"""
snortml_inference/daemon.py
============================
The scoring engine and the loops that feed it.

Engine holds everything that happens to a batch of flows: tag flows a
signature IDS already flagged, score them, look for never-seen services,
aggregate per-host windows, and hand anomalies to the alert manager. The
live daemon and the offline replay both drive the same Engine, so what you
see in a replay is what the daemon would have done.

Capture runs on its own thread and hands flows over through a queue; the
main thread scores. (On Windows nfstream spawns worker processes, and keeping
the capture iterator off the main thread keeps shutdown and timed flushing
simple.)
"""

from __future__ import annotations

import logging
import math
import queue
import signal
import sys
import threading
import time
from collections import Counter
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from snortml_inference.alerter import Alert, AlertManager, severity_from_depth
from snortml_inference.baseline import Bundle, blind_groups, load_bundle, novelty_ready
from snortml_inference.features import SENSITIVE_PORTS
from snortml_inference.flows import FlowRecord, FlowWriter
from snortml_inference.ids import SignatureIndex
from snortml_inference.scorer import Scorer, flow_resembles, host_resembles
from snortml_inference.sources import open_source
from snortml_inference.windows import WindowAggregator

logger = logging.getLogger(__name__)

_SENTINEL = object()


def window_grace(cfg: dict) -> float:
    """How long to wait after a host-window ends before scoring it. Flows are
    reported after they end: nfstream after its idle timeout, Zeek and
    Suricata after their own (longer) connection timeouts."""
    override = (cfg.get("scoring") or {}).get("window_grace_seconds")
    if override is not None:
        return float(override)
    kind = (cfg.get("source") or {}).get("type", "live")
    if kind in ("zeek", "suricata"):
        return 90.0
    return float((cfg.get("capture") or {}).get("idle_timeout", 10)) + 5.0


class Engine:
    """Scores batches of flows and raises alerts."""

    def __init__(self, bundle: Bundle, cfg: dict,
                 sig_index: Optional[SignatureIndex] = None,
                 recorder: Optional[FlowWriter] = None,
                 alerts: Optional[AlertManager] = None):
        self.cfg = cfg
        self.sig = sig_index
        self.recorder = recorder
        self.alerts = alerts or AlertManager(cfg)
        self.on_match = (cfg.get("ids") or {}).get("on_match", "suppress")
        nov = cfg.get("novelty") or {}
        self.novelty_enabled = bool(nov.get("enabled", True))
        self.allow = set(nov.get("allow") or [])
        self.stats: Counter = Counter()
        self._novel: dict = {}  # client -> {service label: last seen}
        self._drift = {"hour": None, "scored": 0, "candidates": 0, "warned": False}
        self.watermark = 0.0
        self.set_bundle(bundle)

    def set_bundle(self, bundle: Bundle) -> None:
        cfg = self.cfg
        sc = cfg.get("scoring") or {}
        self.bundle = bundle
        self.scorer = Scorer(
            bundle,
            home_networks=cfg.get("home_networks") or [],
            alerts_per_day=float(sc.get("alerts_per_day", 10)),
            flow_share=float(sc.get("flow_share", 0.7)),
            suppress_seconds=float((cfg.get("alerts") or {}).get("suppress_seconds", 3600)),
            drift_margin=float(sc.get("drift_margin", 5)),
        )
        self.agg = WindowAggregator(bundle.config.window_seconds, window_grace(cfg))
        self.alpha_flow = self.scorer.alpha_flow
        self.alpha_host = self.scorer.alpha_host
        self.theta_flow = -math.log10(self.alpha_flow) if self.alpha_flow else None
        self.theta_host = -math.log10(self.alpha_host) if self.alpha_host else None
        self.baseline_flows = int(bundle.stats.get("flows_total", 0))
        self.novelty = self.novelty_enabled and novelty_ready(
            bundle.stats, float(sc.get("alerts_per_day", 10)))
        if self.novelty_enabled and not self.novelty:
            logger.info("New-service alerts paused: the baseline was still discovering "
                        "new services; record a longer baseline to enable them")

    # ------------------------------------------------------------------

    def process(self, records: list) -> None:
        if not records:
            return
        if self.sig is not None:
            for r in records:
                if not r.sig_alerted and self.sig.matches(r):
                    r.sig_alerted = True
        # recorded after tagging, so a later re-fit knows which flows the
        # signature IDS had already caught
        if self.recorder is not None:
            self.recorder.write(records)
        self.watermark = max(self.watermark, max(r.ts_end for r in records))
        n_flagged = sum(1 for r in records if r.sig_alerted)
        self.stats["flows"] += len(records)
        self.stats["signature_flagged"] += n_flagged
        if n_flagged and self.on_match == "suppress":
            # The signature layer owns these. They are not scored, not counted
            # in the host windows, and not treated as new services: this layer
            # only looks at what the rules let through.
            self.stats["left_to_signatures"] += n_flagged
            records = [r for r in records if not r.sig_alerted]
            if not records:
                return
        sc = self.scorer
        ft = sc.table(records)
        self.agg.add_table(ft)
        scores = sc.score_flows(ft)

        n_scored = int(ft.eligible.sum())
        self.stats["scored"] += n_scored
        self.stats["no_model"] += int((ft.eligible & (scores.group < 0)).sum())

        if self.alpha_flow:
            hits = np.nonzero(ft.eligible & (scores.p <= self.alpha_flow))[0]
            self._track_drift(float(ft.ts_end.max()), n_scored, hits.size)
            for i in hits:
                rec = records[i]
                kind = sc.flow_kind(ft, scores, i)
                self.alerts.submit(Alert(
                    ts=float(ft.ts_end[i]), layer="flow",
                    kind=kind, resembles=flow_resembles(kind),
                    severity=severity_from_depth(scores.depth[i], self.theta_flow),
                    p=float(scores.p[i]), alpha=self.alpha_flow,
                    client=str(ft.client[i]), server=str(ft.server[i]),
                    client_port=int(ft.client_port[i]), server_port=int(ft.server_port[i]),
                    proto=int(ft.proto[i]), context=sc.flow_context(scores, i),
                    reasons=sc.explain_flow(ft, scores, i),
                    detail={
                        "pkts_toserver": int(ft.fwd_pkts[i]),
                        "pkts_toclient": int(ft.bwd_pkts[i]),
                        "bytes_toserver": int(ft.fwd_bytes[i]),
                        "bytes_toclient": int(ft.bwd_bytes[i]),
                        "duration_ms": float(rec.duration_ms),
                    },
                    signature_flagged=bool(rec.sig_alerted),
                ))

        if self.novelty:
            self._novel_services(ft, records)

    def _novel_services(self, ft, records) -> None:
        window = self.alerts.suppress
        for i, label in self.scorer.novel_services(ft, self.allow):
            rec = records[i]
            client = str(ft.client[i])
            ts = float(ft.ts_end[i])
            seen = self._novel.setdefault(client, {})
            for k in [k for k, v in seen.items() if v[0] < ts - window]:
                del seen[k]
            entry = seen.setdefault(label, [ts, 0])
            entry[0] = ts
            entry[1] += 1
            count, uses = len(seen), entry[1]
            port = int(ft.server_port[i])
            lateral = ft.Xcat[1][i] == "ii"
            if count >= 20:
                severity = "HIGH"
            elif count >= 5 or uses >= 50 or port in SENSITIVE_PORTS or lateral:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            if count > 1:
                examples = ", ".join(sorted(seen)[:6])
                reason = (f"{count} services never seen in the baseline within the "
                          f"last hour, e.g. {examples}")
            else:
                reason = (f"{label} was never used in the baseline "
                          f"({self.baseline_flows:,} flows)")
                if uses >= 50:
                    reason += f"; {uses} connections in the last hour"
            self.alerts.submit(Alert(
                ts=ts, layer="new-service", kind="new-service", severity=severity,
                resembles=("service scan" if count >= 20 else
                           "repeated attempts on a service this host never used"
                           if uses >= 50 else ""),
                client=client, server=str(ft.server[i]),
                client_port=int(ft.client_port[i]), server_port=port,
                proto=int(ft.proto[i]), context=label, reasons=[reason],
                detail={"new_services": sorted(seen)[:50], "count": count},
                signature_flagged=bool(rec.sig_alerted),
            ))
        if len(self._novel) > 20_000:
            self._novel.clear()

    def _track_drift(self, ts: float, scored: int, candidates: int) -> None:
        """Warn when far more flows cross the threshold than the baseline
        predicts: the network has changed and the baseline needs refreshing."""
        d = self._drift
        hour = int(ts // 3600)
        if d["hour"] != hour:
            d.update(hour=hour, scored=0, candidates=0, warned=False)
        d["scored"] += scored
        d["candidates"] += candidates
        expected = d["scored"] * self.alpha_flow
        if (not d["warned"] and d["scored"] >= 1000 and d["candidates"] >= 25
                and d["candidates"] > 20 * max(expected, 1.0)):
            d["warned"] = True
            logger.warning(
                f"{d['candidates']} of {d['scored']} flows this hour are beyond the "
                f"alert threshold (about {expected:.1f} expected). Traffic no longer "
                f"looks like the baseline; re-run the baseline command.")

    def tick(self, watermark: Optional[float]) -> None:
        """Close host windows older than the watermark and score them."""
        windows = self.agg.flush(watermark)
        if not windows:
            return
        self.stats["host_minutes"] += len(windows)
        if not self.alpha_host:
            return
        ws = self.scorer.score_windows(windows)
        if ws is None:
            return
        for i in np.nonzero(ws.p <= self.alpha_host)[0]:
            w = windows[i]
            kind = self.scorer.host_kind(ws, i)
            self.alerts.submit(Alert(
                ts=float(w.t_end), layer="host", kind=kind,
                resembles=host_resembles(kind, w),
                severity=severity_from_depth(ws.depth[i], self.theta_host),
                p=float(ws.p[i]), alpha=self.alpha_host, client=str(w.entity),
                reasons=self.scorer.explain_window(w, ws, i),
                detail={"flows": w.n_flows, "destinations": w.n_dsts,
                        "ports": w.n_ports, "unanswered": w.n_unanswered,
                        "payload_out": int(w.payload_out)},
            ))
        self.alerts.prune(float(windows[-1].t_end))

    def finish(self) -> None:
        self.tick(None)

    def summary(self) -> str:
        s = self.stats
        return (f"flows seen: {s['flows']:,}; scored: {s['scored']:,}; "
                f"in contexts with no model: {s['no_model']:,}; "
                f"host-minutes: {s['host_minutes']:,}; "
                f"left to signature IDS: {s['left_to_signatures']:,}\n"
                + self.alerts.summary())


# ---------------------------------------------------------------------------
# Offline replay
# ---------------------------------------------------------------------------

def replay(engine: Engine, flows: Iterable[FlowRecord], batch_size: int = 2000) -> dict:
    """Run recorded flows through the engine in arrival order."""
    batch = []
    first = last = None
    for rec in flows:
        if first is None:
            first = rec.ts_end
        last = rec.ts_end if last is None else max(last, rec.ts_end)
        batch.append(rec)
        if len(batch) >= batch_size:
            engine.process(batch)
            engine.tick(engine.watermark)
            batch = []
    if batch:
        engine.process(batch)
    engine.finish()
    span = (last - first) if first is not None else 0.0
    return {"seconds": span, "stats": dict(engine.stats),
            "alerts": dict(engine.alerts.counts)}


# ---------------------------------------------------------------------------
# Live daemon
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging") or {}
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = log_cfg.get("file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3,
                                            encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=handlers, force=True)


def _reader(cfg: dict, out: "queue.Queue", stop: threading.Event) -> None:
    src, cap = cfg["source"], cfg["capture"]
    kind = src.get("type", "live")
    target = src.get("interface") if kind == "live" else src.get("path")
    try:
        flows = open_source(
            kind, target, idle_timeout=int(cap["idle_timeout"]),
            active_timeout=int(cap["active_timeout"]),
            follow=bool(src.get("follow", True)), from_start=False, stop=stop)
        for rec in flows:
            if stop.is_set():
                break
            out.put(rec)
    except Exception as e:
        logger.error(f"input stopped: {type(e).__name__}: {e}")
    finally:
        out.put(_SENTINEL)


def run_daemon(cfg: dict) -> int:
    """Score a live source until interrupted. Returns a process exit code."""
    setup_logging(cfg)
    src = cfg["source"]
    kind = src.get("type", "live")
    target = src.get("interface") if kind == "live" else src.get("path")
    if not target:
        logger.error("no input configured: set source.interface (or source.path) in "
                     "config.yaml, or pass --interface")
        return 2
    try:
        bundle = load_bundle(cfg["baseline_dir"])
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 2

    stop = threading.Event()
    sig = None
    ids_cfg = cfg.get("ids") or {}
    if ids_cfg.get("suricata_eve") or ids_cfg.get("snort_json"):
        sig = SignatureIndex()
        for path in (ids_cfg.get("suricata_eve"), ids_cfg.get("snort_json")):
            if path:
                sig.follow(path, stop)
                logger.info(f"Following signature alerts in {path}")
    rec_cfg = cfg.get("record") or {}
    recorder = None
    if rec_cfg.get("enabled", True):
        recorder = FlowWriter(rec_cfg.get("dir", "data"), rec_cfg.get("retention_days"))
    engine = Engine(bundle, cfg, sig_index=sig, recorder=recorder)

    cap = cfg["capture"]
    for name in ("idle_timeout", "active_timeout", "min_packets"):
        used, now = getattr(bundle.config, name), int(cap[name])
        if name != "min_packets" and used != now:
            logger.warning(f"capture.{name} is {now} but the baseline was recorded with "
                           f"{used}; flows will be cut differently than the baseline expects")
    if bundle.source_kind and kind in ("zeek", "suricata") and kind not in bundle.source_kind:
        logger.warning(f"baseline was built from '{bundle.source_kind}' flows but the "
                       f"input is '{kind}'; build the baseline from the same source")

    logger.info("=" * 60)
    logger.info(f"Input      : {kind} {target}")
    logger.info(f"Baseline   : {cfg['baseline_dir']} "
                f"({bundle.stats['active_days'] * 24:.1f} h, "
                f"{len(bundle.flow.groups)} flow contexts)")
    if engine.alpha_flow:
        logger.info(f"Flow alerts: rarer than 1 in {1 / engine.alpha_flow:,.0f}")
    if engine.alpha_host:
        logger.info(f"Host alerts: rarer than 1 in {1 / engine.alpha_host:,.0f}")
    for name, det, alpha in (("flow", bundle.flow, engine.alpha_flow),
                             ("host", bundle.host, engine.alpha_host)):
        blind = blind_groups(det, alpha)
        if blind:
            logger.info(f"{len(blind)} of {len(det.groups)} {name} contexts cannot alert "
                        f"yet (too little baseline for this budget)")
    logger.info("Listening... (Ctrl+C to stop)")
    logger.info("=" * 60)

    def _stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    q: "queue.Queue" = queue.Queue(maxsize=int(cap["batch_size"]) * 50)
    threading.Thread(target=_reader, args=(cfg, q, stop), daemon=True,
                     name="input").start()

    batch_size = int(cap["batch_size"])
    max_wait = float(cap["max_batch_wait_sec"])
    live_clock = kind == "live" or (kind in ("zeek", "suricata")
                                    and bool(src.get("follow", True)))
    manifest = Path(cfg["baseline_dir"]) / "manifest.json"
    manifest_mtime = manifest.stat().st_mtime
    buffer: list = []
    last_flush = last_check = time.time()
    t_start = time.time()
    try:
        while not stop.is_set():
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                item = None
            if item is _SENTINEL:
                break
            if item is not None:
                buffer.append(item)
            now = time.time()
            if buffer and (len(buffer) >= batch_size or now - last_flush >= max_wait):
                try:
                    engine.process(buffer)
                except Exception:
                    logger.exception("scoring failed for a batch; continuing")
                buffer = []
                last_flush = now
            try:
                # live input: the wall clock; file input: the newest flow seen
                engine.tick(now if live_clock else engine.watermark)
            except Exception:
                logger.exception("host-window scoring failed; continuing")
            if now - last_check >= 60:
                last_check = now
                try:
                    mtime = manifest.stat().st_mtime
                    if mtime != manifest_mtime:
                        engine.set_bundle(load_bundle(cfg["baseline_dir"]))
                        manifest_mtime = mtime
                        logger.info("Baseline changed on disk: reloaded")
                except Exception as e:
                    logger.warning(f"could not reload baseline: {e}")
        if buffer:
            engine.process(buffer)
        engine.finish()
    finally:
        stop.set()
        logger.info(f"Stopped after {time.time() - t_start:.0f} s. {engine.summary()}")
    return 0
