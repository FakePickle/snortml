"""
snortml_inference/baseline.py
==============================
Fits a baseline bundle from recorded flows and saves/loads it.

A bundle is a directory with two files:

    manifest.json   settings, context groups, calibration summary
    arrays.npz      numeric model parameters

Neither file is a pickle, so loading a bundle cannot execute code.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from snortml_inference.features import (
    FLOW_CATEGORICAL, FLOW_NUMERIC, FLOW_VOLUME, HOST_CATEGORICAL, HOST_NUMERIC,
    ContextMap, active_days, block_split, build_flow_table, flow_parent, host_parent,
    watched_service,
)
from snortml_inference.flows import FlowRecord, Locality, records_to_columns
from snortml_inference.model import DetectorParams, GroupedDetector
from snortml_inference.windows import WindowAggregator, window_arrays

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1


@dataclass
class FitConfig:
    min_packets: int = 3  # flows smaller than this are left to the host layer
    min_group_flows: int = 500  # flows a service needs to get its own context
    window_seconds: float = 60.0
    home_networks: list = field(default_factory=list)
    autoencoder: bool = True
    idle_timeout: int = 10
    active_timeout: int = 60
    seed: int = 7
    # The alert budget the baseline is sized for (see alert_alpha). A context
    # gets its own model only if it has enough held-out data to reach the
    # alert threshold this budget implies; smaller ones are pooled.
    alerts_per_day: float = 10.0
    flow_share: float = 0.7
    drift_margin: float = 5.0
    # Leave flows the signature IDS alerted on out of the baseline
    exclude_signature_flagged: bool = True


@dataclass
class Bundle:
    ctx: ContextMap
    flow: GroupedDetector
    host: Optional[GroupedDetector]
    config: FitConfig
    stats: dict  # rates and validation results
    source_kind: str = ""
    created: float = 0.0


def _rows_to_reach(budget_per_day: float, rows_per_day: float, drift_margin: float,
                   max_extrapolation: float) -> int:
    """Held-out rows a group needs before its smallest reportable p-value
    (1 / (max_extrapolation * rows)) is below the alert threshold."""
    if rows_per_day <= 0 or budget_per_day <= 0:
        return 0
    alpha = min(max(budget_per_day / rows_per_day / max(drift_margin, 1.0), 1e-9), 0.05)
    return int(np.ceil(1.0 / (max_extrapolation * alpha)))


def fit_bundle(records: Sequence[FlowRecord], cfg: Optional[FitConfig] = None) -> Bundle:
    """Fit both layers from baseline flows."""
    cfg = cfg or FitConfig()
    # "Normal" is what the signature IDS let through. Flows it alerted on are
    # known-bad and must not be learned as part of the baseline.
    n_before = len(records)
    if cfg.exclude_signature_flagged:
        records = [r for r in records if not r.sig_alerted]
    n_excluded = n_before - len(records)
    if len(records) < 1000:
        raise ValueError(
            f"only {len(records)} flows in the baseline; at least a few thousand "
            f"are needed. Record for longer."
        )
    cols = records_to_columns(records)
    locality = Locality(cfg.home_networks)
    source_kinds = sorted(set(cols["source"].tolist()))

    # ---- flow layer ----
    eligible_all = (
        ((cols["fwd_pkts"] + cols["bwd_pkts"]) >= cfg.min_packets)
        & (cols["fwd_pkts"] > 0) & (cols["bwd_pkts"] > 0)
    )
    ctx = ContextMap.learn(cols["proto"][eligible_all], cols["src_port"][eligible_all],
                           cols["dst_port"][eligible_all], cfg.min_group_flows)
    ft = build_flow_table(cols, ctx, locality, cfg.min_packets)
    el = ft.eligible
    split = block_split(ft.ts_end)
    days = max(active_days(ft.ts_end), 1e-6)
    flows_per_day = float(el.sum()) / days
    params = DetectorParams(autoencoder=cfg.autoencoder, seed=cfg.seed,
                            one_sided=list(FLOW_VOLUME))
    params.min_cal = max(params.min_cal, _rows_to_reach(
        cfg.alerts_per_day * cfg.flow_share, flows_per_day, cfg.drift_margin,
        params.max_extrapolation))
    flow_det = GroupedDetector(FLOW_NUMERIC, FLOW_CATEGORICAL, flow_parent, params)
    flow_report = flow_det.fit(ft.keys[el], ft.Xnum[el], [x[el] for x in ft.Xcat], split[el])
    flow_incident = np.array(
        [f"{c}>{s}|{k}" for c, s, k in zip(ft.client[el], ft.server[el], ft.keys[el])],
        dtype=object)

    # ---- host layer ----
    agg = WindowAggregator(cfg.window_seconds, grace_seconds=0.0)
    agg.add_table(ft)
    windows = agg.flush(None)
    host_det, host_summary, windows_per_day = None, None, 0.0
    if len(windows) >= 200:
        wkeys, WX, wts = window_arrays(windows)
        host_params = DetectorParams(
            autoencoder=False, min_fit=120, min_cal=60, seed=cfg.seed,
            one_sided=list(HOST_NUMERIC))
        host_params.min_cal = max(host_params.min_cal, _rows_to_reach(
            cfg.alerts_per_day * (1.0 - cfg.flow_share), len(windows) / days,
            cfg.drift_margin, host_params.max_extrapolation))
        host_det = GroupedDetector(HOST_NUMERIC, HOST_CATEGORICAL, host_parent, host_params)
        try:
            host_report = host_det.fit(wkeys, WX, [], block_split(wts))
            windows_per_day = len(windows) / days
            entities = np.array([w.entity for w in windows], dtype=object)
            host_summary = _summarize(host_report, entities, wts)
        except ValueError as e:
            logger.warning(f"host layer not fitted: {e}")
            host_det = None

    # ---- services seen (for "never used before" detection) ----
    watched = watched_service(ft.proto, ft.server_port)
    services: dict = {}
    first_seen: dict = {}
    for proto_num, port, ts in zip(ft.proto[watched].tolist(),
                                   ft.server_port[watched].tolist(),
                                   ft.ts_end[watched].tolist()):
        d = services.setdefault(str(proto_num), {})
        d[str(port)] = d.get(str(port), 0) + 1
        key = (proto_num, port)
        if key not in first_seen or ts < first_seen[key]:
            first_seen[key] = ts
    # How many services turned up for the first time in the last fifth of the
    # baseline: if that is still happening often, "never seen before" is not
    # yet a meaningful signal on this network.
    late_cut = float(np.quantile(ft.ts_end, 0.8)) if ft.n else 0.0
    services_late = sum(1 for ts in first_seen.values() if ts > late_cut)

    stats = {
        "services": services,
        "services_late": services_late,
        "services_late_days": days * 0.2,
        "flows_excluded_signature": int(n_excluded),
        "flows_total": int(ft.n),
        "flows_scored": int(el.sum()),
        "active_days": days,
        # held-out rows are two of every five time blocks
        "heldout_days": days * 0.4,
        "flows_per_day": flows_per_day,
        "windows_total": len(windows),
        "windows_per_day": windows_per_day,
        "flow": _summarize(flow_report, flow_incident, ft.ts_end[el]),
        "host": host_summary,
    }
    return Bundle(ctx=ctx, flow=flow_det, host=host_det, config=cfg, stats=stats,
                  source_kind=",".join(k for k in source_kinds if k),
                  created=time.time())


HELDOUT_TAIL_ROWS = 3000


def _summarize(report: dict, incident: np.ndarray, ts: np.ndarray) -> dict:
    """Keep the JSON-friendly part of a fit report plus the held-out rows with
    the smallest out-of-fold p-values (with what they would be de-duplicated
    by), so an alert budget can be checked against the baseline itself."""
    oof = report["oof_p"]
    rows = np.nonzero(np.isfinite(oof))[0]
    p = oof[rows]
    out = {
        "groups": report["groups"],
        "features": report["features"],
        "rows": report["rows"],
        "rows_unmodeled": report["rows_unmodeled"],
        "heldout_rows": int(rows.size),
        # the share of held-out rows with p <= alpha should be about alpha
        "heldout_exceed": {str(a): int((p <= a).sum()) for a in (0.1, 0.01, 0.001, 0.0001)},
    }
    order = rows[np.argsort(p)][:HELDOUT_TAIL_ROWS]
    out["heldout_tail"] = [[float(oof[i]), str(incident[i]), float(ts[i])] for i in order]
    return out


def alert_alpha(layer: Optional[dict], rows_per_day: float, heldout_days: float,
                budget_per_day: float, suppress_seconds: float = 3600.0,
                drift_margin: float = 5.0) -> dict:
    """Turn an alert budget into a p-value threshold for one layer.

    The nominal threshold is budget / rows-per-day, divided by a safety margin
    because tomorrow's traffic never matches the baseline exactly: on unseen
    days rare-looking events turn up several times more often than the
    baseline alone predicts. The threshold is then checked against held-out
    baseline traffic with the same de-duplication the daemon applies, and
    tightened further if the baseline itself would exceed the budget."""
    if not layer or rows_per_day <= 0 or budget_per_day <= 0:
        return {"alpha": None, "nominal": None, "heldout_incidents": 0, "allowed": 0}
    nominal = min(max(budget_per_day / rows_per_day / max(drift_margin, 1.0), 1e-9), 0.05)
    allowed = max(1, int(round(budget_per_day * heldout_days)))
    alpha = nominal
    seen = set()
    at_nominal = 0
    for p, incident, ts in layer["heldout_tail"]:
        if p > nominal:
            break
        key = (incident, int(ts // max(suppress_seconds, 1.0)))
        if key in seen:
            continue
        seen.add(key)
        at_nominal += 1
        if len(seen) == allowed + 1:
            alpha = max(p * 0.999, 1e-9)
    return {"alpha": alpha, "nominal": nominal, "heldout_incidents": at_nominal,
            "allowed": allowed}


def save_bundle(bundle: Bundle, directory) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    fmeta, farr = bundle.flow.to_state()
    arrays.update({f"flow__{k}": v for k, v in farr.items()})
    hmeta = None
    if bundle.host is not None:
        hmeta, harr = bundle.host.to_state()
        arrays.update({f"host__{k}": v for k, v in harr.items()})
    manifest = {
        "format": BUNDLE_FORMAT,
        "created": bundle.created,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(bundle.created)),
        "source_kind": bundle.source_kind,
        "config": asdict(bundle.config),
        "ports": {str(p): {str(k): v for k, v in d.items()}
                  for p, d in bundle.ctx.ports.items()},
        "flow": fmeta,
        "host": hmeta,
        "stats": bundle.stats,
    }
    tmp_npz = directory / "arrays.tmp.npz"
    np.savez_compressed(tmp_npz, **arrays)
    tmp_npz.replace(directory / "arrays.npz")
    tmp_json = directory / "manifest.tmp.json"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    tmp_json.replace(directory / "manifest.json")
    return directory


def load_bundle(directory) -> Bundle:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no baseline found in {directory}. Create one first with: "
            f"python -m snortml_inference baseline --source <interface or file>"
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format {manifest.get('format')!r}")
    with np.load(directory / "arrays.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}

    def part(prefix):
        return {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}

    flow = GroupedDetector.from_state(manifest["flow"], part("flow__"), flow_parent)
    host = None
    if manifest.get("host"):
        host = GroupedDetector.from_state(manifest["host"], part("host__"), host_parent)
    return Bundle(
        ctx=ContextMap(manifest["ports"]),
        flow=flow, host=host,
        config=FitConfig(**manifest["config"]),
        stats=manifest["stats"],
        source_kind=manifest.get("source_kind", ""),
        created=float(manifest.get("created", 0.0)),
    )


def blind_groups(det: Optional[GroupedDetector], alpha: Optional[float]) -> list:
    """Contexts whose smallest reportable p-value is above the alert
    threshold: nothing in them can alert until the baseline is longer."""
    if det is None or not alpha:
        return []
    return [g for i, g in enumerate(det.groups) if det.p_floor(i) > alpha]


def novelty_ready(stats: dict, alerts_per_day: float) -> bool:
    """New-service alerts are useful only once the baseline has stopped
    discovering services faster than the alert budget."""
    late = stats.get("services_late", 0)
    days = max(stats.get("services_late_days", 0.0), 1e-9)
    return not (late >= 3 and late / days > alerts_per_day)


def describe_bundle(bundle: Bundle, alerts_per_day: float = 10.0,
                    flow_share: float = 0.7, drift_margin: float = 5.0) -> str:
    """Human-readable summary of a fitted baseline and what to expect from it."""
    s = bundle.stats
    lines = []
    hours = s["active_days"] * 24
    lines.append(f"Baseline covers {hours:.1f} hours of traffic "
                 f"({s['flows_total']:,} flows, {s['flows_scored']:,} large enough to score).")
    if s.get("flows_excluded_signature"):
        lines.append(f"{s['flows_excluded_signature']:,} flows the signature IDS had "
                     f"alerted on were left out: they are not part of normal.")
    f = s["flow"]
    groups = sorted(f["groups"], key=lambda g: -g["n_fit"])
    shown = ", ".join(f"{g['key']} ({g['n_fit'] + g['n_cal']:,})" for g in groups[:8])
    more = f" and {len(groups) - 8} more" if len(groups) > 8 else ""
    lines.append(f"Flow contexts: {len(groups)}: {shown}{more}.")
    lines.append(f"Flow features: {', '.join(f['features'])}.")
    if f["rows_unmodeled"]:
        lines.append(f"{f['rows_unmodeled']:,} baseline flows were in contexts too "
                     f"small to model and will not be scored.")
    if s["host"]:
        lines.append(f"Host contexts: {len(s['host']['groups'])} "
                     f"({s['windows_total']:,} host-minutes).")
    else:
        lines.append("Host layer: not fitted (too few host-minutes in the baseline).")

    for name, layer, per_day, budget in (
        ("Flow", s["flow"], s["flows_per_day"], alerts_per_day * flow_share),
        ("Host", s["host"], s["windows_per_day"], alerts_per_day * (1 - flow_share)),
    ):
        r = alert_alpha(layer, per_day, s["heldout_days"], budget,
                        drift_margin=drift_margin)
        if r["alpha"] is None:
            continue
        line = (f"{name} layer: budget {budget:.1f} alerts/day -> alert when a "
                f"{'flow' if name == 'Flow' else 'host-minute'} is rarer than "
                f"1 in {1 / r['alpha']:,.0f}.")
        if r["alpha"] < r["nominal"]:
            line += (f" (Tightened from 1 in {1 / r['nominal']:,.0f}: at that level the "
                     f"held-out baseline produced {r['heldout_incidents']} incidents "
                     f"where the budget allows {r['allowed']}.)")
        lines.append(line)
        det = bundle.flow if name == "Flow" else bundle.host
        blind = blind_groups(det, r["alpha"])
        if blind:
            shown = ", ".join(blind[:5]) + (" ..." if len(blind) > 5 else "")
            lines.append(f"  {len(blind)} of {len(det.groups)} {name.lower()} contexts have "
                         f"too little baseline to alert at this threshold yet: {shown}")
    n_services = sum(len(v) for v in s.get("services", {}).values())
    if novelty_ready(s, alerts_per_day):
        lines.append(f"New-service alerts: on ({n_services} well-known services seen "
                     f"in the baseline).")
    else:
        lines.append(f"New-service alerts: paused. {s['services_late']} services were "
                     f"first seen in the last fifth of the baseline, so new ones are "
                     f"still routine here; a longer baseline turns this on.")
    if hours < 24:
        lines.append("Note: less than a day of baseline. Expect extra alerts the first "
                     "time routine traffic appears that the baseline never saw; "
                     "a week is a better baseline.")
    return "\n".join(lines)
