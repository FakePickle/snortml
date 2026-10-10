"""
snortml_inference/alerter.py
=============================
Decides which anomalies become alerts and writes them out.

An anomalous flow is rarely alone: a long transfer is seen as one slice per
minute, a retrying client produces the same odd flow again and again. Alerts
are therefore grouped into incidents. The first event of an incident is
emitted, repeats are counted, and a repeat is emitted again only if it is
more severe. A per-hour cap stops an alert storm when the baseline has gone
stale.

Outputs: console, a one-line-per-alert text log, a JSON-lines file with
Suricata-style field names (for Wazuh, Security Onion or any SIEM), and an
optional webhook.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from snortml_inference.scorer import one_in

logger = logging.getLogger(__name__)

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
PROTO_LABEL = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "IPV6-ICMP", 2: "IGMP"}
_COLORS = {"CRITICAL": "\033[91m", "HIGH": "\033[93m", "MEDIUM": "\033[94m", "LOW": ""}


def severity_from_depth(depth: float, theta: float) -> str:
    """depth and theta are -log10 of the event's tail probability and of the
    alert threshold. Each order of magnitude beyond the threshold is a step."""
    steps = int(max(depth - theta, 0.0))
    return SEVERITIES[min(steps, len(SEVERITIES) - 1)]


@dataclass
class Alert:
    ts: float  # event time, epoch seconds
    layer: str  # "flow", "host" or "new-service"
    kind: str
    severity: str
    client: str
    p: Optional[float] = None
    alpha: Optional[float] = None
    server: str = ""
    client_port: int = 0
    server_port: int = 0
    proto: int = 0
    context: str = ""
    resembles: str = ""  # what this pattern usually is; a lead, not a verdict
    reasons: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    signature_flagged: bool = False
    repeats: int = 0  # earlier occurrences folded into this incident

    def key(self) -> tuple:
        if self.layer == "flow":
            return ("flow", self.client, self.server, self.context)
        # a host that scans and a host that uploads are different incidents
        return (self.layer, self.client, self.kind)

    def text(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))
        if self.layer == "flow":
            what = f"flow anomaly ({self.kind})"
            where = (f"{self.client}:{self.client_port} -> "
                     f"{self.server}:{self.server_port} {self.context}")
        elif self.layer == "host":
            what = f"host anomaly ({self.kind})"
            where = self.client
        else:
            what = "new service"
            where = (f"{self.client} -> {self.server}:{self.server_port} {self.context}"
                     if self.server else self.client)
        parts = [f"[{stamp}] [{self.severity}] {what}", where]
        if self.p is not None:
            parts.append(one_in(self.p))
        if self.resembles:
            parts.append(f"resembles: {self.resembles}")
        if self.reasons:
            parts.append("; ".join(self.reasons))
        if self.signature_flagged:
            parts.append("also flagged by signature IDS")
        if self.repeats:
            parts.append(f"repeat {self.repeats}")
        return " | ".join(parts)

    def to_json(self) -> dict:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.ts))
        stamp += f".{int((self.ts % 1) * 1e6):06d}+0000"
        out = {
            "timestamp": stamp,
            "event_type": "anomaly",
            "layer": self.layer,
            "kind": self.kind,
            "severity": self.severity,
            "src_ip": self.client,
        }
        if self.layer != "host":
            out.update({
                "src_port": self.client_port,
                "dest_ip": self.server,
                "dest_port": self.server_port,
                "proto": PROTO_LABEL.get(self.proto, str(self.proto)),
            })
        if self.context:
            out["context"] = self.context
        if self.p is not None:
            out["p_value"] = self.p
            out["one_in"] = round(1.0 / max(self.p, 1e-12))
        if self.alpha is not None:
            out["threshold_one_in"] = round(1.0 / max(self.alpha, 1e-12))
        out["resembles"] = self.resembles
        out["reasons"] = self.reasons
        out["detail"] = self.detail
        out["signature_flagged"] = self.signature_flagged
        out["repeats"] = self.repeats
        return out


class _Webhook:
    """Posts alerts from a background thread so a slow endpoint never stalls scoring."""

    def __init__(self, url: str, fmt: str = "json", timeout: float = 5.0):
        self.url, self.fmt, self.timeout = url, fmt, timeout
        self._q: "queue.Queue" = queue.Queue(maxsize=500)
        self._failed = False
        threading.Thread(target=self._run, daemon=True, name="webhook").start()

    def send(self, alert: Alert) -> None:
        try:
            self._q.put_nowait(alert)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            alert = self._q.get()
            try:
                if self.fmt == "text":
                    body = alert.text().encode("utf-8")
                    headers = {"Content-Type": "text/plain; charset=utf-8",
                               "Title": f"{alert.severity} {alert.layer} anomaly"}
                else:
                    body = json.dumps(alert.to_json()).encode("utf-8")
                    headers = {"Content-Type": "application/json"}
                req = urllib.request.Request(self.url, data=body, headers=headers,
                                             method="POST")
                urllib.request.urlopen(req, timeout=self.timeout).close()
                self._failed = False
            except Exception as e:  # network errors must never reach the scorer
                if not self._failed:
                    logger.warning(f"webhook delivery failed: {e}")
                self._failed = True


class AlertManager:
    """Incident grouping, muting, rate cap and output."""

    def __init__(self, cfg: Optional[dict] = None):
        a = (cfg or {}).get("alerts", {}) or {}
        self.console = bool(a.get("console", True))
        self.color = bool(a.get("color", sys.stdout.isatty()))
        self.text_file = a.get("file")
        self.json_file = a.get("json_file")
        self.suppress = float(a.get("suppress_seconds", 3600))
        self.max_per_hour = int(a.get("max_per_hour", 30))
        self.mute = set(a.get("mute", ["download"]) or [])
        min_sev = str(a.get("min_severity", "LOW")).upper()
        self.min_rank = SEVERITIES.index(min_sev) if min_sev in SEVERITIES else 0
        hook = a.get("webhook")
        if isinstance(hook, str):
            hook = {"url": hook}
        self._webhook = _Webhook(hook["url"], hook.get("format", "json")) if hook else None
        for path in (self.text_file, self.json_file):
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._active: dict = {}
        self._emitted: deque = deque()
        self._storm_noted = -1e18
        self.counts: Counter = Counter()
        self.emitted: list = []  # kept for replay summaries and tests
        self.keep_history = False

    # ------------------------------------------------------------------

    def submit(self, alert: Alert) -> bool:
        """Returns True if the alert was emitted."""
        self.counts["candidates"] += 1
        if alert.kind in self.mute:
            self.counts["muted"] += 1
            return False
        rank = SEVERITIES.index(alert.severity)
        if rank < self.min_rank:
            self.counts["below_min_severity"] += 1
            return False
        key = alert.key()
        state = self._active.get(key)
        if state is not None and alert.ts < state["until"]:
            state["repeats"] += 1
            if rank <= state["rank"]:
                self.counts["repeats"] += 1
                return False
            state["rank"] = rank  # more severe than what was reported: report again
            alert.repeats = state["repeats"]
        else:
            self._active[key] = {"until": alert.ts + self.suppress, "rank": rank,
                                 "repeats": 0}
        # hourly cap
        while self._emitted and self._emitted[0] <= alert.ts - 3600:
            self._emitted.popleft()
        if len(self._emitted) >= self.max_per_hour:
            self.counts["rate_capped"] += 1
            if alert.ts - self._storm_noted >= 3600:
                self._storm_noted = alert.ts
                self._write_line(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.ts))}] "
                    f"[NOTICE] more than {self.max_per_hour} alerts in an hour; further "
                    f"alerts are being held back. The baseline is probably out of date: "
                    f"re-run the baseline command.")
            return False
        self._emitted.append(alert.ts)
        self._emit(alert)
        if len(self._active) > 50_000:
            self.prune(alert.ts)
        return True

    def prune(self, now: float) -> None:
        for key in [k for k, s in self._active.items() if s["until"] <= now]:
            del self._active[key]

    # ------------------------------------------------------------------

    def _emit(self, alert: Alert) -> None:
        self.counts["emitted"] += 1
        self.counts[f"emitted_{alert.layer}"] += 1
        if self.keep_history:
            self.emitted.append(alert)
        self._write_line(alert.text(), severity=alert.severity)
        if self.json_file:
            with open(self.json_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_json()) + "\n")
        if self._webhook is not None:
            self._webhook.send(alert)

    def _write_line(self, line: str, severity: str = "") -> None:
        if self.console:
            color = _COLORS.get(severity, "") if self.color else ""
            try:
                print(f"{color}{line}\033[0m" if color else line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        if self.text_file:
            with open(self.text_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def summary(self) -> str:
        c = self.counts
        return (f"alerts emitted: {c['emitted']} "
                f"(flow {c['emitted_flow']}, host {c['emitted_host']}, "
                f"new-service {c['emitted_new-service']}); "
                f"repeats folded: {c['repeats']}; muted: {c['muted']}; "
                f"held back by hourly cap: {c['rate_capped']}")
