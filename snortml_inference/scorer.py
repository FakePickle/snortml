"""
snortml_inference/scorer.py
============================
Loads a baseline bundle and scores flows and host-windows against it.
Pure numpy: no TensorFlow, no ONNX runtime, no pickle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from snortml_inference.baseline import Bundle, alert_alpha
from snortml_inference.features import (
    FEATURE_INFO, FLOW_VOLUME, FlowTable, build_flow_table, describe_category,
    payload_estimate, watched_service,
)
from snortml_inference.flows import FlowRecord, Locality, records_to_columns
from snortml_inference.windows import window_arrays

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting for explanations
# ---------------------------------------------------------------------------

def fmt_bytes(v: float) -> str:
    v = max(float(v), 0.0)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f} {unit}" if unit == "B" or v >= 100 else f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} GB"


def fmt_ms(v: float) -> str:
    v = max(float(v), 0.0)
    if v < 1000:
        return f"{v:.0f} ms"
    if v < 120_000:
        return f"{v / 1000:.1f} s"
    return f"{v / 60_000:.1f} min"


def fmt_value(feature: str, x: float) -> str:
    """x is in the model's transformed units (log1p for most features)."""
    unit = FEATURE_INFO.get(feature, (feature, "count"))[1]
    if unit == "frac":
        return f"{100 * x:.0f}%"
    raw = math.expm1(x) if feature.startswith("log_") else x
    if unit == "bytes":
        return fmt_bytes(raw)
    if unit == "ms":
        return fmt_ms(raw)
    return f"{raw:,.0f}"


def one_in(p: float) -> str:
    """'1 in 250,000' style rendering of a p-value."""
    n = 1.0 / max(p, 1e-12)
    if n >= 1e9:
        return "1 in more than a billion"
    if n >= 1e6:
        return f"1 in {n / 1e6:.1f} million"
    return f"1 in {n:,.0f}"


_FLOW_RESEMBLES = {
    "upload": "possible data exfiltration",
    "download": "large transfer to this host",
    "internal-download": "bulk data pulled from an internal server",
    "volume": "unusually large exchange",
    "timing": "tunnel or automated traffic (unusual timing)",
    "packet-size": "tunnel or non-standard use of the protocol (unusual packet sizes)",
    "connection-state": "unusual way of opening or closing connections",
    "direction": "service used in an unusual direction",
}


def flow_resembles(kind: str) -> str:
    """What a flow anomaly of this kind most often is. A description of the
    pattern to start an investigation from, not an identification."""
    return _FLOW_RESEMBLES.get(kind, "")


def host_resembles(kind: str, window) -> str:
    """Names the behaviour a host-minute looks like from its own counts."""
    flows, dsts, ports = window.n_flows, max(window.n_dsts, 1), window.n_ports
    if kind == "upload":
        return "possible data exfiltration"
    if ports >= 20 and ports >= 5 * dsts:
        return "port scan"
    if dsts >= 20 and ports <= max(3, dsts // 5):
        return "sweep of many hosts on the same service (scan or worm-like spread)"
    if dsts >= 20:
        return "scan of many hosts and ports"
    if flows >= 100 and flows >= 20 * dsts:
        return "flood or brute force against one target"
    if kind == "unanswered":
        return "probing targets that do not answer"
    return "unusual burst of connections"


@dataclass
class Scores:
    p: np.ndarray
    depth: np.ndarray  # -log10 of the tail estimate without the evidence floor
    group: np.ndarray  # index into detector.groups, -1 if the context has no model
    contrib: np.ndarray


_VOLUME = set(FLOW_VOLUME)
_TIMING = {"log_mean_piat", "log_std_piat", "log_max_piat"}
_SIZES = {"log_min_ps", "log_max_ps", "log_std_ps", "psh_frac"}


class Scorer:
    """Scores flows and host-windows and explains the result."""

    def __init__(self, bundle: Bundle, home_networks: Optional[Sequence[str]] = None,
                 alerts_per_day: float = 10.0, flow_share: float = 0.7,
                 suppress_seconds: float = 3600.0, drift_margin: float = 5.0,
                 alpha_flow: Optional[float] = None, alpha_host: Optional[float] = None):
        self.bundle = bundle
        cfg = bundle.config
        nets = list(cfg.home_networks) + list(home_networks or [])
        self.locality = Locality(nets)
        self.min_packets = cfg.min_packets
        s = bundle.stats
        self.flow_budget = alert_alpha(
            s["flow"], s["flows_per_day"], s["heldout_days"],
            alerts_per_day * flow_share, suppress_seconds, drift_margin)
        self.host_budget = alert_alpha(
            s.get("host"), s["windows_per_day"], s["heldout_days"],
            alerts_per_day * (1.0 - flow_share), suppress_seconds, drift_margin)
        self.alpha_flow = alpha_flow if alpha_flow else self.flow_budget["alpha"]
        self.alpha_host = alpha_host if alpha_host else self.host_budget["alpha"]
        if bundle.host is None:
            self.alpha_host = None

    # ------------------------------------------------------------------

    def table(self, records: Sequence[FlowRecord]) -> FlowTable:
        return build_flow_table(records_to_columns(records), self.bundle.ctx,
                                self.locality, self.min_packets)

    def score_flows(self, ft: FlowTable) -> Scores:
        """Scores every row; rows with ft.eligible False get p = 1."""
        n = ft.n
        p = np.ones(n)
        depth = np.zeros(n)
        group = np.full(n, -1, dtype=np.int64)
        det = self.bundle.flow
        contrib = np.zeros((n, det.d))
        rows = np.nonzero(ft.eligible)[0]
        if rows.size:
            r = det.score(ft.keys[rows], ft.Xnum[rows], [x[rows] for x in ft.Xcat])
            p[rows], group[rows], contrib[rows] = r["p"], r["group"], r["contrib"]
            depth[rows] = r["depth"]
        return Scores(p=p, depth=depth, group=group, contrib=contrib)

    def score_windows(self, windows: list) -> Optional[Scores]:
        det = self.bundle.host
        if det is None or not windows:
            return None
        keys, X, _ = window_arrays(windows)
        r = det.score(keys, X, [])
        return Scores(p=r["p"], depth=r["depth"], group=r["group"], contrib=r["contrib"])

    # ------------------------------------------------------------------
    # Services never seen in the baseline
    # ------------------------------------------------------------------

    def novel_services(self, ft: FlowTable, allow: Sequence[str] = ()) -> list:
        """Rows (index, label) whose well-known or sensitive service port was
        never used in the baseline. Counted only when one of our hosts is the
        client, or one of our hosts answered as the server: unanswered probes
        arriving from outside are background noise."""
        known = self.bundle.stats.get("services", {})
        watched = watched_service(ft.proto, ft.server_port)
        out = []
        for i in np.nonzero(watched)[0]:
            proto, port = int(ft.proto[i]), int(ft.server_port[i])
            if known.get(str(proto), {}).get(str(port), 0) > 0:
                continue
            loc = ft.Xcat[1][i]
            if loc[1] == "m":
                continue
            ours = loc[0] == "i" or (loc[1] == "i" and ft.bwd_pkts[i] > 0)
            if not ours:
                continue
            label = f"{'tcp' if proto == 6 else 'udp'}/{port}"
            if label in allow:
                continue
            out.append((int(i), label))
        return out

    # ------------------------------------------------------------------
    # What kind of anomaly
    # ------------------------------------------------------------------

    def flow_kind(self, ft: FlowTable, scores: Scores, i: int) -> str:
        names = self.bundle.flow.feature_names
        c = np.maximum(scores.contrib[i], 0.0)
        total = float(c.sum()) or 1.0

        def share(group):
            return sum(c[j] for j, n in enumerate(names) if n in group) / total

        if share(_VOLUME) >= 0.5:
            up = float(payload_estimate(ft.fwd_bytes[i], ft.fwd_pkts[i]))
            down = float(payload_estimate(ft.bwd_bytes[i], ft.bwd_pkts[i]))
            if down >= 4 * max(up, 1.0):
                internal = ft.Xcat[1][i] == "ii"
                return "internal-download" if internal else "download"
            if up >= 4 * max(down, 1.0):
                return "upload"
            return "volume"
        best = max(("timing", share(_TIMING)), ("packet-size", share(_SIZES)),
                   ("connection-state", share({"tcp_state"})),
                   ("direction", share({"locality"})), key=lambda t: t[1])
        return best[0]

    def host_kind(self, scores: Scores, i: int) -> str:
        names = self.bundle.host.feature_names
        top = names[int(np.argmax(scores.contrib[i]))]
        return {"log_unanswered": "unanswered", "log_payload_out": "upload"}.get(top, "fan-out")

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def _explain(self, det, gi: int, contrib_row: np.ndarray, x_num: np.ndarray,
                 x_cat: list, top: int = 3) -> list:
        names = det.feature_names
        order = np.argsort(-contrib_row)[:top]
        total = float(contrib_row[contrib_row > 0].sum()) or 1.0
        reasons = []
        for j in order:
            if contrib_row[j] <= 0 or contrib_row[j] / total < 0.08:
                continue
            name = names[j]
            label = FEATURE_INFO.get(name, (name, ""))[0]
            if j < det.dn:
                value = x_num[det.num_idx[j]]
                if not np.isfinite(value):
                    continue
                q05, _, q95 = det.typical_range(gi, name)
                reasons.append(
                    f"{label} {fmt_value(name, value)} "
                    f"(usually {fmt_value(name, q05)} to {fmt_value(name, q95)})")
            else:
                c = j - det.dn
                value = str(x_cat[c])
                usual = det.typical_range(gi, name)
                if value == usual:
                    continue
                reasons.append(
                    f"{label}: {describe_category(name, value)} "
                    f"(usually {describe_category(name, usual)})")
        return reasons

    def explain_flow(self, ft: FlowTable, scores: Scores, i: int) -> list:
        gi = int(scores.group[i])
        if gi < 0:
            return []
        return self._explain(self.bundle.flow, gi, scores.contrib[i], ft.Xnum[i],
                             [x[i] for x in ft.Xcat])

    def explain_window(self, window, scores: Scores, i: int) -> list:
        gi = int(scores.group[i])
        if gi < 0:
            return []
        _, X, _ = window_arrays([window])
        return self._explain(self.bundle.host, gi, scores.contrib[i], X[0], [])

    def flow_context(self, scores: Scores, i: int) -> str:
        gi = int(scores.group[i])
        return self.bundle.flow.groups[gi] if gi >= 0 else "unmodeled"
