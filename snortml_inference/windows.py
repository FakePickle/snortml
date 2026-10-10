"""
snortml_inference/windows.py
=============================
Per-host time windows. A port scan, a flood or a brute-force run is thousands
of tiny flows that each look unremarkable; what gives them away is how many
flows, destinations and unanswered connections one host produces in a minute.

There is exactly one implementation: the baseline is built by feeding recorded
flows through the same aggregator the daemon uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from snortml_inference.features import HOST_NUMERIC, FlowTable, payload_estimate


@dataclass(slots=True)
class Window:
    entity: str  # client address
    kind: str  # "i" internal / "e" external / "m" multicast
    t_end: float  # end of the window, epoch seconds
    n_flows: int
    n_dsts: int  # distinct servers contacted
    n_ports: int  # distinct (protocol, server port) pairs
    n_unanswered: int  # distinct (server, port) targets that never replied
    payload_out: float  # estimated payload bytes sent (bare ACKs count as zero)


class _Acc:
    __slots__ = ("kind", "n_flows", "dsts", "ports", "unanswered", "payload_out")

    def __init__(self, kind: str):
        self.kind = kind
        self.n_flows = 0
        self.dsts = set()
        self.ports = set()
        self.unanswered = set()
        self.payload_out = 0.0


_MAX_SET = 100_000  # bound memory for one host in one window


class WindowAggregator:
    """Groups flows by (client, window of flow end time)."""

    def __init__(self, window_seconds: float = 60.0, grace_seconds: float = 15.0):
        self.window = float(window_seconds)
        self.grace = float(grace_seconds)
        self._open: dict = {}
        self._closed_before = -1  # windows with index < this are finalized

    def add_table(self, ft: FlowTable) -> None:
        """Add every flow in the table (including ones too small for the
        flow layer: scans are made of those)."""
        idx = np.floor(ft.ts_end / self.window).astype(np.int64)
        payload = payload_estimate(ft.fwd_bytes, ft.fwd_pkts)
        for i in range(ft.n):
            w = int(idx[i])
            if w < self._closed_before:
                w = self._closed_before  # late arrival: count it in the oldest open window
            key = (ft.client[i], w)
            acc = self._open.get(key)
            if acc is None:
                acc = _Acc(ft.client_kind[i])
                self._open[key] = acc
            acc.n_flows += 1
            if len(acc.dsts) < _MAX_SET:
                acc.dsts.add(ft.server[i])
            if len(acc.ports) < _MAX_SET:
                acc.ports.add((int(ft.proto[i]), int(ft.server_port[i])))
            if ft.bwd_pkts[i] == 0 and len(acc.unanswered) < _MAX_SET:
                acc.unanswered.add((ft.server[i], int(ft.server_port[i])))
            acc.payload_out += float(payload[i])

    def flush(self, watermark: Optional[float] = None) -> list:
        """Finalize windows that ended more than grace seconds before the
        watermark. watermark=None finalizes everything."""
        if watermark is None:
            limit = None
        else:
            limit = int(np.floor((watermark - self.grace) / self.window))
        done = []
        for key in [k for k in self._open if limit is None or k[1] < limit]:
            acc = self._open.pop(key)
            done.append(Window(
                entity=key[0], kind=acc.kind, t_end=(key[1] + 1) * self.window,
                n_flows=acc.n_flows, n_dsts=len(acc.dsts), n_ports=len(acc.ports),
                n_unanswered=len(acc.unanswered), payload_out=acc.payload_out,
            ))
        if limit is not None and limit > self._closed_before:
            self._closed_before = limit
        done.sort(key=lambda w: (w.t_end, w.entity))
        return done


def window_arrays(windows: list):
    """Windows -> (group keys, numeric matrix, end times)."""
    n = len(windows)
    keys = np.array([f"host:{w.entity}|{w.kind}" for w in windows], dtype=object)
    X = np.zeros((n, len(HOST_NUMERIC)))
    for i, w in enumerate(windows):
        X[i] = (w.n_flows, w.n_dsts, w.n_ports, w.n_unanswered, w.payload_out)
    X = np.log1p(X)
    ts = np.array([w.t_end for w in windows], dtype=np.float64)
    return keys, X, ts
