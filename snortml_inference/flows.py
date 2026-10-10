"""
snortml_inference/flows.py
===========================
The one flow record every input source is normalized to, plus CSV storage
and the internal/external address classifier.

Baseline fitting and live scoring both start from this record, so a feature
can never mean one thing in training and another in the daemon.
"""

from __future__ import annotations

import csv
import gzip
import ipaddress
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

NAN = float("nan")


@dataclass(slots=True)
class FlowRecord:
    """One bidirectional flow. "fwd" is src -> dst as the source reported it."""

    ts_start: float  # epoch seconds
    ts_end: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int  # IP protocol number
    fwd_pkts: int
    bwd_pkts: int
    fwd_bytes: int
    bwd_bytes: int
    duration_ms: float
    # TCP flags seen in either direction (count when known, else 0/1)
    syn: int = 0
    fin: int = 0
    rst: int = 0
    # Packet-level statistics. NaN when the source does not report them
    # (Zeek conn.log and Suricata flow events do not).
    psh: float = NAN
    min_ps: float = NAN
    max_ps: float = NAN
    std_ps: float = NAN
    mean_piat_ms: float = NAN
    std_piat_ms: float = NAN
    max_piat_ms: float = NAN
    # True when a signature IDS already alerted on this flow
    sig_alerted: bool = False
    source: str = ""


FIELD_NAMES = [f.name for f in fields(FlowRecord)]
_INT_FIELDS = {
    "src_port", "dst_port", "proto", "fwd_pkts", "bwd_pkts",
    "fwd_bytes", "bwd_bytes", "syn", "fin", "rst",
}
_STR_FIELDS = {"src_ip", "dst_ip", "source"}
_BOOL_FIELDS = {"sig_alerted"}


# ---------------------------------------------------------------------------
# CSV storage
# ---------------------------------------------------------------------------

def open_text(path, mode: str = "rt"):
    """Open a text file, transparently handling .gz."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def _cell(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return repr(round(value, 3))
    return str(value)


def write_flows(path, records: Iterable[FlowRecord], append: bool = False) -> int:
    """Write records to CSV (optionally .gz). Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not (append and path.exists() and path.stat().st_size > 0)
    n = 0
    with open_text(path, "at" if append else "wt") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELD_NAMES)
        for r in records:
            w.writerow([_cell(getattr(r, name)) for name in FIELD_NAMES])
            n += 1
    return n


def read_flows(path) -> Iterator[FlowRecord]:
    """Read records written by write_flows()."""
    with open_text(path, "rt") as f:
        for row in csv.DictReader(f):
            kwargs = {}
            for name in FIELD_NAMES:
                raw = row.get(name, "")
                if name in _STR_FIELDS:
                    kwargs[name] = raw or ""
                elif name in _BOOL_FIELDS:
                    kwargs[name] = raw in ("1", "True", "true")
                elif name in _INT_FIELDS:
                    kwargs[name] = int(float(raw)) if raw else 0
                else:
                    kwargs[name] = float(raw) if raw else NAN
            yield FlowRecord(**kwargs)


class FlowWriter:
    """Appends flows to one CSV per UTC day: <dir>/flows-YYYYMMDD.csv."""

    def __init__(self, directory, retention_days: Optional[int] = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._last_prune = 0.0

    def write(self, records: Sequence[FlowRecord]) -> None:
        by_day: dict[str, list[FlowRecord]] = {}
        for r in records:
            day = time.strftime("%Y%m%d", time.gmtime(r.ts_end if r.ts_end > 0 else 0))
            by_day.setdefault(day, []).append(r)
        for day, recs in by_day.items():
            write_flows(self.dir / f"flows-{day}.csv", recs, append=True)
        self._prune()

    def _prune(self) -> None:
        if not self.retention_days or time.time() - self._last_prune < 3600:
            return
        self._last_prune = time.time()
        cutoff = time.strftime(
            "%Y%m%d", time.gmtime(time.time() - self.retention_days * 86400)
        )
        for p in self.dir.glob("flows-????????.csv"):
            if p.stem.split("-")[1] < cutoff:
                p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Column view
# ---------------------------------------------------------------------------

def records_to_columns(records: Sequence[FlowRecord]) -> dict:
    """List of records -> dict of numpy arrays, one per field."""
    cols = {}
    for name in FIELD_NAMES:
        values = [getattr(r, name) for r in records]
        if name in _STR_FIELDS:
            cols[name] = np.array(values, dtype=object)
        elif name in _BOOL_FIELDS:
            cols[name] = np.array(values, dtype=bool)
        elif name in _INT_FIELDS:
            cols[name] = np.array(values, dtype=np.int64)
        else:
            cols[name] = np.array(values, dtype=np.float64)
    return cols


# ---------------------------------------------------------------------------
# Address classification
# ---------------------------------------------------------------------------

# Address ranges treated as "ours" without configuration. Listed explicitly:
# Python's is_private also covers documentation and benchmarking ranges.
# 100.64.0.0/10 is carrier-grade NAT space, which overlay VPNs such as
# Tailscale hand out to peers; it is never a public internet host.
_INTERNAL = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
    "127.0.0.0/8", "100.64.0.0/10", "fc00::/7", "fe80::/10", "::1/128",
)]


class Locality:
    """Classifies an address as internal ("i"), external ("e") or
    multicast/broadcast ("m"). Results are cached per address."""

    def __init__(self, home_networks: Sequence[str] = ()):
        self._nets = list(_INTERNAL)
        for n in home_networks or ():
            self._nets.append(ipaddress.ip_network(str(n), strict=False))
        self._cache: dict[str, str] = {}

    def kind(self, ip: str) -> str:
        k = self._cache.get(ip)
        if k is None:
            k = self._classify(ip)
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[ip] = k
        return k

    def kinds(self, ips: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(ips.astype(str), return_inverse=True)
        return np.array([self.kind(u) for u in uniq], dtype=object)[inv]

    def _classify(self, ip: str) -> str:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return "e"
        if addr.is_multicast or ip == "255.255.255.255":
            return "m"
        internal = any(addr.version == net.version and addr in net for net in self._nets)
        if internal and addr.version == 4 and ip.endswith(".255"):
            return "m"  # directed broadcast on one of our subnets
        return "i" if internal else "e"
