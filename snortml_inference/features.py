"""
snortml_inference/features.py
==============================
Turns flow records into model inputs. The same code runs when a baseline is
fitted and when the daemon scores live traffic.

Three ideas do most of the work:

1. Orientation. Each flow is turned so that "fwd" means client -> server,
   whichever end nfstream happened to see first. A 60-second slice of a long
   download no longer looks like a different kind of flow.
2. Context. Each flow is scored against flows of the same service
   (tcp/443, udp/53, icmp, ...), not against all traffic at once. A ping is
   compared with other pings, so being ICMP is not in itself anomalous.
3. Log scale. Byte, packet and time counts are heavy-tailed; they are used
   as log1p values so a large download is a few units from the median, not
   thousands of standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from snortml_inference.flows import Locality

PROTO_NAMES = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp6", 2: "igmp"}

# Numeric flow features. The last seven need per-packet statistics and are
# NaN for Zeek/Suricata input; the detector drops columns that are all-NaN
# in the baseline.
FLOW_NUMERIC = [
    "log_duration", "log_fwd_pkts", "log_bwd_pkts", "log_fwd_bytes", "log_bwd_bytes",
    "log_min_ps", "log_max_ps", "log_std_ps",
    "log_mean_piat", "log_std_piat", "log_max_piat", "psh_frac",
]
FLOW_CATEGORICAL = ["tcp_state", "locality"]
# Size and length features: only "more than usual" is of interest.
FLOW_VOLUME = [
    "log_duration", "log_fwd_pkts", "log_bwd_pkts", "log_fwd_bytes", "log_bwd_bytes",
]

HOST_NUMERIC = [
    "log_flows", "log_dsts", "log_ports", "log_unanswered", "log_payload_out",
]
HOST_CATEGORICAL: list[str] = []

# name -> (human label, unit) for alert explanations
FEATURE_INFO = {
    "log_duration": ("duration", "ms"),
    "log_fwd_pkts": ("packets to server", "count"),
    "log_bwd_pkts": ("packets from server", "count"),
    "log_fwd_bytes": ("bytes to server", "bytes"),
    "log_bwd_bytes": ("bytes from server", "bytes"),
    "log_min_ps": ("smallest packet", "bytes"),
    "log_max_ps": ("largest packet", "bytes"),
    "log_std_ps": ("packet size spread", "bytes"),
    "log_mean_piat": ("mean gap between packets", "ms"),
    "log_std_piat": ("gap spread", "ms"),
    "log_max_piat": ("longest gap between packets", "ms"),
    "psh_frac": ("share of PSH packets", "frac"),
    "tcp_state": ("connection state", "cat"),
    "locality": ("direction", "cat"),
    "log_flows": ("flows in the window", "count"),
    "log_dsts": ("distinct destinations", "count"),
    "log_ports": ("distinct destination ports", "count"),
    "log_unanswered": ("destinations that never replied", "count"),
    "log_payload_out": ("data sent", "bytes"),
}

# High ports that are worth noticing when a host starts using them for the
# first time (remote access, databases, device management, common malware).
SENSITIVE_PORTS = frozenset({
    1080, 1433, 1521, 2049, 2323, 2375, 3306, 3389, 4444, 5432, 5555, 5900,
    5985, 5986, 6379, 6667, 7547, 8291, 9200, 11211, 27017, 37215,
})


def watched_service(proto: np.ndarray, server_port: np.ndarray) -> np.ndarray:
    """True for TCP/UDP flows whose server port is well-known (< 1024) or on
    the sensitive list. Only these are checked for never-seen-before use."""
    is_port_proto = (proto == 6) | (proto == 17)
    low = (server_port > 0) & (server_port < 1024)
    sensitive = np.isin(server_port, np.fromiter(SENSITIVE_PORTS, dtype=np.int64))
    return is_port_proto & (low | sensitive)


# Bytes of headers assumed per packet when estimating payload. Generous on
# purpose: a stream of bare ACKs (the client side of a big download) must
# come out as zero payload, so that only real uploads count as data sent.
HEADER_BYTES = 90


def payload_estimate(bytes_, pkts):
    return np.maximum(bytes_ - HEADER_BYTES * pkts, 0.0)

_LOCALITY_TEXT = {
    "ie": "internal -> internet", "ii": "internal -> internal",
    "ei": "internet -> internal", "ee": "external -> external",
    "im": "internal -> multicast", "em": "external -> multicast",
    "mi": "multicast -> internal", "me": "multicast -> external", "mm": "multicast",
}


def describe_category(feature: str, value: str) -> str:
    if feature == "locality":
        return _LOCALITY_TEXT.get(value, value)
    if feature == "tcp_state":
        if value.startswith("U"):
            return "no reply" if value.endswith("0") else "two-way"
        parts = ["SYN seen" if value.startswith("S") else "no SYN (mid-stream)"]
        if "F" in value:
            parts.append("FIN")
        if "R" in value:
            parts.append("RST")
        if value.endswith("0"):
            parts.append("no reply")
        return ", ".join(parts)
    return value


# ---------------------------------------------------------------------------
# Context groups
# ---------------------------------------------------------------------------

def flow_parent(key: str) -> Optional[str]:
    """Fallback chain for flow contexts: tcp/443 -> tcp/other -> other."""
    if key == "other":
        return None
    if "/" in key and not key.endswith("/other"):
        return key.split("/", 1)[0] + "/other"
    return "other"


def host_parent(key: str) -> Optional[str]:
    """Fallback chain for host contexts: host:1.2.3.4|i -> pool:i -> pool."""
    if key == "pool":
        return None
    if key.startswith("host:"):
        return "pool:" + key.rsplit("|", 1)[-1]
    return "pool"


class ContextMap:
    """Learns which (protocol, port) pairs are services on this network and
    uses that to orient flows and assign them to context groups."""

    def __init__(self, ports: dict):
        # proto -> {port: baseline flow count}
        self.ports = {
            int(p): {int(k): int(v) for k, v in d.items()} for p, d in ports.items()
        }
        self._tbl = {}
        for proto, d in self.ports.items():
            t = np.zeros(65536, dtype=np.int64)
            for port, count in d.items():
                if 0 < port < 65536:
                    t[port] = count
            self._tbl[proto] = t

    @classmethod
    def learn(cls, proto: np.ndarray, sport: np.ndarray, dport: np.ndarray,
              min_flows: int) -> "ContextMap":
        """A port is a service if at least min_flows flows have it as their
        more-frequently-seen end. Ephemeral client ports never reach that."""
        ports = {}
        sport = np.clip(sport, 0, 65535)
        dport = np.clip(dport, 0, 65535)
        for p in (6, 17):
            m = proto == p
            if not m.any():
                continue
            s, d = sport[m], dport[m]
            seen = np.bincount(s, minlength=65536) + np.bincount(d, minlength=65536)
            seen[0] = 0
            service = np.where(seen[s] > seen[d], s, d)
            assigned = np.bincount(service, minlength=65536)
            assigned[0] = 0
            popular = np.nonzero(assigned >= min_flows)[0]
            ports[p] = {int(k): int(assigned[k]) for k in popular}
        return cls(ports)

    def assign(self, proto: np.ndarray, sport: np.ndarray, dport: np.ndarray):
        """Returns (group keys, swap mask, server port). swap is True where
        the record's src is really the server side."""
        n = proto.shape[0]
        sport = np.clip(sport, 0, 65535)
        dport = np.clip(dport, 0, 65535)
        swap = np.zeros(n, dtype=bool)
        server_port = dport.astype(np.int64).copy()
        code = proto.astype(np.int64) * 100_000  # no service port known
        for p in (6, 17):
            m = proto == p
            if not m.any():
                continue
            s, d = sport[m], dport[m]
            tbl = self._tbl.get(p)
            if tbl is None:
                cs = np.zeros(s.shape[0], dtype=np.int64)
                cd = cs
            else:
                cs, cd = tbl[s], tbl[d]
            unknown = (cs == 0) & (cd == 0)
            # neither end is a known service: a low source port talking to a
            # high destination port is a reply seen first
            sw = (cs > cd) | (unknown & (s > 0) & (s < 1024) & (d >= 1024))
            sp = np.where(sw, s, d)
            swap[m] = sw
            server_port[m] = sp
            code[m] = p * 100_000 + np.where(unknown, 0, sp + 1)
        uniq, inv = np.unique(code, return_inverse=True)
        names = []
        for c in uniq:
            p, rest = divmod(int(c), 100_000)
            base = PROTO_NAMES.get(p, f"ip{p}")
            if p in (6, 17):
                names.append(f"{base}/{rest - 1}" if rest else f"{base}/other")
            else:
                names.append(base)
        keys = np.array(names, dtype=object)[inv]
        return keys, swap, server_port


# ---------------------------------------------------------------------------
# Flow features
# ---------------------------------------------------------------------------

@dataclass
class FlowTable:
    """Oriented flows plus their model inputs."""

    n: int
    keys: np.ndarray  # context group key per flow
    client: np.ndarray
    server: np.ndarray
    client_port: np.ndarray
    server_port: np.ndarray
    proto: np.ndarray
    fwd_pkts: np.ndarray
    bwd_pkts: np.ndarray
    fwd_bytes: np.ndarray
    bwd_bytes: np.ndarray
    ts_end: np.ndarray
    client_kind: np.ndarray  # "i" / "e" / "m"
    Xnum: np.ndarray
    Xcat: list
    eligible: np.ndarray  # scored by the flow layer: enough packets, and answered


_TCP_STATE = [
    ("S" if c & 8 else "-") + ("F" if c & 4 else "") + ("R" if c & 2 else "")
    + ("0" if c & 1 else "")
    for c in range(16)
]


def build_flow_table(cols: dict, ctx: ContextMap, locality: Locality,
                     min_packets: int = 3) -> FlowTable:
    """cols is the column dict from flows.records_to_columns()."""
    proto = cols["proto"]
    n = proto.shape[0]
    keys, swap, server_port = ctx.assign(proto, cols["src_port"], cols["dst_port"])

    def pick(a, b):
        return np.where(swap, b, a)

    client = pick(cols["src_ip"], cols["dst_ip"])
    server = pick(cols["dst_ip"], cols["src_ip"])
    client_port = pick(cols["src_port"], cols["dst_port"])
    fwd_pkts = pick(cols["fwd_pkts"], cols["bwd_pkts"]).astype(np.float64)
    bwd_pkts = pick(cols["bwd_pkts"], cols["fwd_pkts"]).astype(np.float64)
    fwd_bytes = pick(cols["fwd_bytes"], cols["bwd_bytes"]).astype(np.float64)
    bwd_bytes = pick(cols["bwd_bytes"], cols["fwd_bytes"]).astype(np.float64)
    total_pkts = fwd_pkts + bwd_pkts

    def log1p(a):
        return np.log1p(np.maximum(a, 0.0))  # NaN passes through

    with np.errstate(invalid="ignore"):
        Xnum = np.column_stack([
            log1p(cols["duration_ms"]),
            log1p(fwd_pkts),
            log1p(bwd_pkts),
            log1p(fwd_bytes),
            log1p(bwd_bytes),
            log1p(cols["min_ps"]),
            log1p(cols["max_ps"]),
            log1p(cols["std_ps"]),
            log1p(cols["mean_piat_ms"]),
            log1p(cols["std_piat_ms"]),
            log1p(cols["max_piat_ms"]),
            cols["psh"] / np.maximum(total_pkts, 1.0),
        ]) if n else np.zeros((0, len(FLOW_NUMERIC)))

    is_tcp = proto == 6
    noreply = (bwd_pkts == 0).astype(np.int64)
    tcp_code = (
        (cols["syn"] > 0).astype(np.int64) * 8
        + (cols["fin"] > 0).astype(np.int64) * 4
        + (cols["rst"] > 0).astype(np.int64) * 2
        + noreply
    )
    tcp_state = np.where(
        is_tcp,
        np.array(_TCP_STATE, dtype=object)[tcp_code],
        np.where(noreply == 1, "U0", "U"),
    ).astype(object)

    client_kind = locality.kinds(client) if n else np.array([], dtype=object)
    server_kind = locality.kinds(server) if n else np.array([], dtype=object)
    loc = np.array([a + b for a, b in zip(client_kind, server_kind)], dtype=object)

    return FlowTable(
        n=n, keys=keys, client=client, server=server,
        client_port=client_port, server_port=server_port, proto=proto,
        fwd_pkts=fwd_pkts, bwd_pkts=bwd_pkts,
        fwd_bytes=fwd_bytes, bwd_bytes=bwd_bytes,
        ts_end=cols["ts_end"], client_kind=client_kind,
        Xnum=Xnum, Xcat=[tcp_state, loc],
        # Unanswered flows (SYN retries to a dead server, scans, multicast
        # announcements) carry no shape to model; the host layer counts them.
        eligible=(total_pkts >= min_packets) & (fwd_pkts > 0) & (bwd_pkts > 0),
    )


# ---------------------------------------------------------------------------
# Train / calibration / validation split
# ---------------------------------------------------------------------------

def block_split(ts: np.ndarray, block_seconds: Optional[float] = None) -> np.ndarray:
    """Assign each row to fit (0), calibration (1) or validation (2) by time
    block, in a repeating 3:1:1 pattern. Whole blocks keep slices of the same
    long-lived flow together, and interleaving keeps every part of the day in
    all three sets."""
    if ts.size == 0:
        return np.zeros(0, dtype=np.int64)
    if block_seconds is None:
        # about 60 blocks of actual traffic; hour-long blocks once the
        # baseline covers days
        block_seconds = min(max(active_days(ts) * 86400.0 / 60.0, 5.0), 3600.0)
    block = np.floor((ts - np.nanmin(ts)) / block_seconds).astype(np.int64)
    role = np.array([0, 0, 0, 1, 2], dtype=np.int64)
    return role[block % 5]


def active_days(ts: np.ndarray, bucket_seconds: float = 300.0) -> float:
    """Length of time actually covered by the data, in days: the number of
    distinct 5-minute buckets that contain traffic. Gaps between captures and
    captures with relative timestamps do not inflate it."""
    if ts.size == 0:
        return 0.0
    span = float(np.nanmax(ts) - np.nanmin(ts))
    if span < bucket_seconds:
        return max(span, 1.0) / 86400.0
    buckets = np.unique(np.floor(ts / bucket_seconds))
    return buckets.size * bucket_seconds / 86400.0
