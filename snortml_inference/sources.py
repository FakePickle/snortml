"""
snortml_inference/sources.py
=============================
Input sources. Each yields FlowRecord objects:

    live / pcap  nfstream (its own capture on an interface, or a pcap file)
    zeek         Zeek conn.log, TSV or JSON
    suricata     Suricata eve.json, "flow" events
    csv          flows previously recorded by this tool

Zeek and Suricata are how the daemon sits downstream of an existing sensor
instead of capturing packets a second time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Iterator, Optional

from snortml_inference.flows import FlowRecord, open_text, read_flows

logger = logging.getLogger(__name__)

SOURCE_KINDS = ("live", "pcap", "zeek", "suricata", "csv")

_PROTO_NUM = {
    "tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58, "ipv6-icmp": 58,
    "igmp": 2, "sctp": 132, "gre": 47, "esp": 50,
}


# ---------------------------------------------------------------------------
# nfstream (live interface or pcap file)
# ---------------------------------------------------------------------------

def iter_nfstream(
    source: str,
    idle_timeout: int = 10,
    active_timeout: int = 60,
    bpf_filter: Optional[str] = None,
) -> Iterator[FlowRecord]:
    """Flows from a network interface or a pcap file."""
    try:
        from nfstream import NFStreamer
    except ImportError as e:
        raise ImportError(
            "nfstream is not installed. Run: pip install nfstream\n"
            "Windows also requires Npcap: https://npcap.com/#download"
        ) from e

    streamer = NFStreamer(
        source=source,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
        statistical_analysis=True,
        splt_analysis=0,
        n_dissections=0,  # no payload dissection: we only use flow statistics
        bpf_filter=bpf_filter,
    )
    for f in streamer:
        yield FlowRecord(
            ts_start=f.bidirectional_first_seen_ms / 1000.0,
            ts_end=f.bidirectional_last_seen_ms / 1000.0,
            src_ip=f.src_ip,
            dst_ip=f.dst_ip,
            src_port=int(f.src_port or 0),
            dst_port=int(f.dst_port or 0),
            proto=int(f.protocol or 0),
            fwd_pkts=int(f.src2dst_packets or 0),
            bwd_pkts=int(f.dst2src_packets or 0),
            fwd_bytes=int(f.src2dst_bytes or 0),
            bwd_bytes=int(f.dst2src_bytes or 0),
            duration_ms=float(f.bidirectional_duration_ms or 0),
            syn=int(f.bidirectional_syn_packets or 0),
            fin=int(f.bidirectional_fin_packets or 0),
            rst=int(f.bidirectional_rst_packets or 0),
            psh=float(f.bidirectional_psh_packets or 0),
            min_ps=float(f.bidirectional_min_ps or 0),
            max_ps=float(f.bidirectional_max_ps or 0),
            std_ps=float(f.bidirectional_stddev_ps or 0),
            mean_piat_ms=float(f.bidirectional_mean_piat_ms or 0),
            std_piat_ms=float(f.bidirectional_stddev_piat_ms or 0),
            max_piat_ms=float(f.bidirectional_max_piat_ms or 0),
            source="nfstream",
        )


# ---------------------------------------------------------------------------
# Line reader with optional tail -f behaviour
# ---------------------------------------------------------------------------

def iter_lines(path: str, follow: bool = False, from_start: bool = True,
               stop=None, poll: float = 0.5) -> Iterator[str]:
    """Yield lines from a file. With follow=True, keep waiting for new lines
    and reopen the file if it is rotated or truncated."""
    if not follow:
        with open_text(path, "rt") as f:
            for line in f:
                yield line
        return

    f = None
    inode = None
    try:
        while stop is None or not stop.is_set():
            if f is None:
                try:
                    f = open(path, "r", encoding="utf-8", errors="replace")
                    inode = os.fstat(f.fileno()).st_ino
                    if not from_start:
                        f.seek(0, os.SEEK_END)
                except FileNotFoundError:
                    time.sleep(poll)
                    continue
            line = f.readline()
            if line:
                if line.endswith("\n"):
                    yield line
                else:
                    # partial line: wait for the writer to finish it
                    pos = f.tell() - len(line.encode("utf-8", errors="replace"))
                    time.sleep(poll)
                    f.seek(max(pos, 0))
                continue
            time.sleep(poll)
            try:
                st = os.stat(path)
                if st.st_ino != inode or st.st_size < f.tell():
                    f.close()
                    f = None
                    from_start = True  # read a rotated-in file from its start
            except FileNotFoundError:
                f.close()
                f = None
    finally:
        if f is not None:
            f.close()


# ---------------------------------------------------------------------------
# Zeek conn.log
# ---------------------------------------------------------------------------

def _num(value, default=0.0) -> float:
    if value is None or value == "-" or value == "" or value == "(empty)":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _zeek_record(row: dict) -> Optional[FlowRecord]:
    try:
        src_ip = row["id.orig_h"]
        dst_ip = row["id.resp_h"]
    except KeyError:
        return None
    proto_name = str(row.get("proto", "")).lower()
    proto = _PROTO_NUM.get(proto_name, 0)
    if proto == 1 and ":" in str(src_ip):
        proto = 58
    ts = row.get("ts")
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        ts = parse_time(ts)  # JSON logs written with ISO-8601 timestamps
    dur_s = _num(row.get("duration"))
    # IP-level byte counts include headers; payload counts are the fallback
    fwd_bytes = _num(row.get("orig_ip_bytes"), -1)
    if fwd_bytes < 0:
        fwd_bytes = _num(row.get("orig_bytes"))
    bwd_bytes = _num(row.get("resp_ip_bytes"), -1)
    if bwd_bytes < 0:
        bwd_bytes = _num(row.get("resp_bytes"))
    history = str(row.get("history") or "")
    is_port_proto = proto in (6, 17)
    return FlowRecord(
        ts_start=ts,
        ts_end=ts + dur_s,
        src_ip=str(src_ip),
        dst_ip=str(dst_ip),
        src_port=int(_num(row.get("id.orig_p"))) if is_port_proto else 0,
        dst_port=int(_num(row.get("id.resp_p"))) if is_port_proto else 0,
        proto=proto,
        fwd_pkts=int(_num(row.get("orig_pkts"))),
        bwd_pkts=int(_num(row.get("resp_pkts"))),
        fwd_bytes=int(fwd_bytes),
        bwd_bytes=int(bwd_bytes),
        duration_ms=dur_s * 1000.0,
        syn=1 if ("S" in history or "s" in history) else 0,
        fin=1 if ("F" in history or "f" in history) else 0,
        rst=1 if ("R" in history or "r" in history) else 0,
        source="zeek",
    )


def iter_zeek_conn(path: str, follow: bool = False, from_start: bool = True,
                   stop=None) -> Iterator[FlowRecord]:
    """Flows from a Zeek conn.log in TSV (with #fields header) or JSON form."""
    names = None
    sep = "\t"
    for line in iter_lines(path, follow=follow, from_start=from_start, stop=stop):
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
        elif line.startswith("#"):
            if line.startswith("#separator"):
                token = line.split(" ", 1)[1] if " " in line else "\\x09"
                sep = token.encode().decode("unicode_escape")
            elif line.startswith("#fields"):
                names = line.split(sep)[1:]
            continue
        else:
            if names is None:
                continue
            row = dict(zip(names, line.split(sep)))
        rec = _zeek_record(row)
        if rec is not None:
            yield rec


# ---------------------------------------------------------------------------
# Suricata eve.json
# ---------------------------------------------------------------------------

def parse_time(value) -> float:
    """ISO-8601 timestamp (as written by Suricata and Snort) -> epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def _suricata_record(ev: dict) -> Optional[FlowRecord]:
    flow = ev.get("flow")
    if ev.get("event_type") != "flow" or not isinstance(flow, dict):
        return None
    proto_name = str(ev.get("proto", "")).lower()
    proto = _PROTO_NUM.get(proto_name, 0)
    start = parse_time(flow.get("start", ev.get("timestamp", 0)))
    end = parse_time(flow.get("end", ev.get("timestamp", 0)))
    tcp = ev.get("tcp") or {}
    return FlowRecord(
        ts_start=start,
        ts_end=end,
        src_ip=str(ev.get("src_ip", "")),
        dst_ip=str(ev.get("dest_ip", "")),
        src_port=int(ev.get("src_port") or 0),
        dst_port=int(ev.get("dest_port") or 0),
        proto=proto,
        fwd_pkts=int(flow.get("pkts_toserver") or 0),
        bwd_pkts=int(flow.get("pkts_toclient") or 0),
        fwd_bytes=int(flow.get("bytes_toserver") or 0),
        bwd_bytes=int(flow.get("bytes_toclient") or 0),
        duration_ms=max(end - start, 0.0) * 1000.0,
        syn=1 if tcp.get("syn") else 0,
        fin=1 if tcp.get("fin") else 0,
        rst=1 if tcp.get("rst") else 0,
        sig_alerted=bool(flow.get("alerted", False)),
        source="suricata",
    )


def iter_suricata_eve(path: str, follow: bool = False, from_start: bool = True,
                      stop=None) -> Iterator[FlowRecord]:
    """Flows from the "flow" events of a Suricata eve.json. Flows Suricata
    already alerted on arrive with sig_alerted=True."""
    for line in iter_lines(path, follow=follow, from_start=from_start, stop=stop):
        if '"flow"' not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec = _suricata_record(ev)
        if rec is not None:
            yield rec


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def open_source(kind: str, target: str, *, idle_timeout: int = 10,
                active_timeout: int = 60, follow: bool = False,
                from_start: bool = True, stop=None) -> Iterator[FlowRecord]:
    """Return a flow iterator for the given source kind and target."""
    if kind in ("live", "pcap"):
        return iter_nfstream(target, idle_timeout=idle_timeout,
                             active_timeout=active_timeout)
    if kind == "zeek":
        return iter_zeek_conn(target, follow=follow, from_start=from_start, stop=stop)
    if kind == "suricata":
        return iter_suricata_eve(target, follow=follow, from_start=from_start, stop=stop)
    if kind == "csv":
        return read_flows(target)
    raise ValueError(f"unknown source type {kind!r}; expected one of {SOURCE_KINDS}")


def guess_kind(target: str) -> str:
    """Infer the source kind from a file name; anything else is an interface."""
    name = os.path.basename(target).lower()
    if not os.path.exists(target):
        return "live"
    if name.endswith((".pcap", ".pcapng", ".cap")):
        return "pcap"
    if name.endswith((".csv", ".csv.gz")):
        return "csv"
    if "eve" in name and name.endswith((".json", ".json.gz", ".log")):
        return "suricata"
    if "conn" in name:
        return "zeek"
    with open_text(target, "rt") as f:
        head = f.readline()
    if head.startswith("#separator") or '"id.orig_h"' in head:
        return "zeek"
    if '"event_type"' in head:
        return "suricata"
    return "csv"
