"""
snortml_inference/ids.py
=========================
Knows which connections a signature IDS has already alerted on, so the
anomaly layer can leave them alone and report only what signatures missed.

Two alert formats are followed as they are written:

    Suricata eve.json   events with "event_type": "alert"
    Snort 3 alert_json  one JSON object per alert (src_addr, dst_addr, ...)

When flows themselves come from Suricata, none of this is needed: each flow
event already carries "alerted": true/false.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from snortml_inference.flows import FlowRecord
from snortml_inference.sources import _PROTO_NUM, iter_lines, parse_time

logger = logging.getLogger(__name__)


class SignatureIndex:
    """Recently alerted connections, matched in either direction."""

    def __init__(self, ttl_seconds: float = 900.0):
        self.ttl = float(ttl_seconds)
        self._seen: dict = {}
        self._lock = threading.Lock()
        self._last_prune = 0.0
        self.added = 0

    @staticmethod
    def _key(proto, ip_a, port_a, ip_b, port_b) -> tuple:
        a, b = (str(ip_a), int(port_a or 0)), (str(ip_b), int(port_b or 0))
        return (int(proto), a, b) if a <= b else (int(proto), b, a)

    def add(self, proto, src_ip, src_port, dst_ip, dst_port, ts: float) -> None:
        key = self._key(proto, src_ip, src_port, dst_ip, dst_port)
        with self._lock:
            self._seen[key] = max(self._seen.get(key, 0.0), float(ts))
            self.added += 1

    def matches(self, rec: FlowRecord) -> bool:
        """True if a signature alert fell inside (or shortly around) the flow."""
        key = self._key(rec.proto, rec.src_ip, rec.src_port, rec.dst_ip, rec.dst_port)
        with self._lock:
            ts = self._seen.get(key)
            if rec.ts_end - self._last_prune > 60:
                self._last_prune = rec.ts_end
                cutoff = rec.ts_end - self.ttl
                for k in [k for k, v in self._seen.items() if v < cutoff]:
                    del self._seen[k]
        if ts is None:
            return False
        return rec.ts_start - self.ttl <= ts <= rec.ts_end + self.ttl

    # ------------------------------------------------------------------

    def load_line(self, line: str) -> bool:
        """Parse one alert line in either format. Returns True if it was an alert."""
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(ev, dict):
            return False
        if "event_type" in ev:  # Suricata
            if ev.get("event_type") != "alert":
                return False
            proto = _PROTO_NUM.get(str(ev.get("proto", "")).lower(), 0)
            self.add(proto, ev.get("src_ip", ""), ev.get("src_port"),
                     ev.get("dest_ip", ""), ev.get("dest_port"),
                     parse_time(ev.get("timestamp", time.time())))
            return True
        if "src_addr" in ev and "dst_addr" in ev:  # Snort 3 alert_json
            proto = _PROTO_NUM.get(str(ev.get("proto", "")).lower(), 0)
            ts = ev.get("seconds")
            if ts is None:
                ts = time.time()  # Snort's "timestamp" field has no year
            self.add(proto, ev["src_addr"], ev.get("src_port"),
                     ev["dst_addr"], ev.get("dst_port"), float(ts))
            return True
        return False

    def load_file(self, path: str) -> int:
        """Read a whole alert file (for offline replay)."""
        n = 0
        for line in iter_lines(path, follow=False):
            n += self.load_line(line)
        return n

    def follow(self, path: str, stop: threading.Event) -> threading.Thread:
        """Follow an alert file as the IDS writes it."""

        def run():
            try:
                for line in iter_lines(path, follow=True, from_start=False, stop=stop):
                    self.load_line(line)
            except Exception as e:
                logger.error(f"stopped following {path}: {e}")

        t = threading.Thread(target=run, daemon=True, name="ids-follow")
        t.start()
        return t
