"""
snortml_inference/alerter.py
==============================
Formats and writes anomaly alerts to console and/or file.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROTO_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}

SEVERITY_LEVELS = [
    (0.95, "CRITICAL"),
    (0.80, "HIGH"),
    (0.60, "MEDIUM"),
    (0.00, "LOW"),
]


class Alerter:
    """Writes anomaly alerts to console and optionally a log file."""

    def __init__(self, config: dict):
        self.console = config["alerts"].get("console", True)
        self.alert_file = config["alerts"].get("file")
        self.threshold = None  # set after scorer loads

        if self.alert_file:
            Path(self.alert_file).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Alert file: {self.alert_file}")

        self._alert_count = 0

    def alert(self, features: dict, score: float, threshold: float) -> None:
        """Format and emit an alert for an anomalous flow."""
        self._alert_count += 1
        meta = features.get("_meta", {})
        severity = self._severity(score, threshold)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        proto = PROTO_NAMES.get(meta.get("protocol", 0), "UNK")

        # One-line alert format
        alert_line = (
            f"[{timestamp}] [{severity}] "
            f"SNORTML+ ANOMALY DETECTED | "
            f"score={score:.4f} threshold={threshold:.4f} | "
            f"{meta.get('src_ip', '?')}:{meta.get('src_port', '?')} -> "
            f"{meta.get('dst_ip', '?')}:{meta.get('dst_port', '?')} "
            f"proto={proto} "
            f"pkts={meta.get('packets', '?')} "
            f"bytes={meta.get('bytes', '?')} "
            f"dur={meta.get('duration_ms', '?')}ms"
        )

        if self.console:
            # Colour code by severity in terminal
            color = {
                "CRITICAL": "\033[91m",  # red
                "HIGH": "\033[93m",  # yellow
                "MEDIUM": "\033[94m",  # blue
                "LOW": "\033[0m",  # default
            }.get(severity, "\033[0m")
            print(f"{color}{alert_line}\033[0m")

        if self.alert_file:
            with open(self.alert_file, "a", encoding="utf-8") as f:
                f.write(alert_line + "\n")

    def summary(self) -> None:
        """Print session summary."""
        print(f"\n[SnortML+] Session complete. Total alerts: {self._alert_count}")

    @staticmethod
    def _severity(score: float, threshold: float) -> str:
        """Map score to severity label based on how far above threshold."""
        ratio = score / max(threshold, 1e-8)
        if ratio >= 3.0:
            return "CRITICAL"
        elif ratio >= 2.0:
            return "HIGH"
        elif ratio >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"
