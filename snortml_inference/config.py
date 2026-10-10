"""
snortml_inference/config.py
============================
Configuration defaults and loading. Paths in the config file are relative to
the config file's own directory.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULTS = {
    "source": {
        "type": "live",  # live | pcap | zeek | suricata | csv
        "interface": None,  # for type: live
        "path": None,  # for the file-based types
        "follow": True,  # keep reading a log file as it grows
    },
    "capture": {
        "min_packets": 3,
        "idle_timeout": 10,
        "active_timeout": 60,
        "batch_size": 64,
        "max_batch_wait_sec": 5,
    },
    "home_networks": [],
    "baseline_dir": "baseline",
    "scoring": {
        "alerts_per_day": 10,
        "drift_margin": 5,
        "flow_share": 0.7,
        "window_seconds": 60,
        "window_grace_seconds": None,  # None: chosen from the input type
        "min_group_flows": 500,
        "autoencoder": True,
    },
    "novelty": {"enabled": True, "allow": []},
    "alerts": {
        "file": "alerts/alerts.log",
        "json_file": "alerts/alerts.jsonl",
        "console": True,
        "suppress_seconds": 3600,
        "max_per_hour": 30,
        "mute": ["download"],
        "min_severity": "LOW",
        "webhook": None,
    },
    "ids": {"suricata_eve": None, "snort_json": None, "on_match": "suppress"},
    "record": {"enabled": True, "dir": "data", "retention_days": 30},
    "logging": {"level": "INFO", "file": "logs/daemon.log"},
}

_PATH_KEYS = [
    ("baseline_dir",), ("alerts", "file"), ("alerts", "json_file"),
    ("record", "dir"), ("logging", "file"), ("source", "path"),
    ("ids", "suricata_eve"), ("ids", "snort_json"),
]


def _merge(base: dict, extra: dict) -> dict:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None) -> dict:
    """Defaults, overlaid with the YAML file if it exists."""
    cfg = copy.deepcopy(DEFAULTS)
    root = Path.cwd()
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        root = Path(path).resolve().parent
        # config files written for the first version of the daemon
        if "interface" in user and "source" not in user:
            user["source"] = {"type": "live", "interface": user.pop("interface")}
        for old in ("models_dir", "performance"):
            user.pop(old, None)
        if isinstance(user.get("alerts"), dict) and "score_override" in user["alerts"]:
            user["alerts"].pop("score_override")
            logger.warning("alerts.score_override is no longer used; the threshold now "
                           "comes from scoring.alerts_per_day")
        _merge(cfg, user)
    elif path:
        logger.warning(f"config file {path} not found; using defaults")
    for keys in _PATH_KEYS:
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        value = node.get(keys[-1])
        if value and not Path(str(value)).is_absolute():
            node[keys[-1]] = str(root / str(value))
    cfg["_root"] = str(root)
    return cfg
