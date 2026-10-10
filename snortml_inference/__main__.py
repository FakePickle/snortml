"""
snortml_inference/__main__.py
==============================
Command line:

    python -m snortml_inference baseline --source eth0 --duration 86400
    python -m snortml_inference run
    python -m snortml_inference replay --source capture.pcap
    python -m snortml_inference fit
    python -m snortml_inference info
    python -m snortml_inference interfaces
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="config.yaml",
                   help="path to config.yaml (default: config.yaml)")


def _add_source(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", help="interface name, pcap, Zeek conn.log, "
                                    "Suricata eve.json or recorded flow CSV")
    p.add_argument("--type", choices=["live", "pcap", "zeek", "suricata", "csv"],
                   help="input type (guessed from the file name if omitted)")


def _resolve_source(args, cfg):
    """(kind, target) from the command line, falling back to the config."""
    from snortml_inference.sources import guess_kind

    if getattr(args, "source", None):
        target = args.source
        kind = args.type or guess_kind(target)
        return kind, target
    src = cfg["source"]
    kind = src.get("type", "live")
    return kind, (src.get("interface") if kind == "live" else src.get("path"))


def _fit_config(cfg):
    from snortml_inference.baseline import FitConfig

    cap, sc = cfg["capture"], cfg["scoring"]
    return FitConfig(
        min_packets=int(cap["min_packets"]),
        min_group_flows=int(sc["min_group_flows"]),
        window_seconds=float(sc["window_seconds"]),
        home_networks=list(cfg.get("home_networks") or []),
        autoencoder=bool(sc["autoencoder"]),
        idle_timeout=int(cap["idle_timeout"]),
        active_timeout=int(cap["active_timeout"]),
        alerts_per_day=float(sc["alerts_per_day"]),
        flow_share=float(sc["flow_share"]),
        drift_margin=float(sc["drift_margin"]),
    )


def _fit_and_save(records, cfg) -> int:
    from snortml_inference.baseline import describe_bundle, fit_bundle, save_bundle

    print(f"Fitting baseline on {len(records):,} flows...")
    t = time.time()
    try:
        bundle = fit_bundle(records, _fit_config(cfg))
    except ValueError as e:
        print(f"Cannot build a baseline: {e}")
        return 1
    out = save_bundle(bundle, cfg["baseline_dir"])
    sc = cfg["scoring"]
    print(f"Saved to {out} ({time.time() - t:.0f} s)\n")
    print(describe_bundle(bundle, float(sc["alerts_per_day"]), float(sc["flow_share"]),
                          float(sc["drift_margin"])))
    return 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_baseline(args) -> int:
    """Record traffic from a source, then fit a baseline from it."""
    import threading

    from snortml_inference.config import load_config
    from snortml_inference.flows import FlowWriter
    from snortml_inference.ids import SignatureIndex
    from snortml_inference.sources import open_source

    cfg = load_config(args.config)
    kind, target = _resolve_source(args, cfg)
    if not target:
        print("No input. Use --source <interface or file>.")
        return 2
    cap = cfg["capture"]
    live = kind == "live"
    if live and not args.duration:
        print("Recording until Ctrl+C. Use --duration SECONDS to stop automatically.")
    print(f"Reading {kind} input: {target}")
    writer = None
    if not args.no_save and kind != "csv":
        writer = FlowWriter(cfg["record"]["dir"])
    # Signature alerts written while (or before) the traffic was recorded:
    # flows they cover are marked, and fitting leaves them out of "normal".
    sig, stop = None, threading.Event()
    ids_paths = [p for p in ((cfg.get("ids") or {}).get("suricata_eve"),
                             (cfg.get("ids") or {}).get("snort_json")) if p]
    if ids_paths and kind != "suricata":
        sig = SignatureIndex()
        for path in ids_paths:
            if live:
                sig.follow(path, stop)
                print(f"Following signature alerts in {path}")
            elif os.path.exists(path):
                print(f"Loaded {sig.load_file(path):,} signature alerts from {path}")
    records, pending = [], []
    t0 = last_note = time.time()
    try:
        for rec in open_source(kind, target, idle_timeout=int(cap["idle_timeout"]),
                               active_timeout=int(cap["active_timeout"]), follow=False):
            if sig is not None and not rec.sig_alerted and sig.matches(rec):
                rec.sig_alerted = True
            records.append(rec)
            pending.append(rec)
            now = time.time()
            if len(pending) >= 5000 or (live and now - last_note >= 30):
                if writer:
                    writer.write(pending)
                pending = []
                if now - last_note >= 30:
                    last_note = now
                    print(f"  {len(records):,} flows after {now - t0:.0f} s")
            if live and args.duration and now - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nStopped recording.")
    stop.set()
    if writer and pending:
        writer.write(pending)
    if writer:
        print(f"Recorded flows kept in {cfg['record']['dir']} (used again by 'fit').")
    return _fit_and_save(records, cfg)


def cmd_fit(args) -> int:
    """Fit a baseline from flows recorded earlier (by 'baseline' or the daemon)."""
    from snortml_inference.config import load_config
    from snortml_inference.flows import read_flows

    cfg = load_config(args.config)
    if args.flows:
        paths = sorted({p for pattern in args.flows for p in glob.glob(pattern)})
    else:
        paths = sorted(glob.glob(os.path.join(cfg["record"]["dir"], "flows-*.csv*")))
        if args.days:
            cutoff = time.strftime("%Y%m%d", time.gmtime(time.time() - args.days * 86400))
            recent = [p for p in paths
                      if Path(p).name.split("-")[1][:8] >= cutoff]
            paths = recent or paths
    if not paths:
        print("No recorded flows found. Run 'baseline' first, or pass --flows FILES.")
        return 2
    records = []
    for p in paths:
        records.extend(read_flows(p))
        print(f"  {p}: {len(records):,} flows so far")
    return _fit_and_save(records, cfg)


def cmd_run(args) -> int:
    """Score live traffic (or a growing log file) and raise alerts."""
    from snortml_inference.config import load_config
    from snortml_inference.daemon import run_daemon

    cfg = load_config(args.config)
    if getattr(args, "interface", None):
        cfg["source"]["type"] = "live"
        cfg["source"]["interface"] = args.interface
    elif getattr(args, "source", None):
        kind, target = _resolve_source(args, cfg)
        cfg["source"]["type"] = kind
        cfg["source"]["interface" if kind == "live" else "path"] = target
    return run_daemon(cfg)


def cmd_replay(args) -> int:
    """Score a recorded capture or log offline and print what would have alerted."""
    from snortml_inference.alerter import AlertManager
    from snortml_inference.baseline import load_bundle
    from snortml_inference.config import load_config
    from snortml_inference.daemon import Engine, replay
    from snortml_inference.ids import SignatureIndex
    from snortml_inference.sources import open_source

    cfg = load_config(args.config)
    kind, target = _resolve_source(args, cfg)
    if not target or not os.path.exists(str(target)):
        print("Replay needs a file: --source capture.pcap | conn.log | eve.json | flows.csv")
        return 2
    cfg["source"]["type"] = kind
    if args.alerts_per_day:
        cfg["scoring"]["alerts_per_day"] = args.alerts_per_day
    cfg["alerts"].update({"file": None, "json_file": args.json, "webhook": None,
                          "console": not args.quiet})
    try:
        bundle = load_bundle(args.baseline or cfg["baseline_dir"])
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 2
    sig = None
    if args.ids_alerts:
        sig = SignatureIndex()
        print(f"Loaded {sig.load_file(args.ids_alerts):,} signature alerts "
              f"from {args.ids_alerts}")
    alerts = AlertManager(cfg)
    engine = Engine(bundle, cfg, sig_index=sig, alerts=alerts)
    cap = cfg["capture"]
    flows = open_source(kind, target, idle_timeout=int(cap["idle_timeout"]),
                        active_timeout=int(cap["active_timeout"]), follow=False)
    result = replay(engine, flows)
    hours = result["seconds"] / 3600.0
    print()
    print(engine.summary())
    if hours >= 1:
        per_day = alerts.counts["emitted"] / hours * 24.0
        print(f"Input spans {hours:.1f} hours: about {per_day:.1f} alerts per day "
              f"at this rate (budget {cfg['scoring']['alerts_per_day']}).")
    else:
        print(f"Input spans {hours * 60:.0f} minutes: too short to project a daily rate.")
    return 0


def cmd_info(args) -> int:
    """Describe the current baseline."""
    from snortml_inference.baseline import describe_bundle, load_bundle
    from snortml_inference.config import load_config

    cfg = load_config(args.config)
    try:
        bundle = load_bundle(cfg["baseline_dir"])
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 2
    sc = cfg["scoring"]
    print(f"Baseline: {cfg['baseline_dir']}")
    print(f"Created : {time.strftime('%Y-%m-%d %H:%M', time.localtime(bundle.created))}"
          f" from '{bundle.source_kind}' flows")
    print(describe_bundle(bundle, float(sc["alerts_per_day"]), float(sc["flow_share"]),
                          float(sc["drift_margin"])))
    return 0


def cmd_interfaces(_args) -> int:
    """List capture interfaces."""
    print("Available interfaces:")
    if sys.platform.startswith("win"):
        import subprocess

        script = ("Get-NetAdapter | ForEach-Object { '{0}|{1}|{2}' -f "
                  "$_.Name, $_.Status, $_.InterfaceGuid }")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception as e:
            print(f"  could not query adapters: {e}")
            return 1
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3:
                name, status, guid = parts
                print(f"  {name} ({status})")
                print(f"      use: \\Device\\NPF_{guid}")
        print("Put the \\Device\\NPF_{...} value in config.yaml under source.interface.")
        return 0
    try:
        import psutil

        names = sorted(psutil.net_if_addrs())
    except ImportError:
        names = sorted(os.listdir("/sys/class/net")) if os.path.isdir("/sys/class/net") else []
    for name in names:
        print(f"  {name}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m snortml_inference",
        description="Network anomaly detection calibrated on your own traffic.")
    # flags of the first version of the daemon still work without a subcommand
    _add_common(parser)
    parser.add_argument("--interface", help=argparse.SUPPRESS)
    parser.add_argument("--list-interfaces", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("baseline", help="record traffic and fit a baseline from it")
    _add_common(p)
    _add_source(p)
    p.add_argument("--duration", type=float, default=0,
                   help="seconds to record from a live interface (default: until Ctrl+C)")
    p.add_argument("--no-save", action="store_true",
                   help="do not keep the recorded flows on disk")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("fit", help="re-fit the baseline from recorded flows")
    _add_common(p)
    p.add_argument("--flows", nargs="+", help="flow CSV files (default: the record directory)")
    p.add_argument("--days", type=float, default=14,
                   help="use recorded flows from the last N days (default: 14)")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("run", help="score live traffic and raise alerts")
    _add_common(p)
    _add_source(p)
    p.add_argument("--interface", help="capture interface (overrides the config)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("replay", help="score a recorded capture or log offline")
    _add_common(p)
    _add_source(p)
    p.add_argument("--baseline", help="baseline directory (default: from the config)")
    p.add_argument("--alerts-per-day", type=float, help="override the alert budget")
    p.add_argument("--ids-alerts", help="Suricata eve.json or Snort alert_json file; "
                                        "flows it alerted on are left to the IDS")
    p.add_argument("--json", help="also write alerts to this JSON-lines file")
    p.add_argument("--quiet", action="store_true", help="print only the summary")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("info", help="describe the current baseline")
    _add_common(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("interfaces", help="list capture interfaces")
    p.set_defaults(func=cmd_interfaces)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "list_interfaces", False):
        return cmd_interfaces(args)
    if not args.command:
        return cmd_run(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
