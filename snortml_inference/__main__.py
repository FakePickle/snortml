"""
snortml_inference/__main__.py
Allows: python -m snortml_inference
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SnortML+ Inference Daemon — Anomaly-based IDS"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--interface", default=None,
        help="Override interface (e.g. eth0, Wi-Fi, wlan0)"
    )
    parser.add_argument(
        "--list-interfaces", action="store_true",
        help="List available network interfaces and exit"
    )
    args = parser.parse_args()

    if args.list_interfaces:
        _list_interfaces()
        sys.exit(0)

    import yaml
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.interface:
        config["interface"] = args.interface

    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=config_path.parent
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    try:
        from snortml_inference.daemon import run
        run(tmp_path)
    finally:
        os.unlink(tmp_path)


def _list_interfaces():
    import subprocess
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-NetAdapter | Select-Object Name,Status | Format-Table -AutoSize"],
        capture_output=True, text=True
    )
    print("\nAvailable interfaces:")
    print(result.stdout)


if __name__ == "__main__":
    main()
