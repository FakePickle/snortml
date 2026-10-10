"""The live path, end to end, in a real process: the daemon follows a growing
Suricata eve.json, scores flows as they are appended, and stops on SIGINT."""

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from snortml_inference.__main__ import main

from conftest import normal_traffic

REPO = Path(__file__).resolve().parent.parent
PROTO = {6: "TCP", 17: "UDP", 1: "ICMP"}


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int(ts % 1 * 1e6):06d}+0000"


def eve_flow(r) -> str:
    ev = {
        "timestamp": _iso(r.ts_end), "event_type": "flow",
        "src_ip": r.src_ip, "src_port": r.src_port, "dest_ip": r.dst_ip,
        "dest_port": r.dst_port, "proto": PROTO[r.proto],
        "flow": {"pkts_toserver": r.fwd_pkts, "pkts_toclient": r.bwd_pkts,
                 "bytes_toserver": r.fwd_bytes, "bytes_toclient": r.bwd_bytes,
                 "start": _iso(r.ts_start), "end": _iso(r.ts_end), "alerted": False},
    }
    if r.proto == 6:
        ev["tcp"] = {"syn": bool(r.syn), "fin": bool(r.fin), "rst": bool(r.rst)}
    return json.dumps(ev)


def _wait_for(predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.25)
    return False


def test_daemon_follows_a_growing_suricata_log(tmp_path):
    eve = tmp_path / "eve.json"
    eve.write_text("")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "source": {"type": "suricata", "path": "eve.json", "follow": True},
        "capture": {"max_batch_wait_sec": 1},
        "scoring": {"window_seconds": 10, "window_grace_seconds": 2, "min_group_flows": 400},
        "record": {"enabled": True, "dir": "data"},
    }))

    # baseline from yesterday's (synthetic) Suricata flow events
    now = time.time()
    history = tmp_path / "eve-history.json"
    with open(history, "w") as f:
        for r in normal_traffic(seed=1, hours=4, start=now - 86400):
            f.write(eve_flow(r) + "\n")
    assert main(["baseline", "--config", str(config), "--source", str(history),
                 "--type", "suricata", "--no-save"]) == 0

    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "snortml_inference", "run", "--config", str(config)],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        log = tmp_path / "logs" / "daemon.log"
        assert _wait_for(lambda: log.exists() and "Listening" in log.read_text(), 30)
        time.sleep(1.5)  # let the reader open the file at its end

        rng = random.Random(4)
        t = time.time()
        with open(eve, "a") as f:
            for r in normal_traffic(seed=9, hours=0.004, start=t - 14):
                f.write(eve_flow(r) + "\n")
            for port in rng.sample(range(1, 6000), 600):  # a port scan, right now
                f.write(json.dumps({
                    "timestamp": _iso(t), "event_type": "flow",
                    "src_ip": "192.168.1.12", "src_port": rng.randint(40000, 60000),
                    "dest_ip": "192.168.1.77", "dest_port": port, "proto": "TCP",
                    "flow": {"pkts_toserver": 1, "pkts_toclient": 0, "bytes_toserver": 58,
                             "bytes_toclient": 0, "start": _iso(t), "end": _iso(t),
                             "alerted": False},
                    "tcp": {"syn": True}}) + "\n")

        alerts = tmp_path / "alerts" / "alerts.jsonl"

        def events():
            if not alerts.exists():
                return []
            return [json.loads(l) for l in alerts.read_text().splitlines() if l.strip()]

        assert _wait_for(lambda: any(e["layer"] == "host" for e in events()), 45), (
            f"no host alert; alerts={events()} log={log.read_text()[-2000:]}")
        got = events()
        host = next(e for e in got if e["layer"] == "host")
        assert host["src_ip"] == "192.168.1.12" and host["kind"] in ("fan-out", "unanswered")
        assert host["severity"] in ("HIGH", "CRITICAL")
        assert any(e["layer"] == "new-service" and e["src_ip"] == "192.168.1.12" for e in got)
        # flows it saw are kept for re-fitting the baseline later
        assert list((tmp_path / "data").glob("flows-*.csv"))
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
    assert proc.returncode == 0, out[-3000:]
    assert "Stopped after" in out and "alerts emitted" in out
