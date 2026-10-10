import json

import yaml

from snortml_inference.__main__ import main
from snortml_inference.flows import write_flows

from conftest import T0, normal_traffic
from test_engine import port_scan


def test_fit_info_and_replay(tmp_path, capsys, baseline_flows, unseen_flows):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "scoring": {"min_group_flows": 400},
        "alerts": {"file": "alerts/alerts.log", "json_file": "alerts/alerts.jsonl"},
    }))
    train = tmp_path / "train.csv.gz"
    write_flows(train, baseline_flows)

    assert main(["fit", "--config", str(config), "--flows", str(train)]) == 0
    out = capsys.readouterr().out
    assert "Baseline covers" in out and "Flow layer: budget" in out
    assert (tmp_path / "baseline" / "manifest.json").exists()  # relative to the config

    assert main(["info", "--config", str(config)]) == 0
    assert "tcp/443" in capsys.readouterr().out

    test = tmp_path / "test.csv.gz"
    scan = port_scan(T0 + 86400 + 1800)
    write_flows(test, sorted(unseen_flows + scan, key=lambda r: r.ts_end))
    alerts = tmp_path / "out.jsonl"
    assert main(["replay", "--config", str(config), "--source", str(test),
                 "--json", str(alerts)]) == 0
    out = capsys.readouterr().out
    assert "host anomaly" in out and "alerts emitted:" in out
    events = [json.loads(l) for l in alerts.read_text().splitlines()]
    assert events and all(e["event_type"] == "anomaly" for e in events)
    assert any(e["layer"] == "host" and e["src_ip"] == "192.168.1.11" for e in events)
    # a replay never writes to the daemon's own alert files
    assert not (tmp_path / "alerts").exists()


def test_helpful_errors(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    assert main(["info", "--config", str(config)]) == 2
    assert "baseline --source" in capsys.readouterr().out
    assert main(["fit", "--config", str(config)]) == 2
    assert "No recorded flows" in capsys.readouterr().out
    small = tmp_path / "small.csv"
    write_flows(small, normal_traffic(seed=3, hours=0.02))
    assert main(["fit", "--config", str(config), "--flows", str(small)]) == 1
    assert "Cannot build a baseline" in capsys.readouterr().out


def test_old_style_config_still_loads(tmp_path):
    from snortml_inference.config import load_config

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "interface": "eth0", "models_dir": "models",
        "capture": {"min_packets": 5},
        "alerts": {"file": "alerts/x.log", "score_override": 0.2},
        "performance": {"threads": 4},
    }))
    cfg = load_config(str(config))
    assert cfg["source"] == {"type": "live", "interface": "eth0", "path": None, "follow": True}
    assert cfg["capture"]["min_packets"] == 5 and cfg["capture"]["idle_timeout"] == 10
    assert "score_override" not in cfg["alerts"]
    assert cfg["alerts"]["file"] == str(tmp_path / "alerts" / "x.log")
