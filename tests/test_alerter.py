import json

from snortml_inference.alerter import Alert, AlertManager, severity_from_depth


def _cfg(tmp_path=None, **alerts):
    a = {"console": False, "file": None, "json_file": None}
    if tmp_path is not None:
        a.update(file=str(tmp_path / "alerts.log"), json_file=str(tmp_path / "alerts.jsonl"))
    a.update(alerts)
    return {"alerts": a}


def _flow(ts=1000.0, severity="LOW", kind="upload", server="1.1.1.1", **kw):
    return Alert(ts=ts, layer="flow", kind=kind, severity=severity, client="10.0.0.5",
                 server=server, client_port=50000, server_port=443, proto=6,
                 context="tcp/443", p=1e-6, alpha=1e-4, reasons=["bytes to server 50 MB"], **kw)


def test_severity_steps():
    assert severity_from_depth(4.2, 4.0) == "LOW"
    assert severity_from_depth(5.2, 4.0) == "MEDIUM"
    assert severity_from_depth(6.9, 4.0) == "HIGH"
    assert severity_from_depth(12.0, 4.0) == "CRITICAL"


def test_repeats_fold_into_one_incident_until_the_window_expires():
    am = AlertManager(_cfg(suppress_seconds=3600))
    assert am.submit(_flow(ts=0))
    for i in range(1, 60):  # one slice a minute of the same connection
        assert not am.submit(_flow(ts=60.0 * i))
    assert am.counts["emitted"] == 1 and am.counts["repeats"] == 59
    assert am.submit(_flow(ts=3601))  # a new hour is a new incident


def test_more_severe_repeat_is_reported_again():
    am = AlertManager(_cfg())
    am.keep_history = True
    assert am.submit(_flow(ts=0, severity="LOW"))
    assert not am.submit(_flow(ts=10, severity="LOW"))
    assert am.submit(_flow(ts=20, severity="HIGH"))
    assert am.emitted[-1].repeats == 2
    assert not am.submit(_flow(ts=30, severity="MEDIUM"))


def test_different_servers_and_kinds_are_different_incidents():
    am = AlertManager(_cfg())
    assert am.submit(_flow(server="1.1.1.1"))
    assert am.submit(_flow(server="2.2.2.2"))
    scan = Alert(ts=1000, layer="host", kind="fan-out", severity="HIGH", client="10.0.0.5")
    upload = Alert(ts=1001, layer="host", kind="upload", severity="LOW", client="10.0.0.5")
    assert am.submit(scan) and am.submit(upload)


def test_mute_and_minimum_severity():
    am = AlertManager(_cfg(mute=["download"], min_severity="MEDIUM"))
    assert not am.submit(_flow(kind="download", severity="CRITICAL"))
    assert not am.submit(_flow(kind="upload", severity="LOW"))
    assert am.submit(_flow(kind="upload", severity="MEDIUM", server="9.9.9.9"))
    assert am.counts["muted"] == 1 and am.counts["below_min_severity"] == 1


def test_hourly_cap_holds_back_a_storm(tmp_path):
    am = AlertManager(_cfg(tmp_path, max_per_hour=5))
    sent = sum(am.submit(_flow(ts=float(i), server=f"1.1.1.{i}")) for i in range(40))
    assert sent == 5 and am.counts["rate_capped"] == 35
    text = (tmp_path / "alerts.log").read_text()
    assert text.count("[NOTICE]") == 1 and "baseline" in text
    assert am.submit(_flow(ts=4000.0, server="8.8.8.8"))  # next hour: flowing again


def test_outputs(tmp_path):
    am = AlertManager(_cfg(tmp_path))
    am.submit(_flow(ts=1_750_000_000.25, signature_flagged=True))
    am.submit(Alert(ts=1_750_000_060, layer="host", kind="fan-out", severity="CRITICAL",
                    client="10.0.0.5", p=1e-5, alpha=1e-3,
                    reasons=["distinct destination ports 1,000 (usually 1 to 4)"]))
    line = (tmp_path / "alerts.log").read_text().splitlines()[0]
    assert line.isascii()
    assert "[LOW] flow anomaly (upload)" in line and "10.0.0.5:50000 -> 1.1.1.1:443" in line
    assert "1 in 1.0 million" in line and "also flagged by signature IDS" in line
    flow, host = [json.loads(l) for l in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert flow["event_type"] == "anomaly" and flow["layer"] == "flow"
    assert (flow["src_ip"], flow["dest_ip"], flow["dest_port"], flow["proto"]) == (
        "10.0.0.5", "1.1.1.1", 443, "TCP")
    assert flow["one_in"] == 1_000_000 and flow["threshold_one_in"] == 10_000
    assert flow["timestamp"].startswith("2025-06-15T") and flow["timestamp"].endswith("+0000")
    assert host["layer"] == "host" and "dest_ip" not in host and host["severity"] == "CRITICAL"
