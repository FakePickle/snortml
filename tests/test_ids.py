import json

from snortml_inference.flows import FlowRecord
from snortml_inference.ids import SignatureIndex


def _rec(src="10.0.0.5", sport=50000, dst="1.2.3.4", dport=80, start=1000.0, end=1010.0):
    return FlowRecord(start, end, src, dst, sport, dport, 6, 5, 5, 500, 500, 10000.0)


def test_suricata_alert_marks_the_flow_in_either_direction():
    idx = SignatureIndex()
    alert = {"timestamp": "1970-01-01T00:16:45.000000+0000", "event_type": "alert",
             "src_ip": "1.2.3.4", "src_port": 80, "dest_ip": "10.0.0.5", "dest_port": 50000,
             "proto": "TCP", "alert": {"signature": "ET MALWARE test"}}
    assert idx.load_line(json.dumps(alert))
    assert not idx.load_line(json.dumps({"event_type": "dns", "src_ip": "10.0.0.5"}))
    assert idx.matches(_rec())  # reversed 5-tuple still matches
    assert not idx.matches(_rec(dport=443))
    assert not idx.matches(_rec(start=90000.0, end=90010.0))  # hours later: unrelated


def test_snort_alert_json():
    idx = SignatureIndex()
    line = json.dumps({"seconds": 1005, "proto": "TCP", "src_addr": "10.0.0.5",
                       "src_port": 50000, "dst_addr": "1.2.3.4", "dst_port": 80,
                       "msg": "test rule"})
    assert idx.load_line(line) and idx.matches(_rec())
    assert not idx.load_line("garbage") and not idx.load_line("[1, 2]")


def test_load_file(tmp_path):
    p = tmp_path / "eve.json"
    rows = [{"event_type": "alert", "timestamp": "1970-01-01T00:16:45.000000+0000",
             "src_ip": "10.0.0.5", "src_port": 50000 + i, "dest_ip": "1.2.3.4",
             "dest_port": 80, "proto": "TCP"} for i in range(3)]
    rows.append({"event_type": "flow", "src_ip": "10.0.0.5"})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    idx = SignatureIndex()
    assert idx.load_file(str(p)) == 3
