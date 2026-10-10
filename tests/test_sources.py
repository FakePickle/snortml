import gzip
import json

from snortml_inference.flows import FlowRecord, Locality, read_flows, write_flows
from snortml_inference.sources import (
    guess_kind, iter_suricata_eve, iter_zeek_conn, parse_time,
)

ZEEK_TSV = (
    "#separator \\x09\n"
    "#set_separator\t,\n"
    "#empty_field\t(empty)\n"
    "#unset_field\t-\n"
    "#path\tconn\n"
    "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\t"
    "duration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\t"
    "history\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents\n"
    "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\tcount\tcount\t"
    "string\tbool\tbool\tcount\tstring\tcount\tcount\tcount\tcount\tset[string]\n"
    "1600000000.500000\tCabc\t10.0.0.5\t51000\t93.184.216.34\t443\ttcp\tssl\t2.500000\t"
    "1200\t54000\tSF\tT\tF\t0\tShADadfF\t14\t1768\t42\t56188\t-\n"
    "1600000003.000000\tCdef\t10.0.0.5\t40000\t8.8.8.8\t53\tudp\tdns\t-\t-\t-\tS0\tT\tF\t0\t"
    "D\t1\t70\t0\t0\t-\n"
    "#close\t2020-09-13-12-30-00\n"
)


def test_zeek_tsv(tmp_path):
    p = tmp_path / "conn.log"
    p.write_text(ZEEK_TSV)
    flows = list(iter_zeek_conn(str(p)))
    assert len(flows) == 2
    f = flows[0]
    assert (f.src_ip, f.src_port, f.dst_ip, f.dst_port, f.proto) == (
        "10.0.0.5", 51000, "93.184.216.34", 443, 6)
    assert (f.fwd_pkts, f.bwd_pkts) == (14, 42)
    assert (f.fwd_bytes, f.bwd_bytes) == (1768, 56188)  # IP-level byte counts
    assert f.duration_ms == 2500.0 and f.ts_end == 1600000003.0
    assert (f.syn, f.fin, f.rst) == (1, 1, 0)
    assert f.max_ps != f.max_ps  # NaN: Zeek has no per-packet statistics
    unset = flows[1]  # "-" fields become zero
    assert unset.proto == 17 and unset.duration_ms == 0 and unset.bwd_pkts == 0


def test_zeek_json(tmp_path):
    p = tmp_path / "conn.json"
    row = {"ts": 1600000000.0, "id.orig_h": "fe80::1", "id.orig_p": 3, "id.resp_h": "ff02::1",
           "id.resp_p": 135, "proto": "icmp", "duration": 1.0, "orig_pkts": 3,
           "orig_ip_bytes": 200, "resp_pkts": 0, "resp_ip_bytes": 0, "history": ""}
    p.write_text(json.dumps(row) + "\n")
    (f,) = iter_zeek_conn(str(p))
    assert f.proto == 58  # ICMP over IPv6
    assert f.src_port == 0 and f.dst_port == 0  # ICMP type/code are not ports


def test_suricata_flow_events(tmp_path):
    p = tmp_path / "eve.json"
    flow = {
        "timestamp": "2026-06-30T18:26:16.000000+0530", "event_type": "flow",
        "src_ip": "10.0.0.5", "src_port": 51000, "dest_ip": "93.184.216.34",
        "dest_port": 443, "proto": "TCP",
        "flow": {"pkts_toserver": 14, "pkts_toclient": 42, "bytes_toserver": 1768,
                 "bytes_toclient": 56188, "start": "2026-06-30T18:26:10.000000+0530",
                 "end": "2026-06-30T18:26:15.500000+0530", "alerted": True},
        "tcp": {"syn": True, "fin": True, "psh": True, "ack": True},
    }
    other = {"timestamp": "2026-06-30T18:26:16.000000+0530", "event_type": "dns",
             "src_ip": "10.0.0.5", "dns": {"rrname": "flow.example"}}
    p.write_text(json.dumps(other) + "\n" + json.dumps(flow) + "\nnot json\n")
    (f,) = iter_suricata_eve(str(p))
    assert f.sig_alerted is True
    assert (f.fwd_pkts, f.bwd_pkts, f.fwd_bytes, f.bwd_bytes) == (14, 42, 1768, 56188)
    assert abs(f.duration_ms - 5500.0) < 1e-6
    assert (f.syn, f.fin, f.rst) == (1, 1, 0)


def test_parse_time_handles_offsets():
    a = parse_time("2026-06-30T18:26:15.000000+0530")
    b = parse_time("2026-06-30T12:56:15.000000+0000")
    assert abs(a - b) < 1e-6


def test_csv_roundtrip(tmp_path):
    recs = [
        FlowRecord(1.5, 2.5, "10.0.0.1", "1.1.1.1", 1234, 443, 6, 3, 4, 500, 600, 1000.0,
                   syn=1, psh=2.0, min_ps=54.0, max_ps=1500.0, std_ps=10.5,
                   mean_piat_ms=1.0, std_piat_ms=2.0, max_piat_ms=3.0, source="nfstream"),
        FlowRecord(3.0, 4.0, "10.0.0.2", "2.2.2.2", 1, 53, 17, 1, 1, 70, 90, 12.0,
                   sig_alerted=True, source="zeek"),
    ]
    p = tmp_path / "flows.csv.gz"
    assert write_flows(p, recs) == 2
    assert gzip.open(p, "rt").readline().startswith("ts_start,")
    back = list(read_flows(p))
    assert back[0] == recs[0]
    assert back[1].sig_alerted is True and back[1].max_ps != back[1].max_ps  # NaN kept


def test_guess_kind(tmp_path):
    for name, kind in (("x.pcap", "pcap"), ("flows.csv", "csv"), ("eve.json", "suricata"),
                       ("conn.log", "zeek")):
        (tmp_path / name).write_text("")
        assert guess_kind(str(tmp_path / name)) == kind
    assert guess_kind("eth0") == "live"


def test_locality():
    loc = Locality(["2a04:7f80::/32"])
    assert loc.kind("192.168.1.5") == "i"
    assert loc.kind("100.100.1.2") == "i"  # overlay VPN (CGNAT range)
    assert loc.kind("8.8.8.8") == "e"
    assert loc.kind("203.0.113.9") == "e"  # documentation range is not "ours"
    assert loc.kind("224.0.0.251") == "m" and loc.kind("192.168.1.255") == "m"
    assert loc.kind("fe80::1") == "i" and loc.kind("2a04:7f80:1::5") == "i"
    assert loc.kind("2001:4860:4860::8888") == "e"
