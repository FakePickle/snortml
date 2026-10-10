"""End-to-end behaviour of the engine on a synthetic network."""

import copy
import random

import pytest

from snortml_inference.alerter import AlertManager
from snortml_inference.baseline import FitConfig, fit_bundle
from snortml_inference.config import DEFAULTS
from snortml_inference.daemon import Engine, replay
from snortml_inference.flows import NAN, FlowRecord
from snortml_inference.scorer import Scorer

from conftest import T0, normal_traffic

DAY2 = T0 + 86400
ATTACKER = "192.168.1.11"


def _cfg(**sections):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["alerts"].update({"console": False, "file": None, "json_file": None})
    for name, values in sections.items():
        cfg[name].update(values)
    return cfg


def _run(bundle, flows, **sections):
    cfg = _cfg(**sections)
    am = AlertManager(cfg)
    am.keep_history = True
    engine = Engine(bundle, cfg, alerts=am)
    replay(engine, sorted(flows, key=lambda r: r.ts_end))
    return engine, am


def _tiny(ts, dst, dport, answered=False):
    return FlowRecord(ts, ts, ATTACKER, dst, random.randint(40000, 60000), dport, 6,
                      1, 1 if answered else 0, 58, 54 if answered else 0, 0.0, syn=1,
                      psh=0, min_ps=58, max_ps=58, std_ps=0, mean_piat_ms=0,
                      std_piat_ms=0, max_piat_ms=0, source="nfstream")


def port_scan(ts):
    rng = random.Random(5)
    return [_tiny(ts + rng.uniform(0, 30), "192.168.1.50", port)
            for port in rng.sample(range(1, 5000), 800)]


def upload(ts, slices=6, sig=False):
    """One long TLS connection pushing ~45 MB a minute, as 60-second slices."""
    return [FlowRecord(ts + 60 * i, ts + 60 * (i + 1), ATTACKER, "45.33.32.156", 51000, 443, 6,
                       31000, 15500, 45_000_000, 930_000, 60000.0, psh=1500, min_ps=54,
                       max_ps=1514, std_ps=690, mean_piat_ms=1.3, std_piat_ms=3,
                       max_piat_ms=40, sig_alerted=sig, source="nfstream")
            for i in range(slices)]


def download(ts):
    return [FlowRecord(ts, ts + 60, ATTACKER, "45.33.32.156", 51001, 443, 6,
                       15500, 31000, 930_000, 45_000_000, 60000.0, psh=1500, min_ps=54,
                       max_ps=1514, std_ps=690, mean_piat_ms=1.3, std_piat_ms=3,
                       max_piat_ms=40, source="nfstream")]


def test_unseen_normal_traffic_stays_within_budget(bundle, unseen_flows):
    engine, am = _run(bundle, unseen_flows)
    assert engine.stats["scored"] > 5000
    assert am.counts["emitted"] <= 2  # three hours at a budget of ten a day


def test_port_scan_is_caught_by_the_host_layer_and_as_new_services(bundle, unseen_flows):
    _, am = _run(bundle, unseen_flows + port_scan(DAY2 + 1800))
    host = [a for a in am.emitted if a.layer == "host" and a.client == ATTACKER]
    assert host and host[0].kind in ("fan-out", "unanswered")
    assert host[0].severity in ("HIGH", "CRITICAL")
    assert any("destination" in r for r in host[0].reasons)
    novel = [a for a in am.emitted if a.layer == "new-service"]
    assert novel and novel[-1].severity == "HIGH"
    assert len(novel) <= 3  # hundreds of new ports, a handful of alerts


def test_upload_is_caught_and_its_slices_fold_into_one_incident(bundle, unseen_flows):
    _, am = _run(bundle, unseen_flows + upload(DAY2 + 1800, slices=10))
    flow = [a for a in am.emitted if a.layer == "flow" and a.server == "45.33.32.156"]
    assert 1 <= len(flow) <= 2 and flow[0].kind == "upload"
    assert any("bytes to server" in r for r in flow[0].reasons)
    assert flow[0].detail["bytes_toserver"] == 45_000_000
    assert am.counts["repeats"] >= 8
    assert any(a.layer == "host" and a.kind == "upload" for a in am.emitted)


def test_internet_downloads_are_muted_unless_asked_for(bundle, unseen_flows):
    flows = unseen_flows + download(DAY2 + 1800)
    _, quiet = _run(bundle, flows)
    assert not [a for a in quiet.emitted if a.server == "45.33.32.156"]
    assert quiet.counts["muted"] >= 1
    _, loud = _run(bundle, flows, alerts={"mute": []})
    assert [a.kind for a in loud.emitted if a.server == "45.33.32.156"] == ["download"]


def test_flows_a_signature_ids_flagged_are_left_to_it(bundle, unseen_flows):
    flows = unseen_flows + upload(DAY2 + 1800, sig=True)
    engine, am = _run(bundle, flows)
    assert not [a for a in am.emitted if a.layer == "flow" and a.server == "45.33.32.156"]
    assert engine.stats["left_to_signatures"] >= 1
    _, tagged = _run(bundle, flows, ids={"on_match": "tag"})
    hit = [a for a in tagged.emitted if a.layer == "flow" and a.server == "45.33.32.156"]
    assert hit and hit[0].signature_flagged


def test_new_service_alert_and_allow_list(bundle, unseen_flows):
    ssh = [_tiny(DAY2 + 1800 + i, "45.33.32.156", 22, answered=True) for i in range(3)]
    _, am = _run(bundle, unseen_flows + ssh)
    novel = [a for a in am.emitted if a.layer == "new-service"]
    assert len(novel) == 1 and novel[0].context == "tcp/22" and novel[0].severity == "LOW"
    assert "never used in the baseline" in novel[0].reasons[0]
    _, allowed = _run(bundle, unseen_flows + ssh, novelty={"allow": ["tcp/22"]})
    assert not [a for a in allowed.emitted if a.layer == "new-service"]
    _, off = _run(bundle, unseen_flows + ssh, novelty={"enabled": False})
    assert not [a for a in off.emitted if a.layer == "new-service"]


def test_unanswered_probes_from_outside_are_not_new_services(bundle, unseen_flows):
    probes = [FlowRecord(DAY2 + 1800 + i, DAY2 + 1800 + i, "45.33.32.156", "192.168.1.10",
                         40000 + i, 23, 6, 1, 0, 58, 0, 0.0, syn=1, source="nfstream")
              for i in range(50)]
    _, am = _run(bundle, unseen_flows + probes)
    assert not [a for a in am.emitted if a.layer == "new-service"]


def test_batch_size_does_not_change_the_result(bundle, unseen_flows):
    flows = sorted(unseen_flows + port_scan(DAY2 + 1800) + upload(DAY2 + 3000),
                   key=lambda r: r.ts_end)

    def keys(batch):
        cfg = _cfg()
        am = AlertManager(cfg)
        am.keep_history = True
        replay(Engine(bundle, cfg, alerts=am), flows, batch_size=batch)
        return sorted((a.layer, a.kind, a.client, a.server, a.severity) for a in am.emitted)

    assert keys(64) == keys(5000)


def test_budget_sets_the_threshold(bundle):
    strict = Scorer(bundle, alerts_per_day=1)
    loose = Scorer(bundle, alerts_per_day=100)
    assert strict.alpha_flow < loose.alpha_flow
    assert strict.alpha_host < loose.alpha_host
    no_margin = Scorer(bundle, alerts_per_day=10, drift_margin=1)
    assert no_margin.alpha_flow >= Scorer(bundle, alerts_per_day=10).alpha_flow


def test_baseline_from_flow_logs_without_packet_statistics():
    """Zeek and Suricata give no per-packet statistics: the model uses what exists."""
    flows = normal_traffic(seed=7, hours=3)
    for r in flows:
        r.psh = r.min_ps = r.max_ps = r.std_ps = NAN
        r.mean_piat_ms = r.std_piat_ms = r.max_piat_ms = NAN
        r.source = "zeek"
    b = fit_bundle(flows, FitConfig(min_group_flows=400))
    assert b.source_kind == "zeek"
    assert b.flow.feature_names == ["log_duration", "log_fwd_pkts", "log_bwd_pkts",
                                    "log_fwd_bytes", "log_bwd_bytes", "tcp_state", "locality"]
    attack = upload(T0 + 5000, slices=3)
    for r in attack:
        r.psh = r.min_ps = r.max_ps = r.std_ps = NAN
        r.mean_piat_ms = r.std_piat_ms = r.max_piat_ms = NAN
    _, am = _run(b, normal_traffic(seed=8, hours=1, start=T0 + 4000) + attack)
    assert any(a.kind == "upload" for a in am.emitted)


def test_too_little_data_is_refused():
    with pytest.raises(ValueError, match="Record for longer"):
        fit_bundle(normal_traffic(seed=1, hours=0.05))


def test_signature_flagged_flows_stay_out_of_every_layer_and_of_the_baseline(
        bundle, baseline_flows, unseen_flows):
    """The hybrid contract: what the rules caught is theirs. It is not scored,
    not counted per host, not reported as a new service, and not learned."""
    scan = port_scan(DAY2 + 1800)
    for r in scan:
        r.sig_alerted = True
    engine, am = _run(bundle, unseen_flows + scan)
    assert engine.stats["left_to_signatures"] == len(scan)
    assert not [a for a in am.emitted if a.client == ATTACKER]
    # the same scan, missed by the rules, is ours to catch
    for r in scan:
        r.sig_alerted = False
    _, am = _run(bundle, unseen_flows + scan)
    host = [a for a in am.emitted if a.layer == "host" and a.client == ATTACKER]
    assert host and host[0].resembles == "port scan"

    flagged = port_scan(T0 + 3000)
    for r in flagged:
        r.sig_alerted = True
    b = fit_bundle(baseline_flows + flagged, FitConfig(min_group_flows=400))
    assert b.stats["flows_excluded_signature"] == len(flagged)
    assert b.stats["flows_total"] == len(baseline_flows)


def test_alerts_say_what_the_pattern_resembles(bundle, unseen_flows):
    _, am = _run(bundle, unseen_flows + upload(DAY2 + 1800, slices=3))
    flow = [a for a in am.emitted if a.layer == "flow" and a.server == "45.33.32.156"]
    assert flow[0].resembles == "possible data exfiltration"
    assert "resembles: possible data exfiltration" in flow[0].text()
    assert flow[0].to_json()["resembles"] == "possible data exfiltration"
