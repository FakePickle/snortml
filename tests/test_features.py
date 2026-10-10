import numpy as np

from snortml_inference.features import (
    ContextMap, active_days, block_split, build_flow_table, flow_parent, host_parent,
    payload_estimate,
)
from snortml_inference.flows import FlowRecord, Locality, records_to_columns
from snortml_inference.windows import WindowAggregator, window_arrays


def _rec(src, sport, dst, dport, proto=6, fp=10, bp=12, fb=1000, bb=9000, ts=100.0, **kw):
    return FlowRecord(ts - 1, ts, src, dst, sport, dport, proto, fp, bp, fb, bb, 1000.0, **kw)


def test_fallback_chains():
    assert flow_parent("tcp/443") == "tcp/other"
    assert flow_parent("tcp/other") == "other"
    assert flow_parent("icmp") == "other"
    assert flow_parent("other") is None
    assert host_parent("host:10.0.0.1|i") == "pool:i"
    assert host_parent("pool:i") == "pool" and host_parent("pool") is None


def test_context_learns_services_not_ephemeral_ports():
    rng = np.random.default_rng(0)
    n = 2000
    proto = np.full(n, 6)
    sport = rng.integers(49152, 65535, n)
    dport = np.where(np.arange(n) < 1500, 443, rng.integers(20000, 40000, n))
    ctx = ContextMap.learn(proto, sport, dport, min_flows=500)
    assert list(ctx.ports[6]) == [443]
    keys, swap, server_port = ctx.assign(proto, sport, dport)
    assert set(keys[:1500]) == {"tcp/443"} and set(keys[1500:]) == {"tcp/other"}
    assert not swap.any()


def test_orientation_flips_reply_first_flows():
    ctx = ContextMap({6: {443: 1000}})
    recs = [
        _rec("10.0.0.5", 50000, "93.184.216.34", 443, fp=10, bp=40, fb=900, bb=50000),
        # the same conversation seen server-first (a mid-stream slice)
        _rec("93.184.216.34", 443, "10.0.0.5", 50000, fp=40, bp=10, fb=50000, bb=900),
    ]
    ft = build_flow_table(records_to_columns(recs), ctx, Locality())
    assert list(ft.client) == ["10.0.0.5", "10.0.0.5"]
    assert list(ft.server_port) == [443, 443]
    np.testing.assert_array_equal(ft.fwd_bytes, [900, 900])
    np.testing.assert_array_equal(ft.bwd_bytes, [50000, 50000])
    np.testing.assert_allclose(ft.Xnum[0], ft.Xnum[1])
    assert list(ft.Xcat[1]) == ["ie", "ie"]


def test_only_answered_flows_with_enough_packets_are_scored():
    ctx = ContextMap({6: {443: 1000}})
    recs = [
        _rec("10.0.0.5", 50000, "1.1.1.1", 443, fp=3, bp=0, fb=180, bb=0),  # SYN retries
        _rec("10.0.0.5", 50001, "1.1.1.1", 443, fp=1, bp=1),  # too small
        _rec("10.0.0.5", 50002, "1.1.1.1", 443, fp=2, bp=1),
    ]
    ft = build_flow_table(records_to_columns(recs), ctx, Locality(), min_packets=3)
    assert list(ft.eligible) == [False, False, True]


def test_payload_estimate_ignores_bare_acks():
    # 20,000 ACKs of 54 bytes each carry no data; 1,000 full packets do
    assert payload_estimate(np.array([20000 * 54.0]), np.array([20000.0]))[0] == 0
    assert payload_estimate(np.array([1000 * 1514.0]), np.array([1000.0]))[0] > 1_000_000


def test_block_split_and_active_days():
    ts = np.arange(0, 6000, 1.0)
    split = block_split(ts)
    shares = np.bincount(split) / split.size
    assert abs(shares[0] - 0.6) < 0.05 and abs(shares[1] - 0.2) < 0.05
    # two one-hour captures a day apart are two hours of traffic, not a day
    two = np.concatenate([np.arange(0, 3600.0), np.arange(86400, 90000.0)])
    assert abs(active_days(two) * 24 - 2.0) < 0.2


def test_window_aggregation_counts_targets():
    ctx = ContextMap({6: {443: 1000}})
    recs = [_rec("10.0.0.5", 40000 + i, "192.168.1.9", 1000 + i, fp=1, bp=0, fb=58, bb=0,
                 ts=30.0) for i in range(50)]
    recs += [_rec("10.0.0.5", 41000 + i, "1.1.1.1", 443, fp=3, bp=0, fb=180, bb=0, ts=40.0)
             for i in range(17)]  # 17 retries to ONE dead server
    recs += [_rec("10.0.0.5", 42000, "1.1.1.1", 443, fp=1000, bp=500, fb=1_500_000, bb=27000,
                  ts=50.0)]
    ft = build_flow_table(records_to_columns(recs), ctx, Locality())
    agg = WindowAggregator(60.0, 0.0)
    agg.add_table(ft)
    (w,) = agg.flush(None)
    assert w.entity == "10.0.0.5" and w.n_flows == 68
    assert w.n_dsts == 2 and w.n_ports == 51
    assert w.n_unanswered == 51  # 50 scanned ports + one dead server, not 67
    assert 1_300_000 < w.payload_out < 1_500_000
    keys, X, ts = window_arrays([w])
    assert keys[0] == "host:10.0.0.5|i" and X.shape == (1, 5) and ts[0] == 60.0


def test_window_watermark_and_late_flows():
    ctx = ContextMap({})
    agg = WindowAggregator(60.0, grace_seconds=15.0)
    first = build_flow_table(records_to_columns([_rec("10.0.0.5", 1, "1.1.1.1", 2, ts=30.0)]),
                             ctx, Locality())
    agg.add_table(first)
    assert agg.flush(70.0) == []  # window 0 ends at 60; grace not over
    assert len(agg.flush(80.0)) == 1
    late = build_flow_table(records_to_columns([_rec("10.0.0.5", 1, "1.1.1.1", 2, ts=59.0)]),
                            ctx, Locality())
    agg.add_table(late)  # belongs to a closed window: counted in the open one
    (w,) = agg.flush(None)
    assert w.t_end == 120.0 and w.n_flows == 1
