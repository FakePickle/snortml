"""Shared fixtures: a synthetic two-day network and a baseline fitted on it."""

import math
import random

import pytest

from snortml_inference.baseline import FitConfig, fit_bundle
from snortml_inference.flows import FlowRecord

T0 = 1_750_000_000  # arbitrary epoch start
CLIENTS = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13"]
WEB = [f"93.184.{a}.{b}" for a in range(10, 14) for b in range(1, 40)]


def _lognorm(rng, mean, sigma):
    return math.exp(rng.gauss(math.log(mean), sigma))


def web_flow(rng, ts, client, port=443):
    """A request/response exchange: small upload, larger download."""
    fwd_pkts = max(2, int(_lognorm(rng, 9, 0.6)))
    bwd_pkts = max(2, int(fwd_pkts * _lognorm(rng, 1.3, 0.3)))
    fwd_bytes = int(fwd_pkts * _lognorm(rng, 140, 0.4))
    bwd_bytes = int(bwd_pkts * _lognorm(rng, 700, 0.5))
    dur = _lognorm(rng, 900, 1.0)
    full = rng.random() < 0.7  # the rest are mid-stream slices
    return FlowRecord(
        ts_start=ts - dur / 1000, ts_end=ts, src_ip=client, dst_ip=rng.choice(WEB),
        src_port=rng.randint(49152, 65535), dst_port=port, proto=6,
        fwd_pkts=fwd_pkts, bwd_pkts=bwd_pkts, fwd_bytes=fwd_bytes, bwd_bytes=bwd_bytes,
        duration_ms=dur, syn=2 if full else 0, fin=2 if full and rng.random() < 0.8 else 0,
        rst=1 if rng.random() < 0.03 else 0, psh=max(1, fwd_pkts // 3),
        min_ps=54, max_ps=min(1514, 200 + bwd_bytes / bwd_pkts * 1.5),
        std_ps=_lognorm(rng, 400, 0.3), mean_piat_ms=dur / (fwd_pkts + bwd_pkts),
        std_piat_ms=dur / (fwd_pkts + bwd_pkts) * _lognorm(rng, 1.5, 0.3),
        max_piat_ms=dur * rng.uniform(0.2, 0.7), source="nfstream")


def dns_flow(rng, ts, client):
    return FlowRecord(
        ts_start=ts - 0.03, ts_end=ts, src_ip=client, dst_ip="192.168.1.1",
        src_port=rng.randint(49152, 65535), dst_port=53, proto=17,
        fwd_pkts=1, bwd_pkts=1, fwd_bytes=rng.randint(70, 95), bwd_bytes=rng.randint(90, 300),
        duration_ms=rng.uniform(5, 60), psh=0, min_ps=70, max_ps=300, std_ps=50,
        mean_piat_ms=20, std_piat_ms=0, max_piat_ms=20, source="nfstream")


def ntp_flow(rng, ts, client):
    return FlowRecord(
        ts_start=ts - 0.05, ts_end=ts, src_ip=client, dst_ip="129.6.15.28",
        src_port=123, dst_port=123, proto=17, fwd_pkts=2, bwd_pkts=2,
        fwd_bytes=180, bwd_bytes=180, duration_ms=rng.uniform(20, 2100), psh=0,
        min_ps=90, max_ps=90, std_ps=0, mean_piat_ms=rng.uniform(10, 700),
        std_piat_ms=rng.uniform(0, 900), max_piat_ms=rng.uniform(20, 2000),
        source="nfstream")


def normal_traffic(seed: int, hours: float, start: float = T0) -> list:
    """Web, DNS and NTP traffic from four clients."""
    rng = random.Random(seed)
    out = []
    t = start
    end = start + hours * 3600
    while t < end:
        t += rng.expovariate(1.6)  # about 1.6 events per second
        client = rng.choice(CLIENTS)
        r = rng.random()
        if r < 0.50:
            out.append(web_flow(rng, t, client, 443))
        elif r < 0.62:
            out.append(web_flow(rng, t, client, 80))
        elif r < 0.97:
            out.append(dns_flow(rng, t, client))
        else:
            out.append(ntp_flow(rng, t, client))
    return out


@pytest.fixture(scope="session")
def baseline_flows():
    return normal_traffic(seed=1, hours=6)


@pytest.fixture(scope="session")
def bundle(baseline_flows):
    return fit_bundle(baseline_flows, FitConfig(min_group_flows=400))


@pytest.fixture(scope="session")
def unseen_flows():
    """Normal traffic the baseline never saw (different seed, later time)."""
    return normal_traffic(seed=2, hours=3, start=T0 + 86400)
