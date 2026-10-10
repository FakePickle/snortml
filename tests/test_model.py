import numpy as np
import pytest

from snortml_inference.baseline import load_bundle, save_bundle
from snortml_inference.features import flow_parent
from snortml_inference.model import DetectorParams, GroupedDetector, fit_gpd, gpd_sf
from snortml_inference.scorer import Scorer


def _correlated(rng, n, d=5, rho=0.9):
    """Lognormal-ish features driven by one shared factor."""
    shared = rng.normal(size=(n, 1))
    X = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(size=(n, d))
    return 3.0 + 1.5 * X


def _split(n):
    return np.array([0, 0, 0, 1, 2])[np.arange(n) % 5]


@pytest.fixture(scope="module")
def det():
    rng = np.random.default_rng(3)
    n = 6000
    keys = np.where(np.arange(n) % 3 == 0, "udp/53", "tcp/443").astype(object)
    X = _correlated(rng, n)
    cat = np.where(rng.random(n) < 0.03, "rare", "common").astype(object)
    d = GroupedDetector([f"f{i}" for i in range(5)], ["state"], flow_parent,
                        DetectorParams(autoencoder=True, seed=1))
    d.fit(keys, X, [cat], _split(n))
    return d


def test_gpd_fit_recovers_exponential_tail():
    rng = np.random.default_rng(0)
    xi, sigma = fit_gpd(rng.exponential(2.0, 4000))
    assert abs(xi) < 0.1 and abs(sigma - 2.0) < 0.25
    assert abs(gpd_sf(np.array([2.0]), xi, sigma)[0] - np.exp(-1)) < 0.05


def test_gpd_fit_sees_heavy_tail_and_never_bounds_it():
    rng = np.random.default_rng(1)
    heavy = (rng.pareto(1 / 0.3, 4000)) * 3.0  # shape 0.3
    xi, _ = fit_gpd(heavy)
    assert 0.15 < xi <= 0.5
    light = rng.uniform(0, 1, 4000)  # truly bounded: fitted shape would be negative
    xi, sigma = fit_gpd(light)
    assert xi == 0.0 and gpd_sf(np.array([50.0]), xi, sigma)[0] > 0


def test_p_values_are_calibrated_on_fresh_normal_data(det):
    rng = np.random.default_rng(11)
    n = 20000
    keys = np.where(np.arange(n) % 3 == 0, "udp/53", "tcp/443").astype(object)
    cat = np.where(rng.random(n) < 0.03, "rare", "common").astype(object)
    r = det.score(keys, _correlated(rng, n), [cat])
    assert (r["group"] >= 0).all()
    assert 0.07 < np.mean(r["p"] <= 0.1) < 0.13
    assert 0.004 < np.mean(r["p"] <= 0.01) < 0.02
    assert np.mean(r["p"] <= 0.0001) < 0.002


def test_extreme_value_gets_small_p_and_is_blamed(det):
    rng = np.random.default_rng(5)
    X = _correlated(rng, 1)
    X[0] = 3.0
    X[0, 2] = 3.0 + 1.5 * 9  # nine standard deviations out, on its own
    r = det.score(np.array(["tcp/443"], dtype=object), X, [np.array(["common"], dtype=object)])
    assert r["p"][0] < 1e-3
    assert int(np.argmax(r["contrib"][0])) == 2


def test_broken_correlation_is_flagged_but_consistent_point_is_not(det):
    key = np.array(["tcp/443"], dtype=object)
    cat = [np.array(["common"], dtype=object)]
    consistent = np.full((1, 5), 3.0 + 1.5 * 1.8)  # all features high together
    broken = np.array([[3.0 + 1.5 * 1.8, 3.0 - 1.5 * 1.8] * 2 + [3.0]])  # alternating
    p_ok = det.score(key, consistent, cat)["p"][0]
    p_bad = det.score(key, broken, cat)["p"][0]
    assert p_ok > 0.02 and p_bad < p_ok / 20


def test_rare_category_alone_is_not_an_alert_but_unseen_is_more_surprising(det):
    key = np.array(["tcp/443"] * 3, dtype=object)
    X = np.full((3, 5), 3.0)
    cats = [np.array(["common", "rare", "never-seen"], dtype=object)]
    p = det.score(key, X, cats)["p"]
    assert p[1] > 1e-3  # a 3% category is unusual, not alarming
    assert p[2] < p[1] < p[0]


def test_evidence_floor(det):
    key = np.array(["tcp/443"], dtype=object)
    X = np.full((1, 5), 500.0)  # absurd
    r = det.score(key, X, [np.array(["common"], dtype=object)])
    gi = int(r["group"][0])
    assert r["p"][0] == pytest.approx(1.0 / (100 * det.cal_n[gi]))
    assert r["depth"][0] > -np.log10(r["p"][0])  # severity still sees how far out it is


def test_small_groups_fall_back_and_unknown_contexts_resolve():
    rng = np.random.default_rng(2)
    n = 3350
    keys = np.array(["tcp/443"] * 3000 + ["tcp/8443"] * 200 + ["udp/9999"] * 150, dtype=object)
    keys = keys[np.random.default_rng(9).permutation(n)]
    d = GroupedDetector(["a", "b"], [], flow_parent, DetectorParams(autoencoder=False))
    d.fit(keys, rng.normal(size=(n, 2)), [], _split(n))
    # tcp/8443 and udp/9999 are too small alone; together they make the "other" pool
    assert d.groups == ["other", "tcp/443"]
    r = d.score(np.array(["tcp/8443", "icmp", "tcp/443"], dtype=object),
                rng.normal(size=(3, 2)), [])
    assert list(r["group"]) == [0, 0, 1]


def test_no_model_means_no_alert():
    rng = np.random.default_rng(2)
    n = 2000
    d = GroupedDetector(["a"], [], flow_parent, DetectorParams(autoencoder=False))
    d.fit(np.array(["tcp/443"] * n, dtype=object), rng.normal(size=(n, 1)), [], _split(n))
    r = d.score(np.array(["udp/53"], dtype=object), np.array([[99.0]]), [])
    assert r["group"][0] == -1 and r["p"][0] == 1.0


def test_all_missing_column_is_dropped_and_missing_values_are_neutral():
    rng = np.random.default_rng(4)
    n = 2000
    X = rng.normal(size=(n, 3))
    X[:, 1] = np.nan  # e.g. packet statistics a Zeek log does not have
    d = GroupedDetector(["a", "b", "c"], [], flow_parent, DetectorParams(autoencoder=False))
    d.fit(np.array(["tcp/443"] * n, dtype=object), X, [], _split(n))
    assert d.feature_names == ["a", "c"]
    key = np.array(["tcp/443"] * 2, dtype=object)
    full = d.score(key, np.array([[0.0, 0.0, 0.0], [np.nan, 5.0, 0.0]]), [])
    assert full["p"][1] >= full["p"][0] * 0.5  # a missing value is not an anomaly


def test_bundle_roundtrip_gives_identical_scores(bundle, unseen_flows, tmp_path):
    save_bundle(bundle, tmp_path / "b")
    again = load_bundle(tmp_path / "b")
    a, b = Scorer(bundle), Scorer(again)
    recs = unseen_flows[:3000]
    np.testing.assert_allclose(a.score_flows(a.table(recs)).p, b.score_flows(b.table(recs)).p)
    assert a.alpha_flow == b.alpha_flow and a.alpha_host == b.alpha_host
    # nothing in the bundle needs pickle to load
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == ["arrays.npz", "manifest.json"]
