"""Tests for the training protocols, windows, reference graphs, metrics and utility features."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

from chronotrace.attack import fit_score as legacy_fit_score
from chronotrace.core import Edge, _bin_index, incident_edges, nodes_of, remove_user, safe_auc, time_bins
from chronotrace.features import _window, build_task, with_window
from chronotrace.metrics_ext import (
    degree_conditioned_expected_auc, degree_conditioned_null_draws, degree_matched_auc,
    pool_size_metrics, positive_band_auc, surprisal_weighted_recall)
from chronotrace.nulls import label_null_aucs, rewire
from chronotrace.protocols import fit_score, release_of, shadow_tasks
from chronotrace.windows import id_cohort_window, activity_bins, oracle_window, shift_window, window_iou


def synthetic_graph(n_nodes=120, n_edges=900, seed=0):
    """Heavy-tailed signed temporal digraph with no duplicate pairs or self-loops."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, n_nodes + 1) ** 0.8
    weights /= weights.sum()
    seen, edges = set(), []
    while len(edges) < n_edges:
        s, d = rng.choice(n_nodes, size=2, replace=False, p=weights)
        if (s, d) in seen:
            continue
        seen.add((s, d))
        edges.append(Edge(int(s), int(d), 1 if rng.random() < 0.85 else -1, int(rng.integers(0, 10_000))))
    edges.sort(key=lambda e: e.time)
    return edges


@pytest.fixture(scope='module')
def graph():
    edges = synthetic_graph()
    return edges, time_bins(edges, 32)


def targets_of(edges, k=4):
    deg = Counter()
    for e in edges:
        deg[e.src] += 1
        deg[e.dst] += 1
    return [v for v, d in sorted(deg.items()) if 6 <= d <= 40][:k]


def test_explicit_universe_matches_default(graph):
    edges, bins = graph
    u = targets_of(edges)[0]
    a = build_task(edges, u, bins, compute_utility=False)
    b = build_task(edges, u, bins, universe=nodes_of(edges), compute_utility=False)
    assert a.nodes == b.nodes
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)


def test_with_window_matches_rebuilt_task(graph):
    edges, bins = graph
    u = targets_of(edges)[1]
    base = build_task(edges, u, bins, keep_hist=True, compute_utility=False)
    np.testing.assert_allclose(with_window(base, *base.window).X, base.X)
    for win in [(0, 3), (5, 9), (28, 31), (10, 10)]:
        rebuilt = build_task(edges, u, bins, window_override=win, compute_utility=False)
        np.testing.assert_allclose(with_window(base, *win).X, rebuilt.X, rtol=1e-6, atol=1e-6)


def test_oracle_window_matches_legacy(graph):
    edges, bins = graph
    for u in targets_of(edges):
        assert oracle_window(incident_edges(edges, u), bins) == _window(incident_edges(edges, u), bins, pad=1)


def test_shift_window_and_iou():
    assert shift_window((4, 8), 3, 32) == (7, 11)
    assert shift_window((4, 8), -10, 32) == (0, 4)
    assert shift_window((25, 30), 5, 32) == (26, 31)  # clamped at the grid end, width kept
    assert window_iou((0, 3), (0, 3)) == 1.0
    assert window_iou((0, 3), (4, 7)) == 0.0
    assert window_iou((0, 3), (2, 5)) == pytest.approx(2 / 6)


def test_id_cohort_window_reads_neighbouring_ids():
    activity = {v: (v // 10, v // 10 + 2) for v in range(100)}
    b0, b1 = id_cohort_window(55, activity, n_bins=32, k=5, pad=0)
    assert (b0, b1) == (5, 7)


def test_release_protocol_never_exposes_test_user(graph):
    edges, bins = graph
    users = targets_of(edges, 4)
    test_user = users[0]
    release = release_of(edges, test_user)
    tasks = shadow_tasks(release, test_user, users, bins, universe=nodes_of(edges))
    assert len(tasks) == len(users) - 1
    for t in tasks:
        assert test_user not in t.nodes
        assert t.user != test_user
    with pytest.raises(ValueError):
        shadow_tasks(edges, test_user, users, bins, universe=nodes_of(edges))


def test_fit_score_matches_legacy_sampling(graph):
    edges, bins = graph
    users = targets_of(edges, 4)
    tasks = [build_task(edges, u, bins, compute_utility=False) for u in users]
    cols = list(range(11))
    new, coef, _ = fit_score(tasks[1:], tasks[0], cols, seed=7)
    old = legacy_fit_score(tasks[1:], tasks[0], cols, seed=7)
    np.testing.assert_allclose(new, old)
    assert coef is not None and coef.shape == (11,)


def test_degree_controls_on_pure_degree_scorer():
    rng = np.random.default_rng(1)
    deg = rng.integers(0, 30, size=2000).astype(float)
    y = (rng.random(2000) < 0.01 + 0.02 * (deg > 15)).astype(int)
    actual = safe_auc(y, np.log1p(deg))
    assert degree_conditioned_expected_auc(y, np.log1p(deg), deg) == pytest.approx(actual)
    assert degree_matched_auc(y, np.log1p(deg), deg) == pytest.approx(0.5)
    noise_scores = rng.random(2000)
    draws = degree_conditioned_null_draws(y, noise_scores, deg, 4000, rng)
    assert draws.mean() == pytest.approx(degree_conditioned_expected_auc(y, noise_scores, deg), abs=5e-3)


def test_rare_neighbour_metrics():
    deg = np.array([1, 1, 50, 50, 2, 3, 40, 2], dtype=float)
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    scores = deg.copy()
    bands = positive_band_auc(y, scores, deg, bands=((0, 3), (4, np.inf)))
    assert bands['n_pos_0-3'] == 1 and bands['n_pos_4-inf'] == 1
    assert bands['4-inf'] > bands['0-3']
    assert 0.0 <= surprisal_weighted_recall(y, scores, deg, k=2) < 1.0


def test_pool_size_metrics_keep_all_positives():
    rng = np.random.default_rng(3)
    y = np.zeros(5000, dtype=int)
    y[:10] = 1
    scores = rng.random(5000) + y
    full = pool_size_metrics(y, scores, pool=10_000, rng=rng)
    small = pool_size_metrics(y, scores, pool=500, rng=rng, n_draws=5)
    assert full['auc'] == pytest.approx(small['auc'], abs=0.05)
    assert small['precision@10'] >= full['precision@10']


def test_label_nulls():
    rng = np.random.default_rng(5)
    deg = np.concatenate([np.ones(900), np.full(100, 50.0)])
    res = label_null_aucs(deg, n_pos=5, rng=rng, n_trials=400)
    assert res['uniform_mean'] == pytest.approx(0.5, abs=0.05)
    assert res['degprop_mean'] > 0.75


@pytest.mark.parametrize('mode', ['config', 'signed_temporal'])
def test_rewire_preserves_marginals(graph, mode):
    edges, bins = graph
    new, info = rewire(edges, bins, mode, seed=11)
    assert info['dropped_self_loops'] <= 0.01 * len(edges)
    if info['dropped_self_loops'] == 0:
        key = (lambda e: ()) if mode == 'config' else (lambda e: (_bin_index(bins, e.time), e.rating > 0))
        out_old = Counter((e.src, key(e)) for e in edges)
        out_new = Counter((e.src, key(e)) for e in new)
        in_old = Counter((e.dst, key(e)) for e in edges)
        in_new = Counter((e.dst, key(e)) for e in new)
        assert out_old == out_new and in_old == in_new
    assert all(e.src != e.dst for e in new)
    assert sorted(e.time for e in new) == sorted(e.time for e in edges) or info['dropped_self_loops'] > 0


def test_edgelist_universe_drops_only_isolated_neighbours(graph):
    from chronotrace.baselines import heuristic_scores
    edges, bins = graph
    for u in targets_of(edges, 4):
        directory = build_task(edges, u, bins, compute_utility=False)
        edgelist = build_task(edges, u, bins, compute_utility=False, candidates_from_release=True)
        zero = directory.released_degree == 0
        assert np.all(directory.y[zero] == 1), 'a candidate left without edges must be a former neighbour'
        assert np.all(edgelist.released_degree > 0)
        assert set(edgelist.nodes) == set(np.asarray(directory.nodes)[~zero].tolist())
        heur = heuristic_scores(edges, u, bins, return_scores=True, candidates_from_release=True)
        assert heur['candidates'] == edgelist.nodes


def test_utility_training_features_hold_out_the_edge(graph):
    """A training edge must be featurized as if unseen, like a test edge: its own sign,
    its own degree contribution and its own adjacency must not enter its features."""
    from chronotrace.core import build_adj, signed_stats
    from chronotrace.utility import _edge_feature, training_features

    edges, _ = graph
    train = edges[:600]
    X = training_features(train)
    for i in (0, 17, 301, 599):
        e = train[i]
        rest = train[:i] + train[i + 1:]
        expected = _edge_feature(e, build_adj(rest), signed_stats(rest))
        assert np.allclose(X[i], expected), i


def test_utility_test_features_unchanged(graph):
    from chronotrace.core import build_adj, signed_stats
    from chronotrace.utility import _edge_feature, test_features

    edges, _ = graph
    train, test = edges[:600], edges[600:650]
    X = test_features(test, train)
    adj, stats = build_adj(train), signed_stats(train)
    assert np.allclose(X, [_edge_feature(e, adj, stats) for e in test])
