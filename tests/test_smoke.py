#!/usr/bin/env python3
"""Smoke tests for ChronoTrace package."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from chronotrace.core import DATASETS  # noqa: E402

requires_otc = pytest.mark.skipif(not DATASETS['bitcoinotc'].exists(),
                                  reason='BitcoinOTC data not found; see data/README.md')


def test_data_loading():
    from chronotrace.core import DATASETS, load_edges, nodes_of
    for name, path in DATASETS.items():
        if not path.exists():
            print(f'  SKIP {name} (data not found at {path})')
            continue
        edges = load_edges(path)
        nodes = nodes_of(edges)
        assert len(edges) > 0, f'{name}: no edges loaded'
        assert len(nodes) > 1, f'{name}: too few nodes'
        assert all(e.rating in (-1, 1) for e in edges), f'{name}: ratings not binarized'
        assert edges == sorted(edges, key=lambda e: e.time), f'{name}: edges not time-sorted'
        print(f'  OK {name}: {len(edges)} edges, {len(nodes)} nodes')


@requires_otc
def test_user_selection():
    from chronotrace.core import DATASETS, load_edges, choose_users
    edges = load_edges(DATASETS['bitcoinotc'])
    users = choose_users(edges, n_users=8, seed=42, min_degree=4, max_degree=128)
    assert len(users) == 8, f'Expected 8 users, got {len(users)}'
    assert len(set(users)) == len(users), 'Duplicate users selected'
    # deterministic
    users2 = choose_users(edges, n_users=8, seed=42, min_degree=4, max_degree=128)
    assert users == users2, 'User selection not deterministic'
    print(f'  OK user selection: {users}')


def test_metrics():
    from chronotrace.core import (
        safe_auc, safe_ap, recall_at_k, precision_at_k,
        success_at_k, best_positive_rank, mean_positive_rank,
    )
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    score = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.15, 0.7, 0.25, 0.4])

    auc = safe_auc(y, score)
    assert 0.5 < auc <= 1.0, f'AUC out of range: {auc}'

    ap = safe_ap(y, score)
    assert 0 < ap <= 1.0, f'AP out of range: {ap}'

    r5 = recall_at_k(y, score, 5)
    assert 0 <= r5 <= 1.0, f'Recall@5 out of range: {r5}'

    p5 = precision_at_k(y, score, 5)
    assert 0 <= p5 <= 1.0, f'Precision@5 out of range: {p5}'

    s5 = success_at_k(y, score, 5)
    assert s5 in (0.0, 1.0), f'Success@5 should be binary: {s5}'

    bpr = best_positive_rank(y, score)
    assert bpr >= 1, f'Best pos rank should be >= 1: {bpr}'

    mpr = mean_positive_rank(y, score)
    assert mpr >= 1, f'Mean pos rank should be >= 1: {mpr}'

    print(f'  OK metrics: AUC={auc:.3f} AP={ap:.3f} R@5={r5:.2f} P@5={p5:.2f} BPR={bpr:.0f}')


@requires_otc
def test_task_building():
    from chronotrace.core import DATASETS, load_edges, time_bins
    from chronotrace.features import build_task, FEATURE_NAMES
    edges = load_edges(DATASETS['bitcoinotc'])
    bins = time_bins(edges, 32)
    task = build_task(edges, user=6, bins=bins)
    assert task.X.shape[1] == len(FEATURE_NAMES), f'Feature dim mismatch: {task.X.shape[1]} vs {len(FEATURE_NAMES)}'
    assert task.y.sum() > 0, 'No positive labels in task'
    assert len(task.nodes) == task.X.shape[0], 'Node count mismatch'
    assert not np.any(np.isnan(task.X)), 'NaN in features'
    print(f'  OK task building: {task.X.shape}, {task.y.sum()} positives / {len(task.y)} total')


@requires_otc
def test_lopo_evaluation():
    from chronotrace.core import DATASETS, load_edges, time_bins, choose_users
    from chronotrace.features import build_task
    from chronotrace.attack import evaluate_lopo
    edges = load_edges(DATASETS['bitcoinotc'])
    bins = time_bins(edges, 32)
    users = choose_users(edges, n_users=3, seed=42, min_degree=4, max_degree=128)
    tasks = [build_task(edges, u, bins) for u in users]
    rows = evaluate_lopo(tasks, seed=42)
    assert len(rows) == 3, f'Expected 3 rows, got {len(rows)}'
    for row in rows:
        assert 'chronotrace' in row['methods']
        auc = row['methods']['chronotrace']['auc']
        assert not np.isnan(auc), 'NaN AUC in LOPO evaluation'
        assert 0 <= auc <= 1, f'AUC out of range: {auc}'
    auc_vals = [f'{r["methods"]["chronotrace"]["auc"]:.3f}' for r in rows]
    print(f'  OK LOPO: {len(rows)} rows, AUCs: {auc_vals}')


@requires_otc
def test_heuristic_baselines():
    from chronotrace.core import DATASETS, load_edges, time_bins
    from chronotrace.baselines import heuristic_scores
    edges = load_edges(DATASETS['bitcoinotc'])
    bins = time_bins(edges, 32)
    result = heuristic_scores(edges, user=6, bins=bins)
    assert 'degree_prior' in result['methods']
    assert 'common_neighbors' in result['methods']
    auc = result['methods']['degree_prior']['auc']
    assert not np.isnan(auc), 'NaN AUC for degree prior'
    print(f'  OK heuristic baselines: degree_prior AUC={auc:.3f}')


if __name__ == '__main__':
    tests = [
        ('Data loading', test_data_loading),
        ('User selection', test_user_selection),
        ('Metrics', test_metrics),
        ('Task building', test_task_building),
        ('LOPO evaluation', test_lopo_evaluation),
        ('Heuristic baselines', test_heuristic_baselines),
    ]
    passed = 0
    failed = 0
    for name, func in tests:
        print(f'\n[TEST] {name}')
        try:
            func()
            passed += 1
        except Exception as e:
            print(f'  FAIL: {e}')
            failed += 1

    print(f'\n{"="*40}')
    print(f'Results: {passed} passed, {failed} failed')
    sys.exit(1 if failed > 0 else 0)
