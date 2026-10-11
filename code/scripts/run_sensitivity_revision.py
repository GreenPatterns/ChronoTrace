#!/usr/bin/env python3
"""Appendix sensitivity analyses (release-only protocol, edge-list universe).

* ``classifier``: random forest (100 trees, depth 10, balanced) against the logistic
  attacker, for the degree, all-release-only and all-window feature sets.
* ``binarization``: the window-informed attacker when ratings above a threshold
  tau in {-1, 0, 1, 5} count as positive (Bitcoin graphs, whose ratings span -10..10).

Outputs: paper/results/revision/<dataset>/sensitivity-<analysis>/seed<S>_u<U>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_revision
from chronotrace.core import DATASETS, load_edges, metrics, nodes_of, time_bins
from chronotrace.features import FEATURE_GROUPS, build_task
from chronotrace.protocols import fit_score, release_of, shadow_tasks
from run_revision import OUT_ROOT, SEEDS, get_graph, load_task, targets, task_path

run_revision.UNIVERSE = 'edgelist'
GROUPS = {'degree': FEATURE_GROUPS['degree'], 'release_all': FEATURE_GROUPS['full_nonoracle'],
          'window_all': FEATURE_GROUPS['chronotrace']}
THRESHOLDS = [-1, 0, 1, 5]


def rf_scores(train, test, cols, seed):
    Xtr = np.vstack([t.X[:, cols] for t in train])
    ytr = np.concatenate([t.y for t in train])
    pos, neg = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    rng = np.random.default_rng(seed + test.user)
    keep = np.concatenate([pos, rng.choice(neg, size=min(len(neg), max(1000, 20 * len(pos))), replace=False)])
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=10,
                                                                 class_weight='balanced', random_state=seed,
                                                                 n_jobs=1))
    clf.fit(Xtr[keep], ytr[keep])
    return clf.predict_proba(test.X[:, cols])[:, 1]


def classifier_unit(dataset, seed, user):
    edges, bins, universe = get_graph(dataset, 'real')
    train_users = [u for u in targets(dataset, 'main', seed) if u != user]
    test = load_task(task_path(dataset, 'real', 'main', seed, user))
    shadows = shadow_tasks(release_of(edges, user), user, train_users, bins, universe, candidates_from_release=True)
    out = {}
    for g, cols in GROUPS.items():
        lr, _, _ = fit_score(shadows, test, cols, seed)
        out[f'lr_{g}'] = float(metrics(test.y, lr)['auc'])
        out[f'rf_{g}'] = float(metrics(test.y, rf_scores(shadows, test, cols, seed))['auc'])
    return out


_BIN_GRAPHS = {}


def binarization_unit(dataset, seed, user):
    out = {}
    for tau in THRESHOLDS:
        key = (dataset, tau)
        if key not in _BIN_GRAPHS:
            e = load_edges(DATASETS[dataset], binarization_threshold=tau)
            _BIN_GRAPHS[key] = (e, time_bins(e, run_revision.N_BINS), nodes_of(e))
        edges, bins, universe = _BIN_GRAPHS[key]
        train_users = [u for u in targets(dataset, 'main', seed) if u != user]
        test = build_task(edges, user, bins, universe=universe, compute_utility=False, candidates_from_release=True)
        shadows = shadow_tasks(release_of(edges, user), user, train_users, bins, universe,
                               candidates_from_release=True)
        s, _, _ = fit_score(shadows, test, GROUPS['window_all'], seed)
        out[str(tau)] = {'auc': float(metrics(test.y, s)['auc']),
                         'distrust_share': float(np.mean([e.rating < 0 for e in edges]))}
    return out


def run_unit(analysis, dataset, seed, user):
    path = OUT_ROOT / dataset / f'sensitivity-{analysis}' / f'seed{seed}_u{user}.json'
    if path.exists():
        return f'skip {path.name}'
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = classifier_unit(dataset, seed, user) if analysis == 'classifier' else binarization_unit(dataset, seed, user)
    path.write_text(json.dumps({'dataset': dataset, 'seed': seed, 'user': user, 'results': res,
                                'elapsed_sec': time.time() - t0}, indent=1))
    return f'done {analysis} {dataset} seed{seed} u{user} ({time.time() - t0:.0f}s)'


def _safe(*args):
    try:
        return run_unit(*args)
    except Exception:
        return 'ERROR ' + repr(args) + '\n' + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', choices=['classifier', 'binarization'], required=True)
    ap.add_argument('--datasets', default='bitcoinotc,bitcoinalpha,epinions_dated')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    units = [(args.analysis, ds, s, u) for ds in args.datasets.split(',') for s in SEEDS
             for u in targets(ds, 'main', s)]
    print(f'[{time.strftime("%H:%M:%S")}] {args.analysis} units={len(units)}', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_safe, *u) for u in units]
        for i, f in enumerate(as_completed(futs), 1):
            print(f'[{time.strftime("%H:%M:%S")}] {i}/{len(units)} {f.result()}', flush=True)


if __name__ == '__main__':
    main()
