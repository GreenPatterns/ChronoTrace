#!/usr/bin/env python3
"""Experiments of the ChronoTrace benchmark: main results, reference graphs and low-degree targets.

Stage ``tasks`` builds and caches every evaluation task: one target deleted from
one graph, every other node featurized (full candidate universe, no
subsampling), per-candidate activity histograms kept, plus the oracle-anchor
heuristic baselines on the real graph.

Stage ``attack`` trains the logistic attacker for every target under both
training protocols (``lodo`` as submitted, ``release`` added in the revision)
and every window mode, scores the held-out target, and saves per-candidate
scores, metrics, coefficients, windows and label-null AUCs.

Graphs: ``real`` or a rewired reference graph ``config<r>`` / ``sigtemp<r>``
(realization r; see chronotrace.nulls). Target sets: ``main`` (incident degree
4-128, 12 per seed, as submitted) and ``low`` (incident degree 1-3; trained on the
same seed's main targets).

Outputs: paper/results/revision/<dataset>/<graph>/<targetset>/{tasks,units}/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chronotrace.baselines import heuristic_scores
from chronotrace.core import (DATASETS, DATA_ROOT, choose_users, incident_edges, load_edges, metrics,
                              nodes_of, remove_user, time_bins)
from chronotrace.features import FEATURE_GROUPS, FEATURE_NAMES, Task, build_task, with_window
from chronotrace.metrics_ext import (degree_conditioned_expected_auc, degree_matched_auc,
                                     positive_band_auc, surprisal_weighted_recall)
from chronotrace.nulls import label_null_aucs, rewire
from chronotrace.protocols import fit_score, release_of, shadow_tasks
from chronotrace.windows import window_for, window_iou

DATASETS.setdefault('epinions_dated', DATA_ROOT / 'edg60000.csv')
OUT_ROOT = Path(__file__).resolve().parents[2] / 'paper' / 'results' / 'revision'
SEEDS = [42, 7, 13, 100, 200]
N_TARGETS = 12
TARGET_RANGES = {'main': (4, 128), 'low': (1, 3)}
N_BINS = 32
WINDOW_GROUPS = ['temporal', 'motif', 'chronotrace']

_GRAPHS: Dict[Tuple[str, str], Tuple[list, np.ndarray, list]] = {}


# ---------------------------------------------------------------- graphs and targets

def get_graph(dataset: str, graph: str):
    """(edges, bins, node universe) for a dataset's real or rewired graph, cached per process.

    Bins always come from the real graph so every variant shares one time grid.
    """
    key = (dataset, graph)
    if key not in _GRAPHS:
        real = load_edges(DATASETS[dataset])
        bins = time_bins(real, N_BINS)
        if graph == 'real':
            edges = real
        else:
            mode = {'config': 'config', 'sigtemp': 'signed_temporal'}[graph.rstrip('0123456789')]
            realization = int(graph[len(graph.rstrip('0123456789')):])
            edges, info = rewire(real, bins, mode, seed=1000 + realization)
            info_path = OUT_ROOT / dataset / graph / 'rewire_info.json'
            info_path.parent.mkdir(parents=True, exist_ok=True)
            info_path.write_text(json.dumps(info, indent=2))
        _GRAPHS[key] = (edges, bins, nodes_of(real))
    return _GRAPHS[key]


def targets(dataset: str, targetset: str, seed: int) -> List[int]:
    """Main targets: 12 per seed stratified over degree quartiles (as submitted).

    Low-degree targets (1-3 incident events) are drawn uniformly, 12 per seed: with only
    three degree values the quartile sampler leaves buckets empty and returns fewer users.
    """
    real, _, _ = get_graph(dataset, 'real')
    lo, hi = TARGET_RANGES[targetset]
    if targetset == 'low':
        deg: Dict[int, int] = {}
        for e in real:
            deg[e.src] = deg.get(e.src, 0) + 1
            deg[e.dst] = deg.get(e.dst, 0) + 1
        pool = sorted(v for v, d in deg.items() if lo <= d <= hi)
        rng = np.random.default_rng([seed, 7919])
        return sorted(int(v) for v in rng.choice(pool, size=N_TARGETS, replace=False))
    return choose_users(real, N_TARGETS, seed, lo, hi)


UNIVERSE = 'edgelist'  # set from --universe; 'directory' keeps every account as a candidate


def ts_dir(targetset: str) -> str:
    return targetset if UNIVERSE == 'directory' else f'{targetset}-{UNIVERSE}'


def unit_dir(dataset: str, graph: str, targetset: str, kind: str) -> Path:
    d = OUT_ROOT / dataset / graph / ts_dir(targetset) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- task cache

def save_task(path: Path, task: Task, heur: Dict | None) -> None:
    arrays = dict(user=task.user, nodes=np.asarray(task.nodes, dtype=np.int64), X=task.X, y=task.y.astype(np.int8),
                  hist=task.hist, signed_hist=task.signed_hist, released_degree=task.released_degree,
                  window=np.asarray(task.window), profile=json.dumps(task.profile))
    if heur is not None:
        for name, sc in heur['scores'].items():
            arrays[f'H__{name}'] = sc
    np.savez_compressed(path, **arrays)


def load_task(path: Path) -> Task:
    z = np.load(path, allow_pickle=False)
    return Task(user=int(z['user']), nodes=z['nodes'].tolist(), X=z['X'], y=z['y'].astype(int),
                profile=json.loads(str(z['profile'])), utility_edges_before=0, utility_edges_after=0,
                utility_sign_auc=float('nan'), utility_sign_ap=float('nan'), utility_train_edges=0,
                utility_test_edges=0, feature_names=list(FEATURE_NAMES), window=tuple(z['window'].tolist()),
                hist=z['hist'], signed_hist=z['signed_hist'], released_degree=z['released_degree'])


def task_path(dataset, graph, targetset, seed, user) -> Path:
    return unit_dir(dataset, graph, targetset, 'tasks') / f'seed{seed}_u{user}.npz'


def build_one(dataset: str, graph: str, targetset: str, seed: int, user: int, universe_mode: str) -> str:
    global UNIVERSE
    UNIVERSE = universe_mode
    path = task_path(dataset, graph, targetset, seed, user)
    if path.exists():
        return f'skip {path.name}'
    edges, bins, universe = get_graph(dataset, graph)
    from_release = UNIVERSE == 'edgelist'
    task = build_task(edges, user, bins, universe=universe, compute_utility=False, keep_hist=True,
                      candidates_from_release=from_release)
    heur = None
    if graph == 'real':
        heur = heuristic_scores(edges, user, bins, return_scores=True, candidates_from_release=from_release)
        if heur['candidates'] != task.nodes:
            raise RuntimeError('heuristic candidates differ from task candidates')
    save_task(path, task, heur)
    return f'built {path.name} n={len(task.nodes)} pos={int(task.y.sum())}'


# ---------------------------------------------------------------- attack units

def score_metrics(y: np.ndarray, s: np.ndarray, deg: np.ndarray) -> Dict:
    m = {k: float(v) for k, v in metrics(y, s).items()}
    m['deg_matched_auc'] = degree_matched_auc(y, s, deg)
    m['deg_cond_auc'] = degree_conditioned_expected_auc(y, s, deg)
    m.update({f'band_{k}': v for k, v in positive_band_auc(y, s, deg).items()})
    m['swr@100'] = surprisal_weighted_recall(y, s, deg, 100)
    m['swr@500'] = surprisal_weighted_recall(y, s, deg, 500)
    return m


def rng_for(seed: int, user: int, mode: str) -> np.random.Generator:
    return np.random.default_rng([seed, user, sum(ord(c) for c in mode)])


def windowed(task: Task, user: int, full_edges_u, release_edges, bins, mode: str, seed: int) -> Task:
    if mode == 'oracle':
        return task
    w = window_for(mode, user, full_edges_u, release_edges, bins, rng_for(seed, user, mode))
    return with_window(task, *w)


def attack_one(dataset: str, graph: str, targetset: str, seed: int, user: int,
               protocols: List[str], modes: List[str], universe_mode: str) -> str:
    global UNIVERSE
    UNIVERSE = universe_mode
    out_npz = unit_dir(dataset, graph, targetset, 'units') / f'seed{seed}_u{user}.npz'
    out_json = out_npz.with_suffix('.json')
    if out_json.exists():
        return f'skip {out_json.name}'
    t0 = time.time()
    edges, bins, universe = get_graph(dataset, graph)
    main_users = targets(dataset, 'main', seed)
    train_users = [u for u in main_users if u != user]
    test = load_task(task_path(dataset, graph, targetset, seed, user))
    deg = test.released_degree.astype(float)
    y = test.y
    full_neighbors = {(e.dst if e.src == user else e.src) for e in incident_edges(edges, user)}
    record = {'dataset': dataset, 'graph': graph, 'targetset': targetset, 'universe': UNIVERSE,
              'seed': seed, 'user': user, 'n_candidates': int(len(y)), 'n_pos': int(y.sum()),
              'n_neighbors': len(full_neighbors), 'n_isolated_neighbors': len(full_neighbors) - int(y.sum()),
              'incident_degree': int(test.profile['degree']),
              'true_window': list(test.window), 'train_users': train_users, 'results': {}, 'windows': {},
              'feature_names': list(FEATURE_NAMES)}
    scores_out = {'nodes': np.asarray(test.nodes, dtype=np.int64), 'y': y.astype(np.int8),
                  'deg': test.released_degree}
    full_u = incident_edges(edges, user)
    release = release_of(edges, user)

    train_sets: Dict[str, List[Task]] = {}
    if 'lodo' in protocols:
        train_sets['lodo'] = [load_task(task_path(dataset, graph, 'main', seed, u)) for u in train_users]
    if 'release' in protocols:
        train_sets['release'] = shadow_tasks(release, user, train_users, bins, universe,
                                             candidates_from_release=UNIVERSE == 'edgelist')
    shadow_releases = {}

    for protocol, train in train_sets.items():
        for mode in (modes if protocol == 'release' else ['oracle']):
            test_m = windowed(test, user, full_u, release, bins, mode, seed)
            if mode != 'oracle':
                train_m = []
                for t in train:
                    if protocol == 'release':
                        if t.user not in shadow_releases:
                            shadow_releases[t.user] = remove_user(release, t.user)
                        rel_j, full_j = shadow_releases[t.user], incident_edges(release, t.user)
                    else:
                        rel_j, full_j = remove_user(edges, t.user), incident_edges(edges, t.user)
                    train_m.append(windowed(t, t.user, full_j, rel_j, bins, mode, seed))
            else:
                train_m = train
            if protocol == 'release':
                record['windows'][mode] = {'test': list(test_m.window),
                                           'iou': window_iou(tuple(test_m.window), tuple(test.window))}
            groups = FEATURE_GROUPS if mode == 'oracle' else {g: FEATURE_GROUPS[g] for g in WINDOW_GROUPS}
            for gname, cols in groups.items():
                s, coef, intercept = fit_score(train_m, test_m, cols, seed)
                key = f'{protocol}__{mode}__{gname}'
                rec = score_metrics(y, s, deg)
                rec['coef'] = None if coef is None else [float(c) for c in coef]
                rec['intercept'] = intercept
                record['results'][key] = rec
                scores_out[f'S__{key}'] = s.astype(np.float32)

    rng = np.random.default_rng([seed, user, 7])
    record['label_null'] = label_null_aucs(deg, int(y.sum()), rng, n_trials=200)
    random_scores = np.random.default_rng(seed + user).random(len(y))
    record['results']['random'] = score_metrics(y, random_scores, deg)
    if graph == 'real':
        z = np.load(task_path(dataset, graph, targetset, seed, user))
        for k in z.files:
            if k.startswith('H__'):
                record['results'][f'heur__{k[3:]}'] = score_metrics(y, z[k].astype(float), deg)
    record['elapsed_sec'] = time.time() - t0
    np.savez_compressed(out_npz, **scores_out)
    out_json.write_text(json.dumps(record, indent=1))
    return f'done {dataset}/{graph}/{targetset} seed{seed} u{user} ({record["elapsed_sec"]:.0f}s)'


# ---------------------------------------------------------------- orchestration

def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return 'ERROR ' + repr(args[:5]) + '\n' + traceback.format_exc()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', choices=['tasks', 'attack'], required=True)
    ap.add_argument('--datasets', default='bitcoinotc,bitcoinalpha,epinions_dated')
    ap.add_argument('--graphs', default='real')
    ap.add_argument('--targetsets', default='main')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--protocols', default='lodo,release')
    ap.add_argument('--modes', default='oracle,shift1,shift2,shift4,shift8,hub,cohort')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--universe', choices=['edgelist', 'directory'], default='edgelist',
                    help='candidates: users still in the edge list (default) or every account')
    args = ap.parse_args()
    global UNIVERSE
    UNIVERSE = args.universe

    units = []
    for ds in args.datasets.split(','):
        for graph in args.graphs.split(','):
            for ts in args.targetsets.split(','):
                for seed in map(int, args.seeds.split(',')):
                    users = targets(ds, ts, seed)
                    if args.stage == 'tasks' and ts == 'low':
                        users = users  # low targets need their own tasks; main tasks are built separately
                    for u in users:
                        units.append((ds, graph, ts, seed, u))
    print(f'[{time.strftime("%H:%M:%S")}] stage={args.stage} universe={args.universe} units={len(units)} '
          f'workers={args.workers}', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        if args.stage == 'tasks':
            futs = [ex.submit(_safe, build_one, *u, args.universe) for u in units]
        else:
            futs = [ex.submit(_safe, attack_one, *u, args.protocols.split(','), args.modes.split(','),
                              args.universe) for u in units]
        for i, f in enumerate(as_completed(futs), 1):
            msg = f.result()
            print(f'[{time.strftime("%H:%M:%S")}] {i}/{len(units)} {msg}', flush=True)
    print(f'[{time.strftime("%H:%M:%S")}] finished in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
