#!/usr/bin/env python3
"""Defense stress tests under the release-only protocol.

For each main target and defense the platform publishes a defended release; the
attacker trains on deletions it simulates on that release with the same (public)
defense applied, then scores the target. Defenses:

* ChronoScrub (platform-side local edge suppression; diagnostic, no guarantee):
  deletion_only, temporal, joint at strength 0.25/0.50/0.75.
* Deletion-only edge-level randomized response at epsilon in {0.5, 1, 2, 4}:
  every released edge is kept with probability e^eps / (1 + e^eps).

Outputs: paper/results/revision/<dataset>/defense/seed<S>_u<U>.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronotrace.core import incident_edges, metrics, remove_user
from chronotrace.features import FEATURE_GROUPS, _window, apply_chronoscrub_edges, apply_randomized_response, build_task
from chronotrace.protocols import fit_score, shadow_tasks
from chronotrace.utility import sign_prediction_utility
from run_revision import OUT_ROOT, SEEDS, get_graph, targets

CHRONOSCRUB = [
    ('deletion_only', 'none', 0.0),
    ('temporal_scrub', 'temporal_smooth', 1.0),
    ('joint_0.25', 'joint', 0.25),
    ('joint_0.50', 'joint', 0.50),
    ('joint_0.75', 'joint', 0.75),
]
EPSILONS = [0.5, 1.0, 2.0, 4.0]
GROUPS = ['degree', 'full_nonoracle', 'chronotrace']


EDGELIST = True  # candidates are users still present in the release; set from --universe


def attack(release, test_task, user, train_users, bins, universe, seed, mode='none', strength=0.0):
    shadows = shadow_tasks(release, user, train_users, bins, universe, defense=mode, defense_strength=strength,
                           defense_hops=2, candidates_from_release=EDGELIST)
    out = {}
    for g in GROUPS:
        s, _, _ = fit_score(shadows, test_task, FEATURE_GROUPS[g], seed)
        m = metrics(test_task.y, s)
        out[g] = {k: float(m[k]) for k in ('auc', 'ap', 'precision@5', 'recall@100', 'recall@500')}
    return out


def run_unit(dataset: str, seed: int, user: int, edgelist: bool = True) -> str:
    global EDGELIST
    EDGELIST = edgelist
    out_path = OUT_ROOT / dataset / ('defense-edgelist' if edgelist else 'defense') / f'seed{seed}_u{user}.json'
    if out_path.exists():
        return f'skip {out_path.name}'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    edges, bins, universe = get_graph(dataset, 'real')
    train_users = [u for u in targets(dataset, 'main', seed) if u != user]
    edges_u = incident_edges(edges, user)
    neighbors = {(e.dst if e.src == user else e.src) for e in edges_u}
    b0, b1 = _window(edges_u, bins, pad=1)
    raw_release = remove_user(edges, user)
    rec = {'dataset': dataset, 'seed': seed, 'user': user, 'chronoscrub': {}, 'rr': {}}

    for name, mode, strength in CHRONOSCRUB:
        release = apply_chronoscrub_edges(raw_release, neighbors, bins, b0, b1, mode, strength, 2)
        test = build_task(edges, user, bins, defense=mode, defense_strength=strength, defense_hops=2,
                          universe=universe, compute_utility=True, candidates_from_release=EDGELIST)
        rec['chronoscrub'][name] = {
            'attack': attack(release, test, user, train_users, bins, universe, seed, mode, strength),
            'retention': len(release) / max(1, len(raw_release)),
            'utility_sign_auc': float(test.utility_sign_auc),
        }

    for eps in EPSILONS:
        release = apply_randomized_response(raw_release, epsilon=eps, seed=seed + user)
        # The attacker featurizes the RR release; build_task deletes nothing further because
        # the user is absent from it, and the universe still lists every platform user.
        test = build_task(release + edges_u, user, bins, universe=universe, compute_utility=False,
                          candidates_from_release=EDGELIST)
        util = sign_prediction_utility(raw_release, release, seed=user)
        rec['rr'][str(eps)] = {
            'attack': attack(release, test, user, train_users, bins, universe, seed),
            'retention': len(release) / max(1, len(raw_release)),
            'utility_sign_auc': float(util['sign_auc']),
        }
    rec['elapsed_sec'] = time.time() - t0
    out_path.write_text(json.dumps(rec, indent=1))
    return f'done {dataset} seed{seed} u{user} ({rec["elapsed_sec"]:.0f}s)'


def _safe(*args):
    try:
        return run_unit(*args)
    except Exception:
        return 'ERROR ' + repr(args) + '\n' + traceback.format_exc()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='bitcoinotc,bitcoinalpha,epinions_dated')
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--universe', choices=['edgelist', 'directory'], default='edgelist')
    args = ap.parse_args()
    units = [(ds, s, u, args.universe == 'edgelist') for ds in args.datasets.split(',')
             for s in map(int, args.seeds.split(',')) for u in targets(ds, 'main', s)]
    print(f'[{time.strftime("%H:%M:%S")}] defense units={len(units)}', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_safe, *u) for u in units]
        for i, f in enumerate(as_completed(futs), 1):
            print(f'[{time.strftime("%H:%M:%S")}] {i}/{len(units)} {f.result()}', flush=True)


if __name__ == '__main__':
    main()
