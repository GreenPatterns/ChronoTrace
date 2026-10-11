#!/usr/bin/env python3
"""Recompute the sign-prediction utility of every defended release with held-out training features.

The first defense runs featurized each training edge with its own sign included in its
endpoints' sign balance, while test edges (unseen) did not have it. This script rebuilds
each defended release exactly (ChronoScrub is a deterministic hash; randomized response is
seeded by seed + user), checks that the rebuilt release reproduces the stored retention and,
with the old featurization, the stored utility, and then replaces ``utility_sign_auc`` with
the corrected value. The old value is kept as ``utility_sign_auc_v1``. Attack results are
not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronotrace.core import incident_edges, remove_user
from chronotrace.features import _window, apply_chronoscrub_edges, apply_randomized_response
from chronotrace.utility import sign_prediction_utility
from run_defense_revision import CHRONOSCRUB, EPSILONS
from run_revision import OUT_ROOT, get_graph

TOL = 1e-9


def _same(a: float, b: float) -> bool:
    return (a != a and b != b) or abs(a - b) <= TOL


def recompute(path: str) -> str:
    p = Path(path)
    rec = json.loads(p.read_text())
    if rec.get('utility_version') == 2:
        return f'skip {p.name}'
    dataset, seed, user = rec['dataset'], int(rec['seed']), int(rec['user'])
    edges, bins, _ = get_graph(dataset, 'real')
    edges_u = incident_edges(edges, user)
    neighbors = {(e.dst if e.src == user else e.src) for e in edges_u}
    b0, b1 = _window(edges_u, bins, pad=1)
    raw = remove_user(edges, user)
    releases = [('chronoscrub', name, apply_chronoscrub_edges(raw, neighbors, bins, b0, b1, mode, strength, 2))
                for name, mode, strength in CHRONOSCRUB]
    releases += [('rr', str(eps), apply_randomized_response(raw, epsilon=eps, seed=seed + user)) for eps in EPSILONS]
    problems = []
    for fam, key, release in releases:
        r = rec[fam][key]
        if not _same(len(release) / max(1, len(raw)), r['retention']):
            problems.append(f'{fam}/{key} retention')
            continue
        old = sign_prediction_utility(raw, release, seed=user, hold_out_training_edge=False)['sign_auc']
        if not _same(old, r['utility_sign_auc']):
            problems.append(f'{fam}/{key} utility {old} != {r["utility_sign_auc"]}')
            continue
        r['utility_sign_auc_v1'] = r['utility_sign_auc']
        r['utility_sign_auc'] = float(sign_prediction_utility(raw, release, seed=user)['sign_auc'])
    if problems:
        return f'MISMATCH {p.name}: ' + '; '.join(problems)
    rec['utility_version'] = 2
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(p)
    return f'done {dataset} {p.name}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='bitcoinotc,bitcoinalpha,epinions_dated')
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()
    paths = [str(f) for ds in args.datasets.split(',') for f in sorted((OUT_ROOT / ds / 'defense-edgelist').glob('*.json'))]
    print(f'[{time.strftime("%H:%M:%S")}] units={len(paths)}', flush=True)
    bad = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(recompute, p) for p in paths]
        for i, f in enumerate(as_completed(futs), 1):
            msg = f.result()
            bad += msg.startswith('MISMATCH')
            print(f'[{time.strftime("%H:%M:%S")}] {i}/{len(paths)} {msg}', flush=True)
    print(f'mismatches: {bad}', flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
