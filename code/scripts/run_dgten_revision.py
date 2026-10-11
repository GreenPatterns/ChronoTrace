#!/usr/bin/env python3
"""DGTEN-variant representation baseline (appendix).

DGTEN is a temporal trust-prediction GNN. Its node embeddings are trained once
per dataset on the full pre-deletion graph, which contains the deleted edges, so
this baseline sees more than a release-only attacker may; any advantage it shows
is an upper bound. The embeddings are appended to the candidate features and the
logistic attacker is trained with the release-only protocol on the edge-list
candidate universe, exactly as in run_revision.py.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python run_dgten_revision.py [--workers N]
Outputs: paper/results/revision/<dataset>/dgten-edgelist/seed<S>_u<U>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_revision
from chronotrace.core import metrics
from chronotrace.dgten_features import compute_dgten_features, load_or_train_embeddings
from chronotrace.features import FEATURE_GROUPS
from chronotrace.protocols import fit_score, release_of, shadow_tasks
from run_revision import OUT_ROOT, SEEDS, get_graph, load_task, targets, task_path

run_revision.UNIVERSE = 'edgelist'
DATASETS = ['bitcoinotc', 'bitcoinalpha']
CACHE = OUT_ROOT / 'dgten'
SEED = 42
EPOCHS = 50


def with_embeddings(task, emb):
    extra = compute_dgten_features(emb['embeddings'], emb['chrono_to_dgten'], task.nodes)
    return replace(task, X=np.hstack([task.X, extra]))


def run_unit(dataset: str, seed: int, user: int) -> str:
    out = OUT_ROOT / dataset / 'dgten-edgelist' / f'seed{seed}_u{user}.json'
    if out.exists():
        return f'skip {out.name}'
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    emb = load_or_train_embeddings(dataset, seed=SEED, epochs=EPOCHS, cache_dir=CACHE)
    edges, bins, universe = get_graph(dataset, 'real')
    train_users = [u for u in targets(dataset, 'main', seed) if u != user]
    test = with_embeddings(load_task(task_path(dataset, 'real', 'main', seed, user)), emb)
    shadows = [with_embeddings(t, emb) for t in
               shadow_tasks(release_of(edges, user), user, train_users, bins, universe,
                            candidates_from_release=True)]
    n_base = len(FEATURE_GROUPS['chronotrace'])
    dg = list(range(n_base, n_base + emb['dim']))
    groups = {'dgten': dg, 'full_nonoracle_dgten': FEATURE_GROUPS['full_nonoracle'] + dg,
              'chronotrace_dgten': FEATURE_GROUPS['chronotrace'] + dg}
    rec = {'dataset': dataset, 'seed': seed, 'user': user, 'embedding_seed': SEED, 'epochs': EPOCHS,
           'dgten_sign_auc': float(emb['performance'].get('AUC', float('nan'))), 'results': {}}
    for name, cols in groups.items():
        s, _, _ = fit_score(shadows, test, cols, seed)
        rec['results'][name] = {k: float(v) for k, v in metrics(test.y, s).items()}
    rec['elapsed_sec'] = time.time() - t0
    out.write_text(json.dumps(rec, indent=1))
    return f'done {dataset} seed{seed} u{user} ({rec["elapsed_sec"]:.0f}s)'


def _safe(*args):
    try:
        return run_unit(*args)
    except Exception:
        return 'ERROR ' + repr(args) + '\n' + traceback.format_exc()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    for ds in DATASETS:  # train (or load) embeddings once, on the GPU, before forking workers
        emb = load_or_train_embeddings(ds, seed=SEED, epochs=EPOCHS, cache_dir=CACHE)
        print(f'{ds}: embedding dim {emb["dim"]}, DGTEN sign-prediction AUC {emb["performance"].get("AUC")}',
              flush=True)
    units = [(ds, s, u) for ds in DATASETS for s in SEEDS for u in targets(ds, 'main', s)]
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_safe, *u) for u in units]
        for i, f in enumerate(as_completed(futs), 1):
            print(f'[{time.strftime("%H:%M:%S")}] {i}/{len(units)} {f.result()}', flush=True)


if __name__ == '__main__':
    main()
