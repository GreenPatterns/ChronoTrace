#!/usr/bin/env python3
"""Can the attacker also recover the sign of a deleted relationship?

For every main target u and every true deleted neighbor v, the deleted
relationship is negative if any rating between u and v (either direction) was
negative. The attacker's predictor is v's residual signed balance in the
release (feature 1, computed after deletion): a lower balance should indicate a
negative relationship. We report the pooled AUC over all (u, v) pairs with a
cluster bootstrap over targets, and the base rate of negative relationships.

Output: paper/results/revision/sign_recovery.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronotrace.core import incident_edges
import run_revision
from run_revision import OUT_ROOT, SEEDS, get_graph, load_task, targets, task_path

run_revision.UNIVERSE = 'edgelist'  # neighbors left without edges are not in an edge-list release

N_BOOT = 10_000


def pairs_for(dataset: str):
    edges, _, _ = get_graph(dataset, 'real')
    out = []  # (target key, is_negative, residual signed balance of v)
    for seed in SEEDS:
        for u in targets(dataset, 'main', seed):
            task = load_task(task_path(dataset, 'real', 'main', seed, u))
            index = {v: i for i, v in enumerate(task.nodes)}
            neg = {}
            for e in incident_edges(edges, u):
                v = e.dst if e.src == u else e.src
                neg[v] = neg.get(v, False) or e.rating < 0
            for v, is_neg in neg.items():
                if v in index:
                    out.append(((seed, u), int(is_neg), float(task.X[index[v], 1])))
    return out


def main() -> None:
    result = {}
    for ds in ('bitcoinotc', 'bitcoinalpha', 'epinions_dated'):
        rows = pairs_for(ds)
        keys = sorted({k for k, _, _ in rows})
        by_key = {k: [(y, s) for kk, y, s in rows if kk == k] for k in keys}
        y = np.array([r[1] for r in rows])
        s = -np.array([r[2] for r in rows])  # lower balance -> more likely negative
        auc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float('nan')
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(keys), size=len(keys))
            yy = np.concatenate([[a for a, _ in by_key[keys[i]]] for i in pick])
            ss = np.concatenate([[-b for _, b in by_key[keys[i]]] for i in pick])
            if 0 < yy.sum() < len(yy):
                boots.append(roc_auc_score(yy, ss))
        result[ds] = {'n_pairs': int(len(y)), 'n_negative': int(y.sum()), 'base_rate': float(y.mean()),
                      'auc': auc, 'lo': float(np.percentile(boots, 2.5)), 'hi': float(np.percentile(boots, 97.5))}
        print(ds, result[ds], flush=True)
    (OUT_ROOT / 'sign_recovery.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
