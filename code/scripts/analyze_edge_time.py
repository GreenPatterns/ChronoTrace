#!/usr/bin/env python3
"""When did a deleted relationship happen?

For every main target u and every former neighbor v present in the edge-list release,
the ground truth is the time bin of the earliest deleted event between u and v. Three
estimators read only what the adversary has once v is identified:

* ``busiest``: v's busiest bin over the whole release (release-only);
* ``window_busiest``: v's busiest bin inside u's window (window-informed);
* ``window_mid``: the middle of u's window (window-informed, ignores v);
* ``global_mode``: the busiest bin of the whole release (a naive baseline that ignores v and u).

We report the share of exact bins, the share within one bin, and the mean absolute
error in bins, with 95% cluster-bootstrap intervals over targets.
Output: paper/results/revision/edge_time.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_revision
from chronotrace.core import _bin_index, incident_edges
from run_revision import OUT_ROOT, SEEDS, get_graph, load_task, targets, task_path

run_revision.UNIVERSE = 'edgelist'
N_BOOT = 10_000
ESTIMATORS = ('global_mode', 'busiest', 'window_busiest', 'window_mid')


def rows_for(dataset: str):
    edges, bins, _ = get_graph(dataset, 'real')
    rows = []  # (target key, true bin, {estimator: predicted bin})
    counts = np.bincount([_bin_index(bins, e.time) for e in edges], minlength=len(bins) - 1)
    global_mode = int(np.argmax(counts))
    for seed in SEEDS:
        for u in targets(dataset, 'main', seed):
            task = load_task(task_path(dataset, 'real', 'main', seed, u))
            index = {v: i for i, v in enumerate(task.nodes)}
            b0, b1 = task.window
            first = {}
            for e in incident_edges(edges, u):
                v = e.dst if e.src == u else e.src
                first[v] = min(first.get(v, e.time), e.time)
            for v, t in first.items():
                if v not in index:
                    continue
                h = task.hist[index[v]]
                inside = h[b0:b1 + 1]
                pred = {'global_mode': global_mode, 'busiest': int(np.argmax(h)),
                        'window_busiest': int(b0 + np.argmax(inside)),
                        'window_mid': int((b0 + b1) // 2)}
                rows.append(((seed, u), _bin_index(bins, t), pred))
    return rows


def summarize(rows):
    keys = sorted({k for k, _, _ in rows})
    by_key = {k: [(t, p) for kk, t, p in rows if kk == k] for k in keys}

    def stats(sample):
        out = {}
        for est in ESTIMATORS:
            err = np.array([abs(p[est] - t) for t, p in sample])
            out[est] = (float(np.mean(err == 0)), float(np.mean(err <= 1)), float(err.mean()))
        return out

    point = stats([r for k in keys for r in by_key[k]])
    rng = np.random.default_rng(42)
    boots = {est: [] for est in ESTIMATORS}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(keys), size=len(keys))
        s = stats([r for i in pick for r in by_key[keys[i]]])
        for est in ESTIMATORS:
            boots[est].append(s[est])
    result = {'n_pairs': len(rows), 'n_targets': len(keys)}
    for est in ESTIMATORS:
        b = np.array(boots[est])
        result[est] = {name: {'value': point[est][j], 'lo': float(np.percentile(b[:, j], 2.5)),
                              'hi': float(np.percentile(b[:, j], 97.5))}
                       for j, name in enumerate(('exact', 'within_one', 'mae_bins'))}
    return result


def main() -> None:
    out = {}
    for ds in ('bitcoinotc', 'bitcoinalpha', 'epinions_dated'):
        out[ds] = summarize(rows_for(ds))
        print(ds, {e: {k: round(v['value'], 3) for k, v in out[ds][e].items()} for e in ESTIMATORS}, flush=True)
    (OUT_ROOT / 'edge_time.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
