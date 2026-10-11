#!/usr/bin/env python3
"""Dataset statistics for the revised paper (datasets table and transfer discussion).

Writes paper/results/revision/dataset_stats.json with, per dataset: users, ratings,
time span and bin width, share of distrust ratings, reciprocity, the median delay
between a rating and its reciprocal rating, Spearman correlation between pseudonymous
ID and first activity, the share of users with at most three distinct neighbors, and
degree heterogeneity (ratio of mean squared degree to squared mean degree).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronotrace.core import build_adj
from run_revision import N_BINS, OUT_ROOT, get_graph


def stats(dataset: str) -> dict:
    edges, bins, _ = get_graph(dataset, 'real')
    t = np.array([e.time for e in edges], dtype=float)
    when = {(e.src, e.dst): e.time for e in edges}
    lags = [abs(when[(d, s)] - ts) / 86400.0 for (s, d), ts in when.items() if (d, s) in when and s < d]
    adj = build_adj(edges)
    nbrs = np.array([len(a - {v}) for v, a in adj.items()], dtype=float)
    first = {}
    for e in edges:  # edges are time-sorted
        first.setdefault(e.src, e.time)
        first.setdefault(e.dst, e.time)
    ids = np.array(sorted(first))
    deg = Counter()
    for e in edges:
        deg[e.src] += 1
        deg[e.dst] += 1
    d = np.array(list(deg.values()), dtype=float)
    span_days = (t.max() - t.min()) / 86400.0
    return {
        'users': len(adj), 'ratings': len(edges),
        'first': float(t.min()), 'last': float(t.max()), 'span_days': span_days, 'bin_days': span_days / N_BINS,
        'distrust_share': float(np.mean([e.rating < 0 for e in edges])),
        'reciprocity': float(sum(1 for (s, dd) in when if (dd, s) in when) / len(when)),
        'reciprocal_lag_median_days': float(np.median(lags)) if lags else float('nan'),
        'reciprocal_lag_within_7d': float(np.mean(np.array(lags) <= 7)) if lags else float('nan'),
        'ratings_reciprocated_within_7d': float(np.mean([((d, s) in when and abs(when[(d, s)] - ts) <= 7 * 86400)
                                                  for (s, d), ts in when.items()])),
        'id_time_spearman': float(spearmanr(ids, [first[v] for v in ids]).correlation),
        'share_le3_neighbors': float(np.mean(nbrs <= 3)),
        'share_le3_events': float(np.mean(d <= 3)),
        'degree_heterogeneity': float((d ** 2).mean() / d.mean() ** 2),
    }


def main() -> None:
    out = {ds: stats(ds) for ds in ('bitcoinotc', 'bitcoinalpha', 'epinions_dated', 'epinions')}
    (OUT_ROOT / 'dataset_stats.json').write_text(json.dumps(out, indent=1))
    for ds, s in out.items():
        print(ds, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == '__main__':
    main()
