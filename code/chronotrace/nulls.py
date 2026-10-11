"""Reference models that calibrate the attack.

Two kinds of reference are provided.

* Label nulls keep the released graph and redraw the "neighbors" of the deleted
  node: uniformly, or with probability proportional to released degree (the
  configuration-model expectation for a random endpoint). They score the
  degree ranker only, using the rank trick, so each trial costs O(|N_u|).
* Graph nulls rewire the original graph before deletion and rerun the full
  attack. ``config`` permutes edge destinations globally and keeps every node's
  in- and out-degree. ``signed_temporal`` permutes destinations only among edges
  of the same time bin and sign, so it also keeps every node's per-bin, per-sign
  in- and out-counts. Both randomize who interacts with whom, including
  reciprocity.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .core import Edge, _bin_index
from .metrics_ext import auc_from_ranks, midranks


def label_null_aucs(deg: np.ndarray, n_pos: int, rng: np.random.Generator,
                    n_trials: int = 200) -> Dict[str, float]:
    """Degree-ranker AUC when |N_u| pseudo-neighbors are drawn from the same candidates.

    ``deg`` is the released incident-edge count of every candidate in the task's
    candidate universe; the ranker scores log(1 + deg), so midranks of ``deg``
    are its ranks.
    """
    ranks = midranks(deg)
    n = len(deg)
    k = int(min(n_pos, n - 1))
    weights = deg.astype(float)
    p = weights / weights.sum() if weights.sum() > 0 else None
    out = {}
    for name, probs in (('uniform', None), ('degprop', p)):
        vals = np.empty(n_trials)
        for t in range(n_trials):
            idx = rng.choice(n, size=k, replace=False, p=probs)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            vals[t] = auc_from_ranks(ranks, mask)
        out[f'{name}_mean'] = float(vals.mean())
        out[f'{name}_std'] = float(vals.std())
    return out


def rewire(edges: List[Edge], bins: np.ndarray, mode: str, seed: int,
           max_fix_rounds: int = 50) -> Tuple[List[Edge], Dict[str, int]]:
    """Stub-matching rewiring that preserves node-level marginals (see module docstring).

    Destinations are permuted within groups (one group for ``config``; one per
    (time bin, sign) for ``signed_temporal``). Self-loops created by the
    permutation are repaired by swapping destinations inside the same group;
    any that cannot be repaired are dropped and counted.
    """
    rng = np.random.default_rng(seed)
    src = np.array([e.src for e in edges], dtype=np.int64)
    dst = np.array([e.dst for e in edges], dtype=np.int64)
    if mode == 'config':
        keys = np.zeros(len(edges), dtype=np.int64)
    elif mode == 'signed_temporal':
        keys = np.array([_bin_index(bins, e.time) * 2 + (0 if e.rating > 0 else 1) for e in edges],
                        dtype=np.int64)
    else:
        raise ValueError(f'unknown rewiring mode {mode!r}')
    new_dst = dst.copy()
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        new_dst[idx] = dst[rng.permutation(idx)]
        for _ in range(max_fix_rounds):
            loops = idx[src[idx] == new_dst[idx]]
            if len(loops) == 0:
                break
            partners = rng.choice(idx, size=len(loops))
            for i, j in zip(loops, partners):
                if src[i] != new_dst[j] and src[j] != new_dst[i]:
                    new_dst[i], new_dst[j] = new_dst[j], new_dst[i]
    keep = src != new_dst
    out = [Edge(int(s), int(d), e.rating, e.time)
           for s, d, e, kp in zip(src, new_dst, edges, keep) if kp]
    out.sort(key=lambda e: e.time)
    pairs = np.stack([src[keep], new_dst[keep]], axis=1)
    n_multi = len(pairs) - len(np.unique(pairs, axis=0))
    return out, {'dropped_self_loops': int((~keep).sum()), 'duplicate_directed_pairs': int(n_multi)}
