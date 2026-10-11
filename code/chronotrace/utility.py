from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import Edge, build_adj, signed_stats


_NO_STATS = {'deg': 0.0, 'pos': 0.0, 'neg': 0.0, 'out': 0.0, 'in': 0.0}


def _row(ns: set, nd: set, ss: Dict[str, float], sd: Dict[str, float]) -> List[float]:
    inter = len(ns & nd)
    union = max(1, len(ns | nd))
    bal_s = (ss['pos'] - ss['neg']) / max(1.0, ss['pos'] + ss['neg'])
    bal_d = (sd['pos'] - sd['neg']) / max(1.0, sd['pos'] + sd['neg'])
    return [
        np.log1p(ss['deg']), np.log1p(sd['deg']),
        np.log1p(ss['out']), np.log1p(sd['in']),
        bal_s, bal_d,
        np.log1p(inter), inter / union,
    ]


def _edge_feature(e: Edge, adj: Dict[int, set], stats: Dict[int, Dict[str, float]]) -> List[float]:
    """Features of an edge that is not part of the graph given by ``adj`` and ``stats``."""
    return _row(adj.get(e.src, set()), adj.get(e.dst, set()), stats.get(e.src, _NO_STATS), stats.get(e.dst, _NO_STATS))


def _held_out(st: Dict[str, float], e: Edge, node: int) -> Dict[str, float]:
    """``node``'s statistics without the contribution of edge ``e`` (as signed_stats adds it)."""
    out = dict(st)
    for n, direction in ((e.src, 'out'), (e.dst, 'in')):
        if n == node:
            out['deg'] -= 1.0
            out[direction] -= 1.0
            out['pos' if e.rating > 0 else 'neg'] -= 1.0
    return out


def test_features(test_edges: List[Edge], train_edges: List[Edge]) -> np.ndarray:
    adj, stats = build_adj(train_edges), signed_stats(train_edges)
    return np.asarray([_edge_feature(e, adj, stats) for e in test_edges], dtype=float)


def training_features(train_edges: List[Edge]) -> np.ndarray:
    """Featurize each training edge with the edge itself held out, exactly as a test edge is
    featurized: its own sign, degree contribution and adjacency do not enter its features."""
    adj, stats = build_adj(train_edges), signed_stats(train_edges)
    pairs = Counter((min(e.src, e.dst), max(e.src, e.dst)) for e in train_edges)
    rows = []
    for e in train_edges:
        ns, nd = adj[e.src], adj[e.dst]
        if pairs[(min(e.src, e.dst), max(e.src, e.dst))] == 1:
            ns, nd = ns - {e.dst}, nd - {e.src}
        rows.append(_row(ns, nd, _held_out(stats[e.src], e, e.src), _held_out(stats[e.dst], e, e.dst)))
    return np.asarray(rows, dtype=float)


def sign_prediction_utility(original_remaining: List[Edge], defended_remaining: List[Edge], seed: int, max_train: int = 12000, max_test: int = 4000, hold_out_training_edge: bool = True) -> Dict[str, float]:
    """Temporal sign-prediction utility for the released graph.

    The test set is fixed from the original post-deletion graph's latest 20 percent
    edges. The training set is built from the defended release's earlier edges.
    Thus privacy scrubbing pays a real utility cost if it removes useful training
    edges or damages local signed structure. Every edge, training or test, is
    featurized from the training graph without that edge, so a training edge's own
    sign never enters its features.
    """
    if len(original_remaining) < 50 or len(defended_remaining) < 20:
        return {'sign_auc': float('nan'), 'sign_ap': float('nan'), 'train_edges': len(defended_remaining), 'test_edges': 0}
    orig = sorted(original_remaining, key=lambda e: e.time)
    cutoff = int(0.8 * len(orig))
    test_edges = orig[cutoff:]
    test_min_time = test_edges[0].time if test_edges else orig[-1].time
    train_edges = [e for e in defended_remaining if e.time < test_min_time]
    if len(train_edges) < 20 or len(test_edges) < 10:
        return {'sign_auc': float('nan'), 'sign_ap': float('nan'), 'train_edges': len(train_edges), 'test_edges': len(test_edges)}
    rng = np.random.default_rng(seed + len(original_remaining) + len(defended_remaining))
    if len(train_edges) > max_train:
        train_edges = list(rng.choice(np.asarray(train_edges, dtype=object), size=max_train, replace=False))
    if len(test_edges) > max_test:
        test_edges = list(rng.choice(np.asarray(test_edges, dtype=object), size=max_test, replace=False))
    # hold_out_training_edge=False reproduces the first revision runs, whose training edges
    # were featurized with their own sign included (kept only to verify recomputation).
    Xtr = training_features(train_edges) if hold_out_training_edge else test_features(train_edges, train_edges)
    ytr = np.asarray([1 if e.rating > 0 else 0 for e in train_edges], dtype=int)
    Xte = test_features(test_edges, train_edges)
    yte = np.asarray([1 if e.rating > 0 else 0 for e in test_edges], dtype=int)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {'sign_auc': float('nan'), 'sign_ap': float('nan'), 'train_edges': len(train_edges), 'test_edges': len(test_edges)}
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed))
    clf.fit(Xtr, ytr)
    score = clf.predict_proba(Xte)[:, 1]
    return {
        'sign_auc': float(roc_auc_score(yte, score)),
        'sign_ap': float(average_precision_score(yte, score)),
        'train_edges': int(len(train_edges)),
        'test_edges': int(len(test_edges)),
    }
