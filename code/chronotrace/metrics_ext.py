"""Metrics added in the revision: degree-controlled AUCs, rare-neighbor recall, pool-size effects.

All functions take one deletion task: labels ``y`` (1 = deleted neighbor), an
attacker's ``scores`` over the same candidates, and the candidates' released
degree ``deg`` (incident-edge count after deletion, the quantity the degree
feature ranks by).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import rankdata



def midranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks of ``scores`` in ascending order, ties sharing their average rank."""
    return rankdata(np.asarray(scores, dtype=float), method='average')


def auc_from_ranks(ranks: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC of the positive set ``pos`` (boolean mask) given midranks over all candidates."""
    n_pos = int(pos.sum())
    n_neg = len(ranks) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def degree_conditioned_expected_auc(y: np.ndarray, scores: np.ndarray, strata: np.ndarray) -> float:
    """Expected AUC of ``scores`` when the positives are re-drawn uniformly within each stratum.

    The re-draw keeps the number of positives in every stratum (for example, every
    exact released degree), so it keeps the true neighbors' degree histogram and
    destroys everything else. By linearity of expectation the expected rank sum of
    the positives is sum_b p_b * mean_rank_b, which gives the AUC in closed form.
    For a pure degree scorer with exact-degree strata it equals the actual AUC.
    """
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = midranks(scores)
    expected_rank_sum = 0.0
    for b in np.unique(strata[y]):
        members = strata == b
        expected_rank_sum += y[members].sum() * ranks[members].mean()
    return float((expected_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def degree_conditioned_null_draws(y: np.ndarray, scores: np.ndarray, strata: np.ndarray,
                                  n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo AUCs under the same within-stratum re-draw (for p-values and spread)."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ranks = midranks(scores)
    groups = []
    for b in np.unique(strata[y]):
        members = np.flatnonzero(strata == b)
        groups.append((ranks[members], int(y[members].sum())))
    out = np.empty(n_draws)
    for t in range(n_draws):
        total = 0.0
        for member_ranks, p_b in groups:
            total += rng.choice(member_ranks, size=p_b, replace=False).sum()
        out[t] = (total - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return out


def degree_matched_auc(y: np.ndarray, scores: np.ndarray, strata: np.ndarray) -> float:
    """AUC over (neighbor, non-neighbor) pairs of equal stratum only.

    A pure degree scorer scores exactly 0.5 here, so any excess is information
    the attacker has beyond the released degree.
    """
    y = np.asarray(y).astype(bool)
    weighted, pairs = 0.0, 0
    for b in np.unique(strata[y]):
        members = strata == b
        yb = y[members]
        p_b = int(yb.sum())
        n_b = int(len(yb) - p_b)
        if p_b == 0 or n_b == 0:
            continue
        weighted += auc_from_ranks(midranks(scores[members]), yb) * p_b * n_b
        pairs += p_b * n_b
    return float(weighted / pairs) if pairs else float('nan')


def positive_band_auc(y: np.ndarray, scores: np.ndarray, deg: np.ndarray,
                      bands: Sequence[tuple] = ((0, 3), (4, 20), (21, np.inf))) -> Dict[str, float]:
    """AUC of each band of true neighbors (by released degree) against all non-neighbors.

    Answers whether the attacker finds low-degree ("rare") neighbors or only hubs.
    """
    y = np.asarray(y).astype(bool)
    ranks = midranks(scores)
    out: Dict[str, float] = {}
    for lo, hi in bands:
        band = (deg >= lo) & (deg <= hi)
        keep = band & y | ~y
        sub_pos = (band & y)[keep]
        out[f'{int(lo)}-{"inf" if np.isinf(hi) else int(hi)}'] = (
            auc_from_ranks(midranks(scores[keep]), sub_pos) if sub_pos.any() else float('nan'))
        out[f'n_pos_{int(lo)}-{"inf" if np.isinf(hi) else int(hi)}'] = int(sub_pos.sum())
    return out


def surprisal_weighted_recall(y: np.ndarray, scores: np.ndarray, deg: np.ndarray, k: int) -> float:
    """Recall@k where each neighbor counts by its surprisal -log q_v, q_v proportional to degree + 1.

    A low-degree neighbor is unlikely under the degree prior, so finding it counts
    more than finding a hub.
    """
    y = np.asarray(y).astype(bool)
    if not y.any():
        return float('nan')
    q = (deg + 1.0) / float((deg + 1.0).sum())
    w = -np.log(q)
    top = np.zeros(len(y), dtype=bool)
    top[np.argsort(-scores, kind='stable')[:min(k, len(y))]] = True
    return float(w[y & top].sum() / w[y].sum())


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """AUC, AP and top-of-ranking metrics from one sort (fast path for repeated subsampling)."""
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return {}
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable')
    hits = y[order]
    pos_ranks = np.flatnonzero(hits) + 1
    cum = np.cumsum(hits)
    out = {'auc': auc_from_ranks(midranks(scores), y.astype(bool)),
           'ap': float((cum[pos_ranks - 1] / pos_ranks).mean()),
           'best_pos_rank': float(pos_ranks[0]), 'mean_pos_rank': float(pos_ranks.mean())}
    for k in (5, 10):
        out[f'precision@{k}'] = float(hits[:k].sum() / min(k, len(y)))
        out[f'success@{k}'] = float(hits[:k].sum() > 0)
    out['recall@100'] = float(hits[:100].sum() / n_pos)
    return out


def pool_size_metrics(y: np.ndarray, scores: np.ndarray, pool: int, rng: np.random.Generator,
                      n_draws: int = 20) -> Dict[str, float]:
    """Mean ranking metrics after keeping every neighbor and subsampling non-neighbors to ``pool``.

    Scores are fixed per candidate, so this is exactly the evaluation a smaller
    candidate universe would produce. AP here is the non-interpolated average
    precision (mean precision at each neighbor's rank), which equals scikit-learn's
    average_precision_score when scores have no ties.
    """
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_neg = pool - len(pos)
    if n_neg >= len(neg) or n_neg <= 0:
        return ranking_metrics(y, scores)
    acc: Dict[str, list] = {}
    for _ in range(n_draws):
        keep = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
        for k, v in ranking_metrics(y[keep], scores[keep]).items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}
