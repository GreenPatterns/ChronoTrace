from __future__ import annotations

from typing import Dict, List

import numpy as np

from .core import Edge, build_adj, incident_edges, metrics, remove_user, signed_stats, temporal_hist, signed_temporal_hist
from .features import _window, apply_chronoscrub_edges


def heuristic_scores(edges: List[Edge], user: int, bins: np.ndarray, defense='none', defense_strength=0.0, defense_hops=2, max_candidates: int = 0, return_scores: bool = False, candidates_from_release: bool = False) -> Dict:
    """Published style and ChronoTrace ablation scores for one deleted user.

    Scores are computed over the full remaining node universe after removing all
    incident edges of the deleted user. Because u is absent after deletion, link
    prediction heuristics are adapted as residual prominence scores: candidates
    are ranked by structural or temporal correlates in the released graph.
    """
    edges_u = incident_edges(edges, user)
    deleted_neighbors = {(e.dst if e.src == user else e.src) for e in edges_u}
    remaining_raw = remove_user(edges, user)
    b0, b1 = _window(edges_u, bins, pad=1)
    remaining = apply_chronoscrub_edges(remaining_raw, deleted_neighbors, bins, b0, b1, defense, defense_strength, defense_hops)
    adj = build_adj(remaining)
    if candidates_from_release:
        nodes = sorted({e.src for e in remaining} | {e.dst for e in remaining})
    else:
        nodes = sorted({e.src for e in edges} | {e.dst for e in edges})
    candidates = [v for v in nodes if v != user]

    # Subsample candidates for large graphs (e.g., Epinions with 131K nodes)
    if max_candidates > 0 and len(candidates) > max_candidates:
        pos_nodes = list(deleted_neighbors)
        neg_candidates = [v for v in candidates if v not in deleted_neighbors]
        rng = np.random.default_rng(user)
        n_neg = max_candidates - len(pos_nodes)
        if n_neg > 0 and len(neg_candidates) > n_neg:
            sampled_neg = rng.choice(neg_candidates, size=n_neg, replace=False).tolist()
        else:
            sampled_neg = neg_candidates
        candidates = pos_nodes + sampled_neg
    stats = signed_stats(remaining)
    h = temporal_hist(remaining, bins)
    sh = signed_temporal_hist(remaining, bins)

    all_degrees = {v: len(adj.get(v, set())) for v in candidates}
    # Residual anchor set: top temporally active nodes in the deletion window.
    win_activity = {}
    for v in candidates:
        hv = h.get(v, np.zeros(len(bins) - 1))
        win_activity[v] = float(hv[b0:b1+1].sum())
    anchor_count = max(5, min(50, len(deleted_neighbors) * 3 if deleted_neighbors else 10))
    anchors = set(sorted(candidates, key=lambda v: win_activity[v], reverse=True)[:anchor_count])

    scores = {name: [] for name in [
        'degree_prior','common_neighbors','jaccard','adamic_adar','preferential_attachment',
        'temporal_burst','temporal_shift','signed_balance','signed_motif','unsupervised_chronotrace'
    ]}
    y = []
    for v in candidates:
        y.append(1 if v in deleted_neighbors else 0)
        neigh = adj.get(v, set())
        deg = all_degrees.get(v, 0)
        cn = len(neigh & anchors)
        union = len(neigh | anchors)
        jac = cn / max(1, union)
        aa = 0.0
        for w in neigh & anchors:
            aa += 1.0 / max(1.0, np.log1p(len(adj.get(w, set()))))
        pa = deg * sum(all_degrees.get(a, 0) for a in anchors) / max(1, len(anchors))
        hv = h.get(v, np.zeros(len(bins) - 1))
        shv = sh.get(v, np.zeros((len(bins) - 1, 2)))
        burst = float(hv[b0:b1+1].sum() / max(1.0, hv.sum()))
        pre = hv[max(0, b0-3):b0].mean() if b0 > 0 else 0.0
        post = hv[b1+1:min(len(hv), b1+4)].mean() if b1 + 1 < len(hv) else 0.0
        shift = float(abs(post - pre))
        st = stats.get(v, {'pos':0.0, 'neg':0.0, 'out':0.0, 'in':0.0})
        signed_bal = float((st['pos'] - st['neg']) / max(1.0, st['pos'] + st['neg']))
        pos_burst = float(shv[b0:b1+1,0].sum() / max(1.0, shv[:,0].sum()))
        neg_burst = float(shv[b0:b1+1,1].sum() / max(1.0, shv[:,1].sum()))
        signed_motif = abs(signed_bal) + pos_burst + neg_burst + np.log1p(cn)
        unsup = np.log1p(deg) + np.log1p(cn) + 2.0*burst + shift + signed_motif
        vals = {
            'degree_prior': np.log1p(deg),
            'common_neighbors': cn,
            'jaccard': jac,
            'adamic_adar': aa,
            'preferential_attachment': pa,
            'temporal_burst': burst,
            'temporal_shift': shift,
            'signed_balance': abs(signed_bal),
            'signed_motif': signed_motif,
            'unsupervised_chronotrace': unsup,
        }
        for k, val in vals.items():
            scores[k].append(float(val))
    y_arr = np.asarray(y, dtype=int)
    method_metrics = {k: metrics(y_arr, np.asarray(v, dtype=float)) for k, v in scores.items()}
    out = {
        'user': user,
        'degree': int(y_arr.sum()),
        'n_candidates': len(candidates),
        'defense': {'mode': defense, 'strength': defense_strength, 'hops': defense_hops},
        'methods': method_metrics,
    }
    if return_scores:
        out['candidates'] = list(candidates)
        out['scores'] = {k: np.asarray(v, dtype=np.float32) for k, v in scores.items()}
    return out
