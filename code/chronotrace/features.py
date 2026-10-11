from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional

import numpy as np

from .core import (
    Edge, build_adj, incident_edges, nodes_of, profile_for, remove_user,
    signed_stats, signed_temporal_hist, temporal_hist,
)
from .utility import sign_prediction_utility

FEATURE_NAMES = [
    'log_degree',
    'signed_balance',
    'out_in_balance',
    'window_activity_ratio',
    'pre_post_shift',
    'positive_window_ratio',
    'negative_window_ratio',
    'log_neighbor_degree',
    'log_twohop',
    'clustering_proxy',
    'temporal_concentration',
]

FEATURE_GROUPS = {
    'degree': [0],
    'topology': [0, 7, 8, 9],
    'full_nonoracle': [0, 1, 2, 7, 8, 9, 10],
    'temporal': [3, 4, 10],
    'motif': [1, 2, 5, 6, 7, 8, 9],
    'chronotrace': list(range(len(FEATURE_NAMES))),
}


def get_dgten_feature_groups(dgten_dim: int) -> Dict[str, List[int]]:
    """Return feature group dicts including DGTEN features.
    
    Args:
        dgten_dim: Dimension of DGTEN embeddings
    
    Returns:
        Dictionary of feature group names to feature indices
    """
    n_base = len(FEATURE_NAMES)
    
    groups = dict(FEATURE_GROUPS)
    
    # DGTEN-only features
    groups['dgten'] = list(range(n_base, n_base + dgten_dim))
    
    # All nonoracle + DGTEN
    groups['full_nonoracle_dgten'] = groups['full_nonoracle'] + groups['dgten']
    
    # ChronoTrace + DGTEN (full feature set)
    groups['chronotrace_dgten'] = list(range(n_base + dgten_dim))
    
    return groups


@dataclass
class Task:
    user: int
    nodes: List[int]
    X: np.ndarray
    y: np.ndarray
    profile: dict
    utility_edges_before: int
    utility_edges_after: int
    utility_sign_auc: float
    utility_sign_ap: float
    utility_train_edges: int
    utility_test_edges: int
    feature_names: List[str]
    window: tuple = (0, 0)
    hist: Optional[np.ndarray] = None         # (n_cand, n_bins) edge counts per bin
    signed_hist: Optional[np.ndarray] = None  # (n_cand, n_bins, 2) positive/negative counts per bin
    released_degree: Optional[np.ndarray] = None  # incident-edge count of each candidate in the release


def _window(edges_u: List[Edge], bins: np.ndarray, pad: int = 1) -> tuple[int, int]:
    if not edges_u:
        return 0, len(bins) - 2
    t0 = min(e.time for e in edges_u)
    t1 = max(e.time for e in edges_u)
    b0 = int(np.searchsorted(bins, t0, side='right') - 1)
    b1 = int(np.searchsorted(bins, t1, side='right') - 1)
    return max(0, b0 - pad), min(len(bins) - 2, b1 + pad)


def residue_nodes(adj: Dict[int, set], deleted_neighbors: set, hops: int) -> set:
    cur = set(deleted_neighbors)
    seen = set(cur)
    for _ in range(max(0, hops - 1)):
        nxt = set()
        for v in cur:
            nxt.update(adj.get(v, set()))
        nxt -= seen
        seen.update(nxt)
        cur = nxt
    return seen


def apply_chronoscrub_edges(
    remaining: List[Edge],
    deleted_neighbors: set,
    bins: np.ndarray,
    b0: int,
    b1: int,
    mode: str,
    strength: float,
    hops: int,
) -> List[Edge]:
    """Joint defense on released graph.

    temporal_smooth: drop a 0.45 x strength fraction of the edges inside the deletion
    window that touch a residue node.
    joint: additionally drop a 0.25 x strength fraction of the edges between two
    residue nodes, at any time. The fractions are realized by a deterministic hash
    of each edge. This is intentionally a simple auditable defense baseline, not a
    claimed formal guarantee.
    """
    if mode == 'none' or strength <= 0:
        return list(remaining)
    adj = build_adj(remaining)
    res = residue_nodes(adj, deleted_neighbors, hops)
    kept: List[Edge] = []
    for e in remaining:
        in_window = b0 <= int(np.searchsorted(bins, e.time, side='right') - 1) <= b1
        touches_residue = e.src in res or e.dst in res
        both_residue = e.src in res and e.dst in res
        drop_score = 0.0
        if mode in {'temporal_smooth', 'joint'} and in_window and touches_residue:
            drop_score = max(drop_score, 0.45 * strength)
        if mode == 'joint' and both_residue:
            drop_score = max(drop_score, 0.25 * strength)
        # Deterministic pseudo-random drop, stable across runs and independent of labels.
        key = (e.src * 1000003 + e.dst * 9176 + e.time * 37 + (1 if e.rating > 0 else 2)) & 0xffffffff
        keep_threshold = (key % 100000) / 100000.0
        if keep_threshold >= drop_score:
            kept.append(e)
    return kept


def apply_randomized_response(
    remaining: List[Edge],
    epsilon: float,
    seed: int = 42,
) -> List[Edge]:
    """Deletion-only edge-level randomized response.

    Each edge is independently retained with probability p = e^ε / (1 + e^ε)
    and dropped with probability 1 / (1 + e^ε). Because no fake edges are added,
    this is not edge-level differential privacy: an absent edge is certainly absent
    from the original graph. Full randomized response would also add each non-edge
    with probability 1/(1+e^ε), which is impractical for graphs of this size.

    Args:
        remaining: Edge list after node deletion
        epsilon: Privacy parameter (lower = more private, higher = more utility)
        seed: Random seed for reproducibility
    """
    if epsilon <= 0:
        return []  # Perfect privacy = drop everything
    p_keep = np.exp(epsilon) / (1.0 + np.exp(epsilon))
    rng = np.random.default_rng(seed)
    kept = [e for e in remaining if rng.random() < p_keep]
    return kept


def build_task(edges: List[Edge], user: int, bins: np.ndarray, defense: str = 'none', defense_strength: float = 0.0, defense_hops: int = 2, max_candidates: int = 0, dgten_data: Optional[Dict] = None, window_override: Optional[tuple] = None, universe: Optional[List[int]] = None, compute_utility: bool = True, keep_hist: bool = False, candidates_from_release: bool = False) -> Task:
    """Build one deletion task: delete ``user`` from ``edges`` and featurize every candidate.

    ``universe`` fixes the candidate node set explicitly (default: every node that
    appears in ``edges``, i.e. a directory release in which every account stays
    listed). With ``candidates_from_release`` the candidates are the nodes that still
    have at least one edge in the published release (an edge-list release); a former
    neighbor whose only edges were with ``user`` then disappears from the candidates.
    """
    edges_u = incident_edges(edges, user)
    deleted_neighbors = {(e.dst if e.src == user else e.src) for e in edges_u}
    remaining_raw = remove_user(edges, user)
    if window_override is not None:
        b0, b1 = int(window_override[0]), int(window_override[1])
    else:
        b0, b1 = _window(edges_u, bins, pad=1)
    remaining = apply_chronoscrub_edges(remaining_raw, deleted_neighbors, bins, b0, b1, defense, defense_strength, defense_hops)

    if candidates_from_release:
        base_nodes = nodes_of(remaining)
    else:
        base_nodes = nodes_of(edges) if universe is None else sorted(universe)
    all_nodes = [v for v in base_nodes if v != user]
    
    # Subsample candidates for large graphs (e.g., Epinions with 131K nodes)
    if max_candidates > 0 and len(all_nodes) > max_candidates:
        pos_nodes = list(deleted_neighbors)
        neg_candidates = [v for v in all_nodes if v not in deleted_neighbors]
        rng = np.random.default_rng(user)
        n_neg = max_candidates - len(pos_nodes)
        if n_neg > 0 and len(neg_candidates) > n_neg:
            sampled_neg = rng.choice(neg_candidates, size=n_neg, replace=False).tolist()
        else:
            sampled_neg = neg_candidates
        all_nodes = pos_nodes + sampled_neg
    adj = build_adj(remaining)
    stats = signed_stats(remaining)
    h = temporal_hist(remaining, bins)
    sh = signed_temporal_hist(remaining, bins)

    X = []
    y = []
    for v in all_nodes:
        st = stats.get(v, {'deg': 0.0, 'pos': 0.0, 'neg': 0.0, 'out': 0.0, 'in': 0.0})
        deg = float(st['deg'])
        pos = float(st['pos'])
        neg = float(st['neg'])
        signed_balance = (pos - neg) / max(1.0, pos + neg)
        out_in = (float(st['out']) - float(st['in'])) / max(1.0, float(st['out']) + float(st['in']))
        hv = h.get(v, np.zeros(len(bins) - 1))
        shv = sh.get(v, np.zeros((len(bins) - 1, 2)))
        window_sum = hv[b0:b1 + 1].sum()
        burst = float(window_sum / max(1.0, hv.sum()))
        pre = hv[max(0, b0 - 3):b0].mean() if b0 > 0 else 0.0
        post = hv[b1 + 1:min(len(hv), b1 + 4)].mean() if b1 + 1 < len(hv) else 0.0
        shift = float(abs(post - pre))
        pos_burst = float(shv[b0:b1 + 1, 0].sum() / max(1.0, shv[:, 0].sum()))
        neg_burst = float(shv[b0:b1 + 1, 1].sum() / max(1.0, shv[:, 1].sum()))
        neigh = adj.get(v, set())
        neigh_deg = float(np.mean([stats.get(w, {'deg': 0.0})['deg'] for w in neigh])) if neigh else 0.0
        twohop_set = set()
        for w in neigh:
            twohop_set.update(adj.get(w, set()))
        twohop = len(twohop_set - neigh - {v})
        clustering_proxy = len(neigh) / max(1.0, np.sqrt(twohop + 1.0))
        temporal_concentration = 0.0
        if hv.sum() > 0:
            p = hv / hv.sum()
            entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(hv)))
            temporal_concentration = 1.0 - entropy
        X.append([
            np.log1p(deg), signed_balance, out_in, burst, shift, pos_burst, neg_burst,
            np.log1p(neigh_deg), np.log1p(twohop), clustering_proxy, temporal_concentration,
        ])
        y.append(1 if v in deleted_neighbors else 0)

    # Convert to numpy array
    X_array = np.asarray(X, dtype=float)
    
    # Append DGTEN features if provided
    feature_names = list(FEATURE_NAMES)
    if dgten_data is not None:
        from .dgten_features import compute_dgten_features, get_dgten_feature_names
        dgten_features = compute_dgten_features(
            dgten_data['embeddings'],
            dgten_data['chrono_to_dgten'],
            all_nodes
        )
        X_array = np.hstack([X_array, dgten_features])
        feature_names = feature_names + get_dgten_feature_names(dgten_data['dim'])

    if compute_utility:
        utility = sign_prediction_utility(remaining_raw, remaining, seed=user)
    else:
        utility = {'sign_auc': float('nan'), 'sign_ap': float('nan'), 'train_edges': 0, 'test_edges': 0}

    return Task(
        user=user,
        nodes=all_nodes,
        X=X_array,
        y=np.asarray(y, dtype=int),
        profile=asdict(profile_for(edges_u, user)),
        utility_edges_before=len(remaining_raw),
        utility_edges_after=len(remaining),
        utility_sign_auc=float(utility['sign_auc']),
        utility_sign_ap=float(utility['sign_ap']),
        utility_train_edges=int(utility['train_edges']),
        utility_test_edges=int(utility['test_edges']),
        feature_names=feature_names,
        window=(int(b0), int(b1)),
        hist=np.asarray([h.get(v, np.zeros(len(bins) - 1)) for v in all_nodes], dtype=np.float32) if keep_hist else None,
        signed_hist=np.asarray([sh.get(v, np.zeros((len(bins) - 1, 2))) for v in all_nodes], dtype=np.float32) if keep_hist else None,
        released_degree=np.asarray([stats.get(v, {'deg': 0.0})['deg'] for v in all_nodes], dtype=np.float32),
    )


WINDOW_COLUMNS = (3, 4, 5, 6)


def window_columns(hist: np.ndarray, signed_hist: np.ndarray, b0: int, b1: int) -> np.ndarray:
    """Vectorized features 3-6 for window [b0, b1]; matches the per-candidate loop in build_task."""
    n_bins = hist.shape[1]
    hist = hist.astype(float)
    signed_hist = signed_hist.astype(float)
    burst = hist[:, b0:b1 + 1].sum(axis=1) / np.maximum(1.0, hist.sum(axis=1))
    pre = hist[:, max(0, b0 - 3):b0].mean(axis=1) if b0 > 0 else np.zeros(len(hist))
    post = hist[:, b1 + 1:min(n_bins, b1 + 4)].mean(axis=1) if b1 + 1 < n_bins else np.zeros(len(hist))
    shift = np.abs(post - pre)
    pos_burst = signed_hist[:, b0:b1 + 1, 0].sum(axis=1) / np.maximum(1.0, signed_hist[:, :, 0].sum(axis=1))
    neg_burst = signed_hist[:, b0:b1 + 1, 1].sum(axis=1) / np.maximum(1.0, signed_hist[:, :, 1].sum(axis=1))
    return np.column_stack([burst, shift, pos_burst, neg_burst])


def with_window(task: Task, b0: int, b1: int) -> Task:
    """Copy of ``task`` whose window features (3-6) use [b0, b1] instead of the task's own window.

    Requires a task built with ``keep_hist=True``. Every other column is unchanged,
    because features 0-2 and 7-10 do not depend on the window.
    """
    if task.hist is None or task.signed_hist is None:
        raise ValueError('with_window needs a task built with keep_hist=True')
    X = task.X.copy()
    X[:, list(WINDOW_COLUMNS)] = window_columns(task.hist, task.signed_hist, int(b0), int(b1))
    return replace(task, X=X, window=(int(b0), int(b1)))
