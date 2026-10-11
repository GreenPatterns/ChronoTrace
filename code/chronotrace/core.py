from __future__ import annotations

import csv
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

def _default_data_root() -> Path:
    """The repository's data/ folder if it holds the datasets, else a shared ../data/cyberdata."""
    repo = Path(__file__).resolve().parents[2]  # code/chronotrace/core.py -> repository root
    for candidate in (repo / 'data', repo.parent / 'data' / 'cyberdata'):
        if (candidate / 'bitcoinotc.csv').exists():
            return candidate
    return repo / 'data'


DATA_ROOT = Path(os.environ.get('CHRONOTRACE_DATA_DIR', str(_default_data_root())))
DATASETS = {
    'bitcoinotc': DATA_ROOT / 'bitcoinotc.csv',
    'bitcoinalpha': DATA_ROOT / 'bitcoinalpha.csv',
    'epinions': DATA_ROOT / 'epinions.csv',
}


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    rating: int
    time: int


@dataclass(frozen=True)
class Profile:
    user: int
    degree: int
    pos_edges: int
    neg_edges: int
    first_time: int
    last_time: int
    span_days: float


def load_edges(path: Path, binarization_threshold: int = 0) -> List[Edge]:
    edges: List[Edge] = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = {h.strip(): h for h in reader.fieldnames or []}
        for row in reader:
            try:
                s = int(row[fields['source']])
                d = int(row[fields['target']])
                r = int(float(row[fields['rating']]))
                t = int(float(row[fields['time']]))
            except Exception as exc:
                warnings.warn(f'Skipping malformed CSV row: {exc}', stacklevel=2)
                continue
            edges.append(Edge(s, d, 1 if r > binarization_threshold else -1, t))
    edges.sort(key=lambda e: e.time)
    return edges


def nodes_of(edges: Iterable[Edge]) -> List[int]:
    nodes = set()
    for e in edges:
        nodes.add(e.src)
        nodes.add(e.dst)
    return sorted(nodes)


def incident_edges(edges: List[Edge], u: int) -> List[Edge]:
    return [e for e in edges if e.src == u or e.dst == u]


def remove_user(edges: List[Edge], u: int) -> List[Edge]:
    return [e for e in edges if e.src != u and e.dst != u]


def profile_for(edges_u: List[Edge], u: int) -> Profile:
    times = [e.time for e in edges_u]
    if not times:
        return Profile(u, 0, 0, 0, 0, 0, 0.0)
    return Profile(
        user=u,
        degree=len(edges_u),
        pos_edges=sum(1 for e in edges_u if e.rating > 0),
        neg_edges=sum(1 for e in edges_u if e.rating < 0),
        first_time=min(times),
        last_time=max(times),
        span_days=(max(times) - min(times)) / 86400.0,
    )


def choose_users(edges: List[Edge], n_users: int, seed: int, min_degree: int, max_degree: int | None) -> List[int]:
    deg: Dict[int, int] = {}
    for e in edges:
        deg[e.src] = deg.get(e.src, 0) + 1
        deg[e.dst] = deg.get(e.dst, 0) + 1
    candidates = [u for u, d in deg.items() if d >= min_degree and (max_degree is None or d <= max_degree)]
    if len(candidates) <= n_users:
        return sorted(candidates)
    qs = np.quantile([deg[u] for u in candidates], [0.25, 0.5, 0.75])
    buckets = [[], [], [], []]
    for u in candidates:
        d = deg[u]
        idx = 0 if d <= qs[0] else 1 if d <= qs[1] else 2 if d <= qs[2] else 3
        buckets[idx].append(u)
    rng = random.Random(seed)
    chosen: List[int] = []
    per = max(1, math.ceil(n_users / 4))
    for b in buckets:
        rng.shuffle(b)
        chosen.extend(b[:per])
    rng.shuffle(chosen)
    return chosen[:n_users]


def build_adj(edges: List[Edge]) -> Dict[int, set]:
    adj: Dict[int, set] = {}
    for e in edges:
        adj.setdefault(e.src, set()).add(e.dst)
        adj.setdefault(e.dst, set()).add(e.src)
    return adj


def signed_stats(edges: List[Edge]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for e in edges:
        for node, direction in ((e.src, 'out'), (e.dst, 'in')):
            st = out.setdefault(node, {'deg': 0.0, 'pos': 0.0, 'neg': 0.0, 'out': 0.0, 'in': 0.0})
            st['deg'] += 1.0
            st[direction] += 1.0
            st['pos' if e.rating > 0 else 'neg'] += 1.0
    return out


def time_bins(edges: List[Edge], n_bins: int) -> np.ndarray:
    times = np.asarray([e.time for e in edges], dtype=float)
    return np.linspace(times.min(), times.max() + 1.0, n_bins + 1)


def _bin_index(bins: np.ndarray, t: int) -> int:
    return max(0, min(len(bins) - 2, int(np.searchsorted(bins, t, side='right') - 1)))


def temporal_hist(edges: List[Edge], bins: np.ndarray) -> Dict[int, np.ndarray]:
    h: Dict[int, np.ndarray] = {}
    for e in edges:
        b = _bin_index(bins, e.time)
        for node in (e.src, e.dst):
            h.setdefault(node, np.zeros(len(bins) - 1, dtype=float))[b] += 1.0
    return h


def signed_temporal_hist(edges: List[Edge], bins: np.ndarray) -> Dict[int, np.ndarray]:
    h: Dict[int, np.ndarray] = {}
    for e in edges:
        b = _bin_index(bins, e.time)
        c = 0 if e.rating > 0 else 1
        for node in (e.src, e.dst):
            h.setdefault(node, np.zeros((len(bins) - 1, 2), dtype=float))[b, c] += 1.0
    return h


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(average_precision_score(y, score))


def recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    if y.sum() <= 0:
        return float('nan')
    idx = np.argsort(-score)[:min(k, len(y))]
    return float(y[idx].sum() / y.sum())


def mcc_at_topk(y: np.ndarray, score: np.ndarray, k: int) -> float:
    pred = np.zeros_like(y)
    pred[np.argsort(-score)[:min(k, len(y))]] = 1
    return float(matthews_corrcoef(y, pred))



def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Fraction of top-k that are true positives."""
    k = min(k, len(y))
    if k <= 0:
        return float('nan')
    idx = np.argsort(-score)[:k]
    return float(y[idx].sum() / k)

def success_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """1 if at least one true positive in top-k, else 0."""
    k = min(k, len(y))
    if k <= 0 or y.sum() <= 0:
        return float('nan')
    idx = np.argsort(-score)[:k]
    return 1.0 if y[idx].sum() > 0 else 0.0

def best_positive_rank(y: np.ndarray, score: np.ndarray) -> float:
    """Rank of highest-ranked true positive (1-indexed, lower = better)."""
    if y.sum() <= 0:
        return float('nan')
    order = np.argsort(-score)
    pos_ranks = np.where(y[order] == 1)[0] + 1
    return float(pos_ranks[0])

def mean_positive_rank(y: np.ndarray, score: np.ndarray) -> float:
    """Mean rank of all true positives (lower = better)."""
    if y.sum() <= 0:
        return float('nan')
    order = np.argsort(-score)
    pos_ranks = np.where(y[order] == 1)[0] + 1
    return float(np.mean(pos_ranks))

def metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    return {
        'auc': safe_auc(y, score),
        'ap': safe_ap(y, score),
        'precision@5': precision_at_k(y, score, 5),
        'precision@10': precision_at_k(y, score, 10),
        'precision@20': precision_at_k(y, score, 20),
        'precision@50': precision_at_k(y, score, 50),
        'success@5': success_at_k(y, score, 5),
        'success@10': success_at_k(y, score, 10),
        'best_pos_rank': best_positive_rank(y, score),
        'mean_pos_rank': mean_positive_rank(y, score),
        'recall@50': recall_at_k(y, score, 50),
        'recall@100': recall_at_k(y, score, 100),
        'recall@500': recall_at_k(y, score, 500),
        'mcc@degree': mcc_at_topk(y, score, int(max(1, y.sum()))),
    }
