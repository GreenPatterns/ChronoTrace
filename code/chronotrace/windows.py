"""Deletion-window estimators for the window-informed attackers.

Every estimator returns an inclusive bin interval ``(b0, b1)`` on the dataset's
time-bin grid. They differ only in the information they read:

* ``oracle_window`` reads the deleted node's own incident edges. Only the
  benchmark has them, so it is an upper bound on timing knowledge.
* ``shift_window`` displaces a window by a fixed number of bins. Applied to the
  oracle window, it models side information about timing that is off by a
  known amount, which lets us trace accuracy as timing knowledge degrades.
* ``hub_median_window`` and ``id_cohort_window`` read only the released graph
  (and, for the cohort estimator, the deleted node's pseudonymous ID, which the
  threat model already gives the attacker).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .core import Edge, _bin_index
from .window_estimator import estimate_window

Window = Tuple[int, int]


def oracle_window(edges_u: Sequence[Edge], bins: np.ndarray, pad: int = 1) -> Window:
    """Bins spanned by the deleted node's incident edges, padded by ``pad`` bins each side."""
    n_bins = len(bins) - 1
    if not edges_u:
        return 0, n_bins - 1
    b0 = int(np.searchsorted(bins, min(e.time for e in edges_u), side='right') - 1)
    b1 = int(np.searchsorted(bins, max(e.time for e in edges_u), side='right') - 1)
    return max(0, b0 - pad), min(n_bins - 1, b1 + pad)


def shift_window(window: Window, delta: int, n_bins: int) -> Window:
    """Move a window by ``delta`` bins, keeping its width where the grid allows."""
    b0, b1 = window
    width = b1 - b0
    lo = b0 + delta
    lo = max(0, min(lo, n_bins - 1 - width))
    return lo, min(n_bins - 1, lo + width)


def window_iou(a: Window, b: Window) -> float:
    """Intersection over union of two inclusive bin intervals."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return float(inter / max(1, union))


def hub_median_window(edges_release: Sequence[Edge], bins: np.ndarray,
                      n_proxy: int = 50, half_width: int = 3) -> Window:
    """Median time of edges at the ``n_proxy`` highest-degree released nodes, +/- ``half_width`` bins."""
    return estimate_window(list(edges_release), bins, n_proxy_neighbors=n_proxy, half_width=half_width)


def activity_bins(edges_release: Iterable[Edge], bins: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """First and last activity bin of every node that appears in the release."""
    first: Dict[int, int] = {}
    last: Dict[int, int] = {}
    for e in edges_release:
        b = _bin_index(bins, e.time)
        for v in (e.src, e.dst):
            if v not in first or b < first[v]:
                first[v] = b
            if v not in last or b > last[v]:
                last[v] = b
    return {v: (first[v], last[v]) for v in first}


def id_cohort_window(user: int, activity: Dict[int, Tuple[int, int]], n_bins: int,
                     k: int = 25, pad: int = 1) -> Window:
    """Window of the ``k`` released nodes whose pseudonymous IDs are closest to ``user``.

    Platforms that assign IDs sequentially at sign-up make the ID a proxy for
    join time. The estimate is the cohort's median first bin and median last
    bin, padded like the oracle window. It reads nothing about the deleted
    node except its ID.
    """
    ids = np.fromiter((v for v in activity if v != user), dtype=np.int64)
    if len(ids) == 0:
        return 0, n_bins - 1
    nearest = ids[np.argsort(np.abs(ids - user), kind='stable')[:k]]
    firsts = np.array([activity[int(v)][0] for v in nearest])
    lasts = np.array([activity[int(v)][1] for v in nearest])
    b0 = int(np.floor(np.median(firsts)))
    b1 = int(np.ceil(np.median(lasts)))
    if b1 < b0:
        b0, b1 = b1, b0
    return max(0, b0 - pad), min(n_bins - 1, b1 + pad)


def window_for(mode: str, user: int, edges_full_u: List[Edge], edges_release: List[Edge],
               bins: np.ndarray, rng: np.random.Generator) -> Window:
    """Dispatch on the attacker's timing knowledge.

    ``mode`` is ``oracle``, ``shift<d>`` (oracle moved by d bins in a random
    direction), ``hub`` or ``cohort``. ``edges_full_u`` are the deleted node's
    edges (read only by the oracle-based modes); ``edges_release`` is what the
    attacker sees.
    """
    n_bins = len(bins) - 1
    if mode == 'oracle':
        return oracle_window(edges_full_u, bins)
    if mode.startswith('shift'):
        delta = int(mode[len('shift'):])
        sign = 1 if rng.random() < 0.5 else -1
        return shift_window(oracle_window(edges_full_u, bins), sign * delta, n_bins)
    if mode == 'hub':
        return hub_median_window(edges_release, bins)
    if mode == 'cohort':
        return id_cohort_window(user, activity_bins(edges_release, bins), n_bins)
    raise ValueError(f'unknown window mode {mode!r}')
