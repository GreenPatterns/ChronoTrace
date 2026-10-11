"""Estimated-window attacker for ChronoTrace.

Bridges the gap between the nonoracle attacker (which ignores any
information about *when* the deleted node was active) and the oracle
window attacker (which knows the exact earliest/latest timestamps of
the deleted incident edges).

The estimator operates *only* on the released graph G^{rel}_u and returns
a window [b_hat0, b_hat1] (in time-bin units) that the supervised attacker
then uses *as if* it were the oracle window. This produces a third,
realistic threat-model setting between nonoracle and oracle.

Estimator (deterministic, released-graph-only)
----------------------------------------------
Idea. Under preferential attachment, former neighbors of a deleted node
tend to be high-degree retained nodes. The median timestamp of edges
incident to those high-degree nodes is therefore a noisy estimator of
when the deleted node was active.

Algorithm.
  1. Sort retained nodes by degree (descending) and take the top-K.
  2. Collect every retained edge incident to any of those proxy nodes.
  3. Take the median of those timestamps as the window center.
  4. Return [center - half_width, center + half_width] in bin indices.

This estimator does *not* use any privileged knowledge: it has access
only to G^{rel}_u (which is what the realistic attacker sees). Its
quality is dataset-dependent; we report Intersection-over-Union (IoU)
with the true window as a calibration metric.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .core import Edge, build_adj


def estimate_window(
    edges_released,
    bins: np.ndarray,
    n_proxy_neighbors: int = 50,
    half_width: int = 3,
) -> Tuple[int, int]:
    """Estimate the deletion window [b_hat0, b_hat1] from the released graph.

    Parameters
    ----------
    edges_released : list of Edge
        Edges of G^{rel}_u (i.e., after the target's incident edges have
        been removed). The estimator uses only this.
    bins : np.ndarray
        Time bin edges (length n_bins+1).
    n_proxy_neighbors : int
        Number of top-degree retained nodes used as proxy former neighbors.
    half_width : int
        Half-width of the returned window in bin indices. The window is
        centered on the median timestamp of proxy-incident edges and
        extends ``half_width`` bins on each side (clipped to graph extent).

    Returns
    -------
    (b_hat0, b_hat1) : tuple of int
        Estimated window in bin indices (inclusive, clipped to
        ``[0, n_bins-1]``).
    """
    n_bins = len(bins) - 1
    if n_bins <= 0 or not edges_released:
        return 0, max(0, n_bins - 1)

    adj = build_adj(edges_released)
    if not adj:
        return 0, n_bins - 1
    nodes_by_deg = sorted(adj.items(), key=lambda kv: -len(kv[1]))
    proxy = {v for v, _ in nodes_by_deg[:max(5, n_proxy_neighbors)]}

    timestamps = [e.time for e in edges_released if e.src in proxy or e.dst in proxy]
    if len(timestamps) < 4:
        timestamps = [e.time for e in edges_released]

    arr = np.asarray(timestamps, dtype=float)
    center = float(np.median(arr))
    c_bin = int(np.searchsorted(bins, center, side='right') - 1)
    b_hat0 = max(0, c_bin - half_width)
    b_hat1 = min(n_bins - 1, c_bin + half_width)
    return b_hat0, b_hat1


def true_window(edges_u, bins: np.ndarray, pad_bins: int = 1) -> Tuple[int, int]:
    """Oracle deletion window from incident edges of u."""
    n_bins = len(bins) - 1
    if not edges_u:
        return 0, max(0, n_bins - 1)
    t0 = min(e.time for e in edges_u)
    t1 = max(e.time for e in edges_u)
    b0 = int(np.searchsorted(bins, t0, side='right') - 1)
    b1 = int(np.searchsorted(bins, t1, side='right') - 1)
    return max(0, b0 - pad_bins), min(n_bins - 1, b1 + pad_bins)


def window_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Intersection-over-union for two bin intervals (inclusive)."""
    inter_lo = max(a[0], b[0])
    inter_hi = min(a[1], b[1])
    inter = max(0, inter_hi - inter_lo + 1)
    union_lo = min(a[0], b[0])
    union_hi = max(a[1], b[1])
    union = max(1, union_hi - union_lo + 1)
    return float(inter / union)
