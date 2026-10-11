"""Training protocols for the supervised attacker.

``lodo`` (the submitted protocol): to score target u_i, the attacker trains on the
deletion tasks of the other targets u_j, each built on the original graph minus
u_j. Those training graphs still contain u_i and its edges, and a training label
can state that u_i was adjacent to u_j.

``release`` (added in the revision): to score target u_i, the attacker trains
only on deletions that it simulates on the release it was given, G - u_i. For
each other target u_j it deletes u_j from G - u_i and labels u_j's neighbors in
G - u_i. No edge of u_i, no neighbor of u_i and no label involving u_i reaches
training, and the attacker needs no historical releases or ground truth.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import Edge, remove_user
from .features import Task, build_task


def fit_score(train_tasks: Sequence[Task], test_task: Task, cols: Sequence[int],
              seed: int) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Logistic attacker on pooled training tasks; returns (test scores, standardized coefs, intercept).

    The sampling matches ``attack.fit_score`` exactly: every positive row plus
    max(1000, 20 x positives) pooled negatives drawn with seed + test user.
    """
    cols = list(cols)
    Xtr = np.vstack([t.X[:, cols] for t in train_tasks])
    ytr = np.concatenate([t.y for t in train_tasks])
    Xte = test_task.X[:, cols]
    if len(np.unique(ytr)) < 2:
        return Xte.sum(axis=1), None, float('nan')
    pos = np.flatnonzero(ytr == 1)
    neg = np.flatnonzero(ytr == 0)
    rng = np.random.default_rng(seed + test_task.user)
    keep_neg = rng.choice(neg, size=min(len(neg), max(1000, len(pos) * 20)), replace=False)
    keep = np.concatenate([pos, keep_neg])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed))
    clf.fit(Xtr[keep], ytr[keep])
    lr = clf[-1]
    return clf.predict_proba(Xte)[:, 1], lr.coef_[0].copy(), float(lr.intercept_[0])


def release_of(graph: List[Edge], test_user: int) -> List[Edge]:
    """The undefended release after deleting ``test_user``: every incident edge removed."""
    return remove_user(graph, test_user)


def shadow_tasks(release: List[Edge], test_user: int, train_users: Sequence[int], bins: np.ndarray,
                 universe: Sequence[int], defense: str = 'none', defense_strength: float = 0.0,
                 defense_hops: int = 2, candidates_from_release: bool = False) -> List[Task]:
    """Simulated deletions of ``train_users`` performed on the attacker's ``release``.

    ``release`` is what the platform published after deleting ``test_user``
    (defended or not). Every shadow task's candidate universe is ``universe``
    minus the test user and the shadow target, so the test user is never a
    training candidate and none of its edges is in any training graph.
    """
    if any(e.src == test_user or e.dst == test_user for e in release):
        raise ValueError('release still contains edges of the deleted test user')
    uni = [v for v in universe if v != test_user]
    return [build_task(release, u, bins, defense=defense, defense_strength=defense_strength,
                       defense_hops=defense_hops, universe=uni, compute_utility=False,
                       keep_hist=True, candidates_from_release=candidates_from_release)
            for u in train_users if u != test_user]
