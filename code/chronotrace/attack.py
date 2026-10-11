from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import metrics
from .features import FEATURE_GROUPS, Task


def fit_score(train_tasks: List[Task], test_task: Task, cols: List[int], seed: int) -> np.ndarray:
    Xtr = np.vstack([t.X[:, cols] for t in train_tasks])
    ytr = np.concatenate([t.y for t in train_tasks])
    Xte = test_task.X[:, cols]
    if len(np.unique(ytr)) < 2:
        return Xte.sum(axis=1)
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    rng = np.random.default_rng(seed + test_task.user)
    keep_neg = rng.choice(neg, size=min(len(neg), max(1000, len(pos) * 20)), replace=False)
    keep = np.concatenate([pos, keep_neg])
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed))
    clf.fit(Xtr[keep], ytr[keep])
    return clf.predict_proba(Xte)[:, 1]


def evaluate_lopo(tasks: List[Task], seed: int, feature_groups: Optional[Dict[str, List[int]]] = None) -> List[Dict]:
    """Evaluate leave-one-deleted-user-out with optional custom feature groups.
    
    Args:
        tasks: List of Task objects
        seed: Random seed
        feature_groups: Optional custom feature groups (default: FEATURE_GROUPS)
    
    Returns:
        List of per-user results
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS
    
    rows = []
    for i, test in enumerate(tasks):
        train = [t for j, t in enumerate(tasks) if j != i]
        methods: Dict[str, Dict] = {}
        rng = np.random.default_rng(seed + test.user)
        methods['random'] = metrics(test.y, rng.random(len(test.y)))
        for name, cols in feature_groups.items():
            # Skip feature groups that exceed the feature dimension
            if max(cols) >= test.X.shape[1]:
                continue
            score = fit_score(train, test, cols, seed)
            methods[name] = metrics(test.y, score)
        methods['profile'] = test.profile
        methods['n_candidates'] = int(len(test.y))
        methods['positives'] = int(test.y.sum())
        methods['utility_edges_before'] = int(test.utility_edges_before)
        methods['utility_edges_after'] = int(test.utility_edges_after)
        methods['utility_edge_retention'] = float(test.utility_edges_after / max(1, test.utility_edges_before))
        methods['utility_sign_auc'] = float(test.utility_sign_auc)
        methods['utility_sign_ap'] = float(test.utility_sign_ap)
        methods['utility_train_edges'] = int(test.utility_train_edges)
        methods['utility_test_edges'] = int(test.utility_test_edges)
        rows.append({'user': test.user, 'methods': methods})
    return rows


def aggregate(rows: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    methods = ['random', 'degree', 'topology', 'full_nonoracle', 'temporal', 'motif', 'chronotrace']
    mets = ['auc', 'ap', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'success@5', 'success@10', 'best_pos_rank', 'mean_pos_rank', 'recall@50', 'recall@100', 'recall@500', 'mcc@degree']
    for m in methods:
        out[m] = {}
        for met in mets:
            vals = [float(r['methods'][m][met]) for r in rows if r['methods'][m][met] == r['methods'][m][met]]
            out[m][met] = {
                'mean': float(np.mean(vals)) if vals else float('nan'),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'n': len(vals),
            }
    ret = [float(r['methods']['utility_edge_retention']) for r in rows]
    sauc = [float(r['methods']['utility_sign_auc']) for r in rows if r['methods']['utility_sign_auc'] == r['methods']['utility_sign_auc']]
    sap = [float(r['methods']['utility_sign_ap']) for r in rows if r['methods']['utility_sign_ap'] == r['methods']['utility_sign_ap']]
    out['utility_proxy'] = {
        'edge_retention_mean': float(np.mean(ret)) if ret else float('nan'),
        'edge_retention_std': float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0,
        'sign_auc_mean': float(np.mean(sauc)) if sauc else float('nan'),
        'sign_auc_std': float(np.std(sauc, ddof=1)) if len(sauc) > 1 else 0.0,
        'sign_ap_mean': float(np.mean(sap)) if sap else float('nan'),
        'sign_ap_std': float(np.std(sap, ddof=1)) if len(sap) > 1 else 0.0,
    }
    return out
