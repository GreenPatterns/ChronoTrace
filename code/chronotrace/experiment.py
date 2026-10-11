from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .attack import aggregate, evaluate_lopo
from .core import DATASETS, choose_users, load_edges, nodes_of, time_bins
from .features import build_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # code/chronotrace/ -> ChronoTrace/
RESULT_ROOT = PROJECT_ROOT / 'paper' / 'results' / 'unified'


def run_dataset(dataset: str, seed: int, n_users: int, min_degree: int, max_degree: int | None, n_bins: int, defenses: List[Dict]) -> Dict:
    edges = load_edges(DATASETS[dataset])
    bins = time_bins(edges, n_bins)
    users = choose_users(edges, n_users=n_users, seed=seed, min_degree=min_degree, max_degree=max_degree)
    ds_out = {
        'path': str(DATASETS[dataset]),
        'n_edges': len(edges),
        'n_nodes': len(nodes_of(edges)),
        'time_min': int(min(e.time for e in edges)),
        'time_max': int(max(e.time for e in edges)),
        'users': users,
        'defenses': {},
    }
    for defense in defenses:
        name = defense['name']
        print(f'[{dataset}] defense={name} build_tasks users={len(users)}')
        tasks = [
            build_task(
                edges, u, bins,
                defense=defense.get('mode', 'none'),
                defense_strength=float(defense.get('strength', 0.0)),
                defense_hops=int(defense.get('hops', 2)),
            )
            for u in users
        ]
        rows = evaluate_lopo(tasks, seed=seed)
        ds_out['defenses'][name] = {
            'config': defense,
            'rows': rows,
            'aggregate': aggregate(rows),
        }
        chrono = ds_out['defenses'][name]['aggregate']['chronotrace']['auc']['mean']
        util = ds_out['defenses'][name]['aggregate']['utility_proxy']['edge_retention_mean']
        print(f'[{dataset}] defense={name} chronotrace_auc={chrono:.3f} edge_retention={util:.3f}')
    return ds_out


def default_defenses() -> List[Dict]:
    return [
        {'name': 'deletion_only', 'mode': 'none', 'strength': 0.0, 'hops': 0},
        {'name': 'chronoscrub_temporal', 'mode': 'temporal_smooth', 'strength': 1.0, 'hops': 2},
        {'name': 'chronoscrub_joint_s025', 'mode': 'joint', 'strength': 0.25, 'hops': 2},
        {'name': 'chronoscrub_joint_s050', 'mode': 'joint', 'strength': 0.50, 'hops': 2},
        {'name': 'chronoscrub_joint_s075', 'mode': 'joint', 'strength': 0.75, 'hops': 2},
    ]


def run_experiment(dataset: str = 'both', seeds: List[int] | None = None, n_users: int = 12, min_degree: int = 4, max_degree: int | None = 128, n_bins: int = 32, defenses: List[Dict] | None = None) -> Path:
    t0 = time.time()
    seeds = seeds or [42, 7, 13]
    defenses = defenses or default_defenses()
    datasets = ['bitcoinotc', 'bitcoinalpha'] if dataset == 'both' else [dataset]
    result = {
        'project': 'ChronoTrace',
        'protocol': 'leave-one-deleted-user-out full candidate universe',
        'config': {
            'dataset': dataset,
            'seeds': seeds,
            'n_users': n_users,
            'min_degree': min_degree,
            'max_degree': max_degree,
            'time_bins': n_bins,
            'defenses': defenses,
        },
        'runs': [],
    }
    for seed in seeds:
        run = {'seed': seed, 'datasets': {}}
        for ds in datasets:
            run['datasets'][ds] = run_dataset(ds, seed, n_users, min_degree, max_degree, n_bins, defenses)
        result['runs'].append(run)
    result['elapsed_sec'] = time.time() - t0
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESULT_ROOT / f'chronotrace_unified_{time.strftime("%Y%m%d_%H%M%S")}.json'
    out.write_text(json.dumps(result, indent=2))
    return out
