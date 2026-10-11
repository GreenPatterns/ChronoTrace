#!/usr/bin/env python3
"""Aggregate the revision runs into paper/results/revision/summary.json and REPORT.md.

Every statistic is computed from the per-target unit files written by
run_revision.py. Intervals are percentile bootstrap 95% CIs over targets
(10,000 resamples, seed 42); paired differences pair the same target; p-values
are two-sided sign-flip permutation tests on per-target differences, adjusted
with Benjamini-Hochberg within each family.
"""
from __future__ import annotations

import argparse
import json
import zlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chronotrace.features import FEATURE_GROUPS, FEATURE_NAMES
from chronotrace.metrics_ext import pool_size_metrics

ROOT = Path(__file__).resolve().parents[2] / 'paper' / 'results' / 'revision'
N_BOOT = 10_000
GROUPS = list(FEATURE_GROUPS)
MODES = ['oracle', 'shift1', 'shift2', 'shift4', 'shift8', 'hub', 'cohort']
HEURISTICS = ['degree_prior', 'common_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment',
              'temporal_burst', 'temporal_shift', 'signed_balance', 'signed_motif', 'unsupervised_chronotrace']
METRICS = ['auc', 'ap', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'success@5', 'success@10',
           'recall@50', 'recall@100',
           'recall@500', 'best_pos_rank', 'mean_pos_rank', 'deg_matched_auc', 'deg_cond_auc', 'swr@100', 'swr@500',
           'band_0-3', 'band_4-20', 'band_21-inf']

Key = Tuple[int, int]


# ---------------------------------------------------------------- statistics

def boot_ci(vals: Iterable[float], seed: int = 42) -> Dict[str, float]:
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1) if len(v) > 1 else v
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            'lo': float(np.percentile(means, 2.5)), 'hi': float(np.percentile(means, 97.5)), 'n': int(len(v))}


def signflip_p(d: np.ndarray, seed: int = 42) -> float:
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(N_BOOT, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (N_BOOT + 1))


def paired(a: Dict[Key, float], b: Dict[Key, float]) -> Dict[str, float]:
    keys = sorted(k for k in a if k in b and a[k] == a[k] and b[k] == b[k])
    d = np.array([a[k] - b[k] for k in keys])
    out = boot_ci(d)
    out['p'] = signflip_p(d)
    return out


def bh(pvals: Dict[str, float]) -> Dict[str, float]:
    names = list(pvals)
    p = np.array([pvals[n] for n in names])
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, 1))):
        running = min(running, p[idx] * len(p) / rank)
        adj[idx] = running
    return {n: float(a) for n, a in zip(names, adj)}


# ---------------------------------------------------------------- loading

SUFFIX = '-edgelist'  # candidate universe of the analysed runs; '' for directory runs


def load_units(ds: str, graph: str = 'real', ts: str = 'main') -> List[dict]:
    d = ROOT / ds / graph / (ts + SUFFIX) / 'units'
    return [json.loads(p.read_text()) for p in sorted(d.glob('*.json'))] if d.exists() else []


def per_task(units: List[dict], key: str, metric: str = 'auc') -> Dict[Key, float]:
    out = {}
    for r in units:
        res = r['results'].get(key)
        if res is not None and metric in res and res[metric] is not None:
            out[(r['seed'], r['user'])] = float(res[metric])
    return out


def graph_null_per_task(ds: str, family: str, key: str, metric: str = 'auc') -> Dict[Key, float]:
    """Per-target mean over every available realization of a rewired reference graph."""
    acc: Dict[Key, list] = {}
    for g in sorted(p.name for p in (ROOT / ds).glob(f'{family}[0-9]*')):
        for k, v in per_task(load_units(ds, g), key, metric).items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------- per-dataset summary

def summarize_dataset(ds: str) -> dict:
    units = load_units(ds)
    if not units:
        return {}
    s: dict = {'n_tasks': len(units)}
    npos = np.array([r['n_pos'] for r in units])
    s['n_pos'] = {'mean': float(npos.mean()), 'median': float(np.median(npos)), 'min': int(npos.min()),
                  'max': int(npos.max())}
    s['n_candidates'] = sorted({r['n_candidates'] for r in units})
    s['incident_degree'] = boot_ci([r['incident_degree'] for r in units])

    keys = ['random'] + [f'{p}__oracle__{g}' for p in ('release', 'lodo') for g in GROUPS]
    keys += [f'release__{m}__{g}' for m in MODES[1:] for g in ('temporal', 'motif', 'chronotrace')]
    keys += [f'heur__{h}' for h in HEURISTICS]
    s['methods'] = {k: {m: boot_ci(per_task(units, k, m).values()) for m in METRICS}
                    for k in keys if any(k in r['results'] for r in units)}

    # Protocol check: release-only minus the original-graph (LODO) protocol, per group.
    s['protocol_release_minus_lodo'] = {
        g: {m: paired(per_task(units, f'release__oracle__{g}', m), per_task(units, f'lodo__oracle__{g}', m))
            for m in ('auc', 'ap', 'precision@5', 'recall@100')}
        for g in GROUPS}

    # Gains over the degree attacker (release protocol), BH within dataset.
    deg = per_task(units, 'release__oracle__degree')
    vs = {g: paired(per_task(units, f'release__oracle__{g}'), deg) for g in GROUPS if g != 'degree'}
    for est in ('cohort', 'hub'):
        vs[f'chronotrace_{est}'] = paired(per_task(units, f'release__{est}__chronotrace'), deg)
    vs['chronotrace_vs_full_nonoracle'] = paired(per_task(units, 'release__oracle__chronotrace'),
                                                 per_task(units, 'release__oracle__full_nonoracle'))
    adj = bh({k: v['p'] for k, v in vs.items()})
    for k in vs:
        vs[k]['p_bh'] = adj[k]
    s['gains'] = vs

    # Reference-model ladder.
    lad: dict = {}
    lab_u = {(r['seed'], r['user']): r['label_null']['uniform_mean'] for r in units}
    lab_d = {(r['seed'], r['user']): r['label_null']['degprop_mean'] for r in units}
    lad['label_uniform'] = boot_ci(lab_u.values())
    lad['label_degprop'] = boot_ci(lab_d.values())
    lad['degree_minus_label_degprop'] = paired(deg, lab_d)
    for fam in ('config', 'sigtemp'):
        fam_out = {}
        for g in ('degree', 'full_nonoracle', 'temporal', 'chronotrace'):
            nul = graph_null_per_task(ds, fam, f'release__oracle__{g}')
            if not nul:
                continue
            real = per_task(units, f'release__oracle__{g}')
            fam_out[g] = {'null_auc': boot_ci(nul.values()), 'real_minus_null': paired(real, nul),
                          'null_deg_matched': boot_ci(graph_null_per_task(ds, fam, f'release__oracle__{g}',
                                                                          'deg_matched_auc').values())}
        if fam_out:
            fam_out['realizations'] = len(list((ROOT / ds).glob(f'{fam}[0-9]*')))
            lad[fam] = fam_out
    s['ladder'] = lad

    # Degree controls: excess over the degree-conditioned permutation expectation.
    s['degree_controls'] = {}
    for k in [f'release__oracle__{g}' for g in GROUPS] + [f'release__{m}__chronotrace' for m in MODES[1:]]:
        auc = per_task(units, k)
        cond = per_task(units, k, 'deg_cond_auc')
        s['degree_controls'][k] = {'deg_matched_auc': boot_ci(per_task(units, k, 'deg_matched_auc').values()),
                                   'excess_over_deg_cond': paired(auc, cond)}

    # Window knowledge.
    oracle_ct = per_task(units, 'release__oracle__chronotrace')
    nonoracle = per_task(units, 'release__oracle__full_nonoracle')
    sweep = {}
    for m in MODES:
        ct = per_task(units, f'release__{m}__chronotrace')
        iou = [r['windows'][m]['iou'] for r in units if m in r.get('windows', {})]
        sweep[m] = {'auc': boot_ci(ct.values()), 'iou': boot_ci(iou),
                    'deg_matched_auc': boot_ci(per_task(units, f'release__{m}__chronotrace',
                                                        'deg_matched_auc').values()),
                    'minus_oracle': paired(ct, oracle_ct), 'minus_full_nonoracle': paired(ct, nonoracle),
                    'minus_degree': paired(ct, deg)}
    s['window_sweep'] = sweep

    # Logistic coefficients (standardized features), release protocol, oracle window.
    coefs = {}
    for g in ('full_nonoracle', 'chronotrace'):
        rows = [r['results'][f'release__oracle__{g}']['coef'] for r in units
                if r['results'][f'release__oracle__{g}']['coef'] is not None]
        arr = np.array(rows)
        coefs[g] = {FEATURE_NAMES[c]: {'mean': float(arr[:, i].mean()), 'std': float(arr[:, i].std(ddof=1)),
                                       'share_positive': float((arr[:, i] > 0).mean())}
                    for i, c in enumerate(FEATURE_GROUPS[g])}
    s['coefficients'] = coefs

    # Stratification by deleted-neighborhood size |N_u| (distinct neighbors).
    q = np.quantile(npos, [1 / 3, 2 / 3])
    strata = {}
    for name, lo, hi in (('small', -1, q[0]), ('medium', q[0], q[1]), ('large', q[1], np.inf)):
        sub = [r for r in units if lo < r['n_pos'] <= hi]
        strata[name] = {'n_tasks': len(sub), 'n_pos_range': [int(min(r['n_pos'] for r in sub)),
                                                             int(max(r['n_pos'] for r in sub))] if sub else [],
                        'auc': {g: boot_ci(per_task(sub, f'release__oracle__{g}').values()) for g in GROUPS}}
    s['strata_by_neighbors'] = strata
    return s


def summarize_low(ds: str) -> dict:
    units = load_units(ds, 'real', 'low')
    if not units:
        return {}
    keys = [f'{p}__oracle__{g}' for p in ('release', 'lodo') for g in GROUPS] + ['release__cohort__chronotrace',
                                                                                    'random']
    out = {'n_tasks': len(units), 'n_pos': boot_ci([r['n_pos'] for r in units]),
           'n_pos_hist': {int(k): int(v) for k, v in zip(*np.unique([r['n_pos'] for r in units],
                                                                     return_counts=True))}}
    out['methods'] = {k: {m: boot_ci(per_task(units, k, m).values())
                          for m in ('auc', 'success@10', 'success@5', 'best_pos_rank', 'mean_pos_rank',
                                    'deg_matched_auc')}
                      for k in keys if any(k in r['results'] for r in units)}
    out['chronotrace_minus_degree'] = paired(per_task(units, 'release__oracle__chronotrace'),
                                             per_task(units, 'release__oracle__degree'))
    return out


def summarize_pool_sizes(ds: str, pools=(2000, 5000, 10000, 20000, 50000)) -> dict:
    """Operational metrics as the candidate pool shrinks, from saved full-universe scores."""
    d = ROOT / ds / 'real' / ('main' + SUFFIX) / 'units'
    files = sorted(d.glob('*.npz')) if d.exists() else []
    if not files:
        return {}
    keys = ['release__oracle__degree', 'release__oracle__full_nonoracle', 'release__oracle__chronotrace']
    acc: Dict[str, Dict[str, Dict[str, list]]] = {}
    for f in files:
        z = np.load(f)
        y = z['y'].astype(int)
        for k in keys:
            if f'S__{k}' not in z.files:
                continue
            sc = z[f'S__{k}'].astype(float)
            for pool in list(pools) + ['full']:
                size = len(y) if pool == 'full' else int(pool)
                if size > len(y):
                    continue
                rng = np.random.default_rng([zlib.crc32(f.name.encode()), size])
                m = pool_size_metrics(y, sc, size, rng, n_draws=10)
                for met in ('auc', 'ap', 'precision@5', 'precision@10', 'success@5', 'success@10', 'recall@100',
                            'best_pos_rank', 'mean_pos_rank'):
                    acc.setdefault(k, {}).setdefault(str(pool), {}).setdefault(met, []).append(m[met])
    return {k: {pool: {met: boot_ci(v) for met, v in mets.items()} for pool, mets in pools_.items()}
            for k, pools_ in acc.items()}


# ---------------------------------------------------------------- report

def fmt(c: dict, nd: int = 3) -> str:
    if not c or c.get('n', 0) == 0:
        return 'n/a'
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


def report(summary: dict) -> str:
    lines = ['# ChronoTrace revision results (auto-generated by analyze_revision.py)', '']
    for ds, s in summary['datasets'].items():
        if not s:
            continue
        lines += [f'## {ds} ({s["n_tasks"]} targets; |N_u| mean {s["n_pos"]["mean"]:.1f}, median '
                  f'{s["n_pos"]["median"]:.0f}, range {s["n_pos"]["min"]}-{s["n_pos"]["max"]}; candidates '
                  f'{s["n_candidates"]})', '']
        lines += ['| method | AUC | AP | P@5 | S@10 | R@100 | deg-matched AUC |', '|---|---|---|---|---|---|---|']
        for k, v in s['methods'].items():
            lines.append(f"| {k} | {fmt(v['auc'])} | {fmt(v['ap'])} | {fmt(v['precision@5'])} | "
                         f"{fmt(v['success@10'])} | {fmt(v['recall@100'])} | {fmt(v['deg_matched_auc'])} |")
        lines += ['', '**Release-only minus submitted LODO (AUC):** ' + '; '.join(
            f"{g} {fmt(v['auc'], 4)} p={v['auc']['p']:.3f}" for g, v in s['protocol_release_minus_lodo'].items()), '']
        lines += ['**Gains (release, AUC):** ' + '; '.join(
            f"{k} {fmt(v, 4)} p_BH={v['p_bh']:.4f}" for k, v in s['gains'].items()), '']
        lad = s['ladder']
        lines += [f"**Ladder:** label uniform {fmt(lad['label_uniform'])}; label degprop {fmt(lad['label_degprop'])};"
                  f" degree - degprop {fmt(lad['degree_minus_label_degprop'], 4)} p={lad['degree_minus_label_degprop']['p']:.4f}"]
        for fam in ('config', 'sigtemp'):
            if fam in lad:
                lines.append(f"- {fam} ({lad[fam]['realizations']} realizations): " + '; '.join(
                    f"{g}: null {fmt(v['null_auc'])}, real-null {fmt(v['real_minus_null'], 4)} "
                    f"(p={v['real_minus_null']['p']:.4f}), null deg-matched {fmt(v['null_deg_matched'])}"
                    for g, v in lad[fam].items() if g != 'realizations'))
        lines += ['', '**Degree controls:**']
        for k, v in s['degree_controls'].items():
            lines.append(f"- {k}: deg-matched {fmt(v['deg_matched_auc'])}; AUC - E[AUC | degree histogram] "
                         f"{fmt(v['excess_over_deg_cond'], 4)} p={v['excess_over_deg_cond']['p']:.4f}")
        lines += ['', '**Window sweep (chronotrace, release):**']
        for m, v in s['window_sweep'].items():
            lines.append(f"- {m}: AUC {fmt(v['auc'])}, IoU {fmt(v['iou'])}, deg-matched {fmt(v['deg_matched_auc'])},"
                         f" minus full-nonoracle {fmt(v['minus_full_nonoracle'], 4)} (p={v['minus_full_nonoracle']['p']:.4f}),"
                         f" minus degree {fmt(v['minus_degree'], 4)}")
        lines += ['', '**Coefficients (standardized; mean, share > 0):**']
        for g, c in s['coefficients'].items():
            lines.append(f"- {g}: " + ', '.join(f"{f} {v['mean']:+.2f} ({v['share_positive']:.2f})" for f, v in c.items()))
        lines += ['', '**Strata by |N_u|:** ' + '; '.join(
            f"{n} (n={v['n_tasks']}, |N_u| {v['n_pos_range']}): degree {fmt(v['auc']['degree'])}, chrono "
            f"{fmt(v['auc']['chronotrace'])}" for n, v in s['strata_by_neighbors'].items()), '']
    for ds, s in summary.get('low', {}).items():
        if not s:
            continue
        lines += [f'## Low-degree targets: {ds} ({s["n_tasks"]} targets, |N_u| hist {s["n_pos_hist"]})', '']
        for k, v in s['methods'].items():
            lines.append(f"- {k}: AUC {fmt(v['auc'])}, S@10 {fmt(v['success@10'])}, best rank "
                         f"{fmt(v['best_pos_rank'], 0)}, deg-matched {fmt(v['deg_matched_auc'])}")
        lines.append(f"- chronotrace - degree: {fmt(s['chronotrace_minus_degree'], 4)}")
        lines.append('')
    for ds, s in summary.get('pool_sizes', {}).items():
        if not s:
            continue
        lines += [f'## Pool-size sensitivity: {ds}', '']
        for k, pools in s.items():
            lines.append(f'- {k}: ' + '; '.join(
                f"{p}: AUC {v['auc']['mean']:.3f}, AP {v['ap']['mean']:.3f}, P@5 {v['precision@5']['mean']:.3f}, "
                f"S@10 {v['success@10']['mean']:.3f}, best rank {v['best_pos_rank']['mean']:.0f}"
                for p, v in pools.items()))
        lines.append('')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='bitcoinotc,bitcoinalpha,epinions_dated,epinions')
    ap.add_argument('--pool-datasets', default='epinions_dated,epinions')
    ap.add_argument('--universe', choices=['edgelist', 'directory'], default='edgelist')
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = '' if args.universe == 'directory' else '-edgelist'
    dsets = args.datasets.split(',')
    summary = {'datasets': {ds: summarize_dataset(ds) for ds in dsets},
               'low': {ds: summarize_low(ds) for ds in dsets},
               'pool_sizes': {ds: summarize_pool_sizes(ds) for ds in args.pool_datasets.split(',')}}
    tag = '' if args.universe == 'edgelist' else '_directory'
    (ROOT / f'summary{tag}.json').write_text(json.dumps(summary, indent=1))
    (ROOT / f'REPORT{tag}.md').write_text(report(summary))
    print(f'wrote {ROOT / f"summary{tag}.json"} and {ROOT / f"REPORT{tag}.md"}')


if __name__ == '__main__':
    main()
