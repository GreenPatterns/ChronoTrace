#!/usr/bin/env python3
"""Every figure of the revised paper, drawn from saved results (no hand-entered values).

Inputs: paper/results/revision/summary.json (edge-list release), per-target unit files,
defense files. Outputs: paper/images/fig_*.pdf. One visual language throughout:
blues = release-only feature sets, greens = ChronoTrace with a window estimated from the
release, oranges = feature sets that read the exact window.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'paper' / 'results' / 'revision'
IMG = ROOT / 'paper' / 'images'
DS = {'otc': 'bitcoinotc', 'alpha': 'bitcoinalpha', 'epi': 'epinions_dated'}
NAMES = {'otc': 'BitcoinOTC', 'alpha': 'BitcoinAlpha', 'epi': 'Epinions'}
FULL, COL = 7.0, 3.35

# key, legend label, color, hatch
ADV = [
    ('random', 'Random', '#BDBDBD', ''),
    ('release__oracle__degree', 'Degree', '#08519C', ''),
    ('release__oracle__topology', 'Topology', '#3182BD', ''),
    ('release__oracle__full_nonoracle', 'Full-R', '#9ECAE1', ''),
    ('release__hub__chronotrace', 'ChronoTrace, hub window', '#C7E9C0', '////'),
    ('release__cohort__chronotrace', 'ChronoTrace, ID-cohort window', '#41AB5D', '////'),
    ('release__oracle__temporal', 'Temporal', '#FDD0A2', ''),
    ('release__oracle__motif', 'Sgn-Struct', '#FD8D3C', ''),
    ('release__oracle__chronotrace', 'ChronoTrace, exact window', '#D94801', ''),
]
COLOR = {k: c for k, _, c, _ in ADV}
LABEL = {k: l for k, l, _, _ in ADV}

plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'font.family': 'serif', 'pdf.fonttype': 42,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'grid.linewidth': 0.4})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def units(ds: str, ts: str = 'main-edgelist') -> List[dict]:
    d = RES / ds / 'real' / ts / 'units'
    return [json.loads(p.read_text()) for p in sorted(d.glob('*.json'))] if d.exists() else []


def err(c):
    return [[c['mean'] - c['lo']], [c['hi'] - c['mean']]]


def save(fig, name):
    fig.savefig(IMG / name, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- RQ1

def fig_attack_auc(S):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), sharey=True)
    for ax, (p, ds) in zip(axes, DS.items()):
        M = S['datasets'][ds]['methods']
        for i, (k, lab, col, hatch) in enumerate(ADV):
            c = M[k]['auc']
            ax.bar(i, c['mean'], color=col, edgecolor='black', linewidth=0.4, hatch=hatch, label=lab)
            ax.errorbar(i, c['mean'], yerr=err(c), color='black', lw=0.7, capsize=1.6)
        cfg = S['datasets'][ds]['ladder'].get('config', {}).get('degree')
        if cfg:
            ax.axhline(cfg['null_auc']['mean'], color='black', ls=(0, (4, 2)), lw=0.9,
                       label='Degree on degree reference')
        ax.axhline(0.5, color='0.5', ls=':', lw=0.8)
        ax.set_title(NAMES[p])
        ax.set_xticks([])
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis='y', alpha=0.5)
    axes[0].set_ylabel('AUC (95% CI)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.2), columnspacing=0.9,
               handlelength=1.5)
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_attack_auc.pdf')


def fig_paired_deltas(S):
    keys = ['release__oracle__topology', 'release__oracle__full_nonoracle', 'release__cohort__chronotrace',
            'release__oracle__temporal', 'release__oracle__motif', 'release__oracle__chronotrace']
    gain_key = {'release__oracle__topology': 'topology', 'release__oracle__full_nonoracle': 'full_nonoracle',
                'release__cohort__chronotrace': 'chronotrace_cohort', 'release__oracle__temporal': 'temporal',
                'release__oracle__motif': 'motif', 'release__oracle__chronotrace': 'chronotrace'}
    short = {'release__oracle__topology': 'Topology', 'release__oracle__full_nonoracle': 'Full-R',
             'release__cohort__chronotrace': 'ChronoTrace (ID cohort)', 'release__oracle__temporal': 'Temporal',
             'release__oracle__motif': 'Sgn-Struct', 'release__oracle__chronotrace': 'ChronoTrace (exact)'}
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.35), sharey=True)
    rng = np.random.default_rng(42)
    for ax, (p, ds) in zip(axes, DS.items()):
        uu = units(ds)
        deg = {(r['seed'], r['user']): r['results']['release__oracle__degree']['auc'] for r in uu}
        for i, k in enumerate(keys):
            d = [r['results'][k]['auc'] - deg[(r['seed'], r['user'])] for r in uu
                 if k in r['results'] and r['results'][k]['auc'] == r['results'][k]['auc']]
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(d)), d, s=4, color=COLOR[k], alpha=0.55, lw=0)
            g = S['datasets'][ds]['gains'][gain_key[k]]
            ax.errorbar(i, g['mean'], yerr=err(g), fmt='s', color='black', ms=3, capsize=2.2, lw=0.9)
        ax.axhline(0, color='black', ls=(0, (4, 2)), lw=0.7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([short[k] for k in keys], rotation=35, ha='right', rotation_mode='anchor')
        ax.set_title(NAMES[p])
        ax.grid(axis='y', alpha=0.5)
    axes[0].set_ylabel(r'$\Delta$AUC vs Degree')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_paired_deltas.pdf')


# ---------------------------------------------------------------- RQ2

def fig_ladder(S):
    steps = ['Uniform', 'Degree\nreference', 'Activity\nreference', 'Real\ngraph']
    series = [('degree', 'release__oracle__degree', 'o'), ('chronotrace', 'release__oracle__chronotrace', 's')]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.9), sharey=True)
    for ax, (p, ds) in zip(axes, DS.items()):
        d = S['datasets'][ds]
        lad = d['ladder']
        for g, key, mk in series:
            if 'config' not in lad or 'sigtemp' not in lad:
                continue
            vals = [lad['label_uniform'], lad['config'][g]['null_auc'], lad['sigtemp'][g]['null_auc'],
                    d['methods'][key]['auc']]
            y = [v['mean'] for v in vals]
            yerr = np.array([[v['mean'] - v['lo'] for v in vals], [v['hi'] - v['mean'] for v in vals]])
            ax.errorbar(range(4), y, yerr=yerr, color=COLOR[key], marker=mk, ms=3.5, lw=1.2, capsize=2,
                        label=LABEL[key] if g == 'degree' else 'ChronoTrace, exact window')
        ax.set_xticks(range(4))
        ax.set_xticklabels(steps)
        ax.set_title(NAMES[p])
        ax.set_ylim(0.45, 1.0)
        ax.grid(axis='y', alpha=0.5)
    axes[0].set_ylabel('AUC (95% CI)')
    axes[0].legend(frameon=False, loc='lower right')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_ladder.pdf')


# ---------------------------------------------------------------- RQ3

def fig_window(S):
    shifts = [0, 1, 2, 4, 8]
    modes = ['oracle', 'shift1', 'shift2', 'shift4', 'shift8']
    colors = {'otc': '#0072B2', 'alpha': '#D55E00', 'epi': '#009E73'}
    markers = {'otc': 'o', 'alpha': 's', 'epi': '^'}
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    x = np.arange(len(shifts))
    est_x = {'cohort': len(shifts) + 0.6, 'hub': len(shifts) + 1.6}
    for k, (p, ds) in enumerate(DS.items()):
        sw = S['datasets'][ds]['window_sweep']
        y = np.array([sw[m]['auc']['mean'] for m in modes])
        e = np.array([[sw[m]['auc']['mean'] - sw[m]['auc']['lo'] for m in modes],
                      [sw[m]['auc']['hi'] - sw[m]['auc']['mean'] for m in modes]])
        off = (k - 1) * 0.08
        ax.errorbar(x + off, y, yerr=e, color=colors[p], marker=markers[p], ms=3.5, lw=1.3, capsize=2, label=NAMES[p])
        deg = S['datasets'][ds]['methods']['release__oracle__degree']['auc']['mean']
        ax.axhline(deg, color=colors[p], ls=(0, (4, 2)), lw=0.9)
        for mode, xx in est_x.items():
            c = sw[mode]['auc']
            ax.errorbar(xx + off, c['mean'], yerr=err(c), color=colors[p], marker=markers[p], ms=3.5, capsize=2,
                        ls='none', mfc='white')
    ax.axvline(len(shifts) - 0.2, color='0.6', lw=0.5)
    ax.set_xticks(list(x) + list(est_x.values()))
    ax.set_xticklabels([str(v) for v in shifts] + ['ID\ncohort', 'hub'])
    ax.set_xlim(-0.4, len(shifts) + 2.1)
    ax.set_xlabel(r'window shift $\delta$ (bins)  $\mid$  estimated window')
    ax.set_ylabel('AUC of ChronoTrace')
    ax.grid(axis='y', alpha=0.5)
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout(pad=0.3)
    save(fig, 'fig_window.pdf')


# ---------------------------------------------------------------- RQ4

def fig_strata(S):
    keys = ['release__oracle__degree', 'release__oracle__full_nonoracle', 'release__oracle__chronotrace']
    groups = {'release__oracle__degree': 'degree', 'release__oracle__full_nonoracle': 'full_nonoracle',
              'release__oracle__chronotrace': 'chronotrace'}
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), sharey=True)
    w = 0.26
    for ax, (p, ds) in zip(axes, DS.items()):
        st = S['datasets'][ds]['strata_by_neighbors']
        low = S.get('low', {}).get(ds) or {}
        labels = []
        cols = []
        if low:
            labels.append(f"1\u20133 events\nn={low['methods']['release__oracle__degree']['auc']['n']}")
            cols.append({k: low['methods'][k]['auc'] for k in keys})
        for name in ('small', 'medium', 'large'):
            g = st[name]
            lo, hi = g['n_pos_range']
            labels.append(f"$|N_u|$ {lo}\u2013{hi}\nn={g['n_tasks']}")
            cols.append({k: g['auc'][groups[k]] for k in keys})
        for j, k in enumerate(keys):
            ys = [c[k]['mean'] for c in cols]
            es = np.array([[c[k]['mean'] - c[k]['lo'] for c in cols], [c[k]['hi'] - c[k]['mean'] for c in cols]])
            ax.bar(np.arange(len(cols)) + (j - 1) * w, ys, w, color=COLOR[k], edgecolor='black', lw=0.4,
                   label=LABEL[k])
            ax.errorbar(np.arange(len(cols)) + (j - 1) * w, ys, yerr=es, fmt='none', color='black', lw=0.6,
                        capsize=1.4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title(NAMES[p])
        ax.set_ylim(0.6, 1.0)
        ax.grid(axis='y', alpha=0.5)
    axes[0].set_ylabel('AUC (95% CI)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_strata.pdf')


# ---------------------------------------------------------------- RQ5

def fig_defense():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), sharey=True)
    any_data = False
    short = {'release__oracle__degree': 'Degree', 'release__oracle__chronotrace': 'ChronoTrace (exact)'}
    for ax, (p, ds) in zip(axes, DS.items()):
        recs = [json.loads(f.read_text()) for f in sorted((RES / ds / 'defense-edgelist').glob('*.json'))]
        if len(recs) < 60:
            ax.text(0.5, 0.5, 'pending', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(NAMES[p])
            continue
        any_data = True
        for g, key in (('degree', 'release__oracle__degree'), ('chronotrace', 'release__oracle__chronotrace')):
            rr = [('none', recs[0]['chronoscrub']['deletion_only'])] if False else []
            xs, ys = [], []
            base_auc = np.mean([r['chronoscrub']['deletion_only']['attack'][g]['auc'] for r in recs])
            xs.append(100.0)
            ys.append(base_auc)
            for eps in ('4.0', '2.0', '1.0', '0.5'):
                xs.append(100 * np.mean([r['rr'][eps]['retention'] for r in recs]))
                ys.append(np.mean([r['rr'][eps]['attack'][g]['auc'] for r in recs]))
            ax.plot(xs, ys, color=COLOR[key], marker='o', ms=3, lw=1.1,
                    label=f'{short[key]}, RR')
            sx = [100 * np.mean([r['chronoscrub'][v]['retention'] for r in recs])
                  for v in ('temporal_scrub', 'joint_0.25', 'joint_0.50', 'joint_0.75')]
            sy = [np.mean([r['chronoscrub'][v]['attack'][g]['auc'] for r in recs])
                  for v in ('temporal_scrub', 'joint_0.25', 'joint_0.50', 'joint_0.75')]
            ax.scatter(sx, sy, color=COLOR[key], marker='^', s=16, edgecolor='black', lw=0.4, zorder=3,
                       label=f'{short[key]}, ChronoScrub')
        ax.set_title(NAMES[p])
        ax.set_xlabel('events published (%)')
        ax.set_xlim(58, 102)
        ax.grid(alpha=0.5)
    axes[0].set_ylabel('AUC')
    if any_data:
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_defense.pdf')


# ---------------------------------------------------------------- appendix

def fig_heterogeneity(S):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.95), sharey=True)
    for ax, (p, ds) in zip(axes, DS.items()):
        uu = units(ds)
        for key, mk in (('release__oracle__degree', 'o'), ('release__oracle__chronotrace', 's')):
            xs = [r['n_pos'] for r in uu]
            ys = [r['results'][key]['auc'] for r in uu]
            ax.scatter(xs, ys, s=7, color=COLOR[key], marker=mk, alpha=0.75, lw=0, label=LABEL[key])
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100'])
        ax.minorticks_off()
        ax.axhline(0.5, color='0.5', ls=':', lw=0.8)
        ax.set_title(NAMES[p])
        ax.set_xlabel(r'$|N_u|$ in the release (log scale)')
        ax.grid(alpha=0.5)
    axes[0].set_ylabel('per-target AUC')
    axes[0].legend(frameon=False, loc='lower right')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_heterogeneity.pdf')


def fig_precision_recall(S):
    keys = ['release__oracle__degree', 'release__oracle__full_nonoracle', 'release__cohort__chronotrace',
            'release__oracle__chronotrace']
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.4))
    for j, (p, ds) in enumerate(DS.items()):
        M = S['datasets'][ds]['methods']
        for k in keys:
            ks = [5, 10, 20, 50]
            ys = [M[k][f'precision@{q}']['mean'] for q in ks]
            es = np.array([[M[k][f'precision@{q}']['mean'] - M[k][f'precision@{q}']['lo'] for q in ks],
                           [M[k][f'precision@{q}']['hi'] - M[k][f'precision@{q}']['mean'] for q in ks]])
            axes[0, j].errorbar(ks, ys, yerr=es, color=COLOR[k], marker='o', ms=3, lw=1.1, capsize=1.6, label=LABEL[k])
            rk = [50, 100, 500]
            ys = [M[k][f'recall@{q}']['mean'] for q in rk]
            es = np.array([[M[k][f'recall@{q}']['mean'] - M[k][f'recall@{q}']['lo'] for q in rk],
                           [M[k][f'recall@{q}']['hi'] - M[k][f'recall@{q}']['mean'] for q in rk]])
            axes[1, j].errorbar(rk, ys, yerr=es, color=COLOR[k], marker='s', ms=3, lw=1.1, capsize=1.6, label=LABEL[k])
        axes[0, j].set_title(NAMES[p])
        axes[0, j].set_xscale('log')
        axes[1, j].set_xscale('log')
        axes[0, j].set_xticks([5, 10, 20, 50])
        axes[0, j].set_xticklabels(['5', '10', '20', '50'])
        axes[1, j].set_xticks([50, 100, 500])
        axes[1, j].set_xticklabels(['50', '100', '500'])
        axes[0, j].minorticks_off()
        axes[1, j].minorticks_off()
        axes[1, j].set_xlabel('$k$')
        for a in axes[:, j]:
            a.grid(alpha=0.5)
    axes[0, 0].set_ylabel('P@$k$')
    axes[1, 0].set_ylabel('R@$k$')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(w_pad=0.4, h_pad=0.4)
    save(fig, 'fig_precision_recall.pdf')


def fig_heuristics(S):
    heur = [('degree_prior', 'Neighbor-count prior', '#08519C'), ('signed_balance', '|Sign balance|', '#3182BD'),
            ('preferential_attachment', 'Preferential attachment', '#FDD0A2'),
            ('common_neighbors', 'Common neighbors', '#FDAE6B'),
            ('jaccard', 'Jaccard', '#FD8D3C'), ('adamic_adar', 'Adamic\u2013Adar', '#F16913'),
            ('temporal_burst', 'Window activity share', '#D94801'), ('temporal_shift', 'Activity shift at window', '#A63603'),
            ('signed_motif', 'Signed motif', '#7F2704'), ('unsupervised_chronotrace', 'CT-Unsupervised', '#41AB5D')]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), sharey=True)
    for ax, (p, ds) in zip(axes, DS.items()):
        M = S['datasets'][ds]['methods']
        for i, (k, lab, col) in enumerate(heur):
            c = M[f'heur__{k}']['auc']
            ax.bar(i, c['mean'], color=col, edgecolor='black', lw=0.4, label=lab)
            ax.errorbar(i, c['mean'], yerr=err(c), color='black', lw=0.6, capsize=1.4)
        ax.axhline(M['release__oracle__chronotrace']['auc']['mean'], color=COLOR['release__oracle__chronotrace'],
                   ls=(0, (4, 2)), lw=0.9, label='ChronoTrace, exact window')
        ax.axhline(0.5, color='0.5', ls=':', lw=0.8)
        ax.set_xticks([])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(NAMES[p])
        ax.grid(axis='y', alpha=0.5)
    axes[0].set_ylabel('AUC (95% CI)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.24), columnspacing=0.8)
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig_heuristics.pdf')


def fig_pool(S):
    pools = S.get('pool_sizes', {})
    avail = [(ds, lab) for ds, lab in (('epinions_dated', 'Epinions (dated)'), ('epinions', 'Epinions (static)'))
             if pools.get(ds)]
    if not avail:
        return
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.9))
    keys = ['release__oracle__degree', 'release__oracle__full_nonoracle', 'release__oracle__chronotrace']
    for ds, lab in avail:
        P = pools[ds]
        for k in keys:
            if k not in P:
                continue
            order = sorted(P[k], key=lambda s: float('inf') if s == 'full' else int(s))
            xs = [int(s) if s != 'full' else None for s in order]
            full_n = {'epinions_dated': 56498, 'epinions': 131827}[ds]
            xs = [x if x is not None else full_n for x in xs]
            ls = '-' if ds == 'epinions_dated' else '--'
            for ax, met in zip(axes, ('auc', 'ap', 'precision@5')):
                ax.plot(xs, [P[k][s][met]['mean'] for s in order], color=COLOR[k], ls=ls, marker='o', ms=2.5,
                        lw=1.0, label=f'{LABEL[k]}, {lab}')
    for ax, t in zip(axes, ('AUC', 'AP', 'P@5')):
        ax.set_xscale('log')
        ax.set_xticks([2000, 5000, 20000, 50000, 131827])
        ax.set_xticklabels(['2k', '5k', '20k', '50k', '132k'])
        ax.minorticks_off()
        ax.set_xlabel('candidate pool size')
        ax.set_title(t)
        ax.grid(alpha=0.5)
    axes[0].set_ylim(0.8, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig_pool.pdf')


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    S = load('summary.json')
    fig_attack_auc(S)
    fig_paired_deltas(S)
    fig_ladder(S)
    fig_window(S)
    fig_strata(S)
    fig_defense()
    fig_heterogeneity(S)
    fig_precision_recall(S)
    fig_heuristics(S)
    fig_pool(S)
    print('figures written to', IMG)


if __name__ == '__main__':
    main()
