#!/usr/bin/env python3
"""Generate every number and table of the revised paper from saved results.

Inputs (all under paper/results/revision/): summary.json (edge-list release, from
analyze_revision.py), summary_directory.json (directory release, the submitted setting),
dataset_stats.json, sign_recovery.json, edge_time.json, and the per-target unit,
defense, DGTEN and sensitivity files. Outputs: paper/generated/macros.tex and
paper/generated/tab_*.tex. Figures are drawn by make_figures.py. A value whose
experiment has not finished is written as \\pending{...}, so the PDF shows the gap instead
of a stale number.

Feature-set names used everywhere: Degree, Topology, Full-R (release-only); Temporal,
Sgn-Struct, ChronoTrace (window-based). The window is none, estimated from the release
(hub, ID cohort: still release-only knowledge), or exact (window-informed knowledge).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'paper' / 'results' / 'revision'
GEN = ROOT / 'paper' / 'generated'
DS = {'otc': 'bitcoinotc', 'alpha': 'bitcoinalpha', 'epi': 'epinions_dated'}
NAMES = {'otc': 'BitcoinOTC', 'alpha': 'BitcoinAlpha', 'epi': 'Epinions'}
N_BOOT = 10_000
MACROS: Dict[str, str] = {}

SETS = {  # unit-file key -> (name in tables, knowledge)
    'random': ('Random', '--'),
    'release__oracle__degree': ('Degree', 'none'),
    'release__oracle__topology': ('Topology', 'none'),
    'release__oracle__full_nonoracle': ('Full-R', 'none'),
    'release__hub__chronotrace': ('ChronoTrace', 'hub (from release)'),
    'release__cohort__chronotrace': ('ChronoTrace', 'ID cohort (from release)'),
    'release__oracle__temporal': ('Temporal', 'exact'),
    'release__oracle__motif': ('Sgn-Struct', 'exact'),
    'release__oracle__chronotrace': ('ChronoTrace', 'exact'),
}
FEATS = {'random': '--', 'release__oracle__degree': '0', 'release__oracle__topology': '0, 7--9',
         'release__oracle__full_nonoracle': '0--2, 7--10', 'release__hub__chronotrace': '0--10',
         'release__cohort__chronotrace': '0--10', 'release__oracle__temporal': '3, 4, 10',
         'release__oracle__motif': '1, 2, 5--9', 'release__oracle__chronotrace': '0--10'}


# ---------------------------------------------------------------- helpers

def load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def data_file(name: str):
    """An input data file: $CHRONOTRACE_DATA_DIR, else <repo>/data, else ../data/cyberdata (None if absent)."""
    roots = [Path(os.environ['CHRONOTRACE_DATA_DIR'])] if os.environ.get('CHRONOTRACE_DATA_DIR') else []
    for root in roots + [ROOT / 'data', ROOT.parent / 'data' / 'cyberdata']:
        if (root / name).exists():
            return root / name
    return None


def f3(x) -> str:
    return 'n/a' if x is None or x != x else f'{x:.3f}'


def f2(x) -> str:
    return 'n/a' if x is None or x != x else f'{x:.2f}'


def minus(s: str) -> str:
    """Typeset a leading minus as a math minus, not a hyphen."""
    return s.replace('-', '$-$')


def sgn(x, nd: int = 3) -> str:
    if x is None or x != x:
        return 'n/a'
    s = f'{x:+.{nd}f}'
    return f'{0:.{nd}f}' if float(s) == 0 else minus(s)


def pct(x, nd: int = 0) -> str:
    return 'n/a' if x is None or x != x else f'{100 * x:.{nd}f}\\%'


def thousands(n: int) -> str:
    return f'{n:,}'.replace(',', '{,}')


def ci(c: dict, nd: int = 3, signed: bool = False) -> str:
    fmt = (lambda v: sgn(v, nd)) if signed else (lambda v: minus(f'{v:.{nd}f}'))
    return f'[{fmt(c["lo"])}, {fmt(c["hi"])}]'


def pm(c: dict, nd: int = 3) -> str:
    return f'{c["mean"]:.{nd}f}$\\pm${c["std"]:.{nd}f}'


def floor2(x: float) -> str:
    return f'{np.floor(x * 100) / 100:.2f}'


def ceil2(x: float) -> str:
    return f'{np.ceil(x * 100) / 100:.2f}'


def rng_str(vals: Iterable[float], nd: int = 2) -> str:
    v = [x for x in vals if x == x]
    lo, hi = f'{min(v):.{nd}f}', f'{max(v):.{nd}f}'
    return lo if lo == hi else f'{lo}--{hi}'


def pval(p: float) -> str:
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def macro(name: str, value) -> None:
    MACROS[name] = str(value)


def pending(name: str, what: str) -> None:
    MACROS.setdefault(name, f'\\pending{{{what}}}')


def boot(vals, seed: int = 42) -> dict:
    v = np.asarray([x for x in vals if x == x], dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)), 'lo': float(np.percentile(means, 2.5)),
            'hi': float(np.percentile(means, 97.5)), 'n': int(len(v))}


def units(ds: str, graph: str = 'real', ts: str = 'main-edgelist') -> List[dict]:
    d = RES / ds / graph / ts / 'units'
    return [json.loads(p.read_text()) for p in sorted(d.glob('*.json'))] if d.exists() else []


def dirjson(ds: str, sub: str) -> List[dict]:
    d = RES / ds / sub
    return [json.loads(p.read_text()) for p in sorted(d.glob('*.json'))] if d.exists() else []


def write(name: str, text: str) -> None:
    (GEN / name).write_text(text)


PLACE_APP = 'app'  # appendix floats: !tbp for one-column tables, !tp for table* (which cannot go at the bottom)


def table(env: str, caption: str, label: str, colspec: str, header: str, rows: List[str], size: str = r'\footnotesize',
          sep: str = '4pt', resize: bool = False, place: str = 't', note: str = '') -> str:
    if place == PLACE_APP:
        place = '!tp' if env == 'table*' else '!tbp'
    inner = table_inner(caption, label, colspec, header, rows, size, sep, resize, env, note)
    return f'\\begin{{{env}}}[{place}]\n\\centering\n{inner}\\end{{{env}}}\n'


def table_inner(caption: str, label: str, colspec: str, header: str, rows: List[str], size: str = r'\footnotesize',
                sep: str = '4pt', resize: bool = False, env: str = 'table', note: str = '') -> str:
    """Caption, label and tabular of one table, without the float environment."""
    body = '\n'.join(rows)
    tab = (f'\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n'
           f'\\end{{tabular}}')
    if resize:
        width = r'\textwidth' if env == 'table*' else r'\columnwidth'
        tab = f'\\resizebox{{{width}}}{{!}}{{%\n{tab}}}'
    extra = f'\n\\\\[2pt]{{\\footnotesize {note}}}' if note else ''
    return (f'\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n'
            f'\\setlength{{\\tabcolsep}}{{{sep}}}\n{tab}{extra}\n')


def side_by_side(left: str, right: str, width: float = 0.48) -> str:
    """Two tables, each with its own caption and number, side by side in one full-width appendix float."""
    return ('\\begin{table*}[!tp]\n'
            f'\\begin{{minipage}}[t]{{{width}\\textwidth}}\n\\centering\n{left}\\end{{minipage}}\\hfill\n'
            f'\\begin{{minipage}}[t]{{{width}\\textwidth}}\n\\centering\n{right}\\end{{minipage}}\n'
            '\\end{table*}\n')


# ---------------------------------------------------------------- macros

def main_macros(S: dict, SD: dict, stats: dict, sign: dict) -> None:
    deg_aucs, rel_dm, win_dm, p5, ex_p = [], [], [], [], []
    for p, ds in DS.items():
        d = S['datasets'][ds]
        M = d['methods']
        get = lambda k, m='auc': M[k][m]['mean']
        macro(f'{p}DegAUC', f3(get('release__oracle__degree')))
        macro(f'{p}RelAUC', f3(get('release__oracle__full_nonoracle')))
        macro(f'{p}WinAUC', f3(get('release__oracle__chronotrace')))
        macro(f'{p}WinGain', sgn(d['gains']['chronotrace']['mean']))
        macro(f'{p}DegPfive', f2(get('release__oracle__degree', 'precision@5')))
        macro(f'{p}WinPfive', f2(get('release__oracle__chronotrace', 'precision@5')))
        macro(f'{p}DegSten', pct(get('release__oracle__degree', 'success@10')))
        macro(f'{p}DegRfive', pct(get('release__oracle__degree', 'recall@500')))
        dc = d['degree_controls']
        macro(f'{p}DmRel', f2(dc['release__oracle__full_nonoracle']['deg_matched_auc']['mean']))
        macro(f'{p}DmWin', f2(dc['release__oracle__chronotrace']['deg_matched_auc']['mean']))
        macro(f'{p}ExRel', sgn(dc['release__oracle__full_nonoracle']['excess_over_deg_cond']['mean']))
        macro(f'{p}ExWin', sgn(dc['release__oracle__chronotrace']['excess_over_deg_cond']['mean']))
        macro(f'{p}BandLowDeg', f2(get('release__oracle__degree', 'band_0-3')))
        macro(f'{p}BandLowWin', f2(get('release__oracle__chronotrace', 'band_0-3')))
        macro(f'{p}BandLowWinCI', '95\\% CI ' + ci(M['release__oracle__chronotrace']['band_0-3'], 2))
        coh = dc['release__cohort__chronotrace']
        macro(f'{p}CohortDM', f2(coh['deg_matched_auc']['mean']))
        macro(f'{p}CohortEx', sgn(coh['excess_over_deg_cond']['mean']))
        macro(f'{p}CohortExP', '$p' + ('<0.001' if coh['excess_over_deg_cond']['p'] < 0.001
                                       else f'={coh["excess_over_deg_cond"]["p"]:.3f}') + '$')
        if p != 'epi':
            ex_p.append(dc['release__oracle__chronotrace']['excess_over_deg_cond']['p'])
        sw = d['window_sweep']
        macro(f'{p}HubIoU', f2(sw['hub']['iou']['mean']))
        macro(f'{p}CohortIoU', f2(sw['cohort']['iou']['mean']))
        macro(f'{p}CohortOverDeg', sgn(sw['cohort']['minus_degree']['mean']))
        macro(f'{p}CohortOverDegCI', '95\\% CI ' + ci(sw['cohort']['minus_degree'], signed=True))
        macro(f'{p}ShiftTwoOverDeg', sgn(sw['shift2']['minus_degree']['mean']))
        macro(f'{p}ShiftTwoOverDegCI', '95\\% CI ' + ci(sw['shift2']['minus_degree'], signed=True))
        macro(f'{p}ShiftEightOverDeg', sgn(sw['shift8']['minus_degree']['mean']))
        macro(f'{p}IDrho', f3(stats[ds]['id_time_spearman']))
        macro(f'{p}RecipSeven', pct(stats[ds]['ratings_reciprocated_within_7d']))
        macro(f'{p}RecipPairSeven', pct(stats[ds]['reciprocal_lag_within_7d']))
        macro(f'{p}RecipLagDays', f"{stats[ds]['reciprocal_lag_median_days']:.0f}")
        st = d['strata_by_neighbors']
        macro(f'{p}SmallWin', f3(st['small']['auc']['chronotrace']['mean']))
        macro(f'{p}LargeWin', f3(st['large']['auc']['chronotrace']['mean']))
        macro(f'{p}Cands', thousands(stats[ds]['users'] - 1))
        if sign and ds in sign:
            macro(f'{p}SignAUC', f2(sign[ds]['auc']))
            macro(f'{p}SignBase', pct(sign[ds]['base_rate']))
        deg_aucs.append(get('release__oracle__degree'))
        rel_dm.append(dc['release__oracle__full_nonoracle']['deg_matched_auc']['mean'])
        if p != 'epi':
            win_dm.append(dc['release__oracle__chronotrace']['deg_matched_auc']['mean'])
        p5 += [M[k]['precision@5']['mean'] for k in M if k.startswith('release__')]
        dd = SD['datasets'][ds]
        macro(f'{p}DirDegAUC', f3(dd['methods']['release__oracle__degree']['auc']['mean']))
        uu = units(ds)
        iso = sum(r['n_isolated_neighbors'] for r in uu) / max(1, sum(r['n_neighbors'] for r in uu))
        macro(f'{p}IsoShare', pct(iso, 1))
        macro(f'{p}IsoTasks', pct(np.mean([r['n_isolated_neighbors'] > 0 for r in uu])))
        best = [r['results']['release__oracle__chronotrace']['best_pos_rank'] for r in uu]
        macro(f'{p}WinBestRank', f'{np.median(best):.0f}')
        macro(f'{p}WinBestRankMean', f'{np.mean(best):.0f}')
    macro('AUCdegRange', rng_str(deg_aucs))
    free = [S['datasets'][ds]['methods'][k]['auc']['mean'] for ds in DS.values()
            for k in ('release__oracle__degree', 'release__oracle__topology', 'release__oracle__full_nonoracle')]
    macro('AUCfreeRange', rng_str(free))
    topo = [S['datasets'][ds]['degree_controls']['release__oracle__topology']['deg_matched_auc'] for ds in DS.values()]
    macro('topoDmCIs', ', '.join(f'{f2(t["mean"])} {ci(t, 2)}' for t in topo[:-1]) + f' and {f2(topo[-1]["mean"])} '
          f'{ci(topo[-1], 2)}')
    macro('bandHighMin', floor2(min(S['datasets'][ds]['methods']['release__oracle__degree']['band_21-inf']['mean']
                                    for ds in DS.values())))
    macro('dmRelRange', rng_str(rel_dm))
    macro('dmWinRange', rng_str(win_dm))
    macro('PfiveMax', ceil2(max(p5)))
    macro('PfiveMissMin', pct(1 - float(ceil2(max(p5)))))
    macro('exWinP', '$p<0.001$' if max(ex_p) < 0.001 else f'$p\\le{max(ex_p):.3f}$')
    seen = {}
    for ds in DS.values():
        for r in units(ds):
            seen.setdefault((ds, r['user']), set()).add(r['seed'])
    macro('dupMain', sum(len(v) > 1 for v in seen.values()))
    if sign:
        macro('signRange', rng_str([sign[ds]['auc'] for ds in DS.values() if ds in sign]))
    le3 = [stats[ds]['share_le3_neighbors'] for ds in DS.values()]
    ev3 = [stats[ds]['share_le3_events'] for ds in DS.values()]
    macro('lowEventMin', pct(min(ev3)))
    macro('lowEventMax', pct(max(ev3)))
    macro('leThreeMin', pct(min(le3)))
    macro('leThreeMax', pct(max(le3)))
    npos = [r['n_neighbors'] for ds in DS.values() for r in units(ds)]
    macro('nposMin', min(npos))
    macro('nposMax', max(npos))
    macro('nposMedian', f'{np.median(npos):.0f}')
    prov = json.loads(data_file('edg60000-provenance.json').read_text())
    macro('epiPlaceholderShare', pct(prov['raw_placeholder_statements'] / prov['raw_edges'], 1))
    diffs = [abs(v['auc']['mean']) for summ in (S, SD) for ds in DS.values()
             for v in summ['datasets'][ds]['protocol_release_minus_lodo'].values()]
    macro('protoMaxDiff', f'{max(diffs):.4f}' if max(diffs) >= 0.0005 else '0.001')
    co = {p: S['datasets'][ds]['coefficients'] for p, ds in DS.items()}
    macro('otcCoefWinShare', sgn(co['otc']['chronotrace']['window_activity_ratio']['mean'], 2))
    macro('alphaCoefPosWin', sgn(co['alpha']['chronotrace']['positive_window_ratio']['mean'], 2))


def edge_time_macros() -> None:
    et = load('edge_time.json')
    if not et:
        for p in DS:
            for n in ('TimeModeExact', 'TimeBusyExact', 'TimeBusyWithin', 'TimeWinWithin'):
                pending(f'{p}{n}', 'edge time')
            pending(f'{p}TimeMidWithin', 'edge time')
        return
    for p, ds in DS.items():
        e = et[ds]
        macro(f'{p}TimeModeExact', pct(e['global_mode']['exact']['value']))
        macro(f'{p}TimeBusyExact', pct(e['busiest']['exact']['value']))
        macro(f'{p}TimeBusyWithin', pct(e['busiest']['within_one']['value']))
        macro(f'{p}TimeWinWithin', pct(e['window_busiest']['within_one']['value']))
        macro(f'{p}TimeMidWithin', pct(e['window_mid']['within_one']['value']))


# ---------------------------------------------------------------- main-body tables

def table_datasets(stats: dict) -> None:
    import datetime as dt
    rows = []
    for p, ds in DS.items():
        s = stats[ds]
        y0 = dt.datetime.utcfromtimestamp(s['first']).strftime('%Y-%m')
        y1 = dt.datetime.utcfromtimestamp(s['last']).strftime('%Y-%m')
        rows.append(' & '.join([NAMES[p], thousands(s['users']), thousands(s['ratings']), f'{y0} to {y1}',
                                f'{s["bin_days"]:.0f}', pct(s['distrust_share']), f2(s['reciprocity']),
                                pct(s['ratings_reciprocated_within_7d']), f2(s['id_time_spearman']),
                                pct(s['share_le3_neighbors']), f'{s["degree_heterogeneity"]:.0f}']) + r' \\')
    write('tab_datasets.tex', table(
        'table*', r'Datasets. Bin: days per time bin. Distrust: share of negative ratings. Recip.: share of ratings '
                  r'whose reverse exists; $\le$7d: reversed within 7 days. ID $\rho$: Spearman correlation of user ID '
                  r'and first activity. $\le$3 nbrs: users with at most three neighbors. Het.: degree heterogeneity '
                  r'$\overline{d^2}/\bar d^{\,2}$.',
        'tab:datasets', 'lrrlrrrrrrr',
        r'Dataset & Users & Ratings & Period & Bin & Distrust & Recip. & $\le$7d & ID $\rho$ & $\le$3 nbrs & Het. \\',
        rows))


KNOW = ['random', 'release__oracle__degree', 'release__oracle__topology', 'release__oracle__full_nonoracle',
        'release__hub__chronotrace', 'release__cohort__chronotrace', 'release__oracle__temporal',
        'release__oracle__motif', 'release__oracle__chronotrace']


def table_main(S: dict) -> None:
    rows = []
    for p, ds in DS.items():
        M = S['datasets'][ds]['methods']
        best_auc = max(M[k]['auc']['mean'] for k in KNOW)
        for i, k in enumerate(KNOW):
            m = M[k]
            name, know = SETS[k]
            auc = f3(m['auc']['mean'])
            auc = f'\\textbf{{{auc}}}' if m['auc']['mean'] == best_auc else auc
            first = f'\\multirow{{{len(KNOW)}}}{{*}}{{{NAMES[p]}}}' if i == 0 else ''
            rows.append(' & '.join([first, name, know, FEATS[k], f'{auc} {ci(m["auc"])}', f3(m['ap']['mean']),
                                    f2(m['precision@5']['mean']), f2(m['success@10']['mean']),
                                    f3(m['recall@100']['mean']), f3(m['recall@500']['mean']),
                                    f2(m['deg_matched_auc']['mean']) if k != 'random' else '--']) + r' \\')
        rows.append(r'\midrule')
    write('tab_main.tex', table(
        'table*', r'Former-neighbor ranking on edge-list releases (mean over 60 targets; AUC with 95\% CI). Random: '
                  r'random scores. Window: none, estimated from the published graph, or exact. Features: indices from '
                  r'Table~\ref{tab:featuremap}. DM-AUC: degree-matched AUC. Bold: highest AUC.',
        'tab:main', 'lllllcccccc',
        r'Dataset & Feature set & Window & Features & AUC [95\% CI] & AP & P@5 & S@10 & R@100 & R@500 & DM-AUC \\',
        rows[:-1]))


def table_ladder(S: dict) -> None:
    rows = []
    for p, ds in DS.items():
        d = S['datasets'][ds]
        lad, dc = d['ladder'], d['degree_controls']
        cfg, act = lad.get('config', {}), lad.get('sigtemp', {})
        if 'degree' in cfg:
            macro(f'{p}CfgDeg', f3(cfg['degree']['null_auc']['mean']))
            g = cfg['degree']['real_minus_null']
            macro(f'{p}GapCfgDeg', sgn(g['mean']))
            macro(f'{p}GapCfgDegCI', '95\\% CI ' + ci(g, signed=True))
        else:
            for n in ('CfgDeg', 'GapCfgDeg', 'GapCfgDegCI'):
                pending(f'{p}{n}', 'config')
        if 'chronotrace' in act:
            macro(f'{p}ActWin', f3(act['chronotrace']['null_auc']['mean']))
            ex = act['chronotrace']['real_minus_null']
            macro(f'{p}ExActWin', sgn(ex['mean']))
            macro(f'{p}ExActWinCI', '95\\% CI ' + ci(ex, signed=True))
            macro(f'{p}CfgWin', f3(cfg['chronotrace']['null_auc']['mean']))
            real_w = d['methods']['release__oracle__chronotrace']['auc']['mean']
            cfg_w, act_w = cfg['chronotrace']['null_auc']['mean'], act['chronotrace']['null_auc']['mean']
            macro(f'{p}ActShare', pct((act_w - cfg_w) / (real_w - cfg_w)))
        else:
            for n in ('ActWin', 'ExActWin', 'ExActWinCI', 'CfgWin'):
                pending(f'{p}{n}', 'activity')
        for j, (g, gname) in enumerate((('degree', 'Degree'), ('full_nonoracle', 'Full-R'),
                                        ('chronotrace', 'ChronoTrace (exact)'))):
            key = f'release__oracle__{g}'
            dcg = dc[key]
            cells = [NAMES[p] if j == 0 else '', gname, f3(d['methods'][key]['auc']['mean']),
                     f3(lad['label_degprop']['mean']) if g == 'degree' else '--',
                     f3(cfg[g]['null_auc']['mean']) if g in cfg else '--',
                     f3(act[g]['null_auc']['mean']) if g in act else '--',
                     f2(dcg['deg_matched_auc']['mean']) + ' ' + ci(dcg['deg_matched_auc'], 2),
                     sgn(dcg['excess_over_deg_cond']['mean']) + ('$^\\ast$' if dcg['excess_over_deg_cond']['p'] < 0.05
                                                                  else '')]
            rows.append(' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    write('tab_ladder.tex', table(
        'table*', r'AUC on the real graph and on reference graphs (rewired: mean over independent rewirings), with '
                  r'degree controls; 60 targets per dataset. Degree (labels): label reference. DM-AUC: degree-matched AUC.',
        'tab:ladder', 'llcccccc',
        r' & & & \multicolumn{3}{c}{Reference-graph AUC} & & \\ \cmidrule(lr){4-6}' '\n'
        r'Dataset & Feature set & Real AUC & Degree (labels) & Degree (rewired) & Activity (rewired) & DM-AUC [95\% CI] & Excess over degree \\',
        rows[:-1], resize=True, note=r'$^\ast$ $p<0.05$, two-sided sign-flip permutation test (unadjusted).'))


def table_coef(S: dict) -> None:
    feats = ['log_degree', 'signed_balance', 'out_in_balance', 'window_activity_ratio', 'pre_post_shift',
             'positive_window_ratio', 'negative_window_ratio', 'log_neighbor_degree', 'log_twohop',
             'clustering_proxy', 'temporal_concentration']
    label = ['Degree', 'Sign balance', 'Direction balance', 'Window activity share', 'Activity shift at window',
             'Positive window share', 'Negative window share', 'Neighbor degree', 'Two-hop reach', 'Closure ratio',
             'Activity concentration']
    rows = []
    for i, f in enumerate(feats):
        cells = [str(i), label[i]]
        for g in ('full_nonoracle', 'chronotrace'):
            for ds in DS.values():
                c = S['datasets'][ds]['coefficients'][g]
                cells.append(sgn(c[f]['mean'], 2) if f in c else '')
        rows.append(' & '.join(cells) + r' \\')
    write('tab_coef.tex', table(
        'table', 'Mean standardized logistic coefficients over 60 targets (OTC: BitcoinOTC; Alpha: BitcoinAlpha; '
                 'Epin.: Epinions).', 'tab:coef', 'rlrrrrrr',
        r' & & \multicolumn{3}{c}{Full-R} & \multicolumn{3}{c}{ChronoTrace (exact)} \\ '
        r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}' '\n'
        r'\# & Feature & OTC & Alpha & Epin. & OTC & Alpha & Epin. \\', rows, size=r'\scriptsize', sep='2.5pt'))


def table_bands(S: dict) -> None:
    keys = [('release__oracle__degree', 'Degree'), ('release__oracle__full_nonoracle', 'Full-R'),
            ('release__oracle__chronotrace', 'ChronoTrace (exact)')]
    rows = []
    for k, name in keys:
        cells = [name]
        for ds in DS.values():
            m = S['datasets'][ds]['methods'][k]
            cells += [f2(m[f'band_{b}']['mean']) for b in ('0-3', '4-20', '21-inf')]
        rows.append(' & '.join(cells) + r' \\')
    counts = []
    for ds in DS.values():
        uu = units(ds)
        for b in ('0-3', '4-20', '21-inf'):
            counts.append(str(sum(r['results']['release__oracle__degree'].get(f'band_n_pos_{b}', 0) for r in uu)))
    rows += [r'\midrule', 'Former neighbors & ' + ' & '.join(counts) + r' \\']
    write('tab_bands.tex', table(
        'table', r'AUC of former neighbors grouped by their degree in $\rel$, each group against all non-neighbors.',
        'tab:bands', 'lccccccccc',
        r' & \multicolumn{3}{c}{BitcoinOTC} & \multicolumn{3}{c}{BitcoinAlpha} & \multicolumn{3}{c}{Epinions} \\ '
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}' '\n'
        r'Degree in $\rel$ & 1--3 & 4--20 & $>$20 & 1--3 & 4--20 & $>$20 & 1--3 & 4--20 & $>$20 \\',
        rows, size=r'\scriptsize', sep='2.2pt'))


HEUR = [('degree_prior', 'Neighbor-count prior', 'R'), ('signed_balance', r'$|$Sign balance$|$', 'R'),
        ('preferential_attachment', 'Preferential attachment', 'W'), ('common_neighbors', 'Common neighbors', 'W'),
        ('jaccard', 'Jaccard', 'W'), ('adamic_adar', 'Adamic--Adar', 'W'),
        ('temporal_burst', 'Window activity share', 'W'), ('temporal_shift', 'Activity shift at window', 'W'),
        ('signed_motif', 'Signed motif', 'W'),
        ('unsupervised_chronotrace', 'CT-Unsupervised', 'W')]


def table_heuristics(S: dict) -> None:
    rows = []
    for key, name, know in HEUR:
        cells = [name, know]
        for ds in DS.values():
            m = S['datasets'][ds]['methods'][f'heur__{key}']
            cells += [pm(m['auc']), f3(m['ap']['mean']), f3(m['recall@500']['mean'])]
        rows.append(' & '.join(cells) + r' \\')
    write('tab_heuristics.tex', table(
        'table*', r'Adapted link-prediction heuristics (mean$\pm$std over 60 targets). R: release-only; W: reads the '
                  r'exact window and the number of former neighbors.',
        'tab:heuristics', 'llccccccccc',
        r' & & \multicolumn{3}{c}{BitcoinOTC} & \multicolumn{3}{c}{BitcoinAlpha} & \multicolumn{3}{c}{Epinions} \\ '
        r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}' '\n'
        r'Heuristic & & AUC & AP & R@500 & AUC & AP & R@500 & AUC & AP & R@500 \\', rows, size=r'\scriptsize',
        sep='3pt'))


DEF_NAMES = {'deletion_only': 'None', 'temporal_scrub': 'ChronoScrub, temporal', 'joint_0.25': 'ChronoScrub, joint 0.25',
             'joint_0.50': 'ChronoScrub, joint 0.50', 'joint_0.75': 'ChronoScrub, joint 0.75',
             '0.5': r'RR, $\epsilon=0.5$', '1.0': r'RR, $\epsilon=1$', '2.0': r'RR, $\epsilon=2$', '4.0': r'RR, $\epsilon=4$'}


def defense_records():
    out = {}
    for p, ds in DS.items():
        recs = dirjson(ds, 'defense-edgelist')
        if len(recs) >= 60:
            out[p] = recs
    return out


def table_defense() -> None:
    D = defense_records()
    order = [('chronoscrub', k) for k in ('deletion_only', 'temporal_scrub', 'joint_0.25', 'joint_0.50', 'joint_0.75')]
    order += [('rr', e) for e in ('0.5', '1.0', '2.0', '4.0')]
    rows, drops, deg_drops, removed = [], [], [], []
    for fam, key in order:
        cells = [DEF_NAMES[key]]
        for p in DS:
            if p not in D:
                cells += ['--'] * 4
                continue
            recs = D[p]
            vals = {g: np.mean([r[fam][key]['attack'][g]['auc'] for r in recs]) for g in ('degree', 'chronotrace')}
            ret = np.mean([r[fam][key]['retention'] for r in recs])
            util = np.nanmean([r[fam][key]['utility_sign_auc'] for r in recs])
            cells += [f3(vals['degree']), f3(vals['chronotrace']), pct(ret), f3(util)]
            base = {g: np.mean([r['chronoscrub']['deletion_only']['attack'][g]['auc'] for r in recs])
                    for g in ('degree', 'chronotrace')}
            if fam == 'chronoscrub' and key != 'deletion_only':
                drops.append(base['chronotrace'] - vals['chronotrace'])
                deg_drops.append(base['degree'] - vals['degree'])
                removed.append(1 - ret)
            if fam == 'rr' and key == '0.5':
                macro(f'{p}RRDegHalf', f3(vals['degree']))
                macro(f'{p}RRDegNone', f3(base['degree']))
                macro(f'{p}RRRemovedHalf', pct(1 - ret))
        rows.append(' & '.join(cells) + r' \\')
        if key == 'joint_0.75':
            rows.append(r'\midrule')
    if drops:
        macro('scrubMaxDrop', f3(max(drops)))
        macro('scrubMaxDegDrop', f3(max(deg_drops)))
        macro('scrubMaxRemoved', pct(max(removed)))
        macro('rrRemovedHalf', MACROS.get('otcRRRemovedHalf', 'n/a'))
    for n in ('scrubMaxDrop', 'scrubMaxDegDrop', 'scrubMaxRemoved', 'rrRemovedHalf', 'otcRRDegHalf', 'otcRRDegNone'):
        pending(n, 'defense')
    write('tab_defense.tex', table(
        'table*', r'Defenses: mean AUC of the Degree and ChronoTrace (exact window) adversaries over 60 targets. '
                  r'Kept: share of events published. Utility: sign-prediction AUC on the defended release. RR: '
                  r'randomized response.',
        'tab:defense', 'lcccccccccccc',
        r' & \multicolumn{4}{c}{BitcoinOTC} & \multicolumn{4}{c}{BitcoinAlpha} & \multicolumn{4}{c}{Epinions} \\ '
        r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}' '\n'
        r'Defense & Degree & ChronoTrace & Kept & Utility & Degree & ChronoTrace & Kept & Utility & Degree & ChronoTrace & Kept & Utility \\',
        rows, size=r'\scriptsize', sep='3pt'))


# ---------------------------------------------------------------- appendix tables

def table_richer(S: dict) -> None:
    keys = KNOW[1:]
    rows = []
    for p, ds in DS.items():
        M = S['datasets'][ds]['methods']
        for i, k in enumerate(keys):
            m = M[k]
            name, know = SETS[k]
            first = f'\\multirow{{{len(keys)}}}{{*}}{{{NAMES[p]}}}' if i == 0 else ''
            rows.append(' & '.join([first, name, know] + [f2(m[q]['mean']) for q in ('precision@5', 'precision@10',
                                                                                     'precision@20', 'success@5',
                                                                                     'success@10')]
                                   + [f'{m["best_pos_rank"]["mean"]:.0f}', f'{m["mean_pos_rank"]["mean"]:.0f}']) + r' \\')
        rows.append(r'\midrule')
    write('tab_richer.tex', table(
        'table*', r'Ranking metrics at fixed depths (mean over 60 targets). S@$k$: at least one former neighbor in the '
                  r'top $k$. Best and mean: rank of the best-ranked and the average rank of the former neighbors.',
        'tab:richer', 'llllccccccc',
        r'Dataset & Feature set & Window & P@5 & P@10 & P@20 & S@5 & S@10 & Best & Mean \\',
        rows[:-1], size=r'\scriptsize', sep='3.5pt', place=PLACE_APP))


def table_paired(S: dict) -> None:
    comps = [('topology', 'Topology'), ('full_nonoracle', 'Full-R'), ('chronotrace_hub', 'ChronoTrace, hub window'),
             ('chronotrace_cohort', 'ChronoTrace, ID-cohort window'), ('temporal', 'Temporal'),
             ('motif', 'Sgn-Struct'), ('chronotrace', 'ChronoTrace, exact window'),
             ('chronotrace_vs_full_nonoracle', 'ChronoTrace (exact) minus Full-R')]
    rows = []
    for p, ds in DS.items():
        G = S['datasets'][ds]['gains']
        for i, (k, name) in enumerate(comps):
            g = G[k]
            first = f'\\multirow{{{len(comps)}}}{{*}}{{{NAMES[p]}}}' if i == 0 else ''
            rows.append(' & '.join([first, name, sgn(g['mean']), ci(g, signed=True), pval(g['p']), pval(g['p_bh'])])
                        + r' \\')
        rows.append(r'\midrule')
    write('tab_paired.tex', table(
        'table*', r'Paired differences in AUC against the Degree adversary (last row of each block: against Full-R), '
                  r'with 95\% bootstrap CIs over 60 targets, two-sided sign-flip $p$-values, and Benjamini--Hochberg '
                  r'adjusted $p$-values within each dataset.',
        'tab:paired', 'llcccc', r'Dataset & Comparison & $\Delta$AUC & 95\% CI & $p$ & $p_{\mathrm{BH}}$ \\',
        rows[:-1], size=r'\scriptsize', sep='4pt', place=PLACE_APP))


def table_protocol(S: dict, SD: dict) -> None:
    names = {'degree': 'Degree', 'topology': 'Topology', 'full_nonoracle': 'Full-R', 'temporal': 'Temporal',
             'motif': 'Sgn-Struct', 'chronotrace': 'ChronoTrace (exact)'}
    rows = []
    for summ, label in ((S, 'edge-list'), (SD, 'directory')):
        for j, (g, gn) in enumerate(names.items()):
            cells = [label if j == 0 else '', gn]
            for ds in DS.values():
                v = summ['datasets'][ds]['protocol_release_minus_lodo'][g]['auc']
                cells.append(f'{sgn(v["mean"], 4)} {ci(v, 4, signed=True)}')
            rows.append(' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    write('tab_protocol.tex', table(
        'table*', r'AUC with training on erasures simulated on the published graph minus AUC with the submitted '
                  r'protocol, in both release formats (mean [95\% CI] over 60 targets).',
        'tab:protocol', 'llccc', r'Release & Feature set & BitcoinOTC & BitcoinAlpha & Epinions \\', rows[:-1],
        size=r'\scriptsize', sep='4pt', place=PLACE_APP))


def table_low(S: dict) -> None:
    keys = [('release__oracle__degree', 'Degree'), ('release__oracle__full_nonoracle', 'Full-R'),
            ('release__cohort__chronotrace', 'ChronoTrace, ID-cohort window'),
            ('release__oracle__chronotrace', 'ChronoTrace, exact window')]
    low = S.get('low', {})
    rows = []
    for p, ds in DS.items():
        L = low.get(ds) or {}
        if not L:
            for x in ('LowDegAUC', 'LowWinAUC', 'LowN'):
                pending(f'{p}{x}', 'low')
            continue
        for i, (k, kn) in enumerate(keys):
            m = L['methods'][k]
            rows.append(' & '.join([NAMES[p] if i == 0 else '', kn, f3(m['auc']['mean']) + ' ' + ci(m['auc']),
                                    f2(m['success@10']['mean']), f'{m["best_pos_rank"]["mean"]:.0f}',
                                    str(m['auc']['n'])]) + r' \\')
        macro(f'{p}LowDegAUC', f3(L['methods']['release__oracle__degree']['auc']['mean']))
        macro(f'{p}LowWinAUC', f3(L['methods']['release__oracle__chronotrace']['auc']['mean']))
        macro(f'{p}LowN', L['methods']['release__oracle__degree']['auc']['n'])
        c = L['chronotrace_minus_degree']
        macro(f'{p}LowGain', sgn(c['mean']))
        macro(f'{p}LowGainCI', '95\\% CI ' + ci(c, signed=True))
        rows.append(r'\midrule')
    write('tab_low.tex', table(
        'table*', r'Targets with one to three events (60 drawn per dataset). Best: mean rank of the best-ranked former '
                  r'neighbor. $n$: targets with at least one former neighbor in the edge-list release.',
        'tab:low', 'llcccc', r'Dataset & Feature set & AUC [95\% CI] & S@10 & Best & $n$ \\', rows[:-1],
        size=r'\scriptsize', sep='4pt', place=PLACE_APP))


def table_static(S: dict) -> None:
    pools = S.get('pool_sizes', {})
    rows = []
    for ds, label in (('epinions_dated', 'Epinions (dated)'), ('epinions', 'Epinions (static)')):
        P = pools.get(ds) or {}
        k = 'release__oracle__degree'
        if k not in P:
            continue
        order = sorted(P[k], key=lambda s: float('inf') if s == 'full' else int(s))
        for i, pool in enumerate(order):
            v = P[k][pool]
            size = 'all' if pool == 'full' else thousands(int(pool))
            rows.append(' & '.join([label if i == 0 else '', size, f3(v['auc']['mean']), f3(v['ap']['mean']),
                                    f2(v['precision@5']['mean']), f2(v['success@10']['mean']),
                                    f'{v["best_pos_rank"]["mean"]:.0f}']) + r' \\')
        rows.append(r'\midrule')
    dated = pools.get('epinions_dated', {}).get('release__oracle__degree', {})
    if dated:
        macro('epiPoolPfiveTwoK', f2(dated['2000']['precision@5']['mean']))
        macro('epiPoolPfiveFull', f2(dated['full']['precision@5']['mean']))
    else:
        pending('epiPoolPfiveTwoK', 'pool')
        pending('epiPoolPfiveFull', 'pool')
    pool_tab = table_inner(r'Degree adversary as the candidate pool shrinks (every former neighbor kept; mean over '
                           r'targets).', 'tab:pool', 'llccccr',
                           r'Dataset & Pool & AUC & AP & P@5 & S@10 & Best \\',
                           rows[:-1] if rows else [r'\multicolumn{7}{c}{\pending{pool sizes}} \\'],
                           size=r'\scriptsize', sep='3pt')
    static = S['datasets'].get('epinions') or {}
    srows = []
    if static:
        dc = static['degree_controls']
        for k in ('release__oracle__degree', 'release__oracle__topology', 'release__oracle__full_nonoracle',
                  'release__oracle__chronotrace'):
            m, e = static['methods'][k], dc[k]['excess_over_deg_cond']
            star = '$^\\ast$' if e['p'] < 0.05 else ''
            srows.append(' & '.join([SETS[k][0], f'{f3(m["auc"]["mean"])} {ci(m["auc"])}', f3(m['ap']['mean']),
                                     f2(m['precision@5']['mean']), f2(dc[k]['deg_matched_auc']['mean']),
                                     sgn(e['mean']) + star]) + r' \\')
        macro('episDegAUC', f3(static['methods']['release__oracle__degree']['auc']['mean']))
        macro('episTopoDM', f2(dc['release__oracle__topology']['deg_matched_auc']['mean']))
        macro('episRelDM', f2(dc['release__oracle__full_nonoracle']['deg_matched_auc']['mean']))
        macro('episTopoEx', sgn(dc['release__oracle__topology']['excess_over_deg_cond']['mean']))
        gt = static['gains']['topology']
        macro('episTopoGain', sgn(gt['mean']))
        macro('episTopoGainCI', '95\\% CI ' + ci(gt, signed=True))
        g = static['gains']['chronotrace']
        macro('episWinGain', sgn(g['mean']))
        macro('episWinGainCI', '95\\% CI ' + ci(g, signed=True))
        nc = static['n_candidates']
        macro('episCands', thousands(int(max(nc) if isinstance(nc, list) else nc)))
        # Share of consecutive static-Epinions records whose rater ID does not decrease; cached so that the
        # paper can be regenerated from the saved results without the dataset.
        cache, src = RES / 'static_epinions_order.json', data_file('epinions.csv')
        if src is not None:
            rec = np.loadtxt(src, delimiter=',', skiprows=1, usecols=(0,), dtype=np.int64)
            order = {'records': int(len(rec)), 'rater_id_nondecreasing_share': float(np.mean(np.diff(rec) >= 0))}
            cache.write_text(json.dumps(order, indent=1))
        else:
            order = json.loads(cache.read_text())
        macro('episOrderShare', pct(order['rater_id_nondecreasing_share']))
    else:
        srows = [r'\multicolumn{6}{c}{\pending{static Epinions}} \\']
        for n in ('episDegAUC', 'episTopoDM', 'episRelDM', 'episTopoEx', 'episTopoGain', 'episTopoGainCI',
                  'episWinGain', 'episWinGainCI', 'episCands', 'episOrderShare'):
            pending(n, 'static')
    static_tab = table_inner(r'Static SNAP Epinions, edge-list release, full candidate universe (60 targets). '
                             r'DM-AUC: degree-matched AUC. Excess: excess over degree.',
                             'tab:static', 'lccccc', r'Feature set & AUC [95\% CI] & AP & P@5 & DM-AUC & Excess \\',
                             srows, size=r'\scriptsize', sep='3pt',
                             note=r'$^\ast$ $p<0.05$, two-sided sign-flip permutation test (unadjusted).')
    write('tab_static.tex', side_by_side(pool_tab, static_tab))


def table_classifier() -> str:
    rows = []
    for p, ds in DS.items():
        recs = dirjson(ds, 'sensitivity-classifier')
        if len(recs) < 60:
            continue
        cells = [NAMES[p]]
        for g in ('degree', 'release_all', 'window_all'):
            cells += [f3(np.mean([r['results'][f'lr_{g}'] for r in recs])),
                      f3(np.mean([r['results'][f'rf_{g}'] for r in recs]))]
        rows.append(' & '.join(cells) + r' \\')
    return table_inner(
        'Logistic regression (LR) and random forest (RF) attackers trained on the same simulated erasures '
        '(mean AUC over 60 targets).', 'tab:classifier', 'lcccccc',
        r' & \multicolumn{2}{c}{Degree} & \multicolumn{2}{c}{Full-R} & \multicolumn{2}{c}{ChronoTrace (exact)} \\ '
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}' '\n'
        r'Dataset & LR & RF & LR & RF & LR & RF \\', rows or [r'\multicolumn{7}{c}{\pending{classifier}} \\'],
        size=r'\scriptsize', sep='3pt')


def table_binarization() -> str:
    rows = []
    for p, ds in (('otc', 'bitcoinotc'), ('alpha', 'bitcoinalpha')):
        recs = dirjson(ds, 'sensitivity-binarization')
        if len(recs) < 60:
            continue
        cells = [NAMES[p]]
        for tau in ('-1', '0', '1', '5'):
            cells.append(f3(np.mean([r['results'][tau]['auc'] for r in recs])))
        cells.append(pct(np.mean([recs[0]['results'][tau]['distrust_share'] for tau in ('0',)])))
        rows.append(' & '.join(cells) + r' \\')
    return table_inner(
        r'ChronoTrace (exact window) when ratings above $\tau$ count as positive (mean AUC over 60 targets).',
        'tab:binarization', 'lccccc', r'Dataset & $\tau=-1$ & $\tau=0$ & $\tau=1$ & $\tau=5$ & Distrust at $\tau=0$ \\',
        rows or [r'\multicolumn{6}{c}{\pending{binarization}} \\'], size=r'\scriptsize', sep='3pt')


def table_dgten(S: dict) -> None:
    rows = []
    for p, ds in (('otc', 'bitcoinotc'), ('alpha', 'bitcoinalpha')):
        recs = dirjson(ds, 'dgten-edgelist')
        if len(recs) < 60:
            continue
        main = {(r['seed'], r['user']): r['results'] for r in units(ds)}
        base = {k: main[k]['release__oracle__degree']['auc'] for k in main}
        entries = [('Degree', {k: main[k]['release__oracle__degree']['auc'] for k in main}),
                   ('DGTEN embedding', {(r['seed'], r['user']): r['results']['dgten']['auc'] for r in recs}),
                   ('Full-R + DGTEN', {(r['seed'], r['user']): r['results']['full_nonoracle_dgten']['auc'] for r in recs}),
                   ('ChronoTrace (exact)', {k: main[k]['release__oracle__chronotrace']['auc'] for k in main}),
                   ('ChronoTrace (exact) + DGTEN',
                    {(r['seed'], r['user']): r['results']['chronotrace_dgten']['auc'] for r in recs})]
        for i, (name, vals) in enumerate(entries):
            keys = sorted(k for k in vals if k in base)
            b = boot([vals[k] for k in keys])
            dlt = boot([vals[k] - base[k] for k in keys]) if name != 'Degree' else None
            rows.append(' & '.join([NAMES[p] if i == 0 else '', name, pm(b),
                                    '--' if dlt is None else f'{sgn(dlt["mean"])} {ci(dlt, signed=True)}']) + r' \\')
        rows.append(r'\midrule')
        macro(f'{p}DgtenAUC', f3(np.mean([r['results']['dgten']['auc'] for r in recs])))
    write('tab_dgten.tex', table(
        'table*', r'Node embeddings of a DGTEN variant as attacker features (mean$\pm$std over 60 targets; '
                  r'$\Delta$: paired difference to Degree with 95\% CI).', 'tab:dgten', 'llcc',
        r'Dataset & Features & AUC & $\Delta$ vs Degree \\', rows[:-1] if rows else [r'\multicolumn{4}{c}{\pending{DGTEN}} \\'],
        size=r'\footnotesize', sep='5pt', place=PLACE_APP))


def table_defense_full() -> None:
    D = defense_records()
    order = [('chronoscrub', k) for k in ('deletion_only', 'temporal_scrub', 'joint_0.25', 'joint_0.50', 'joint_0.75')]
    order += [('rr', e) for e in ('0.5', '1.0', '2.0', '4.0')]
    rows = []
    for p in DS:
        if p not in D:
            continue
        recs = D[p]
        for i, (fam, key) in enumerate(order):
            cells = [NAMES[p] if i == 0 else '', DEF_NAMES[key]]
            for g in ('degree', 'chronotrace'):
                cells.append(pm(boot([r[fam][key]['attack'][g]['auc'] for r in recs])))
            cells.append(f3(np.mean([r[fam][key]['attack']['chronotrace']['recall@500'] for r in recs])))
            cells.append(pm(boot([r[fam][key]['retention'] for r in recs]), 2))
            cells.append(pm(boot([r[fam][key]['utility_sign_auc'] for r in recs]), 3))
            rows.append(' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    write('tab_defense_full.tex', table(
        'table*', r'Defenses in full (mean$\pm$std over 60 targets): AUC of both adversaries, R@500 of ChronoTrace, '
                  r'share of events kept, and sign-prediction utility. RR: randomized response.', 'tab:defensefull',
        'llccccc',
        r'Dataset & Defense & Degree AUC & ChronoTrace AUC & ChronoTrace R@500 & Kept & Utility \\',
        rows[:-1] if rows else [r'\multicolumn{7}{c}{\pending{defense}} \\'], size=r'\scriptsize', sep='3.5pt',
        place=PLACE_APP))


def table_directory(SD: dict) -> None:
    keys = ['release__oracle__degree', 'release__oracle__topology', 'release__oracle__full_nonoracle',
            'release__oracle__temporal', 'release__oracle__motif', 'release__oracle__chronotrace']
    rows = []
    for p, ds in DS.items():
        M = SD['datasets'][ds]['methods']
        for i, k in enumerate(keys):
            m = M[k]
            rows.append(' & '.join([NAMES[p] if i == 0 else '', SETS[k][0], f3(m['auc']['mean']) + ' ' + ci(m['auc']),
                                    f3(m['ap']['mean']), f2(m['precision@5']['mean']),
                                    f2(m['deg_matched_auc']['mean'])]) + r' \\')
        lad = SD['datasets'][ds]['ladder']['degree_minus_label_degprop']
        rows.append(' & '.join(['', r'Degree minus label reference', f'{sgn(lad["mean"])} {ci(lad, signed=True)}',
                                '', '', '']) + r' \\')
        rows.append(r'\midrule')
    write('tab_directory.tex', table(
        'table*', r'The submitted evaluation setting: directory releases, in which users left without events remain '
                  r'candidates (mean over 60 targets).', 'tab:directory', 'llcccc',
        r'Dataset & Feature set & AUC [95\% CI] & AP & P@5 & DM-AUC \\', rows[:-1], size=r'\scriptsize', sep='4pt',
        place=PLACE_APP))


def table_rewire() -> None:
    rows = []
    for p, ds in DS.items():
        for fam, label in (('config', 'degree'), ('sigtemp', 'activity')):
            infos = [json.loads((g / 'rewire_info.json').read_text()) for g in sorted((RES / ds).glob(f'{fam}[0-9]*'))
                     if (g / 'rewire_info.json').exists() and (g / 'main-edgelist' / 'units').exists()]
            if not infos:
                continue
            rows.append(' & '.join([NAMES[p], label, str(len(infos)),
                                    f'{np.mean([i["dropped_self_loops"] for i in infos]):.0f}',
                                    f'{np.mean([i["duplicate_directed_pairs"] for i in infos]):.0f}']) + r' \\')
    write('tab_rewire.tex', table(
        'table', 'Reference graphs: rewirings per dataset, and mean numbers of events dropped as self-loops and of '
                 'repeated rater--rated pairs per rewiring.', 'tab:rewire', 'llrrr',
        r'Dataset & Reference & Rewirings & Self-loops & Duplicates \\', rows, size=r'\scriptsize', sep='4pt',
        place=PLACE_APP))


def table_windows(S: dict) -> None:
    modes = [('oracle', 'exact'), ('shift1', r'shifted, $\delta=1$'), ('shift2', r'shifted, $\delta=2$'),
             ('shift4', r'shifted, $\delta=4$'), ('shift8', r'shifted, $\delta=8$'), ('cohort', 'ID cohort'),
             ('hub', 'hub')]
    rows = []
    for p, ds in DS.items():
        sw = S['datasets'][ds]['window_sweep']
        dc = S['datasets'][ds]['degree_controls']
        for i, (m, name) in enumerate(modes):
            v = sw[m]
            e = dc[f'release__{m}__chronotrace']['excess_over_deg_cond']
            rows.append(' & '.join([NAMES[p] if i == 0 else '', name, f2(v['iou']['mean']),
                                    f3(v['auc']['mean']) + ' ' + ci(v['auc']),
                                    f'{sgn(v["minus_degree"]["mean"])} {ci(v["minus_degree"], signed=True)}',
                                    f2(v['deg_matched_auc']['mean']),
                                    sgn(e['mean']) + ('$^\\ast$' if e['p'] < 0.05 else '')]) + r' \\')
        rows.append(r'\midrule')
    write('tab_windows.tex', table(
        'table*', r'ChronoTrace under each level of timing knowledge (mean over 60 targets). IoU: overlap with the '
                  r'exact window. $\Delta$: paired difference to the Degree adversary. Excess: excess over degree.',
        'tab:windows', 'llccccc',
        r'Dataset & Window & IoU & AUC [95\% CI] & $\Delta$ vs Degree [95\% CI] & DM-AUC & Excess \\', rows[:-1],
        size=r'\scriptsize', sep='4pt', place=PLACE_APP,
        note=r'$^\ast$ $p<0.05$, two-sided sign-flip permutation test (unadjusted).'))


def table_window_sets(S: dict) -> None:
    """AUC of every window-based feature set under each level of timing knowledge."""
    modes = [('oracle', 'exact'), ('shift1', r'shifted, $\delta=1$'), ('shift2', r'shifted, $\delta=2$'),
             ('shift4', r'shifted, $\delta=4$'), ('shift8', r'shifted, $\delta=8$'), ('cohort', 'ID cohort'),
             ('hub', 'hub')]
    sets = ('temporal', 'motif', 'chronotrace')
    cells = [r'none (Degree)']
    for ds in DS.values():
        cells.append(r'\multicolumn{3}{c}{' + f3(S['datasets'][ds]['methods']['release__oracle__degree']['auc']['mean']) + '}')
    rows = [' & '.join(cells) + r' \\', r'\midrule']
    for m, name in modes:
        cells = [name]
        for ds in DS.values():
            M = S['datasets'][ds]['methods']
            cells += [f3(M[f'release__{m}__{g}']['auc']['mean']) for g in sets]
        rows.append(' & '.join(cells) + r' \\')
    for p, ds in DS.items():
        macro(f'{p}TempShiftEight', f3(S['datasets'][ds]['methods']['release__shift8__temporal']['auc']['mean']))
    write('tab_windowsets.tex', table(
        'table*', r'AUC of the three window-based feature sets under each level of timing knowledge (mean over 60 '
                  r'targets); first row: the Degree adversary.', 'tab:windowsets', 'lccccccccc',
        r' & \multicolumn{3}{c}{BitcoinOTC} & \multicolumn{3}{c}{BitcoinAlpha} & \multicolumn{3}{c}{Epinions} \\ '
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}' '\n'
        r'Window & Temporal & Sgn-Struct & ChronoTrace & Temporal & Sgn-Struct & ChronoTrace & Temporal & Sgn-Struct & '
        r'ChronoTrace \\', rows, size=r'\scriptsize', sep='3.5pt', place=PLACE_APP))


def table_swr(S: dict) -> None:
    """Recall and surprisal-weighted recall, in which rare (low-degree) former neighbors count more."""
    keys = [('release__oracle__degree', 'Degree', 'none'), ('release__oracle__full_nonoracle', 'Full-R', 'none'),
            ('release__cohort__chronotrace', 'ChronoTrace', 'ID cohort'),
            ('release__oracle__chronotrace', 'ChronoTrace', 'exact')]
    rows = []
    for p, ds in DS.items():
        M = S['datasets'][ds]['methods']
        for i, (k, name, win) in enumerate(keys):
            m = M[k]
            rows.append(' & '.join([NAMES[p] if i == 0 else '', name, win, f3(m['recall@100']['mean']),
                                    f3(m['swr@100']['mean']), f3(m['recall@500']['mean']),
                                    f3(m['swr@500']['mean'])]) + r' \\')
        rows.append(r'\midrule')
        for k, tag in (('release__oracle__degree', 'Deg'), ('release__oracle__chronotrace', 'Win')):
            r100, w100 = M[k]['recall@100']['mean'], M[k]['swr@100']['mean']
            macro(f'{p}SwrDrop{tag}', pct((r100 - w100) / r100))
    write('tab_swr.tex', table(
        'table', r'Recall (R) and surprisal-weighted recall (SWR) at depths 100 and 500 (mean over 60 targets).',
        'tab:swr', 'lllcccc', r'Dataset & Feature set & Window & R@100 & SWR@100 & R@500 & SWR@500 \\', rows[:-1],
        size=r'\scriptsize', sep='2.5pt', resize=True, place=PLACE_APP))


def table_signtime(sign: dict, et: dict) -> None:
    """Sign and time of deleted relationships (Section 6.4)."""
    ests = ('global_mode', 'busiest', 'window_busiest', 'window_mid')
    rows = []
    for p, ds in DS.items():
        sg, e = sign[ds], et[ds]
        cells = [NAMES[p], thousands(sg['n_pairs']), pct(sg['base_rate']),
                 f"{f2(sg['auc'])} [{f2(sg['lo'])}, {f2(sg['hi'])}]"]
        for crit in ('exact', 'within_one'):
            cells += [pct(e[est][crit]['value']) for est in ests]
        rows.append(' & '.join(cells) + r' \\')
    write('tab_signtime.tex', table(
        'table*', r'Sign and time of deleted relationships, over pairs of a main target and a former neighbor present in '
                  r'the edge-list release. Distrust: share of pairs with a negative rating. Mode: the dataset\textquoteright s '
                  r'busiest bin; Busiest: the neighbor\textquoteright s busiest bin; $W_u$: exact window.',
        'tab:signtime', 'lrrcrrrrrrrr',
        r' & & & & \multicolumn{4}{c}{Same bin} & \multicolumn{4}{c}{Within one bin} \\ '
        r'\cmidrule(lr){5-8}\cmidrule(lr){9-12}' '\n'
        r'Dataset & Pairs & Distrust & Sign AUC [95\% CI] & Mode & Busiest & Busiest in $W_u$ & Middle of $W_u$ & Mode & '
        r'Busiest & Busiest in $W_u$ & Middle of $W_u$ \\', rows, size=r'\scriptsize', sep='3pt', place=PLACE_APP))


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    S, SD = load('summary.json'), load('summary_directory.json')
    stats, sign = load('dataset_stats.json'), load('sign_recovery.json')
    main_macros(S, SD, stats, sign)
    edge_time_macros()
    table_datasets(stats)
    table_main(S)
    table_ladder(S)
    table_coef(S)
    table_bands(S)
    table_heuristics(S)
    table_defense()
    table_richer(S)
    table_paired(S)
    table_protocol(S, SD)
    table_low(S)
    table_static(S)
    write('tab_sensitivity.tex', side_by_side(table_classifier(), table_binarization()))
    table_dgten(S)
    table_defense_full()
    table_directory(SD)
    table_rewire()
    table_windows(S)
    table_window_sets(S)
    table_swr(S)
    table_signtime(sign, load('edge_time.json'))
    lines = ['% Generated by code/scripts/make_paper_assets.py from paper/results/revision. Do not edit.',
             '\\providecommand{\\pending}[1]{\\textbf{[pending: #1]}}']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(MACROS.items())]
    (GEN / 'macros.tex').write_text('\n'.join(lines) + '\n')
    n_pending = sum('pending' in v for v in MACROS.values())
    print(f'wrote {len(MACROS)} macros ({n_pending} pending) and tables to {GEN}')


if __name__ == '__main__':
    main()
