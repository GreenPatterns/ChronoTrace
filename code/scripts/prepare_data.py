#!/usr/bin/env python3
"""Build the input files of every experiment from the public raw releases.

Put the four raw files in one folder and pass it as --raw (see data/README.md for download links):

  soc-sign-bitcoinotc.csv.gz     SNAP, Bitcoin OTC trust network
  soc-sign-bitcoinalpha.csv.gz   SNAP, Bitcoin Alpha trust network
  soc-sign-epinions.txt.gz       SNAP, Epinions signed network (static snapshot, no timestamps)
  soc-epinions-trust-dir.edges   Network Repository, Extended Epinions with creation dates (unzipped)

Outputs, written to --out (default: the repository's data/ folder):

  bitcoinotc.csv, bitcoinalpha.csv   SNAP rows in file order, timestamps rounded to whole seconds
  bitcoinotc-rating.txt,             input of the DGTEN-variant GNN baseline, in time order: rank of the
  bitcoinalpha-rating.txt            rater and of the rated user among all user IDs, trust, distrust
  epinions.csv                       static SNAP Epinions; users numbered by first appearance;
                                     time = record index (the snapshot has no timestamps)
  edg60000.csv,                      dated Epinions: every Extended Epinions statement with a genuine
  edg60000-provenance.json           creation date, without self-ratings; users numbered by first appearance

Each output is compared with data/SHA256SUMS, the checksums of the files used for the paper, so a
successful run proves that the inputs are byte-identical to ours.

Usage: python code/scripts/prepare_data.py --raw /path/to/raw [--out data]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HEADER = 'source,target,rating,time'
N_KEEP = 60000  # top users by degree in the dated-Epinions builder; all 56,499 dated users qualify


def _open(path: Path):
    return gzip.open(path, 'rt') if path.suffix == '.gz' else open(path)


def _write_csv(rows, path: Path, header: str = HEADER) -> None:
    # The original files use CRLF line endings, including after the last row.
    with open(path, 'w', newline='') as f:
        f.write(header + '\r\n')
        for s, d, r, t in rows:
            f.write(f'{s},{d},{r},{t}\r\n')


def bitcoin(raw: Path, name: str, out: Path) -> None:
    """SNAP soc-sign-bitcoin{otc,alpha}: SOURCE,TARGET,RATING,TIME without header; TIME in seconds."""
    rows = []
    with _open(raw / f'soc-sign-{name}.csv.gz') as f:
        for line in f:
            s, d, r, t = line.strip().split(',')
            rows.append((int(s), int(d), int(r), int(round(float(t)))))
    # The BitcoinAlpha file used for the paper carries a stray space in its header ("target ");
    # it is kept so that the checksum matches. The loader strips header whitespace.
    header = 'source,target ,rating,time' if name == 'bitcoinalpha' else HEADER
    _write_csv(rows, out / f'{name}.csv', header)
    # Rating file of the GNN baseline: one line per rating in time order (a stable sort keeps file order
    # within a timestamp), users indexed by their rank among all IDs, then trust and distrust flags.
    # The two files used for the paper differ in line endings (OTC: CRLF without a final newline;
    # Alpha: LF with one); both are reproduced byte for byte.
    rank = {v: i for i, v in enumerate(sorted({x for s, d, _, _ in rows for x in (s, d)}))}
    ordered = sorted(rows, key=lambda row: row[3])
    lines = [f'{rank[s]}\t{rank[d]}\t{int(r > 0)}\t{int(r <= 0)}' for s, d, r, _ in ordered]
    text = '\r\n'.join(lines) if name == 'bitcoinotc' else '\n'.join(lines) + '\n'
    with open(out / f'{name}-rating.txt', 'w', newline='') as f:
        f.write(text)


def epinions_static(raw: Path, out: Path) -> None:
    """SNAP soc-sign-epinions: FromNodeId ToNodeId Sign (no time). Users renumbered by first appearance."""
    rows, ids = [], {}
    with _open(raw / 'soc-sign-epinions.txt.gz') as f:
        for line in f:
            if line.startswith('#'):
                continue
            s, d, sign = (int(x) for x in line.split())
            for v in (s, d):
                ids.setdefault(v, len(ids))
            rows.append((ids[s], ids[d], 10 * sign, len(rows)))
    _write_csv(rows, out / 'epinions.csv')


def epinions_dated(raw: Path, out: Path) -> None:
    """Extended Epinions (Network Repository soc-epinions-trust-dir): source target sign unix_time.

    68.8% of the statements carry the release's first date, a placeholder for statements created before
    Epinions recorded dates. We drop that block and the self-ratings, keep the subgraph induced by the
    N_KEEP users with the most statements (all remaining users qualify), sort by time and number the
    users in order of first appearance: raters by their first rating given, then users only rated.
    """
    E = np.loadtxt(raw / 'soc-epinions-trust-dir.edges', comments='%', dtype=np.int64)
    df = pd.DataFrame(E, columns=['source', 'target', 'sign', 'time'])
    n_raw, n_loops = len(df), int((df.source == df.target).sum())
    df = df[df.source != df.target]
    placeholder = int(df.time.min())
    n_placeholder = int((df.time == placeholder).sum())
    df = df[df.time > placeholder]
    deg = pd.concat([df.source, df.target]).value_counts()
    keep = set(deg.index[:N_KEEP])
    sub = df[df.source.isin(keep) & df.target.isin(keep)].copy()
    sub = sub.sort_values('time', kind='mergesort').reset_index(drop=True)
    nodes = pd.unique(pd.concat([sub.source, sub.target]))
    remap = {int(v): i for i, v in enumerate(nodes)}
    sub['source'] = sub.source.map(remap)
    sub['target'] = sub.target.map(remap)
    sub['rating'] = np.where(sub.sign > 0, 10, -10)
    sub[['source', 'target', 'rating', 'time']].to_csv(out / f'edg{N_KEEP}.csv', index=False)
    t0 = int(df.time.min())
    stats = {
        'source': 'Network Repository soc-epinions-trust-dir (Extended Epinions, Massa & Avesani 2005)',
        'raw_edges': n_raw, 'raw_self_loops_dropped': n_loops, 'top_n_by_degree': N_KEEP,
        'placeholder_time': placeholder, 'raw_placeholder_statements': n_placeholder,
        'placeholder_block_dropped': True,
        'nodes': int(len(nodes)), 'edges': int(len(sub)),
        'distrust_share': round(float((sub.sign < 0).mean()), 4),
        'initial_date_share': round(float((sub.time == t0).mean()), 4),
        'time_min': int(sub.time.min()), 'time_max': int(sub.time.max()),
    }
    with open(out / f'edg{N_KEEP}-provenance.json', 'w') as f:
        json.dump(stats, f, indent=2)


def verify(out: Path, sums: Path) -> bool:
    expected = dict(line.split()[::-1] for line in sums.read_text().splitlines() if line.strip())
    ok = True
    for name, digest in sorted(expected.items()):
        p = out / name
        if not p.exists():
            print(f'  MISSING  {name}')
            ok = False
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f'  {"OK      " if got == digest else "MISMATCH"} {name}')
        ok &= got == digest
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, required=True, help='folder with the four raw files')
    ap.add_argument('--out', type=Path, default=ROOT / 'data', help='output folder (default: data/)')
    ap.add_argument('--sums', type=Path, default=ROOT / 'data' / 'SHA256SUMS', help='expected checksums')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print('BitcoinOTC ...', flush=True)
    bitcoin(args.raw, 'bitcoinotc', args.out)
    print('BitcoinAlpha ...', flush=True)
    bitcoin(args.raw, 'bitcoinalpha', args.out)
    print('static Epinions ...', flush=True)
    epinions_static(args.raw, args.out)
    print('dated Epinions ...', flush=True)
    epinions_dated(args.raw, args.out)
    if args.sums.exists():
        print(f'checking against {args.sums}:')
        sys.exit(0 if verify(args.out, args.sums) else 1)


if __name__ == '__main__':
    main()
