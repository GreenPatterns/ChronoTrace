# Data

The experiments use four public graphs. They are **not redistributed** in this repository: download the raw
files from their original sources, then build the exact input files with one script. The script checks every
output against [`SHA256SUMS`](SHA256SUMS), the checksums of the files used for the paper, so a successful
run guarantees byte-identical inputs.

## 1. Download the raw files

Put the four files in one folder, for example `data/raw/` (ignored by git):

| File | Dataset | Source |
|---|---|---|
| `soc-sign-bitcoinotc.csv.gz` | BitcoinOTC: who-trusts-whom ratings on the Bitcoin OTC marketplace, −10 to +10, with timestamps | SNAP, <https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html> |
| `soc-sign-bitcoinalpha.csv.gz` | BitcoinAlpha: the same kind of ratings on the Bitcoin Alpha marketplace | SNAP, <https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html> |
| `soc-sign-epinions.txt.gz` | Epinions trust and distrust statements, static snapshot (no timestamps) | SNAP, <https://snap.stanford.edu/data/soc-sign-epinions.html> |
| `soc-epinions-trust-dir.edges` | Extended Epinions trust and distrust statements with creation dates | Network Repository, <https://networkrepository.com/soc-epinions-trust-dir.php> (unzip the download) |

For example:

```bash
mkdir -p data/raw && cd data/raw
curl -O https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
curl -O https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz
curl -O https://snap.stanford.edu/data/soc-sign-epinions.txt.gz
# Extended Epinions: download the zip from the Network Repository page above, then
unzip soc-epinions-trust-dir.zip    # yields soc-epinions-trust-dir.edges
cd ../..
```

## 2. Build the input files

```bash
python code/scripts/prepare_data.py --raw data/raw
```

The script writes seven files into `data/` and ends with one `OK` line per file:

| File | Content | Used by |
|---|---|---|
| `bitcoinotc.csv` | SNAP rows in file order; timestamps rounded to whole seconds | all experiments |
| `bitcoinalpha.csv` | SNAP rows in file order | all experiments |
| `epinions.csv` | static SNAP Epinions; users numbered by first appearance; `time` is the record index, because the snapshot has no timestamps | Appendix E only |
| `edg60000.csv` | dated Epinions: every Extended Epinions statement with a genuine creation date, without self-ratings | all experiments ("Epinions" in the paper) |
| `edg60000-provenance.json` | how the dated graph was built; also committed here because the paper reports the placeholder share from it | `prepare_data.py`, `make_paper_assets.py` |
| `bitcoinotc-rating.txt`, `bitcoinalpha-rating.txt` | input of the optional GNN baseline: rank of the rater and of the rated user among all user IDs, trust flag, distrust flag, in time order | `run_dgten_revision.py` only |

All CSV files share the columns `source,target,rating,time`. A rating above zero counts as positive. Epinions
statements are stored as +10 (trust) and −10 (distrust).

### How dated Epinions is built

The Extended Epinions release stamps 68.8% of its statements with its first date. That date is a placeholder
for statements created before Epinions recorded dates. The script:

1. drops that block and the self-ratings;
2. keeps the subgraph induced by the 60,000 users with the most statements, which includes every remaining user;
3. sorts the statements by time;
4. numbers users in order of first appearance: users who rated others come first, ordered by their first
   rating given, followed by users who were only rated.

The result has 56,499 users and 261,803 statements. Because of this numbering, the ID–time correlation of
dated Epinions reflects the dataset construction, not the platform's pseudonyms.

## Location

By default the code reads the datasets from this folder. To keep them elsewhere, set
`CHRONOTRACE_DATA_DIR=/path/to/folder`.

## Terms of use and citation

The datasets remain subject to the terms of their sources. If you use them, cite the original work:

- **BitcoinOTC and BitcoinAlpha:** S. Kumar, F. Spezzano, V.S. Subrahmanian, C. Faloutsos. *Edge Weight
  Prediction in Weighted Signed Networks.* ICDM 2016.
- **Epinions:** P. Massa, P. Avesani. *Controversial Users Demand Local Trust Metrics: An Experimental
  Study on Epinions.com Community.* AAAI 2005.
- **Epinions:** J. Leskovec, D. Huttenlocher, J. Kleinberg. *Signed Networks in Social Media.* CHI 2010.
- **Network Repository:** R.A. Rossi, N.K. Ahmed. *The Network Data Repository with Interactive Graph
  Analytics and Visualization.* AAAI 2015.
