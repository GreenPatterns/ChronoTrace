# ChronoTrace

**Calibrating residual deleted-neighbor inference after node deletion in dynamic trust graphs.**

When a trust or reputation platform erases a user, it deletes every rating the user gave or received.
Each deleted rating also had a counterparty, though, and the counterparties' other ratings stay public.
ChronoTrace measures what the remaining, published graph still reveals about **whom the erased user
interacted with**, and **why**.

ChronoTrace has two parts:

- **Attack.** An adversary ranks every remaining user as a likely former neighbor of the erased user.
  It sees either the published graph alone (*release-only*) or also knows when the user was active
  (*window-informed*). It trains only on erasures it simulates on the published graph itself, so it needs
  no ground truth about the erased user.
- **Calibration.** The same adversary also runs on *reference graphs* that preserve every user's degree, or
  every user's per-month activity, while randomizing who interacted with whom. Comparing accuracies separates
  what any graph with those properties would reveal from what remains beyond them. Two degree controls refine
  the comparison: the degree-matched AUC and the excess over degree.

This repository contains everything behind the paper:

- the Python code of the benchmark and of every experiment;
- every per-target result record (4,000+ JSON files) and the run logs;
- the scripts that turn those records into every number, table and figure of the paper;
- the paper itself, as a PDF ([`paper/ChronoTrace.pdf`](paper/ChronoTrace.pdf)), and the same paper with every change since its previous version marked ([`paper/ChronoTrace_changes.pdf`](paper/ChronoTrace_changes.pdf); additions underlined in blue, deletions struck through in red); the manuscript sources are not distributed;
- a data-preparation script that rebuilds the exact input files from the public datasets, verified by checksum.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Requirements](#requirements)
3. [Quick start: regenerate the paper's numbers, tables and figures](#quick-start-regenerate-the-papers-numbers-tables-and-figures)
4. [Data](#data)
5. [Reproducing every experiment](#reproducing-every-experiment)
6. [Where each table and figure comes from](#where-each-table-and-figure-comes-from)
7. [Result records](#result-records)
8. [The benchmark in brief](#the-benchmark-in-brief)
9. [Code overview](#code-overview)
10. [Tests](#tests)
11. [Notes and caveats](#notes-and-caveats)

---

## Repository layout

```
.
├── README.md                     this file
├── requirements.txt              pinned Python dependencies
├── code/
│   ├── chronotrace/              the benchmark library (features, adversaries, windows, calibration, metrics, defenses)
│   ├── mycode/                   vendored GNN (a deterministic DGTEN variant) for the optional GNN baseline
│   └── scripts/                  data preparation, experiment runners, analyses, table and figure generators
├── data/
│   ├── README.md                 where to download the datasets and how they are prepared
│   ├── SHA256SUMS                checksums of the exact input files used for the paper
│   └── edg60000-provenance.json  provenance of the dated Epinions build
├── paper/
│   ├── ChronoTrace.pdf           the paper
│   ├── ChronoTrace_changes.pdf   the paper with changes since its previous version marked
│   └── results/revision/         every result record, summary and log (see "Result records")
└── tests/                        pytest suite
```

Every number, data table and plot in the paper is produced by two scripts from the saved records in
`paper/results/revision/`; nothing is typed by hand. `make_paper_assets.py` writes every number, as macro
definitions in `paper/generated/macros.tex`, and every data table, as `paper/generated/tab_*.tex`.
`make_figures.py` writes every plot to `paper/images/fig_*.pdf`. These outputs are regenerated locally and
are not tracked by git.

---

## Requirements

- **Python 3.10** (tested with 3.10.14). Install the dependencies with

  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

  The core pipeline needs only `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib` and `pytest`.
  `torch`, `torch-scatter`, `einops`, `easydict`, `networkx`, `texttable` and `tqdm` are needed only for
  the optional GNN baseline (Appendix D.11). That baseline trains on a CUDA GPU, but its embeddings are
  cached in this repository. Remove those lines from `requirements.txt` if you do not need it.
- **Hardware** (only to rerun the experiments): any multi-core Linux machine. The runners parallelize over
  targets with worker processes and pin BLAS to one thread per worker. The run times quoted below are for
  22 workers. Dated Epinions (56,499 users) is the most time- and memory-intensive dataset; lower
  `WORKERS` if memory is short.

---

## Quick start: regenerate the paper's numbers, tables and figures

No dataset, no GPU and no experiment run is needed. About two minutes:

```bash
python code/scripts/make_paper_assets.py     # -> paper/generated/macros.tex (every number) and tab_*.tex (every data table)
python code/scripts/make_figures.py          # -> paper/images/fig_*.pdf (every plot)
```

Compare the outputs with [`paper/ChronoTrace.pdf`](paper/ChronoTrace.pdf). Each macro in `macros.tex` is
one number of the text, such as `\otcDegAUC` (the Degree adversary's AUC on BitcoinOTC). Each `tab_*.tex`
is one table, and each `fig_*.pdf` is one figure. `make_paper_assets.py` reports how many macros it wrote
and how many are pending. A pending value comes from an experiment that has not finished.

---

## Data

The four datasets are public but are not redistributed here. [`data/README.md`](data/README.md) gives the
download links and details. In short:

```bash
# put soc-sign-bitcoinotc.csv.gz, soc-sign-bitcoinalpha.csv.gz, soc-sign-epinions.txt.gz (SNAP)
# and soc-epinions-trust-dir.edges (Network Repository) into data/raw/, then
python code/scripts/prepare_data.py --raw data/raw
```

The script rebuilds the seven input files and checks each against [`data/SHA256SUMS`](data/SHA256SUMS);
every line must read `OK`. By default the code reads `data/`; set `CHRONOTRACE_DATA_DIR` to use another
folder.

| Name in the paper | File | Users | Ratings | Period |
|---|---|---:|---:|---|
| BitcoinOTC | `bitcoinotc.csv` | 5,881 | 35,592 | 2010-11 to 2016-01 |
| BitcoinAlpha | `bitcoinalpha.csv` | 3,783 | 24,186 | 2010-11 to 2016-01 |
| Epinions (dated) | `edg60000.csv` | 56,499 | 261,803 | 2001-01 to 2003-08 |
| static SNAP Epinions (Appendix E only) | `epinions.csv` | 131,828 | 841,372 | no timestamps (record order) |

---

## Reproducing every experiment

The experiments write into `paper/results/revision/`. Each runner skips targets whose record already
exists, so delete or move the existing records (or work in a fresh clone) to recompute them. The runners
also write `.npz` caches next to the records: per-target feature matrices in `tasks/` and per-candidate
scores in `units/`, about 3.7 GB in total. These caches are excluded from git (see `.gitignore`) and are
recreated by the runs below.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. Main experiments on edge-list releases, in priority order (about 3 hours with 22 workers):
#    real graphs (all windows, both training protocols), degree- and activity-preserving reference graphs,
#    low-degree targets, defenses, and the static Epinions snapshot. Logs: paper/results/revision/logs/.
WORKERS=22 PY=python code/scripts/run_revision_all.sh

# 2. Directory releases (the submitted setting; Appendix E and the protocol check), about 10 min
python code/scripts/run_revision.py --universe directory --stage tasks  --workers 22
python code/scripts/run_revision.py --universe directory --stage attack --workers 22

# 3. Sensitivity analyses (Appendices D.9 and D.10), about 50 min
python code/scripts/run_sensitivity_revision.py --analysis classifier --workers 22
python code/scripts/run_sensitivity_revision.py --analysis binarization --datasets bitcoinotc,bitcoinalpha --workers 22

# 4. GNN baseline (Appendix D.11), minutes; trains on a GPU only if the cached embeddings in
#    paper/results/revision/dgten/ are missing
CUDA_VISIBLE_DEVICES=0 python code/scripts/run_dgten_revision.py --workers 8

# 5. Analyses (each about a minute)
python code/scripts/dataset_stats.py           # -> dataset_stats.json (Table 3)
python code/scripts/analyze_sign_recovery.py   # -> sign_recovery.json (Section 6.4)
python code/scripts/analyze_edge_time.py       # -> edge_time.json (Section 6.4)
python code/scripts/analyze_revision.py --universe edgelist
python code/scripts/analyze_revision.py --universe directory --datasets bitcoinotc,bitcoinalpha,epinions_dated

# 6. Numbers, tables and figures of the paper
python code/scripts/make_paper_assets.py
python code/scripts/make_figures.py
```

Approximate wall-clock times of the edge-list batches with 22 workers:

| Batch | Time |
|---|---|
| main targets, real graphs | ~10 min |
| reference graphs: Bitcoin / Epinions | ~7 min / ~43 min |
| low-degree targets | ~30 min |
| defenses | ~60 min |
| static Epinions | ~24 min |

**Determinism.** Every random choice is seeded: target sampling, negative sampling, window shifts,
rewiring, label references, randomized response, bootstrap and permutation tests. With the pinned
library versions and one BLAS thread per worker, a rerun reproduces the saved records.

**Optional reference-graph runs on directory releases.** The records in `<dataset>/{config,sigtemp}<r>/main/`
are directory-release runs on the Bitcoin reference graphs. No result in the paper uses them. To recompute
them, add `--graphs config0,config1,config2,config3,config4,sigtemp0,sigtemp1,sigtemp2,sigtemp3,sigtemp4
--datasets bitcoinotc,bitcoinalpha --protocols release --modes oracle` to both stages of step 2.

**Utility values written before the held-out fix.** `recompute_defense_utility.py` recomputed the
sign-prediction utility in defense records created before each training edge was featurized without
itself. It rebuilds each defended release, checks that it reproduces the stored values, then replaces them.
Fresh runs already use the corrected featurization, so you do not need this script for a fresh run.

---

## Where each table and figure comes from

Table and figure numbers refer to [`paper/ChronoTrace.pdf`](paper/ChronoTrace.pdf). `R/` stands for
`paper/results/revision/`.

| Paper element | Produced by | From |
|---|---|---|
| Figures 1 and 2 (running example, pipeline), Tables 1 and 2 (positioning, feature map) | written in the paper | — |
| Table 3 (datasets) | `make_paper_assets.py` | `R/dataset_stats.json` |
| Table 4 (all adversaries), Figure 3 (AUC), Figure 4 (per-target differences) | `make_paper_assets.py`, `make_figures.py` | `R/summary.json`, `R/<ds>/real/main-edgelist/units/*.json` |
| Table 5 and Figure 5 (calibration ladder) | both | `R/summary.json` (ladder, degree controls) from `R/<ds>/{config,sigtemp}<r>/main-edgelist/` |
| Table 6 (coefficients), Table 7 (degree bands) | `make_paper_assets.py` | `R/summary.json` |
| Figure 6, Tables 14 and 15 (timing knowledge, every window-based set) | both | `R/summary.json` (window sweep, methods) |
| Figure 7 (target groups), Table 16 (low-degree targets) | both | `R/summary.json` (strata, low) from `R/<ds>/real/low-edgelist/` |
| Section 6.4, Table 17 (sign and time of deleted relationships) | `make_paper_assets.py` | `R/sign_recovery.json`, `R/edge_time.json` |
| Table 18 (surprisal-weighted recall) | `make_paper_assets.py` | `R/summary.json` |
| Table 8, Figure 8, Table 22 (defenses) | both | `R/<ds>/defense-edgelist/*.json` |
| Table 9 and Figure 11 (adapted heuristics) | both | `R/summary.json` (`heur__*`) |
| Table 10 (reference graphs) | `make_paper_assets.py` | `R/<ds>/{config,sigtemp}<r>/rewire_info.json` |
| Table 11 (training-protocol check) | `make_paper_assets.py` | `R/summary.json`, `R/summary_directory.json` |
| Tables 12 and 13, Figures 9 and 10 (fixed depths, paired tests, PR curves, per-target AUC) | both | `R/summary.json`, unit records |
| Tables 19 and 20 (random forest, sign binarization) | `make_paper_assets.py` | `R/<ds>/sensitivity-{classifier,binarization}/*.json` |
| Table 21 (GNN embeddings) | `make_paper_assets.py` | `R/<ds>/dgten-edgelist/*.json` |
| Table 23 (directory releases) | `make_paper_assets.py` | `R/summary_directory.json` from `R/<ds>/real/main/` |
| Tables 24 and 25, Figure 12 (pool size, static Epinions) | both | `R/summary.json` (pool sizes) from `R/epinions/real/main-edgelist/` |

`R/REPORT.md` and `R/REPORT_directory.md` are human-readable digests of the two summaries.

---

## Result records

### Folder structure

```
paper/results/revision/
├── summary.json, REPORT.md                  edge-list releases: aggregates of every record (analyze_revision.py)
├── summary_directory.json, REPORT_directory.md   the same for directory releases
├── dataset_stats.json                       dataset statistics (Table 3)
├── sign_recovery.json, sign_recovery_directory.json, edge_time.json   Section 6.4
├── static_epinions_order.json               record-order statistic of the static snapshot (Appendix E)
├── logs/                                    stdout of every batch
├── dgten/                                   cached GNN node embeddings
└── <dataset>/                               bitcoinotc, bitcoinalpha, epinions_dated, epinions (static)
    ├── real/<targetset>[-edgelist]/units/   one JSON record per target (seed, user)
    │                                        targetset: main (60 targets) or low (60 targets with 1-3 events);
    │                                        suffix -edgelist: edge-list release; no suffix: directory release
    ├── config<r>/, sigtemp<r>/              degree reference (config) and activity reference (sigtemp), rewiring r,
    │                                        with rewire_info.json and the same units/ layout
    ├── defense-edgelist/                    one record per target: ChronoScrub variants and randomized response
    ├── sensitivity-classifier/              logistic regression vs random forest
    ├── sensitivity-binarization/            ChronoTrace AUC per sign threshold (Bitcoin only)
    └── dgten-edgelist/                      GNN-embedding adversaries (Bitcoin only)
```

### One target's record (`units/seed<S>_u<U>.json`)

| Field | Meaning |
|---|---|
| `dataset`, `graph`, `targetset`, `universe`, `seed`, `user` | which erasure this is |
| `n_candidates` | candidates ranked (users still in the release) |
| `n_neighbors`, `n_pos`, `n_isolated_neighbors` | former neighbors in total; among the candidates; left without events (identified exactly by a directory release) |
| `incident_degree` | the target's number of events |
| `true_window` | the exact activity window `[b0, b1]` in time bins (32 bins per dataset), padded by one bin |
| `train_users` | the shadow users whose simulated erasures trained the adversary |
| `windows` | per window mode, the window used (`test`) and its IoU with the exact window |
| `label_null` | AUC of the Degree ranking against uniform and degree-proportional random labels (label references) |
| `results` | one entry per method (below), each a dictionary of metrics |

Method keys follow `<protocol>__<window>__<feature set>`:

| Part | Values |
|---|---|
| protocol | `release`: trained on erasures simulated on the published graph (used throughout the paper); `lodo`: trained on erasures from the original graph (the submitted protocol, compared in Appendix C.2) |
| window | `oracle` (exact window), `shift1`/`shift2`/`shift4`/`shift8` (shifted by δ bins), `hub` and `cohort` (estimated from the published graph) |
| feature set | `degree` (Degree), `topology` (Topology), `full_nonoracle` (Full-R), `temporal` (Temporal), `motif` (Sgn-Struct), `chronotrace` (ChronoTrace) |

The other keys are `random` (random scores) and `heur__<name>` (adapted link-prediction heuristics; the
names follow the code, so `degree_prior` is the paper's neighbor-count prior, `temporal_burst` its window
activity share and `temporal_shift` its activity shift).

Metrics in each entry:

| Metric | Meaning |
|---|---|
| `auc`, `ap` | AUC (ties count one half) and average precision over all candidates |
| `precision@k`, `recall@k`, `success@k` | share of the top k that are former neighbors; share of former neighbors in the top k; at least one in the top k |
| `best_pos_rank`, `mean_pos_rank` | rank (1 = top) of the best-ranked former neighbor and mean rank of all former neighbors |
| `deg_matched_auc` | AUC over pairs of a former neighbor and a non-neighbor with the same degree |
| `deg_cond_auc` | expected AUC when the former neighbors are redrawn within degree classes (excess over degree = `auc − deg_cond_auc`) |
| `band_<a-b>`, `band_n_pos_<a-b>` | AUC of the former neighbors with a–b events (each band against all non-neighbors), and their number |
| `coef`, `intercept` | standardized logistic-regression coefficients (learned adversaries) |
| `mcc@degree`, `swr@k` | Matthews correlation at the true neighbor count; surprisal-weighted recall (computed, not reported in the paper) |

### Summaries

`summary.json` has one entry per dataset with these parts:

- `methods`: mean, standard deviation and 95% bootstrap interval of every metric;
- `gains`: paired differences to Degree, with sign-flip p-values and Benjamini–Hochberg adjustment;
- `degree_controls`: degree-matched AUC and excess over degree, with p-values;
- `ladder`: label references and the rewired references;
- `coefficients`, `window_sweep`, `strata_by_neighbors`, `protocol_release_minus_lodo`;
- the top-level `low` (low-degree targets) and `pool_sizes` (candidate-pool analysis).

Intervals are percentile bootstraps over targets with 10,000 resamples. The p-values are two-sided and come
from 10,000 random sign flips.

---

## The benchmark in brief

The paper (Sections 3 and 4 and Appendix B) gives the full definitions.

- **Erasure.** Deleting user u removes every event u gave or received. The *published graph* is what
  remains. An *edge-list release* lists events only, so users left without events disappear. A *directory
  release* lists every account, so users left without events are exactly identified former neighbors.
- **Knowledge levels.**
  - A *release-only* adversary knows only the published graph and u's pseudonym. It can use window-free
    features, or estimate u's activity window from the published graph: the *hub* estimator, or the
    *ID-cohort* estimator, which uses the activity of users with nearby IDs.
  - A *window-informed* adversary also holds the exact window (an upper bound) or a window shifted by δ bins.
- **Features.** Eleven features per candidate, defined in `chronotrace/features.py` and Appendix A:
  - degree;
  - sign and direction balance;
  - four window features: activity share, activity shift, positive share, negative share;
  - neighbor degree, two-hop reach, closure ratio;
  - activity concentration.

  The window-free feature sets are Degree, Topology and Full-R. The window-based sets are Temporal,
  Sgn-Struct and ChronoTrace (all eleven features).
- **Training.** A logistic regression with balanced class weights trains on *shadow erasures*: other users
  deleted from the published graph itself. No event, label or window of the target enters training
  (`chronotrace/protocols.py`).
- **Calibration.** The degree reference permutes the rated end of every event, so every user keeps its
  events given and received. The activity reference permutes within (time bin, sign), so every user keeps its
  per-bin, per-sign activity (`chronotrace/nulls.py`). The degree controls are in
  `chronotrace/metrics_ext.py`.
- **Defenses.** ChronoScrub (local edge suppression around the erased user's neighbors, diagnostic) and
  deletion-only edge-level randomized response (`chronotrace/features.py`). The sign-prediction utility is in
  `chronotrace/utility.py`.

---

## Code overview

| Path | Purpose |
|---|---|
| `code/chronotrace/core.py` | data loading, time bins, target sampling, ranking metrics, data location (`CHRONOTRACE_DATA_DIR`) |
| `code/chronotrace/features.py` | erasure, the eleven candidate features, windows, ChronoScrub, randomized response, `build_task` |
| `code/chronotrace/protocols.py` | shadow-erasure training (release-only) and model fitting |
| `code/chronotrace/windows.py`, `window_estimator.py` | exact, shifted, hub and ID-cohort windows |
| `code/chronotrace/nulls.py` | degree- and activity-preserving rewiring; label references |
| `code/chronotrace/metrics_ext.py` | degree-matched AUC, excess over degree, degree bands, pool-size metrics |
| `code/chronotrace/baselines.py` | adapted link-prediction heuristics and CT-Unsupervised |
| `code/chronotrace/utility.py` | sign-prediction utility of a defended release |
| `code/chronotrace/dgten_features.py`, `code/mycode/` | GNN-embedding baseline (deterministic DGTEN variant) |
| `code/chronotrace/attack.py`, `experiment.py` | the original experiment driver; kept because the package (`__init__.py`) and the tests import it |
| `code/scripts/prepare_data.py` | builds the input files from the public raw data and checks their checksums |
| `code/scripts/run_revision.py`, `run_revision_all.sh` | main experiments: tasks, then attacks, per dataset, graph, target set, window and protocol |
| `code/scripts/run_defense_revision.py` | defense experiments |
| `code/scripts/run_sensitivity_revision.py`, `run_dgten_revision.py` | appendix sensitivity analyses and GNN baseline |
| `code/scripts/analyze_revision.py` | aggregates records into `summary*.json` and `REPORT*.md` |
| `code/scripts/analyze_sign_recovery.py`, `analyze_edge_time.py`, `dataset_stats.py` | Section 6.4 analyses and dataset statistics |
| `code/scripts/make_paper_assets.py`, `make_figures.py` | every macro, table and figure of the paper |
| `code/scripts/recompute_defense_utility.py` | one-off correction of utility values in older defense records |

Run scripts from the repository root. They add `code/` to the Python path themselves.

---

## Tests

```bash
python -m pytest tests/
```

`tests/test_revision.py` uses synthetic graphs. It checks:

- the release-only protocol never exposes the test user;
- window shifting and estimation;
- degree controls and rewiring marginals;
- the edge-list universe;
- the held-out featurization of the utility metric.

`tests/test_smoke.py` runs end-to-end checks on BitcoinOTC and skips them when the data is not prepared.

---

## Notes and caveats

- **Saved records versus caches.** The repository holds every per-target JSON record, which is enough to
  regenerate the paper. Two analyses read the large `.npz` caches that the runners write: the pool-size
  analysis in `analyze_revision.py`, and the Section 6.4 analyses. Their outputs are saved, so rerun them
  only after rerunning the experiments.
- **Dated Epinions IDs** were assigned when the dataset was built, in order of first rating. Its ID–time
  correlation therefore says nothing about the platform's pseudonyms. BitcoinOTC and BitcoinAlpha keep
  their SNAP IDs.
- **Static Epinions** has no timestamps. Its `time` column is the record order, which largely follows rater
  IDs, so its timing features read file order (Appendix E).
- **The hub window** depends on the target only through the deleted events, and in practice it is the same
  for every target of a dataset.
- **The GNN baseline** is a deterministic variant of DGTEN, without its uncertainty modeling. It is trained
  on the first seven of ten time snapshots of each Bitcoin graph, with input features smoothed over the whole
  graph.
- **Binarization.** Bitcoin ratings are never zero, so the thresholds τ = −1 and τ = 0 give identical signs.
