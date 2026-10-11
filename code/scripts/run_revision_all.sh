#!/usr/bin/env bash
# All experiment batches, in priority order (see run_revision.py).
# Candidate universe: users still present in the release (edge-list release, the default).
# Logs go to paper/results/revision/logs/. Finished units are skipped on rerun.
set -euo pipefail
cd "$(dirname "$0")/../.."

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -z "${CHRONOTRACE_DATA_DIR:-}" ]; then
  if [ -f data/bitcoinotc.csv ]; then CHRONOTRACE_DATA_DIR="$(pwd)/data"; else CHRONOTRACE_DATA_DIR="$(cd .. && pwd)/data/cyberdata"; fi
fi
export CHRONOTRACE_DATA_DIR
PY="${PY:-python}"
W="${WORKERS:-20}"
U="${UNIVERSE:-edgelist}"
LOG=paper/results/revision/logs
mkdir -p "$LOG"

run() {
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name"
  "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] end   $name: $(tail -1 "$LOG/$name.log")"
  if grep -q 'ERROR' "$LOG/$name.log"; then echo "  !! errors in $name"; fi
}
R="code/scripts/run_revision.py --universe $U --workers $W"

# 1. Real graphs, main targets: both protocols, every window mode.
run "$U-tasks_real_main"  $R --stage tasks  --datasets bitcoinotc,bitcoinalpha,epinions_dated
run "$U-attack_real_main" $R --stage attack --datasets bitcoinotc,bitcoinalpha,epinions_dated

# 2. Rewired reference graphs: release protocol, oracle window.
G_BTC=config0,config1,config2,config3,config4,sigtemp0,sigtemp1,sigtemp2,sigtemp3,sigtemp4
run "$U-tasks_ref_btc"  $R --stage tasks  --datasets bitcoinotc,bitcoinalpha --graphs "$G_BTC"
run "$U-attack_ref_btc" $R --stage attack --datasets bitcoinotc,bitcoinalpha --graphs "$G_BTC" --protocols release --modes oracle
G_EPI=config0,config1,config2,sigtemp0,sigtemp1,sigtemp2
run "$U-tasks_ref_epi"  $R --stage tasks  --datasets epinions_dated --graphs "$G_EPI"
run "$U-attack_ref_epi" $R --stage attack --datasets epinions_dated --graphs "$G_EPI" --protocols release --modes oracle

# 3. Low-degree targets (incident degree 1-3), trained on the same seed's main targets.
run "$U-tasks_low"  $R --stage tasks  --datasets bitcoinotc,bitcoinalpha,epinions_dated --targetsets low
run "$U-attack_low" $R --stage attack --datasets bitcoinotc,bitcoinalpha,epinions_dated --targetsets low --modes oracle,cohort

# 4. Defense stress tests (ChronoScrub, randomized response) under the release-only protocol.
run "$U-defense" code/scripts/run_defense_revision.py --universe "$U" --workers "$W" --datasets bitcoinotc,bitcoinalpha,epinions_dated

# 5. Static SNAP Epinions (as submitted, proxy record-order time) on its full candidate universe.
run "$U-tasks_epi_static"  $R --stage tasks  --datasets epinions
run "$U-attack_epi_static" $R --stage attack --datasets epinions --modes oracle

echo "[$(date +%H:%M:%S)] all batches finished"
