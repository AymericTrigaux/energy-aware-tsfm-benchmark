#!/usr/bin/env bash
# Sequential re-run of the 4 FMs with --energy_tool=all (CC + CT + nvidia_smi).
# Sequential so codecarbon attributes system energy correctly and nvidia_smi
# polling isn't shared with another GPU job.
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

export HF_HOME="${HF_HOME:-/volume1/no_backup/r1062653/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PROJECT_DIR/lag-llama:${PYTHONPATH:-}"

PY="${BENCH_PY:-/volume1/no_backup/r1062653/thesis_fm_env/bin/python}"
TS=$(date -u +%Y%m%d_%H%M%S)
ROOT="${BENCH_RESULTS:-$PROJECT_DIR/results}/energy_fm_all_${TS}"
mkdir -p "$ROOT" logs/energy_fm_all
LOG="logs/energy_fm_all/${TS}.log"
echo "tag=$TS  root=$ROOT" | tee -a "$LOG"

for m in chronos_bolt_mini timesfm_200m moirai_small lag_llama; do
    rdir="$ROOT/$m"
    mkdir -p "$rdir"
    echo "" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] >>> $m" | tee -a "$LOG"
    $PY -u run_tsfm_benchmark.py --models "$m" \
        --horizons 1 4 96 --test_years 1.0 --stride 24 \
        --num_samples 100 --device cuda --gpu_index 0 \
        --energy_tool all --results_dir "$rdir" --no_plots \
        2>&1 | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] <<< $m done" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "[$(date -u +%H:%M:%S)] ALL DONE  ROOT=$ROOT" | tee -a "$LOG"
echo "$TS"   > logs/energy_fm_all/last_tag.txt
echo "$ROOT" > logs/energy_fm_all/last_root.txt
