#!/usr/bin/env bash
# Sequential run of the four LARGER FM variants on GPU.
#
# One model at a time — strictly sequential — per sysadmin OOM warning.
# Uses the *safe-optimised* defaults baked into the model classes:
#   - Chronos T5  : num_samples=50  (class default; median of 50 ≈ median of 100)
#   - TimesFM 500M: context_length=1024 (class default; ~2× faster than 2048)
#   - Batched walk-forward at batch_size=16 (Blackwell-friendly, fits all FMs).
# Stride 2 = origin every 30 min over the 1-year test window (~17.5k origins).
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export HF_HOME="${HF_HOME:-/volume1/no_backup/r1062653/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PROJECT_DIR/lag-llama:${PYTHONPATH:-}"

PY="${BENCH_PY:-/volume1/no_backup/r1062653/thesis_fm_env/bin/python}"
TS=$(date -u +%Y%m%d_%H%M%S)
ROOT="${BENCH_RESULTS:-$PROJECT_DIR/results}/fm_large_${TS}"
mkdir -p "$ROOT" logs/fm_large
LOG="logs/fm_large/${TS}.log"
echo "tag=$TS  root=$ROOT" | tee -a "$LOG"

# Order: smallest/fastest first so the pipeline is verified before the long runs.
for m in chronos_bolt_base moirai_large chronos_large timesfm_500m; do
    rdir="$ROOT/$m"
    mkdir -p "$rdir"
    echo "" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] >>> $m" | tee -a "$LOG"
    free -g | head -2 | tee -a "$LOG"
    $PY -u run_tsfm_benchmark.py --models "$m" \
        --horizons 1 4 96 --test_years 1.0 --stride 2 \
        --batch_size 16 --device cuda --gpu_index 0 \
        --energy_tool all --results_dir "$rdir" --no_plots \
        2>&1 | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] <<< $m done" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "[$(date -u +%H:%M:%S)] ALL DONE  ROOT=$ROOT" | tee -a "$LOG"
echo "$TS"   > logs/fm_large/last_tag.txt
echo "$ROOT" > logs/fm_large/last_root.txt
