#!/usr/bin/env bash
# Sequential stride=2 run for every model NOT already in fm_large_stride2_*.
# 11 models total = 4 classical + 6 FM small/mini + 1 new Moirai-2.0.
#
# Sequential is required so codecarbon attributes system-wide energy cleanly
# (parallel runs muddle attribution — see scripts/run_energy_pass.sh comment).
# Classical models use --energy_tool both (CC+CT; NV is 0 for CPU-only runs
# and not exposed by run_benchmark.py).
# FM models use --energy_tool all (CC+CT+NV).
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export HF_HOME=/volume1/no_backup/r1062653/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PROJECT_DIR/lag-llama:${PYTHONPATH:-}"

PY=/volume1/no_backup/r1062653/thesis_fm_env/bin/python
TS=$(date -u +%Y%m%d_%H%M%S)
ROOT="$PROJECT_DIR/results/stride2_remaining_${TS}"
mkdir -p "$ROOT" logs/stride2_remaining
LOG="logs/stride2_remaining/${TS}.log"
echo "tag=$TS  root=$ROOT" | tee -a "$LOG"

# ── Classical (4): CPU, --energy_tool both ──────────────────────────────────
for m in naive linear lgbm arima; do
    rdir="$ROOT/$m"
    mkdir -p "$rdir"
    echo "" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] >>> $m (classical)" | tee -a "$LOG"
    free -g | head -2 | tee -a "$LOG"
    tuned=""
    if [ "$m" = "linear" ] || [ "$m" = "lgbm" ]; then
        tuned="--tuned_params results/tuned_params.json"
    fi
    $PY -u run_benchmark.py --models "$m" --horizons 1 4 96 --stride 2 \
        $tuned --energy_tool both --results_dir "$rdir" --no_plots \
        2>&1 | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] <<< $m done" | tee -a "$LOG"
done

# ── FM small/mini + Moirai-2 (7): GPU, --energy_tool all ────────────────────
# Order: cheapest first so the slow ones run last (lag_llama is ~10-15 min).
for m in chronos_bolt_mini moirai2_small moirai_small moirai_base chronos_mini timesfm_200m lag_llama; do
    rdir="$ROOT/$m"
    mkdir -p "$rdir"
    echo "" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] >>> $m (FM)" | tee -a "$LOG"
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
echo "$TS"   > logs/stride2_remaining/last_tag.txt
echo "$ROOT" > logs/stride2_remaining/last_root.txt
