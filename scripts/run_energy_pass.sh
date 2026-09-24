#!/usr/bin/env bash
# Sequential re-run of all 8 best-version models with full energy tracking
# (codecarbon + carbontracker). Sequential matters: codecarbon attributes
# system-wide energy to the process, so parallel runs would muddle attribution.
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

export HF_HOME=/volume1/no_backup/r1062653/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PROJECT_DIR/lag-llama:${PYTHONPATH:-}"

PY=/volume1/no_backup/r1062653/thesis_fm_env/bin/python
TS=$(date -u +%Y%m%d_%H%M%S)
ROOT="$PROJECT_DIR/results/energy_pass_${TS}"
mkdir -p "$ROOT" logs/energy_pass
LOG="logs/energy_pass/${TS}.log"
echo "Energy-pass tag: $TS"  | tee -a "$LOG"
echo "Results root: $ROOT"   | tee -a "$LOG"

run_one() {
    local model="$1" kind="$2"
    local rdir="$ROOT/$model"
    mkdir -p "$rdir"
    echo "" | tee -a "$LOG"
    echo "[$(date -u +%H:%M:%S)] >>> $model ($kind)" | tee -a "$LOG"
    if [ "$kind" = "classical" ]; then
        local tuned=""
        if [ "$model" = "linear" ] || [ "$model" = "lgbm" ]; then
            tuned="--tuned_params results/tuned_params.json"
        fi
        $PY -u run_benchmark.py --models "$model" --horizons 1 4 96 --stride 24 \
            $tuned --energy_tool both --results_dir "$rdir" --no_plots \
            2>&1 | tee -a "$LOG"
    else
        $PY -u run_tsfm_benchmark.py --models "$model" \
            --horizons 1 4 96 --test_years 1.0 --stride 24 \
            --num_samples 100 --device cuda --gpu_index 0 \
            --energy_tool both --results_dir "$rdir" --no_plots \
            2>&1 | tee -a "$LOG"
    fi
    echo "[$(date -u +%H:%M:%S)] <<< $model done" | tee -a "$LOG"
}

# Classical first (CPU-only, fast)
for m in naive linear lgbm arima; do
    run_one "$m" classical
done
# FMs second (GPU)
for m in chronos_bolt_mini timesfm_200m moirai_small lag_llama; do
    run_one "$m" fm
done

echo "" | tee -a "$LOG"
echo "[$(date -u +%H:%M:%S)] ALL DONE  ROOT=$ROOT" | tee -a "$LOG"
echo "$TS"   > logs/energy_pass/last_tag.txt
echo "$ROOT" > logs/energy_pass/last_root.txt
