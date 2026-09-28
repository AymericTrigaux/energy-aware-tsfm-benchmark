#!/usr/bin/env bash
# Repeat the stride-2 walk-forward evaluation N times per model, to measure
# run-to-run variance in energy and accuracy.
#
# Flags mirror the final May 2026 stride-2 pass (scripts/run_stride2_remaining.sh):
#   classical -> run_benchmark.py      --energy_tool both  (+ --tuned_params)
#   FM        -> run_tsfm_benchmark.py --energy_tool all   --batch_size 16 --device cuda
#
# Runs are STRICTLY SEQUENTIAL. CodeCarbon attributes system-wide energy to the
# running process, so two concurrent runs would each absorb the other's draw.
#
# Layout (run_id is generated inside the python entry points and is not a CLI
# flag, so the repeat index lives in the directory name + a sidecar JSON):
#
#   results/repeats_<TS>/<model>/rep01/{wf,tsfm}_<timestamp>/metrics.csv
#   results/repeats_<TS>/<model>/rep01/repeat_meta.json
#
# Usage:
#   scripts/run_repeats.sh                          # 10x lgbm, chronos_bolt_mini, moirai2_small
#   scripts/run_repeats.sh --repeats 3
#   scripts/run_repeats.sh --models naive --repeats 1 --stride 96 --test_years 0.05   # smoke test
#   scripts/run_repeats.sh --dry_run
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Same environment as the May 2026 runs.
export HF_HOME="${HF_HOME:-/volume1/no_backup/r1062653/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PROJECT_DIR/lag-llama:${PYTHONPATH:-}"

PY="${BENCH_PY:-/volume1/no_backup/r1062653/thesis_fm_env/bin/python}"

# --- defaults (the task-A cohort) ---
MODELS=(lgbm chronos_bolt_mini moirai2_small)
REPEATS=10
STRIDE=2
TEST_YEARS=1.0
HORIZONS="1 4 96"
TAG=""
DRY_RUN=0
SKIP_IDLE_CHECK=0
BMC_BASELINE_S=60
# Cool-down between the end of a run and its after-baseline, and the pause
# between consecutive jobs, so the next before-window is settled.
CHASSIS_COOLDOWN_S=120
JOB_PAUSE_S=120
SKIP_SWEEP=0

# The nine remaining foundation-model variants, each run once with --bmc so that
# every model in the thesis ends up with one BMC-measured whole-chassis figure.
SWEEP_MODELS=(chronos_mini chronos_large chronos_bolt_base
              timesfm_200m timesfm_500m
              moirai_small moirai_base moirai_large lag_llama)

# May 2026 stride-2 eval wall clocks (s), from MODELS_STRIDE2_REPORT.md §"Energy
# ranking". Used only to estimate total wall time up front.
may_eval_s() {
    case "$1" in
        naive)             echo 5.2 ;;
        linear)            echo 23.5 ;;
        lgbm)              echo 26.6 ;;
        arima)             echo 876.8 ;;
        chronos_bolt_mini) echo 16.7 ;;
        moirai2_small)     echo 19.8 ;;
        moirai_small)      echo 96.5 ;;
        chronos_bolt_base) echo 52.5 ;;
        moirai_base)       echo 146.9 ;;
        timesfm_200m)      echo 328.6 ;;
        moirai_large)      echo 262.1 ;;
        timesfm_500m)      echo 1479.9 ;;
        chronos_mini)      echo 484.1 ;;
        lag_llama)         echo 3794.1 ;;
        chronos_large)     echo 4767.9 ;;
        *)                 echo 60 ;;
    esac
}
# Rough fit / weight-load overhead per run (s), on top of eval.
setup_s() { case "$1" in naive|linear) echo 5 ;; lgbm) echo 30 ;; arima) echo 10 ;; *) echo 25 ;; esac; }

usage() {
    sed -n '2,26p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --models)     shift; MODELS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --repeats)    REPEATS="$2";    shift 2 ;;
        --stride)     STRIDE="$2";     shift 2 ;;
        --test_years) TEST_YEARS="$2"; shift 2 ;;
        --horizons)   shift; HORIZONS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do HORIZONS="$HORIZONS $1"; shift; done; HORIZONS="${HORIZONS# }" ;;
        --tag)        TAG="$2";        shift 2 ;;
        --dry_run)    DRY_RUN=1;       shift ;;
        --skip_idle_check) SKIP_IDLE_CHECK=1; shift ;;
        --bmc_baseline_s)  BMC_BASELINE_S="$2"; shift 2 ;;
        --skip_sweep)      SKIP_SWEEP=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown flag: $1" >&2; usage 1 ;;
    esac
done

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "error: --models given with no values" >&2
    exit 1
fi

# Model families, mirroring the --models choices of the two entry points.
is_classical() {
    case "$1" in
        naive|linear|lgbm|arima|sarima) return 0 ;;
        *) return 1 ;;
    esac
}

# --- wall-time estimate, from the May 2026 stride-2 wall clocks ---
# Every run carries --bmc, which adds 2 x BMC_BASELINE_S of idle sampling.
estimate_total_s() {
    local total=0 m i n
    for m in "${MODELS[@]}"; do
        n=$(awk -v e="$(may_eval_s "$m")" -v s="$(setup_s "$m")" -v b="$BMC_BASELINE_S" \
                -v c="$CHASSIS_COOLDOWN_S" -v p="$JOB_PAUSE_S" \
                -v r="$REPEATS" 'BEGIN{printf "%.0f", r*(e+s+2*b+c+p)}')
        total=$((total + n))
    done
    if [ "$SKIP_SWEEP" -eq 0 ]; then
        for m in "${SWEEP_MODELS[@]}"; do
            n=$(awk -v e="$(may_eval_s "$m")" -v s="$(setup_s "$m")" -v b="$BMC_BASELINE_S" \
                    -v c="$CHASSIS_COOLDOWN_S" -v p="$JOB_PAUSE_S" \
                    'BEGIN{printf "%.0f", e+s+2*b+c+p}')
            total=$((total + n))
        done
    fi
    echo "$total"
}

print_estimate() {
    local m n sub total
    echo ""
    echo "Estimated wall time (from May 2026 stride-2 eval wall clocks;"
    echo "each run adds 2 x ${BMC_BASELINE_S}s of baseline sampling, ${CHASSIS_COOLDOWN_S}s of"
    echo "cool-down and a ${JOB_PAUSE_S}s pause before the next job):"
    printf "  %-20s %8s %6s %12s\n" MODEL "MAY_EVAL" "RUNS" "EST"
    printf "  %s\n" "------------------------------------------------------"
    for m in "${MODELS[@]}"; do
        sub=$(awk -v e="$(may_eval_s "$m")" -v s="$(setup_s "$m")" -v b="$BMC_BASELINE_S" \
                  -v c="$CHASSIS_COOLDOWN_S" -v p="$JOB_PAUSE_S" \
                  -v r="$REPEATS" 'BEGIN{printf "%.0f", r*(e+s+2*b+c+p)}')
        printf "  %-20s %7ss %6s %10.1f m\n" "$m" "$(may_eval_s "$m")" "$REPEATS" \
               "$(awk -v v="$sub" 'BEGIN{print v/60}')"
    done
    if [ "$SKIP_SWEEP" -eq 0 ]; then
        printf "  %s\n" "--- BMC coverage sweep (1 run each) -------------------"
        for m in "${SWEEP_MODELS[@]}"; do
            sub=$(awk -v e="$(may_eval_s "$m")" -v s="$(setup_s "$m")" -v b="$BMC_BASELINE_S" \
                      -v c="$CHASSIS_COOLDOWN_S" -v p="$JOB_PAUSE_S" \
                      'BEGIN{printf "%.0f", e+s+2*b+c+p}')
            printf "  %-20s %7ss %6s %10.1f m\n" "$m" "$(may_eval_s "$m")" "1" \
                   "$(awk -v v="$sub" 'BEGIN{print v/60}')"
        done
    fi
    total=$(estimate_total_s)
    printf "  %s\n" "------------------------------------------------------"
    awk -v t="$total" 'BEGIN{printf "  TOTAL ESTIMATE: %.0f s = %.1f min = %.2f h\n", t, t/60, t/3600}'
    echo "  (excludes queueing behind other tenants; lag_llama and chronos_large dominate)"
    echo ""
}

print_estimate

# --- pre-flight: refuse to measure energy on a busy shared host ---
# Skipped for --dry_run (nothing is measured) and overridable with
# --skip_idle_check for a deliberate run on a non-idle machine.
if [ "$DRY_RUN" -eq 0 ] && [ "$SKIP_IDLE_CHECK" -eq 0 ]; then
    echo "Running host idle check before starting repeats …"
    if ! "$PROJECT_DIR/scripts/host_idle_check.sh"; then
        echo ""
        echo "ABORTING: host is not idle, so energy measurements would be" >&2
        echo "contaminated by other tenants. Re-run when quiet, or pass" >&2
        echo "--skip_idle_check to override deliberately." >&2
        exit 1
    fi
    echo ""
fi

TS=$(date -u +%Y%m%d_%H%M%S)
SUFFIX=""
[ -n "$TAG" ] && SUFFIX="_${TAG}"
ROOT="${BENCH_RESULTS:-$PROJECT_DIR/results}/repeats_${TS}${SUFFIX}"
mkdir -p "$ROOT" logs/repeats
LOG="logs/repeats/${TS}${SUFFIX}.log"

{
    echo "tag=${TS}${SUFFIX}"
    echo "root=$ROOT"
    echo "models=${MODELS[*]}  repeats=$REPEATS  stride=$STRIDE  test_years=$TEST_YEARS  horizons=$HORIZONS"
    echo "bmc=on  bmc_baseline_s=$BMC_BASELINE_S"
    echo "chassis_cooldown_s=$CHASSIS_COOLDOWN_S  job_pause_s=$JOB_PAUSE_S"
    if [ "$SKIP_SWEEP" -eq 0 ]; then
        echo "bmc_sweep=${SWEEP_MODELS[*]}"
    else
        echo "bmc_sweep=disabled (--skip_sweep)"
    fi
    echo "python=$PY"
    [ "$DRY_RUN" -eq 1 ] && echo "DRY RUN - commands are printed, not executed"
} | tee -a "$LOG"

# Build the job list: the repeat cohort first, then one BMC coverage run for
# each remaining foundation-model variant. One loop serves both so the two
# phases cannot drift apart.
JOB_MODEL=(); JOB_IDX=(); JOB_TOTAL=(); JOB_PHASE=()
for m in "${MODELS[@]}"; do
    for i in $(seq 1 "$REPEATS"); do
        JOB_MODEL+=("$m"); JOB_IDX+=("$i"); JOB_TOTAL+=("$REPEATS"); JOB_PHASE+=("repeat")
    done
done
if [ "$SKIP_SWEEP" -eq 0 ]; then
    for m in "${SWEEP_MODELS[@]}"; do
        JOB_MODEL+=("$m"); JOB_IDX+=(1); JOB_TOTAL+=(1); JOB_PHASE+=("bmc_sweep")
    done
fi

total=${#JOB_MODEL[@]}
done_n=0

for j in "${!JOB_MODEL[@]}"; do
    m="${JOB_MODEL[$j]}"
    i="${JOB_IDX[$j]}"
    reps_total="${JOB_TOTAL[$j]}"
    phase="${JOB_PHASE[$j]}"
    if is_classical "$m"; then kind=classical; else kind=fm; fi

    {
        rep=$(printf "rep%02d" "$i")
        rdir="$ROOT/$m/$rep"
        mkdir -p "$rdir"
        done_n=$(( done_n + 1 ))

        echo "" | tee -a "$LOG"
        echo "[$(date -u +%H:%M:%S)] >>> $m $rep ($kind, $phase)  [$done_n/$total]" | tee -a "$LOG"
        free -g | head -2 | tee -a "$LOG"

        t_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        t0=$(date +%s)

        if [ "$kind" = "classical" ]; then
            # Only linear/lgbm consume tuned params, same as the May 2026 pass.
            tuned=""
            if [ "$m" = "linear" ] || [ "$m" = "lgbm" ]; then
                tuned="--tuned_params results/tuned_params.json"
            fi
            # shellcheck disable=SC2086
            cmd=($PY -u run_benchmark.py --models "$m"
                 --horizons $HORIZONS --stride "$STRIDE" --test_years "$TEST_YEARS"
                 $tuned --energy_tool both --results_dir "$rdir" --no_plots
                 --bmc --bmc_baseline_s "$BMC_BASELINE_S"
                 --rapl --rapl_baseline_s "$BMC_BASELINE_S"
                 --chassis_cooldown_s "$CHASSIS_COOLDOWN_S")
        else
            # shellcheck disable=SC2086
            cmd=($PY -u run_tsfm_benchmark.py --models "$m"
                 --horizons $HORIZONS --stride "$STRIDE" --test_years "$TEST_YEARS"
                 --batch_size 16 --device cuda --gpu_index 0
                 --energy_tool all --results_dir "$rdir" --no_plots
                 --bmc --bmc_baseline_s "$BMC_BASELINE_S"
                 --rapl --rapl_baseline_s "$BMC_BASELINE_S"
                 --chassis_cooldown_s "$CHASSIS_COOLDOWN_S")
        fi

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "DRY: ${cmd[*]}" | tee -a "$LOG"
            rc=0
        else
            "${cmd[@]}" 2>&1 | tee -a "$LOG"
            rc=${PIPESTATUS[0]}
        fi

        t1=$(date +%s)
        t_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)

        # The entry point mints its own run_id; recover it from the one
        # wf_*/ or tsfm_*/ directory it just created.
        run_id=""
        for d in "$rdir"/wf_* "$rdir"/tsfm_*; do
            [ -d "$d" ] && run_id="$(basename "$d")"
        done

        # Sidecar: the repeat index cannot live in run_id, so it lives here.
        cat > "$rdir/repeat_meta.json" <<JSON
{
  "tag": "${TS}${SUFFIX}",
  "model": "$m",
  "kind": "$kind",
  "phase": "$phase",
  "repeat_index": $i,
  "repeats_total": $reps_total,
  "bmc": true,
  "bmc_baseline_s": $BMC_BASELINE_S,
  "chassis_cooldown_s": $CHASSIS_COOLDOWN_S,
  "job_pause_s": $JOB_PAUSE_S,
  "run_id": "$run_id",
  "run_dir": "$(realpath --relative-to="$PROJECT_DIR" "$rdir")",
  "stride": $STRIDE,
  "test_years": $TEST_YEARS,
  "horizons": [$(echo "$HORIZONS" | tr ' ' ',')],
  "energy_tool": "$([ "$kind" = classical ] && echo both || echo all)",
  "command": "${cmd[*]}",
  "started_utc": "$t_start",
  "ended_utc": "$t_end",
  "wall_clock_s": $(( t1 - t0 )),
  "exit_code": $rc,
  "dry_run": $DRY_RUN
}
JSON

        echo "[$(date -u +%H:%M:%S)] <<< $m $rep done (exit=$rc, $(( t1 - t0 ))s, run_id=${run_id:-none})" | tee -a "$LOG"

        # Let the chassis settle before the next job's before-window.
        if [ "$DRY_RUN" -eq 0 ] && [ "$done_n" -lt "$total" ]; then
            echo "[$(date -u +%H:%M:%S)] ... pause ${JOB_PAUSE_S}s before the next job" | tee -a "$LOG"
            sleep "$JOB_PAUSE_S"
        fi
    }
done

echo "" | tee -a "$LOG"
echo "[$(date -u +%H:%M:%S)] ALL DONE  ROOT=$ROOT" | tee -a "$LOG"
echo "Summarise with:  $PY scripts/summarise_repeats.py --root $ROOT" | tee -a "$LOG"
echo "${TS}${SUFFIX}" > logs/repeats/last_tag.txt
echo "$ROOT"          > logs/repeats/last_root.txt
