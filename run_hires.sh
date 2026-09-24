#!/usr/bin/env bash
# Plan B: 10 Hz (power.draw.instant) power profiling for the thesis benchmark.
# Runs 8 models with --nvsmi_interval_ms 100, rebuilds thesis figures, then
# sanity-checks sample density and lag_llama peak power against the cmp2 (1 Hz) run.
set -uo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
PY="${BENCH_PY:-/volume1/no_backup/r1062653/thesis_fm_env/bin/python}"
LOG="$PROJ/logs/hires_run.log"

cd "$PROJ"
export HF_HOME="${HF_HOME:-/volume1/no_backup/r1062653/hf_cache}"

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

{
  ts "=== Plan B hi-res (10 Hz, power.draw.instant) run started ==="
  ts "python: $PY"
  ts "tag=hires  --nvsmi_interval_ms 100  --cc_poll_interval_s 1.0"

  # model:n_origins, in the required order (classical first, then foundation models)
  models=(
    "naive:1457"
    "linear:1457"
    "lgbm:1457"
    "arima:800"
    "chronos_bolt_mini:1200"
    "timesfm_200m:700"
    "moirai_small:500"
    "lag_llama:60"
  )

  for entry in "${models[@]}"; do
    m="${entry%%:*}"
    n="${entry##*:}"
    ts ">>> START measure: $m (n_origins=$n)"
    "$PY" power_curves.py measure --model "$m" --n_origins "$n" \
      --nvsmi_interval_ms 100 --cc_poll_interval_s 1.0 --tag hires
    rc=$?
    ts "<<< DONE measure: $m (exit=$rc)"
  done

  ts "=== All 8 models finished. Rebuilding thesis figures ==="
  ts ">>> composite -> figures/power_curves_composite.png"
  "$PY" power_curves.py composite --tag hires --out figures/power_curves_composite.png
  ts "<<< composite (exit=$?)"
  ts ">>> analyze --tag hires"
  "$PY" power_curves.py analyze --tag hires
  ts "<<< analyze (exit=$?)"

  ts "=== Sanity checks (hires vs cmp2) ==="
  "$PY" - <<'PY'
import glob, os, csv

def newest(pattern):
    fs = glob.glob(pattern)
    return max(fs, key=os.path.getmtime) if fs else None

def count_rows(f):
    with open(f) as fh:
        return sum(1 for _ in fh) - 1  # drop header

def max_watt(f):
    mx = float("-inf")
    with open(f) as fh:
        for row in csv.DictReader(fh):
            try:
                mx = max(mx, float(row["power_w"]))
            except (ValueError, KeyError):
                pass
    return mx

print("--- Sample row counts: hires should be ~10x cmp2 (10 Hz vs 1 Hz) ---")
for m in ("lag_llama", "chronos_bolt_mini"):
    h = newest(f"results/powercurve_{m}_*_hires/nvsmi_power_log.csv")
    c = newest(f"results/powercurve_{m}_*_cmp2/nvsmi_power_log.csv")
    if not h or not c:
        print(f"{m}: MISSING (hires={h}, cmp2={c}) -> FAIL")
        continue
    hr, cr = count_rows(h), count_rows(c)
    ratio = (hr / cr) if cr else float("inf")
    ok = ratio >= 5  # lenient band around the expected ~10x density increase
    print(f"{m:18s} hires={hr:5d} rows  cmp2={cr:5d} rows  ratio={ratio:5.1f}x  -> "
          f"{'PASS' if ok else 'FAIL (expected ~10x)'}")

print("--- lag_llama hires peak power (cmp2 peak was ~462 W) ---")
h = newest("results/powercurve_lag_llama_*_hires/nvsmi_power_log.csv")
if not h:
    print("lag_llama hires nvsmi_power_log.csv MISSING -> cannot check peak")
else:
    peak = max_watt(h)
    old = 462.0
    if peak < 0.7 * old:
        print(f"lag_llama hires peak = {peak:.1f} W  -> WARNING: significantly lower "
              f"than cmp2 ~{old:.0f} W (instant power.draw should be >= averaged)")
    else:
        print(f"lag_llama hires peak = {peak:.1f} W  -> OK (cmp2 ~{old:.0f} W)")
PY

  ts "=== Plan B hi-res run COMPLETE ==="
} >> "$LOG" 2>&1
