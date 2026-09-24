# CLAUDE.md — thesis_arima_energy_benchmark

## Project overview

KU Leuven master's thesis. Energy-aware walk-forward benchmark for Belgian electricity load forecasting (Elia 15-min dataset). Compares four model families on accuracy **and** energy/carbon cost (CodeCarbon), following the thesis methodology.

## Repository structure

```
# Entry points (run from project root)
run_benchmark.py        # Classical models walk-forward (naive, linear, lgbm, arima, sarima)
run_tsfm_benchmark.py   # Foundation models walk-forward (chronos_*, timesfm, moirai_*, lag_llama)
tune_hyperparams.py     # Optuna tuning for Ridge and LightGBM
prepare_dataset.py      # One-time dataset preparation / resampling (Elia 15-min)
prepare_etth1.py        # Fetch ETTh1 (hourly oil temperature) from HuggingFace
visualize_results.py    # Publication-quality prediction plots (uses MODEL_COLORS palette)
plot_energy.py          # Energy / CO₂ / per-prediction efficiency plots
compare_datasets.py     # Side-by-side comparison of two benchmark runs (Elia vs ETTh1)

# Library code
src/
  data.py               # load_timeseries, clean_timeseries, split_last_n_years
  models.py             # NaivePredictor, ARIMAPredictor, MLPredictor, DirectForecaster, factories
  models_tsfm.py        # ChronosForecaster, ChronosBoltForecaster, TimesFMForecaster,
                        # LagLlamaForecaster, MoiraiForecaster (+ Base/Large variants), TSFM_MODELS registry
  metrics.py            # mae, mape, rmse, EnergyMeter (CodeCarbon), CarbonTrackerMeter, NvidiaSmiMeter
  evaluate.py           # Walk-forward loop helpers

# Helper / orchestration scripts
scripts/
  run_energy_pass.sh    # Sequential classical-models run with --energy_tool=both (clean CC+CT attribution)
  run_energy_fm_all.sh  # Sequential FM run with --energy_tool=all (CC + CT + nvidia_smi)
  setup_fm_env.sh       # One-time foundation-model env setup

# Data and outputs
config/
  config_example.json   # Reference config (data paths, split ratios, ARIMA orders)
data/                   # Input CSV/Parquet (not committed — large files)
results/
  metrics_runs.csv      # Append-only run log (accuracy + energy per model/horizon)
  tuned_params.json     # Best hyperparameters from tune_hyperparams.py
  <run_id>/metrics.csv  # Per-run metrics (run_id = wf_TS for classical, tsfm_TS for FMs)
  <run_id>/predictions.parquet  # Long-form predictions
  <run_id>/{codecarbon,carbontracker,nvidia_smi}/  # Raw energy-tracker outputs
figures/                # Generated plot outputs (per-run subdirs)
logs/                   # Per-launch log files
```

## Running the benchmark

```bash
# Activate venv first
source .venv/bin/activate

# All models
python run_benchmark.py --models naive linear lgbm arima sarima

# Single model, no energy tracking
python run_benchmark.py --models naive --energy_tool none

# Skip plots (faster / headless)
python run_benchmark.py --models linear --no_plots

# Run on ETTh1 (hourly oil temperature)
python prepare_etth1.py
python run_benchmark.py --data data/processed/etth1.csv \
    --timestamp_col datetime --target_col ot --freq h \
    --test_years 0.25 --stride 24 --horizons 1 24 168 \
    --naive_s 24 --models naive linear lgbm arima

# Tune ARIMA orders
python tune_hyperparams.py
```

## Models

| Key | Class | Description |
|-----|-------|-------------|
| `naive` | `NaivePredictor` | Seasonal naive: ŷ[t+h] = y[t+h−s], s=96 (daily). Zero-cost baseline. |
| `arima` | `ARIMAPredictor` | ARIMA(2,1,2) — Kalman-filter state updates, no re-fitting per origin. |
| `sarima` | `ARIMAPredictor` | SARIMA(2,1,2)(1,1,1,96) — adds daily seasonality. |
| `linear` | `MLPredictor` | Ridge regression, direct multi-step, lag + cyclic calendar features. |
| `lgbm` | `MLPredictor` | LightGBM, same direct multi-step approach. |

All models expose the same `fit` / `set_series` / `predict` interface so the walk-forward loop in `run_benchmark.py` treats them uniformly.

## Key design decisions

- **Walk-forward (rolling-origin) evaluation**: model is fit once on train, then Kalman-filter `.append(refit=False)` advances the state at each origin (~365× faster than re-fitting).
- **Direct multi-step**: ML models train one estimator per horizon `h` — avoids recursive error accumulation.
- **`max_train_obs=4000`** cap on ARIMA/SARIMA fitting to keep wall-clock time tractable on a laptop.
- **Horizons**: h=4 (1 h ahead) and h=96 (24 h ahead) at 15-min resolution.
- **Energy tracking**: `EnergyMeter` in `src/metrics.py` wraps CodeCarbon; disabled with `--energy_tool none`.
- **Results**: every run appends a row to `results/metrics_runs.csv` (never overwritten).

## Datasets

The benchmark is dataset-agnostic — any CSV/parquet with a timestamp column and a
single target column works. Two datasets are wired up out of the box:

### Elia (primary, 15-min electricity load)

- Source: Elia Belgian electricity load, 15-minute resolution (~5 years).
- Path: `data/processed/elia_load_15min.csv` (columns: `datetime`, `totalload`).
- Frequency: `15min`. `--naive_s 96`, default horizons `[1, 4, 96]`.
- Build: `python prepare_dataset.py`.

### ETTh1 (secondary, hourly oil temperature)

- Source: ETT-small (Zhou et al.) — fetched from the HuggingFace `ett`/`h1`
  config; falls back to GitHub raw if HF is unreachable.
- Path: `data/processed/etth1.csv` (columns: `datetime`, `ot`). 14,400 hours
  (≈ 1.6 years), 2016-07-01 → 2018-02-20.
- Frequency: `h`. `--naive_s 24`, horizons `[1, 24, 168]` (1h / 1d / 1w),
  stride 24, `test_years 0.25` (≈ 91-day test window).
- Build: `python prepare_etth1.py`.

### Cross-dataset comparison

`compare_datasets.py` takes one or more `metrics.csv` folders per dataset and
produces side-by-side accuracy + energy plots, pairing horizons by their
duration in minutes. Example:

```bash
python compare_datasets.py \
  --dataset_a Elia  results/wf_energy_full_20260512 \
  --dataset_b ETTh1 results/wf_20260514_171100 results/tsfm_20260514_171217
```

## Dependencies

```
numpy, pandas, matplotlib, scikit-learn
statsmodels          # SARIMAX
codecarbon           # energy / carbon measurement
pyarrow              # parquet IO (optional)
tqdm                 # progress bars (optional)
lightgbm             # optional, needed for --models lgbm
holidays             # optional, needed for --use_holidays flag
```

Install: `pip install -r requirements.txt`

## System resource limits (heimdall)

**Never oversubscribe memory.** The host has 256 GB RAM shared with other users. A previous parallel run (20 SARIMA workers × ~12 GB state-space each) pushed total usage past 240 GB, caused swap thrashing, and prompted a complaint from the sysadmin (Rik) threatening forceful termination.

Rules:
- Always check `free -g` before launching parallel work and budget per-process RSS conservatively. Combined RSS must stay well under 256 GB with headroom for other users.
- **SARIMA / SARIMAX walk-forward: single-worker only.** Do not spawn `mp.Pool` of SARIMAX fits — each results object can hold many GB of Kalman state at s=96 seasonality.
- 4 classical CPU models (naive/linear/lgbm/arima) in parallel is fine — each is < 2 GB.
- 4 foundation models on GPU in parallel is fine — each is only ~2 GB host RAM; the GPU multiplexes the compute.
- If a heavy job dies unexpectedly, suspect OOM / swap kill before debugging the code. Check `free -g`, `dmesg | grep -i oom`, and whether `/proc/$PID` still exists.

## Development notes

- Python 3.14 (`.venv` uses cpython 3.14).
- `enforce_stationarity=False` and `enforce_invertibility=False` are intentional — safe after differencing, avoids constrained optimisation issues.
- SARIMA fitting suppresses all `statsmodels` warnings via `warnings.catch_warnings` — this is intentional.
- Do not refactor the `fit` / `set_series` / `predict` interface without updating all model classes and the walk-forward loop.
- `results/metrics_runs.csv` is append-only by design — do not truncate or rewrite it.
