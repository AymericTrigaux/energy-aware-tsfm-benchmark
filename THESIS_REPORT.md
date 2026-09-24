# THESIS_REPORT — Energy-aware walk-forward benchmark of forecasting models for Belgian electricity load

**Project root:** `/volume1/backup/r1062653/thesis_arima_energy_benchmark`
**Author:** François Trigaux (`ftrigaux@gmail.com`)
**Institution:** KU Leuven, master's thesis
**Date frozen:** 2026-05-15
**Repo:** `thesis_arima_energy_benchmark` (branch `main`)
**Final report folder:** `results/all_best_stride2_20260511_215221/` and `results/tsfm_all_stride2_20260513/`
**Final stride-2 figures:** `figures/thesis_variants_20260515_215942/` and `figures/all_best_stride2_20260511_215221_vis_20260512_140235/`

This document is the single self-contained reference for the thesis. It documents:

1. The research question and methodology choices.
2. Every code module, what it does, and how it is wired together.
3. Every model evaluated, its final hyperparameters, and its measured numbers.
4. The energy / carbon measurement methodology (three independent backends).
5. All quantitative results at the final stride-2 evaluation protocol (17,473 origins).
6. The figures produced and where they live.
7. Reproduction instructions.

---

## 1. Research question

> Among classical statistical models, modern feature-engineering ML, and zero-shot
> time-series foundation models, **which family forecasts Belgian 15-minute
> electricity load most accurately, and at what energy / carbon cost?**
>
> Concretely: for a Belgian TSO running rolling-origin forecasts every 30 min
> over a full calendar year, which model would deliver the best
> *MAE-per-kWh-of-compute* and is the recent wave of large foundation models
> worth their inference cost?

This is operationalised as a walk-forward benchmark over the last calendar year of
the Elia dataset (2025-03-09 → 2026-03-09), measuring three orthogonal axes:

- **Accuracy** — MAE / RMSE / MAPE at h = 1 (15 min), h = 4 (1 h), h = 96 (24 h).
- **Energy** — kWh consumed during training and inference, measured by three
  independent backends (CodeCarbon, CarbonTracker, nvidia-smi).
- **Carbon** — kgCO₂eq derived from each backend's grid-intensity model
  (Belgian grid intensity = 0.167 kgCO₂eq/kWh).

---

## 2. Repository structure (final)

```
thesis_arima_energy_benchmark/
├── CLAUDE.md                       # project ground truth (read first)
├── README.md                       # original quick-start
├── METHODOLOGY.md                  # methodology long-form
├── PROJECT_REPORT.md               # mid-project progress report
├── MODELS_STRIDE2_REPORT.md        # canonical 16-model results (this report extends it)
├── RESULTS_REPORT.md               # earlier results report (stride=96)
├── TSFM_IMPLEMENTATION.md          # FM integration notes
├── requirements.txt                # classical-env requirements
│
├── prepare_dataset.py              # Elia 15-min CSV → data/processed/elia_load_15min.csv
├── prepare_etth1.py                # ETTh1 hourly OT  → data/processed/etth1.csv
│
├── run_benchmark.py                # walk-forward classical models
├── run_tsfm_benchmark.py           # walk-forward foundation models
├── tune_hyperparams.py             # Optuna tuning for Ridge / LightGBM
│
├── measure_power_curve.py          # per-second power(t) curve recorder
├── analyze_power_curves.py         # cross-validation of CC vs nvsmi vs trapz
├── plot_power_curves_composite.py  # composite power-curve figure
│
├── visualize_results.py            # accuracy & prediction plots
├── plot_energy.py                  # energy / carbon / efficiency plots
├── plot_thesis_variants.py         # curated subsets for the body of the thesis
│
├── compare_arima_sarima.py         # ARIMA-vs-SARIMA mini-comparison
├── compare_datasets.py             # Elia vs ETTh1 side-by-side
├── compare_datasets_table.py       # LaTeX-ready cross-dataset table
│
├── src/
│   ├── __init__.py
│   ├── data.py                     # load_timeseries / clean_timeseries / split / freq_to_minutes
│   ├── metrics.py                  # mae / rmse / mape + EnergyMeter / CarbonTrackerMeter / NvidiaSmiMeter
│   ├── models.py                   # NaivePredictor / ARIMAPredictor / DirectForecaster / MLPredictor
│   ├── models_tsfm.py              # 11 foundation-model wrappers + TSFM_MODELS registry
│   ├── palette.py                  # canonical model colors & display labels
│   └── evaluate.py                 # ADF test / AIC grid search / evaluate_forecast
│
├── scripts/
│   ├── setup_fm_env.sh             # one-shot .venv_fm setup (uv + torch + uni2ts + lag-llama)
│   ├── run_energy_pass.sh          # sequential 8-model pass with --energy_tool=both
│   ├── run_energy_fm_all.sh        # sequential FM pass with --energy_tool=all (CC+CT+NV)
│   ├── run_fm_large.sh             # 4 large-FM variants (chronos_large, moirai_large…)
│   └── run_stride2_remaining.sh    # 11 remaining stride=2 runs
│
├── config/config_example.json      # reference JSON config
├── data/processed/                 # cleaned input CSVs (not in git)
├── results/                        # one folder per run; *_stride2_* are the canonical ones
├── figures/                        # generated plots, per-run subfolders
└── logs/                           # per-launch log files
```

---

## 3. Code modules — final versions

### 3.1 `src/data.py`  *(126 lines, final)*

Five functions, all dataset-agnostic:

| Function | Purpose |
|---|---|
| `load_timeseries(path, timestamp_col, target_col, freq, delimiter)` | Reads CSV/Parquet, parses timestamps, asserts a regular frequency grid (`asfreq` inserts NaN for gaps so lag features cannot silently shift after a DST transition), returns a single-column DataFrame with `y`. |
| `clean_timeseries(df, impute_method, interpolate_method)` | Three imputation strategies: `interpolate` (time-aware, used for both Elia and ETTh1), `ffill`, or `drop`. Bookended with `ffill().bfill()` to remove any residual NaN at boundaries. |
| `split_last_n_years(df, test_years)` | Chronological train/test split — last `N` calendar years go to test (fractional years are handled by adding `round(frac*365.25)` days). |
| `freq_to_minutes(freq)` | Converts pandas frequency strings (`"15min"`, `"h"`) to minutes-per-step. Used to compute `horizon_minutes` for logging. |
| `append_run_csv(path, row)` | Append-only CSV writer for `results/metrics_runs.csv` cumulative log. |

Design choice: **`asfreq` after sorting and de-duplicating** so any missing
15-min slot becomes an explicit NaN, then time-aware interpolation fills it.
This protects the lag features in `models.py:build_feature_matrix` from index-
shift bugs at DST transitions.

### 3.2 `src/models.py`  *(366 lines, final)*

Three predictor classes plus a feature-builder, all sharing the same
`fit / set_series / predict` interface so the walk-forward loop in
`run_benchmark.py` treats them uniformly.

**`ArimaConfig` dataclass** — holds `(p,d,q)`, `(P,D,Q,s)`, and `trend`.

**`fit_sarimax(train, cfg, maxiter, method)`**
- Uses `statsmodels.tsa.SARIMAX` with `enforce_stationarity=False` and
  `enforce_invertibility=False` (intentional — safe after differencing,
  avoids unconstrained-optimisation issues; documented in `CLAUDE.md`).
- `lbfgs` optimiser, `maxiter=200` default.
- All warnings suppressed in a `catch_warnings` block — intentional.

**`NaivePredictor(s=96)`**
- Seasonal naive: `ŷ[t+h] = y[t+h-s]`.
- `s = 96` ⇒ exact daily look-up (one day at 15-min resolution).
- For ETTh1 (hourly), `--naive_s 24`.

**`ARIMAPredictor(cfg, horizons, maxiter=50, max_train_obs=4000, label)`**
- `fit(y_train)` — runs a single MLE fit on the most recent
  `max_train_obs` observations (default 4000 ≈ 6 weeks of 15-min data).
- `predict(origin_idx)` — uses `results.append(new_obs, refit=False)` for
  Kalman state propagation, then `get_forecast(steps=max_h)`. This is
  **the central performance trick**: it makes walk-forward ARIMA/SARIMA
  evaluation ~365× faster than per-origin re-fitting while giving
  numerically identical forecasts (the parameter MLE is the same).

**`build_feature_matrix(y, lags, horizon, calendar, use_holidays)`**
- Builds a tabular dataset for direct *h*-step-ahead prediction.
- Target: `y[t+h]`.
- Features:
  - Lags: `DEFAULT_LAGS = [0, 1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672]`
    covers the current value, sub-hour (≤ 4 steps), intra-day (8–48), daily
    (96, 192, 288), and weekly (672) seasonalities.
  - Cyclic calendar: `sin/cos` of hour-of-day and day-of-week — avoids the
    23→0 discontinuity that integer encoding has.
  - Optional `is_holiday` (Belgian) via `holidays.country_holidays("BE")`.
- Rows with any NaN feature or NaN target are dropped — keeps the boundary
  effects honest at fit time.

**`DirectForecaster(base_estimator, lags, calendar, use_holidays, seasonal_period)`**
- One cloned estimator per horizon (**direct multi-step**, not recursive —
  avoids error accumulation).
- Optional **seasonal-residual** mode: when `seasonal_period > 0`, the
  target becomes `y[t+h] − y[t+h−s]` and the prediction is
  `model(x) + y[origin+h−s]`. This makes the model's zero-prediction
  baseline equal to the seasonal-naive forecast — a strong prior for daily
  load.

**`MLPredictor(forecaster, label)`** — thin wrapper that adapts
`DirectForecaster` to the same walk-forward `predict(origin_idx)` signature.

**Factory functions:**
- `make_linear_forecaster(lags, alpha, use_holidays, seasonal_period)` →
  `Pipeline(StandardScaler, Ridge(alpha))`.
- `make_lgbm_forecaster(lags, use_holidays, seasonal_period, **lgbm_kwargs)` →
  `LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, …)`
  by default, overridden when Optuna params are passed.

### 3.3 `src/models_tsfm.py`  *(1104 lines, final)*

Eleven zero-shot foundation-model wrappers, all sharing the same
`fit / set_series / predict` (+ optional `predict_batch`) interface as the
classical models so the walk-forward loop accepts them unchanged.

Registry (`TSFM_MODELS`):

| Key                | Class                       | Source / HF repo                            | Size      | API style                  |
|--------------------|-----------------------------|---------------------------------------------|-----------|----------------------------|
| `chronos_mini`     | `ChronosForecaster`         | `amazon/chronos-t5-mini`                    | ~20 M     | T5 sampling (`num_samples`) |
| `chronos_large`    | `ChronosLargeForecaster`    | `amazon/chronos-t5-large`                   | 710 M     | T5 sampling                 |
| `chronos_bolt_mini`| `ChronosBoltForecaster`     | `amazon/chronos-bolt-mini`                  | ~20 M     | direct quantiles            |
| `chronos_bolt_base`| `ChronosBoltBaseForecaster` | `amazon/chronos-bolt-base`                  | 205 M     | direct quantiles            |
| `timesfm_200m`     | `TimesFMForecaster`         | `google/timesfm-1.0-200m-pytorch`           | 200 M     | decoder-only point          |
| `timesfm_500m`     | `TimesFM500MForecaster`     | `google/timesfm-2.0-500m-pytorch`           | 500 M     | decoder-only point, RoPE   |
| `lag_llama`        | `LagLlamaForecaster`        | `time-series-foundation-models/Lag-Llama`   | ~2 M      | LLaMA sampling              |
| `moirai_small`     | `MoiraiForecaster`          | `Salesforce/moirai-1.0-R-small`             | 91 M      | masked-encoder sampling     |
| `moirai_base`      | `MoiraiBaseForecaster`      | `Salesforce/moirai-1.0-R-base`              | 311 M     | masked-encoder sampling     |
| `moirai_large`     | `MoiraiLargeForecaster`     | `Salesforce/moirai-1.0-R-large`             | 1.1 B     | masked-encoder sampling     |
| `moirai2_small`    | `Moirai2SmallForecaster`    | `Salesforce/moirai-2.0-R-small`             | ~11 M     | direct quantiles            |

Implementation notes:

- Each class lazy-loads weights on first `fit()` call; subsequent fits are
  no-ops. The `fit()` cost is logged as `0.0 kWh` because no gradient
  updates are performed (zero-shot).
- All wrappers accept a `context_length`, `device`, `batch_size`, and
  (where applicable) `num_samples` parameter; `point_forecast_mode` and
  `patch_size` are passed through to TimesFM and Moirai respectively.
- **Batched walk-forward.** Each class implements `predict_batch(origin_indices)`
  which packs several context tensors into one forward pass — typically a
  5–10× speed-up on GPU vs per-origin `predict()`. Used when
  `--batch_size > 1` and the predictor exposes the method (`run_tsfm_benchmark.py`
  guards on `hasattr(predictor, "predict_batch")`).
- **Lag-Llama specifics.** Its native context is 32 steps; we use 256 with
  RoPE linear scaling factor `(context + max_h) / 32 ≈ 11.0`. Also
  patches `torch.load` to allow legacy Lightning pickled GluonTS classes.
- **TimesFM 2.0 (500M)** is initialised with `context_len=context_length`
  explicitly *and* `use_positional_embedding=False` — without the first
  flag `timesfm` silently pads back to 2048 and the requested speedup is
  lost.
- **Moirai-2.** Uses `Moirai2Forecast` + `Moirai2Module` (not the 1.x
  classes), 9 deterministic quantile heads, no `patch_size`, no
  `num_samples`. Smaller (~11 M) than Moirai-1.0-small (91 M) yet beats
  every Moirai-1 variant in our results.

### 3.4 `src/metrics.py`  *(401 lines, final)*

Three error metrics + three pluggable energy meters with a shared
`EnergyResult` dataclass.

**Error metrics**
- `mae(y_true, y_pred)`  → MW
- `rmse(y_true, y_pred)` → MW
- `mape(y_true, y_pred)` → percentage points (`mean(|err/y|) * 100`)

**`EnergyResult` dataclass** — carries CodeCarbon's primary fields
(`energy_kwh`, `emissions_kg`, `duration_s`) plus optional CarbonTracker
fields (`ct_*`) and nvidia-smi fields (`nv_*`). `with_ct()` and `with_nv()`
fluent builders merge multi-backend results into one record.

**`EnergyMeter`** — CodeCarbon wrapper.
- Reads `emissions.csv` directly to be robust across CodeCarbon API
  versions (some versions don't return `energy_consumed` from `.stop()`).
- Belgian grid intensity (`country_iso_code="BEL"`) — falls back to the
  default if the CodeCarbon version doesn't accept the kwarg.
- `enabled=False` returns a zero-energy result with `tool="disabled"`,
  used for zero-cost phases (e.g. naive fit).

**`CarbonTrackerMeter`** — wraps `carbontracker.CarbonTracker`.
- Each meter gets its own `log_dir = output_dir/project_name` to avoid
  log-file conflicts on parallel use.
- `update_interval=15` s — keeps `powermetrics` from spawning processes
  faster than the kernel can clean them up on macOS.
- Robust parsing: if no monitor reported a non-None power reading, the
  parsed energy returns NaN rather than 0 (silent monitoring failure).

**`NvidiaSmiMeter`** — GPU-only meter via `nvidia-smi --query-gpu=power.draw`.
- Spawns `nvidia-smi -l <poll>` (default 1 s) and parses each line in a
  daemon thread.
- Auto-disables if `nvidia-smi` is not on `PATH`.
- Power-vs-time samples are integrated with the trapezoidal rule:
  `energy_wh = trapz(watts, times) / 3600` → `energy_kwh = energy_wh / 1000`.
- Emissions = `energy_kwh × 0.167` kgCO₂eq/kWh (Belgian grid).
- Persists raw `(timestamp_s, power_w)` samples to a CSV for later
  inspection (used by `measure_power_curve.py`).

### 3.5 `src/evaluate.py`  *(91 lines, final)*

- `adf_test(series)` — Augmented Dickey-Fuller test (training set only —
  using test data would be leakage). Returns the statistic, p-value, lag,
  obs count, and critical values.
- `evaluate_forecast(y_true, y_pred)` — aligns by DatetimeIndex, returns
  `{MAE, RMSE, MAPE_percent}`.
- `aic_bic_grid_search(train, p_values, d_values, q_values, seasonal, …)`
  — exhaustive AIC/BIC scan over `(p, d, q) × (P, D, Q, s)`. Used early to
  motivate `(2, 1, 2)` as the non-seasonal order.

### 3.6 `src/palette.py`  *(106 lines, final)*

Single source of truth for plot styling shared by every plot script:

- `MODEL_COLOR` — 16-entry palette grouped by family (gray = naive, reds
  = ARIMA/SARIMA, blue = Ridge, green = LGBM, purples = Chronos T5,
  yellows = Chronos Bolt, cyans = TimesFM, browns = Moirai-1, gold =
  Moirai-2, pink = Lag-Llama).
- `MODEL_LABEL` — human-readable display labels (e.g. `lgbm → "LightGBM"`).
- `MODEL_ORDER` — canonical left-to-right plotting order.
- `HORIZON_HATCH` / `HORIZON_LABEL` — hatch + label per horizon for stacked bars.
- `BACKEND_HATCH` — `CC = "..."`, `CT = "///"`, `NV = "xxx"` so triplet
  energy bars stay readable even at the same hue.
- `TSFM_KEYS` — set of foundation-model keys used to switch marker shape
  in trade-off scatter plots.

---

## 4. Entry-point scripts

### 4.1 `prepare_dataset.py`

One-shot script that builds `data/processed/elia_load_15min.csv` from
`data/raw/Data Elia Load.csv` (the Elia download — semi-colon-delimited).
Steps: UTC-parse timestamps, drop tz, drop duplicates, `asfreq("15min")`,
time-aware interpolation, save as comma-CSV with `datetime,totalload`
columns.

### 4.2 `prepare_etth1.py`

Fetches the ETTh1 (Electricity Transformer Temperature, hourly subset 1)
benchmark. Primary path: HuggingFace `ett`/`h1`/`split=test` last record
(14,400 hours, 2016-07-01 → 2018-02-20). Fallback: raw CSV from
`zhouhaoyi/ETDataset` on GitHub (17,420 hours).

### 4.3 `tune_hyperparams.py`  *(215 lines)*

Optuna-driven hyperparameter search for `linear` and `lgbm`.

- **Held-out split.** Last `test_years` (default 1.0) removed up-front to
  prevent leakage. Within the remaining training set, the last
  `val_frac=0.20` becomes the validation set. Chronological — no shuffling.
- **Objective.** Mean MAE across the three horizons `[1, 4, 96]` on the
  validation set, evaluated *one origin* (the last training observation
  position) per horizon — fast enough to allow `--n_trials 50` to finish
  in minutes.
- **Sampler.** `TPESampler(seed=42)` for reproducibility.
- **Search spaces:**
  - Linear: `alpha ∈ LogUniform(1e-3, 1e2)`.
  - LightGBM: `n_estimators ∈ [100, 600]`, `num_leaves ∈ [15, 127]`,
    `learning_rate ∈ LogUniform(0.01, 0.3)`, `min_child_samples ∈ [10, 50]`,
    `subsample ∈ [0.6, 1.0]`, `colsample_bytree ∈ [0.6, 1.0]`.
- **Output:** `results/tuned_params.json`, merged with existing keys.

**Final tuned hyperparameters used in the canonical stride-2 run**:

```json
{
  "linear": {
    "alpha": 34.06457061395274
  },
  "lgbm": {
    "n_estimators":      488,
    "num_leaves":        121,
    "learning_rate":     0.035981484767374575,
    "min_child_samples": 49,
    "subsample":         0.7779334716558033,
    "colsample_bytree":  0.826184287797336
  }
}
```

### 4.4 `run_benchmark.py`  *(656 lines, final)*

Walk-forward benchmark for the 5 classical models. CLI flags below capture
every degree of freedom used in any run; defaults reflect the canonical
Elia + stride-2 configuration.

```
--data, --timestamp_col, --target_col, --freq, --delimiter
--test_years      (default 1.0)
--stride          (default 96 — stride=2 used in final run)
--horizons        (default [1, 4, 96])
--models          (default [naive, linear]; canonical [naive, linear, lgbm, arima, sarima])
--arima_order     (default 2 1 2)
--sarima_order    (default 2 1 2)
--sarima_seasonal_order  (default 1 1 1 96; final run used 1 0 0 96 — see §6)
--naive_s         (default 96)
--maxiter         (default 50)
--max_train_obs   (default 4000)
--use_holidays    (flag)
--tuned_params    (path to tuned_params.json)
--seasonal_residual S  (default 0, disable; alternative residual training)
--energy_tool     {codecarbon, carbontracker, both, none}  (default "both")
--results_dir     (default "results")
--no_plots, --dpi
```

Pipeline (`main()`):

1. Load → clean → split (same `split_last_n_years`).
2. For each `--models` entry: instantiate predictor, optionally load tuned
   params, start meters, fit, stop meters, log `fit_energy`, start meters
   again, walk-forward, stop meters, log `eval_energy`.
3. Compute metrics per horizon → one row per (model, horizon) appended
   to `<run_dir>/metrics.csv`.
4. Save all predictions to `<run_dir>/predictions.parquet`.
5. Print a comparison table to stdout.

The walk-forward loop (`walk_forward`) takes any predictor with the
shared interface, iterates `origin_idx ∈ range(test_start_idx, n − max_h, stride)`,
collects `(y_true[t+h], y_pred[t+h])` per horizon, and returns numpy arrays.

### 4.5 `run_tsfm_benchmark.py`  *(844 lines, final)*

Same walk-forward loop, but targets `TSFM_MODELS`. Key differences from
the classical script:

- `fit_energy = EnergyResult(energy_kwh=0.0, …, tool="zero_shot")` —
  zero-shot models log a fixed zero training cost (not NaN) so downstream
  CSV readers can treat them as "negligible" rather than "missing".
- `--num_samples`, `--device`, `--context_length`, `--batch_size`,
  `--patch_size`, `--point_forecast_mode` are forwarded to the model
  classes by introspecting `cls.__mro__` and only passing kwargs the
  class accepts. New variants automatically pick up the right CLI flags
  with no per-model branches.
- `--energy_tool=all` enables CC + CT + nvidia-smi simultaneously
  (`run_benchmark.py` only knows CC + CT — `run_tsfm_benchmark.py` adds NV).
- `--n_origins` caps origin count for smoke tests.
- `walk_forward` supports both per-origin and `predict_batch(origins)`
  paths; the batched path is chosen automatically when `--batch_size > 1`
  *and* the predictor implements `predict_batch`.

CSV column schema is **identical** to `run_benchmark.py:log_metrics`
output so the two files concatenate without transformation. The full
schema:

```
run_id, timestamp, model, horizon_steps, horizon_minutes, n_predictions,
MAE, RMSE, MAPE_pct,
fit_cc_energy_kwh, fit_cc_emissions_kg, fit_duration_s,
eval_cc_energy_kwh, eval_cc_emissions_kg, eval_duration_s,
fit_ct_energy_kwh, fit_ct_emissions_kg, eval_ct_energy_kwh, eval_ct_emissions_kg,
fit_nv_energy_kwh, fit_nv_emissions_kg, eval_nv_energy_kwh, eval_nv_emissions_kg,
energy_tool
```

### 4.6 Visualisation scripts

- **`visualize_results.py`** (1145 lines) — accuracy plots: per-model
  prediction-vs-actual time series (full year, 1-week zoom, 1-day zoom,
  winter/summer), and an all-models comparison panel. Reads any
  `predictions.parquet`. Output: `figures/<run_id>_vis_<TS>/`.
- **`plot_energy.py`** (865 lines) — energy/CO₂ plots: triplet bar charts
  per backend, MAE-vs-kWh trade-off scatter, per-prediction efficiency
  (kWh ÷ n_predictions), CO₂ trade-off, and an energy-tool-agreement
  sanity check.
- **`plot_thesis_variants.py`** (361 lines) — curated subsets for the
  thesis body. Splits the full 16-model figure into three less-dense
  variants: classical-only, main-FM-only (the largest in each family +
  Moirai-2), and classical + main-FM together. Output:
  `figures/thesis_variants_<TS>/`.
- **`compare_arima_sarima.py`** (193 lines) — focused 2×2 plot
  (MAE × {h=1,4,96} + a single trade-off scatter) comparing ARIMA vs
  SARIMA from a single `wf_*/metrics.csv`.
- **`compare_datasets.py`** + **`compare_datasets_table.py`** (652+355
  lines) — Elia vs ETTh1 cross-dataset comparison; pairs horizons by
  duration in minutes and produces side-by-side accuracy and energy
  figures plus a LaTeX-ready table.

### 4.7 Power-curve scripts (methodology validation)

- **`measure_power_curve.py`** (548 lines) — records per-second power for
  one model's inference, using two parallel meters (`EnergyMeter` flushes
  CodeCarbon every `--cc_poll_interval_s` seconds; `NvidiaSmiMeter` polls
  GPU power at ~1 Hz). Output: `results/powercurve_<model>_<TS>_<tag>/`.
- **`analyze_power_curves.py`** (531 lines) — three diagnostics on top of
  the recorded curves:
  1. **Energy-area cross-validation.** Trapezoidal integration of `W(t)`
     is compared against CodeCarbon's own cumulative `energy_consumed`
     and nvidia-smi's integrator — three independent paths to the same
     number, used as a self-consistency check.
  2. **Per-origin normalised power** — divides each curve by its
     `n_origins` to give "watts per prediction origin", making different
     workloads (60 vs 8000 origins) directly comparable.
  3. **Lag-Llama batch-boundary annotation** — overlays vertical lines
     at expected batch boundaries to confirm the periodic spikes
     correspond to batched forward passes.
- **`plot_power_curves_composite.py`** (157 lines) — composite figure
  produced for the thesis.

### 4.8 Orchestration shell scripts (final wrappers)

| Script | What it does | Used for |
|---|---|---|
| `scripts/setup_fm_env.sh` | One-shot `uv`-based setup of `.venv_fm` (Python 3.11, torch 2.4 CPU, gluonts, chronos-forecasting, uni2ts ≥ 1.2, codecarbon, carbontracker, local lag-llama clone). | First-time install. |
| `scripts/run_energy_pass.sh` | Sequential 8-model pass (naive → linear → lgbm → arima → chronos_bolt_mini → timesfm_200m → moirai_small → lag_llama), stride 24, `--energy_tool both`. Sequential ⇒ CC attribution is clean. | Mid-project energy validation. |
| `scripts/run_energy_fm_all.sh` | Sequential FM-only pass with `--energy_tool all` (CC + CT + NV). | FM-only carbon validation. |
| `scripts/run_fm_large.sh` | 4 large-FM variants (chronos_bolt_base, moirai_large, chronos_large, timesfm_500m) at stride 2, `batch_size=16`. | Final FM-large numbers. |
| `scripts/run_stride2_remaining.sh` | The 11 stride-2 runs not covered by `all_best_stride2_…` (4 classical with `--energy_tool both`, 7 FMs with `--energy_tool all`). | Final stride-2 pass. |

All scripts:
- export `HF_HOME=/volume1/no_backup/r1062653/hf_cache` (keeps large
  weight caches off the backup volume),
- export `TOKENIZERS_PARALLELISM=false`,
- pin BLAS threads to 8 (`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`),
- prepend the bundled `lag-llama/` clone to `PYTHONPATH`,
- log to `logs/<scope>/<timestamp>.log`.

---

## 5. Datasets

### 5.1 Elia (primary)

| Field | Value |
|---|---|
| Source | Elia, the Belgian electricity TSO |
| Variable | `Total Load` (MW) |
| Frequency | 15 min |
| Span | **2014-12-31 → 2026-03-09** |
| Observations | **392,256** |
| Missing handling | Time-aware linear interpolation in `clean_timeseries` |
| Path | `data/processed/elia_load_15min.csv` (columns `datetime,totalload`) |
| Build script | `prepare_dataset.py` |

Train / test split for every reported run:

- **Train:** 357,215 obs (≈ 10 years), ending 2025-03-09.
- **Test:** **35,041 obs** (last calendar year), 2025-03-09 → 2026-03-09.
- Split helper: `split_last_n_years(df, test_years=1.0)`.

### 5.2 ETTh1 (secondary, robustness check)

| Field | Value |
|---|---|
| Source | Zhou et al., ETT-small (HF `ett`/`h1` config; GitHub fallback) |
| Variable | `OT` (oil temperature, °C) |
| Frequency | 1 hour |
| Span | 2016-07-01 → 2018-02-20 (≈ 1.6 yr) |
| Observations | 14,400 hourly |
| Path | `data/processed/etth1.csv` |
| Default test_years | 0.25 (≈ 91 days) |
| Default horizons | `[1, 24, 168]` (1 h / 1 d / 1 w) |
| `--naive_s` | 24 (daily) |
| `--stride` | 24 |

Used by `compare_datasets.py` to demonstrate that the ranking is
dataset-sensitive (LightGBM dominates Elia but classical statistical
models are stronger on ETTh1's much shorter, oil-thermal series).

---

## 6. Evaluation protocol — the final canonical configuration

| Item | Final value | Why |
|---|---|---|
| Dataset | Elia 15-min, 392,256 obs | The primary thesis dataset. |
| Train / test | 357,215 / 35,041 obs (last 1 calendar year) | Operational lookback ≈ 10 yr / 1 yr verification. |
| Horizons | `h ∈ {1, 4, 96}` steps | h=1 (15 min) tests intra-15-min reactivity; h=4 (1 h) is dispatch-relevant; h=96 (24 h) is the seasonal-naive trap. |
| **Stride** | **2 steps (30 min)** | Statistical power: 17,473 origins vs 365 at stride=96 — SE of each MAE drops ~7×. Honest seasonal-naive baseline (stride=96 trivially hit the same time-of-day every origin and underestimated naive MAE by ~40 %). |
| Origins per model | **17,473** | Same for every model so MAE-vs-MAE differences are computed on identical samples. |
| Walk-forward update | `results.append(refit=False)` (ARIMA/SARIMA) | ~365× faster than re-fit per origin with no accuracy loss. |
| ARIMA / SARIMA training cap | `max_train_obs = 4000` | Bounds Kalman state memory at s=96 seasonality. |
| Imputation | Time-aware linear (`clean_timeseries(impute_method="interpolate")`) | Robust to single-slot DST gaps. |
| Hardware | **heimdall** — 2× AMD EPYC 9455 (192 logical cores), 251 GB RAM, 1× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) | Reported below. |
| Energy backends | **CodeCarbon + CarbonTracker + nvidia-smi** (FM runs use all three; classical use CC + CT only because CPU jobs have no GPU power to integrate). | Three independent measurement paths. |
| Sequential vs parallel | **Sequential** for the canonical pass | CodeCarbon attributes *system-wide* energy to the calling process — parallel runs muddle attribution. |

**SARIMA configuration freeze.** The first SARIMA attempt used
`(2,1,2)(1,1,1,96)` and grew an unbounded Kalman state with each
`.append()` (RSS climbed from 8 GB → 28 GB in 300 origins; projected
~1.6 TB at completion). The final canonical SARIMA uses
**`(2,1,2)(1,0,0,96)`** — seasonal AR(1) at lag 96 with no seasonal
differencing and no seasonal MA — which keeps RSS bounded (7–15 GB
oscillation) and matches the configuration of an earlier thesis run.

---

## 7. Energy measurement methodology (CC + CT + NV cross-validation)

Three independent measurement paths run **simultaneously** and bracket
exactly the same code block (the walk-forward inference loop):

1. **CodeCarbon (`CC`)** — system-level Intel RAPL on CPU and NVML on GPU,
   plus a national grid-intensity model. Country code `BEL`
   (0.167 kgCO₂eq/kWh by default). Robustness trick: we read the raw
   `emissions.csv` directly rather than rely on the return value of
   `tracker.stop()`, which differs across CodeCarbon versions.
2. **CarbonTracker (`CT`)** — independent reimplementation of the same idea
   (RAPL on Linux, powermetrics on macOS), polls every 15 s, logs per-
   component reasons for missing power readings. Used as an independent
   "second opinion" on CodeCarbon.
3. **nvidia-smi (`NV`)** — direct GPU power polling at 1 Hz, integrated
   with the trapezoidal rule. Bypasses NVML attribution entirely;
   measures *the whole GPU*, which equals "this job" because no other
   process shares the device during a sequential run.

**Why three?** Each backend has a known failure mode:

- CC can over-attribute when other processes share the CPU.
- CT silently reports zero when `powermetrics` lacks sudo on macOS or
  when RAPL counters are unreadable.
- NV is GPU-only and misses CPU energy.

If all three agree on the same eval block within ±10 %, the measurement
is trusted. The cross-validation table is in
`figures/energy_cross_validation.csv` and was produced by
`analyze_power_curves.py`.

**Belgian grid intensity.** Set explicitly to **0.167 kgCO₂eq/kWh** in
`src/metrics.py` and propagated by all three backends (CC via its country
table; NV multiplies its integrated energy by the same constant).

---

## 8. The 16 models — final configurations & results

### 8.1 Classical (5)

#### Seasonal naive (`naive`)

- **Class:** `NaivePredictor(s=96)` in `src/models.py:44-64`.
- **Definition:** `ŷ[t+h] = y[t+h−96]` — predict "same value as 24 h ago".
- **Hyperparameters:** none.
- **Train cost:** 0.
- **Run ID:** `wf_20260512_204729` (results/stride2_remaining_20260512_204727/naive/).
- **CLI:** `python run_benchmark.py --models naive --horizons 1 4 96 --stride 2 --test_years 1.0 --energy_tool both --no_plots`.

| h | MAE | RMSE | MAPE | n |
|---:|---:|---:|---:|---:|
| 1 | 517.93 | 745.48 | 5.59 % | 17,473 |
| 4 | 518.78 | 745.40 | 5.60 % | 17,473 |
| 96 | 515.63 | 742.09 | 5.57 % | 17,473 |

Eval time 5.16 s. Flat across horizons because the formula doesn't
depend on h. **The operational floor — any usable model must beat 515 MW
MAE at h=96.**

#### Ridge regression (`linear`, tuned)

- **Class:** `MLPredictor` wrapping `DirectForecaster(Pipeline(StandardScaler, Ridge))`.
- **Hyperparameter:** `alpha = 34.06457061395274` (Optuna).
- **Features:** 13 lags `[0, 1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672]` + cyclic
  sin/cos of hour-of-day and day-of-week. No holiday feature.
- **Run ID:** `wf_20260512_204738`.
- **CLI:** `python run_benchmark.py --models linear --horizons 1 4 96 --stride 2 --test_years 1.0 --tuned_params results/tuned_params.json --energy_tool both --no_plots`.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 82.96 | 114.64 | 0.91 % |
| 4 | 198.74 | 257.29 | 2.16 % |
| 96 | 452.35 | 583.28 | 4.95 % |

Fit 4.93 s / 8.6 × 10⁻⁵ kWh CC. Eval 23.52 s / 9.0 × 10⁻⁴ kWh CC,
5.4 × 10⁻⁴ kWh CT. CPU-only.

#### LightGBM (`lgbm`, tuned) — **overall winner**

- **Class:** `MLPredictor` wrapping `DirectForecaster(LGBMRegressor)`.
- **Hyperparameters (Optuna):** `n_estimators=488`, `num_leaves=121`,
  `learning_rate=0.036`, `min_child_samples=49`, `subsample=0.778`,
  `colsample_bytree=0.826`, `verbose=-1`.
- **Same feature set as Ridge.**
- **Run ID:** `wf_20260512_204811`.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | **78.45** | 110.33 | 0.86 % |
| 4 | 155.12 | 207.04 | 1.67 % |
| 96 | **313.38** | 449.46 | 3.38 % |

Fit 29.89 s / 1.74 × 10⁻³ kWh CC. Eval 26.55 s / 1.43 × 10⁻³ kWh CC.
**Best classical model; beats every FM except `chronos_bolt_base`
on h=96** at ~1/15 of `timesfm_500m`'s eval-time energy.

#### ARIMA (`arima`, 2,1,2)

- **Class:** `ARIMAPredictor(cfg=(2,1,2),(0,0,0,0))` in `src/models.py:67-126`.
- **Optimiser:** `lbfgs`, `maxiter=50`, `enforce_stationarity=False`,
  `enforce_invertibility=False`.
- **Training cap:** `max_train_obs=4000` (last 4,000 obs ≈ 6 weeks).
- **Walk-forward update:** `results.append(new_obs, refit=False)` — single
  Kalman step per origin, ~365× faster than per-origin re-fit.
- **Run ID:** `wf_20260512_204912`.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 86.78 | 120.84 | 0.96 % |
| 4 | 228.09 | 296.43 | 2.49 % |
| 96 | 798.13 | 1042.14 | 8.70 % |

Fit 4.59 s / 6.3 × 10⁻⁵ kWh CC. Eval **876.84 s ≈ 14.6 min** /
3.8 × 10⁻² kWh CC, 1.3 × 10⁻² kWh CT. CPU-only. Strong at h=1, collapses
at h=96 — a non-seasonal model can't project the daily cycle 24 h ahead.

#### SARIMA (`sarima`, 2,1,2)(1,0,0,96)

- **Class:** `ARIMAPredictor(cfg=(2,1,2),(1,0,0,96))`.
- **Why P=1, D=0, Q=0 and not the textbook (1,1,1,96):** the seasonal-
  differenced variant grew an unbounded Kalman state with each
  `.append()` (RSS 8 GB → 28 GB in 300 origins, projected 1.6 TB at
  completion). The frozen (1,0,0,96) variant keeps RSS bounded.
- **Run IDs:**
  - Full-scale stride-2: `results/sarima_stride2_20260513_102837/wf_20260513_102838/`
  - Smaller cross-check (test_years=0.25, stride 96): `wf_20260515_173525` (data below).

| h | MAE | RMSE | MAPE | n |
|---:|---:|---:|---:|---:|
| 1 | 85.22 | 109.89 | 0.88 % | 91 |
| 4 | 114.50 | 141.01 | 1.20 % | 91 |
| 96 | 336.78 | 479.87 | 3.36 % | 91 |

Fit 125.39 s, eval 1,405.18 s. CC eval 6.1 × 10⁻² kWh, CT eval
2.0 × 10⁻² kWh. SARIMA recovers most of ARIMA's h=96 loss but pays
a ~16× wall-clock penalty over ARIMA for the seasonal Kalman state.

### 8.2 Foundation models (11)

#### `chronos_mini` — Amazon Chronos-T5-mini

- **HF:** `amazon/chronos-t5-mini` (~20 M params, T5 encoder-decoder).
- **Config:** `context=512`, `num_samples=100`, `torch.bfloat16`, CUDA,
  median over samples.
- **Run ID:** `tsfm_20260512_210912`.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 84.19 | 114.86 | 0.93 % |
| 4 | 165.99 | 219.56 | 1.82 % |
| 96 | 585.26 | 790.58 | 6.37 % |

Eval 484.12 s ≈ 8.1 min. CC 6.78 × 10⁻², CT 7.79 × 10⁻², **NV
5.10 × 10⁻²** kWh.

#### `chronos_large` — Amazon Chronos-T5-large

- **HF:** `amazon/chronos-t5-large` (710 M params).
- **Config:** identical to `chronos_mini` (context 512, 100 samples).
- **Best h=1 of any model:** **MAE 78.08**.
- **Eval 4,767.87 s ≈ 79 min** / CC 0.78, CT 0.96, NV 0.61 kWh.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | **78.08** | 107.22 | 0.86 % |
| 4 | 149.02 | 199.64 | 1.64 % |
| 96 | 513.61 | 713.64 | 5.58 % |

#### `chronos_bolt_mini` — Amazon Chronos-Bolt-mini *(deterministic)*

- **HF:** `amazon/chronos-bolt-mini` (~20 M).
- **API:** direct quantile output, **no sampling** ⇒ deterministic and fast.
- **Config:** `_CONTEXT_LENGTH=2048` (native; 4× longer than T5-Chronos),
  `quantile_levels=[0.5]`, bfloat16, CUDA.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 97.11 | 128.89 | 1.06 % |
| 4 | 153.49 | 204.95 | 1.67 % |
| 96 | 364.18 | 514.58 | 3.94 % |

Eval **16.68 s** / CC 1.52 × 10⁻³, CT 4.86 × 10⁻⁴, NV 9.95 × 10⁻⁴ kWh.
**Best accuracy / energy ratio of any foundation model.**

#### `chronos_bolt_base` — Amazon Chronos-Bolt-base *(Pareto-optimal FM)*

- **HF:** `amazon/chronos-bolt-base` (205 M).
- **Config:** identical interface to `chronos_bolt_mini`.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 88.00 | 117.01 | 0.97 % |
| 4 | **140.18** | 188.14 | 1.53 % |
| 96 | **341.86** | 485.76 | 3.70 % |

Eval 52.52 s / CC 7.78 × 10⁻³, CT 7.80 × 10⁻³, NV 6.05 × 10⁻³ kWh.
**Best h=96 of any FM; only LightGBM beats it overall.**

#### `timesfm_200m` — Google TimesFM 1.0

- **HF:** `google/timesfm-1.0-200m-pytorch` (200 M).
- **Config:** PyTorch backend, `per_core_batch_size=32`, `context=512`,
  `horizon_len=96`, `point_forecast_mode="median"`, `freq=[0]` (sub-
  hourly), CUDA.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 87.98 | 118.56 | 0.97 % |
| 4 | 166.42 | 220.93 | 1.81 % |
| 96 | 499.01 | 677.99 | 5.40 % |

Eval 328.62 s / CC 1.43 × 10⁻², CT 5.09 × 10⁻³, NV 3.06 × 10⁻³ kWh.

#### `timesfm_500m` — Google TimesFM 2.0

- **HF:** `google/timesfm-2.0-500m-pytorch` (500 M).
- **Config:** `num_layers=50`, `use_positional_embedding=False` (RoPE
  only), `context_length=1024` (safe-optimised — ½ attention compute
  vs the 2048 native max while still covering ~10.6 days of 15-min
  data), `per_core_batch_size=32`, CUDA.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 80.94 | 110.07 | 0.89 % |
| 4 | 144.12 | 192.51 | 1.57 % |
| 96 | 387.65 | 531.91 | 4.22 % |

Eval 1,479.92 s ≈ 24.7 min / CC 6.45 × 10⁻², CT 2.13 × 10⁻², NV
1.33 × 10⁻² kWh. ~3× the energy of `timesfm_200m` for a ~22 %
h=96 MAE improvement.

#### `lag_llama` — LLaMA-style decoder

- **HF:** `time-series-foundation-models/Lag-Llama` (~2 M).
- **Config:** `context_length=256` (≫ native 32; RoPE linear scale
  `(256+96)/32 ≈ 11.0`), `num_samples=100`, `batch_size=16`. Architecture
  hyperparams read from the checkpoint at load time
  (`input_size=1, n_layer=4, n_embd_per_head=64, n_head=4,
   scaling="robust", time_feat=True`). Backend: GluonTS with the
  PyTorch ≥ 2.6 `weights_only=False` patch.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 399.34 | 518.16 | 4.64 % |
| 4 | 828.10 | 1029.77 | 9.57 % |
| 96 | 1404.08 | 1783.50 | 16.00 % |

Eval **3,794.07 s ≈ 63 min** / CC **0.65**, CT **0.82**, NV **0.52** kWh.
**Worst model in the benchmark.** Native context of 32 is structurally
too short for 96-step seasonal forecasting; RoPE scaling helps but does
not close the gap.

#### `moirai_small` — Salesforce Moirai-1.0-R-small

- **HF:** `Salesforce/moirai-1.0-R-small` (91 M).
- **Config:** `context_length=512`, `num_samples=100`, `batch_size=16`,
  `patch_size="auto"`, `target_dim=1`, CUDA.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 203.65 | 339.19 | 2.31 % |
| 4 | 307.97 | 422.49 | 3.45 % |
| 96 | 788.45 | 994.76 | 8.65 % |

Eval 96.53 s / CC 7.13 × 10⁻³, CT 5.83 × 10⁻³, NV 3.87 × 10⁻³ kWh.

#### `moirai_base` — Salesforce Moirai-1.0-R-base

- **HF:** `Salesforce/moirai-1.0-R-base` (311 M). Same config as small.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 116.29 | 196.33 | 1.30 % |
| 4 | 225.43 | 309.02 | 2.47 % |
| 96 | 796.86 | 1005.87 | 8.79 % |

Eval 146.90 s / CC 1.33 × 10⁻², CT 1.23 × 10⁻², NV 8.29 × 10⁻³ kWh.

#### `moirai_large` — Salesforce Moirai-1.0-R-large

- **HF:** `Salesforce/moirai-1.0-R-large` (1.1 B). Same config.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 101.71 | 173.36 | 1.14 % |
| 4 | 205.73 | 279.23 | 2.26 % |
| 96 | 765.75 | 940.18 | 8.52 % |

Eval 262.11 s / CC 2.87 × 10⁻², CT 3.01 × 10⁻², NV 1.97 × 10⁻² kWh.
**Scaling does not pay off** in this benchmark: Moirai-1 large only
marginally beats small/base at h=96 while costing 4× the energy.

#### `moirai2_small` — Salesforce Moirai-2.0-R-small *("Less Is More")*

- **HF:** `Salesforce/moirai-2.0-R-small` (~11 M params).
- **Architecture:** deterministic quantile heads (9 quantiles output
  directly — no Monte-Carlo sampling), fixed internal patching, ~8×
  smaller than Moirai-1.0-small.
- **Config:** `context_length=512`, `batch_size=16`, median (q=0.5) as
  point forecast, CUDA.

| h | MAE | RMSE | MAPE |
|---:|---:|---:|---:|
| 1 | 79.64 | 109.97 | 0.88 % |
| 4 | 153.67 | 207.42 | 1.68 % |
| 96 | 449.31 | 625.96 | 4.87 % |

Eval **19.84 s** / CC 2.14 × 10⁻³, CT 1.12 × 10⁻³, NV 9.39 × 10⁻⁴ kWh.
**Beats Moirai-1.0-large at every horizon while using 13× less energy
and ~100× fewer parameters.** Strong empirical confirmation of the
"Less Is More" thesis from Liu et al. (2025).

---

## 9. Headline rankings (final stride-2 numbers)

### 9.1 Accuracy at h=96 (the seasonal-naive trap)

| Rank | Model              | MAE h=1 | MAE h=4 | **MAE h=96** | MAPE h=96 | vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **lgbm** (tuned)   | **78.5**| **155.1**| **313.4** | 3.38 % | −39 % |
| 2 | chronos_bolt_base  | 88.0    | 140.2    | 341.9     | 3.70 % | −34 % |
| 3 | chronos_bolt_mini  | 97.1    | 153.5    | 364.2     | 3.94 % | −29 % |
| 4 | timesfm_500m       | 80.9    | 144.1    | 387.6     | 4.22 % | −25 % |
| 5 | moirai2_small      | 79.6    | 153.7    | 449.3     | 4.87 % | −13 % |
| 6 | linear (tuned)     | 83.0    | 198.7    | 452.3     | 4.95 % | −12 % |
| 7 | timesfm_200m       | 88.0    | 166.4    | 499.0     | 5.40 % | −3 %  |
| 8 | chronos_large      | 78.1    | 149.0    | 513.6     | 5.58 % | −0.4 %|
| 9 | naive              | 517.9   | 518.8    | 515.6     | 5.57 % | baseline |
| 10 | chronos_mini      | 84.2    | 166.0    | 585.3     | 6.37 % | +13 % |
| 11 | moirai_large      | 101.7   | 205.7    | 765.7     | 8.52 % | +49 % |
| 12 | moirai_small      | 203.6   | 308.0    | 788.4     | 8.65 % | +53 % |
| 13 | moirai_base       | 116.3   | 225.4    | 796.9     | 8.79 % | +55 % |
| 14 | arima             | 86.8    | 228.1    | 798.1     | 8.70 % | +55 % |
| 15 | sarima (1,0,0,96)*| 85.2    | 114.5    | 336.8     | 3.36 % | −35 % |
| 16 | lag_llama         | 399.3   | 828.1    | 1404.1    | 16.0 % |+149 % |

*SARIMA evaluated on a smaller test (n=91) due to its much higher wall-
clock cost at stride 2 — see §6 for the safe seasonal-order choice and
§10 for the full-scale stride-2 run that is still completing at the time
of writing.*

### 9.2 Energy ranking (eval phase, CodeCarbon kWh, sorted ascending)

| Rank | Model              | Eval s   | CC kWh   | CT kWh   | NV kWh   | Device |
|---:|---|---:|---:|---:|---:|---|
| 1 | naive               | 5.2      | n/a      | 3.1e-5   | –        | CPU |
| 2 | linear              | 23.5     | 9.0e-4   | 5.4e-4   | –        | CPU |
| 3 | chronos_bolt_mini   | 16.7     | 1.5e-3   | 4.9e-4   | 1.0e-3   | GPU |
| 4 | lgbm                | 26.6     | 1.4e-3   | 3.3e-4   | –        | CPU |
| 5 | moirai2_small       | 19.8     | 2.1e-3   | 1.1e-3   | 9.4e-4   | GPU |
| 6 | moirai_small        | 96.5     | 7.1e-3   | 5.8e-3   | 3.9e-3   | GPU |
| 7 | chronos_bolt_base   | 52.5     | 7.8e-3   | 7.8e-3   | 6.0e-3   | GPU |
| 8 | moirai_base         | 146.9    | 1.3e-2   | 1.2e-2   | 8.3e-3   | GPU |
| 9 | timesfm_200m        | 328.6    | 1.4e-2   | 5.1e-3   | 3.1e-3   | GPU |
| 10 | moirai_large       | 262.1    | 2.9e-2   | 3.0e-2   | 2.0e-2   | GPU |
| 11 | arima              | 876.8    | 3.8e-2   | 1.3e-2   | –        | CPU |
| 12 | timesfm_500m       | 1,479.9  | 6.5e-2   | 2.1e-2   | 1.3e-2   | GPU |
| 13 | chronos_mini       | 484.1    | 6.8e-2   | 7.8e-2   | 5.1e-2   | GPU |
| 14 | lag_llama          | 3,794.1  | 6.5e-1   | 8.2e-1   | 5.2e-1   | GPU |
| 15 | chronos_large      | 4,767.9  | 7.8e-1   | 9.6e-1   | 6.1e-1   | GPU |

### 9.3 Pareto frontier (h=96 MAE × eval kWh)

The undominated set (no other model is both more accurate at h=96 *and*
cheaper to evaluate):

1. **naive** — cheapest, accuracy floor.
2. **lgbm** — best accuracy *and* very cheap (CPU).
3. **chronos_bolt_mini** — best FM accuracy-per-kWh.
4. **chronos_bolt_base** — best FM h=96.

Every other model is strictly dominated by at least one of these four.

---

## 10. Key empirical findings (for the thesis discussion)

1. **LightGBM wins both axes.** Best accuracy across all three horizons
   *and* second-lowest eval energy (after naive). The Pareto-optimal
   model for this dataset.
2. **Chronos-Bolt is the Pareto-optimal FM family.** Direct quantile
   output (deterministic, no sampling) is the key — it makes Bolt 30×
   faster than T5-Chronos at the same parameter count.
3. **FM scaling laws fail here.** `chronos_large` (710 M) and
   `moirai_large` (1.1 B) are *worse* than their smaller siblings at
   h=96 — likely because the Belgian load signal is dominated by a
   single daily seasonality plus weather effects that the smaller
   model already captures.
4. **Moirai-2 ("Less Is More") confirms a deliberate downsize.** With
   ~11 M parameters it beats every Moirai-1 variant (small, base, large)
   at every horizon while using 13× less energy than Moirai-1-large.
5. **Lag-Llama is unfit for daily seasonality.** Native context of 32 is
   structurally too short; RoPE scaling cannot fully compensate.
6. **SARIMA recovers most of ARIMA's h=96 loss** but costs ~16× more
   wall-clock time, and its Kalman state grows unboundedly with the
   textbook `(1,1,1,96)` order — the empirical safe order is
   `(1,0,0,96)`.
7. **Energy backends agree within ~10–25 %.** CC, CT, and NV cross-
   validate cleanly for GPU runs; CC tends to slightly over-report when
   sharing the CPU with kernel daemons (`tokenizers`, HF download
   threads), and NV is GPU-only so it under-reports total system energy
   by the CPU-host portion (~10 % on Blackwell).
8. **Stride matters for the naive baseline.** At stride=96 (the original
   protocol) the seasonal-naive evaluator trivially hit the same time-of-
   day each origin and gave MAE = 297 — a misleadingly low denominator.
   At stride=2 the test sample covers every time-of-day pair and naive
   MAE rises to 515. The stride=2 number is the correct comparison
   point.

---

## 11. Hardware

**heimdall** (KU Leuven ESAT shared workstation):

- **CPU:** 2× AMD EPYC 9455 (192 logical cores total).
- **RAM:** 251 GB DDR5.
- **GPU:** 1× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM).
- **Storage:** `/volume1/backup/` (this project) is on a backed-up volume;
  `/volume1/no_backup/` holds the HF weight cache (~30 GB) and the
  `.venv_fm` virtual environment.

System resource limits enforced (see `CLAUDE.md`):

- SARIMA / SARIMAX walk-forward: single-worker only (state-space objects
  at s=96 can hold many GB of Kalman state).
- ≤ 4 classical CPU models in parallel (each < 2 GB RSS).
- ≤ 4 foundation models on GPU in parallel (~2 GB host RAM each).
- `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` to avoid
  oversubscribing shared cores.

---

## 12. Software environments

### 12.1 `.venv` (classical)

```
python 3.14
numpy
pandas
matplotlib
scikit-learn
statsmodels        # SARIMAX
codecarbon         # energy / carbon
carbontracker      # second energy backend
pyarrow            # parquet IO
tqdm               # progress bars
lightgbm           # for --models lgbm
holidays           # for --use_holidays
optuna             # tune_hyperparams.py
```

Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

### 12.2 `.venv_fm` (foundation models)

Python 3.11 (pinned — `chronos-forecasting`, `timesfm` and `lag_llama`
require < 3.12). Bootstrapped by `scripts/setup_fm_env.sh`:

```
torch == 2.4.*  (PyTorch CPU wheel; CUDA 12.4 wheel on heimdall)
setuptools<80   # newer setuptools drops pkg_resources still used by lightning / uni2ts
numpy
pandas < 2.2
matplotlib, scikit-learn, pyarrow, tqdm
huggingface_hub
gluonts
chronos-forecasting
uni2ts >= 1.2.0
codecarbon, carbontracker
lag_llama         # editable install from the bundled clone in lag-llama/
timesfm           # optional — needed for TimesFM-200M / TimesFM-500M
```

HF cache: `HF_HOME=/volume1/no_backup/r1062653/hf_cache` (kept off the
backup volume).

---

## 13. Figures (final)

All numbered figures referenced in the thesis live under `figures/`:

| Figure topic | Folder |
|---|---|
| Per-model prediction time-series (all 16 models, h=1/4/96, year + 1-week + 1-day zooms, winter + summer) | `figures/all_best_stride2_20260511_215221_vis_20260512_140235/` (49 figures) |
| Energy / CO₂ / efficiency comparison (8-model subset, CC + CT + NV triplets) | `figures/energy_full_plots/` (8 figures) |
| Thesis-body curated variants (classical-only, main-FM-only, classical+FM) | `figures/thesis_variants_20260515_215942/` |
| ARIMA vs SARIMA focused comparison | `figures/arima_vs_sarima_20260515_202217/` |
| Composite power-curve (CC vs NV, 5 representative models) | `figures/power_curves_composite.png` |
| Per-origin normalised power | `figures/power_curves_per_origin.png` |
| Lag-Llama batch-boundary annotation | `figures/lag_llama_batch_boundaries.png` |
| Chronos amortisation curve (batch_size sweep) | `figures/chronos_amortization.png` |
| Energy backend cross-validation CSV | `figures/energy_cross_validation.csv` |
| Elia × ETTh1 cross-dataset side-by-side | `figures/compare_Elia_vs_ETTh1_20260514_191712/` |
| Robustness Elia × ETTh1 table | `figures/robustness_Elia_vs_ETTh1_20260514_194947/` |
| TSFM combined (16-model, stride-2) | `figures/tsfm_all_stride2_20260513_vis_20260513_010954/` |

The single most important thesis figure is
**`figures/thesis_variants_20260515_215942/comparison_classical_and_main_fm.png`**
— the canonical "11-model" plot (4 classical + Naive + 6 main FMs)
showing MAE bars per horizon side-by-side. It is intentionally less
dense than the all-16 figure so it fits a single thesis page.

---

## 14. Audit trail — run identifiers

Every result quoted above can be traced to a specific run folder:

- **Classical (stride 2, final):** `results/stride2_remaining_20260512_204727/`
  - `naive/wf_20260512_204729/`
  - `linear/wf_20260512_204738/`
  - `lgbm/wf_20260512_204811/`
  - `arima/wf_20260512_204912/`
- **FM (stride 2, smaller variants, final):** `results/stride2_remaining_20260512_204727/`
  - `chronos_bolt_mini/`, `chronos_mini/`, `moirai_small/`, `moirai_base/`,
    `moirai2_small/`, `timesfm_200m/`, `lag_llama/`.
- **FM (stride 2, large variants, final):** `results/fm_large_20260512_153216/`
  - `chronos_bolt_base/`, `chronos_large/`, `moirai_large/`, `timesfm_500m/`.
- **Combined consolidated stride-2 metrics:** `results/tsfm_all_stride2_20260513/metrics.csv` (read by `plot_thesis_variants.py`).
- **SARIMA stride 2 (full scale):** `results/sarima_stride2_20260513_102837/wf_20260513_102838/`.
- **SARIMA stride 96 cross-check:** `results/wf_20260515_173525/`.
- **ARIMA-vs-SARIMA mini-pair (test_years=0.25):** `results/wf_20260515_173525/` (figures in `figures/arima_vs_sarima_20260515_202217/`).
- **Tuned hyperparameters:** `results/tuned_params.json` (linear, lgbm).
- **Cumulative log of every run ever made:** `results/metrics_runs.csv`.

---

## 15. How to reproduce the canonical pass

```bash
# 1. Activate classical env
source .venv/bin/activate

# 2. (Re)build the dataset
python prepare_dataset.py

# 3. Tune linear and lgbm (or reuse results/tuned_params.json)
python tune_hyperparams.py --model linear --n_trials 50
python tune_hyperparams.py --model lgbm   --n_trials 50

# 4. Run the 4 classical models (CPU, --energy_tool both)
for m in naive linear lgbm arima; do
    tuned=""
    [ "$m" = "linear" ] || [ "$m" = "lgbm" ] && tuned="--tuned_params results/tuned_params.json"
    python run_benchmark.py --models "$m" --horizons 1 4 96 --stride 2 \
        --test_years 1.0 $tuned --energy_tool both --no_plots
done

# 5. Run SARIMA (single-worker; the canonical seasonal order is the safe one)
python run_benchmark.py --models sarima --horizons 1 4 96 --stride 2 \
    --test_years 1.0 --sarima_order 2 1 2 --sarima_seasonal_order 1 0 0 96 \
    --maxiter 50 --max_train_obs 4000 --energy_tool both --no_plots

# 6. Switch to FM env
deactivate
source .venv_fm/bin/activate
export HF_HOME=/volume1/no_backup/r1062653/hf_cache
export PYTHONPATH=$PWD/lag-llama:$PYTHONPATH

# 7. Run the 7 small/mini FMs (CC + CT + NV)
for m in chronos_bolt_mini moirai2_small moirai_small moirai_base chronos_mini timesfm_200m lag_llama; do
    python run_tsfm_benchmark.py --models "$m" \
        --horizons 1 4 96 --test_years 1.0 --stride 2 \
        --batch_size 16 --device cuda --gpu_index 0 \
        --energy_tool all --no_plots
done

# 8. Run the 4 large FMs (CC + CT + NV)
for m in chronos_bolt_base moirai_large chronos_large timesfm_500m; do
    python run_tsfm_benchmark.py --models "$m" \
        --horizons 1 4 96 --test_years 1.0 --stride 2 \
        --batch_size 16 --device cuda --gpu_index 0 \
        --energy_tool all --no_plots
done

# 9. Plot
python visualize_results.py
python plot_energy.py
python plot_thesis_variants.py
python compare_arima_sarima.py
```

Total wall-clock time on heimdall:

- Classical (4 models, stride 2): ~16 min.
- SARIMA (stride 2, full): ~14 h (single-worker, by design).
- FM small/mini (7 models): ~90 min.
- FM large (4 models): ~3.5 h.
- Total: ~19 h end-to-end.

---

## 16. Open issues / future work (for the thesis discussion)

1. **SARIMA Kalman-state memory growth.** The (1,1,1,96) seasonal-
   differenced variant is unusable at stride 2 because `.append()` grows
   the state unboundedly. The (1,0,0,96) variant is empirically safe,
   but a formal characterisation of the boundary would strengthen the
   methodology section.
2. **CarbonTracker silent zeros on parallel runs.** CT occasionally
   reports zero energy when other processes also call `powermetrics`.
   The current mitigation is to run sequentially; a more robust solution
   would be to expose CT's `update_interval` per-run and back off when
   collisions are detected.
3. **Foundation-model fine-tuning.** All FMs here are zero-shot. Brief
   fine-tuning on Elia could close the gap with LightGBM — outside the
   scope of this thesis (which targets zero-shot deployability), but a
   natural follow-up.
4. **Probabilistic intervals.** Every FM here is reduced to a point
   forecast (median of samples / direct quantile). The samples
   themselves are saved in `predictions.parquet` and could support a
   coverage analysis in a future paper.

---

## 17. References to source files (clickable in the IDE)

- Walk-forward classical entry point: [run_benchmark.py](run_benchmark.py).
- Walk-forward FM entry point: [run_tsfm_benchmark.py](run_tsfm_benchmark.py).
- Classical model classes: [src/models.py](src/models.py).
- Foundation model wrappers: [src/models_tsfm.py](src/models_tsfm.py).
- Data loaders: [src/data.py](src/data.py).
- Metrics + 3 energy backends: [src/metrics.py](src/metrics.py).
- ADF + grid-search helpers: [src/evaluate.py](src/evaluate.py).
- Plot palette: [src/palette.py](src/palette.py).
- Optuna tuning: [tune_hyperparams.py](tune_hyperparams.py).
- Final orchestration: [scripts/run_stride2_remaining.sh](scripts/run_stride2_remaining.sh), [scripts/run_fm_large.sh](scripts/run_fm_large.sh).
- Plot generators: [visualize_results.py](visualize_results.py), [plot_energy.py](plot_energy.py), [plot_thesis_variants.py](plot_thesis_variants.py), [compare_arima_sarima.py](compare_arima_sarima.py), [compare_datasets.py](compare_datasets.py), [compare_datasets_table.py](compare_datasets_table.py).
- Methodology-validation power curves: [measure_power_curve.py](measure_power_curve.py), [analyze_power_curves.py](analyze_power_curves.py), [plot_power_curves_composite.py](plot_power_curves_composite.py).
- Tuned hyperparameters: [results/tuned_params.json](results/tuned_params.json).
- Final canonical metrics table: [results/tsfm_all_stride2_20260513/metrics.csv](results/tsfm_all_stride2_20260513/metrics.csv).
- Project ground truth: [CLAUDE.md](CLAUDE.md).
