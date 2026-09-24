# Stride-2 walk-forward benchmark — full model specification & results report

**Generated:** 2026-05-13
**Source data:** `results/tsfm_all_stride2_20260513/metrics.csv` (consolidated stride-2 metrics for the 15 completed models) + live in-progress SARIMA run `results/sarima_stride2_20260513_102837/`

## Common evaluation protocol (applies to **every** model below)

| Item | Value |
|---|---|
| Dataset | Elia Belgian electricity load, 15-min resolution |
| Span | 2014-12-31 → 2026-03-09 (392,256 observations) |
| Train window | 357,215 obs (≈ 10 years, ending 2025-03-09) |
| Test window | last calendar year = 35,041 obs (2025-03-09 → 2026-03-09) |
| Test split helper | `split_last_n_years(df, test_years=1.0)` in [src/data.py](src/data.py) |
| Walk-forward stride | **2 steps = 30 min** between consecutive forecast origins |
| Number of origins | **17,473** per model (per horizon) |
| Horizons | h = 1 (15 min), h = 4 (1 h), h = 96 (24 h ahead) |
| Hardware | heimdall — 2× AMD EPYC 9455 (192 logical cores), 251 GB RAM, 1× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |
| Energy backends | CodeCarbon (CC) + CarbonTracker (CT) on every run; nvidia-smi (NV) added for GPU runs |
| Imputation | linear interpolation of NaN gaps (`clean_timeseries(impute_method="interpolate")`) |

## Headline accuracy results (MAE in MW, lower = better)

Sorted by h = 96 MAE (the 24 h-ahead horizon, hardest to beat against seasonal-naive):

| Rank | Model | h=1 | h=4 | **h=96** | MAPE h=96 |
|---:|---|---:|---:|---:|---:|
| 1 | **lgbm** (tuned) | 78.5 | 155.1 | **313.4** | 3.38 % |
| 2 | **chronos_bolt_base** | 88.0 | 140.2 | 341.9 | 3.70 % |
| 3 | chronos_bolt_mini | 97.1 | 153.5 | 364.2 | 3.94 % |
| 4 | timesfm_500m | 80.9 | 144.1 | 387.6 | 4.22 % |
| 5 | moirai2_small | 79.6 | 153.7 | 449.3 | 4.87 % |
| 6 | linear (tuned) | 83.0 | 198.7 | 452.3 | 4.95 % |
| 7 | timesfm_200m | 88.0 | 166.4 | 499.0 | 5.40 % |
| 8 | chronos_large | 78.1 | 149.0 | 513.6 | 5.58 % |
| 9 | naive | 517.9 | 518.8 | 515.6 | 5.57 % |
| 10 | chronos_mini | 84.2 | 166.0 | 585.3 | 6.37 % |
| 11 | moirai_large | 101.7 | 205.7 | 765.7 | 8.52 % |
| 12 | moirai_small | 203.6 | 308.0 | 788.4 | 8.65 % |
| 13 | moirai_base | 116.3 | 225.4 | 796.9 | 8.79 % |
| 14 | arima | 86.8 | 228.1 | 798.1 | 8.70 % |
| 15 | lag_llama | 399.3 | 828.1 | 1404.1 | 16.00 % |
|  – | **sarima** *(in progress — see §16)* | — | — | — | — |

## Model specifications

Each section: (a) **what** the model is, (b) **size / source**, (c) the **exact configuration used in this stride-2 run**, (d) **results**, (e) **energy / runtime**.

All classical models share the walk-forward methodology: **fit once on the train window, then advance state per origin without re-fitting** ([src/models.py:67-126](src/models.py#L67-L126)). All foundation models are **zero-shot**: weights are downloaded once, no gradient updates, context is re-sliced at each origin ([src/models_tsfm.py](src/models_tsfm.py)).

---

### 1. `naive` — Seasonal-naive baseline

**Class:** `NaivePredictor` ([src/models.py:44-64](src/models.py#L44-L64))
**Definition:** ŷ[t + h] = y[t + h − s], with s = 96 (one calendar day at 15-min resolution)
**Parameters / hyper-params:** none — pure look-up
**Run id:** `wf_20260512_204729` (results/stride2_remaining_20260512_204727/naive/)
**CLI:** `python run_benchmark.py --models naive --horizons 1 4 96 --stride 2 --test_years 1.0 --energy_tool both --no_plots`

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 517.93 | 745.48 | 5.59 | 17,473 |
| 4 | 518.78 | 745.40 | 5.60 | 17,473 |
| 96 | 515.63 | 742.09 | 5.57 | 17,473 |

**Energy** — fit = 0 (no training); eval = **5.16 s**, CT 3.06 × 10⁻⁵ kWh. CodeCarbon column blank because the script skips CC for the zero-cost fit phase.

Flat across horizons because the formula is independent of h. This is the operational lower bound: any useful model must beat 515 MW MAE at h = 96.

---

### 2. `linear` — Ridge regression, direct multi-step

**Class:** `MLPredictor` wrapping a `DirectForecaster(Ridge)` ([src/models.py:231-340](src/models.py#L231-L340))
**Architecture:** `Pipeline([StandardScaler, Ridge])`, one cloned estimator per horizon (h = 1, 4, 96).
**Feature set (`DEFAULT_LAGS`, [src/models.py:132](src/models.py#L132)):**
- Lags `[0, 1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672]` → current value, sub-hour, intra-day, daily (96), weekly (672) coverage
- Cyclic sin/cos encoding of hour-of-day and day-of-week
- **No holiday feature** (`--use_holidays` was not passed)
- **No seasonal-residual transform** (`--seasonal_residual 0`)

**Hyper-parameter (tuned by Optuna in `tune_hyperparams.py`):**
- `alpha = 34.06457061395274` ([results/tuned_params.json](results/tuned_params.json))

**Run id:** `wf_20260512_204738`
**CLI:** `python run_benchmark.py --models linear --horizons 1 4 96 --stride 2 --test_years 1.0 --tuned_params results/tuned_params.json --energy_tool both --no_plots`

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 82.96 | 114.64 | 0.91 | 17,473 |
| 4 | 198.74 | 257.29 | 2.16 | 17,473 |
| 96 | 452.35 | 583.28 | 4.95 | 17,473 |

**Energy** — fit = 4.93 s / 8.6 × 10⁻⁵ kWh (CC); eval = **23.52 s** / 9.0 × 10⁻⁴ kWh (CC), 5.4 × 10⁻⁴ kWh (CT). CPU-only.

---

### 3. `lgbm` — LightGBM gradient boosting, direct multi-step

**Class:** `MLPredictor` wrapping a `DirectForecaster(LGBMRegressor)` ([src/models.py:342-365](src/models.py#L342-L365))
**Architecture:** one LightGBM regressor per horizon; same lag + calendar feature matrix as `linear`.
**Hyper-parameters (Optuna-tuned, [results/tuned_params.json](results/tuned_params.json)):**

| param | value |
|---|---:|
| `n_estimators` | 488 |
| `num_leaves` | 121 |
| `learning_rate` | 0.0360 |
| `min_child_samples` | 49 |
| `subsample` | 0.778 |
| `colsample_bytree` | 0.826 |
| `verbose` | −1 (suppress) |

**Run id:** `wf_20260512_204811`
**CLI:** `python run_benchmark.py --models lgbm --horizons 1 4 96 --stride 2 --test_years 1.0 --tuned_params results/tuned_params.json --energy_tool both --no_plots`

**Results — best classical model in the benchmark**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | **78.45** | 110.33 | 0.86 | 17,473 |
| 4 | 155.12 | 207.04 | 1.67 | 17,473 |
| 96 | **313.38** | 449.46 | 3.38 | 17,473 |

**Energy** — fit = 29.89 s / 1.74 × 10⁻³ kWh (CC); eval = **26.55 s** / 1.43 × 10⁻³ kWh (CC). CPU-only.

Beats every foundation model except `chronos_bolt_base` at h = 96, and does so at roughly **1/15 of the eval-time energy of timesfm_500m**.

---

### 4. `arima` — Non-seasonal ARIMA(2, 1, 2)

**Class:** `ARIMAPredictor` ([src/models.py:67-126](src/models.py#L67-L126)) over `statsmodels.tsa.SARIMAX`.
**Configuration:**
- `order = (p=2, d=1, q=2)`, `seasonal_order = (0, 0, 0, 0)`, `trend = "n"`
- `enforce_stationarity = False`, `enforce_invertibility = False` (intentional, see [CLAUDE.md](CLAUDE.md))
- MLE optimiser: `lbfgs`, `maxiter = 50`, `disp = False`
- **Training cap: `max_train_obs = 4000`** (last 4,000 obs ≈ 6 weeks). The earlier 350k obs are ignored — by design, ARIMA accuracy is dominated by the recent regime and full-history fitting blows up SARIMAX memory.

**Walk-forward state update** ([src/models.py:110-125](src/models.py#L110-L125)): `results.append(new_obs, refit=False)` — one Kalman step per origin, **no re-optimisation**. ~365× faster than re-fit per origin.

**Run id:** `wf_20260512_204912`
**CLI:** `python run_benchmark.py --models arima --horizons 1 4 96 --stride 2 --test_years 1.0 --arima_order 2 1 2 --maxiter 50 --max_train_obs 4000 --energy_tool both --no_plots`

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 86.78 | 120.84 | 0.96 | 17,473 |
| 4 | 228.09 | 296.43 | 2.49 | 17,473 |
| 96 | 798.13 | 1042.14 | 8.70 | 17,473 |

**Energy** — fit = 4.59 s / 6.3 × 10⁻⁵ kWh (CC); eval = **876.84 s ≈ 14.6 min** / 3.8 × 10⁻² kWh (CC), 1.3 × 10⁻² kWh (CT). CPU-only.

Strong at h = 1 (Kalman state propagates current observation), collapses at h = 96 because the non-seasonal model cannot project the daily cycle 24 h ahead.

---

### 5. `sarima` — Seasonal ARIMA(2, 1, 2)(1, 0, 0, 96) **— RUN CURRENTLY IN PROGRESS**

**Class:** same `ARIMAPredictor` as ARIMA, but with seasonal terms.
**Configuration of the in-progress run** (started 2026-05-13 10:28 UTC):
- `order = (p=2, d=1, q=2)`
- `seasonal_order = (P=1, D=0, Q=0, s=96)` — **seasonal AR(1) at lag 96 (one day), no seasonal differencing**
- `max_train_obs = 4000`, `maxiter = 50`, `lbfgs`
- Same Kalman-update walk-forward as ARIMA

**Why (1, 0, 0, 96) and not (1, 1, 1, 96):**
A first attempt at the seasonal-differencing variant `(1, 1, 1, 96)` was killed earlier today after observation that its Kalman state at s = 96 was growing unboundedly with each `.append()` — RSS went from 8 GB → 28 GB in 300 origins, projected ~1.6 TB at completion. The D = 0 / Q = 0 variant matches the prior thesis SARIMA configuration (run `benchmark_20260310_162746`) and keeps RSS bounded.

**Run id:** `wf_20260513_102838`
**CLI:** `python run_benchmark.py --models sarima --horizons 1 4 96 --stride 2 --test_years 1.0 --sarima_order 2 1 2 --sarima_seasonal_order 1 0 0 96 --maxiter 50 --max_train_obs 4000 --energy_tool both --no_plots`
**Output dir:** `results/sarima_stride2_20260513_102837/`
**Log:** `logs/sarima_stride2/20260513_102837.log`

**Status (snapshot at 14:30 UTC):**
- Fit: **57.7 s** (already complete)
- Walk-forward: ~2,800 / 17,473 origins done (≈ 16 %)
- Pace: ~18–22 origins/min, RSS oscillating 7–15 GB, no memory growth trend
- Projected completion: 14 h from start ≈ tomorrow morning 02:00 UTC

**Results: not yet available** — will be written to `results/sarima_stride2_20260513_102837/wf_20260513_102838/metrics.csv` on completion. Energy (CC + CT) tracked.

---

### 6. `chronos_mini` — Amazon Chronos-T5-mini

**Source:** Ansari et al. 2024, *"Chronos: Learning the Language of Time Series"*
**HF repo:** `amazon/chronos-t5-mini` (~20M params, T5 encoder-decoder)
**Class:** `ChronosForecaster` ([src/models_tsfm.py:41-206](src/models_tsfm.py#L41-L206))
**Configuration:**
- `_CONTEXT_LENGTH = 512` (hard-coded)
- `num_samples = 100` (CLI default `--num_samples 100`, overrides class default 50)
- `torch_dtype = bfloat16`, GPU (CUDA)
- Point forecast = **median across the 100 sample trajectories**

**Run id:** `tsfm_20260512_210912`
**CLI:** `python run_tsfm_benchmark.py --models chronos_mini --horizons 1 4 96 --stride 2 --test_years 1.0 --num_samples 100 --device cuda --gpu_index 0 --energy_tool all --no_plots`

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 84.19 | 114.86 | 0.93 | 17,473 |
| 4 | 165.99 | 219.56 | 1.82 | 17,473 |
| 96 | 585.26 | 790.58 | 6.37 | 17,473 |

**Energy** — fit (= model load) 2.65 s; eval **484.12 s ≈ 8.1 min** — CC 6.78 × 10⁻², CT 7.79 × 10⁻², **NV 5.10 × 10⁻² kWh**.

---

### 7. `chronos_large` — Amazon Chronos-T5-large

**HF repo:** `amazon/chronos-t5-large` (**710M params**, largest T5-based Chronos)
**Class:** `ChronosLargeForecaster` (subclass of `ChronosForecaster`, [src/models_tsfm.py:1018-1023](src/models_tsfm.py#L1018-L1023))
**Configuration:** identical to `chronos_mini` (context 512, 100 samples, bfloat16, CUDA, median).

**Run id:** `tsfm_all_stride2_20260513` (chronos_large rows, dated 2026-05-12 19:45 UTC)
**CLI:** `python run_tsfm_benchmark.py --models chronos_large --horizons 1 4 96 --stride 2 --num_samples 100 --device cuda --energy_tool all --no_plots`

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | **78.08** | 107.22 | 0.86 | 17,473 |
| 4 | 149.02 | 199.64 | 1.64 | 17,473 |
| 96 | 513.61 | 713.64 | 5.58 | 17,473 |

**Energy** — eval **4,767.87 s ≈ 79 min**, CC **0.78 kWh**, CT **0.96 kWh**, NV **0.61 kWh**. The most expensive *non-Lag-Llama* foundation model in the benchmark.

Best h=1 of all models, but slows badly at h=96 because the T5 encoder treats the long-range signal probabilistically rather than autoregressively.

---

### 8. `chronos_bolt_mini` — Amazon Chronos-Bolt-mini (deterministic)

**Source:** Amazon (2024) — Bolt variant of Chronos with **direct quantile output, no sampling**.
**HF repo:** `amazon/chronos-bolt-mini` (~20M params)
**Class:** `ChronosBoltForecaster` ([src/models_tsfm.py:776-881](src/models_tsfm.py#L776-L881))
**Configuration:**
- `_CONTEXT_LENGTH = 2048` (native; 4× longer than T5-based Chronos)
- `quantile_levels = [0.5]` → median is the point forecast (deterministic, no sampling)
- `torch_dtype = bfloat16`, CUDA

**Run id:** `tsfm_20260512_210358`
**CLI:** `python run_tsfm_benchmark.py --models chronos_bolt_mini --horizons 1 4 96 --stride 2 --device cuda --energy_tool all --no_plots`

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 97.11 | 128.89 | 1.06 | 17,473 |
| 4 | 153.49 | 204.95 | 1.67 | 17,473 |
| 96 | 364.18 | 514.58 | 3.94 | 17,473 |

**Energy** — eval **16.68 s** (deterministic → very fast), CC 1.52 × 10⁻³, CT 4.86 × 10⁻⁴, NV 9.95 × 10⁻⁴ kWh.

**Best accuracy / energy ratio of any foundation model.**

---

### 9. `chronos_bolt_base` — Amazon Chronos-Bolt-base

**HF repo:** `amazon/chronos-bolt-base` (~205M params, largest Bolt variant)
**Class:** `ChronosBoltBaseForecaster` (subclass, [src/models_tsfm.py:1025-1029](src/models_tsfm.py#L1025-L1029))
**Configuration:** identical to `chronos_bolt_mini` (context 2048, median, bfloat16, CUDA).

**Run id:** `tsfm_all_stride2_20260513` (chronos_bolt_base, 2026-05-12 18:21 UTC)
**Results — best FM at h=96 overall**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 88.00 | 117.01 | 0.97 | 17,473 |
| 4 | **140.18** | 188.14 | 1.53 | 17,473 |
| 96 | **341.86** | 485.76 | 3.70 | 17,473 |

**Energy** — eval **52.52 s**, CC 7.78 × 10⁻³, CT 7.80 × 10⁻³, NV 6.05 × 10⁻³ kWh. Pareto-optimal among FMs.

---

### 10. `timesfm_200m` — Google TimesFM 1.0 (200M params)

**Source:** Das et al. 2024, *"A decoder-only foundation model for time-series forecasting"*
**HF repo:** `google/timesfm-1.0-200m-pytorch`
**Class:** `TimesFMForecaster` ([src/models_tsfm.py:213-353](src/models_tsfm.py#L213-L353))
**Configuration:**
- Backend `pytorch`, `per_core_batch_size = 32`
- `context_length = 512`, `horizon_len = max(horizons) = 96`
- `point_forecast_mode = "median"`
- `freq = [0]` (sub-hourly tag, matches 15-min Elia)
- CUDA

**Run id:** `tsfm_20260512_211725`
**CLI:** `python run_tsfm_benchmark.py --models timesfm_200m --horizons 1 4 96 --stride 2 --device cuda --energy_tool all --no_plots`

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 87.98 | 118.56 | 0.97 | 17,473 |
| 4 | 166.42 | 220.93 | 1.81 | 17,473 |
| 96 | 499.01 | 677.99 | 5.40 | 17,473 |

**Energy** — eval **328.62 s ≈ 5.5 min**, CC 1.43 × 10⁻², CT 5.09 × 10⁻³, NV 3.06 × 10⁻³ kWh.

---

### 11. `timesfm_500m` — Google TimesFM 2.0 (500M params)

**HF repo:** `google/timesfm-2.0-500m-pytorch`
**Class:** `TimesFM500MForecaster` (subclass, [src/models_tsfm.py:1036-1088](src/models_tsfm.py#L1036-L1088))
**Configuration:**
- `num_layers = 50` (up from 20 in v1.0)
- `use_positional_embedding = False` (RoPE-only)
- `context_length = 1024` (the "safe-optimised" default — covers ~10.6 days of 15-min data; halves attention compute vs the 2048 native max)
- `per_core_batch_size = 32`, `horizon_len = 96`, `point_forecast_mode = "median"`, CUDA

**Run id:** `tsfm_all_stride2_20260513` (timesfm_500m, 2026-05-12 20:10 UTC)

**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 80.94 | 110.07 | 0.89 | 17,473 |
| 4 | 144.12 | 192.51 | 1.57 | 17,473 |
| 96 | 387.65 | 531.91 | 4.22 | 17,473 |

**Energy** — eval **1,479.92 s ≈ 24.7 min**, CC 6.45 × 10⁻², CT 2.13 × 10⁻², NV 1.33 × 10⁻² kWh.

3× more energy than `timesfm_200m` for a ~22 % MAE-at-h=96 improvement.

---

### 12. `lag_llama` — Lag-Llama (time-series-foundation-models)

**Source:** Rasul et al. 2023, *"Lag-Llama: Towards Foundation Models for Time Series Forecasting"*
**HF repo:** `time-series-foundation-models/Lag-Llama` (LLaMA-style decoder, ~2M params core)
**Class:** `LagLlamaForecaster` ([src/models_tsfm.py:360-591](src/models_tsfm.py#L360-L591))
**Configuration:**
- `context_length = 256` (>> native 32; RoPE linear scaling factor = (256 + 96) / 32 ≈ 11.0)
- `num_samples = 100`; median of sampled trajectories = point forecast
- `batch_size = 16`
- Architecture hparams read from checkpoint at load time: `input_size=1`, `n_layer=4`, `n_embd_per_head=64`, `n_head=4`, `scaling="robust"`, `time_feat=True`
- Backend: GluonTS with PyTorch ≥ 2.6 (custom `weights_only=False` patch for the legacy Lightning checkpoint, [src/models_tsfm.py:443-451](src/models_tsfm.py#L443-L451))
- CUDA

**Run id:** `tsfm_20260512_215941`
**CLI:** `python run_tsfm_benchmark.py --models lag_llama --horizons 1 4 96 --stride 2 --num_samples 100 --device cuda --energy_tool all --no_plots`

**Results — worst model in the benchmark**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 399.34 | 518.16 | 4.64 | 17,473 |
| 4 | 828.10 | 1029.77 | 9.57 | 17,473 |
| 96 | 1404.08 | 1783.50 | 16.00 | 17,473 |

**Energy** — eval **3,794.07 s ≈ 63 min**, CC **0.65 kWh**, CT **0.82 kWh**, NV **0.52 kWh**.

Worst accuracy + second-worst energy. Lag-Llama's native context of 32 is structurally too short for 96-step seasonal forecasting; RoPE scaling helps but does not close the gap.

---

### 13. `moirai_small` — Salesforce Moirai-1.0-R-small

**Source:** Woo et al. 2024, *"Unified Training of Universal Time Series Forecasting Transformers"*
**HF repo:** `Salesforce/moirai-1.0-R-small` (~91M params)
**Class:** `MoiraiForecaster` ([src/models_tsfm.py:598-769](src/models_tsfm.py#L598-L769))
**Configuration:**
- `context_length = 512`, `num_samples = 100`, `batch_size = 16`
- `patch_size = "auto"` (model picks based on context + horizon)
- `target_dim = 1`, no exogenous features
- Backend: GluonTS with `uni2ts`. CUDA.

**Run id:** `tsfm_20260512_210452`
**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 203.65 | 339.19 | 2.31 | 17,473 |
| 4 | 307.97 | 422.49 | 3.45 | 17,473 |
| 96 | 788.45 | 994.76 | 8.65 | 17,473 |

**Energy** — eval **96.53 s**, CC 7.13 × 10⁻³, CT 5.83 × 10⁻³, NV 3.87 × 10⁻³ kWh.

---

### 14. `moirai_base` — Salesforce Moirai-1.0-R-base

**HF repo:** `Salesforce/moirai-1.0-R-base` (~311M params)
**Class:** `MoiraiBaseForecaster` (subclass — sets `variant="base"`, otherwise identical config)
**Configuration:** context 512, 100 samples, batch 16, patch "auto", CUDA.

**Run id:** `tsfm_20260512_210637`
**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 116.29 | 196.33 | 1.30 | 17,473 |
| 4 | 225.43 | 309.02 | 2.47 | 17,473 |
| 96 | 796.86 | 1005.87 | 8.79 | 17,473 |

**Energy** — eval **146.90 s**, CC 1.33 × 10⁻², CT 1.23 × 10⁻², NV 8.29 × 10⁻³ kWh.

---

### 15. `moirai_large` — Salesforce Moirai-1.0-R-large

**HF repo:** `Salesforce/moirai-1.0-R-large` (~1.1B params, largest published Moirai-1.x)
**Class:** `MoiraiLargeForecaster` (subclass)
**Configuration:** identical interface to small / base — context 512, 100 samples, batch 16, patch "auto", CUDA.

**Run id:** `tsfm_all_stride2_20260513` (moirai_large, 2026-05-12 18:26 UTC)
**Results**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 101.71 | 173.36 | 1.14 | 17,473 |
| 4 | 205.73 | 279.23 | 2.26 | 17,473 |
| 96 | 765.75 | 940.18 | 8.52 | 17,473 |

**Energy** — eval **262.11 s**, CC 2.87 × 10⁻², CT 3.01 × 10⁻², NV 1.97 × 10⁻² kWh.

In this benchmark, Moirai-1.0 **scaling does not pay off**: large only marginally beats small/base at h = 96, while costing 4× the energy. Almost certainly because the model is patch-based and the small daily cycle of electricity load is captured equally well by the smaller variant.

---

### 16. `moirai2_small` — Salesforce Moirai-2.0-R-small ("Less Is More")

**Source:** Liu et al. 2025, *"When Less Is More"* (Moirai-2.0)
**HF repo:** `Salesforce/moirai-2.0-R-small` (~11M params — smaller than Moirai-1 small)
**Class:** `Moirai2SmallForecaster` ([src/models_tsfm.py:900-1011](src/models_tsfm.py#L900-L1011))
**Architectural differences from Moirai-1.x:**
- **Deterministic quantile heads** (9 quantiles output directly — no Monte-Carlo sampling)
- Fixed internal patching (no `patch_size` parameter)
- Smaller model that *outperforms* all three Moirai-1.0-R variants in this benchmark

**Configuration:** `context_length = 512`, `batch_size = 16`, median (q=0.5) as point forecast, CUDA.

**Run id:** `tsfm_20260512_210423`
**Results — best Moirai variant, top-5 overall**

| h | MAE | RMSE | MAPE % | n |
|---:|---:|---:|---:|---:|
| 1 | 79.64 | 109.97 | 0.88 | 17,473 |
| 4 | 153.67 | 207.42 | 1.68 | 17,473 |
| 96 | 449.31 | 625.96 | 4.87 | 17,473 |

**Energy** — eval **19.84 s**, CC 2.14 × 10⁻³, CT 1.12 × 10⁻³, NV 9.39 × 10⁻⁴ kWh.

Beats Moirai-1.0-large at every horizon while using **13× less energy** and ~100× fewer parameters.

---

## Energy ranking (eval phase only, lower = better)

CodeCarbon kWh, eval phase, sorted ascending:

| Rank | Model | Eval s | CC kWh | CT kWh | NV kWh | Device |
|---:|---|---:|---:|---:|---:|---|
| 1 | naive | 5.2 | n/a | 3.1 × 10⁻⁵ | – | CPU |
| 2 | linear | 23.5 | 9.0 × 10⁻⁴ | 5.4 × 10⁻⁴ | – | CPU |
| 3 | chronos_bolt_mini | 16.7 | 1.5 × 10⁻³ | 4.9 × 10⁻⁴ | 1.0 × 10⁻³ | GPU |
| 4 | lgbm | 26.6 | 1.4 × 10⁻³ | 3.3 × 10⁻⁴ | – | CPU |
| 5 | moirai2_small | 19.8 | 2.1 × 10⁻³ | 1.1 × 10⁻³ | 9.4 × 10⁻⁴ | GPU |
| 6 | moirai_small | 96.5 | 7.1 × 10⁻³ | 5.8 × 10⁻³ | 3.9 × 10⁻³ | GPU |
| 7 | chronos_bolt_base | 52.5 | 7.8 × 10⁻³ | 7.8 × 10⁻³ | 6.0 × 10⁻³ | GPU |
| 8 | moirai_base | 146.9 | 1.3 × 10⁻² | 1.2 × 10⁻² | 8.3 × 10⁻³ | GPU |
| 9 | timesfm_200m | 328.6 | 1.4 × 10⁻² | 5.1 × 10⁻³ | 3.1 × 10⁻³ | GPU |
| 10 | moirai_large | 262.1 | 2.9 × 10⁻² | 3.0 × 10⁻² | 2.0 × 10⁻² | GPU |
| 11 | arima | 876.8 | 3.8 × 10⁻² | 1.3 × 10⁻² | – | CPU |
| 12 | timesfm_500m | 1,479.9 | 6.5 × 10⁻² | 2.1 × 10⁻² | 1.3 × 10⁻² | GPU |
| 13 | chronos_mini | 484.1 | 6.8 × 10⁻² | 7.8 × 10⁻² | 5.1 × 10⁻² | GPU |
| 14 | lag_llama | 3,794.1 | 6.5 × 10⁻¹ | 8.2 × 10⁻¹ | 5.2 × 10⁻¹ | GPU |
| 15 | chronos_large | 4,767.9 | 7.8 × 10⁻¹ | 9.6 × 10⁻¹ | 6.1 × 10⁻¹ | GPU |
|  – | sarima | (TBD) | (TBD) | (TBD) | – | CPU |

## Pareto frontier — accuracy at h=96 vs energy

Models that are **not dominated** (no other model is both more accurate at h=96 AND cheaper to evaluate):

1. **naive** — cheapest, accuracy floor
2. **lgbm** — best accuracy / cheap CPU
3. **chronos_bolt_mini** — best FM accuracy / energy
4. **chronos_bolt_base** — best h=96 of the FMs

All other models are dominated by at least one of these four.

## Files and run identifiers (audit trail)

- **Classical 15-model combined metrics:** `results/tsfm_all_stride2_20260513/metrics.csv`
- **Per-model raw runs (1):** `results/all_best_stride2_20260511_215221/{naive,linear,lgbm,arima,chronos_bolt_mini,lag_llama,moirai_small,timesfm_200m}/wf_*|tsfm_*/`
- **Per-model raw runs (2 — the "last" stride-2 runs cited above):** `results/stride2_remaining_20260512_204727/{naive,linear,lgbm,arima,chronos_bolt_mini,chronos_mini,lag_llama,moirai2_small,moirai_base,moirai_small,timesfm_200m}/`
- **Large-FM raw runs:** `results/tsfm_fm_large_stride2_20260512/` (chronos_bolt_base, chronos_large, moirai_large, timesfm_500m)
- **SARIMA (running):** `results/sarima_stride2_20260513_102837/wf_20260513_102838/` — metrics + predictions written on completion
- **Tuned hyper-parameters:** [results/tuned_params.json](results/tuned_params.json) (linear, lgbm only)
- **Codebase entry points:** [run_benchmark.py](run_benchmark.py), [run_tsfm_benchmark.py](run_tsfm_benchmark.py)
- **Model definitions:** [src/models.py](src/models.py) (classical), [src/models_tsfm.py](src/models_tsfm.py) (foundation)
- **Walk-forward + metrics helpers:** [src/data.py](src/data.py), [src/metrics.py](src/metrics.py), [src/evaluate.py](src/evaluate.py)
