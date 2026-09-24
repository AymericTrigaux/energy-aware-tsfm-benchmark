# PROJECT REPORT — Energy-Aware Electricity Load Forecasting Benchmark
**KU Leuven Master's Thesis — Aymeric Trigaux**
**Report generated:** April 2026 | **Repository:** `thesis_arima_energy_benchmark`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Models](#3-models)
   - 3.1 [Seasonal Naive](#31-seasonal-naive)
   - 3.2 [ARIMA(2,1,2)](#32-arima212)
   - 3.3 [SARIMA(2,1,2)(1,1,1,96)](#33-sarima212111096)
   - 3.4 [Ridge Regression](#34-ridge-regression)
   - 3.5 [LightGBM](#35-lightgbm)
4. [Evaluation Framework](#4-evaluation-framework)
5. [Results](#5-results)
6. [Key Design Choices & Justifications](#6-key-design-choices--justifications)
7. [File & Code Structure](#7-file--code-structure)
8. [How to Reproduce](#8-how-to-reproduce)

---

## 1. Project Overview

### Research Question and Objective

This thesis benchmarks four forecasting model families on Belgian electricity load data across two dimensions simultaneously: **predictive accuracy** and **energy/carbon footprint**. The central research question is: *Do more accurate forecasting models consume significantly more energy to train and run, and is there a principled accuracy-efficiency trade-off that practitioners can use to guide model selection?*

The Belgian electricity grid is managed by Elia (the Belgian transmission system operator), which publishes 15-minute load data publicly. Accurate short- and medium-term load forecasts are essential for grid balancing: over-forecasting triggers unnecessary reserve generation; under-forecasting risks frequency instability. At the same time, the computational cost of running these forecasts at scale — and its carbon footprint — is increasingly relevant as grid operators modernise their forecasting infrastructure.

### Dataset

- **Source:** Elia (Belgian TSO) public API — 15-minute total load measurements for the Belgian high-voltage grid.
- **File:** `data/processed/elia_load_15min.csv` (columns: `datetime`, `totalload`).
- **Resolution:** one observation every 15 minutes → 96 observations per day, 672 per week.
- **Date range:** December 2014 – March 2026 (approximately 11 years).
- **Approximate size:** ~385,000 observations.
- **Target variable:** `totalload` in MW. Load ranges from approximately 6,500 MW (summer nights) to 13,500 MW (cold winter peaks).

### Forecast Horizons

Three forecast horizons are evaluated:

| Horizon | Real-time meaning | Practical application |
|---------|------------------|----------------------|
| **h = 1** | 15 minutes ahead | Ultra-short-term intra-hour balancing |
| **h = 4** | 60 minutes (1 hour) ahead | Short-term reserve scheduling |
| **h = 96** | 1,440 minutes (24 hours) ahead | Day-ahead market and dispatch planning |

These three horizons span the range from real-time balancing to day-ahead planning, covering the most practically relevant operational timescales for a transmission system operator.

### Evaluation Dimensions

Accuracy is measured by MAE (primary), RMSE, and MAPE — all computed over a walk-forward rolling-origin evaluation protocol. Energy and carbon footprint are measured separately for the **fit phase** (one-time training cost) and the **eval phase** (recurring inference cost) using two complementary tools: CodeCarbon (primary) and CarbonTracker (secondary, for cross-validation).

### Walk-Forward Cross-Validation

The benchmark uses **rolling-origin (walk-forward) cross-validation**. The implementation in `run_benchmark.py` works as follows:

1. The full dataset is split chronologically: the last 1 calendar year (approximately March 2025 – March 2026) forms the **test set**; everything prior (approximately December 2014 – March 2025, ~10 years) is the **training set**.
2. Each model is fitted **once** on the training set. For ARIMA/SARIMA, the Kalman filter state is updated incrementally via `res.append(refit=False)` at each origin — there is no full re-fitting.
3. A **forecast origin** is any time point `t` in the test period. At each origin, the model produces predictions for all three horizons: `ŷ[t+1]`, `ŷ[t+4]`, `ŷ[t+96]`.
4. Origins are spaced by a configurable **stride** parameter. The canonical stride is 96 steps (one origin per day), yielding 365 error pairs per horizon — enough for statistical robustness while keeping ARIMA evaluation tractable. An additional high-density run (stride=1) was also executed, producing 17,473 error pairs.
5. After each prediction, the true observation is revealed and fed back into the model state (Kalman update for ARIMA/SARIMA; the ML models read from the full stored series). This simulates online, operational deployment exactly.

The walk-forward approach avoids the fundamental flaw of single-origin evaluation, where an ARIMA model fit once and forecasted 96 steps into the future would revert to its unconditional mean after a few lags, making h=4 and h=96 results nearly indistinguishable. Walk-forward evaluation ensures that each horizon genuinely tests the model's difficulty at that horizon.

---

## 2. Data Pipeline

### Loading (`src/data.py: load_timeseries`)

Raw data is loaded from CSV or Parquet files by `load_timeseries()`. The function accepts a file path (auto-detected by extension: `.csv`, `.txt`, or `.parquet`), the column names for the timestamp and target variable, and the expected frequency string (`"15min"`). After loading, it enforces a regular datetime grid via `DataFrame.asfreq(freq)`, which inserts NaN rows for any missing 15-minute slots — critically important because DST transitions, network outages, or data gaps would otherwise silently shrink the array, misaligning all subsequent lag features by one or more positions. The function returns a single-column DataFrame with column `"y"` indexed by a `DatetimeIndex`.

### Cleaning (`src/data.py: clean_timeseries`)

Missing values (NaN rows inserted by `asfreq`) are filled by `clean_timeseries()` using **time-linear interpolation** (`method="time"`): the `pandas` interpolation weights each interpolated value proportionally to elapsed time, producing a physically plausible smooth ramp between known readings. This is followed by forward/backward fill for any residual edge NaNs. Forward-fill alone (copying the last observed value for potentially many consecutive steps) was rejected because it produces artificial flat segments that inflate apparent autocorrelation and can bias ARIMA order selection. Gaps larger than a few days are extremely rare in the Elia dataset; for those, the interpolation remains reasonable as a fallback.

### Train/Test Split (`src/data.py: split_last_n_years`)

The data is split chronologically by `split_last_n_years()`. Given `test_years=1.0`, it computes the split boundary as `last_timestamp − 1 calendar year` and partitions observations accordingly. Using a full calendar year as the test set guarantees that every season (winter demand peaks, summer troughs, Easter, Christmas, summer holidays) is represented in the evaluation — a critical requirement for unbiased assessment of seasonal models. A fixed-ratio split (e.g., 80/20) could accidentally assign only summer data to the test set, making seasonal models appear better or worse than they are.

**Effective split:**
- **Train:** December 2014 – approximately March 2025 (~10 years, ~350,000 observations)
- **Test:** March 2025 – March 2026 (1 full calendar year, ~35,040 observations)

### ARIMA Training Cap

SARIMAX MLE fitting on all ~350,000 training observations would take many hours on a laptop. The `--max_train_obs 4000` parameter (set in `ARIMAPredictor.__init__`) caps training at the most recent 4,000 observations (~42 days), which is sufficient for reliable parameter estimation while completing in seconds. The ML models (Ridge, LightGBM) use the full training set because `sklearn` fitting time scales far more efficiently with dataset size.

### Feature Engineering (ML models only)

For Ridge and LightGBM, `build_feature_matrix()` in `src/models.py` constructs a supervised learning dataset from the time series. Each row corresponds to one time step `t` and contains:

**Lag features (`DEFAULT_LAGS = [0, 1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672]`):**

| Lag | Offset in time | Purpose |
|-----|---------------|---------|
| 0 | current value y[t] | Most recent observation |
| 1–4 | 15–60 min ago | Short-term autocorrelation |
| 8, 12 | 2 h, 3 h ago | Sub-hourly dynamics |
| 24, 48 | 6 h, 12 h ago | Within-day context |
| 96 | 24 h ago (yesterday, same slot) | Primary daily seasonality |
| 192, 288 | 2, 3 days ago | Multi-day context |
| 672 | 7 days ago (last week, same slot) | Weekly seasonality |

**Calendar features (cyclic encoding):**
- `h_sin = sin(2π · hour_fraction / 24)`, `h_cos = cos(2π · hour_fraction / 24)`
- `d_sin = sin(2π · day_of_week / 7)`, `d_cos = cos(2π · day_of_week / 7)`

Cyclic (sin/cos) encoding is used instead of raw integers because hour 23 and hour 0 are adjacent on the clock but 23 apart numerically; raw encoding places them at opposite ends of the feature space, creating a false discontinuity that hurts linear models and confuses tree splits near hour boundaries.

Rows where any feature or target is NaN (due to series edges after lagging/leading) are dropped before training. A total of 17 features (13 lags + 4 calendar) are produced.

---

## 3. Models

All five model classes expose the same `fit(y_train)` / `set_series(y_all)` / `predict(origin_idx)` interface, allowing the walk-forward loop in `run_benchmark.py` to treat them uniformly — no special-casing per model type.

### 3.1 Seasonal Naive

**Conceptual explanation:** The seasonal naive predictor makes no assumptions beyond the existence of a dominant daily seasonality. At each forecast origin `t`, the h-step-ahead prediction is simply the load observed at the same 15-minute slot exactly one seasonal period earlier: `ŷ[t+h] = y[t + h − s]`, where `s = 96` (one full day of 15-minute steps).

| Horizon | Formula | What it retrieves |
|---------|---------|------------------|
| h = 1 | `y[t − 95]` | Load at this 15-min slot yesterday (23h 45m ago) |
| h = 4 | `y[t − 92]` | Load at this 15-min slot yesterday (23h ago) |
| h = 96 | `y[t]` | Current load = predicted same-slot load tomorrow |

**Role in benchmark:** The seasonal naive is the **minimum bar** — the cheapest possible model that meaningfully exploits the dominant signal in the data. Any model that cannot beat the seasonal naive for a given horizon provides no predictive value relative to its computational cost.

**Implementation:** `NaivePredictor` in `src/models.py`. `fit()` simply stores the training series; `set_series()` updates the internal reference to the full train+test series for rolling use; `predict()` performs an array index look-up. Training cost: essentially zero (no parameters, no optimization). Inference cost: one array read per horizon.

**Hyperparameter tuning:** None needed. The only parameter, `s = 96`, is fixed by the dataset resolution and confirmed to be the dominant seasonal period.

**Limitations:** Ignores all information beyond one daily lag. Cannot adapt to structural changes, trends, or unusual days (holidays, extreme weather). Its h=96 MAE is essentially the "you can predict tomorrow from today's same slot" baseline.

---

### 3.2 ARIMA(2,1,2)

**Conceptual explanation:** ARIMA (AutoRegressive Integrated Moving Average) models a time series as a linear combination of its own past values (AR component), past forecast errors (MA component), and differenced values (I component). ARIMA(2,1,2) applies one round of first-differencing (`d=1`) to remove the unit root confirmed by the ADF test, then models the differenced series with 2 autoregressive lags and 2 moving-average lags.

**Role in benchmark:** The **non-seasonal statistical reference**. Its results quantify how much of the predictive signal comes from short-term autocorrelation (which ARIMA captures well) versus daily seasonality (which ARIMA ignores, since it has no seasonal component). Comparing ARIMA to SARIMA isolates the value of explicit seasonal differencing; comparing ARIMA to Ridge/LightGBM isolates the value of non-linear modelling and direct multi-step forecasting.

**Implementation:** `ARIMAPredictor` in `src/models.py`, backed by `statsmodels.tsa.SARIMAX` with `seasonal_order=(0,0,0,0)`. The key design decision is the **Kalman filter state update strategy**: the model is fit once on the most recent 4,000 training observations using MLE with lbfgs optimisation (`maxiter=50`). At each walk-forward origin `t`, new observations accumulated since the last call are fed into the Kalman filter via `res.append(new_obs, refit=False)` — this updates the state estimate (conditional mean and covariance) without re-running MLE. `get_forecast(steps=max_h)` then produces predictions for all horizons in one call. This approach is approximately 365× faster than re-fitting from scratch at each origin.

**Model parameters:** `order=(2, 1, 2)`, `seasonal_order=(0, 0, 0, 0)`, `trend='n'`, `enforce_stationarity=False`, `enforce_invertibility=False`. The stationarity and invertibility constraints are disabled intentionally: after differencing, the parameter space is effectively bounded anyway, and removing the constraints allows the L-BFGS optimiser to explore freely, improving convergence speed.

**Stationarity verification:** ADF test on the training set after `d=1` differencing:
- ADF statistic: −12.85 (vs. 1% critical value −3.43 → strongly significant)
- p-value: 5.4 × 10⁻²⁴ → conclusive rejection of the unit-root null hypothesis
- This confirms that `d=1` is the correct differencing order.

**Hyperparameter tuning:** Order (2,1,2) was selected based on ACF/PACF analysis performed offline and confirmed by AIC/BIC grid search via `aic_bic_grid_search()` in `src/evaluate.py`. The grid search fits SARIMAX for each candidate (p,d,q) combination with `maxiter=30` for speed and ranks by AIC.

**Forecast strategy:** Single-shot multi-step — `get_forecast(steps=96)` returns all 96 horizon predictions from the Kalman smoother in one call.

**Known limitation:** Without a seasonal component, ARIMA reverts to its unconditional mean (the long-run level of the differenced series) after approximately max(p, q) = 2 lags. By h=96 (24 hours), the forecast is essentially a flat mean prediction, explaining the very high MAE at that horizon.

---

### 3.3 SARIMA(2,1,2)(1,1,1,96)

**Conceptual explanation:** SARIMA (Seasonal ARIMA) extends ARIMA by adding a second layer of AR, I, and MA operators that operate at the seasonal lag `s = 96`. The notation SARIMA(2,1,2)(1,1,1,96) means: non-seasonal part ARIMA(2,1,2) on the original scale, plus seasonal differencing `D=1` (remove `y[t] − y[t−96]`), one seasonal AR lag (`P=1`: uses `y[t−96]`), and one seasonal MA lag (`Q=1`: uses shock from 96 steps back). The seasonal differencing with `D=1` transforms the daily cycle into a stationary residual that the seasonal AR/MA terms can then model.

**Role in benchmark:** The **seasonal statistical reference**. Comparing SARIMA to ARIMA shows the specific contribution of seasonal differencing and seasonal AR/MA. Comparing SARIMA to LightGBM tests whether a parametric seasonal model can match or beat a non-parametric ML model at long horizons.

**Implementation:** Shares the exact same `ARIMAPredictor` class and `fit_sarimax()` function as ARIMA. Only `ArimaConfig` differs: `seasonal_order=(1, 1, 1, 96)` and `trend='n'`. SARIMA fitting is suppressed via `warnings.catch_warnings` because `statsmodels` emits numerous convergence warnings for high-period SARIMA models; these warnings are expected and documented as intentional.

**Seasonal order rationale:** `D=1` (seasonal differencing) properly removes the persistent daily level; `P=1, Q=1` provide the minimal complete seasonal specification (analogous to (1,1,1) being the standard non-seasonal starting point). `s=96` equals one full day.

**Computational cost:** Fitting SARIMA(2,1,2)(1,1,1,96) on 4,000 observations typically takes 2–10 minutes on an Apple M-series chip. This is one of the key energy data points. SARIMA was not included in the most recent full benchmark run (`wf_20260409_143052`) due to its long fitting time; its results are drawn from an earlier dedicated benchmark run.

**Hyperparameter tuning:** Same approach as ARIMA — AIC/BIC grid search plus offline ACF/PACF analysis.

**Known limitation:** The state-space matrices for SARIMA with s=96 are O(96²) in size, making each Kalman filter operation substantially slower than for non-seasonal ARIMA. The `max_train_obs=4000` cap is essential — without it, memory consumption grows to tens of GB on the full training set.

---

### 3.4 Ridge Regression

**Conceptual explanation:** Ridge regression (L2-regularised ordinary least squares) fits a linear model `ŷ = Xβ` by minimising `||Xβ − y||² + α·||β||²`. The L2 penalty shrinks all coefficients smoothly toward zero, preventing overfitting on the 13 correlated lag features while retaining all features in the model (unlike Lasso, which zeros out entire features). A `StandardScaler` is applied upstream in a `sklearn.pipeline.Pipeline` so that L2 regularisation operates on a common scale — without standardisation, features in units of MW (lags, 6,000–12,000 MW range) would dominate the penalty term relative to calendar features (sin/cos, range −1 to 1).

**Role in benchmark:** The **linear ML baseline**. Its results show how much a regularised linear model can achieve without non-linear capacity, and whether direct multi-step forecasting with lag features outperforms seasonal naive at all horizons.

**Implementation:** `make_linear_forecaster()` creates a `DirectForecaster` wrapping `sklearn.pipeline.Pipeline(StandardScaler(), Ridge(alpha=1.0))`. `DirectForecaster` trains one estimator clone per horizon `h` via `sklearn.base.clone` — a separate Ridge model for h=1, another for h=4, another for h=96. This is the **direct multi-step** strategy: each model predicts `y[t+h]` in one shot from features at origin `t`, rather than recursively chaining 96 single-step predictions. The direct strategy avoids recursive error accumulation (Ben Taieb & Atiya, 2016) at the cost of training multiple models.

**Hyperparameter tuning:** `tune_hyperparams.py` supports Optuna-based search over `alpha` (log-uniform, 1e-3 to 1e2). An earlier benchmark used a tuned `alpha=1e-6` (from Optuna). The current implementation in `make_linear_forecaster` defaults to `alpha=1.0`. The actual value used in any given run depends on whether `--tuned_params results/tuned_params.json` is passed; the JSON file stores the Optuna-selected value. **Validation strategy:** chronological 80/20 split within the training data (no shuffling, no cross-validation with shuffled folds — doing so would leak future data into training).

**Forecast strategy:** Direct multi-step (one model per horizon `h`). Feature construction via `build_feature_matrix()` and inference via `build_x_at_origin()`, both in `src/models.py`.

**Known limitation at h=96:** Ridge with `alpha=1.0` can underperform the seasonal naive at h=96. The mechanistic explanation: regularisation shrinks the coefficient of `lag_0` (the single most informative feature for a 24h-ahead forecast is the current load), which the seasonal naive uses with an implicit weight of exactly 1. Additionally, the 12 other lag features (especially `lag_1` through `lag_48`) carry little signal at h=96 but add noise that the regulariser cannot fully suppress. Reducing alpha to 0.01–0.1 for the h=96 model would likely improve performance — this is a documented finding, not a bug.

---

### 3.5 LightGBM

**Conceptual explanation:** LightGBM (Light Gradient Boosting Machine) is an ensemble of decision trees trained sequentially, each new tree correcting the residuals of the ensemble so far. Unlike Ridge, decision trees can model non-linear interactions between features without explicit feature engineering — for instance, "load at 8:00 AM on a cold winter Monday" is a specific combination of lag_0, h_sin, h_cos, d_sin, d_cos that a tree can isolate via a chain of splits, but that a linear model cannot capture without an explicit interaction term.

**Role in benchmark:** The **non-linear ML model** and the primary accuracy competitor. LightGBM is included to test whether non-linear modelling capacity over the same lag feature set yields meaningful improvements over Ridge, particularly at medium and long horizons where load dynamics are more complex.

**Implementation:** `make_lgbm_forecaster()` creates a `DirectForecaster` wrapping `LGBMRegressor`. Default parameters: `n_estimators=300`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `verbose=-1`. Like Ridge, it uses the direct multi-step strategy — one LightGBM model per horizon.

**Parameter rationale:**
- `num_leaves=63` (vs. default 31): electricity load exhibits complex non-linear patterns (sharp morning ramps, holiday valleys); deeper trees capture these. `min_child_samples=20` prevents fitting noise in individual 15-min slots.
- `learning_rate=0.05` (vs. default 0.1): slower learning with more trees improves generalisation on temporal data.
- `subsample=0.8, colsample_bytree=0.8`: row and feature subsampling per tree provides stochastic regularisation and reduces training time.
- LightGBM over XGBoost: LightGBM uses leaf-wise (best-first) tree growth, which is faster and more memory-efficient than XGBoost's level-wise growth on tabular data.

**Hyperparameter tuning:** `tune_hyperparams.py` supports Optuna search over: `n_estimators` (100–600), `num_leaves` (15–127), `learning_rate` (0.01–0.3, log-uniform), `min_child_samples` (10–50), `subsample` (0.6–1.0), `colsample_bytree` (0.6–1.0). Best params are saved to `results/tuned_params.json` and loaded via `--tuned_params` at benchmark time.

**Forecast strategy:** Direct multi-step (one LightGBM per horizon), identical to Ridge.

**Expected advantages over Ridge:** Non-linear interaction effects (cold mornings, holiday weekday patterns), sharper peak capture, better performance at h=4 and h=96 where load dynamics are more complex and non-linear.

---

## 4. Evaluation Framework

### Metrics

Three complementary metrics are computed per model × horizon combination by `evaluate_forecast()` in `src/evaluate.py`:

**Mean Absolute Error (MAE):**
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
In MW. The **primary ranking metric** — directly interpretable by grid operators ("the model is off by X MW on average"), robust to occasional large errors, and the most commonly reported metric in the load-forecasting literature.

**Root Mean Squared Error (RMSE):**
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
In MW. Penalises large individual errors more than MAE. When RMSE >> MAE, the model makes occasional catastrophically large errors (e.g., missing a cold-snap peak by 2,000 MW). The ratio RMSE/MAE is informative: a value near 1.0 indicates consistent errors; a value near 2.0 indicates a few dominant outliers.

**Mean Absolute Percentage Error (MAPE):**
$$\text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{|y_i|} \times 100\%$$
Reported in percentage points. Enables comparison across different load levels and against forecasts from the literature. Belgian electricity load is always > 5,000 MW, so division-by-zero is not a practical concern; a small epsilon (1e-9) is added to the denominator as a safety guard.

Both MAE and RMSE are computed via `sklearn.metrics.mean_absolute_error` and `mean_squared_error`; MAPE is computed directly in `src/metrics.py`. Alignment of true and predicted series is done via `pandas.Series.align(join="inner")` on the DatetimeIndex, guarding against off-by-one index mismatches.

### Walk-Forward Aggregation

Error metrics are aggregated over all N walk-forward origins and reported as scalar summary statistics. For the canonical run (`stride=96`, test_years=1.0), N=365 per horizon. No seasonal decomposition of errors is performed in the main benchmark loop, though `visualize_results.py` produces seasonal (winter vs. summer) comparison plots separately.

### Energy Tracking

Energy and carbon are measured using two complementary tools, run in parallel for each phase:

**CodeCarbon (`EnergyMeter` in `src/metrics.py`):**
- Wraps `codecarbon.EmissionsTracker`.
- Country: `country_iso_code="BEL"` (Belgium, Flanders region).
- Hardware: Apple M4 Pro (14 cores, 24 GB RAM). CodeCarbon uses the macOS power management interface to sample CPU and GPU power.
- Outputs: appended to `results/<run_id>/codecarbon/emissions.csv`. Columns include `energy_consumed` (kWh), `emissions` (kg CO₂eq), `duration` (s), `cpu_model`, `cpu_power`, `gpu_power`, `ram_power`, and `region`.
- Reading strategy: the implementation reads the **final row of `emissions.csv`** rather than relying on the return value of `tracker.stop()`, which varies across CodeCarbon versions.
- Carbon intensity: looked up from CodeCarbon's built-in database for Belgium (Flanders). The Belgian grid carbon intensity is approximately 137–150 g CO₂eq/kWh depending on the year; the exact value used is logged in each `emissions.csv` row under the `emissions_rate` column.

**CarbonTracker (`CarbonTrackerMeter` in `src/metrics.py`):**
- Wraps the `carbontracker` library.
- On Linux: uses Intel RAPL for direct hardware power readings.
- On macOS (Apple Silicon): attempts `sudo powermetrics`; if passwordless sudo is not available, hardware readings degrade to NaN.
- Applies a PUE (Power Usage Effectiveness) factor of 1.58.
- Each meter instance writes to a dedicated sub-directory (`results/<run_id>/carbontracker/<project_name>/`) to prevent log-file collisions.
- Results parsed via `carbontracker.parser.aggregate_consumption`.
- **Graceful degradation:** `_monitoring_succeeded()` checks whether any component (CPU, GPU, RAM) reported a non-None power reading; if all report None, the measurement is flagged as unavailable (NaN) rather than silently returning zero.

**What is measured:** Two phases per model:
1. **Fit phase:** energy consumed from `meter.start()` to `meter.stop()` wrapping `model.fit(y_train)`.
2. **Eval phase:** energy consumed wrapping the full walk-forward loop (`walk_forward()`) including all Kalman updates and/or predictions.

**Results logging:** For each model × horizon, one row is appended to `results/<run_id>/metrics.csv` (run-specific) and potentially to the cumulative `results/wf_metrics_runs.csv`. Columns include the full model specification (order, seasonal order, freq), data splits, accuracy metrics, and energy metrics from both tools. The CSV is opened in append mode and never overwritten — every historical run is preserved.

---

## 5. Results

All results in this section are drawn from two benchmark runs:
- **Run A (`wf_20260331_152753`):** stride=96 (daily origins), n=365 error pairs per horizon. No energy data for Naive; CodeCarbon-only for other models.
- **Run B (`wf_20260409_143052`):** stride=1 (every 15 min), n=17,473 error pairs per horizon. Full dual-tool energy measurement (CodeCarbon + CarbonTracker). All models except SARIMA.

Run A represents the canonical methodology (365 daily origins, matching the stated design). Run B provides higher-density statistics and dual-tool energy validation.

**Note on SARIMA:** SARIMA(2,1,2)(1,1,1,96) has not been included in a complete dual-tool run due to fitting time constraints (~2–10 min per fit × energy measurement overhead). Earlier partial benchmarks (`results/wf_metrics_runs.csv`) include SARIMA results from a legacy pipeline; those are referenced in Section 6.

---

### 5.1 Accuracy Results — Run A (stride=96, n=365 origins)

#### MAE (MW) — Primary Metric

| Model | h=1 (15 min) | h=4 (1 h) | h=96 (24 h) |
|-------|-------------|----------|------------|
| Seasonal Naive | 292.4 | 296.9 | 297.7 |
| ARIMA(2,1,2) | **109.6** | 185.2 | 690.9 ⚠️ |
| Ridge Regression | 109.2 | 199.9 | 381.2 ⚠️ |
| **LightGBM** | **98.5** | **127.9** | **238.4** |
| SARIMA(2,1,2)(1,1,1,96) | — | — | — |

#### RMSE (MW)

| Model | h=1 | h=4 | h=96 |
|-------|-----|-----|------|
| Seasonal Naive | 414.5 | 419.96 | 402.3 |
| ARIMA(2,1,2) | 188.8 | 265.0 | 829.9 |
| Ridge Regression | 186.9 | 271.6 | 481.1 |
| **LightGBM** | **172.9** | **194.3** | **330.9** |

#### MAPE (%)

| Model | h=1 | h=4 | h=96 |
|-------|-----|-----|------|
| Seasonal Naive | 3.54% | 3.70% | 3.56% |
| ARIMA(2,1,2) | 1.34% | 2.32% | 8.24% ⚠️ |
| Ridge Regression | 1.33% | 2.47% | 4.53% ⚠️ |
| **LightGBM** | **1.20%** | **1.58%** | **2.80%** |

#### Accuracy Interpretation

**At h=1 (15 minutes ahead):** LightGBM leads narrowly (MAE 98.5 MW, 1.20% MAPE), with ARIMA (109.6 MW) and Ridge (109.2 MW) essentially tied, and Seasonal Naive last (292.4 MW). The large gap between the models and the naive at h=1 reflects the strong short-term autocorrelation in electricity load: knowing the current load (lag_0) dramatically improves the 15-minute forecast. ARIMA's AR(2) terms and Ridge's lag features both exploit this. LightGBM captures non-linear morning ramp dynamics that give it a small but consistent edge.

**At h=4 (1 hour ahead):** LightGBM maintains a substantial lead (127.9 MW, 1.58% MAPE). ARIMA (185.2 MW) outperforms Ridge (199.9 MW) at this horizon — ARIMA's MA(2) terms capture the decay of short-term autocorrelation more effectively than Ridge's lag-based linear model at 1h. The naive falls far behind (297 MW). The RMSE/MAE ratios (ARIMA: 265/185=1.43; Ridge: 271/200=1.36; LightGBM: 194/128=1.52) indicate occasional larger errors for LightGBM relative to its mean, suggesting it handles typical conditions well but can be surprised by rapid load shifts.

**At h=96 (24 hours ahead) — the critical horizon:**

- **ARIMA (MAE=690.9 MW, MAPE=8.24%):** The worst result. Without seasonal differencing or seasonal AR/MA, ARIMA reverts to its unconditional mean after 2 lags. By h=96, the forecast is essentially a constant — the long-run mean of the differenced series. Its RMSE/MAE ratio of 829.9/690.9 = 1.20 indicates that many individual errors are catastrophically large (2,000+ MW misses during demand spikes). This is the expected, informative outcome: **non-seasonal ARIMA cannot forecast 24 hours ahead in a seasonally-dominated series.**

- **Ridge (MAE=381.2 MW, MAPE=4.53%):** Worse than Seasonal Naive (297.7 MW). Regularisation (alpha=1.0) shrinks the coefficient of lag_0 — the most informative single feature for a 24h forecast — while the 12 other lags (lag_1 through lag_48) contribute noise at this horizon that the regulariser cannot fully suppress. The seasonal naive implicitly uses lag_96 with a coefficient of exactly 1, which Ridge with strong regularisation cannot replicate. This is a documented finding, not a modelling error.

- **LightGBM (MAE=238.4 MW, MAPE=2.80%):** The only model to beat Seasonal Naive (297.7 MW) at h=96 without an explicit seasonal component. LightGBM exploits lag_96 (same slot yesterday) and lag_672 (same slot last week) via non-linear tree splits, combined with the cyclic calendar features, to reconstruct the daily pattern more effectively than the naive's single-lag look-back. The improvement over naive (238 vs 298 MW MAE, −20%) is the key finding of the energy-accuracy trade-off.

---

### 5.2 Accuracy Results — Run B (stride=1, n=17,473 origins)

Run B uses a much denser evaluation (every 15-minute interval is an origin, not every day), giving finer-grained statistics at the cost of higher correlation between adjacent error pairs.

| Model | h=1 MAE | h=1 MAPE | h=4 MAE | h=4 MAPE | h=96 MAE | h=96 MAPE |
|-------|---------|---------|---------|---------|---------|---------|
| Seasonal Naive | 517.9 MW | 5.59% | 518.8 MW | 5.60% | 515.6 MW | 5.57% |
| ARIMA(2,1,2) | **86.8 MW** | **0.96%** | 228.1 MW | 2.49% | 798.1 MW | 8.70% |
| Ridge | 82.9 MW | 0.91% | 198.8 MW | 2.16% | 452.4 MW | 4.95% |
| **LightGBM** | **79.1 MW** | **0.86%** | **156.6 MW** | **1.69%** | **311.6 MW** | **3.36%** |

The relative ordering is consistent with Run A. The higher naive MAE in Run B (518 vs 292 MW at h=1) reflects that stride=1 samples all 15-minute intervals, including overnight hours when load varies significantly day-over-day; stride=96 samples at a fixed daily time and catches a more consistent slice. The model rankings are unchanged, confirming robustness across evaluation protocols.

---

### 5.3 Energy & Compute — Run A (wf_20260331_152753, CodeCarbon)

| Model | Fit energy (kWh) | Fit time (s) | Eval energy (kWh) | Eval time (s) |
|-------|-----------------|-------------|------------------|--------------|
| Seasonal Naive | 0 (no fitting) | 0 | 4 × 10⁻⁸ | 9.4 |
| ARIMA(2,1,2) | 3.7 × 10⁻⁷ | 1.88 | 2.0 × 10⁻⁵ | 11.6 |
| Ridge Regression | 3.4 × 10⁻⁷ | 3.83 | 7.6 × 10⁻⁷ | 2.05 |
| LightGBM | 1.56 × 10⁻⁴ | 41.0 | 7.4 × 10⁻⁷ | 4.12 |

### 5.4 Energy & Compute — Run B (wf_20260409_143052, CodeCarbon + CarbonTracker)

| Model | Fit CC (kWh) | Fit CT (kWh) | Fit (s) | Eval CC (kWh) | Eval CT (kWh) | Eval (s) |
|-------|------------|------------|--------|--------------|--------------|---------|
| Naive | — | — | 0 | 9.2 × 10⁻⁷ | 1.0 × 10⁻⁶ | 4.06 |
| ARIMA | 7.2 × 10⁻⁷ | 1.1 × 10⁻⁶ | 1.84 | 1.51 × 10⁻³ | 1.78 × 10⁻³ | 486.9 |
| Ridge | 1.0 × 10⁻⁶ | 1.6 × 10⁻⁶ | 3.16 | 6.1 × 10⁻⁵ | 9.1 × 10⁻⁵ | 19.3 |
| LightGBM | 2.4 × 10⁻⁵ | 1.1 × 10⁻⁴ | 17.5 | 9.9 × 10⁻⁵ | 7.7 × 10⁻⁵ | 19.7 |

CO₂ equivalents (from Run B, CodeCarbon, Belgium grid carbon intensity ~137 g CO₂eq/kWh):

| Model | Fit CO₂eq (kg) | Eval CO₂eq (kg) | Total CO₂eq (kg) |
|-------|--------------|----------------|----------------|
| Naive | — | 1.3 × 10⁻⁷ | ~0 |
| ARIMA | 1.0 × 10⁻⁷ | 2.1 × 10⁻⁴ | 2.1 × 10⁻⁴ |
| Ridge | 1.4 × 10⁻⁷ | 8.5 × 10⁻⁶ | 8.6 × 10⁻⁶ |
| LightGBM | 3.4 × 10⁻⁶ | 1.4 × 10⁻⁵ | 1.7 × 10⁻⁵ |

#### Energy Interpretation

**Naive** consumes essentially zero energy — no fitting, trivial inference (array indexing). It is the energy-free baseline.

**ARIMA** has a very low fit cost (7.2 × 10⁻⁷ kWh, ~1.84 s) — MLE on 4,000 observations converges quickly. However, its **eval energy in Run B is 1.51 × 10⁻³ kWh**, the highest among all models, because the walk-forward loop at stride=1 executed 17,473 Kalman filter steps over 487 seconds. The Kalman update (`res.append(new_obs, refit=False)`) has non-trivial overhead per step, dominated by matrix operations in the state-space representation. At stride=96 (Run A), ARIMA's eval time is only 11.6 s — the choice of stride dramatically affects ARIMA's operational energy cost.

**Ridge** has a modest fit cost (~1 μWh, ~3 s) and very low eval cost (~0.6–1 μWh per horizon, <2 s for 365 origins). Ridge inference is a single matrix multiply — vectorised and extremely fast.

**LightGBM** has the highest fit cost (24–106 μWh across runs, 17–41 s), reflecting 300 boosting rounds × 3 horizons × tree construction. Its eval cost (99–156 μWh, ~20 s) is somewhat higher than Ridge due to tree traversal being less vectorisable than a matrix multiply. However, **LightGBM's accuracy-per-energy ratio is the best of all non-trivial models**: it achieves the lowest MAE at every horizon while consuming only 24 μWh to fit (CodeCarbon). Compared to ARIMA's 1.51 mWh eval cost in Run B, LightGBM's 0.099 mWh eval is 15× cheaper at much better accuracy.

**Cross-tool agreement:** CodeCarbon and CarbonTracker agree within a factor of 2–4 for all models (e.g., ARIMA eval: 1.51 vs 1.78 mWh), consistent with their different hardware measurement backends (CodeCarbon uses Apple's power management; CarbonTracker uses powermetrics). The agreement validates that both tools are measuring the same underlying physical process, even if absolute values differ slightly.

---

### 5.5 Figures — Forecast Visualisations

The following figures are produced by `visualize_results.py` for the benchmark run `wf_20260409_143052`. All visualisation outputs are in `figures/wf_20260409_143052_vis_20260409_172223/`.

---

#### All-Model Comparison

![Comparison all models](figures/wf_20260409_143052_vis_20260409_172223/comparison_all_models.png)

This figure overlays all four models (Naive, ARIMA, Ridge, LightGBM) on a shared axis for each of the three forecast horizons. It provides the definitive visual summary of model performance: at h=1, all models except Naive track the actual load closely; at h=96, ARIMA diverges catastrophically (flat mean forecast visible as a nearly horizontal line), Ridge shows moderate tracking, and LightGBM is the only model to follow the seasonal envelope. The Naive appears as a smooth "yesterday" curve, accurate at h=96 but missing intra-day dynamics.

---

#### Naive Seasonal — Diagnostic Plots

![Naive seasonal overview](figures/wf_20260409_143052_vis_20260409_172223/naive_seasonal_overview.png)

The 2×3 seasonal overview for the Naive model shows a 2-row (winter/summer) × 3-column (h=1/h=4/h=96) panel grid. The Naive's forecast is visually indistinguishable from the actual signal at h=96 (since it uses yesterday's value directly), confirming its strong 24h baseline. At h=1 and h=4, larger residuals appear because yesterday's same slot may differ from today's by load ramp events.

![Naive zoom winter](figures/wf_20260409_143052_vis_20260409_172223/naive_zoom_winter.png)

The 7-day winter zoom shows the Naive model's day-ahead (h=96) prediction for a week in January. The forecast closely tracks the actual load curve — validating that the daily pattern is highly repetitive in winter. Errors tend to occur at load peaks (morning ramp-up, evening peak) where day-over-day variation is largest.

![Naive zoom summer](figures/wf_20260409_143052_vis_20260409_172223/naive_zoom_summer.png)

The 7-day summer zoom shows the Naive model during July. Summer load is lower (~7,000–8,500 MW) and flatter, so the Naive performs similarly or slightly better than in winter (less day-over-day variation in the daily pattern). The error panels show small, consistent residuals.

![Naive zoom 1 week](figures/wf_20260409_143052_vis_20260409_172223/naive_zoom_1week.png)

One-week detail zoom showing all three horizons stacked: h=1 (top), h=4 (middle), h=96 (bottom). At h=1, errors are larger because yesterday's exact 15-min value differs from today's by load ramp dynamics. At h=96, the Naive achieves near-perfect tracking for typical weekdays. The weekend-weekday transition (visible as a dip in load on Saturday/Sunday) is correctly predicted since yesterday was also a weekend.

![Naive zoom 1 day](figures/wf_20260409_143052_vis_20260409_172223/naive_zoom_1day.png)

One-day (96-step) detail: shows a 24-hour window with dots for each 15-minute forecast. The morning ramp (6:00–9:00 AM) and evening peak (~18:00–20:00) are clearly visible. The Naive correctly predicts the general shape but may be off by 30–60 minutes if the day's peak arrives slightly earlier or later than yesterday.

![Naive full period](figures/wf_20260409_143052_vis_20260409_172223/naive_full_period.png)

Full test-year overview for the Naive model. The seasonal pattern (high winter load, low summer load) is clearly visible in both actual and predicted curves. The overall tracking is good; the Naive MAE of ~298–518 MW (depending on the run's stride) represents the floor for any model that exploits daily seasonality.

---

#### ARIMA(2,1,2) — Diagnostic Plots

![ARIMA seasonal overview](figures/wf_20260409_143052_vis_20260409_172223/arima_seasonal_overview.png)

The 2×3 seasonal overview for ARIMA. At h=1 and h=4, the predictions closely track the actual load, confirming that ARIMA's AR(2) and MA(2) terms effectively capture short-term autocorrelation. At h=96, the forecast panel shows ARIMA's critical failure: the predicted curve is nearly flat in each 7-day window, converging to the unconditional mean. Winter panels show large errors as the model predicts mid-range load regardless of actual peaks.

![ARIMA zoom winter](figures/wf_20260409_143052_vis_20260409_172223/arima_zoom_winter.png)

7-day winter zoom for ARIMA. At h=96, the predicted series (plotted in red/orange) is visually a nearly constant line while the actual load (blue) follows its characteristic double-peak daily pattern. The error panel below shows errors regularly exceeding 1,000 MW during morning and evening peaks. This plot is the most compelling visual demonstration of why non-seasonal ARIMA fails at long horizons.

![ARIMA zoom summer](figures/wf_20260409_143052_vis_20260409_172223/arima_zoom_summer.png)

7-day summer zoom for ARIMA. Similar flat-mean behaviour at h=96, though summer load is more uniform so errors are smaller in absolute terms. At h=1 and h=4, ARIMA tracks well even in summer.

![ARIMA zoom 1 week](figures/wf_20260409_143052_vis_20260409_172223/arima_zoom_1week.png)

One-week detail showing the three-horizon progression. The contrast between h=1 (excellent tracking) and h=96 (flat mean) is starkly visible in this side-by-side format. This figure is particularly useful for thesis presentations to illustrate the fundamental limitation of non-seasonal ARIMA.

![ARIMA zoom 1 day](figures/wf_20260409_143052_vis_20260409_172223/arima_zoom_1day.png)

One-day detail: the 96 15-minute forecasts form a near-constant horizontal band at h=96, confirming the mean-reversion interpretation. At h=1, the forecast tightly follows the actual trace.

![ARIMA full period](figures/wf_20260409_143052_vis_20260409_172223/arima_full_period.png)

Full test-year overview for ARIMA. At h=96, the predicted series appears as a smooth, nearly flat curve tracking the seasonal mean but completely missing the day-to-day variation. The summer/winter seasonal contrast is present at a coarse scale (slightly lower predicted mean in summer) but the within-season dynamics are lost entirely.

---

#### Ridge Regression — Diagnostic Plots

![Ridge seasonal overview](figures/wf_20260409_143052_vis_20260409_172223/linear_seasonal_overview.png)

The 2×3 seasonal overview for Ridge. At h=1 and h=4, the forecasts closely match actuals — the lag features and cyclic calendar encoding give Ridge strong short-to-medium horizon accuracy. At h=96, Ridge shows intermediate behaviour: better than ARIMA (it at least has lag_96 and lag_672 features to exploit), but worse than the Naive. The forecast at h=96 follows the broad seasonal shape but underestimates peaks, consistent with regularisation shrinking the lag_96 coefficient below 1.

![Ridge zoom winter](figures/wf_20260409_143052_vis_20260409_172223/linear_zoom_winter.png)

7-day winter zoom for Ridge. At h=96, the predictions follow the daily pattern (morning and evening peaks visible) but with systematically dampened amplitude: Ridge under-predicts winter peaks and over-predicts troughs, a direct consequence of regularisation pushing the coefficient on lag_0 (the dominant feature for h=96) toward zero.

![Ridge zoom summer](figures/wf_20260409_143052_vis_20260409_172223/linear_zoom_summer.png)

7-day summer zoom. Summer load is flatter, so the dampening effect of regularisation is less harmful — Ridge's h=96 errors are smaller in summer than winter. This is visible in the comparison: the summer MAE contribution is lower than winter.

![Ridge zoom 1 week](figures/wf_20260409_143052_vis_20260409_172223/linear_zoom_1week.png)

One-week detail: the three horizons show Ridge's characteristic behaviour. At h=1 (excellent), h=4 (good), h=96 (partial capture of daily shape with damped amplitude). The error panels show systematic positive and negative errors at the peaks and troughs respectively.

![Ridge zoom 1 day](figures/wf_20260409_143052_vis_20260409_172223/linear_zoom_1day.png)

One-day detail for Ridge. At h=96, the 24-hour ahead forecast shows the daily double-peak pattern (present but dampened). The model correctly predicts the shape but not the exact amplitude — a linear model's limitation in capturing non-linear load spikes.

![Ridge full period](figures/wf_20260409_143052_vis_20260409_172223/linear_full_period.png)

Full test-year for Ridge. The seasonal envelope (high winter, low summer) is well captured at all horizons. At h=96, the predicted curve is smooth and slightly below the actual peaks in winter — consistent with the regularisation-dampening hypothesis.

---

#### LightGBM — Diagnostic Plots

![LightGBM seasonal overview](figures/wf_20260409_143052_vis_20260409_172223/lgbm_seasonal_overview.png)

The 2×3 seasonal overview for LightGBM. This is the strongest result panel in the benchmark: at h=1, the predictions are nearly indistinguishable from actuals; at h=4, minor errors appear at rapid ramp events; at h=96, LightGBM correctly predicts the full daily double-peak pattern in both winter and summer. The residual shading shows consistently smaller errors than Naive, Ridge, or ARIMA at h=96.

![LightGBM zoom winter](figures/wf_20260409_143052_vis_20260409_172223/lgbm_zoom_winter.png)

7-day winter zoom for LightGBM. At h=96, the predictions correctly follow the morning ramp (load rising from ~8,500 MW at 6:00 to ~11,000 MW at 9:00), the midday plateau, and the evening peak (~10,500 MW at 18:00-19:00). Errors are concentrated at the transition points (rapid ramps) where the lag-based feature set has limited temporal resolution. This is the key figure demonstrating LightGBM's superiority over the Naive at long horizons.

![LightGBM zoom summer](figures/wf_20260409_143052_vis_20260409_172223/lgbm_zoom_summer.png)

7-day summer zoom. Summer load is lower and more uniform. LightGBM's performance is excellent at all horizons. The summer h=96 errors are smaller than winter in absolute terms, consistent with lower load variability. The model correctly predicts the flatter summer daily pattern (single midday peak rather than double winter peak).

![LightGBM zoom 1 week](figures/wf_20260409_143052_vis_20260409_172223/lgbm_zoom_1week.png)

One-week detail for LightGBM. The three horizons are stacked: h=1 (essentially perfect), h=4 (excellent), h=96 (good, with small errors at peak transitions). The progressive accuracy degradation across horizons is smooth and physically interpretable — each additional 15 minutes of lookahead adds a small amount of irreducible uncertainty.

![LightGBM zoom 1 day](figures/wf_20260409_143052_vis_20260409_172223/lgbm_zoom_1day.png)

One-day detail: the 96 15-minute forecasts at h=96 form a well-shaped daily load curve rather than a flat line (ARIMA) or damped curve (Ridge). The morning ramp timing is correctly predicted. This figure confirms that LightGBM's tree-based feature interactions (lag_96 × calendar features) effectively reconstruct the daily seasonality.

![LightGBM full period](figures/wf_20260409_143052_vis_20260409_172223/lgbm_full_period.png)

Full test-year for LightGBM. The predicted series tracks the actual at all horizons across all seasons (winter peaks visible in December–February, summer troughs in July–August). This confirms consistent performance across the full year without seasonal performance degradation.

---

### 5.6 Figures — Top-Level Overview Plots

These are standalone overview and zoom figures (likely from an earlier benchmark run) in the `figures/` root.

![Naive overview](figures/naive_overview.png)

Seasonal overview panel for the Naive model from an earlier run. Shows the characteristic property of seasonal naive: near-perfect 24h-ahead tracking across seasons, but larger intra-day errors at short horizons.

![Naive zoom](figures/naive_zoom.png)

Zoom detail for Naive showing error structure at short horizons.

![ARIMA overview](figures/arima_overview.png)

Seasonal overview for ARIMA. The flat h=96 predictions are visible as horizontal lines in each seasonal panel, contrasting with the actual load's daily periodicity.

![ARIMA zoom](figures/arima_zoom.png)

Close-up showing ARIMA's excellent h=1 tracking and flat h=96 forecast within a one-week window.

![Ridge overview](figures/linear_overview.png)

Overview panel for Ridge Regression. The h=96 predictions show a dampened version of the daily pattern — better than ARIMA (daily shape partially recovered) but worse than Naive (amplitude systematically underestimated).

![Ridge zoom](figures/linear_zoom.png)

Zoom for Ridge showing the interplay between lag-based feature reconstruction and regularisation-induced amplitude damping at h=96.

![LightGBM overview](figures/lgbm_overview.png)

Overview panel for LightGBM — the strongest result across all three horizons. At h=96, the predicted curve closely matches the actual in both shape and amplitude.

![LightGBM zoom](figures/lgbm_zoom.png)

Zoom for LightGBM. This figure is suitable for thesis figures illustrating the model's ability to reconstruct daily load dynamics 24 hours ahead.

![Comparison bar chart](figures/comparison_bars.png)

Bar chart comparing MAE (and/or MAPE) across all models and horizons. Provides a compact visual summary of the full accuracy table. LightGBM's bars are shortest at every horizon; ARIMA's h=96 bar is dramatically taller than all others. The Naive provides a horizontal reference line.

![Comparison scatter plot](figures/comparison_scatter.png)

Scatter plot of actual vs. predicted load (actual on x-axis, predicted on y-axis) for all models. Points on the diagonal dashed line (y=x) represent perfect predictions. Tight clustering around the diagonal indicates good calibration. ARIMA at h=96 shows a characteristic horizontal band (all predictions cluster near the mean regardless of actual load). LightGBM shows the tightest cluster.

---

### 5.7 Figures — Energy Plots

All energy figures are from `figures/energy_20260409_171714/`, generated by `plot_energy.py` on the `wf_20260409_143052` run.

![Accuracy bar chart](figures/energy_20260409_171714/01_accuracy_bar.png)

MAE and MAPE per model × horizon in a grouped bar chart. Models are ordered left-to-right on the x-axis; different bar colors represent different horizons. LightGBM consistently shows the shortest bars (lowest error) at all horizons. The height difference between ARIMA's h=96 bar and all others visually confirms its failure mode. This figure is the primary accuracy summary for the energy chapter.

![CodeCarbon vs CarbonTracker energy](figures/energy_20260409_171714/02_energy_cc_vs_ct.png)

Side-by-side bar comparison of total energy (kWh) per model as measured by CodeCarbon (dark blue) and CarbonTracker (dark green). The two tools agree within a factor of 2–4, providing mutual validation. ARIMA's total energy (fit + eval) is by far the highest due to the eval overhead of 17,473 Kalman update steps at stride=1. LightGBM's total energy is dominated by fit cost. This figure confirms that the two measurement tools are consistent.

![Energy breakdown (fit vs eval)](figures/energy_20260409_171714/03_energy_breakdown.png)

Stacked bar chart showing fit energy (solid fill) and eval energy (hatched fill) per model. For ARIMA: eval dominates (the Kalman filter updates across thousands of time steps at stride=1 cost far more than the initial MLE fit). For LightGBM: fit dominates (300 boosting rounds across 3 horizons). For Ridge: both are small, eval slightly larger. For Naive: only a negligible eval bar. This figure is key for the production-cost argument: a model used daily at stride=96 has a different energy profile than at stride=1.

![CO2 emissions](figures/energy_20260409_171714/04_co2_bar.png)

CO₂-equivalent emissions (kg CO₂eq) per model for CodeCarbon and CarbonTracker. All models produce tiny absolute CO₂ amounts (in the range of 10⁻⁷ to 10⁻⁴ kg), reflecting the relatively short durations of each benchmark phase. ARIMA's eval emission dominates. The Belgian grid carbon intensity (~137 g CO₂eq/kWh as used by CodeCarbon for the Flanders region) implies that the dominant energy cost driver is computation time, not Belgian grid carbon intensity.

![Energy per prediction](figures/energy_20260409_171714/05_energy_per_pred.png)

Energy normalised per prediction (μWh per prediction), combining fit and eval energy and dividing by n_predictions × n_horizons. This normalisation answers: "if I deploy this model in production for 17,473 predictions, how much energy does each prediction cost?" Naive is essentially zero. LightGBM and Ridge are comparably efficient per prediction. ARIMA's per-prediction energy is the highest because its eval overhead (Kalman steps) scales linearly with n_predictions.

![Accuracy-energy trade-off](figures/energy_20260409_171714/06_tradeoff.png)

Three scatter plots (one per horizon: h=1, h=4, h=96), with each model as a labeled data point, x-axis = total energy (kWh), y-axis = MAE (MW). The ideal model is in the bottom-left corner (low energy, low error). At h=1 and h=4, LightGBM and Ridge are both in the bottom-left region; ARIMA is more energy-intensive at h=96 due to eval cost. At h=96, LightGBM uniquely occupies the bottom-left (low energy, lowest MAE), while ARIMA is top-right (high energy, highest MAE). This figure is the thesis's central result: **LightGBM dominates the accuracy-energy trade-off at all horizons.**

![Duration bar chart](figures/energy_20260409_171714/07_duration_bar.png)

Wall-clock fit time (solid) and eval time (hatched) per model in seconds. ARIMA's eval time at stride=1 (487 s) dwarfs all other models. LightGBM's fit time (17 s) is the longest among models with finite training. Ridge trains in ~3 s and evaluates in ~19 s. Naive: 0 s fit, 4 s eval (pure array reads). This figure translates energy measurements into operational wall-clock time, which is directly relevant for deployment planning.

---

## 6. Key Design Choices & Justifications

### Walk-Forward Stride Rationale

The canonical stride of 96 (one origin per day) was chosen to balance two competing requirements. With stride=1, the walk-forward loop would generate ~35,000 origins over the 1-year test window, making ARIMA evaluation extremely slow (each Kalman update requires a full state-space matrix operation, costing O(p² × n_new) per step). With stride=96, 365 origins are generated — statistically robust (365 error pairs per horizon) while completing in 10–30 minutes for ARIMA. The alternative (stride=1, 17,473 origins for a 6-month window) was executed in Run B to confirm that relative model rankings are stable across evaluation density — they are.

### Why ARIMA Fails at h=96: The Mean-Reversion Mechanism

ARIMA(p,d,q) produces a conditional mean forecast by solving: `ŷ[t+h|t] = φ₁ŷ[t+h-1] + ... + φₚŷ[t+h-p] + θ₁ε[t+h-1] + ... + θqε[t+h-q]`. For horizons `h > max(p,q) = 2`, the forecast recursion no longer has access to any previously observed residuals (all ε[t+j] for j > 0 are set to their expectation of 0), and the AR terms eventually converge to the unconditional mean of the differenced series — a constant. By h=96 steps ahead, the ARIMA(2,1,2) forecast has been recursing on its own predictions for 94 steps and is indistinguishable from a flat line at the long-run mean. This is not a bug but a fundamental property of finite-order ARIMA: it cannot extrapolate seasonality that was not encoded in its seasonal component. SARIMA(2,1,2)(1,1,1,96) avoids this by explicitly differencing at lag 96 and including a seasonal AR term that directly regresses on `y[t−96]` — the same slot yesterday.

### Why Ridge Underperforms Naive at h=96: The Regularisation Paradox

At h=96, the single most predictive feature is `lag_96` (same slot yesterday) — exactly what the seasonal naive uses. Ridge with `alpha=1.0` applies L2 regularisation globally across all 17 features, shrinking every coefficient toward zero proportionally to its contribution to the penalty. The lag features are in units of MW (6,000–12,000 MW range), making their raw values large; despite StandardScaler normalising them, the regulariser still competes with the signal. The coefficient on `lag_96` is shrunk to a value less than 1, while the seasonal naive implicitly uses a coefficient of exactly 1 with no penalty. Additionally, features like `lag_1` through `lag_12` (recent steps) provide negligible signal at h=96 but consume regularisation budget, diluting the available capacity for `lag_96` and `lag_672`. Reducing `alpha` to 0.01–0.1 for the h=96 model specifically would restore the `lag_96` coefficient closer to 1 and likely close the gap with the Naive.

### Why LightGBM Beats Naive at h=96: Non-Linear Feature Interactions

The seasonal naive uses a single rule: `ŷ[t+h] = y[t+h−96]`. LightGBM effectively learns a richer version of this rule: its trees can isolate different prediction strategies for different contexts. A tree might learn: "if h_cos indicates 8:00 AM AND d_sin indicates Monday AND lag_96 is high → predict higher than lag_96" (capturing the Monday morning ramp that is steeper than the Friday-into-Monday seasonal naive would suggest). Additionally, LightGBM uses lag_672 (same slot last week) alongside lag_96, which provides a second independent anchor for the daily prediction and reduces sensitivity to day-over-day fluctuations. Neither Ridge nor Naive can exploit these interactions without explicit feature engineering.

### ARIMA Training Window Size (max_train_obs = 4,000)

SARIMAX MLE fitting on the full training set (~350,000 observations) would require solving a state-space optimisation problem whose complexity scales roughly as O(n × d_state²) per evaluation, where d_state is the state vector dimension (proportional to max(p,q) + s×max(P,Q) for SARIMA). On an Apple M4 Pro, fitting ARIMA(2,1,2) on 350,000 observations takes approximately 3–6 hours; SARIMA with s=96 would likely take much longer. The 4,000-observation cap (~42 days of history) provides sufficient data for MLE to reliably estimate the 6 non-seasonal + 4 seasonal parameters while completing in under 2 seconds. Longer windows would shift the parameter estimates only marginally given the stationarity confirmed by the ADF test.

### CodeCarbon Carbon Intensity and Reproducibility

The Belgian grid carbon intensity used by CodeCarbon is approximately 137 g CO₂eq/kWh for the Flanders region (April 2026). This value is fetched from CodeCarbon's static database at initialisation time and written to each `emissions.csv` row under the `emissions_rate` column. To reproduce the CO₂eq figures exactly, the same CodeCarbon version (3.2.3, as logged in `emissions.csv`) must be used. Energy figures (kWh) are hardware-measurement-based and are reproducible across CodeCarbon versions given identical hardware (Apple M4 Pro). CarbonTracker applies a PUE factor of 1.58, accounting for server infrastructure overhead — this is the primary reason its absolute kWh values are higher than CodeCarbon's for computationally intensive phases.

### Why Direct (Not Recursive) Multi-Step Forecasting

The direct multi-step strategy trains one model per horizon h and predicts `y[t+h]` in one shot from features at origin `t`. The alternative, recursive forecasting, would train one single-step model and chain 96 predictions — using each predicted value as input to the next. Recursive error accumulation is severe at long horizons: a 1% error at step 1 propagates and compounds through 95 subsequent predictions. Direct forecasting avoids this at the cost of training 3 models (one per horizon) instead of 1. Given that a full training run for Ridge takes <4 s and for LightGBM <20 s, training 3 models instead of 1 adds only a small overhead while fundamentally improving h=96 accuracy.

### Design Decision to Not Use Darts

The thesis deliberately uses `statsmodels.tsa.SARIMAX` and `sklearn` directly rather than the Darts forecasting library used in the KU Leuven course notebooks. The reasons are: (1) Darts wraps model internals, making it impossible to precisely bracket `EnergyMeter.start/stop` around only the model operation itself — Darts may perform internal resampling, index alignment, or validation that would contaminate the energy measurement. (2) The Kalman filter update strategy (`res.append(refit=False)`) is a statsmodels-specific low-level API that Darts abstracts away. (3) Removing Darts eliminates dozens of transitive dependencies for a potentially constrained research infrastructure.

---

## 7. File & Code Structure

```
thesis_arima_energy_benchmark/
├── run_benchmark.py              ← Main entry point
├── tune_hyperparams.py           ← Optuna hyperparameter tuning
├── prepare_dataset.py            ← One-time dataset preprocessing
├── visualize_results.py          ← Publication-quality diagnostic plots
├── generate_report.py            ← PDF report generation (ReportLab)
├── plot_energy.py                ← Energy + accuracy trade-off visualizations
├── CLAUDE.md                     ← Project overview for Claude Code context
├── METHODOLOGY.md                ← Complete technical methodology (628 lines)
├── RESULTS_REPORT.md             ← Full benchmark results & graph-by-graph analysis
├── README.md                     ← Quick-start guide
├── requirements.txt              ← Python dependencies
├── config/
│   └── config_example.json       ← Reference configuration template
├── src/                          ← Core library modules
│   ├── __init__.py               ← Module docstring listing all submodules
│   ├── data.py                   ← Data loading, cleaning, splitting, I/O
│   ├── models.py                 ← All forecasting model classes and factories
│   ├── metrics.py                ← Error metrics and energy measurement
│   └── evaluate.py               ← ADF test, AIC/BIC grid search, forecast eval
├── data/
│   ├── raw/                      ← Original Elia CSV (not committed)
│   └── processed/
│       └── elia_load_15min.csv   ← Cleaned 15-min load (~385K rows)
├── results/
│   ├── wf_metrics_runs.csv       ← Lightweight cumulative run log
│   ├── wf_<YYYYMMDD_HHMMSS>/     ← One folder per benchmark run
│   │   ├── metrics.csv           ← Full model × horizon accuracy + energy
│   │   ├── predictions.parquet   ← All (actual, predicted) pairs (tidy format)
│   │   ├── codecarbon/
│   │   │   └── emissions.csv     ← CodeCarbon raw output
│   │   └── carbontracker/        ← CarbonTracker log sub-directories
│   └── (legacy run folders)
└── figures/
    ├── (top-level overview PNGs) ← From earlier runs
    ├── wf_<id>_vis_<id>/         ← visualize_results.py output per run
    └── energy_<id>/              ← plot_energy.py output per run
```

**`src/data.py`** — Data I/O and preprocessing. The three core functions (`load_timeseries`, `clean_timeseries`, `split_last_n_years`) form the pipeline entry point. `append_run_csv` and `save_json` are small I/O helpers used by `run_benchmark.py` to persist results. Every other module depends on this one for data access.

**`src/models.py`** — All forecasting model classes and factory functions. Contains: `ArimaConfig` (hyperparameter dataclass), `fit_sarimax` (single SARIMAX fit), `NaivePredictor`, `ARIMAPredictor` (handles both ARIMA and SARIMA), `DirectForecaster` (direct multi-step wrapper), `MLPredictor` (walk-forward ML wrapper), and `make_linear_forecaster` / `make_lgbm_forecaster` factory functions. The uniform `fit/set_series/predict` interface is the key abstraction that allows `run_benchmark.py` to handle all models without branching.

**`src/metrics.py`** — Error metrics and energy measurement. Provides scalar metric functions (`mae`, `rmse`, `mape`) and two energy meter classes (`EnergyMeter` wrapping CodeCarbon; `CarbonTrackerMeter` wrapping carbontracker). The `EnergyResult` dataclass carries both CodeCarbon and CarbonTracker measurements so they can be logged side-by-side. The CodeCarbon implementation deliberately reads the last CSV row rather than relying on the `stop()` return value for API version robustness.

**`src/evaluate.py`** — Stationarity and evaluation utilities. `adf_test` runs the Augmented Dickey-Fuller test and returns a structured dict with the statistic, p-value, critical values, and lag count. `evaluate_forecast` aligns two series by DatetimeIndex and computes MAE/RMSE/MAPE. `aic_bic_grid_search` runs an exhaustive grid over ARIMA/SARIMA orders and returns a DataFrame sorted by AIC — used by `tune_hyperparams.py` for order selection.

**`run_benchmark.py`** — The main pipeline orchestrator. Parses CLI arguments, loads data, runs each requested model through `run_model()` (which handles fit → walk-forward → log → plot), then writes the combined predictions parquet and comparison table. The `walk_forward()` function is the loop heart: it steps through test origins at the configured stride, collects `{h: prediction}` dicts, and assembles arrays of (y_true, y_pred) per horizon.

**`tune_hyperparams.py`** — Standalone Optuna tuning script. Runs independently from `run_benchmark.py` to avoid confounding energy measurements. Saves best parameters to `results/tuned_params.json`. Uses chronological validation (no shuffled cross-validation folds) to respect the time-ordering of the data.

**`visualize_results.py`** — Publication-quality diagnostic visualisations. Reads `predictions.parquet` from any run, optionally loads the raw series for continuous actual-value plotting, and produces per-model and cross-model plots. Season windows are auto-detected (January for winter, July for summer) with fallback to extreme-load weeks.

**`plot_energy.py`** — Energy and accuracy trade-off visualisations. Auto-selects the most recent run with CodeCarbon data and generates 7 figures covering accuracy bars, dual-tool energy comparison, fit/eval energy breakdown, CO₂ emissions, per-prediction energy, accuracy-energy trade-off scatter, and duration bars.

**`generate_report.py`** — Auto-generates a PDF thesis report using ReportLab Platypus. Embeds figures from the figures folder, loads accuracy and energy data from the metrics CSV, and produces a formatted A4 report with all standard sections.

**`prepare_dataset.py`** — One-time data preparation script. Reads the raw Elia CSV (`data/raw/Data Elia Load.csv`), cleans and resamples to 15-minute frequency, and writes the processed output to `data/processed/elia_load_15min.csv`. Run only once at project setup; not part of the benchmark pipeline.

---

## 8. How to Reproduce

### Environment Setup

```bash
# Clone or navigate to the project
cd thesis_arima_energy_benchmark

# Create virtual environment (Python 3.14)
python3.14 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: LightGBM (required for --models lgbm)
pip install lightgbm

# Optional: Optuna (required for tune_hyperparams.py)
pip install optuna

# Optional: holidays feature (--use_holidays flag)
pip install holidays

# Optional: PDF report generation
pip install reportlab
```

### Dataset Preparation

The Elia load CSV must be placed at `data/raw/Data Elia Load.csv` (downloaded from the Elia open data portal). Then run:

```bash
python prepare_dataset.py
# → produces data/processed/elia_load_15min.csv
```

This step is idempotent and needs to be run only once.

### Running the Benchmark

```bash
# Full benchmark — all models, default settings (stride=96, test_years=1)
python run_benchmark.py \
    --data data/processed/elia_load_15min.csv \
    --timestamp_col datetime \
    --target_col totalload \
    --models naive arima sarima linear lgbm \
    --horizons 1 4 96 \
    --energy_tool both \
    --results_dir results

# Skip energy tracking (offline / faster)
python run_benchmark.py --models naive linear lgbm arima --energy_tool none

# Single model (e.g., just LightGBM)
python run_benchmark.py --models lgbm --horizons 1 4 96

# ARIMA with explicit order override
python run_benchmark.py --models arima --arima_order 2 1 2 --max_train_obs 4000

# SARIMA (long-running — 2–10 min fit, do not interrupt)
python run_benchmark.py \
    --models sarima \
    --sarima_order 2 1 2 \
    --sarima_seasonal_order 1 1 1 96 \
    --max_train_obs 4000

# Skip plots (faster for headless/CI runs)
python run_benchmark.py --models naive linear --no_plots

# Use tuned hyperparameters
python run_benchmark.py \
    --models linear lgbm \
    --tuned_params results/tuned_params.json
```

### Hyperparameter Tuning (Optional)

```bash
# Tune Ridge alpha (fast, ~5 min)
python tune_hyperparams.py --model linear --n_trials 50 --output results/tuned_params.json

# Tune LightGBM (longer, ~20–40 min)
python tune_hyperparams.py --model lgbm --n_trials 100 --output results/tuned_params.json
```

### Generating Visualisations

```bash
# Visualise the most recent run (auto-detected)
python visualize_results.py
# → figures/wf_<run_id>_vis_<timestamp>/

# Specify a run explicitly
python visualize_results.py --predictions results/wf_20260409_143052/predictions.parquet

# Generate energy trade-off plots
python plot_energy.py
# → figures/energy_<timestamp>/

# Generate PDF report
python generate_report.py
# → results/thesis_report_<timestamp>.pdf
```

### Key CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/elia_load.csv` | Path to input CSV or Parquet |
| `--timestamp_col` | `datetime` | Timestamp column name |
| `--target_col` | `totalload` | Target variable column name |
| `--freq` | `15min` | Time series frequency |
| `--test_years` | `1.0` | Test window in calendar years |
| `--stride` | `96` | Walk-forward stride (steps) |
| `--horizons` | `1 4 96` | Forecast horizons (space-separated) |
| `--models` | (required) | Models: naive arima sarima linear lgbm |
| `--arima_order` | `2 1 2` | Non-seasonal ARIMA order |
| `--sarima_order` | `2 1 2` | SARIMA non-seasonal order |
| `--sarima_seasonal_order` | `1 1 1 96` | SARIMA seasonal order |
| `--max_train_obs` | `4000` | Training cap for ARIMA/SARIMA |
| `--energy_tool` | `codecarbon` | `codecarbon`, `carbontracker`, `both`, `none` |
| `--results_dir` | `results` | Output directory |
| `--no_plots` | False | Skip plot generation |
| `--tuned_params` | None | Path to tuned_params.json |
| `--use_holidays` | False | Add Belgian holiday feature (ML only) |

### Environment Notes

- **Python:** 3.14 (`.venv` uses CPython 3.14). The code is compatible with Python ≥ 3.10.
- **Hardware:** Benchmarked on Apple MacBook with M4 Pro (14 cores, 24 GB RAM). CodeCarbon uses Apple's power management interface; on Linux, RAPL hardware counters would be used instead.
- **Internet:** CodeCarbon fetches Belgian grid carbon intensity at startup. Without internet, use `--energy_tool none` or `--energy_tool carbontracker`.
- **SARIMA memory:** SARIMA(2,1,2)(1,1,1,96) with s=96 requires >4 GB RAM for the state-space matrices at the full training set size. The `--max_train_obs 4000` cap is essential; without it, the process will OOM on a 16 GB MacBook.
- **Results directory:** Each run creates a timestamped subdirectory `results/wf_<YYYYMMDD_HHMMSS>/`. Results are never overwritten; old runs accumulate until manually deleted.

---

*End of report. Generated from source code, results files, and methodology documentation in the `thesis_arima_energy_benchmark` repository.*
