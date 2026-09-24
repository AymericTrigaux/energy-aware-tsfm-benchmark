# Benchmark Methodology — Electricity Load Forecasting

> Complete technical plan covering every design decision in the codebase.
> Use this document to justify methodology choices in the thesis.

---

## 1. Dataset

**Source:** Elia 15-minute electricity load data for Belgium (`data/processed/elia_load_15min.csv`).
**Columns:** `datetime` (timestamp), `totalload` (MW).
**Range:** December 2014 → March 2026 (≈ 11 years, ≈ 385 000 observations).
**Frequency:** one observation every 15 minutes → 96 observations per day, 672 per week.

### Why this dataset?
Belgian load data covers multiple full seasonal cycles (winters, summers, bank holidays),
which is the minimum required to train and evaluate seasonal models honestly. The 11-year
span allows a 1-year test window that is entirely out-of-sample.

---

## 2. Preprocessing (`scripts/utils.py`)

### 2.1 Loading (`load_timeseries`)
1. Read CSV; parse `datetime` column with `pd.to_datetime`.
2. Set datetime index, sort chronologically, remove duplicate timestamps (keep first).
3. Call `df.asfreq("15min")` to enforce the regular grid — this inserts NaN rows for
   any missing 15-minute slots (e.g. DST transitions, missing readings).

**Why enforce frequency explicitly?**
Without `asfreq`, a gap of e.g. 2 hours would silently shrink the array. ARIMA's
Kalman filter and the lag-based ML features both assume equidistant time steps.
A missing row would misalign every subsequent lag by one position.

### 2.2 Cleaning (`clean_timeseries`)
Missing values are filled by **time-linear interpolation** (`method="time"`), then
forward/backward fill for any edge NaNs.

**Why interpolation and not forward-fill?**
Forward-fill copies the last observed value for potentially many steps, producing
artificial flat segments that inflate the apparent autocorrelation and confuse the
ARIMA order selection. Linear interpolation gives a smooth, physically plausible
imputation for gaps up to a few hours (load ramps gradually). Gaps larger than a
few days are extremely rare in the Elia dataset.

---

## 3. Data Split (`split_last_n_years` in `run_walk_forward_benchmark.py`)

```
─────────────────────────────────────────────────
 TRAIN (≈ 10 years)  |  TEST (last 1 year)
─────────────────────────────────────────────────
 Dec 2014 → Mar 2025    Mar 2025 → Mar 2026
```

**Why a calendar-year test window?**
One full calendar year guarantees the test set contains every season (winter demand
peak, summer trough, Easter, Christmas). A fixed ratio split (e.g. 80/20) could
accidentally land in a single season, making the evaluation unrepresentative.

**Why the last year and not a random year?**
Time series must be split chronologically — the future cannot be used to train the past.
Using the most recent year also evaluates the model on conditions as close as possible
to actual deployment.

**ARIMA/SARIMA train cap (`--max_train_obs 4000`)**
Full training on ≈ 350 000 observations would take many hours for SARIMAX MLE fitting.
Capping at 4 000 observations (≈ 42 days) makes fitting tractable on a MacBook while
still providing enough data for the optimizer to estimate parameters reliably.
The ML models (Ridge, LightGBM) use the full training set because sklearn fit time
scales much better.

---

## 4. Stationarity Test (`adf_test` in `scripts/arima.py`)

### The test used: Augmented Dickey-Fuller (ADF)

**H₀:** the series has a unit root (non-stationary).
**H₁:** the series is stationary.

A p-value < 0.05 → reject H₀ → series is stationary.

**Result on the Elia dataset (after first differencing):**
```
ADF statistic : −12.85
p-value       :  5.4 × 10⁻²⁴
Critical value: −2.86 (5%)
→ Strong evidence of stationarity after d=1 differencing
```

### Why ADF and not KPSS?

| Test | H₀ | Verdict when p < 0.05 |
|------|-----|----------------------|
| **ADF** | Series has a unit root (non-stationary) | **Stationary** ✓ |
| KPSS | Series is stationary | **Non-stationary** |
| PP (Phillips-Perron) | Same as ADF | Stationary |

**ADF is the standard choice** in energy forecasting because:
1. It directly tests what ARIMA needs: absence of a unit root.
2. It is implemented in `statsmodels.adfuller` with automatic lag selection (AIC).
3. The null hypothesis matches ARIMA's assumption — we want to *reject* non-stationarity.

**Why not KPSS in addition?**
The two tests are complementary (KPSS tests the opposite null), but for the purpose
of choosing `d` in ARIMA, ADF is sufficient. With |ADF stat| = 12.85 >> 3.43 (1%
critical value), the result is unambiguous.

**Why not the Phillips-Perron test?**
PP corrects for serial correlation non-parametrically; ADF adds lagged differences.
Both are asymptotically equivalent. ADF with `autolag="AIC"` is slightly more
interpretable and is the most commonly cited test in the forecasting literature.

**Important:** The ADF test is run on the training set only. Using the test set to
decide `d` would constitute data leakage.

---

## 5. Models

### 5.1 Baseline — Seasonal Naive (`NaivePredictor`)

**Formula:** `ŷ[t+h] = y[t+h−s]`  where `s = 96` (one daily cycle).

At origin `t`, the h-step-ahead forecast is the load at the same 15-minute slot
exactly one day earlier.

| Horizon | Prediction | What it means |
|---------|-----------|---------------|
| h = 1 | y[t − 95] | Same slot as 23h45 ago |
| h = 4 | y[t − 92] | Same slot as 23h ago |
| h = 96 | y[t] | Current load = prediction for same time tomorrow |

**Why this and not a simpler naive (last-value or mean)?**
Electricity load has a very strong **daily seasonality**. The last observed value
(y[t]) is a poor predictor of y[t+h] for h > a few steps because load varies
substantially within a day. The seasonal naive directly exploits the most
important pattern in the data: same time yesterday ≈ same time today.

**Role in the thesis:** establishes a minimum bar. Any model that cannot beat the
seasonal naive for all horizons is not useful in practice.

---

### 5.2 ARIMA (`ARIMAPredictor`, order (2,1,2))

**Implemented via:** `statsmodels.tsa.SARIMAX` with `seasonal_order=(0,0,0,0)`.

**Order choice (2,1,2):**
- `d=1`: one round of first-differencing removes the (mild) non-stationarity
  confirmed by the ADF test.
- `p=2, q=2`: two AR and two MA lags capture short-term autocorrelation structure.
  This is the standard starting point for energy load; grids search (AIC) typically
  selects small p, q values.

**Walk-forward strategy (`append(refit=False)`):**
The model is fit once on the (capped) training set. At each forecast origin in the
test period, new observations are incorporated via `res.append(new_obs, refit=False)`.
This advances the Kalman filter state without re-optimising MLE parameters. Cost:
O(n_new × p²) per step — negligible vs a full re-fit.

**Why ARIMA at all, given it ignores seasonality?**
ARIMA(2,1,2) serves as the **non-seasonal reference**. Comparing its results to
SARIMA isolates the value of the seasonal component. If SARIMA at h=96 is only
marginally better than ARIMA, the extra complexity and energy cost of SARIMA is
questionable.

**Known limitation:** ARIMA reverts to its unconditional mean after a few lags.
By h=96 (24 hours), the forecast is essentially flat. This is WHY ARIMA at h=96
shows MAE ≈ 691 MW — more than double the seasonal naive. This is an expected,
informative result, not a bug.

---

### 5.3 SARIMA (`ARIMAPredictor`, order (2,1,2)(1,1,1,96))

Same code path as ARIMA; the only difference is `seasonal_order=(1,1,1,96)`.

**Seasonal order rationale:**
- `D=1`: one seasonal difference removes the daily periodicity (y[t] − y[t−96])
  → series becomes stationary across days.
- `P=1, Q=1`: one seasonal AR and one seasonal MA lag. This is the minimal but
  complete seasonal ARIMA specification — analogous to (1,1,1) being the most
  common non-seasonal ARIMA starting point.
- `s=96`: the seasonal period is one full day (96 × 15min = 24h).

**Why (1,1,1,96) and not (1,0,0,96)?**
The earlier benchmark used `(1,0,0,96)` (no seasonal differencing). Setting `D=1`
adds seasonal differencing which properly removes the persistent daily level. This
is the standard recommendation (Box & Jenkins). With `D=0`, the model tries to
capture the seasonal mean via AR/MA terms alone, which requires higher P or Q and
often leads to convergence issues.

**Computational cost:** fitting SARIMA(2,1,2)(1,1,1,96) on 4000 observations
typically takes 2–10 minutes on a MacBook M-series chip. This is a key data point
for the energy-cost section of the thesis.

---

### 5.4 Ridge Regression (`make_linear_forecaster`, `DirectForecaster`)

**Approach:** Direct multi-step forecasting — one Ridge model per horizon h.

**Features (17 total):**

| Feature | Description |
|---------|-------------|
| lag_0 | y[t] — current observation |
| lag_1 … lag_4 | y[t−1] … y[t−4] — recent steps (auto-correlation) |
| lag_8, lag_12 | ≈ 2h and 3h ago |
| lag_24, lag_48 | ≈ 6h and 12h ago |
| lag_96 | y[t−96] — same slot yesterday |
| lag_192, lag_288 | 2, 3 days ago |
| lag_672 | y[t−672] — same slot last week |
| h_sin, h_cos | hour-of-day (cyclic encoding) |
| d_sin, d_cos | day-of-week (cyclic encoding) |

**Why cyclic encoding for calendar features?**
Hour 23 and hour 0 are adjacent in time but far apart numerically. Encoding as
`sin(2π·h/24), cos(2π·h/24)` places them at adjacent points on the unit circle,
preserving the periodic structure without artificial discontinuities.

**Why Ridge (L2) and not Lasso or OLS?**
- OLS is unstable with 17 correlated lag features (multicollinearity).
- Lasso (L1) tends to zero out entire features, which is too aggressive for lags
  that are all mildly informative.
- Ridge (L2) shrinks all coefficients smoothly toward zero, which is appropriate
  when many features contribute small, distributed signals.

**Why direct (not recursive) multi-step?**
Recursive forecasting chains single-step predictions — the h=96 forecast is built
from 96 recursively predicted values, each adding its own error. Direct forecasting
trains a separate model for each h and predicts in one shot. Error accumulation is
avoided at the cost of training multiple models (Ben Taieb & Atiya, 2016).

**Known limitation at h=96:** Ridge with `alpha=1.0` sometimes performs worse than
the seasonal naive at h=96. This happens because:
1. Regularization shrinks the coefficient of lag_0 (the single most informative
   feature for h=96 prediction) toward zero, while the naive uses it with weight 1.
2. The 12 other lags add noise at long horizons that the regularizer cannot fully
   suppress.
Possible fix: reduce `alpha` (try 0.01–0.1) for the h=96 model, or clip lags to
only those relevant for long horizons. This result is an interesting finding in
itself — it shows that regularized linear models do not automatically beat simple
baselines at all horizons.

---

### 5.5 LightGBM (`make_lgbm_forecaster`, `DirectForecaster`)

Same direct multi-step architecture as Ridge, same feature set.

**Model parameters:**
```python
n_estimators=300, learning_rate=0.05, num_leaves=63,
min_child_samples=20, subsample=0.8, colsample_bytree=0.8
```

**Why LightGBM instead of XGBoost or Random Forest?**
- LightGBM is significantly faster than XGBoost on tabular data (leaf-wise growth
  vs level-wise).
- It handles the 17 lag features efficiently and naturally models non-linear
  interactions (e.g. load spike on cold Monday morning).
- Random Forest trains faster but is generally less accurate for time series
  regression.

**Why `num_leaves=63` (not the default 31)?**
Load forecasting exhibits complex non-linear patterns (sharp morning ramps,
holiday valleys). Deeper trees (more leaves) capture these patterns; 63 balances
expressiveness with overfitting risk. `min_child_samples=20` prevents the tree
from fitting noise in individual 15-min slots.

**Expected advantage over Ridge:** LightGBM can model interaction effects (e.g.
cold Monday morning at 8:00 AM = high load peak) that Ridge cannot capture without
explicit feature engineering. This advantage is most pronounced at medium horizons
(h=4, h=24).

---

## 6. Walk-Forward Evaluation (`walk_forward` in `run_walk_forward_benchmark.py`)

### What it is
At each forecast origin `t` in the test period (every `stride=96` steps = 1 day):
1. Compute predictions `ŷ[t+h]` for all horizons h ∈ {1, 4, 96}.
2. Record (ŷ[t+h], y[t+h]) as one error pair per horizon.

With `test_years=1.0` and `stride=96`, this produces **365 error pairs** per horizon.

### Why walk-forward and not single-origin?

**Single-origin bug (the original problem):**
In the earlier benchmark (`run_arima_energy_benchmark.py`), the model was fit once
and forecast 96 steps ahead from a single point. After ~10 steps, ARIMA reverts to
its unconditional mean. Result: h=4 and h=96 showed nearly identical errors because
both were essentially "mean forecasts." This does not reflect real deployment — in
production, the model is re-evaluated at every new 15-min interval.

**Walk-forward fix:**
Each origin uses the model with up-to-date information. At stride=96, the model
has seen one fresh day of data before each forecast, so h=96 genuinely tests
24-hour-ahead accuracy and is harder than h=4.

**Stride=96 (one per day) and not stride=1:**
Using stride=1 would give 35 000 origins (one per 15-min interval). This would
make ARIMA/SARIMA evaluation extremely slow (each step requires a Kalman filter
update + forecast). Stride=96 gives 365 origins — statistically robust (365 error
pairs) while completing in minutes for ARIMA.

---

## 7. Error Metrics (`scripts/utils.py`)

Three complementary metrics are reported per model × horizon combination:

| Metric | Formula | What it tells you |
|--------|---------|------------------|
| **MAE** | mean(|y − ŷ|) | Average absolute error in MW. Primary metric — directly interpretable. |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more. If RMSE >> MAE, there are a few large prediction spikes. |
| **MAPE** | mean(|y − ŷ| / |y|) × 100 | Relative error in %. Allows comparison across different load levels. |

**Why MAPE as percentage and not fraction?**
Standard convention in energy forecasting literature expresses MAPE in %. A MAPE
of 1.3% for Ridge at h=1 means the model's average error is 1.3% of the true load.

**MAPE caveat:** MAPE is undefined when `y=0`. Belgian load is always > 0 (floor
≈ 5000 MW), so this is not an issue here. The code still adds a tiny epsilon
(1e-9) to the denominator as a safety guard.

**Primary metric for ranking models: MAE.** MAE is:
- In the original unit (MW) → directly interpretable by grid operators.
- Robust to outliers (RMSE can be dominated by a few winter peaks).
- The most commonly reported metric in the load-forecasting literature.

---

## 8. Energy Measurement (`EnergyMeter` in `scripts/utils.py`)

**Tool:** [CodeCarbon](https://codecarbon.io/) — Python package that tracks CPU/GPU
power draw and converts to kWh and kg CO₂ equivalent using the grid carbon intensity
for the specified country (Belgium: `country_iso_code="BEL"`).

**What is measured:**
- `fit_energy_kwh` / `fit_emissions_kg`: energy consumed during model fitting only.
- `eval_energy_kwh` / `eval_emissions_kg`: energy consumed during the full
  walk-forward evaluation loop.
- `fit_duration_s` / `eval_duration_s`: wall-clock time for each phase.

**Why separate fit vs eval energy?**
In production, the model is trained once and deployed for many months. The fit
energy is a one-time cost; the eval energy is the recurring operational cost.
A model that trains expensively but predicts cheaply (ARIMA) differs fundamentally
from one that trains cheaply but predicts expensively (LightGBM with many trees).

**Why CodeCarbon and not powermetrics directly?**
`powermetrics` (macOS) requires sudo and produces very granular system-level data
that needs custom parsing. CodeCarbon abstracts over multiple hardware interfaces
(Intel RAPL, Apple PowerMetrics, GPU counters) and writes to a standard CSV format.
This makes the energy measurement reproducible across different hardware.

**Offline fallback:** CodeCarbon fetches the Belgian grid carbon-intensity from an
external API at startup. Without internet, use `--energy_tool none` to skip energy
tracking and still run the benchmark.

---

## 9. Output Structure

Each benchmark run creates:

```
results/wf_<YYYYMMDD_HHMMSS>/
├── metrics.csv                     ← all model × horizon metrics in one table
├── codecarbon/
│   └── emissions.csv               ← CodeCarbon raw output
├── <model>_h<h>_week.png           ← 2-week time-series plot (one per model×horizon)
├── <model>_h<h>_scatter.png        ← actual-vs-predicted scatter (one per model×horizon)
├── <model>_zoom14d.png             ← 14-day zoom, all horizons stacked (1 per model)
├── <model>_week_all.png            ← 7-day overlay, all horizons (1 per model)
└── <model>_seasonal.png            ← winter vs summer comparison (1 per model)
```

**Why one folder per run?**
Multiple runs (one per model, or repeated experiments) produce independent folders
with no risk of overwriting. Short filenames (`naive_h96_week.png`) avoid the long
prefixed names from the earlier benchmark.

---

## 10. Current Results (walk-forward, n=365 origins, stride=96)

Results from valid runs (`wf_20260316_*`):

| Model | h=1 (15min) | h=4 (1h) | h=96 (24h) |
|-------|------------|----------|------------|
| **Naive** | MAE=292 / MAPE=3.5% | MAE=297 / MAPE=3.7% | MAE=298 / MAPE=3.6% |
| **Linear (Ridge)** | MAE=109 / MAPE=1.3% | MAE=200 / MAPE=2.5% | MAE=381 / MAPE=4.5% ⚠️ |
| **ARIMA(2,1,2)** | MAE=110 / MAPE=1.3% | MAE=185 / MAPE=2.3% | MAE=691 / MAPE=8.2% ⚠️ |
| **SARIMA(2,1,2)(1,1,1,96)** | — | — | — (not yet run) |
| **LightGBM** | — | — | — (not yet run) |

### Interpreting the results

**h=1 (15 min ahead):**
Ridge ≈ ARIMA (both ≈ 109 MAE) >> Naive (292 MAE).
Short-term autocorrelation (lag_1, lag_2 features) provides a huge gain over the
seasonal naive, which ignores the most recent observations.

**h=4 (1 hour ahead):**
ARIMA (185) < Ridge (200) < Naive (297).
Both ML and ARIMA still substantially outperform naive. ARIMA's AR(2) captures
the 1-hour dynamics; Ridge's lag features also work well.

**h=96 (24 hours ahead) — key finding:**
- ARIMA (691 MAE): terrible. Without seasonality, ARIMA reverts to mean by h=96.
  This is WHY SARIMA is needed.
- Ridge (381 MAE): worse than naive (298 MAE). The regularized linear model
  cannot exploit the daily pattern as efficiently as the pure seasonal naive.
  This is an informative finding: regularization (alpha=1.0) shrinks the lag_0
  coefficient that naive uses directly. Reducing alpha or using dedicated
  horizon-specific feature selection could improve this.
- Naive (298 MAE): the best available result at h=96 until SARIMA is run.

### Expected results after running SARIMA and LightGBM

| h | Expected winner | Reason |
|---|----------------|--------|
| h=1 | LightGBM ≈ Ridge | Both exploit recent lags |
| h=4 | LightGBM > ARIMA > Ridge | LightGBM captures non-linear ramps |
| h=96 | SARIMA > Naive > LightGBM >> ARIMA | Seasonal differencing directly models daily cycle |

---

## 11. Terminal Commands

```bash
# Run each model independently (recommended for macBook)
python -m scripts.run_walk_forward_benchmark --models naive   --horizons 1 4 96
python -m scripts.run_walk_forward_benchmark --models linear  --horizons 1 4 96
python -m scripts.run_walk_forward_benchmark --models lgbm    --horizons 1 4 96
python -m scripts.run_walk_forward_benchmark --models arima   --horizons 1 4 96
python -m scripts.run_walk_forward_benchmark --models sarima  --horizons 1 4 96

# Without internet (skip CodeCarbon API call)
python -m scripts.run_walk_forward_benchmark --models naive --energy_tool none

# Skip plots (faster, for headless testing)
python -m scripts.run_walk_forward_benchmark --models linear --no_plots
```

---

## 12. Design Decisions Summary

| Decision | Choice | Reason |
|----------|--------|--------|
| Stationarity test | ADF | Tests unit root directly; AIC lag selection |
| Differencing order | d=1 | ADF confirms stationarity after 1 difference |
| Seasonal period | s=96 | 96 × 15min = 1 day (dominant seasonality) |
| SARIMA seasonal order | (1,1,1,96) | Minimal complete seasonal spec; D=1 for proper differencing |
| Walk-forward stride | 96 (1 day) | Balance: robust (365 origins) vs fast (ARIMA tractable) |
| Test window | last 1 calendar year | Covers all seasons; no data leakage |
| Train cap (ARIMA) | 4000 obs ≈ 42 days | Fitting time tractable; enough for parameter estimation |
| ARIMA state update | append(refit=False) | Kalman filter step only; 365× faster than re-fitting |
| ML approach | Direct multi-step | Avoids recursive error accumulation |
| ML regularization | Ridge α=1.0 | L2 shrinkage for correlated lags |
| Calendar encoding | sin/cos cyclic | Preserves hour/day periodicity |
| Primary metric | MAE | MW unit; interpretable; outlier-robust |
| Energy tool | CodeCarbon | Standard; multi-hardware; CSV output |

---

## 13. Comparison with Course Notebooks

The 6 course notebooks (KUL time-series forecasting course) cover similar material but
with different goals: teaching general tooling vs. building a reproducible energy-aware
benchmark. This section documents where the thesis diverges from the course, and why.

### 13.1 Library choice — Darts vs. statsmodels + sklearn

**Course (Notebooks 5 & 6):** use the [Darts](https://unit8co.github.io/darts/) library
(`NaiveSeasonal`, `RegressionModel`, `LightGBMModel`, `model.historical_forecasts()`).
Darts provides a unified `TimeSeries` API over many models.

**Thesis:** uses `statsmodels.tsa.SARIMAX` for ARIMA/SARIMA and `sklearn` (`Ridge`,
`LightGBM`) directly — **no Darts**.

**Why bypass Darts?**
1. **Energy transparency**: Darts wraps model internals, making it harder to isolate
   exactly what computation is measured by CodeCarbon (e.g. internal resampling,
   index alignment overhead). Direct API calls ensure that `EnergyMeter.start/stop`
   brackets only the model operation itself.
2. **Walk-forward control**: the thesis walk-forward loop (`walk_forward()`) needs
   explicit control over the Kalman filter state (`res.append(refit=False)`) for ARIMA
   — a detail that Darts' `historical_forecasts()` abstracts away.
3. **Portability**: removing Darts eliminates a heavy dependency (dozens of sub-packages)
   for an environment that may run on constrained research infrastructure.

---

### 13.2 Feature scaling — Ridge uses StandardScaler

**Course (Notebook 6):** experiments with multiple scalers (`StandardScaler`,
`MinMaxScaler`, `MaxAbsScaler`, `QuantileTransformer`) as a hyperparameter to tune.

**Thesis (`make_linear_forecaster` in `scripts/models_ml.py`):**
```python
make_pipeline(StandardScaler(), Ridge(alpha=1.0))
```
StandardScaler **is applied** — it is baked into the sklearn `Pipeline`. The scaler
is fit on the training features and applied consistently at predict time without any
data leakage. The choice of `StandardScaler` (mean=0, std=1) is appropriate for Ridge
because L2 regularization is scale-sensitive: without standardisation, features with
large numerical ranges (e.g. lag values in MW) would dominate the penalty term and
shrink smaller-range features disproportionately.

The thesis fixes `StandardScaler` rather than tuning the scaler type (see §13.4 on
hyperparameter tuning).

---

### 13.3 Belgian holidays covariate — absent (limitation)

**Course (Notebook 5):** adds `holidays.CountryHoliday('BE')` as an exogenous binary
feature in the `RegressionModel`. Belgian public holidays (Christmas, Easter, National
Day…) typically show load patterns resembling Sundays regardless of the weekday, so
the holiday flag captures a genuine effect.

**Thesis:** does **not** include a holiday feature.

**Impact:** the `d_sin` / `d_cos` day-of-week encoding does not distinguish a Tuesday
holiday from a regular Tuesday. On the ≈ 10 Belgian public holidays in the test year,
the ML models may over-predict (expecting a typical weekday load). This is a
**known limitation** — adding `holidays.CountryHoliday('BE')` as a binary feature
(or as an interaction term with `d_sin`/`d_cos`) is a natural next step.

---

### 13.4 Hyperparameter tuning — absent (deliberate)

**Course (Notebook 6):** uses [Optuna](https://optuna.org/) for hyperparameter search
over lag windows, scaler type, Ridge alpha, and LightGBM parameters. Both grid search
and random search strategies are demonstrated.

**Thesis:** all hyperparameters are **fixed** (Ridge `alpha=1.0`, LightGBM defaults
with `n_estimators=300, num_leaves=63`, lag set hard-coded in `DEFAULT_LAGS`).

**Why no tuning?**
The thesis objective is to benchmark the **energy cost of training and inference** at
a fixed model configuration, not to maximise predictive accuracy. Running Optuna
with 50–300 trials would itself consume significant energy and would confound the
energy measurement (tuning energy ≠ production training energy). Fixed hyperparameters
ensure every model run is comparable and that the measured energy reflects a single
fit-and-evaluate cycle.

This is an acknowledged trade-off: the reported accuracy numbers are not the best
achievable for each model class, but they are reproducible and comparable across models.

---

### 13.5 Calendar feature encoding — improvement over course

**Course (Notebooks 5 & 6):** uses raw integer features: `day_of_week ∈ {0…6}`,
`hour_of_day ∈ {0…23}` as regression inputs.

**Thesis:** uses **cyclic sin/cos encoding**:
```
h_sin = sin(2π · hour / 24),  h_cos = cos(2π · hour / 24)
d_sin = sin(2π · dow / 7),    d_cos = cos(2π · dow / 7)
```

**Why this is an improvement:**
Raw integer encoding creates a false discontinuity — hour 23 and hour 0 are adjacent
on the clock but differ by 23 in value. A linear model treats them as nearly orthogonal.
Cyclic encoding places consecutive hours at adjacent points on the unit circle, so the
model can smoothly interpolate the midnight transition. The same argument applies to
day-of-week (Sunday=6, Monday=0 are adjacent in the weekly cycle).

---

### 13.6 Evaluation metrics — more complete than course

**Course:** reports only **RMSE** (using `from darts.metrics import rmse`).

**Thesis:** reports three metrics per model × horizon:
- **MAE** (primary): directly interpretable in MW; robust to large errors.
- **RMSE**: penalises large errors more; identifies models prone to rare spikes.
- **MAPE**: relative (%) error; enables cross-scale comparisons.

RMSE alone can be dominated by a few extreme winter-peak misses and mislead the
ranking. Reporting all three metrics allows a more nuanced comparison: a model with
low MAE but high RMSE has occasional large errors (spikes); a model with low MAPE but
high MAE is relatively accurate but still off by many MW in absolute terms.

---

### 13.7 EDA techniques used in course but not in thesis scope

**Course (Notebook 4)** covers:
- **STL / MSTL decomposition** (`statsmodels.tsa.seasonal.STL`, `MSTL`): decomposes
  a series into trend, seasonal, and residual components.
- **Pearson correlation heatmap** (`df.corr()`, `sns.heatmap(annot=True)`): measures
  linear relationships between variables.
- **Autocorrelation plots** (`sm.graphics.tsa.plot_acf`, `plot_pacf`): used to
  determine ARIMA order manually (p from PACF, q from ACF).

**Thesis:** these EDA steps are informative but outside the scope of the benchmarking
pipeline. The ARIMA order (2,1,2) was selected based on ACF/PACF analysis performed
offline; the chosen order is fixed in the config. STL decomposition is not used as
a preprocessing step or model component (the seasonal structure is handled by SARIMA's
seasonal differencing and by the lag_96 / lag_672 features in the ML models).

---

### Summary table

| Aspect | Course notebooks | Thesis | Reason for divergence |
|--------|-----------------|--------|----------------------|
| Forecasting library | Darts | statsmodels + sklearn | Energy transparency, walk-forward control |
| Feature scaling | Tuned (4 scalers, Optuna) | Fixed: StandardScaler in Pipeline | Comparable energy measurement |
| Holidays feature | `CountryHoliday('BE')` binary flag | Absent | Known limitation; future work |
| Hyperparameter tuning | Optuna (grid + random) | None — fixed params | Energy benchmark requires fixed config |
| Calendar encoding | Raw integers (0–23, 0–6) | Cyclic sin/cos | Correct periodicity for linear models |
| Evaluation metrics | RMSE only (Darts) | MAE + RMSE + MAPE | More complete; less dominated by outliers |
| STL/MSTL decomposition | Used for EDA (Notebook 4) | Not in scope | Seasonal structure captured by model design |
| Correlation analysis | Pearson heatmap (Notebook 4) | Not in scope | EDA step, not part of benchmark pipeline |
