# Time Series Foundation Models — Implementation Reference

> Technical documentation of the four zero-shot foundation model forecasters
> implemented in this project. Covers model origins, architecture, Python
> packages, interface design, walk-forward integration, and energy measurement.
> Written as a self-contained reference for the thesis methodology chapter.

---

## 1. What Are Time Series Foundation Models?

Time series foundation models (TSFMs) are large neural networks pre-trained on
massive, diverse collections of real-world time series — covering finance,
energy, retail, weather, transport, and more — and released as open-weight
checkpoints on HuggingFace Hub. The key property is **zero-shot transfer**: the
model is never re-trained or fine-tuned on the target dataset. It receives a
window of recent observations (the *context*) and directly outputs forecasts,
relying entirely on patterns absorbed during pre-training.

This contrasts with every classical model in this project:

| Property | Classical (ARIMA, Ridge, LightGBM) | Foundation Model |
|---|---|---|
| Parameters estimated from target data | Yes | No — weights are frozen |
| `fit()` cost | Minutes to hours | Weight download only |
| Inference mechanism | Statistical equations / tree splits | Neural network forward pass |
| Training energy on Elia data | Non-zero | **Zero** |
| Inference energy per origin | Negligible | Depends on model size |
| Domain knowledge source | Elia series only | Pre-training corpus (millions of series) |

The connection to AI is direct: two of the four models (Chronos, TimesFM) apply
**transformer architectures** originally developed for large language models.
Chronos literally tokenises numerical values and runs a T5 encoder-decoder on
them. Lag-Llama applies the LLaMA decoder architecture. Moirai uses a masked
encoder transformer with patch-based tokenisation.

---

## 2. Models Implemented

Four models are implemented in `src/models_tsfm.py` and registered in the
`TSFM_MODELS` dictionary:

| Registry key | Class | Organisation | Architecture | Params |
|---|---|---|---|---|
| `chronos_mini` | `ChronosForecaster` | Amazon | T5 encoder-decoder, value tokenisation | 46 M |
| `timesfm_200m` | `TimesFMForecaster` | Google | Decoder-only transformer, patching | 200 M |
| `lag_llama` | `LagLlamaForecaster` | McGill / Mila | LLaMA decoder, probabilistic, GluonTS | ~50 M |
| `moirai_small` | `MoiraiForecaster` | Salesforce | Masked encoder transformer, patching | 91 M |

All four are used **zero-shot** — `fit()` loads weights but performs no
gradient updates.

---

## 3. Model Details

### 3.1 Chronos (Amazon)

**Paper:** Ansari et al. (2024), *Chronos: Learning the Language of Time Series*

**Architecture:** Chronos frames time series forecasting as a language modelling
problem. Numerical values are quantised into discrete tokens (similar to word
tokens in NLP) and fed into a T5 encoder-decoder transformer. The decoder
autoregressively samples future tokens, which are then de-tokenised back into
numerical forecasts.

**Variant used:** `amazon/chronos-t5-mini` (46 M parameters). The full model
family ranges from `tiny` (8 M) to `large` (710 M) and a newer `Bolt` variant.

**Output:** Probabilistic — the model draws `num_samples` trajectory samples.
This implementation takes the **median** across samples as the point forecast.

**Context window:** 512 observations (≈ 5.3 days at 15-min resolution).

**Python package:** `chronos-forecasting` (pip installable, Python ≤ 3.11).

**HuggingFace model ID:** `amazon/chronos-t5-mini`

**Device handling:** PyTorch. Auto-detects `mps` on Apple Silicon, falls back
to `cpu`. On a GPU cluster, pass `device="cuda"`.

---

### 3.2 TimesFM (Google)

**Paper:** Das et al. (2024), *A Decoder-Only Foundation Model for Time-Series Forecasting*

**Architecture:** TimesFM is a decoder-only transformer (similar to GPT) that
operates on *patches* of time series values rather than individual points. A
patch is a short sub-sequence treated as a single token, which reduces sequence
length and allows the model to scale to longer contexts. It was pre-trained on
a large corpus of real-world time series from Google's internal datasets plus
public sources.

**Variant used:** `google/timesfm-1.0-200m-pytorch` (200 M parameters). A
newer `timesfm-2.0-500m` also exists.

**Output:** Point forecast directly (no sampling). A confidence interval can be
requested but is not used here.

**Context window:** 512 observations (configurable via `--context_length`).

**`freq` parameter:** Set to `0` (high-frequency / sub-hourly), which is the
correct setting for 15-minute electricity data.

**Python package:** `timesfm` (pip installable, requires Python ≤ 3.11 and
PyTorch).

**HuggingFace model ID:** `google/timesfm-1.0-200m-pytorch`

**Device handling:** PyTorch backend; MPS / CPU / CUDA supported.

---

### 3.3 Lag-Llama

**Paper:** Rasul et al. (2023), *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting*

**Architecture:** Lag-Llama adapts the LLaMA decoder-only transformer to time
series. Instead of word embeddings, it uses *lag features* — past values at
fixed offsets (e.g. t−1, t−2, t−7, ...) — as the input representation. The
model outputs a parametric probability distribution (Student's t by default)
at each forecast horizon. It is integrated with the **GluonTS** probabilistic
forecasting library for data handling and prediction.

**Variant:** One public checkpoint (`lag-llama.ckpt`, ~50 M parameters). No
larger variant exists.

**Output:** Probabilistic — draws `num_samples` sample trajectories. The
**median** across samples is used as the point forecast.

**Context window:** 256 observations by default (configurable). Lag-Llama's
native training context is only 32 steps; the implementation applies **RoPE
linear scaling** to extend positional encodings to longer contexts:

```python
rope_factor = max(1.0, (context_length + max_h) / 32)
```

**Python package:** Not on PyPI. Must be cloned from GitHub and installed with
`pip install -e .`:

```
git clone https://github.com/time-series-foundation-models/lag-llama
cd lag-llama && pip install -e .
```

Also requires `gluonts` and `huggingface_hub`.

**HuggingFace model ID:** `time-series-foundation-models/Lag-Llama`

**Device handling:** Auto-detects `mps` on Apple Silicon, falls back to `cpu`.

---

### 3.4 Moirai (Salesforce)

**Paper:** Woo et al. (2024), *Unified Training of Universal Time Series Forecasting Transformers*

**Architecture:** Moirai uses a **masked encoder transformer** (similar to BERT
rather than GPT) trained with a unified multi-dataset, multi-variate objective.
Values are divided into *patches* before being tokenised, allowing variable
patch sizes that adapt to the context and horizon length. It is distributed via
the `uni2ts` Python package and uses the GluonTS predictor interface.

**Variant used:** `moirai_small` (91 M parameters). The code supports all three
sizes via the `variant` parameter: `"small"` (91 M), `"base"` (311 M),
`"large"` (1.1 B).

**Output:** Probabilistic — `num_samples` trajectories, **median** used as
point forecast.

**Context window:** 512 observations.

**Python package:** `uni2ts` (pip installable). Also requires `gluonts`.

**HuggingFace model IDs:**
- `Salesforce/moirai-1.0-R-small`
- `Salesforce/moirai-1.0-R-base`
- `Salesforce/moirai-1.0-R-large`

**Device handling:** via GluonTS predictor; CPU by default on the small variant.

---

## 4. Common Interface

All four classes implement the same three-method interface as the classical
models in `src/models.py`:

```python
model.fit(y_train: pd.Series) -> self
model.set_series(y_all: pd.Series, train_end_idx: int) -> None
model.predict(origin_idx: int) -> Dict[int, float]
```

This design means the walk-forward benchmark loop in `run_tsfm_benchmark.py`
treats foundation models identically to ARIMA or Ridge regression — no
special-casing required.

### fit()
For zero-shot models, `fit()` does not estimate any parameters. It calls the
internal `_load()` method, which downloads the checkpoint from HuggingFace Hub
on the first call and caches it for the lifetime of the object (lazy loading).
Subsequent calls to `fit()` return immediately.

### set_series()
Stores the full concatenated time series (train + test) so that `predict()` can
extract context windows at arbitrary origins without re-passing data.
`train_end_idx` is accepted for interface compatibility but is unused by
zero-shot models.

### predict()
Extracts the last `context_length` observations ending at `origin_idx`, runs a
single forward pass, and returns a dictionary `{h: ŷ}` for every configured
horizon. For probabilistic models (Chronos, Lag-Llama, Moirai) the median of
`num_samples` trajectories is returned. For TimesFM (deterministic) the raw
point forecast is returned directly.

---

## 5. Walk-Forward Benchmark Integration

Foundation models are evaluated using the same rolling-origin protocol as the
classical models (`run_benchmark.py`), implemented in `run_tsfm_benchmark.py`.

### Protocol
- **Test window:** last 1 calendar year of the Elia dataset (≈ 35 040
  observations at 15-min resolution).
- **Stride:** 96 steps (= 1 day) between consecutive forecast origins. This
  gives ≈ 365 evaluation origins per model.
- **Horizons:** h = 1 (15 min), h = 4 (1 h), h = 96 (24 h).
- At each origin *t*, the model predicts ŷ[t+h] for all h simultaneously from
  a single forward pass. No iterative multi-step prediction is used.

### Context Window at Each Origin
```
y_all:  [..., t-511, t-510, ..., t-1, t]  →  forward pass  →  ŷ[t+1], ..., ŷ[t+96]
                      ↑ context (512 steps)
```
The context window slides forward by `stride` steps at each origin; no weights
are updated between origins.

### Why This Is Faster Than Re-Fitting
Classical ARIMA advances its Kalman filter state without re-fitting MLE. TSFMs
have the equivalent advantage built in: the weights are completely frozen and a
context window extraction + forward pass takes milliseconds to seconds, not the
minutes required for MLE re-fitting.

### Results Schema
`run_tsfm_benchmark.py` writes one row per (model, horizon) to
`results/<run_id>/metrics.csv` using the **exact same column schema** as
`run_benchmark.py`. The two CSV files can be concatenated directly for thesis
comparison tables without any transformation.

---

## 6. Energy Measurement

### What Is Measured
The `EnergyMeter` (CodeCarbon wrapper in `src/metrics.py`) brackets **only the
walk-forward inference loop**. Model loading (weight download and GPU/MPS
transfer) happens inside `fit()` and is deliberately excluded from energy
measurement, because it is a one-time cost that would inflate the per-run
figure unpredictably depending on whether the HuggingFace cache is warm.

### Fit Energy
Because no gradient updates are performed, the fit energy is recorded as
**exactly 0.0 kWh** for all four foundation models:

```python
fit_energy = EnergyResult(energy_kwh=0.0, emissions_kg=0.0, ...)
```

This is a deliberate design choice: the zero value communicates that zero-shot
models incur **no training cost on the target dataset**, which is a key thesis
argument. The model-loading wall-clock time is still recorded in
`fit_duration_s`.

### Eval Energy
The inference loop energy is measured with two complementary tools running
simultaneously:

| Tool | Backend | Carbon intensity source | Output |
|---|---|---|---|
| **CodeCarbon** (`EnergyMeter`) | CPU/GPU power draw via RAPL or Apple PowerMetrics | electricityMap / CO₂ Signal | `codecarbon/emissions.csv` |
| **CarbonTracker** (`CarbonTrackerMeter`) | RAPL (Linux) / `sudo powermetrics` (macOS) | IPCC carbon factors, PUE=1.58 | `carbontracker/<tag>.log` |

Both tools expose the same `start()` / `stop()` interface and return an
`EnergyResult` dataclass. The results are merged with `cc_result.with_ct(ct_result)`:

```
EnergyResult
  ├── energy_kwh          ← CodeCarbon primary measurement
  ├── emissions_kg        ← CodeCarbon CO₂eq
  ├── ct_energy_kwh       ← CarbonTracker measurement
  └── ct_emissions_kg     ← CarbonTracker CO₂eq
```

Having two independent tools side-by-side allows cross-validation of the energy
figures and is a methodological strength of the thesis.

### Country Setting
CodeCarbon is configured with `country_iso_code="BEL"` so that the
CO₂-per-kWh conversion uses the Belgian grid carbon intensity, which is
appropriate since the models are evaluated in Belgium.

---

## 7. Probabilistic → Point Forecast Conversion

Chronos, Lag-Llama, and Moirai are inherently probabilistic: they output a
distribution over future values rather than a single number. This implementation
converts them to point forecasts by taking the **median (0.5 quantile)** of
`num_samples` drawn trajectories:

```python
# Chronos (PyTorch tensors)
median = torch.quantile(forecasts.float(), 0.5, dim=1)[0].numpy()

# Lag-Llama / Moirai (GluonTS SampleForecast)
median = forecasts[0].quantile(0.5)   # ndarray of shape (prediction_length,)
```

The median is preferred over the mean because it is robust to heavy-tailed
sample distributions and is the standard reporting convention in probabilistic
forecasting benchmarks (e.g. M4, Monash).

---

## 8. Software Environment

Foundation models require a **separate virtual environment** (`.venv_fm`)
because their dependencies conflict with the `statsmodels` package used by
ARIMA/SARIMA:

| Environment | Python | Key packages |
|---|---|---|
| `.venv` (classical) | 3.14 | `statsmodels`, `scikit-learn`, `lightgbm` |
| `.venv_fm` (TSFMs) | **3.11** | `chronos-forecasting`, `timesfm`, `lag_llama` (from GitHub), `uni2ts`, `gluonts`, `torch` |

The TSFM benchmark deliberately imports **only** `src/data.py`,
`src/metrics.py`, and `src/models_tsfm.py` — not `src/models.py` or
`src/evaluate.py` — because those modules depend on `statsmodels`, which is
absent in `.venv_fm`.

### Activating and Running

```bash
source .venv_fm/bin/activate

# All four models
python run_tsfm_benchmark.py --models chronos_mini timesfm_200m lag_llama moirai_small

# Chronos only, no energy tracking (faster for testing)
python run_tsfm_benchmark.py --models chronos_mini --energy_tool none

# Reduce number of origins for a quick smoke test
python run_tsfm_benchmark.py --models timesfm_200m --n_origins 10
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--models` | all four | Which foundation models to run |
| `--context_length` | 512 | Historical observations fed as context |
| `--num_samples` | 100 | Trajectory samples for probabilistic models |
| `--stride` | 96 | Steps between walk-forward origins (96 = 1 day) |
| `--horizons` | 1 4 96 | Forecast horizons in 15-min steps |
| `--energy_tool` | both | `codecarbon`, `carbontracker`, `both`, or `none` |
| `--device` | auto | `cpu`, `mps`, or `cuda` |
| `--n_origins` | None | Cap origins (e.g. `5` for smoke test) |

---

## 9. Key Design Decisions and Justifications

### Zero-shot evaluation only (no fine-tuning)
Fine-tuning a foundation model on the Elia training set would require GPU
compute, hyperparameter search, and careful regularisation to avoid catastrophic
forgetting. The thesis research question is specifically about the **energy
efficiency of zero-shot models**: can a model trained elsewhere be deployed
without any target-domain training cost and still achieve competitive accuracy?
Fine-tuning would invalidate this comparison.

### Context length of 512 steps (≈ 5.3 days)
512 steps at 15-min resolution captures more than 5 full daily cycles, which is
sufficient for the model to observe the periodic pattern. It also matches the
pre-training context length of Chronos and TimesFM, ensuring the model operates
in its designed regime. Shorter contexts would deprive the model of seasonality
information; longer contexts would exceed the pre-training regime for some
models (especially Lag-Llama, whose native context is 32 steps).

### Median as point forecast
Using the median of probabilistic samples rather than the mean is consistent
with the evaluation metrics (MAE is minimised by the median; RMSE by the mean).
Since MAE is the primary comparison metric in this thesis (it is in MW and
directly interpretable), the median is the theoretically correct choice.

### num_samples = 100
100 samples gives a stable median estimate (standard error ≈ 1.25 × IQR / √100)
without dominating inference time. Increasing to 500 samples would reduce
estimation noise by a factor of √5 ≈ 2.2 at 5× the cost.

### Energy brackets inference only, not model loading
Model loading time varies by orders of magnitude depending on network speed and
whether the HuggingFace cache is warm. Including it would make energy figures
non-reproducible across machines. Inference energy is the operationally relevant
cost: in production, a model would be loaded once and used for many predictions.

### Same CSV schema as classical benchmark
By enforcing the same column names and structure in both `run_benchmark.py` and
`run_tsfm_benchmark.py`, the two result files can be concatenated with
`pd.concat` for a unified comparison table. No post-processing or column
renaming is needed.

### RoPE scaling for Lag-Llama
Lag-Llama was trained with a context of 32 steps. Using a 256-step context
without adjustment would produce incoherent positional encodings (positions
33–256 were never seen during training). Rotary Position Embedding (RoPE)
linear scaling multiplies the position indices by `1/factor`, effectively
compressing the larger context back into the model's trained range:

```python
rope_factor = max(1.0, (context_length + max_h) / 32)
# e.g. (256 + 96) / 32 = 11.0
```

This is the standard technique used by LLaMA to extend context windows beyond
the training length.

---

## 10. File Map

| File | Role |
|---|---|
| `src/models_tsfm.py` | All four forecaster classes + `TSFM_MODELS` registry |
| `run_tsfm_benchmark.py` | CLI entry point: walk-forward loop, energy metering, CSV logging |
| `src/metrics.py` | `EnergyMeter` (CodeCarbon), `CarbonTrackerMeter`, `EnergyResult` |
| `src/data.py` | `load_timeseries`, `clean_timeseries`, `split_last_n_years`, `append_run_csv` |
| `results/<run_id>/metrics.csv` | Per-run results (same schema as classical benchmark) |
| `results/<run_id>/predictions.parquet` | Full per-origin forecast arrays |
| `results/<run_id>/codecarbon/` | Raw CodeCarbon `emissions.csv` |
| `results/<run_id>/carbontracker/` | Raw CarbonTracker log files |

---

## 11. References

- Ansari, A. B. et al. (2024). *Chronos: Learning the Language of Time Series.* arXiv:2403.07815.
- Das, A. et al. (2024). *A Decoder-Only Foundation Model for Time-Series Forecasting.* arXiv:2310.10688.
- Rasul, K. et al. (2023). *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting.* arXiv:2310.08278.
- Woo, G. et al. (2024). *Unified Training of Universal Time Series Forecasting Transformers.* arXiv:2402.02592.
- Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.* International Journal of Forecasting. (GluonTS basis)
- Lannelongue, L., Grealey, J., & Inouye, M. (2021). *Green Algorithms: Quantifying the Carbon Footprint of Computation.* Advanced Science. (CarbonTracker methodology)
- Bannour, B. et al. (2021). *Sustainable AI: Environmental Implications, Challenges, and Opportunities.* (CodeCarbon methodology)
