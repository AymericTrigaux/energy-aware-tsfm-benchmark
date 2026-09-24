"""Foundation model wrappers (Chronos, TimesFM, Lag-Llama, Moirai) for the walk-forward benchmark.

All classes expose the same fit / set_series / predict interface as src.models.
Zero-shot: fit() loads weights but performs no gradient updates.
Run with .venv_fm (Python 3.11): source .venv_fm/bin/activate

Every checkpoint is pinned to an explicit commit via the `_REVISION` class
attribute, so a re-run loads the same weights as the May 2026 thesis runs even
if the upstream repo's `main` moves. See docs/checkpoint_revisions.md for the
revision table and how it was recovered. A pinned revision that is neither
cached nor still on the Hub fails loudly rather than silently falling back.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --- Chronos (Amazon) ---

class ChronosForecaster:
    """Zero-shot wrapper for Amazon Chronos-T5-mini (probabilistic, median point forecast)."""

    name: str = "chronos_mini"
    _CONTEXT_LENGTH: int = 512
    _MODEL_ID: str = "amazon/chronos-t5-mini"
    _REVISION: str = "bd6a4fde8403b8469acd0abd52852b29dbe61c7b"

    def __init__(
        self,
        horizons: List[int],
        device: Optional[str] = None,
        num_samples: int = 50,
    ) -> None:
        self.horizons = horizons
        self.max_h = max(horizons)
        self.device = device
        self.num_samples = num_samples
        self._pipeline = None
        self._y: Optional[pd.Series] = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from chronos import ChronosPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Run: pip install chronos-forecasting"
            ) from exc

        import torch  # type: ignore

        dev = self.device
        if dev is None:
            if torch.cuda.is_available():
                dev = "cuda"
            elif torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"

        print(f"  [{self.name}] loading {self._MODEL_ID}@{self._REVISION[:8]} "
              f"on device={dev} …")
        self._pipeline = ChronosPipeline.from_pretrained(
            self._MODEL_ID,
            revision=self._REVISION,
            device_map=dev,
            torch_dtype=torch.bfloat16,
        )

    def fit(self, y_train: pd.Series) -> "ChronosForecaster":
        self._load()
        return self

    def set_series(self, y_all: pd.Series, train_end_idx: int) -> None:
        self._y = y_all

    def predict(self, origin_idx: int) -> Dict[int, float]:
        import torch  # type: ignore

        start = max(0, origin_idx - self._CONTEXT_LENGTH + 1)
        context = self._y.iloc[start : origin_idx + 1].values.astype(np.float32)
        ctx_tensor = torch.tensor(context).unsqueeze(0)  # (1, seq_len)

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*prediction length.*")
            # forecasts: (1, num_samples, max_h)
            forecasts = self._pipeline.predict(
                ctx_tensor,
                prediction_length=self.max_h,
                num_samples=self.num_samples,
            )

        # Median over the sample dimension → (max_h,)
        median = torch.quantile(forecasts.float(), 0.5, dim=1)[0].numpy()
        return {h: float(median[h - 1]) for h in self.horizons}

    def predict_batch(self, origin_indices: List[int]) -> Dict[int, Dict[int, float]]:
        """Batch predict: one forward pass for many origins (5–10× faster)."""
        import torch  # type: ignore

        contexts: List["torch.Tensor"] = []
        for oi in origin_indices:
            start = max(0, oi - self._CONTEXT_LENGTH + 1)
            ctx = self._y.iloc[start: oi + 1].values.astype(np.float32)
            contexts.append(torch.tensor(ctx))

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*prediction length.*")
            forecasts = self._pipeline.predict(
                contexts,
                prediction_length=self.max_h,
                num_samples=self.num_samples,
            )
        # forecasts: (B, num_samples, max_h)
        median = torch.quantile(forecasts.float(), 0.5, dim=1).cpu().numpy()
        return {
            oi: {h: float(median[i, h - 1]) for h in self.horizons}
            for i, oi in enumerate(origin_indices)
        }


# --- TimesFM (Google) ---

class TimesFMForecaster:
    """Zero-shot wrapper for Google TimesFM-1.0-200M (PyTorch backend)."""

    name: str = "timesfm_200m"
    _MODEL_ID: str = "google/timesfm-1.0-200m-pytorch"
    _REVISION: str = "0581e2c56cb06feb51cfd98fc2b4005b74f7187b"
    # TimesFmCheckpoint has no `revision` field, but TimesFmTorch.load_from_checkpoint
    # uses `checkpoint.path` verbatim when it is set and only falls back to
    # snapshot_download(repo_id) otherwise. So resolving the pinned commit to a
    # concrete file ourselves is what actually enforces the pin here.
    _CKPT_FILENAME: str = "torch_model.ckpt"

    @classmethod
    def _pinned_ckpt_path(cls) -> str:
        from huggingface_hub import hf_hub_download  # type: ignore
        return hf_hub_download(
            repo_id=cls._MODEL_ID,
            filename=cls._CKPT_FILENAME,
            revision=cls._REVISION,
        )

    def __init__(
        self,
        horizons: List[int],
        context_length: int = 512,
        point_forecast_mode: str = "median",
    ) -> None:
        try:
            import timesfm  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "timesfm is not installed. "
                "Activate the .venv_fm environment and run: pip install timesfm"
            ) from exc

        self.horizons = horizons
        self.max_h = max(horizons)
        self.context_length = context_length
        self.point_forecast_mode = point_forecast_mode
        self._tfm = None
        self._y: Optional[pd.Series] = None

    def _load(self) -> None:
        if self._tfm is not None:
            return

        import timesfm  # type: ignore

        print(f"  [{self.name}] loading {self._MODEL_ID}@{self._REVISION[:8]} …")
        self._tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="pytorch",
                per_core_batch_size=32,
                horizon_len=self.max_h,
                point_forecast_mode=self.point_forecast_mode,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self._MODEL_ID,
                path=self._pinned_ckpt_path(),
            ),
        )
        # timesfm >= 1.2 loads weights in the constructor; initialize() was
        # removed in 1.3.0.  No explicit call needed.

    def fit(self, y_train: pd.Series) -> "TimesFMForecaster":
        self._load()
        return self

    def set_series(self, y_all: pd.Series, train_end_idx: int) -> None:
        self._y = y_all

    def predict(self, origin_idx: int) -> Dict[int, float]:
        start = max(0, origin_idx - self.context_length + 1)
        context = self._y.iloc[start : origin_idx + 1].values.astype(np.float32)

        # TimesFM expects a list of 1-D arrays (one per time series).
        # freq=0 → high-frequency (sub-hourly/hourly).
        point_forecast, _ = self._tfm.forecast(
            inputs=[context],
            freq=[0],
        )
        # point_forecast: (1, max_h) → take row 0
        fc = point_forecast[0]
        return {h: float(fc[h - 1]) for h in self.horizons}

    def predict_batch(self, origin_indices: List[int]) -> Dict[int, Dict[int, float]]:
        """Batch predict: one forecast() call for many origins."""
        contexts: List[np.ndarray] = []
        for oi in origin_indices:
            start = max(0, oi - self.context_length + 1)
            ctx = self._y.iloc[start: oi + 1].values.astype(np.float32)
            contexts.append(ctx)

        point_forecast, _ = self._tfm.forecast(
            inputs=contexts,
            freq=[0] * len(contexts),
        )
        fc = np.asarray(point_forecast)  # (B, max_h)
        return {
            oi: {h: float(fc[i, h - 1]) for h in self.horizons}
            for i, oi in enumerate(origin_indices)
        }


# --- Lag-Llama (time-series-foundation-models) ---

class LagLlamaForecaster:
    """Zero-shot wrapper for Lag-Llama via GluonTS.

    Install: git clone https://github.com/time-series-foundation-models/lag-llama && pip install -e .
    """

    name: str = "lag_llama"
    _HF_REPO: str = "time-series-foundation-models/Lag-Llama"
    _HF_FILENAME: str = "lag-llama.ckpt"
    _REVISION: str = "72dcfc29da106acfe38250a60f4ae29d1e56a3d9"
    _NATIVE_CONTEXT: int = 32  # Lag-Llama's training context length

    def __init__(
        self,
        horizons: List[int],
        context_length: int = 256,
        num_samples: int = 100,
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "lag_llama is not installed.  Clone and install it:\n"
                "  git clone https://github.com/time-series-foundation-models/lag-llama\n"
                "  cd lag-llama && pip install -e .\n"
                "Then activate .venv_fm and re-run."
            ) from exc

        self.horizons = horizons
        self.max_h = max(horizons)
        self.context_length = context_length
        self.num_samples = num_samples
        self.device = device
        self.batch_size = batch_size
        self._predictor = None
        self._y: Optional[pd.Series] = None
        self._freq: Optional[str] = None

    def _load(self) -> None:
        if self._predictor is not None:
            return

        import torch  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore

        # PyTorch ≥2.6 defaults torch.load to weights_only=True; Lag-Llama's
        # Lightning checkpoint pickles GluonTS classes that aren't on the safe
        # list. Force the legacy behavior just for the load — we trust the HF
        # checkpoint we pulled.
        _orig_torch_load = torch.load
        def _trusted_load(*a, **kw):
            kw["weights_only"] = False
            return _orig_torch_load(*a, **kw)
        torch.load = _trusted_load
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator  # type: ignore
        finally:
            pass  # leave the patch in place for load_from_checkpoint below

        dev = self.device
        if dev is None:
            if torch.cuda.is_available():
                dev = "cuda"
            elif torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"

        print(f"  [{self.name}] downloading checkpoint from "
              f"{self._HF_REPO}@{self._REVISION[:8]} …")
        ckpt_path = hf_hub_download(
            repo_id=self._HF_REPO,
            filename=self._HF_FILENAME,
            revision=self._REVISION,
        )

        # RoPE linear scaling so context_length > 32 stays coherent
        rope_factor = max(
            1.0,
            (self.context_length + self.max_h) / self._NATIVE_CONTEXT,
        )

        print(
            f"  [{self.name}] building predictor "
            f"(context={self.context_length}, max_h={self.max_h}, "
            f"rope_factor={rope_factor:.1f}, device={dev}) …"
        )
        # Read architecture hyperparameters from the checkpoint so the
        # estimator constructs a model that matches the saved weights.
        _ckpt = torch.load(ckpt_path, map_location="cpu")
        _model_kw = _ckpt.get("hyper_parameters", {}).get("model_kwargs", {})
        del _ckpt

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=self.max_h,
            context_length=self.context_length,
            input_size=_model_kw.get("input_size", 1),
            n_layer=_model_kw.get("n_layer", 4),
            n_embd_per_head=_model_kw.get("n_embd_per_head", 64),
            n_head=_model_kw.get("n_head", 4),
            scaling=_model_kw.get("scaling", "robust"),
            time_feat=_model_kw.get("time_feat", True),
            device=torch.device(dev) if not isinstance(dev, torch.device) else dev,
            batch_size=self.batch_size,
            num_parallel_samples=self.num_samples,
            rope_scaling={
                "type": "linear",
                "factor": rope_factor,
            },
        )
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self._predictor = estimator.create_predictor(
            transformation, lightning_module
        )

    def fit(self, y_train: pd.Series) -> "LagLlamaForecaster":
        # pandas 2.x str(freq) returns "<15 * Minutes>" which GluonTS rejects;
        # use .freqstr which returns the canonical alias "15min" / "h" etc.
        if hasattr(y_train.index, "freq") and y_train.index.freq is not None:
            self._freq = y_train.index.freq.freqstr
        else:
            self._freq = "15min"
        self._load()
        return self

    def set_series(self, y_all: pd.Series, train_end_idx: int) -> None:
        self._y = y_all

    def predict(self, origin_idx: int) -> Dict[int, float]:
        from gluonts.dataset.common import ListDataset  # type: ignore

        start_idx = max(0, origin_idx - self.context_length + 1)
        context = self._y.iloc[start_idx : origin_idx + 1].values.astype(np.float32)
        start_ts = self._y.index[start_idx]

        dataset = ListDataset(
            [{"start": start_ts, "target": context}],
            freq=self._freq or "15min",
        )

        forecasts = list(self._predictor.predict(dataset))
        # SampleForecast.quantile(0.5) → ndarray of shape (prediction_length,)
        median = forecasts[0].quantile(0.5)
        return {h: float(median[h - 1]) for h in self.horizons}

    def predict_batch(self, origin_indices: List[int]) -> Dict[int, Dict[int, float]]:
        from gluonts.dataset.common import ListDataset  # type: ignore

        records = []
        for oi in origin_indices:
            start_idx = max(0, oi - self.context_length + 1)
            ctx = self._y.iloc[start_idx: oi + 1].values.astype(np.float32)
            records.append({"start": self._y.index[start_idx], "target": ctx})

        dataset = ListDataset(records, freq=self._freq or "15min")
        forecasts = list(self._predictor.predict(dataset))
        return {
            oi: {h: float(forecasts[i].quantile(0.5)[h - 1]) for h in self.horizons}
            for i, oi in enumerate(origin_indices)
        }


# --- Moirai (Salesforce) ---

class MoiraiForecaster:
    """Zero-shot wrapper for Salesforce Moirai via GluonTS.

    Variants: moirai-1.0-R-small (13.8M), -base (91.4M), -large (311.0M).
    Counts are from the pinned checkpoints; see docs/checkpoint_revisions.md.
    Install: pip install uni2ts
    """

    name: str = "moirai_small"
    _HF_REPO_TMPL: str = "Salesforce/moirai-1.0-R-{variant}"
    # Keyed by variant because the repo id is built from a template.
    _REVISION_BY_VARIANT: Dict[str, str] = {
        "small": "f5bd5d01f0d67de856107d43abdd637380aae0a3",
        "base":  "4fa939a8800d9da346c0280f3d9aeba0d2d35877",
        "large": "b6b84e3fbeb9e6927d4271d625441fb51ab6bbd8",
    }

    def __init__(
        self,
        horizons: List[int],
        context_length: int = 512,
        num_samples: int = 100,
        variant: str = "small",
        patch_size: str = "auto",
        batch_size: int = 16,
    ) -> None:
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "uni2ts is not installed. "
                "Activate the .venv_fm environment and run: pip install uni2ts"
            ) from exc

        self.horizons = horizons
        self.max_h = max(horizons)
        self.context_length = context_length
        self.num_samples = num_samples
        self.variant = variant
        # Allow the user to pass a numeric patch size as a string from the CLI;
        # uni2ts accepts both "auto" and ints (8/16/32/64/128).
        if isinstance(patch_size, str) and patch_size.isdigit():
            patch_size = int(patch_size)
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.name = f"moirai_{variant}"
        self._hf_repo = self._HF_REPO_TMPL.format(variant=variant)
        if variant not in self._REVISION_BY_VARIANT:
            raise ValueError(
                f"No pinned revision for Moirai variant '{variant}'. "
                f"Known: {sorted(self._REVISION_BY_VARIANT)}. "
                f"Add it to _REVISION_BY_VARIANT and docs/checkpoint_revisions.md."
            )
        self._revision = self._REVISION_BY_VARIANT[variant]
        self._predictor = None
        self._y: Optional[pd.Series] = None
        self._freq: Optional[str] = None

    def _load(self) -> None:
        if self._predictor is not None:
            return

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # type: ignore

        print(
            f"  [{self.name}] loading {self._hf_repo}@{self._revision[:8]} "
            f"(context={self.context_length}, max_h={self.max_h}) …"
        )
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self._hf_repo,
                                                revision=self._revision),
            prediction_length=self.max_h,
            context_length=self.context_length,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = model.create_predictor(batch_size=self.batch_size)

    def fit(self, y_train: pd.Series) -> "MoiraiForecaster":
        # pandas 2.x str(freq) returns "<15 * Minutes>" which GluonTS rejects;
        # use .freqstr which returns the canonical alias "15min" / "h" etc.
        if hasattr(y_train.index, "freq") and y_train.index.freq is not None:
            self._freq = y_train.index.freq.freqstr
        else:
            self._freq = "15min"
        self._load()
        return self

    def set_series(self, y_all: pd.Series, train_end_idx: int) -> None:
        self._y = y_all

    def predict(self, origin_idx: int) -> Dict[int, float]:
        from gluonts.dataset.common import ListDataset  # type: ignore

        start_idx = max(0, origin_idx - self.context_length + 1)
        context = self._y.iloc[start_idx : origin_idx + 1].values.astype(np.float32)
        start_ts = self._y.index[start_idx]

        dataset = ListDataset(
            [{"start": start_ts, "target": context}],
            freq=self._freq or "15min",
        )

        forecasts = list(self._predictor.predict(dataset))
        median = forecasts[0].quantile(0.5)  # (prediction_length,)
        return {h: float(median[h - 1]) for h in self.horizons}

    def predict_batch(self, origin_indices: List[int]) -> Dict[int, Dict[int, float]]:
        from gluonts.dataset.common import ListDataset  # type: ignore

        records = []
        for oi in origin_indices:
            start_idx = max(0, oi - self.context_length + 1)
            ctx = self._y.iloc[start_idx: oi + 1].values.astype(np.float32)
            records.append({"start": self._y.index[start_idx], "target": ctx})

        dataset = ListDataset(records, freq=self._freq or "15min")
        forecasts = list(self._predictor.predict(dataset))
        return {
            oi: {h: float(forecasts[i].quantile(0.5)[h - 1]) for h in self.horizons}
            for i, oi in enumerate(origin_indices)
        }


class ChronosBoltForecaster:
    """Zero-shot wrapper for Amazon Chronos-Bolt-mini (deterministic quantile, context 2048)."""

    name: str = "chronos_bolt_mini"
    _CONTEXT_LENGTH: int = 2048
    _MODEL_ID: str = "amazon/chronos-bolt-mini"
    _REVISION: str = "251268337516a88e253628c43e1d26ec577b376b"

    def __init__(
        self,
        horizons: List[int],
        device: Optional[str] = None,
        num_samples: int = 100,  # accepted for parity; Bolt is deterministic
    ) -> None:
        self.horizons = horizons
        self.max_h = max(horizons)
        self.device = device
        self.num_samples = num_samples
        self._pipeline = None
        self._y: Optional[pd.Series] = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from chronos import BaseChronosPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is not installed. "
                "Run: pip install chronos-forecasting"
            ) from exc

        import torch  # type: ignore
        dev = self.device
        if dev is None:
            if torch.cuda.is_available():
                dev = "cuda"
            elif torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"

        print(f"  [{self.name}] loading {self._MODEL_ID}@{self._REVISION[:8]} "
              f"on device={dev} …")
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self._MODEL_ID,
            revision=self._REVISION,
            device_map=dev,
            torch_dtype=torch.bfloat16,
        )

    def fit(self, y_train: pd.Series) -> "ChronosBoltForecaster":
        self._load()
        return self

    def set_series(self, y_all: pd.Series, train_end_idx: int) -> None:
        self._y = y_all

    def predict(self, origin_idx: int) -> Dict[int, float]:
        import torch  # type: ignore

        start = max(0, origin_idx - self._CONTEXT_LENGTH + 1)
        context = self._y.iloc[start: origin_idx + 1].values.astype(np.float32)
        ctx_tensor = torch.tensor(context).unsqueeze(0)  # (1, seq_len)

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*prediction length.*")
            quantiles, _mean = self._pipeline.predict_quantiles(
                inputs=ctx_tensor,
                prediction_length=self.max_h,
                quantile_levels=[0.5],
            )
        # quantiles shape: (1, max_h, n_quantiles) — take the only quantile (median)
        median = quantiles[0, :, 0].float().cpu().numpy()
        return {h: float(median[h - 1]) for h in self.horizons}

    def predict_batch(self, origin_indices: List[int]) -> Dict[int, Dict[int, float]]:
        import torch  # type: ignore

        contexts: List["torch.Tensor"] = []
        for oi in origin_indices:
            start = max(0, oi - self._CONTEXT_LENGTH + 1)
            ctx = self._y.iloc[start: oi + 1].values.astype(np.float32)
            contexts.append(torch.tensor(ctx))

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*prediction length.*")
            quantiles, _ = self._pipeline.predict_quantiles(
                inputs=contexts,
                prediction_length=self.max_h,
                quantile_levels=[0.5],
            )
        # quantiles: (B, max_h, 1) → take the median (last dim = single quantile)
        median = quantiles[..., 0].float().cpu().numpy()  # (B, max_h)
        return {
            oi: {h: float(median[i, h - 1]) for h in self.horizons}
            for i, oi in enumerate(origin_indices)
        }


class MoiraiBaseForecaster(MoiraiForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="base", **kwargs)


class MoiraiLargeForecaster(MoiraiForecaster):
    """Moirai-1.0-R-large (1.1B params)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="large", **kwargs)


# --- Moirai 2.0 ---

class Moirai2SmallForecaster:
    """Zero-shot wrapper for Salesforce Moirai-2.0-R-small (~11M params, deterministic quantile)."""

    name: str = "moirai2_small"
    _HF_REPO: str = "Salesforce/moirai-2.0-R-small"
    _REVISION: str = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"

    def __init__(
        self,
        horizons: List[int],
        context_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "uni2ts>=2.0 with moirai2 support is not installed. "
                "Activate .venv_fm and run: pip install -U uni2ts"
            ) from exc

        self.horizons = horizons
        self.max_h = max(horizons)
        self.context_length = context_length
        self.batch_size = batch_size
        self._predictor = None
        self._y: Optional[pd.Series] = None
        self._freq: Optional[str] = None

    def _load(self) -> None:
        if self._predictor is not None:
            return

        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module  # type: ignore

        print(
            f"  [{self.name}] loading {self._HF_REPO}@{self._REVISION[:8]} "
            f"(context={self.context_length}, max_h={self.max_h}) …"
        )
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self._HF_REPO,
                                                 revision=self._REVISION),
            prediction_length=self.max_h,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = model.create_predictor(batch_size=self.batch_size)

    def fit(self, y_train: pd.Series) -> "Moirai2SmallForecaster":
        if hasattr(y_train.index, "freq") and y_train.index.freq is not None:
            self._freq = y_train.index.freq.freqstr
        else:
            self._freq = "15min"
        self._load()
        return self

    def set_series(self, y_all: pd.Series, train_end_idx: int) -> None:
        self._y = y_all

    def predict(self, origin_idx: int) -> Dict[int, float]:
        from gluonts.dataset.common import ListDataset  # type: ignore

        start_idx = max(0, origin_idx - self.context_length + 1)
        context = self._y.iloc[start_idx: origin_idx + 1].values.astype(np.float32)
        dataset = ListDataset(
            [{"start": self._y.index[start_idx], "target": context}],
            freq=self._freq or "15min",
        )
        forecasts = list(self._predictor.predict(dataset))
        median = forecasts[0].quantile(0.5)
        return {h: float(median[h - 1]) for h in self.horizons}

    def predict_batch(self, origin_indices: List[int]) -> Dict[int, Dict[int, float]]:
        from gluonts.dataset.common import ListDataset  # type: ignore

        records = []
        for oi in origin_indices:
            start_idx = max(0, oi - self.context_length + 1)
            ctx = self._y.iloc[start_idx: oi + 1].values.astype(np.float32)
            records.append({"start": self._y.index[start_idx], "target": ctx})

        dataset = ListDataset(records, freq=self._freq or "15min")
        forecasts = list(self._predictor.predict(dataset))
        return {
            oi: {h: float(forecasts[i].quantile(0.5)[h - 1]) for h in self.horizons}
            for i, oi in enumerate(origin_indices)
        }


# --- Larger Chronos variants ---

class ChronosLargeForecaster(ChronosForecaster):
    """Amazon Chronos-T5-large (710M params)."""

    name: str = "chronos_large"
    _MODEL_ID: str = "amazon/chronos-t5-large"
    _REVISION: str = "0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a"


class ChronosBoltBaseForecaster(ChronosBoltForecaster):
    """Amazon Chronos-Bolt-base (205M params)."""

    name: str = "chronos_bolt_base"
    _MODEL_ID: str = "amazon/chronos-bolt-base"
    _REVISION: str = "5d9f166d69f47aef3401367a7b842e78fe97b121"


# --- TimesFM 2.0 ---

class TimesFM500MForecaster(TimesFMForecaster):
    """Zero-shot wrapper for Google TimesFM-2.0-500M (50-layer, context up to 2048)."""

    name: str = "timesfm_500m"
    _MODEL_ID: str = "google/timesfm-2.0-500m-pytorch"
    _REVISION: str = "dc2443792ce5516872b89b37cf1bc058c3bf0c10"
    _NATIVE_CONTEXT: int = 2048

    def __init__(
        self,
        horizons: List[int],
        context_length: int = 1024,
        point_forecast_mode: str = "median",
    ) -> None:
        super().__init__(
            horizons=horizons,
            context_length=context_length,
            point_forecast_mode=point_forecast_mode,
        )

    def _load(self) -> None:
        if self._tfm is not None:
            return

        import timesfm  # type: ignore

        print(f"  [{self.name}] loading {self._MODEL_ID}@{self._REVISION[:8]} "
              f"(context_len={self.context_length}) …")
        self._tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="pytorch",
                per_core_batch_size=32,
                horizon_len=self.max_h,
                num_layers=50,
                use_positional_embedding=False,
                # Tie hparams.context_len to the constructor arg so attention
                # actually shrinks when context_length < 2048. Without this,
                # timesfm pads back to its own hparams and the speedup is lost.
                context_len=self.context_length,
                point_forecast_mode=self.point_forecast_mode,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self._MODEL_ID,
                path=self._pinned_ckpt_path(),
            ),
        )


TSFM_MODELS: Dict[str, type] = {
    "chronos_mini":      ChronosForecaster,
    "chronos_large":     ChronosLargeForecaster,
    "chronos_bolt_mini": ChronosBoltForecaster,
    "chronos_bolt_base": ChronosBoltBaseForecaster,
    "timesfm_200m":      TimesFMForecaster,
    "timesfm_500m":      TimesFM500MForecaster,
    "lag_llama":         LagLlamaForecaster,
    "moirai_small":      MoiraiForecaster,
    "moirai_base":       MoiraiBaseForecaster,
    "moirai_large":      MoiraiLargeForecaster,
    "moirai2_small":     Moirai2SmallForecaster,
}
