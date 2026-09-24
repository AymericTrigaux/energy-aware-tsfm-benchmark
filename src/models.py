from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


# --- ARIMA / SARIMA ---

@dataclass
class ArimaConfig:
    """ARIMA/SARIMA order specification."""
    order: Tuple[int, int, int] = (2, 1, 2)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: str = "n"


def fit_sarimax(
    train: pd.Series,
    cfg: ArimaConfig,
    maxiter: int = 200,
    method: str = "lbfgs",
) -> sm.tsa.statespace.sarimax.SARIMAXResults:
    """Fit a SARIMAX model and return the results object."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = sm.tsa.SARIMAX(
            train,
            order=cfg.order,
            seasonal_order=cfg.seasonal_order,
            trend=cfg.trend,
            # Safe after differencing; avoids constrained optimisation issues.
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, method=method, maxiter=maxiter)
    return res


class NaivePredictor:
    """Seasonal naive: ŷ[t+h] = y[t+h-s]. Zero training cost baseline."""

    def __init__(self, s: int = 96):
        self.s = s
        self._y: Optional[pd.Series] = None

    def fit(self, y_train: pd.Series) -> "NaivePredictor":
        self._y = y_train
        return self

    def set_series(self, y_all: pd.Series) -> None:
        self._y = y_all

    def predict(self, origin_idx: int, horizons: List[int]) -> Dict[int, float]:
        preds = {}
        n = len(self._y)
        for h in horizons:
            src = origin_idx + h - self.s
            preds[h] = float(self._y.iloc[src]) if 0 <= src < n else float("nan")
        return preds


class ARIMAPredictor:
    """Walk-forward ARIMA/SARIMA. fit() trains once; predict() advances state via Kalman (no re-fit)."""

    def __init__(
        self,
        cfg: ArimaConfig,
        horizons: List[int],
        maxiter: int = 50,
        max_train_obs: int = 4000,
        label: str = "arima",
    ):
        self.cfg = cfg
        self.horizons = horizons
        self.max_h = max(horizons)
        self.maxiter = maxiter
        self.max_train_obs = max_train_obs
        self.label = label
        self._res = None
        self._y: Optional[pd.Series] = None
        self._last_appended: int = -1

    def fit(self, y_train: pd.Series) -> "ARIMAPredictor":
        if len(y_train) > self.max_train_obs:
            y_fit = y_train.iloc[-self.max_train_obs:]
            print(f"  [{self.label}] capping train to last {self.max_train_obs} obs "
                  f"(full train: {len(y_train):,})")
        else:
            y_fit = y_train

        print(f"  [{self.label}] fitting on {len(y_fit):,} obs …")
        self._res = fit_sarimax(y_fit, self.cfg, maxiter=self.maxiter)
        self._y = y_train
        self._last_appended = len(y_train) - 1
        return self

    def set_series(self, y_all: pd.Series, train_end_idx: int) -> None:
        self._y = y_all
        self._last_appended = train_end_idx - 1

    def predict(self, origin_idx: int) -> Dict[int, float]:
        new_start = self._last_appended + 1
        if origin_idx >= new_start:
            new_obs = self._y.iloc[new_start: origin_idx + 1]
            if len(new_obs) > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self._res = self._res.append(new_obs.values, refit=False)
                self._last_appended = origin_idx

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fc = self._res.get_forecast(steps=self.max_h)
        predicted = fc.predicted_mean

        return {h: float(predicted.iloc[h - 1]) for h in self.horizons}


# --- ML models — shared feature engineering ---

# Covers: current value (lag_0), recent steps (1–4), sub-hourly (8, 12),
# intra-day (24, 48), daily seasonality (96, 192, 288), weekly (672).
DEFAULT_LAGS: List[int] = [0, 1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 288, 672]


def _holiday_feature(index: pd.DatetimeIndex, country: str = "BE") -> np.ndarray:
    """Return a binary holiday flag array (1.0 on public holidays, 0.0 otherwise)."""
    try:
        import holidays as hol_pkg
    except ImportError:
        warnings.warn(
            "Package 'holidays' not installed — is_holiday feature set to 0. "
            "Fix: pip install holidays",
            stacklevel=3,
        )
        return np.zeros(len(index), dtype=float)

    try:
        be_hol = hol_pkg.country_holidays(country)
    except AttributeError:
        be_hol = hol_pkg.CountryHoliday(country)  # legacy API

    return np.array([1.0 if ts.date() in be_hol else 0.0 for ts in index], dtype=float)


def _calendar_features(
    index: pd.DatetimeIndex,
    use_holidays: bool = False,
) -> Dict[str, np.ndarray]:
    """Cyclic sin/cos encoding for hour-of-day and day-of-week."""
    h = index.hour + index.minute / 60.0
    dow = index.dayofweek.astype(float)
    feats: Dict[str, np.ndarray] = {
        "h_sin": np.sin(2 * np.pi * h / 24),
        "h_cos": np.cos(2 * np.pi * h / 24),
        "d_sin": np.sin(2 * np.pi * dow / 7),
        "d_cos": np.cos(2 * np.pi * dow / 7),
    }
    if use_holidays:
        feats["is_holiday"] = _holiday_feature(index)
    return feats


def build_feature_matrix(
    y: pd.Series,
    lags: List[int],
    horizon: int,
    calendar: bool = True,
    use_holidays: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build (X, y_target) for direct h-step-ahead prediction. Drops boundary NaN rows."""
    feats = {f"lag_{k}": y.shift(k) for k in lags}
    X = pd.DataFrame(feats, index=y.index)

    if calendar and hasattr(y.index, "hour"):
        for name, arr in _calendar_features(y.index, use_holidays=use_holidays).items():
            X[name] = arr

    target = y.shift(-horizon)
    valid = X.notna().all(axis=1) & target.notna()
    return X[valid], target[valid]


def build_x_at_origin(
    y: pd.Series,
    origin_idx: int,
    lags: List[int],
    calendar: bool = True,
    use_holidays: bool = False,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for inference at a given origin index."""
    n = len(y)
    data: Dict[str, float] = {}

    for k in lags:
        idx = origin_idx - k
        data[f"lag_{k}"] = float(y.iloc[idx]) if 0 <= idx < n else np.nan

    if calendar:
        ts = y.index[min(origin_idx, n - 1)]
        h_frac = ts.hour + ts.minute / 60.0
        dow = float(ts.dayofweek)
        data["h_sin"] = np.sin(2 * np.pi * h_frac / 24)
        data["h_cos"] = np.cos(2 * np.pi * h_frac / 24)
        data["d_sin"] = np.sin(2 * np.pi * dow / 7)
        data["d_cos"] = np.cos(2 * np.pi * dow / 7)
        if use_holidays:
            data["is_holiday"] = float(_holiday_feature(pd.DatetimeIndex([ts]))[0])

    return pd.DataFrame([data])


class DirectForecaster:
    """Direct multi-step forecaster: one sklearn estimator clone per horizon."""

    def __init__(
        self,
        base_estimator,
        lags: Optional[List[int]] = None,
        calendar: bool = True,
        use_holidays: bool = False,
        seasonal_period: int = 0,
    ):
        self._base = base_estimator
        self.lags: List[int] = lags if lags is not None else DEFAULT_LAGS
        self.calendar = calendar
        self.use_holidays = use_holidays
        self.seasonal_period = int(seasonal_period)
        self._models: Dict[int, object] = {}
        self._y: Optional[pd.Series] = None

    def fit(self, y: pd.Series, horizons: List[int]) -> "DirectForecaster":
        from sklearn.base import clone
        self._y = y
        s = self.seasonal_period
        for h in horizons:
            X, y_h = build_feature_matrix(
                y, self.lags, horizon=h,
                calendar=self.calendar,
                use_holidays=self.use_holidays,
            )
            target = y_h.values
            if s > 0:
                # baseline aligned with X.index: baseline[t] = y[t + h - s]
                baseline = y.shift(-(h - s)).reindex(X.index).values
                valid = ~np.isnan(baseline)
                X = X.iloc[valid]
                target = target[valid] - baseline[valid]
            m = clone(self._base)
            m.fit(X.values, target)
            self._models[h] = m
            print(f"  [ML] h={h:3d}  trained on {len(target):,} samples"
                  f"{' (seasonal-residual)' if s > 0 else ''}")
        return self

    def set_series(self, y_all: pd.Series) -> None:
        self._y = y_all

    def predict(self, origin_idx: int) -> Dict[int, float]:
        x = build_x_at_origin(
            self._y, origin_idx, self.lags,
            calendar=self.calendar,
            use_holidays=self.use_holidays,
        )
        s = self.seasonal_period
        n = len(self._y)
        preds: Dict[int, float] = {}
        for h, m in self._models.items():
            p = float(m.predict(x)[0])
            if s > 0:
                base_idx = origin_idx + h - s
                if 0 <= base_idx < n:
                    p += float(self._y.iloc[base_idx])
                else:
                    p = float("nan")
            preds[h] = p
        return preds


class MLPredictor:
    """Walk-forward ML predictor wrapping a DirectForecaster. Fit once, evaluate many times."""

    def __init__(self, forecaster: DirectForecaster, label: str = "linear"):
        self._forecaster = forecaster
        self.label = label

    def fit(self, y_train: pd.Series, horizons: List[int]) -> "MLPredictor":
        print(f"  [{self.label}] fitting on {len(y_train):,} obs …")
        self._forecaster.fit(y_train, horizons)
        return self

    def set_series(self, y_all: pd.Series) -> None:
        self._forecaster.set_series(y_all)

    def predict(self, origin_idx: int) -> Dict[int, float]:
        return self._forecaster.predict(origin_idx)


# --- Factory functions ---

def make_linear_forecaster(
    lags: Optional[List[int]] = None,
    alpha: float = 1.0,
    use_holidays: bool = False,
    seasonal_period: int = 0,
) -> DirectForecaster:
    """Ridge regression with StandardScaler. L2 regularisation handles correlated lags."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    est = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return DirectForecaster(est, lags=lags, use_holidays=use_holidays,
                            seasonal_period=seasonal_period)


def make_lgbm_forecaster(
    lags: Optional[List[int]] = None,
    use_holidays: bool = False,
    seasonal_period: int = 0,
    **lgbm_kwargs,
) -> DirectForecaster:
    """LightGBM gradient-boosting forecaster."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm") from exc

    params = {
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "verbose":           -1,
        **lgbm_kwargs,
    }
    return DirectForecaster(LGBMRegressor(**params), lags=lags, use_holidays=use_holidays,
                            seasonal_period=seasonal_period)
