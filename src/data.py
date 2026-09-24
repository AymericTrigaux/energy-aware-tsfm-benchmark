from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_timeseries(
    path: str,
    timestamp_col: str,
    target_col: str,
    freq: str,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    """Load a time series from CSV or Parquet into a regular datetime grid."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".csv", ".txt"):
        read_kwargs = {}
        if delimiter is not None:
            read_kwargs["delimiter"] = delimiter
        df = pd.read_csv(path, **read_kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use CSV or Parquet.")

    if timestamp_col not in df.columns:
        raise KeyError(f"timestamp_col='{timestamp_col}' not found in columns: {list(df.columns)}")
    if target_col not in df.columns:
        raise KeyError(f"target_col='{target_col}' not found in columns: {list(df.columns)}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=False, errors="coerce")
    if df[timestamp_col].isna().any():
        bad = df[df[timestamp_col].isna()].head(5)
        raise ValueError(f"Some timestamps could not be parsed. Example bad rows:\n{bad}")

    df = df[[timestamp_col, target_col]].copy()
    df = df.set_index(timestamp_col).sort_index()

    # asfreq inserts NaN rows for missing slots (e.g. DST gaps) — without this,
    # a gap shifts every subsequent lag feature by one position.
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq(freq)

    return df.rename(columns={target_col: "y"})


def clean_timeseries(
    df: pd.DataFrame,
    impute_method: str = "interpolate",
    interpolate_method: str = "time",
) -> pd.DataFrame:
    """Fill NaNs in the 'y' column. Methods: 'interpolate', 'ffill', 'drop'."""
    out = df.copy()
    if "y" not in out.columns:
        raise KeyError("Expected column 'y' in DataFrame.")

    if impute_method == "drop":
        out = out.dropna(subset=["y"])
    elif impute_method == "ffill":
        out["y"] = out["y"].ffill().bfill()
    elif impute_method == "interpolate":
        out["y"] = out["y"].interpolate(method=interpolate_method, limit_direction="both")
        out["y"] = out["y"].ffill().bfill()
    else:
        raise ValueError(f"Unknown impute_method='{impute_method}'")

    return out


def split_last_n_years(
    df: pd.DataFrame,
    test_years: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, test) where test covers the last N calendar years."""
    last_ts = df.index[-1]
    whole_years = int(test_years)
    frac_years = test_years - whole_years
    extra_days = int(round(frac_years * 365.25))
    test_start = last_ts - pd.DateOffset(years=whole_years, days=extra_days)

    train = df[df.index < test_start]
    test = df[df.index >= test_start]

    if len(train) == 0:
        raise ValueError(f"test_years={test_years} consumes the entire dataset.")
    if len(test) == 0:
        raise ValueError(
            f"test_years={test_years}: no data in the last {test_years} year(s). "
            f"Check the dataset range."
        )
    return train, test


def freq_to_minutes(freq: str) -> int:
    """Convert a pandas frequency string to minutes per step."""
    return int(pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds() // 60)


def append_run_csv(path: str, row: dict) -> None:
    """Append a single result row to a cumulative CSV, creating it if absent."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_prev = pd.read_csv(path)
        df_out = pd.concat([df_prev, df_row], ignore_index=True)
    else:
        df_out = df_row
    df_out.to_csv(path, index=False)


def save_json(path: str, obj: dict) -> None:
    """Write a dict to an indented JSON file, creating parent directories."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
