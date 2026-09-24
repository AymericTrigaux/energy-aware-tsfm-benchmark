#!/usr/bin/env python3
"""Publication-quality diagnostic plots for the walk-forward benchmark.

Generates four plot families per run into a timestamped sub-folder under figures/.

Usage:
    python visualize_results.py
    python visualize_results.py --predictions results/classical/wf_xxx/predictions.parquet
    python visualize_results.py --predictions results/classical/wf_xxx/predictions.parquet \\
        --energy results/classical/wf_xxx/codecarbon/emissions.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Style ---
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
})

# --- Palette (shared with plot_energy.py — see src/palette.py) ---
ACTUAL_COLOR = "#1a237e"

from src.palette import (
    MODEL_COLOR as MODEL_COLORS,
    MODEL_LABEL as MODEL_DISPLAY,
    MODEL_ORDER,
)

HORIZON_LABELS: Dict[int, str] = {
    1:  "h=1  (15 min)",
    4:  "h=4  (1 hour)",
    96: "h=96 (24 hours)",
}


# --- Metrics ---

def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def _mae(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs(yt - yp)))

def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)

# --- Data loading ---

def load_predictions(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(path) if p.suffix == ".parquet" else pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    for alias, canon in [("datetime", "timestamp"), ("h", "horizon"), ("date", "timestamp")]:
        if alias in df.columns and canon not in df.columns:
            df = df.rename(columns={alias: canon})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["horizon"]   = df["horizon"].astype(int)
    df["model"]     = df["model"].str.strip().str.lower()
    return df.sort_values(["model", "horizon", "timestamp"]).reset_index(drop=True)


def load_raw_series(path: Optional[str]) -> Optional[pd.Series]:
    if path is None or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        ts_col  = next((c for c in df.columns if c in ("datetime", "timestamp", "date")), None)
        val_col = next(
            (c for c in df.columns if any(k in c for k in ("totalload", "load", "mw", "value", "y"))),
            None,
        )
        if ts_col is None or val_col is None:
            return None
        s = pd.Series(
            pd.to_numeric(df[val_col], errors="coerce").values,
            index=pd.to_datetime(df[ts_col]),
            name="actual",
        ).sort_index().dropna()
        return s
    except Exception as exc:
        print(f"  [warn] Could not load raw series: {exc}")
        return None


def load_energy(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except Exception as exc:
        print(f"  [warn] Could not load energy file: {exc}")
        return None


# --- Season window detection ---

def _find_month_window(
    timestamps: pd.DatetimeIndex,
    month: int,
    n_weeks: int = 3,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) for n_weeks starting at the first occurrence of month."""
    in_month = timestamps[timestamps.month == month]
    if len(in_month) < n_weeks * 5:   # need at least n_weeks of daily points
        return None
    start = pd.Timestamp(in_month[0]).normalize()
    end   = start + pd.Timedelta(weeks=n_weeks)
    end   = min(end, pd.Timestamp(timestamps[-1]))
    return start, end


def _find_extreme_window(
    df_h1: pd.DataFrame,
    n_weeks: int = 3,
    high_load: bool = True,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Find the n_weeks window with the highest (high_load=True) or lowest mean actual load."""
    s = df_h1.set_index("timestamp")["actual"].sort_index()
    window = n_weeks * 7
    if len(s) < window:
        return s.index[0], s.index[-1]
    rolling = s.rolling(window).mean()
    center  = rolling.idxmax() if high_load else rolling.idxmin()
    start   = pd.Timestamp(center) - pd.Timedelta(weeks=n_weeks // 2)
    end     = start + pd.Timedelta(weeks=n_weeks)
    start   = max(start, s.index[0])
    end     = min(end,   s.index[-1])
    return start, end


def get_season_windows(df: pd.DataFrame) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    all_ts = pd.DatetimeIndex(sorted(df["timestamp"].unique()))
    df_h1  = (
        df[df["horizon"] == 1][["timestamp", "actual"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )

    winter = _find_month_window(all_ts, month=1, n_weeks=3)
    if winter is None:
        print("  [info] January not in test set — using highest-load 3-week window for winter.")
        winter = _find_extreme_window(df_h1, n_weeks=3, high_load=True)

    summer = _find_month_window(all_ts, month=7, n_weeks=3)
    if summer is None:
        print("  [info] July not in test set — using lowest-load 3-week window for summer.")
        summer = _find_extreme_window(df_h1, n_weeks=3, high_load=False)

    return {"winter": winter, "summer": summer}


# --- Tick helpers ---

def _ticks_every_n_days(ax: plt.Axes, n: int = 3) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=n))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d/%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)


def _ticks_6h(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H\n%-d/%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=7)


def _ticks_monthly(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)


# --- Sparse-prediction helpers ---

def _pred_median_gap_hours(ts_pred: np.ndarray) -> float:
    """Median gap between consecutive prediction timestamps, in hours."""
    if len(ts_pred) < 2:
        return float("inf")
    diffs = np.diff(ts_pred.astype("datetime64[m]")).astype(float)  # minutes
    return float(np.median(diffs)) / 60.0


def _plot_sparse_needles(
    ax: plt.Axes,
    ts_pred: np.ndarray,
    yt: np.ndarray,
    yp: np.ndarray,
    color: str,
    label: str,
) -> None:
    """Lollipop plot for sparse predictions: each diamond = predicted value, line = error."""
    first = True
    for ts, a, p in zip(ts_pred, yt, yp):
        ts_dt = pd.Timestamp(ts)
        ax.plot([ts_dt, ts_dt], [a, p], color=color, lw=1.5, alpha=0.65, zorder=3)
        ax.scatter([ts_dt], [p],
                   color=color, s=55, marker="D", alpha=0.95, zorder=6,
                   label=label if first else "")
        ax.scatter([ts_dt], [a],
                   color=ACTUAL_COLOR, s=22, marker="o", alpha=0.70, zorder=5)
        first = False


# --- Actual-series retrieval ---

def _actual_in_range(
    raw: Optional[pd.Series],
    sub: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, values) for actual load in [t0, t1].
    Prefers full-resolution raw series; falls back to daily actuals in sub."""
    if raw is not None:
        clipped = raw.loc[t0:t1]
        return clipped.index.values, clipped.values
    mask = (sub["timestamp"] >= t0) & (sub["timestamp"] <= t1)
    tmp  = sub[mask].drop_duplicates("timestamp").sort_values("timestamp")
    return tmp["timestamp"].values, tmp["actual"].values


# --- Plot 1 — Seasonal panels (2×3 grid: winter/summer × h=1/h=4/h=96) ---

def plot_seasonal_panels(
    model_key: str,
    df: pd.DataFrame,
    season_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    raw: Optional[pd.Series],
    out_dir: str,
    horizons: List[int],
    dpi: int,
) -> None:
    color  = MODEL_COLORS.get(model_key, "#555")
    label  = MODEL_DISPLAY.get(model_key, model_key.upper())
    df_mod = df[df["model"] == model_key]

    seasons = ["winter", "summer"]
    n_rows, n_cols = len(seasons), len(horizons)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
    fig.suptitle(f"{label} — Seasonal Behavior (Test Period)", fontsize=13, y=1.01)

    # Compute global y-range for consistent axes within this model
    y_lo, y_hi = np.inf, -np.inf
    for season in seasons:
        t0, t1 = season_windows[season]
        for h in horizons:
            sub = df_mod[(df_mod["horizon"] == h) &
                         (df_mod["timestamp"] >= t0) & (df_mod["timestamp"] <= t1)]
            if sub.empty:
                continue
            y_lo = min(y_lo, sub[["actual", "predicted"]].min().min())
            y_hi = max(y_hi, sub[["actual", "predicted"]].max().max())

    if np.isfinite(y_lo):
        margin = (y_hi - y_lo) * 0.06
        ylim   = (y_lo - margin, y_hi + margin)
    else:
        ylim = None

    ln_actual = ln_pred = ln_error = None   # for shared legend

    for ri, season in enumerate(seasons):
        t0, t1 = season_windows[season]

        for ci, h in enumerate(horizons):
            ax  = axes[ri, ci]
            sub = df_mod[df_mod["horizon"] == h].sort_values("timestamp")
            win = sub[(sub["timestamp"] >= t0) & (sub["timestamp"] <= t1)]

            if win.empty:
                ax.set_title(f"{season.capitalize()} — {HORIZON_LABELS[h]}\n(no data)")
                continue

            ts_pred = win["timestamp"].values
            yt = win["actual"].values.astype(float)
            yp = win["predicted"].values.astype(float)

            # Actual load (continuous if raw available, else daily)
            ts_act, val_act = _actual_in_range(raw, win, t0, t1)

            sparse = _pred_median_gap_hours(ts_pred) > 3.0

            ln_actual, = ax.plot(ts_act, val_act, color=ACTUAL_COLOR, lw=1.0,
                                 label="Actual", zorder=3)

            if sparse:
                _plot_sparse_needles(ax, ts_pred, yt, yp, color, label)
                ln_pred  = ax.collections[-1] if ax.collections else None
                fill     = None
            else:
                err_lo = np.minimum(yt, yp)
                err_hi = np.maximum(yt, yp)
                fill = ax.fill_between(ts_pred, err_lo, err_hi,
                                       color="#e53935", alpha=0.15, zorder=2,
                                       label="Error envelope")
                ln_pred, = ax.plot(ts_pred, yp, color=color, lw=1.0, alpha=0.85,
                                   label=label, zorder=4)

            win_rmse = _rmse(yt, yp)
            win_mae  = _mae(yt, yp)
            with np.errstate(divide="ignore", invalid="ignore"):
                win_mape = float(np.mean(np.abs((yt - yp) / yt))) * 100.0 if len(yt) else float("nan")
            ax.set_title(
                f"{season.capitalize()} — {HORIZON_LABELS[h]}\n"
                f"RMSE={win_rmse:.0f} MW   MAE={win_mae:.0f} MW   MAPE={win_mape:.2f}%",
                fontsize=10,
            )
            ax.set_ylabel("Load (MW)", fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            _ticks_every_n_days(ax, n=3)

            if ln_error is None and fill is not None:
                ln_error = fill

    # Shared legend at figure bottom (outside panels)
    handles = []
    if ln_actual: handles.append(ln_actual)
    if ln_pred:   handles.append(ln_pred)
    if ln_error:  handles.append(ln_error)
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=9, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_key}_seasonal_overview.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"    saved {path}")


# --- Plot 2 — 7-day zoom (one file per model × season) ---

def _annotate_max_error(ax: plt.Axes,
                        ts: np.ndarray, yt: np.ndarray, yp: np.ndarray) -> None:
    if len(ts) == 0:
        return
    errs = np.abs(yt - yp)
    i    = int(np.argmax(errs))
    x_pt = pd.Timestamp(ts[i])
    sign_err = float(yt[i] - yp[i])
    y_ann = float(yp[i]) + sign_err * 0.5
    ax.annotate(
        f"{sign_err:+.0f} MW",
        xy=(x_pt, float(yp[i])),
        xytext=(x_pt, y_ann),
        fontsize=7,
        color="#b71c1c",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=0.8),
    )


def plot_zoom(
    model_key: str,
    df: pd.DataFrame,
    season_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    raw: Optional[pd.Series],
    out_dir: str,
    horizons: List[int],
    dpi: int,
) -> None:
    color = MODEL_COLORS.get(model_key, "#555")
    label = MODEL_DISPLAY.get(model_key, model_key.upper())
    df_mod = df[df["model"] == model_key]

    for season, (t0_3w, t1_3w) in season_windows.items():
        # Days 8–14 of the 3-week window, snapped to Monday → Sunday
        zoom_start = _snap_to_monday(t0_3w + pd.Timedelta(days=7))
        zoom_end   = zoom_start + pd.Timedelta(days=7)
        zoom_end   = min(zoom_end, t1_3w)

        n_h = len(horizons)
        # Height ratios: [3, 1] per horizon
        height_ratios = []
        for _ in range(n_h):
            height_ratios.extend([3, 1])

        fig = plt.figure(figsize=(12, 14))
        gs  = fig.add_gridspec(n_h * 2, 1, height_ratios=height_ratios, hspace=0.45)
        fig.suptitle(
            f"{label} — 7-day Zoom  "
            f"{zoom_start.strftime('%-d/%m')} – {zoom_end.strftime('%-d/%m/%Y')}"
            f"  ({season.capitalize()})",
            fontsize=12, y=1.01,
        )

        for hi, h in enumerate(horizons):
            ax_main  = fig.add_subplot(gs[hi * 2])
            ax_err   = fig.add_subplot(gs[hi * 2 + 1], sharex=ax_main)

            sub  = df_mod[df_mod["horizon"] == h].sort_values("timestamp")
            win  = sub[(sub["timestamp"] >= zoom_start) & (sub["timestamp"] <= zoom_end)]

            if win.empty:
                ax_main.set_title(f"{HORIZON_LABELS[h]} — no data in zoom window")
                continue

            ts_pred = win["timestamp"].values
            yt = win["actual"].values.astype(float)
            yp = win["predicted"].values.astype(float)

            # Continuous actual
            ts_act, val_act = _actual_in_range(raw, win, zoom_start, zoom_end)
            sparse = _pred_median_gap_hours(ts_pred) > 3.0

            # ── Main panel ────────────────────────────────────────────────────
            ax_main.plot(ts_act, val_act, color=ACTUAL_COLOR, lw=1.0,
                         label="Actual", zorder=3)
            if sparse:
                _plot_sparse_needles(ax_main, ts_pred, yt, yp, color, label)
            else:
                ax_main.plot(ts_pred, yp, color=color, lw=1.2, alpha=0.85,
                             label=label, zorder=4)

            # Vertical day separators
            for day in pd.date_range(
                zoom_start.normalize() + pd.Timedelta(days=1), zoom_end, freq="D"
            ):
                ax_main.axvline(day, color="gray", lw=0.6, ls="--", alpha=0.3, zorder=1)

            win_rmse = _rmse(yt, yp)
            win_mae  = _mae(yt, yp)
            with np.errstate(divide="ignore", invalid="ignore"):
                win_mape = float(np.mean(np.abs((yt - yp) / yt))) * 100.0 if len(yt) else float("nan")
            ax_main.set_title(
                f"{HORIZON_LABELS[h]}  RMSE={win_rmse:.0f} MW   MAE={win_mae:.0f} MW   MAPE={win_mape:.2f}%",
                fontsize=10,
            )
            ax_main.set_ylabel("Load (MW)", fontsize=9)
            ax_main.legend(loc="upper right", fontsize=8, framealpha=0.85)
            _annotate_max_error(ax_main, ts_pred, yt, yp)
            plt.setp(ax_main.get_xticklabels(), visible=False)

            # ── Error panel ───────────────────────────────────────────────────
            signed = yt - yp
            bar_colors = ["#1565C0" if e >= 0 else "#c62828" for e in signed]
            if sparse:
                bar_w = pd.Timedelta(hours=20)   # wide bars for 1/day predictions
            elif len(ts_pred) > 1:
                bar_w = pd.Timedelta(hours=14)
            else:
                bar_w = pd.Timedelta(hours=8)
            ax_err.bar(ts_pred, signed, width=bar_w,
                       color=bar_colors, alpha=0.75, zorder=3)
            ax_err.axhline(0, color="black", lw=0.8, zorder=4)
            ax_err.set_ylabel("Error\n(MW)", fontsize=7)
            ax_err.tick_params(axis="both", labelsize=7)

            # Tick labels only on last error panel — daily to avoid overcrowding
            if hi < n_h - 1:
                plt.setp(ax_err.get_xticklabels(), visible=False)
            else:
                ax_err.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%-d/%m"))
                ax_err.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
                plt.setp(ax_err.get_xticklabels(), rotation=0, ha="center", fontsize=7)

        fig.tight_layout()
        path = os.path.join(out_dir, f"{model_key}_zoom_{season}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close("all")
        print(f"    saved {path}")


# --- Plot 2b — 1-week detail zoom ---

def _snap_to_monday(t: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday of the week containing t (at midnight)."""
    t = t.normalize()
    return t - pd.Timedelta(days=t.weekday())  # weekday() 0=Mon … 6=Sun


def _pick_detail_window(
    season_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    df_mod: pd.DataFrame,
    horizons: List[int],
    n_days: int,
    snap_to_monday: bool = False,
) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    """
    Return (start, end, label) for the detail window.

    Strategy: use the first horizon's predictions in the winter window,
    offset by 7 days to avoid cold-start, then pick n_days.
    Falls back to the summer window if winter has too few points.
    If snap_to_monday=True the start is snapped to the Monday of that week.
    """
    for season, (t0, t1) in season_windows.items():
        anchor = t0 + pd.Timedelta(days=7)
        if snap_to_monday:
            anchor = _snap_to_monday(anchor)
        anchor = anchor.normalize()
        end    = anchor + pd.Timedelta(days=n_days)
        end    = min(end, t1)
        h      = horizons[0]
        sub    = df_mod[df_mod["horizon"] == h]
        pts    = sub[(sub["timestamp"] >= anchor) & (sub["timestamp"] <= end)]
        if not pts.empty:
            label = f"{anchor.strftime('%-d/%m/%Y')} – {end.strftime('%-d/%m/%Y')} ({season})"
            return anchor, end, label
    # absolute fallback: use very start of test period
    t0 = df_mod["timestamp"].min() + pd.Timedelta(days=7)
    if snap_to_monday:
        t0 = _snap_to_monday(t0)
    t1 = t0 + pd.Timedelta(days=n_days)
    return t0, t1, f"{t0.strftime('%-d/%m/%Y')} – {t1.strftime('%-d/%m/%Y')}"


def _plot_detail_zoom(
    model_key: str,
    df: pd.DataFrame,
    season_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    raw: Optional[pd.Series],
    out_dir: str,
    horizons: List[int],
    dpi: int,
    n_days: int,
    filename_suffix: str,
) -> None:
    color  = MODEL_COLORS.get(model_key, "#555")
    label  = MODEL_DISPLAY.get(model_key, model_key.upper())
    df_mod = df[df["model"] == model_key]

    # Snap 7-day windows to Monday so each week runs Mon – Sun
    t0, t1, win_label = _pick_detail_window(
        season_windows, df_mod, horizons, n_days,
        snap_to_monday=(n_days >= 7),
    )

    n_h  = len(horizons)
    fig, axes = plt.subplots(n_h, 1, figsize=(14, 4 * n_h + 1), sharex=False)
    if n_h == 1:
        axes = [axes]

    fig.suptitle(
        f"{label} — {n_days}-day Detail  |  {win_label}",
        fontsize=12, y=1.01,
    )

    for ax, h in zip(axes, horizons):
        sub = df_mod[df_mod["horizon"] == h].sort_values("timestamp")
        win = sub[(sub["timestamp"] >= t0) & (sub["timestamp"] <= t1)]

        if win.empty:
            ax.set_title(f"{HORIZON_LABELS[h]} — no predictions in window")
            continue

        ts_pred = win["timestamp"].values
        yt = win["actual"].values.astype(float)
        yp = win["predicted"].values.astype(float)

        # Continuous actual from raw series (full 15-min resolution)
        ts_act, val_act = _actual_in_range(raw, win, t0, t1)
        sparse = _pred_median_gap_hours(ts_pred) > 3.0

        ax.plot(ts_act, val_act, color=ACTUAL_COLOR, lw=1.2,
                label="Actual (15 min)", zorder=3)
        if sparse:
            _plot_sparse_needles(ax, ts_pred, yt, yp, color, f"{label} predicted")
        else:
            ax.fill_between(ts_pred, yt, yp,
                            color="#e53935", alpha=0.12, zorder=2, label="Error")
            ax.plot(ts_pred, yp, color=color, lw=1.4, alpha=0.88,
                    marker="o" if len(ts_pred) <= 48 else None,
                    markersize=3,
                    label=f"{label} predicted", zorder=4)

        # Vertical day separators
        for day in pd.date_range(t0.normalize() + pd.Timedelta(days=1), t1, freq="D"):
            ax.axvline(day, color="gray", lw=0.6, ls="--", alpha=0.35, zorder=1)

        win_rmse = _rmse(yt, yp)
        win_mae  = _mae(yt, yp)
        with np.errstate(divide="ignore", invalid="ignore"):
            win_mape = float(np.mean(np.abs((yt - yp) / yt))) * 100.0 if len(yt) else float("nan")
        ax.set_title(
            f"{HORIZON_LABELS[h]}    RMSE={win_rmse:.0f} MW    MAE={win_mae:.0f} MW    MAPE={win_mape:.2f}%",
            fontsize=10,
        )
        ax.set_ylabel("Load (MW)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # X-axis ticks scaled to window length
        if n_days <= 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-Hh"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%-d/%m"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_key}_zoom_{filename_suffix}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"    saved {path}")


def plot_zoom_1week(
    model_key: str,
    df: pd.DataFrame,
    season_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    raw: Optional[pd.Series],
    out_dir: str,
    horizons: List[int],
    dpi: int,
) -> None:
    _plot_detail_zoom(model_key, df, season_windows, raw, out_dir,
                      horizons, dpi, n_days=7, filename_suffix="1week")


def plot_zoom_1day(
    model_key: str,
    df: pd.DataFrame,
    season_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    raw: Optional[pd.Series],
    out_dir: str,
    horizons: List[int],
    dpi: int,
) -> None:
    _plot_detail_zoom(model_key, df, season_windows, raw, out_dir,
                      horizons, dpi, n_days=1, filename_suffix="1day")


# --- Plot 3 — Cross-model comparison ---

def plot_cross_model_comparison(
    df: pd.DataFrame,
    energy_df: Optional[pd.DataFrame],
    model_keys: List[str],
    horizons: List[int],
    out_dir: str,
    dpi: int,
) -> None:
    has_energy = energy_df is not None
    if not has_energy:
        print("  [warn] No energy file — skipping Row 3 (energy vs RMSE).")

    # Layout: Row 1 = scatter per horizon (cols), Row 2 = metric bars (MAE/RMSE/MAPE),
    # optional Row 3 = energy vs RMSE
    n_cols = len(horizons)
    n_rows = 3 if has_energy else 2
    fig = plt.figure(figsize=(16, 13 if has_energy else 10))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.45, wspace=0.30)
    axes = np.empty((n_rows, n_cols), dtype=object)
    for ci in range(n_cols):
        axes[0, ci] = fig.add_subplot(gs[0, ci])
    # Row 2: 3 panels (MAE, RMSE, MAPE) regardless of n_cols
    bar_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[1, 2 if n_cols >= 3 else 1])]
    if has_energy:
        # Row 3 spans all cols
        axes[2, 0] = None
    fig.suptitle("Cross-Model Comparison", fontsize=13)

    # Pre-compute metrics including MAPE
    metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    for mk in model_keys:
        metrics[mk] = {}
        for h in horizons:
            sub = df[(df["model"] == mk) & (df["horizon"] == h)]
            if sub.empty:
                metrics[mk][h] = {"RMSE": np.nan, "MAE": np.nan,
                                   "MAPE": np.nan, "R2": np.nan}
            else:
                yt = sub["actual"].values.astype(float)
                yp = sub["predicted"].values.astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mape = float(np.mean(np.abs((yt - yp) / yt))) * 100.0
                metrics[mk][h] = {
                    "RMSE": _rmse(yt, yp),
                    "MAE":  _mae(yt, yp),
                    "MAPE": mape,
                    "R2":   _r2(yt, yp),
                }

    act_all = df["actual"].dropna().values
    ax_lo = float(act_all.min()) * 0.95
    ax_hi = float(act_all.max()) * 1.05
    rng   = np.random.default_rng(42)

    # ── Row 1: Scatter ────────────────────────────────────────────────────────
    for ci, h in enumerate(horizons):
        ax = axes[0, ci]
        ax.set_title(f"Predicted vs Actual — {HORIZON_LABELS[h]}", fontsize=10)
        ax.set_xlabel("Actual (MW)", fontsize=9)
        ax.set_ylabel("Predicted (MW)", fontsize=9)
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)

        for mk in model_keys:
            sub = df[(df["model"] == mk) & (df["horizon"] == h)]
            if sub.empty:
                continue
            yt = sub["actual"].values.astype(float)
            yp = sub["predicted"].values.astype(float)
            n  = min(2000, len(yt))
            idx = rng.choice(len(yt), size=n, replace=False)
            r2v = metrics[mk][h]["R2"]
            lbl = f"{MODEL_DISPLAY.get(mk, mk)}  R²={r2v:.3f}"
            ax.scatter(yt[idx], yp[idx], s=6, alpha=0.35,
                       color=MODEL_COLORS.get(mk, "#555"),
                       label=lbl, rasterized=True, zorder=3)

        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", lw=1.2, label="y = x", zorder=5)
        # Move the legend OUTSIDE the plot area — model names were overlapping
        # the data cloud at upper-left. bbox_inches="tight" in savefig will
        # accommodate the extra width without manual figsize tuning.
        leg = ax.legend(fontsize=6.5, loc="upper left",
                        bbox_to_anchor=(1.015, 1.0), framealpha=0.95,
                        title="Model · R²", title_fontsize=7,
                        labelspacing=0.25, borderpad=0.4, handletextpad=0.4)
        # Color each legend label with the model's color for instant recognition.
        for txt, mk in zip(leg.get_texts(), list(model_keys) + ["__y_eq_x__"]):
            if mk == "__y_eq_x__":
                continue
            txt.set_color(MODEL_COLORS.get(mk, "#555"))
        ax.set_aspect("equal", adjustable="datalim")

    # ── Row 2: 3 panels (MAE, RMSE, MAPE) — bars colored by model,
    #          horizons distinguished by hatch pattern ────────────────────────
    bar_x = np.arange(len(model_keys))
    n_h   = len(horizons)
    bar_w = 0.8 / max(1, n_h)
    # Horizon → hatch pattern (visually distinct, monochrome)
    HATCHES = ["", "///", "xxx", "...", "\\\\\\", "++", "--"]
    h_hatch = {h: HATCHES[i % len(HATCHES)] for i, h in enumerate(horizons)}

    from matplotlib.patches import Patch as _Patch

    metric_specs = [
        ("MAE",  "MAE (MW)"),
        ("RMSE", "RMSE (MW)"),
        ("MAPE", "MAPE (%)"),
    ]
    for ax, (mk_key, ylab) in zip(bar_axes, metric_specs):
        ax.set_title(ylab, fontsize=11)
        ax.set_ylabel(ylab, fontsize=9)
        for hi, h in enumerate(horizons):
            vals = [metrics[mk][h][mk_key] for mk in model_keys]
            xs = bar_x + (hi - (n_h - 1) / 2.0) * bar_w
            for xi, v in enumerate(vals):
                if np.isnan(v):
                    continue
                ax.bar(xs[xi], v, width=bar_w,
                       color=MODEL_COLORS.get(model_keys[xi], "#555"),
                       hatch=h_hatch[h], edgecolor="white",
                       linewidth=0.5, alpha=0.92, zorder=3)
                ax.text(xs[xi], v, f"{v:.0f}" if mk_key != "MAPE" else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6)
        # Naive baseline as horizontal lines per horizon
        for hi, h in enumerate(horizons):
            base = metrics.get("naive", {}).get(h, {}).get(mk_key, np.nan)
            if not np.isnan(base):
                ax.axhline(base, color=MODEL_COLORS["naive"],
                           lw=0.8, ls="--", alpha=0.5, zorder=2)
        ax.set_xticks(bar_x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(mk, mk) for mk in model_keys],
            fontsize=8, rotation=25, ha="right",
        )
        # Colour each xtick label with its model's bar colour, bold for
        # readability against the white background. Lets the reader map
        # bar→name→legend without scanning back to the colour key.
        for tick_lbl, mk in zip(ax.get_xticklabels(), model_keys):
            tick_lbl.set_color(MODEL_COLORS.get(mk, "#555"))
            tick_lbl.set_fontweight("bold")
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.25)

    # Single legend for horizon → pattern, placed on the first bar panel
    legend_els = [
        _Patch(facecolor="lightgray", hatch=h_hatch[h], edgecolor="white",
               label=HORIZON_LABELS[h])
        for h in horizons
    ]
    bar_axes[0].legend(handles=legend_els, fontsize=7,
                       loc="upper left", framealpha=0.9,
                       title="horizon (pattern)", title_fontsize=7)

    # ── Row 3: Energy vs RMSE (single panel spanning all horizons) ────────────
    if has_energy:
        ax_e = fig.add_subplot(gs[2, :])  # full-width bottom row

        # Sum fit + eval energy per model (kWh → Wh)
        energy_wh: Dict[str, float] = {}
        for mk in model_keys:
            mask = energy_df["project_name"].str.lower().str.contains(mk, na=False)
            total_kwh = energy_df.loc[mask, "energy_consumed"].sum()
            energy_wh[mk] = total_kwh * 1000.0

        ax_e.set_title("Total Energy vs RMSE (one point per model × horizon)", fontsize=10)
        ax_e.set_xlabel("Total Model Energy (Wh, log scale)", fontsize=9)
        ax_e.set_ylabel("RMSE (MW)", fontsize=9)
        ax_e.set_xscale("log")

        for mk in model_keys:
            eng = energy_wh.get(mk, np.nan)
            if np.isnan(eng):
                continue
            c   = MODEL_COLORS.get(mk, "#555")
            lbl = MODEL_DISPLAY.get(mk, mk)
            rmse_pts = [metrics[mk][h]["RMSE"] for h in horizons]

            # Connect the 3 horizon points for this model with a thin line
            ax_e.plot(
                [eng] * len(horizons), rmse_pts,
                color=c, lw=0.8, alpha=0.5, zorder=2,
            )
            for h, rv in zip(horizons, rmse_pts):
                if not np.isnan(rv):
                    ax_e.scatter([eng], [rv], s=60, color=c, zorder=5)
                    ax_e.annotate(
                        f"{lbl}\nh={h}",
                        (eng, rv),
                        textcoords="offset points",
                        xytext=(6, 3),
                        fontsize=6.5,
                        color=c,
                    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "comparison_all_models.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"    saved {path}")


# --- Plot 4 — Full test period overview ---

def plot_full_period(
    model_key: str,
    df: pd.DataFrame,
    raw: Optional[pd.Series],
    out_dir: str,
    horizons: List[int],
    dpi: int,
) -> None:
    color  = MODEL_COLORS.get(model_key, "#555")
    label  = MODEL_DISPLAY.get(model_key, model_key.upper())
    df_mod = df[df["model"] == model_key]

    n_h  = len(horizons)
    fig, axes = plt.subplots(n_h, 1, figsize=(14, 12), sharex=False)
    if n_h == 1:
        axes = [axes]
    fig.suptitle(f"{label} — Full Test Period", fontsize=13, y=1.01)

    t_start = df["timestamp"].min()
    t_end   = df["timestamp"].max()

    for ax, h in zip(axes, horizons):
        sub = df_mod[df_mod["horizon"] == h].sort_values("timestamp")
        if sub.empty:
            ax.set_title(f"{HORIZON_LABELS[h]} — no data")
            continue

        ts = sub["timestamp"].values
        yt = sub["actual"].values.astype(float)
        yp = sub["predicted"].values.astype(float)

        # Actual: full 15-min resolution when available
        if raw is not None:
            clipped = raw.loc[t_start:t_end]
            ax.plot(clipped.index, clipped.values, color=ACTUAL_COLOR,
                    lw=0.4, alpha=0.8, label="Actual (15 min)", zorder=2)
        else:
            ax.plot(ts, yt, color=ACTUAL_COLOR, lw=0.4, label="Actual (daily)", zorder=2)

        ax.plot(ts, yp, color=color, lw=0.6, alpha=0.8, label=label, zorder=3)

        full_rmse = _rmse(yt, yp)
        full_mae  = _mae(yt, yp)
        ax.set_title(
            f"{HORIZON_LABELS[h]}  |  Full-period RMSE={full_rmse:.0f} MW   MAE={full_mae:.0f} MW",
            fontsize=10,
        )
        ax.set_ylabel("Load (MW)", fontsize=9)

        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        _ticks_monthly(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_key}_full_period.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"    saved {path}")


# --- Summary table ---

def print_summary(df: pd.DataFrame, model_keys: List[str], horizons: List[int]) -> None:
    cw = 10
    sep = "=" * (18 + cw * 2 * len(horizons) + 2)

    # Header
    hdr = f"{'Model':<18}"
    for h in horizons:
        hdr += f"{'h='+str(h)+' RMSE':>{cw}}"
    for h in horizons:
        hdr += f"{'h='+str(h)+' MAE':>{cw}}"

    print(f"\n{sep}")
    print("  BENCHMARK SUMMARY  —  RMSE and MAE (MW)")
    print(sep)
    print(hdr)
    print("-" * len(sep))

    for mk in model_keys:
        row = f"{MODEL_DISPLAY.get(mk, mk):<18}"
        for h in horizons:
            sub = df[(df["model"] == mk) & (df["horizon"] == h)]
            if sub.empty:
                row += f"{'N/A':>{cw}}"
            else:
                row += f"{_rmse(sub['actual'].values, sub['predicted'].values):>{cw}.1f}"
        for h in horizons:
            sub = df[(df["model"] == mk) & (df["horizon"] == h)]
            if sub.empty:
                row += f"{'N/A':>{cw}}"
            else:
                row += f"{_mae(sub['actual'].values, sub['predicted'].values):>{cw}.1f}"
        print(row)

    print(sep)


# --- Auto-detect helpers ---

def _auto_predictions() -> str:
    """Find the most recent predictions.parquet under results/."""
    candidates = sorted(
        list(Path("results").glob("classical/wf_*/predictions.parquet"))
        + list(Path("results").glob("foundation_models/tsfm_*/predictions.parquet"))
        # legacy: flat layout
        + list(Path("results").glob("wf_*/predictions.parquet"))
        + list(Path("results").glob("tsfm_*/predictions.parquet"))
    )
    if candidates:
        return str(candidates[-1])
    candidates = sorted(
        list(Path("results").glob("classical/wf_*/*.parquet"))
        + list(Path("results").glob("foundation_models/tsfm_*/*.parquet"))
        + list(Path("results").glob("wf_*/*.parquet"))
        + list(Path("results").glob("tsfm_*/*.parquet"))
    )
    if candidates:
        return str(candidates[-1])
    return "results/predictions.parquet"


def _auto_energy(pred_path: str) -> Optional[str]:
    c = Path(pred_path).parent / "codecarbon" / "emissions.csv"
    return str(c) if c.exists() else None


# --- CLI + Main ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publication-quality diagnostic plots for walk-forward benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predictions", default=None,
                   help="Path to predictions (.parquet or .csv). "
                        "Defaults to the most recent wf_*/predictions.parquet.")
    p.add_argument("--energy", default=None,
                   help="Path to codecarbon emissions.csv. "
                        "Auto-detected from predictions folder if not given.")
    p.add_argument("--data", default="data/processed/elia_load_15min.csv",
                   help="Raw 15-min series for continuous actual line.")
    p.add_argument("--out", default="figures",
                   help="Base output directory. A timestamped sub-folder is created each run.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 96])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    pred_path   = args.predictions or _auto_predictions()
    energy_path = args.energy      or _auto_energy(pred_path)

    # ── Create a new output folder for this run ───────────────────────────────
    run_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_stem = Path(pred_path).parent.name   # e.g. "wf_20260331_152753"
    out_dir   = os.path.join(args.out, f"{pred_stem}_vis_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading predictions: {pred_path}")
    df = load_predictions(pred_path)

    model_keys = [mk for mk in MODEL_ORDER if mk in df["model"].unique()]
    for mk in sorted(df["model"].unique()):
        if mk not in model_keys:
            model_keys.append(mk)

    horizons = sorted(h for h in args.horizons if h in df["horizon"].unique())
    print(f"  Models   : {model_keys}")
    print(f"  Horizons : {horizons}")
    print(f"  Period   : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Rows     : {len(df):,}  ({len(df) // len(model_keys) // len(horizons)} origins/model/horizon)")

    raw = load_raw_series(args.data)
    if raw is not None:
        print(f"  Raw series: {len(raw):,} obs at 15-min resolution")
    else:
        print("  [warn] Raw series not found — using daily actuals from predictions file.")

    energy_df = load_energy(energy_path)
    if energy_df is not None:
        print(f"  Energy: {energy_path}  ({len(energy_df)} rows)")
    else:
        print("  [warn] Energy file not found.")

    # ── Season windows ────────────────────────────────────────────────────────
    season_windows = get_season_windows(df)
    for name, (t0, t1) in season_windows.items():
        print(f"  {name.capitalize()} window : {t0.date()} → {t1.date()}")

    # ══════════════════════════════════════════════════════════════════════════
    # Generate plots
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\nGenerating Plot 1: seasonal panels  ({len(model_keys)} models) ...")
    for mk in model_keys:
        print(f"  [{MODEL_DISPLAY.get(mk, mk)}]")
        plot_seasonal_panels(mk, df, season_windows, raw, out_dir, horizons, args.dpi)

    print(f"\nGenerating Plot 2: 7-day zoom  ({len(model_keys)} models × 2 seasons) ...")
    for mk in model_keys:
        print(f"  [{MODEL_DISPLAY.get(mk, mk)}]")
        plot_zoom(mk, df, season_windows, raw, out_dir, horizons, args.dpi)

    print(f"\nGenerating Plot 2b: 1-week detail zoom  ({len(model_keys)} models) ...")
    for mk in model_keys:
        print(f"  [{MODEL_DISPLAY.get(mk, mk)}]")
        plot_zoom_1week(mk, df, season_windows, raw, out_dir, horizons, args.dpi)

    print(f"\nGenerating Plot 2c: 1-day detail zoom  ({len(model_keys)} models) ...")
    for mk in model_keys:
        print(f"  [{MODEL_DISPLAY.get(mk, mk)}]")
        plot_zoom_1day(mk, df, season_windows, raw, out_dir, horizons, args.dpi)

    print("\nGenerating Plot 3: cross-model comparison ...")
    plot_cross_model_comparison(df, energy_df, model_keys, horizons, out_dir, args.dpi)

    print(f"\nGenerating Plot 4: full test period  ({len(model_keys)} models) ...")
    for mk in model_keys:
        print(f"  [{MODEL_DISPLAY.get(mk, mk)}]")
        plot_full_period(mk, df, raw, out_dir, horizons, args.dpi)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(df, model_keys, horizons)

    n_figs = (
        len(model_keys)          # Plot 1: seasonal
        + len(model_keys) * 2    # Plot 2: zoom winter + summer
        + len(model_keys)        # Plot 2b: 1-week detail
        + len(model_keys)        # Plot 2c: 1-day detail
        + 1                      # Plot 3: comparison
        + len(model_keys)        # Plot 4: full period
    )
    print(f"\nDone. {n_figs} figures saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
