"""plot_thesis.py — All thesis-specific figures in one place.

Subcommands:
  bars          Five thesis bar figures (accuracy + energy by family)
  variants      Curated thesis-ready accuracy variants (stacked + scatter)
  carbon        Five carbon-accounting figures
  arima_sarima  ARIMA vs SARIMA comparison (accuracy + energy + duration)
  arima_lgbm    ARIMA vs LightGBM weekly zoom at h=96
  pareto        Three-horizon accuracy–energy Pareto panels
  pareto_h96    Single-horizon Pareto at h=96
  backends      Eval-phase energy by measurement backend
  params        Foundation-model size vs h=96 accuracy
  unified       Chapter 6 definitive figures + verification report

Usage:
    python plot_thesis.py bars
    python plot_thesis.py unified
    python plot_thesis.py pareto --metrics results/combined/combined_all_models_final/metrics.csv
    python plot_thesis.py arima_sarima --results_dir results
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.palette import (
    MODEL_COLOR, MODEL_LABEL, MODEL_ORDER,
    HORIZON_HATCH, HORIZON_LABEL,
    TSFM_KEYS, BACKEND_HATCH, ANNOT_COLOR,
)

warnings.filterwarnings("ignore")

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

DPI = 150

# --- Shared helpers ---

def _color(m: str) -> str:
    return MODEL_COLOR.get(m, "#607D8B")

def _label(m: str) -> str:
    return MODEL_LABEL.get(m, m.upper())

def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def _mae(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs(yt - yp)))

def _mape(yt: np.ndarray, yp: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(np.abs((yt - yp) / yt))) * 100.0

def _savefig(fig: plt.Figure, out_dir: str, name: str, dpi: int = DPI) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

def _out_dir(name: str) -> str:
    d = os.path.join("figures", name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


# --- bars ---

_BARS_CLASSICAL: List[str] = ["naive", "arima", "linear", "lgbm"]
_BARS_FM_BY_FAMILY: List[List[str]] = [
    ["chronos_mini", "chronos_large"],
    ["chronos_bolt_mini", "chronos_bolt_base"],
    ["timesfm_200m", "timesfm_500m"],
    ["moirai_small", "moirai_base", "moirai_large"],
    ["moirai2_small"],
    ["lag_llama"],
]
_BARS_FM_FLAT: List[str] = [m for fam in _BARS_FM_BY_FAMILY for m in fam]
_BARS_HORIZONS: List[int] = [1, 4, 96]


def _bars_energy_total(df: pd.DataFrame, model: str, backend: str) -> float:
    fit_col  = f"fit_{backend}_energy_kwh"
    eval_col = f"eval_{backend}_energy_kwh"
    sub = df[df["model"] == model]
    if sub.empty:
        return np.nan
    f = float(sub[fit_col].fillna(0).iloc[0]) if fit_col in df.columns else 0.0
    e = float(sub[eval_col].fillna(0).iloc[0]) if eval_col in df.columns else 0.0
    tot = f + e
    return tot if tot > 0 else np.nan


def _bars_energy_mean(df: pd.DataFrame, model: str) -> float:
    vals = [_bars_energy_total(df, model, b) for b in ("cc", "ct")]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.mean(vals)) if vals else np.nan


def _bars_sort_metric(df, models, metric, horizon=96):
    def _v(m):
        row = df[(df["model"] == m) & (df["horizon_steps"] == horizon)]
        return float(row[metric].iloc[0]) if not row.empty else float("inf")
    return sorted(models, key=_v)


def _bars_sort_energy(df, models):
    def _v(m):
        v = _bars_energy_mean(df, m)
        return v if np.isfinite(v) and v > 0 else float("inf")
    return sorted(models, key=_v)


def _bars_draw_horizon_grouped(ax, df, models, metric, log_scale=False):
    n_h = len(_BARS_HORIZONS)
    bw  = 0.78 / n_h
    x   = np.arange(len(models))
    vmax = 0.0
    for i, h in enumerate(_BARS_HORIZONS):
        offset = (i - n_h / 2 + 0.5) * bw
        hatch  = HORIZON_HATCH.get(h, "")
        for j, m in enumerate(models):
            row = df[(df["model"] == m) & (df["horizon_steps"] == h)]
            if row.empty:
                continue
            val = float(row[metric].iloc[0])
            if not np.isfinite(val) or val <= 0:
                continue
            ax.bar(x[j] + offset, val, bw * 0.92,
                   color=_color(m), hatch=hatch,
                   alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
            vmax = max(vmax, val)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right")
    legend_els = [Patch(facecolor="lightgray", hatch=HORIZON_HATCH.get(h, ""),
                        edgecolor="white", alpha=0.85,
                        label=HORIZON_LABEL.get(h, f"h={h}"))
                  for h in _BARS_HORIZONS]
    ax.legend(handles=legend_els, title="Horizon", framealpha=0.85,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))
    if log_scale:
        ax.set_yscale("log")
    return vmax


def _bars_separator(ax, between):
    for x in between:
        ax.axvline(x + 0.5, color="#9E9E9E", lw=0.7, ls=":", alpha=0.55, zorder=1)


def _bars_fig_classical_accuracy(df, out_dir):
    models = _bars_sort_metric(df, _BARS_CLASSICAL, "MAE", horizon=96)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    vmax = _bars_draw_horizon_grouped(ax, df, models, "MAE")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Classical models — MAE")
    ax.set_ylim(bottom=0, top=vmax * 1.10 if vmax > 0 else None)
    fig.tight_layout()
    _savefig(fig, out_dir, "classical_accuracy_bars.png")


def _bars_fig_classical_energy(df, out_dir):
    models = _bars_sort_energy(df, [m for m in _BARS_CLASSICAL if m != "sarima"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    backends = ["cc", "ct", "mean"]
    bw = 0.78 / len(backends)
    x = np.arange(len(models))
    vmax = 0.0
    bhatch = {"cc": BACKEND_HATCH["CC"], "ct": BACKEND_HATCH["CT"], "mean": ""}
    balpha = {"cc": 0.90, "ct": 0.65, "mean": 0.35}
    blabel = {"cc": "CodeCarbon", "ct": "CarbonTracker", "mean": "Mean"}
    for i, b in enumerate(backends):
        offset = (i - len(backends) / 2 + 0.5) * bw
        for j, m in enumerate(models):
            kwh = _bars_energy_mean(df, m) if b == "mean" else _bars_energy_total(df, m, b)
            if not np.isfinite(kwh) or kwh <= 0:
                continue
            wh = kwh * 1e3
            ax.bar(x[j] + offset, wh, bw * 0.92,
                   color=_color(m), hatch=bhatch[b], alpha=balpha[b],
                   edgecolor="white", linewidth=0.3, zorder=3)
            vmax = max(vmax, wh)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Energy (Wh) — log scale")
    ax.set_title("Classical models — energy per backend")
    ax.set_ylim(bottom=max(1e-3, vmax * 1e-5), top=vmax * 3 if vmax > 0 else None)
    legend_els = [Patch(facecolor="lightgray", hatch=bhatch[b], alpha=balpha[b],
                        edgecolor="#444", label=blabel[b]) for b in backends]
    ax.legend(handles=legend_els, title="Backend", framealpha=0.85,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    _savefig(fig, out_dir, "classical_energy_bars.png")


def _bars_fig_fm_accuracy(df, out_dir):
    fm_sorted        = _bars_sort_metric(df, _BARS_FM_FLAT, "MAE", horizon=96)
    classical_sorted = _bars_sort_metric(df, _BARS_CLASSICAL, "MAE", horizon=96)
    models = fm_sorted + classical_sorted
    n_fm   = len(fm_sorted)
    n_h = len(_BARS_HORIZONS)
    bw  = 0.78 / n_h
    x   = np.arange(len(models))
    vmax = 0.0
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 0.78), 5.0))
    for i, h in enumerate(_BARS_HORIZONS):
        offset = (i - n_h / 2 + 0.5) * bw
        hatch  = HORIZON_HATCH.get(h, "")
        for j, m in enumerate(models):
            row = df[(df["model"] == m) & (df["horizon_steps"] == h)]
            if row.empty:
                continue
            val = float(row["MAE"].iloc[0])
            if not np.isfinite(val):
                continue
            ax.bar(x[j] + offset, val, bw * 0.92,
                   color=_color(m), hatch=hatch,
                   alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
            vmax = max(vmax, val)
    naive_mae = df[df["model"] == "naive"]["MAE"].min()
    if np.isfinite(naive_mae):
        ax.axhline(naive_mae, color=MODEL_COLOR["naive"], lw=1.2, ls="--", zorder=2,
                   label=f"Seasonal Naive ({naive_mae:.0f} MW)")
    _bars_separator(ax, [n_fm - 1])
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Foundation models (left) vs classical references (right) — MAE")
    ax.set_ylim(bottom=0, top=vmax * 1.10 if vmax > 0 else None)
    legend_els = [Patch(facecolor="lightgray", hatch=HORIZON_HATCH.get(h, ""),
                        edgecolor="white", alpha=0.85,
                        label=HORIZON_LABEL.get(h, f"h={h}")) for h in _BARS_HORIZONS]
    ax.legend(framealpha=0.85, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              handles=legend_els + ax.get_legend_handles_labels()[0], title="Horizon")
    fig.tight_layout()
    _savefig(fig, out_dir, "fm_accuracy_bars.png")


def _bars_fig_fm_energy(df, out_dir):
    classical_cmp = [m for m in _BARS_CLASSICAL if m != "sarima"]
    fm_sorted        = _bars_sort_energy(df, _BARS_FM_FLAT)
    classical_sorted = _bars_sort_energy(df, classical_cmp)
    models = fm_sorted + classical_sorted
    n_fm   = len(fm_sorted)
    x = np.arange(len(models))
    vmax = 0.0
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 0.78), 5.0))
    for j, m in enumerate(models):
        kwh = _bars_energy_mean(df, m)
        if not np.isfinite(kwh) or kwh <= 0:
            continue
        wh = kwh * 1e3
        ax.bar(x[j], wh, 0.72, color=_color(m), alpha=0.88,
               edgecolor="white", linewidth=0.3, zorder=3)
        vmax = max(vmax, wh)
    _bars_separator(ax, [n_fm - 1])
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Mean energy, mean(CC, CT) (Wh) — log scale")
    ax.set_title("Foundation models (left) vs classical references (right) — energy")
    ax.set_ylim(bottom=max(1e-3, vmax * 1e-6), top=vmax * 3 if vmax > 0 else None)
    fig.tight_layout()
    _savefig(fig, out_dir, "fm_energy_bars.png")


def _bars_fig_backend_disagreement(df, out_dir):
    def _cc_over_nv(m):
        nv = _bars_energy_total(df, m, "nv")
        cc = _bars_energy_total(df, m, "cc")
        if not (np.isfinite(nv) and nv > 0 and np.isfinite(cc)):
            return float("inf")
        return cc / nv
    models = sorted(_BARS_FM_FLAT, key=_cc_over_nv)
    x  = np.arange(len(models))
    bw = 0.36
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 0.78), 4.8))
    cc_ratios = []
    ct_ratios = []
    for m in models:
        nv = _bars_energy_total(df, m, "nv")
        cc = _bars_energy_total(df, m, "cc")
        ct = _bars_energy_total(df, m, "ct")
        cc_ratios.append(cc / nv if (np.isfinite(nv) and nv > 0 and np.isfinite(cc)) else np.nan)
        ct_ratios.append(ct / nv if (np.isfinite(nv) and nv > 0 and np.isfinite(ct)) else np.nan)
    for j, m in enumerate(models):
        c = _color(m)
        if np.isfinite(cc_ratios[j]):
            ax.bar(x[j] - bw / 2, cc_ratios[j], bw,
                   color=c, alpha=0.90, hatch=BACKEND_HATCH["CC"],
                   edgecolor="white", linewidth=0.3, zorder=3)
        if np.isfinite(ct_ratios[j]):
            ax.bar(x[j] + bw / 2, ct_ratios[j], bw,
                   color=c, alpha=0.65, hatch=BACKEND_HATCH["CT"],
                   edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(1.0, color="#424242", lw=1.0, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("Energy ratio vs nvidia-smi")
    ax.set_title("Backend disagreement — CC/NV (left) and CT/NV (right)")
    legend_els = [
        Patch(facecolor="lightgray", alpha=0.90, hatch=BACKEND_HATCH["CC"],
              edgecolor="#444", label="CC / NV"),
        Patch(facecolor="lightgray", alpha=0.65, hatch=BACKEND_HATCH["CT"],
              edgecolor="#444", label="CT / NV"),
    ]
    ax.legend(handles=legend_els, framealpha=0.85,
              loc="upper left", bbox_to_anchor=(1.005, 1.0), title="ratio")
    fig.tight_layout()
    _savefig(fig, out_dir, "backend_disagreement.png")


def cmd_bars(args: argparse.Namespace) -> None:
    metrics_path = args.metrics or "results/combined/20260513_all_models_stride2_elia/metrics.csv"
    df = pd.read_csv(metrics_path)
    df["model"] = df["model"].str.lower().str.strip()
    print(f"Loaded {metrics_path}: {df['model'].nunique()} models × {df['horizon_steps'].nunique()} horizons")
    out_dir = args.out_dir or _out_dir("thesis_bars")
    print(f"Output: {out_dir}\n")
    _bars_fig_classical_accuracy(df, out_dir)
    _bars_fig_classical_energy(df, out_dir)
    _bars_fig_fm_accuracy(df, out_dir)
    _bars_fig_fm_energy(df, out_dir)
    _bars_fig_backend_disagreement(df, out_dir)
    print(f"\nDone. {len(os.listdir(out_dir))} figures in {out_dir}/")


# --- variants ---

_VAR_DEFAULT_METRICS     = "results/combined/20260513_all_models_stride2_elia/metrics.csv"
_VAR_DEFAULT_PREDICTIONS = "results/combined/20260513_all_models_stride2_elia/predictions.parquet"
_VAR_CLASSICAL: List[str] = ["naive", "arima", "linear", "lgbm"]
_VAR_MAIN_FM:   List[str] = [
    "chronos_large", "chronos_bolt_base", "timesfm_500m",
    "moirai_large", "moirai2_small", "lag_llama",
]
_VAR_HORIZON_LABELS: Dict[int, str] = {
    1:  "h=1  (15 min)",
    4:  "h=4  (1 hour)",
    96: "h=96 (24 hours)",
}


def _var_plot_accuracy_stacked(df, out_dir, dpi, sort_horizon=96):
    df = df[df["model"] != "sarima"].copy()
    df = (df.sort_values("n_predictions", ascending=False)
            .drop_duplicates(subset=["model", "horizon_steps"])
            .sort_index())
    horizons = sorted(df["horizon_steps"].unique())
    all_models = df["model"].unique().tolist()
    n_h = len(horizons)
    bw  = 0.78 / n_h
    panels = [("MAE", "MAE (MW)"), ("MAPE_pct", "MAPE (%)")]
    n_m = len(all_models)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, n_m * 0.85), 9.5))
    for ax, (metric, ylabel) in zip(axes, panels):
        sort_sub = df[df["horizon_steps"] == sort_horizon].set_index("model")[metric]
        ordered  = sort_sub.sort_values(ascending=True).index.tolist()
        for m in all_models:
            if m not in ordered:
                ordered.append(m)
        x = np.arange(len(ordered))
        ax_max = 0.0
        for i, h in enumerate(horizons):
            offset = (i - n_h / 2 + 0.5) * bw
            hatch  = HORIZON_HATCH.get(h, "")
            for j, m in enumerate(ordered):
                val = df.loc[(df["model"] == m) & (df["horizon_steps"] == h), metric].mean()
                if np.isnan(val):
                    continue
                ax.bar(x[j] + offset, val, bw * 0.92,
                       color=_color(m), hatch=hatch,
                       alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
                ax_max = max(ax_max, val)
                pad = val * 0.02 + (0.5 if metric == "MAPE_pct" else 5)
                ax.text(x[j] + offset, val + pad, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7, rotation=90,
                        color=ANNOT_COLOR)
        naive_ref = df.loc[(df["model"] == "naive") & (df["horizon_steps"] == 96), metric].mean()
        if np.isfinite(naive_ref):
            unit = "%" if metric == "MAPE_pct" else " MW"
            ax.axhline(naive_ref, color=_color("naive"), lw=1.0, ls="--",
                       alpha=0.45, zorder=1,
                       label=f"Seasonal Naive h=96 ({naive_ref:.0f}{unit})")
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in ordered], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=11)
        ax.set_ylim(bottom=0, top=ax_max * 1.22 if ax_max > 0 else None)
        legend_els = [Patch(facecolor="gray", hatch=HORIZON_HATCH.get(h, ""),
                            edgecolor="white", alpha=0.85,
                            label=HORIZON_LABEL.get(h, f"h={h}")) for h in horizons]
        line_handles = ax.get_legend_handles_labels()[0]
        ax.legend(handles=legend_els + line_handles, title="Horizon", framealpha=0.85,
                  fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    _savefig(fig, out_dir, "01_accuracy_bar_stacked.png", dpi)


def _var_r2(yt, yp):
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float("nan") if ss_tot == 0 else float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)


def _var_plot_comparison_subset(preds, model_keys, horizons, out_dir, filename,
                                 title, dpi, stride=24):
    model_keys = [m for m in model_keys if m in preds["model"].unique()]
    if not model_keys:
        print(f"  [warn] none of requested models present — skipping {filename}")
        return
    horizons = [h for h in horizons if h in preds["horizon"].unique()]
    n_h = len(horizons)
    r2_table: Dict[str, Dict[int, float]] = {}
    for mk in model_keys:
        r2_table[mk] = {}
        for h in horizons:
            sub = preds[(preds["model"] == mk) & (preds["horizon"] == h)]
            if sub.empty:
                r2_table[mk][h] = float("nan")
            else:
                yt = sub["actual"].values.astype(float)
                yp = sub["predicted"].values.astype(float)
                r2_table[mk][h] = _var_r2(yt, yp)
    act_all = preds["actual"].dropna().values
    ax_lo = float(act_all.min()) * 0.95
    ax_hi = float(act_all.max()) * 1.05
    fig, axes = plt.subplots(n_h, 1, figsize=(9, 6.0 * n_h))
    if n_h == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, y=1.00)
    for ax, h in zip(axes, horizons):
        ax.set_title(f"Predicted vs Actual — {_VAR_HORIZON_LABELS.get(h, f'h={h}')}", fontsize=11)
        ax.set_xlabel("Actual (MW)", fontsize=9)
        ax.set_ylabel("Predicted (MW)", fontsize=9)
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)
        for mk in model_keys:
            sub = (preds[(preds["model"] == mk) & (preds["horizon"] == h)]
                   .sort_values("timestamp"))
            if sub.empty:
                continue
            sub = sub.iloc[::stride]
            yt = sub["actual"].values.astype(float)
            yp = sub["predicted"].values.astype(float)
            r2v = r2_table[mk][h]
            ax.scatter(yt, yp, s=10, alpha=0.50, color=_color(mk),
                       label=f"{_label(mk)}  R²={r2v:.3f}",
                       rasterized=True, zorder=3)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", lw=1.2, label="y = x", zorder=5)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.015, 1.0),
                  framealpha=0.95, title="Model · R²", title_fontsize=8,
                  labelspacing=0.35, borderpad=0.5, handletextpad=0.5)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _savefig(fig, out_dir, filename, dpi)


def cmd_variants(args: argparse.Namespace) -> None:
    metrics_path = args.metrics or _VAR_DEFAULT_METRICS
    pred_path    = args.predictions or _VAR_DEFAULT_PREDICTIONS
    out_dir = args.out_dir or _out_dir("thesis_variants")
    print(f"\nOutput folder: {out_dir}")
    df = pd.read_csv(metrics_path)
    df["model"] = df["model"].str.lower().str.strip()
    preds = pd.read_parquet(pred_path)
    preds["model"] = preds["model"].str.lower().str.strip()
    preds["horizon"] = preds["horizon"].astype(int)
    horizons = sorted(preds["horizon"].unique())
    print("\nGenerating variants …")
    _var_plot_accuracy_stacked(df, out_dir, args.dpi, sort_horizon=args.sort_horizon)
    for model_keys, fname, title in [
        (_VAR_CLASSICAL,
         "comparison_classical_only.png",
         "Cross-Model Comparison — Classical models only"),
        (_VAR_MAIN_FM,
         "comparison_main_fm_only.png",
         "Cross-Model Comparison — Main foundation-model variants"),
        (_VAR_CLASSICAL + _VAR_MAIN_FM,
         "comparison_classical_and_main_fm.png",
         "Cross-Model Comparison — Classical + main foundation models"),
    ]:
        _var_plot_comparison_subset(preds, model_keys, horizons, out_dir,
                                     fname, title, args.dpi, args.scatter_stride)
    print(f"\nDone. {len(os.listdir(out_dir))} figures in {out_dir}/")


# --- carbon ---

_CARBON_CLASSICAL = ["naive", "arima", "linear", "lgbm"]
_CARBON_FM_FLAT   = [
    "chronos_mini", "chronos_large",
    "chronos_bolt_mini", "chronos_bolt_base",
    "timesfm_200m", "timesfm_500m",
    "moirai_small", "moirai_base", "moirai_large",
    "moirai2_small", "lag_llama",
]
_INTENSITY_ANNUAL = 138.0
_INTENSITY_OFF    = 60.0
_INTENSITY_PEAK   = 280.0
_PUE_DEFAULT = 1.56
# Elia open data (ods192), consumption-based CO₂ intensity, Wed 28 May 2026 (CEST)
_HOURLY_INTENSITY = np.array([
    231.1, 228.2, 225.4, 218.8, 210.6, 206.3,
    187.3, 172.6, 127.4, 105.3,  94.8,  89.2,
     83.5,  79.9,  80.1,  78.7,  85.8, 101.3,
    144.9, 195.9, 199.2, 190.4, 196.3, 219.8,
], dtype=float)
_PUE_REFS = [
    (1.09, "Google fleet 2024"),
    (1.20, "Best-in-class new build"),
    (1.56, "Uptime Inst. 2024 avg"),
    (1.65, "Older European DC"),
    (1.80, "Older retrofit (IEA)"),
]


def _carbon_mean_eval_kwh(df, model):
    sub = df[df["model"] == model]
    if sub.empty:
        return np.nan
    cc = float(sub["eval_cc_energy_kwh"].iloc[0]) if pd.notna(sub["eval_cc_energy_kwh"].iloc[0]) else np.nan
    ct = float(sub["eval_ct_energy_kwh"].iloc[0]) if pd.notna(sub["eval_ct_energy_kwh"].iloc[0]) else np.nan
    vals = [v for v in [cc, ct] if np.isfinite(v) and v > 0]
    return float(np.mean(vals)) if vals else np.nan


def _carbon_n_pred(df, model):
    sub = df[df["model"] == model]
    return int(sub["n_predictions"].iloc[0]) if not sub.empty else 1


def _carbon_sort(models, key_fn):
    def _v(m):
        v = key_fn(m)
        return v if (np.isfinite(v) and v > 0) else float("inf")
    return sorted(models, key=_v)


def _carbon_fig1(df, out_dir):
    models = _carbon_sort(_CARBON_CLASSICAL,
                           lambda m: _carbon_mean_eval_kwh(df, m) / _carbon_n_pred(df, m))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(models))
    for j, m in enumerate(models):
        kwh = _carbon_mean_eval_kwh(df, m)
        n   = _carbon_n_pred(df, m)
        if not np.isfinite(kwh):
            continue
        nwh = kwh / n * 1e9
        ax.bar(x[j], nwh, 0.65, color=_color(m), alpha=0.88,
               edgecolor="white", linewidth=0.3, zorder=3)
        ax.text(x[j], nwh * 1.08, f"{nwh:.1f}",
                ha="center", va="bottom", fontsize=8, color="#424242")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Energy per origin (nWh) — log scale")
    ax.set_title("Classical models — eval energy per origin, mean(CC, CT)")
    fig.tight_layout()
    _savefig(fig, out_dir, "01_classical_per_origin_energy.png")


def _carbon_fig2(df, out_dir):
    models = _carbon_sort(_CARBON_CLASSICAL, lambda m: _carbon_mean_eval_kwh(df, m))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(models))
    for j, m in enumerate(models):
        kwh = _carbon_mean_eval_kwh(df, m)
        if not np.isfinite(kwh):
            continue
        co2_mg = kwh * _PUE_DEFAULT * _INTENSITY_ANNUAL * 1e3
        ax.bar(x[j], co2_mg, 0.65, color=_color(m), alpha=0.88,
               edgecolor="white", linewidth=0.3, zorder=3)
        lbl = f"{co2_mg / 1000:.2f} g" if co2_mg >= 1000 else f"{co2_mg:.1f} mg"
        ax.text(x[j], co2_mg * 1.08, lbl,
                ha="center", va="bottom", fontsize=8, color="#424242")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("CO₂eq (mg) — log scale")
    ax.set_title(f"Classical models — eval CO₂eq\n(138 g/kWh annual avg, PUE {_PUE_DEFAULT})")
    fig.tight_layout()
    _savefig(fig, out_dir, "02_classical_eval_co2.png")


def _carbon_fig3(df, out_dir, all_models=False):
    pool = (_CARBON_FM_FLAT + _CARBON_CLASSICAL) if all_models else _CARBON_FM_FLAT
    models = _carbon_sort(pool, lambda m: _carbon_mean_eval_kwh(df, m))
    scenarios = [
        ("Off-peak (60 g/kWh)",    _INTENSITY_OFF,    "",    0.90),
        ("Annual avg (138 g/kWh)", _INTENSITY_ANNUAL, "///", 0.85),
        ("Peak (280 g/kWh)",       _INTENSITY_PEAK,   "xxx", 0.80),
    ]
    n_s = len(scenarios)
    bw  = 0.75 / n_s
    x   = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 0.9), 5.2))
    for si, (slabel, intensity, hatch, alpha) in enumerate(scenarios):
        offset = (si - n_s / 2 + 0.5) * bw
        for j, m in enumerate(models):
            kwh = _carbon_mean_eval_kwh(df, m)
            if not np.isfinite(kwh):
                continue
            co2_g = kwh * _PUE_DEFAULT * intensity
            ax.bar(x[j] + offset, co2_g, bw * 0.92, color=_color(m),
                   hatch=hatch, alpha=alpha, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("CO₂eq per eval run (g) — log scale")
    tag = "All" if all_models else "Foundation"
    ax.set_title(f"{tag} models — eval CO₂eq under three intensity scenarios (PUE {_PUE_DEFAULT})")
    legend_els = [Patch(facecolor="lightgray", hatch=h, alpha=a,
                        edgecolor="#444", label=l) for l, _, h, a in scenarios]
    ax.legend(handles=legend_els, framealpha=0.85,
              loc="upper left", bbox_to_anchor=(1.005, 1.0), title="Intensity")
    fig.tight_layout()
    fname = "03b_all_eval_co2_scenarios.png" if all_models else "03_fm_eval_co2_scenarios.png"
    _savefig(fig, out_dir, fname)


def _carbon_fig4(df, out_dir):
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    for m in ["timesfm_500m", "chronos_mini"]:
        kwh = _carbon_mean_eval_kwh(df, m)
        if not np.isfinite(kwh):
            continue
        co2_g = kwh * _PUE_DEFAULT * _HOURLY_INTENSITY
        ax.plot(hours, co2_g, marker="o", markersize=5, lw=2.0,
                color=_color(m), label=_label(m), zorder=3)
        ax.fill_between(hours, 0, co2_g, color=_color(m), alpha=0.08)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}h" for h in hours], fontsize=7)
    ax.set_xlabel("Dispatch hour")
    ax.set_ylabel("CO₂eq per eval run (g)")
    ax.set_title("CO₂eq by dispatch hour: Belgium — Wed 28 May 2026")
    ax.legend(framealpha=0.85, loc="center right")
    ax2 = ax.twinx()
    ax2.plot(hours, _HOURLY_INTENSITY, ls="--", lw=1.0, color="#9E9E9E", alpha=0.6, zorder=1)
    ax2.axhline(_INTENSITY_OFF,  ls=":", lw=0.9, color="#4CAF50", alpha=0.7)
    ax2.axhline(_INTENSITY_PEAK, ls=":", lw=0.9, color="#C62828", alpha=0.7)
    ax2.text(23.3, _INTENSITY_OFF  + 4, f"{_INTENSITY_OFF:.0f}",  fontsize=7, color="#4CAF50")
    ax2.text(23.3, _INTENSITY_PEAK + 4, f"{_INTENSITY_PEAK:.0f}", fontsize=7, color="#C62828")
    ax2.set_ylabel("Grid intensity (gCO₂/kWh)", color="#9E9E9E", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#9E9E9E", labelsize=7)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#BDBDBD")
    fig.tight_layout()
    _savefig(fig, out_dir, "04_dispatch_hour_co2.png")


def _carbon_fig5(df, out_dir):
    model = "chronos_bolt_mini"
    kwh   = _carbon_mean_eval_kwh(df, model)
    pues   = [p for p, _ in _PUE_REFS]
    labels = [f"PUE {p:.2f}\n{l}" for p, l in _PUE_REFS]
    co2_g  = [kwh * p * _INTENSITY_ANNUAL for p in pues]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    x    = np.arange(len(pues))
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=min(pues) - 0.1, vmax=max(pues) + 0.1)
    for j, (pue, g) in enumerate(zip(pues, co2_g)):
        c = cmap(norm(pue))
        ax.bar(x[j], g, 0.65, color=c, alpha=0.88,
               edgecolor="white", linewidth=0.3, zorder=3)
        ax.text(x[j], g + max(co2_g) * 0.02, f"{g:.2f} g",
                ha="center", va="bottom", fontsize=8, color="#424242")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.set_ylabel("CO₂eq per eval run (g)")
    ax.set_title(f"{_label(model)} — CO₂eq at 138 g/kWh under five PUE values")
    ax.set_ylim(bottom=0, top=max(co2_g) * 1.15)
    fig.tight_layout()
    _savefig(fig, out_dir, "05_pue_sensitivity.png")


def cmd_carbon(args: argparse.Namespace) -> None:
    metrics_path = args.metrics or "results/combined/combined_all_models_final/metrics.csv"
    df = pd.read_csv(metrics_path)
    df["model"] = df["model"].str.lower().str.strip()
    out_dir = args.out_dir or _out_dir("carbon_accounting")
    print(f"Output: {out_dir}\n")
    _carbon_fig1(df, out_dir)
    _carbon_fig2(df, out_dir)
    _carbon_fig3(df, out_dir, all_models=False)
    _carbon_fig3(df, out_dir, all_models=True)
    _carbon_fig4(df, out_dir)
    _carbon_fig5(df, out_dir)
    print(f"\nDone. {len(os.listdir(out_dir))} figures in {out_dir}/")


# --- arima_sarima ---

def _as_load_run(results_dir: str, run_id: Optional[str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "wf_*/metrics.csv")))
    if not files:
        raise FileNotFoundError(f"No wf_*/metrics.csv found under {results_dir}")
    dfs  = [pd.read_csv(f).assign(_file=f) for f in files]
    full = pd.concat(dfs, ignore_index=True)
    full["model"] = full["model"].str.lower().str.strip()
    if run_id:
        sub = full[full["run_id"] == run_id]
        if sub.empty:
            raise ValueError(f"run_id '{run_id}' not found")
        return sub
    by_run     = full.groupby("run_id")["model"].unique()
    candidates = [rid for rid, mods in by_run.items()
                  if {"arima", "sarima"}.issubset(set(mods))]
    if not candidates:
        raise SystemExit("No run found containing both arima and sarima.")
    chosen = sorted(candidates)[-1]
    print(f"  Selected run: {chosen}")
    return full[full["run_id"] == chosen]


def _as_summary_row(df, model):
    sub = df[df["model"] == model]
    return sub[["fit_cc_energy_kwh", "eval_cc_energy_kwh",
                "fit_ct_energy_kwh", "eval_ct_energy_kwh",
                "fit_duration_s", "eval_duration_s"]].iloc[0]


def _as_plot(df, out_dir, dpi):
    horizons = sorted(df["horizon_steps"].unique())
    models   = ["arima", "sarima"]
    colors   = [MODEL_COLOR[m] for m in models]
    labels   = [MODEL_LABEL[m] for m in models]
    x = np.arange(len(horizons))
    w = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    for i, m in enumerate(models):
        vals = [df[(df["model"] == m) & (df["horizon_steps"] == h)]["MAE"].mean()
                for h in horizons]
        ax.bar(x + (i - 0.5) * w, vals, w, color=colors[i], label=labels[i],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABEL.get(h, f"h={h}") for h in horizons])
    ax.set_ylabel("MAE (MW)")
    ax.set_title("MAE")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, ls="--")

    ax = axes[0, 1]
    for i, m in enumerate(models):
        vals = [df[(df["model"] == m) & (df["horizon_steps"] == h)]["MAPE_pct"].mean()
                for h in horizons]
        ax.bar(x + (i - 0.5) * w, vals, w, color=colors[i], label=labels[i],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABEL.get(h, f"h={h}") for h in horizons])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE")
    ax.grid(axis="y", alpha=0.3, ls="--")

    bw = 0.38
    xm = np.arange(len(models))
    ax = axes[1, 0]
    for i, m in enumerate(models):
        r     = _as_summary_row(df, m)
        fit_e = (r["fit_cc_energy_kwh"] or 0) * 1e6
        ev_e  = (r["eval_cc_energy_kwh"] or 0) * 1e6
        ax.bar(xm[i] - bw / 2, fit_e, bw, color=colors[i],
               edgecolor="white", linewidth=0.4, label="fit" if i == 0 else None)
        ax.bar(xm[i] + bw / 2, ev_e, bw, color=colors[i], alpha=0.55,
               hatch="///", edgecolor="white", linewidth=0.4,
               label="eval" if i == 0 else None)
    ax.set_xticks(xm)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (μWh, CodeCarbon)")
    ax.set_title("Energy: fit vs eval")
    ax.set_yscale("log")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, ls="--", which="both")

    ax = axes[1, 1]
    for i, m in enumerate(models):
        r      = _as_summary_row(df, m)
        fit_s  = float(r["fit_duration_s"] or 0)
        eval_s = float(r["eval_duration_s"] or 0)
        ax.bar(xm[i] - bw / 2, fit_s, bw, color=colors[i],
               edgecolor="white", linewidth=0.4, label="fit" if i == 0 else None)
        ax.bar(xm[i] + bw / 2, eval_s, bw, color=colors[i], alpha=0.55,
               hatch="///", edgecolor="white", linewidth=0.4,
               label="eval" if i == 0 else None)
    ax.set_xticks(xm)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title("Duration: fit vs eval")
    ax.set_yscale("log")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, ls="--", which="both")

    fig.tight_layout()
    path = os.path.join(out_dir, "arima_vs_sarima.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def cmd_arima_sarima(args: argparse.Namespace) -> None:
    out_dir = args.out_dir or _out_dir("arima_vs_sarima")
    print(f"Output: {out_dir}")
    df = _as_load_run(args.results_dir, args.run_id)
    _as_plot(df, out_dir, args.dpi)


# --- arima_lgbm ---

def cmd_arima_lgbm(args: argparse.Namespace) -> None:
    from visualize_results import (
        ACTUAL_COLOR, HORIZON_LABELS as VR_HORIZON_LABELS,
        _actual_in_range, load_predictions, load_raw_series,
    )
    pred_path = args.predictions or "results/combined/combined_all_models_final/predictions.parquet"
    raw_path  = args.raw or "data/processed/elia_load_15min.csv"
    out_path  = args.out or "figures/arima_vs_lgbm_h96_1week.png"

    if not Path(pred_path).exists():
        raise SystemExit(f"predictions file not found: {pred_path}")

    df  = load_predictions(pred_path)
    raw = load_raw_series(raw_path)
    if raw is None:
        raise SystemExit(f"raw series not found: {raw_path}")

    WEEK_START = pd.Timestamp(args.week_start)
    WEEK_END   = pd.Timestamp(args.week_end)
    HORIZON    = args.horizon

    def _slice(model):
        return (df[(df["model"] == model) & (df["horizon"] == HORIZON)
                   & (df["timestamp"] >= WEEK_START)
                   & (df["timestamp"] <= WEEK_END)]
                .sort_values("timestamp").reset_index(drop=True))

    win_arima = _slice("arima")
    win_lgbm  = _slice("lgbm")
    if win_arima.empty or win_lgbm.empty:
        raise SystemExit("no predictions in window for arima or lgbm")

    ts_act, val_act = _actual_in_range(raw, win_arima, WEEK_START, WEEK_END)
    all_vals = np.concatenate([np.asarray(val_act, dtype=float),
                               win_arima["actual"].values.astype(float),
                               win_arima["predicted"].values.astype(float),
                               win_lgbm["predicted"].values.astype(float)])
    y_lo = float(np.nanmin(all_vals))
    y_hi = float(np.nanmax(all_vals))
    margin = (y_hi - y_lo) * 0.06
    ylim = (y_lo - margin, y_hi + margin)

    PANEL_TITLES = {
        "arima": f"ARIMA(2,1,2) — h={HORIZON}",
        "lgbm":  f"LightGBM — h={HORIZON}",
    }

    def _draw(ax, model_key, win, show_xlabels):
        color = MODEL_COLOR[model_key]
        ts_pred = win["timestamp"].values
        yt = win["actual"].values.astype(float)
        yp = win["predicted"].values.astype(float)
        ts_a, val_a = _actual_in_range(raw, win, WEEK_START, WEEK_END)
        ax.plot(ts_a, val_a, color=ACTUAL_COLOR, lw=1.2, label="Actual (15 min)", zorder=3)
        ax.fill_between(ts_pred, yt, yp, color="#e53935", alpha=0.12, zorder=2, label="Error")
        ax.plot(ts_pred, yp, color=color, lw=1.4, alpha=0.88,
                label=f"{MODEL_LABEL[model_key]} predicted", zorder=4)
        for day in pd.date_range(WEEK_START.normalize() + pd.Timedelta(days=1), WEEK_END, freq="D"):
            ax.axvline(day, color="gray", lw=0.6, ls="--", alpha=0.35, zorder=1)
        rmse_v = _rmse(yt, yp)
        mae_v  = _mae(yt, yp)
        mape_v = _mape(yt, yp)
        ax.set_title(f"{PANEL_TITLES[model_key]}    RMSE={rmse_v:.0f} MW    "
                     f"MAE={mae_v:.0f} MW    MAPE={mape_v:.2f}%", fontsize=10)
        ax.set_ylabel("Load (MW)", fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_xlim(WEEK_START, WEEK_END)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%-d/%m"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        if not show_xlabels:
            plt.setp(ax.get_xticklabels(), visible=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"ARIMA vs LightGBM, h={HORIZON} (24 hours ahead) — "
                 f"winter week {WEEK_START.date()} to {WEEK_END.date()}",
                 fontsize=12, y=1.01)
    _draw(axes[0], "arima", win_arima, show_xlabels=False)
    _draw(axes[1], "lgbm",  win_lgbm,  show_xlabels=True)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close("all")
    print(f"saved: {out_path}")


# --- pareto ---

_PARETO_HORIZONS  = [1, 4, 96]
_PARETO_FRONTIER  = ["naive", "lgbm", "chronos_bolt_mini", "chronos_bolt_base"]
_PARETO_COLOR     = "#D32F2F"
_PARETO_FR_LABEL  = {"naive": "Naive", "lgbm": "LightGBM",
                     "chronos_bolt_mini": "CB-mini", "chronos_bolt_base": "CB-base"}
_PARETO_FR_OFFSET = {"naive": (7, 4), "lgbm": (-3, -15),
                     "chronos_bolt_mini": (2, 9), "chronos_bolt_base": (9, -2)}
_PARETO_FR_HA     = {"naive": "left", "lgbm": "center",
                     "chronos_bolt_mini": "left", "chronos_bolt_base": "left"}


def _pareto_load(metrics_path):
    df = pd.read_csv(metrics_path)
    df = df[df["model"] != "sarima"].copy()
    n_full = df["n_predictions"].max()
    df = df[df["n_predictions"] == n_full]
    df["mean_kwh"] = df[["eval_cc_energy_kwh", "eval_ct_energy_kwh"]].mean(axis=1)
    return df


def cmd_pareto(args: argparse.Namespace) -> None:
    df     = _pareto_load(args.metrics)
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    out_dir = args.out_dir or _out_dir("pareto_panels")

    fig, axes = plt.subplots(1, len(_PARETO_HORIZONS), figsize=(16, 5.4),
                              sharex=True, sharey=False)
    for ax, h in zip(axes, _PARETO_HORIZONS):
        s  = df[df["horizon_steps"] == h].set_index("model")
        fr = s.loc[_PARETO_FRONTIER].sort_values("mean_kwh")
        ax.plot(fr["mean_kwh"].values, fr["MAE"].values, "-",
                color=_PARETO_COLOR, linewidth=2.6, zorder=2, alpha=0.9)
        for m in models:
            marker = "^" if m in TSFM_KEYS else "o"
            ax.scatter(s.loc[m, "mean_kwh"], s.loc[m, "MAE"],
                       color=MODEL_COLOR.get(m, "#607D8B"), s=120, marker=marker,
                       edgecolors="white", linewidths=0.9, zorder=5)
        for m in _PARETO_FRONTIER:
            ax.annotate(_PARETO_FR_LABEL[m], (s.loc[m, "mean_kwh"], s.loc[m, "MAE"]),
                        xytext=_PARETO_FR_OFFSET[m], textcoords="offset points",
                        fontsize=7.5, color="#222222", ha=_PARETO_FR_HA[m], zorder=6)
        ax.set_xscale("log")
        ax.set_xlabel("Mean eval energy (kWh) — log scale")
        ax.set_title(HORIZON_LABEL.get(h, f"h={h}") + f"  (h={h})")
    axes[0].set_ylabel("MAE (MW)")
    handles = [
        Line2D([0], [0], marker=("^" if m in TSFM_KEYS else "o"), linestyle="none",
               markerfacecolor=MODEL_COLOR.get(m, "#607D8B"),
               markeredgecolor="white", markersize=9, label=MODEL_LABEL.get(m, m))
        for m in models
    ] + [
        Line2D([0], [0], marker="None", linestyle="none", label="—"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="gray",
               markeredgecolor="white", markersize=8, label="Classical"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="gray",
               markeredgecolor="white", markersize=8, label="Foundation"),
        Line2D([0], [0], color=_PARETO_COLOR, linewidth=2.6,
               label="Multi-horizon Pareto frontier"),
    ]
    axes[-1].legend(handles=handles, framealpha=0.9, fontsize=8,
                    title="model", title_fontsize=8,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"pareto_panels.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# --- pareto_h96 ---

def cmd_pareto_h96(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.metrics)
    df = df[df["model"] != "sarima"].copy()
    n_full = df["n_predictions"].max()
    df = df[(df["n_predictions"] == n_full) & (df["horizon_steps"] == 96)]
    df["mean_kwh"] = df[["eval_cc_energy_kwh", "eval_ct_energy_kwh"]].mean(axis=1)
    s  = df.groupby("model")[["MAE", "mean_kwh"]].first()
    models = [m for m in MODEL_ORDER if m in s.index]
    out_dir = args.out_dir or _out_dir("pareto_h96")

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    fr = s.loc[_PARETO_FRONTIER].sort_values("mean_kwh")
    ax.plot(fr["mean_kwh"].values, fr["MAE"].values, "-",
            color=_PARETO_COLOR, linewidth=2.6, zorder=2, alpha=0.9)
    for m in models:
        marker = "^" if m in TSFM_KEYS else "o"
        ax.scatter(s.loc[m, "mean_kwh"], s.loc[m, "MAE"],
                   color=MODEL_COLOR.get(m, "#607D8B"), s=140, marker=marker,
                   edgecolors="white", linewidths=0.9, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Mean eval energy (kWh) — log scale")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Accuracy–energy Pareto at h=96 (24 h ahead)")
    model_handles = [
        Line2D([0], [0], marker=("^" if m in TSFM_KEYS else "o"), linestyle="none",
               markerfacecolor=MODEL_COLOR.get(m, "#607D8B"),
               markeredgecolor="white", markersize=9, label=MODEL_LABEL.get(m, m))
        for m in models
    ]
    shape_handles = [
        Line2D([0], [0], marker="None", linestyle="none", label="—"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="gray",
               markeredgecolor="white", markersize=8, label="Classical"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="gray",
               markeredgecolor="white", markersize=8, label="Foundation"),
        Line2D([0], [0], color=_PARETO_COLOR, linewidth=2.6,
               label="Multi-horizon Pareto frontier"),
    ]
    ax.legend(handles=model_handles + shape_handles, framealpha=0.9, fontsize=8,
              title="model", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"pareto_h96.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# --- backends ---

_BE_EVAL_COLS = {
    "eval_cc": "eval_cc_energy_kwh",
    "eval_ct": "eval_ct_energy_kwh",
    "eval_nv": "eval_nv_energy_kwh",
}
_BE_SERIES = [
    ("CodeCarbon",    "eval_cc",    BACKEND_HATCH["CC"], 0.85),
    ("CarbonTracker", "eval_ct",    BACKEND_HATCH["CT"], 0.70),
    ("Mean (CC, CT)", "mean_cc_ct", "",                  0.97),
    ("nvidia_smi",    "eval_nv",    BACKEND_HATCH["NV"], 0.55),
]


def cmd_backends(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.metrics)
    df = df[df["model"] != "sarima"].copy()
    n_full = df["n_predictions"].max()
    df = df[df["n_predictions"] == n_full]
    summary = df.groupby("model")[list(_BE_EVAL_COLS.values())].first()
    summary = summary.rename(columns={v: k for k, v in _BE_EVAL_COLS.items()})
    summary *= 1e3
    summary["mean_cc_ct"] = summary[["eval_cc", "eval_ct"]].mean(axis=1)
    summary = summary.sort_values("mean_cc_ct")
    models  = list(summary.index)
    out_dir = args.out_dir or _out_dir("eval_energy_backends")

    n_b = len(_BE_SERIES)
    x   = np.arange(len(models))
    w   = 0.8 / n_b
    fig, ax = plt.subplots(figsize=(max(13, len(models) * 1.1), 5.6))
    for bi, (name, col, hatch, alpha) in enumerate(_BE_SERIES):
        xs   = x + (bi - (n_b - 1) / 2.0) * w
        vals = summary[col].values
        for i, m in enumerate(models):
            v = vals[i]
            if not np.isfinite(v) or v <= 0:
                continue
            ax.bar(xs[i], v, w, color=MODEL_COLOR.get(m, "#607D8B"),
                   alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0.4, zorder=3)
    floor = float(summary.loc["lgbm", "mean_cc_ct"])
    ax.axhline(floor, linestyle=":", color="#333333", linewidth=1.4, zorder=2)
    ax.set_yscale("log")
    finite = summary[[c for _, c, _, _ in _BE_SERIES]].values
    finite = finite[np.isfinite(finite) & (finite > 0)]
    ax.set_ylim(bottom=finite.min() * 0.5, top=finite.max() * 3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m.upper()) for m in models],
                       rotation=30, ha="right")
    ax.set_ylabel("Eval energy (Wh) — log scale")
    ax.set_title("Eval-phase energy by model and measurement backend")
    legend_els = [Patch(facecolor="lightgray", alpha=alpha, hatch=hatch,
                        edgecolor="#444", label=name)
                  for name, _, hatch, alpha in _BE_SERIES]
    legend_els.append(Line2D([0], [0], linestyle=":", color="#333333", linewidth=1.4,
                              label=f"LightGBM eval ({floor:.2f} Wh)"))
    ax.legend(handles=legend_els, framealpha=0.9, fontsize=8,
              title="measurement backend", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"eval_energy_backends.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# --- params ---

_PARAMS_M = {
    "chronos_mini":      20,   "chronos_large":     710,
    "chronos_bolt_mini": 21,   "chronos_bolt_base": 205,
    "timesfm_200m":      200,  "timesfm_500m":      500,
    "moirai_small":      14,   "moirai_base":        91, "moirai_large":  311,
    "moirai2_small":     11,   "lag_llama":         2.45,
}
_QUANTILE = {"chronos_bolt_mini", "chronos_bolt_base",
             "timesfm_200m", "timesfm_500m", "moirai2_small"}


def cmd_params(args: argparse.Namespace) -> None:
    df  = pd.read_csv(args.metrics)
    df  = df[df["n_predictions"] == df["n_predictions"].max()]
    mae = df[df["horizon_steps"] == 96].set_index("model")["MAE"]
    models  = [m for m in MODEL_ORDER if m in TSFM_KEYS and m in _PARAMS_M]
    out_dir = args.out_dir or _out_dir("params_vs_mae")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for m in models:
        c = MODEL_COLOR.get(m, "#607D8B")
        x, y = _PARAMS_M[m], mae[m]
        if m in _QUANTILE:
            ax.scatter(x, y, s=130, marker="D", facecolors="none",
                       edgecolors=c, linewidths=2.0, zorder=5)
        else:
            ax.scatter(x, y, s=130, marker="o", color=c,
                       edgecolors="white", linewidths=0.9, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (millions) — log scale")
    ax.set_ylabel("MAE at h=96 (MW)")
    ax.set_title("Foundation-model size vs accuracy (h=96)")
    model_handles = [
        Line2D([0], [0], linestyle="none",
               marker=("D" if m in _QUANTILE else "o"),
               markerfacecolor=("none" if m in _QUANTILE else MODEL_COLOR.get(m, "#607D8B")),
               markeredgecolor=(MODEL_COLOR.get(m, "#607D8B") if m in _QUANTILE else "white"),
               markeredgewidth=(1.8 if m in _QUANTILE else 0.8),
               markersize=9, label=MODEL_LABEL.get(m, m))
        for m in models
    ]
    shape_handles = [
        Line2D([0], [0], marker="None", linestyle="none", label="—"),
        Line2D([0], [0], linestyle="none", marker="o", markerfacecolor="gray",
               markeredgecolor="white", markersize=9, label="Sampled trajectory"),
        Line2D([0], [0], linestyle="none", marker="D", markerfacecolor="none",
               markeredgecolor="gray", markeredgewidth=1.8, markersize=9,
               label="Deterministic quantile"),
    ]
    ax.legend(handles=model_handles + shape_handles, framealpha=0.9, fontsize=8,
              title="model / output type", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"params_vs_mae.{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# --- unified ---

def cmd_unified(args: argparse.Namespace) -> None:
    import shutil as _shutil
    import subprocess as _subprocess
    import sys as _sys
    from src.data import load_timeseries, clean_timeseries, split_last_n_years

    ROOT         = os.path.abspath(".")
    MAIN_METRICS = (os.path.abspath(args.metrics) if getattr(args, "metrics", None)
                    else os.path.join(ROOT, "results", "combined_all", "metrics.csv"))
    OUT_DIR      = (os.path.abspath(args.out_dir) if getattr(args, "out_dir", None)
                    else os.path.join(ROOT, "figures", "chapter6_unified"))
    # 300 dpi for print; each figure is also written as a sibling .pdf.
    _DPI         = getattr(args, "dpi", None) or 300

    MODELS_UF: Dict[str, dict] = {
        "naive":             dict(disp="Seasonal Naive",    cls="classical", fam="Naive",     out=None,       params=None),
        "arima":             dict(disp="ARIMA",             cls="classical", fam="ARIMA",     out=None,       params=None),
        "linear":            dict(disp="Ridge",             cls="classical", fam="Ridge",     out=None,       params=None),
        "lgbm":              dict(disp="LightGBM",          cls="classical", fam="LightGBM",  out=None,       params=None),
        "chronos_mini":      dict(disp="Chronos-T5 mini",  cls="fm",        fam="Chronos",   out="sampled",  params=20e6),
        "chronos_large":     dict(disp="Chronos-T5 large", cls="fm",        fam="Chronos",   out="sampled",  params=710e6),
        "chronos_bolt_mini": dict(disp="Chronos-Bolt mini",cls="fm",        fam="Chronos",   out="quantile", params=20e6),
        "chronos_bolt_base": dict(disp="Chronos-Bolt base",cls="fm",        fam="Chronos",   out="quantile", params=205e6),
        "timesfm_200m":      dict(disp="TimesFM 200M",     cls="fm",        fam="TimesFM",   out="quantile", params=200e6),
        "timesfm_500m":      dict(disp="TimesFM 500M",     cls="fm",        fam="TimesFM",   out="quantile", params=500e6),
        "moirai_small":      dict(disp="Moirai-1.0 small", cls="fm",        fam="Moirai",    out="sampled",  params=14e6),
        "moirai_base":       dict(disp="Moirai-1.0 base",  cls="fm",        fam="Moirai",    out="sampled",  params=91e6),
        "moirai_large":      dict(disp="Moirai-1.0 large", cls="fm",        fam="Moirai",    out="sampled",  params=311e6),
        "moirai2_small":     dict(disp="Moirai-2.0 small", cls="fm",        fam="Moirai",    out="quantile", params=11e6),
        "lag_llama":         dict(disp="Lag-Llama",        cls="fm",        fam="Lag-Llama", out="sampled",  params=2e6),
    }
    ORDER_UF = list(MODELS_UF.keys())

    def _disp(k):  return MODELS_UF[k]["disp"]
    def _fam(k):   return MODELS_UF[k]["fam"]
    def _col(k):   return MODEL_COLOR[k]
    def _is_fm(k): return MODELS_UF[k]["cls"] == "fm"

    REFERENCE = {
        "naive":            (517.9, 518.8, 515.6, 3.5e-5),
        "arima":            (86.8, 228.1, 798.1, 2.7e-2),
        "linear":           (82.9, 198.8, 452.4, 6.0e-4),
        "lgbm":             (79.1, 156.6, 311.6, 8.1e-4),
        "chronos_mini":     (84.2, 166.0, 585.3, 7.29e-2),
        "chronos_large":    (78.1, 149.0, 513.6, 8.71e-1),
        "chronos_bolt_mini":(97.1, 153.5, 364.2, 1.00e-3),
        "chronos_bolt_base":(88.0, 140.2, 341.9, 7.79e-3),
        "timesfm_200m":     (88.0, 166.4, 499.0, 9.70e-3),
        "timesfm_500m":     (80.9, 144.1, 387.6, 4.29e-2),
        "moirai_small":     (203.6, 308.0, 788.4, 6.48e-3),
        "moirai_base":      (116.3, 225.4, 796.9, 1.28e-2),
        "moirai_large":     (101.7, 205.7, 765.7, 2.94e-2),
        "moirai2_small":    (79.6, 153.7, 449.3, 1.63e-3),
        "lag_llama":        (399.3, 828.1, 1404.1, 7.34e-1),
    }
    # TimesFM omitted: thesis Table B.6 note c — TimesFM pretraining energy is not
    # estimable, so it must not get a descending amortisation curve.
    PRETRAIN_KWH = {
        "naive": 0.0, "arima": 0.0, "linear": 0.0, "lgbm": 0.0,
        "chronos_bolt_mini": 80.0, "chronos_bolt_base": 130.0,
        "chronos_mini": 37.4, "chronos_large": 294.7,
        "moirai_small": 3.0, "moirai_base": 190.0, "moirai_large": 650.0,
        "moirai2_small": 45.0, "lag_llama": 18.7,
    }
    REPORT: List[str] = []
    SAVED:  List[str] = []
    ACCENT = "#C026A0"

    def log(line=""):
        print(line)
        REPORT.append(str(line))

    def dominates(a, b, axes):
        return all(a[k] <= b[k] for k in axes) and any(a[k] < b[k] for k in axes)

    def frontier(points, axes):
        return [m for m in points
                if not any(dominates(points[o], points[m], axes)
                           for o in points if o is not m)]

    def ring(ax, x, y, color=ACCENT, s=460):
        ax.scatter([x], [y], s=s, facecolors="none", edgecolors=color, linewidths=2.2, zorder=5)

    def _save(fig, name, tight=True):
        """Write PNG (at _DPI) plus a sibling PDF.

        tight=False skips bbox_inches="tight" for figures that are authored at
        their exact printed size: "tight" re-crops to the artists' extent, which
        changes the output width and would silently rescale the point sizes.
        """
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, name + ".png")
        pdf_path = os.path.join(OUT_DIR, name + ".pdf")
        kw = {"bbox_inches": "tight"} if tight else {}
        fig.savefig(path, dpi=_DPI, **kw)
        fig.savefig(pdf_path, **kw)
        plt.close(fig)
        if _shutil.which("pngquant"):
            _subprocess.run(["pngquant", "--quality", "82-98", "--skip-if-larger",
                             "--force", "--ext", ".png", path], check=False)
        SAVED.append(os.path.relpath(path, ROOT))
        print(f"  saved: {os.path.relpath(path, ROOT)} (+ .pdf)")

    def _model_legend(ax, keys, title="model / class"):
        handles = [Line2D([0], [0], marker=("^" if _is_fm(k) else "o"), linestyle="none",
                          markerfacecolor=MODEL_COLOR[k], markeredgecolor="white",
                          markersize=8, label=_disp(k)) for k in keys]
        handles += [
            Line2D([0], [0], marker="None", linestyle="none", label="—"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="gray",
                   markeredgecolor="white", markersize=8, label="Classical"),
            Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="gray",
                   markeredgecolor="white", markersize=8, label="Foundation"),
        ]
        ax.legend(handles=handles, fontsize=7.5, framealpha=0.9, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), title=title, title_fontsize=8)

    # Load + sanity check
    if not os.path.exists(MAIN_METRICS):
        raise FileNotFoundError(MAIN_METRICS)
    df_uf = pd.read_csv(MAIN_METRICS)
    df_uf = df_uf[df_uf["model"] != "sarima"].copy()
    n_origins = int(df_uf["n_predictions"].max())
    df_uf = df_uf[df_uf["n_predictions"] == n_origins]
    df_uf["mean_kwh"] = df_uf[["eval_cc_energy_kwh", "eval_ct_energy_kwh"]].mean(axis=1)
    log(f"  main metrics: {os.path.relpath(MAIN_METRICS, ROOT)}  n_origins={n_origins}")

    rows = []
    for k in ORDER_UF:
        sub = df_uf[df_uf["model"] == k]
        if sub.empty:
            log(f"  [WARN] model {k} missing"); continue
        mae_vals = {h: float(sub[sub["horizon_steps"] == h]["MAE"].iloc[0]) for h in (1, 4, 96)}
        mean_kwh = float(sub["mean_kwh"].iloc[0])
        cc = float(sub["eval_cc_energy_kwh"].iloc[0])
        ct = float(sub["eval_ct_energy_kwh"].iloc[0])
        rows.append(dict(
            key=k, disp=_disp(k), cls=MODELS_UF[k]["cls"], fam=_fam(k),
            out=MODELS_UF[k]["out"], params=MODELS_UF[k]["params"],
            mae_h1=mae_vals[1], mae_h4=mae_vals[4], mae_h96=mae_vals[96],
            cc_kwh=cc, ct_kwh=ct, mean_kwh=mean_kwh,
            per_origin_uwh=mean_kwh / n_origins * 1e9,
        ))
    T = pd.DataFrame(rows).set_index("key").reindex(
        [k for k in ORDER_UF if k in df_uf["model"].unique()])

    log("=" * 78)
    log("SANITY CHECK vs verified headline numbers (1% MAE, 5% energy)")
    log("=" * 78)
    flagged = 0
    for k in T.index:
        r1, r4, r96, e = REFERENCE[k]
        got = T.loc[k]
        for name, ref, val, tol in [
            ("MAE_h1", r1, got.mae_h1, 0.01), ("MAE_h4", r4, got.mae_h4, 0.01),
            ("MAE_h96", r96, got.mae_h96, 0.01), ("mean_kwh", e, got.mean_kwh, 0.05),
        ]:
            if ref == 0:
                continue
            dev = abs(val - ref) / ref
            if dev > tol:
                flagged += 1
                log(f"  [DRIFT] {k:18} {name:9} got={val:.4g} ref={ref:.4g} dev={dev*100:.1f}%")
    log("  all OK" if flagged == 0 else f"  {flagged} value(s) drifted")

    # Pareto
    pts = {k: dict(h1=T.loc[k].mae_h1, h4=T.loc[k].mae_h4,
                   h96=T.loc[k].mae_h96, energy=T.loc[k].mean_kwh) for k in T.index}
    A = frontier(pts, ["h1", "h4", "h96", "energy"])
    B = frontier(pts, ["h4", "h96", "energy"])
    C = [m for m in B if m != "linear"]

    # fig1 — pareto h96
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for k in T.index:
        ax.scatter(T.loc[k].mean_kwh, T.loc[k].mae_h96, s=130,
                   marker=("^" if _is_fm(k) else "o"), color=_col(k),
                   edgecolors="white", linewidths=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(T["mean_kwh"].min() * 0.35, T["mean_kwh"].max() * 3.2)
    ax.set_ylim(T["mae_h96"].min() - 60, T["mae_h96"].max() * 1.10)
    ax.set_xlabel("Mean eval energy (kWh, log scale)")
    ax.set_ylabel("MAE at h=96 (MW)")
    ax.grid(True, which="both", alpha=0.3)
    for k in C:
        ring(ax, T.loc[k].mean_kwh, T.loc[k].mae_h96, s=520)
    _model_legend(ax, list(T.index))
    _save(fig, "pareto_frontier_h96")

    # fig2 — pareto all horizons
    horizons_uf = [("mae_h1", "h = 1"), ("mae_h4", "h = 4"), ("mae_h96", "h = 96")]
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))
    for ax, (col, lab) in zip(axes, horizons_uf):
        for k in T.index:
            ax.scatter(T.loc[k].mean_kwh, T.loc[k][col], s=95,
                       marker=("^" if _is_fm(k) else "o"), color=_col(k),
                       edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(T["mean_kwh"].min() * 0.35, T["mean_kwh"].max() * 3.2)
        ymax = T[col].max()
        ax.set_ylim(T[col].min() - ymax * 0.10, ymax * 1.12)
        ax.set_ylabel("MAE (MW)")
        ax.grid(True, which="both", alpha=0.3)
        ax.annotate(lab, xy=(0.04, 0.93), xycoords="axes fraction",
                    fontsize=13, fontweight="bold")
        for k in C:
            ring(ax, T.loc[k].mean_kwh, T.loc[k][col], s=320)
    axes[-1].set_xlabel("Mean eval energy (kWh, log scale)")
    _model_legend(axes[0], list(T.index))
    fig.tight_layout()
    _save(fig, "pareto_all_horizons")

    # EAS
    EAS_MODELS_UF = ["lgbm", "chronos_bolt_mini", "chronos_bolt_base", "naive",
                     "moirai2_small", "linear", "arima", "chronos_large", "lag_llama"]
    ref_mae = T.loc["lgbm"].mae_h96
    ref_e   = T.loc["lgbm"].mean_kwh
    lam_uf  = np.logspace(-4, 1, 200)
    eas_uf  = {}
    for k in EAS_MODELS_UF:
        eas_uf[k] = (T.loc[k].mae_h96 / ref_mae, T.loc[k].mean_kwh / ref_e)
    nmae_naive, nen_naive = eas_uf["naive"]
    crossover = (nmae_naive - 1) / (1 - nen_naive)

    LS_UF = {"chronos_bolt_mini": "-", "chronos_bolt_base": "--", "chronos_large": ":"}
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    handles_uf = []
    for k in EAS_MODELS_UF:
        nmae, nen = eas_uf[k]
        y = nmae + lam_uf * nen
        line, = ax.plot(lam_uf, y, color=_col(k), lw=2.0, ls=LS_UF.get(k, "-"),
                        zorder=3, label=_disp(k))
        handles_uf.append(line)
    for xv in (0.01, 0.1, 1):
        ax.axvline(xv, color="gray", ls="--", lw=0.8, alpha=0.6)
    cross_line = ax.axvline(crossover, color=ACCENT, ls=":", lw=1.8,
                            label=f"naive/LightGBM crossover  λ≈{crossover:.2f}")
    handles_uf.append(cross_line)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.1e1)
    ax.set_ylim(0, 7.8)
    ax.set_xlabel("λ  (energy weight, log scale)")
    ax.set_ylabel("EAS = nMAE + λ·nEnergy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(handles=handles_uf, fontsize=8.5, framealpha=0.9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), title="model (EAS curve)", title_fontsize=9)
    _save(fig, "eas_lambda_sweep")

    # fig4 — arch vs size
    fm_uf = [k for k in T.index if _is_fm(k)]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for k in fm_uf:
        quant = MODELS_UF[k]["out"] == "quantile"
        if quant:
            ax.scatter(T.loc[k].params, T.loc[k].mae_h96, s=140, marker="D",
                       facecolors="none", edgecolors=_col(k), linewidths=2.0, zorder=3)
        else:
            ax.scatter(T.loc[k].params, T.loc[k].mae_h96, s=140, marker="o",
                       color=_col(k), edgecolors="white", linewidths=0.9, zorder=3)
        ax.annotate(_disp(k), (T.loc[k].params, T.loc[k].mae_h96),
                    xytext=(7, 3), textcoords="offset points", fontsize=7, color="#333", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("MAE at h=96 (MW)")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "arch_vs_size")

    # fig5 — amortization
    #
    # Print geometry: placed at 0.85 x 5.5 in NeurIPS textwidth = 4.68 in. Authored
    # at exactly that width so 1 pt on the canvas is 1 pt on the page — the font
    # sizes below are what the reader sees. (Authored at the old 9.6 in and scaled
    # to fit, 11 pt type would have printed at 5.4 pt.)
    SCENARIO_A = 35040; SCENARIO_B = 3504000; SCENARIO_C = 35040000
    SCENARIOS_UF = [("A", SCENARIO_A), ("B", SCENARIO_B), ("C", SCENARIO_C)]
    marginal = {k: T.loc[k].mean_kwh / n_origins for k in T.index}
    lgbm_floor = marginal["lgbm"]
    N = np.logspace(3, 12, 300)
    amort_rc = {
        "font.size":       11,
        "axes.titlesize":  12,
        "axes.labelsize":  11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize":  9,
    }
    with plt.rc_context(amort_rc):
        fig, ax = plt.subplots(figsize=(4.68, 4.5))
        for k in [m for m in T.index if not _is_fm(m)]:
            ax.plot(N, np.full_like(N, marginal[k]), color=_col(k),
                    lw=(3.6 if k == "lgbm" else 2.2), ls="-",
                    alpha=(1.0 if k == "lgbm" else 0.8), zorder=(5 if k == "lgbm" else 3))
        for k in [m for m in T.index if _is_fm(m) and m in PRETRAIN_KWH]:
            y = PRETRAIN_KWH[k] / N + marginal[k]
            ax.plot(N, y, color=_col(k), lw=2.2, ls="--", alpha=0.85, zorder=3)
        ax.annotate("LightGBM floor", (N[-1], lgbm_floor), xytext=(-4, 6),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=MODEL_COLOR["lgbm"], fontweight="bold")
        ytop = ax.get_ylim()[1]
        for sc, per_year in SCENARIOS_UF:
            for mult, tag in [(1, "1 yr"), (30, "30 yr")]:
                nval = per_year * mult
                ax.axvline(nval, color="#555555", ls=":", lw=1.1, alpha=0.65)
                ax.annotate(f"{sc} ({tag})", (nval, ytop), xytext=(2, -3),
                            textcoords="offset points", rotation=90, va="top",
                            ha="center", fontsize=9, color="#555555")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N forecasts over lifetime (log scale)")
        ax.set_ylabel("Total energy per forecast (kWh, log scale)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        _save(fig, "pretraining_amortization", tight=False)

    # fig6 — robustness
    try:
        etth = pd.concat([
            pd.read_csv(os.path.join(ROOT, "results/classical/20260514_173941_lgbm_naive_etth1/metrics.csv")),
            pd.read_csv(os.path.join(ROOT, "results/foundation_models/20260514_174014_chronos_bolt_mini_etth1/metrics.csv")),
        ], ignore_index=True)
        etth["mean_kwh"] = etth[["eval_cc_energy_kwh", "eval_ct_energy_kwh"]].mean(axis=1)
        ROB_MODELS_UF = ["naive", "lgbm", "chronos_bolt_mini"]
        D_elia = 532.4  # pre-computed MASE denominator
        D_etth = 2.25

        def etth_row(k):
            sub = etth[etth["model"] == k]
            return (float(sub[sub.horizon_steps == 1]["MAE"].iloc[0]),
                    float(sub[sub.horizon_steps == 24]["MAE"].iloc[0]),
                    float(sub["mean_kwh"].iloc[0]),
                    int(sub["n_predictions"].iloc[0]))

        data_rob = {}
        for k in ROB_MODELS_UF:
            e1h, e1d, e_kwh, e_n = etth_row(k)
            data_rob[k] = dict(
                elia_mase_1h=T.loc[k].mae_h4 / D_elia, elia_mase_1d=T.loc[k].mae_h96 / D_elia,
                etth_mase_1h=e1h / D_etth, etth_mase_1d=e1d / D_etth,
                elia_uwh=T.loc[k].per_origin_uwh, etth_uwh=e_kwh / e_n * 1e9,
            )

        order_rob = sorted(ROB_MODELS_UF, key=lambda k: data_rob[k]["elia_mase_1d"])
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))
        groups = [("1 h", "elia_mase_1h", "etth_mase_1h"), ("1 d", "elia_mase_1d", "etth_mase_1d")]
        nb = len(order_rob) * 2
        w  = 0.8 / nb
        for gi, (glab, ce, ct) in enumerate(groups):
            for mi, k in enumerate(order_rob):
                base = gi + (mi * 2 - nb / 2 + 0.5) * w
                axL.bar(base, data_rob[k][ce], w, color=_col(k), edgecolor="white", zorder=3)
                axL.bar(base + w, data_rob[k][ct], w, color=_col(k), hatch="///",
                        edgecolor="white", zorder=3)
        axL.set_xticks(range(len(groups)))
        axL.set_xticklabels([g[0] for g in groups])
        axL.set_ylabel("MASE")
        axL.grid(True, axis="y", alpha=0.3)
        w2 = 0.8 / 2
        for di, (dlab, key, hatch) in enumerate([("Elia", "elia_uwh", ""), ("ETTh1", "etth_uwh", "///")]):
            for mi, k in enumerate(order_rob):
                axR.bar(mi + (di - 0.5) * w2, data_rob[k][key], w2, color=_col(k),
                        hatch=hatch, edgecolor="white", zorder=3)
        axR.set_yscale("log")
        axR.set_xticks(range(len(order_rob)))
        axR.set_xticklabels([_disp(k) for k in order_rob], rotation=15, ha="right")
        axR.set_ylabel("Per-prediction energy (µWh, log scale)")
        axR.grid(True, axis="y", which="both", alpha=0.3)
        handles_rob = [Patch(facecolor="gray", edgecolor="white", label="Elia (solid)"),
                       Patch(facecolor="gray", hatch="///", edgecolor="white", label="ETTh1 (hatched)")]
        handles_rob += [Patch(facecolor=_col(k), edgecolor="white", label=_disp(k))
                        for k in order_rob]
        axR.legend(handles=handles_rob, fontsize=8, framealpha=0.9, loc="upper left",
                   bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        _save(fig, "robustness")
    except FileNotFoundError as e:
        log(f"  [SKIP] robustness figure: {e}")

    # fig7 — all energy sorted
    s_uf = T.sort_values("mean_kwh")
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    y = np.arange(len(s_uf))
    ax.barh(y, s_uf["mean_kwh"].values, color=[_col(k) for k in s_uf.index],
            edgecolor="white", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_disp(k) for k in s_uf.index])
    ax.set_xscale("log")
    ax.set_xlabel("Mean eval energy (kWh, log scale)")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    for yi, k in zip(y, s_uf.index):
        wh = s_uf.loc[k].mean_kwh * 1e3
        ax.annotate(f"{wh:.2f} Wh" if wh < 100 else f"{wh:.0f} Wh",
                    (s_uf.loc[k].mean_kwh, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color="#333")
    ax.set_xlim(s_uf["mean_kwh"].min() * 0.4, s_uf["mean_kwh"].max() * 4)
    _save(fig, "all_models_energy_sorted")

    log(f"\n[DONE] {len(SAVED)} figures saved to {os.path.relpath(OUT_DIR, ROOT)}/")


# --- CLI dispatcher ---

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # bars
    p_bars = sub.add_parser("bars", help="Five thesis bar figures (accuracy + energy)")
    p_bars.add_argument("--metrics", default=None)
    p_bars.add_argument("--out_dir", default=None)

    # variants
    p_var = sub.add_parser("variants", help="Curated accuracy variants (stacked bars + scatter)")
    p_var.add_argument("--metrics", default=None)
    p_var.add_argument("--predictions", default=None)
    p_var.add_argument("--out_dir", default=None)
    p_var.add_argument("--dpi", type=int, default=DPI)
    p_var.add_argument("--sort_horizon", type=int, default=96)
    p_var.add_argument("--scatter_stride", type=int, default=24)

    # carbon
    p_car = sub.add_parser("carbon", help="Five carbon-accounting figures")
    p_car.add_argument("--metrics", default=None)
    p_car.add_argument("--out_dir", default=None)

    # arima_sarima
    p_as = sub.add_parser("arima_sarima", help="ARIMA vs SARIMA comparison")
    p_as.add_argument("--results_dir", default="results")
    p_as.add_argument("--run_id", default=None)
    p_as.add_argument("--out_dir", default=None)
    p_as.add_argument("--dpi", type=int, default=DPI)

    # arima_lgbm
    p_al = sub.add_parser("arima_lgbm", help="ARIMA vs LightGBM weekly zoom at h=96")
    p_al.add_argument("--predictions", default=None)
    p_al.add_argument("--raw", default=None)
    p_al.add_argument("--out", default=None)
    p_al.add_argument("--week_start", default="2026-01-05")
    p_al.add_argument("--week_end",   default="2026-01-12")
    p_al.add_argument("--horizon", type=int, default=96)
    p_al.add_argument("--dpi", type=int, default=DPI)

    # pareto
    p_par = sub.add_parser("pareto", help="Three-horizon Pareto panels")
    p_par.add_argument("--metrics", default="results/combined/combined_all_models_final/metrics.csv")
    p_par.add_argument("--out_dir", default=None)
    p_par.add_argument("--dpi", type=int, default=200)

    # pareto_h96
    p_p96 = sub.add_parser("pareto_h96", help="Single-horizon Pareto at h=96")
    p_p96.add_argument("--metrics", default="results/combined/combined_all_models_final/metrics.csv")
    p_p96.add_argument("--out_dir", default=None)
    p_p96.add_argument("--dpi", type=int, default=200)

    # backends
    p_be = sub.add_parser("backends", help="Eval-phase energy by measurement backend")
    p_be.add_argument("--metrics", default="results/combined/combined_all_models_final/metrics.csv")
    p_be.add_argument("--out_dir", default=None)
    p_be.add_argument("--dpi", type=int, default=200)

    # params
    p_pm = sub.add_parser("params", help="Foundation-model size vs h=96 accuracy")
    p_pm.add_argument("--metrics", default="results/combined/combined_all_models_final/metrics.csv")
    p_pm.add_argument("--out_dir", default=None)
    p_pm.add_argument("--dpi", type=int, default=200)

    # unified
    p_un = sub.add_parser("unified", help="Chapter 6 definitive figures + verification report")
    p_un.add_argument("--dpi", type=int, default=300)
    p_un.add_argument("--metrics", default=None,
                      help="Combined metrics.csv. Defaults to results/combined_all/metrics.csv.")
    p_un.add_argument("--out_dir", default=None,
                      help="Output directory. Defaults to figures/chapter6_unified/.")

    args = p.parse_args()
    dispatch = {
        "bars":         cmd_bars,
        "variants":     cmd_variants,
        "carbon":       cmd_carbon,
        "arima_sarima": cmd_arima_sarima,
        "arima_lgbm":   cmd_arima_lgbm,
        "pareto":       cmd_pareto,
        "pareto_h96":   cmd_pareto_h96,
        "backends":     cmd_backends,
        "params":       cmd_params,
        "unified":      cmd_unified,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
