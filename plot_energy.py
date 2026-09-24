#!/usr/bin/env python3
"""Energy + accuracy visualisation for the walk-forward benchmark.

Reads results/*/*/metrics.csv and produces 8 figures (accuracy, energy, CO₂,
efficiency, trade-off, duration) saved to figures/energy_<timestamp>/.

Usage:
    python plot_energy.py
    python plot_energy.py --run_id wf_20260407_122418
    python plot_energy.py --results_dir results --dpi 150
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
})

# --- Palette (shared with visualize_results.py — see src/palette.py) ---
from src.palette import (
    MODEL_COLOR, MODEL_LABEL, MODEL_ORDER,
    HORIZON_HATCH, HORIZON_LABEL,
    BACKEND_HATCH, TSFM_KEYS, ANNOT_COLOR,
)


# --- Data loading ---

def _load_run(results_dir: str, run_id: Optional[str]) -> pd.DataFrame:
    """Return a tidy DataFrame for the requested (or best) run."""
    all_files = sorted(
        glob.glob(os.path.join(results_dir, "classical/wf_*/metrics.csv"))
        + glob.glob(os.path.join(results_dir, "foundation_models/tsfm_*/metrics.csv"))
        + glob.glob(os.path.join(results_dir, "combined/*/metrics.csv"))
        # legacy: flat layout
        + glob.glob(os.path.join(results_dir, "wf_*/metrics.csv"))
        + glob.glob(os.path.join(results_dir, "tsfm_*/metrics.csv"))
    )
    if not all_files:
        raise FileNotFoundError(
            f"No metrics.csv found under {results_dir}/classical/, "
            f"{results_dir}/foundation_models/, or {results_dir}/combined/"
        )

    all_dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df["_file"] = f
            all_dfs.append(df)
        except Exception:
            pass
    full = pd.concat(all_dfs, ignore_index=True)

    # Normalise legacy column names (pre-dual-tracker runs used different names)
    if "fit_energy_kwh" in full.columns and "fit_cc_energy_kwh" not in full.columns:
        full = full.rename(columns={
            "fit_energy_kwh":    "fit_cc_energy_kwh",
            "fit_emissions_kg":  "fit_cc_emissions_kg",
            "eval_energy_kwh":   "eval_cc_energy_kwh",
            "eval_emissions_kg": "eval_cc_emissions_kg",
        })
    for col in ["fit_ct_energy_kwh", "fit_ct_emissions_kg",
                "eval_ct_energy_kwh", "eval_ct_emissions_kg"]:
        if col not in full.columns:
            full[col] = float("nan")

    if run_id:
        df = full[full["run_id"] == run_id].copy()
        if df.empty:
            raise ValueError(f"run_id '{run_id}' not found. "
                             f"Available: {sorted(full['run_id'].unique())}")
        return df

    # Auto-select: most recent run with CC energy data and ≥ 2 models
    has_cc = full["eval_cc_energy_kwh"].notna()
    candidates = full[has_cc].groupby("run_id")["model"].nunique()
    candidates = candidates[candidates >= 2].sort_index(ascending=False)
    if candidates.empty:
        # Fall back to the most recent run regardless
        chosen = full["run_id"].sort_values(ascending=False).iloc[0]
        print(f"  [warn] No run with CC energy data found. Using latest run: {chosen}")
    else:
        chosen = candidates.index[0]

    df = full[full["run_id"] == chosen].copy()
    print(f"  Selected run: {chosen}  ({df['model'].nunique()} models, "
          f"{df['horizon_steps'].nunique()} horizons)")
    return df


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per model per horizon (deduplicate, sort, set model order)."""
    df = df.copy()
    df["model"] = df["model"].str.lower().str.strip()
    # Deduplicate: if a model appears multiple times keep the row with most energy data
    df["_energy_score"] = df["eval_cc_energy_kwh"].notna().astype(int)
    df = (df.sort_values("_energy_score", ascending=False)
            .drop_duplicates(["model", "horizon_steps"])
            .drop(columns="_energy_score"))
    # Sort models in display order
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["_order"] = df["model"].map(lambda m: order_map.get(m, 99))
    df = df.sort_values(["_order", "horizon_steps"]).drop(columns="_order")
    return df.reset_index(drop=True)


def _models(df: pd.DataFrame) -> List[str]:
    known   = [m for m in MODEL_ORDER if m in df["model"].unique()]
    unknown = sorted(m for m in df["model"].unique() if m not in MODEL_ORDER)
    return known + unknown


def _models_sorted_by(df: pd.DataFrame, value_by_model) -> List[str]:
    """Return models ordered ascending by `value_by_model(model)`. NaN/inf last."""
    models = _models(df)
    def _v(m):
        v = value_by_model(m)
        try:
            return float(v) if (v is not None and np.isfinite(float(v)) and float(v) > 0) else float("inf")
        except (TypeError, ValueError):
            return float("inf")
    return sorted(models, key=_v)


def _drop_stride_mismatched(df: pd.DataFrame) -> pd.DataFrame:
    """Drop SARIMA rows for energy/duration/CO2 comparisons.

    SARIMA was run at a much larger stride than the rest of the cohort, so its
    eval energy and wall-clock are not on the same scale. It belongs only in
    the dedicated ARIMA-vs-SARIMA panel (compare_arima_sarima.py).
    """
    return df[df["model"] != "sarima"].copy()


def _horizons(df: pd.DataFrame) -> List[int]:
    return sorted(df["horizon_steps"].unique())


def _color(model: str) -> str:
    return MODEL_COLOR.get(model, "#607D8B")


def _label(model: str) -> str:
    return MODEL_LABEL.get(model, model.upper())


def _savefig(fig: plt.Figure, out_dir: str, name: str, dpi: int) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# --- Figure 1 — Accuracy ---

def plot_accuracy(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Accuracy bars: each model keeps its own color; horizon shown by hatch."""
    from matplotlib.patches import Patch
    df = _drop_stride_mismatched(df)
    horizons = sorted(_horizons(df))
    h_sort   = horizons[-1]  # sort by MAE at the longest horizon (most discriminating)
    def _mae_at(m):
        sub = df[(df["model"] == m) & (df["horizon_steps"] == h_sort)]
        return float(sub["MAE"].iloc[0]) if not sub.empty else float("inf")
    models = _models_sorted_by(df, _mae_at)
    n_h  = len(horizons)
    n_m  = len(models)
    x    = np.arange(n_m)
    bw   = 0.70 / n_h          # bar width per horizon slot

    fig, axes = plt.subplots(1, 2, figsize=(max(13, n_m * 1.5), 6.0))

    for ax, metric, ylabel in zip(axes, ["MAE", "MAPE_pct"], ["MAE (MW)", "MAPE (%)"]):
        # Track max value across this axis so we can size headroom for annotations
        ax_max = 0.0
        for i, h in enumerate(horizons):
            offset = (i - n_h / 2 + 0.5) * bw
            hatch  = HORIZON_HATCH.get(h, "")
            for j, m in enumerate(models):
                val = df.loc[(df["model"] == m) & (df["horizon_steps"] == h), metric].mean()
                if np.isnan(val):
                    continue
                c = _color(m)
                ax.bar(x[j] + offset, val, bw * 0.92,
                       color=c, hatch=hatch,
                       alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
                ax_max = max(ax_max, val)
                # Vertical text in gray — fits comfortably above each bar
                pad = val * 0.02 + (0.5 if metric == "MAPE_pct" else 5)
                ax.text(x[j] + offset, val + pad, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7, rotation=90,
                        color=ANNOT_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"Forecast accuracy — {ylabel.split('(')[0].strip()}", fontsize=11)
        # Extra headroom (~20%) so the vertical annotations clear the legend.
        ax.set_ylim(bottom=0, top=ax_max * 1.22 if ax_max > 0 else None)

        legend_els = [
            Patch(facecolor="gray", hatch=HORIZON_HATCH.get(h, ""),
                  edgecolor="white", alpha=0.85,
                  label=HORIZON_LABEL.get(h, f"h={h}"))
            for h in horizons
        ]
        ax.legend(handles=legend_els, title="Horizon", framealpha=0.85, fontsize=8,
                  loc="upper left", bbox_to_anchor=(1.005, 1.0))

    fig.suptitle("Walk-forward accuracy — each model's colour, horizon by fill pattern",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "01_accuracy_bar.png", dpi)


# --- Figure 2 — CodeCarbon vs CarbonTracker total energy ---

def plot_energy_cc_vs_ct(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Energy per model: 3 bars side by side — CodeCarbon (solid), CarbonTracker
    (///), nvidia_smi (xxx). Model color preserved; log scale."""
    from matplotlib.patches import Patch
    df = _drop_stride_mismatched(df)
    cols = ["fit_cc_energy_kwh", "eval_cc_energy_kwh",
            "fit_ct_energy_kwh", "eval_ct_energy_kwh",
            "fit_nv_energy_kwh", "eval_nv_energy_kwh"]
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    summary = (df.groupby("model")[cols].first().reset_index())
    summary["total_cc"] = (summary["fit_cc_energy_kwh"].fillna(0)
                           + summary["eval_cc_energy_kwh"].fillna(0))
    summary["total_ct"] = (summary["fit_ct_energy_kwh"].fillna(0)
                           + summary["eval_ct_energy_kwh"].fillna(0))
    summary["total_nv"] = (summary["fit_nv_energy_kwh"].fillna(0)
                           + summary["eval_nv_energy_kwh"].fillna(0))
    # Sort ascending by mean total across the populated backends
    def _mean_total(m):
        row = summary[summary["model"] == m]
        if row.empty: return float("inf")
        vals = [row[c].iloc[0] for c in ("total_cc","total_ct")
                if np.isfinite(row[c].iloc[0]) and row[c].iloc[0] > 0]
        return float(np.mean(vals)) if vals else float("inf")
    models = _models_sorted_by(df, _mean_total)
    summary = summary.set_index("model").reindex(models)

    cc_vals = summary["total_cc"].values * 1e6   # kWh → μWh
    ct_vals = summary["total_ct"].values * 1e6
    nv_vals = summary["total_nv"].values * 1e6
    colors  = [_color(m) for m in models]

    has_ct = np.any(np.nan_to_num(ct_vals) > 0)
    has_nv = np.any(np.nan_to_num(nv_vals) > 0)
    backends = [("CodeCarbon", cc_vals, BACKEND_HATCH["CC"], 0.92)]
    if has_ct: backends.append(("CarbonTracker", ct_vals, BACKEND_HATCH["CT"], 0.75))
    if has_nv: backends.append(("nvidia_smi",    nv_vals, BACKEND_HATCH["NV"], 0.60))
    n_b = len(backends)

    x = np.arange(len(models))
    w = 0.8 / n_b
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 5.6))

    max_v = 0.0
    for bi, (name, vals, hatch, alpha) in enumerate(backends):
        xs = x + (bi - (n_b - 1) / 2.0) * w
        for i, (v, c) in enumerate(zip(vals, colors)):
            if not np.isfinite(v) or v <= 0:
                continue
            ax.bar(xs[i], v, w, color=c, alpha=alpha,
                   hatch=hatch, edgecolor="white", linewidth=0.4, zorder=3)
            max_v = max(max_v, v)
            ax.text(xs[i], v * 1.18, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                    color=ANNOT_COLOR)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.05, top=max_v * 6 if max_v > 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Total energy (μWh) — log scale", fontsize=9)
    ax.set_title("Energy consumption per model — CodeCarbon vs CarbonTracker vs nvidia_smi\n"
                 "(fit + eval, all horizons combined)", fontsize=11)

    legend_els = [Patch(facecolor="lightgray", alpha=alpha, hatch=hatch,
                        edgecolor="#444", label=name)
                  for name, _, hatch, alpha in backends]
    ax.legend(handles=legend_els, framealpha=0.85, fontsize=8,
              title="energy backend", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    _savefig(fig, out_dir, "02_energy_cc_vs_ct.png", dpi)


# --- Figure 3 — Fit vs eval energy breakdown ---

def plot_energy_breakdown(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Per model: 3 grouped bars (CC, CT, NV), each split into fit (solid) + eval (hatched).

    Every backend uses its own hatch on the *eval* portion (the visible part on
    log scale), so the three bars are visually distinct even when the fit cost
    is zero (zero-shot foundation models). Backend label sits *above* each bar's
    annotation rather than on the x-axis baseline, to avoid overlap on log-scale.
    """
    from matplotlib.patches import Patch
    df = _drop_stride_mismatched(df)
    cols = ["fit_cc_energy_kwh", "eval_cc_energy_kwh",
            "fit_ct_energy_kwh", "eval_ct_energy_kwh",
            "fit_nv_energy_kwh", "eval_nv_energy_kwh"]
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    tmp = (df.groupby("model")[cols].first().reset_index())
    cc_ct_cols = [c for c in cols if "_cc_" in c or "_ct_" in c]
    tmp["total_any"] = tmp[cc_ct_cols].fillna(0).sum(axis=1)
    # Sort ascending by total summed across CC+CT backends/phases
    def _tot(m):
        r = tmp[tmp["model"] == m]
        return float(r["total_any"].iloc[0]) if not r.empty else float("inf")
    models = _models_sorted_by(df, _tot)
    summary = tmp.set_index("model").reindex(models)

    backends = [("CC", "fit_cc_energy_kwh", "eval_cc_energy_kwh", BACKEND_HATCH["CC"]),
                ("CT", "fit_ct_energy_kwh", "eval_ct_energy_kwh", BACKEND_HATCH["CT"]),
                ("NV", "fit_nv_energy_kwh", "eval_nv_energy_kwh", BACKEND_HATCH["NV"])]
    has = {b[0]: summary[b[2]].fillna(0).sum() > 0 for b in backends}
    backends = [b for b in backends if has[b[0]]]
    n_b = len(backends)

    colors = [_color(m) for m in models]
    x = np.arange(len(models))
    w = 0.8 / n_b
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 1.5), 5.6))

    max_total = 0.0
    for bi, (bname, fcol, ecol, bhatch) in enumerate(backends):
        xs = x + (bi - (n_b - 1) / 2.0) * w
        fit_vals  = summary[fcol].fillna(0).values * 1e6
        eval_vals = summary[ecol].fillna(0).values * 1e6
        for i, (m, f, e, c) in enumerate(zip(models, fit_vals, eval_vals, colors)):
            if f + e <= 0:
                continue
            # Fit (solid color, no hatch) — only visible for classical models
            if f > 0:
                ax.bar(xs[i], f, w, color=c, alpha=0.95,
                       edgecolor="white", linewidth=0.4, zorder=3)
            # Eval (hatched per backend) — dominant on log scale for FMs
            ax.bar(xs[i], e, w, color=c, alpha=0.60,
                   bottom=f, hatch=bhatch, edgecolor="white",
                   linewidth=0.4, zorder=3)
            total = f + e
            if total > 0.01:
                max_total = max(max_total, total)
                ax.text(xs[i], total * 1.18, f"{total:.1f}",
                        ha="center", va="bottom", fontsize=7, rotation=90,
                        color=ANNOT_COLOR)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.05, top=max_total * 8 if max_total > 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Energy (μWh) — log scale", fontsize=9)
    ax.set_title("Energy breakdown: fit vs evaluation, per backend (CC / CT / NV)", fontsize=11)

    # Combined legend: backends (with their own hatches) + fit/eval pattern key.
    legend_els = [
        Patch(facecolor="lightgray", alpha=0.60, hatch=bhatch,
              edgecolor="#444", label=f"{bname} (eval)")
        for bname, _, _, bhatch in backends
    ]
    legend_els.append(
        Patch(facecolor="lightgray", alpha=0.95, edgecolor="#444", label="Fit (any backend)")
    )
    ax.legend(handles=legend_els, framealpha=0.85, fontsize=8,
              title="energy backend / phase", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    _savefig(fig, out_dir, "03_energy_breakdown.png", dpi)


# --- Figure 4 — CO₂ emissions ---

def plot_co2(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    df = _drop_stride_mismatched(df)
    summary = (df.groupby("model")[
        ["fit_cc_emissions_kg", "eval_cc_emissions_kg",
         "fit_ct_emissions_kg", "eval_ct_emissions_kg",
         "fit_cc_energy_kwh",   "eval_cc_energy_kwh",
         "fit_ct_energy_kwh",   "eval_ct_energy_kwh"]
    ].first().reset_index())
    summary["total_cc"] = (summary["fit_cc_emissions_kg"].fillna(0)
                           + summary["eval_cc_emissions_kg"].fillna(0))
    summary["total_ct"] = (summary["fit_ct_emissions_kg"].fillna(0)
                           + summary["eval_ct_emissions_kg"].fillna(0))
    summary["total_cc_energy"] = (summary["fit_cc_energy_kwh"].fillna(0)
                                  + summary["eval_cc_energy_kwh"].fillna(0))
    summary["total_ct_energy"] = (summary["fit_ct_energy_kwh"].fillna(0)
                                  + summary["eval_ct_energy_kwh"].fillna(0))
    # Sort ascending by mean CO2 across populated backends
    def _mean_co2(m):
        r = summary[summary["model"] == m]
        if r.empty: return float("inf")
        vals = [r[c].iloc[0] for c in ("total_cc","total_ct")
                if np.isfinite(r[c].iloc[0]) and r[c].iloc[0] > 0]
        return float(np.mean(vals)) if vals else float("inf")
    models = _models_sorted_by(df, _mean_co2)
    summary = summary.set_index("model").reindex(models)

    # Diagnostic: implied carbon intensity per model
    print("\n[Fig 04 CO₂ diagnostic — implied intensity (g CO₂/kWh)]")
    print(f"  {'model':<10} {'cc_µWh':>10} {'ct_µWh':>10} {'cc_mg':>12} {'ct_mg':>12} {'cc_g/kWh':>10} {'ct_g/kWh':>10}")
    for m in models:
        row = summary.loc[m]
        cc_e  = row["total_cc_energy"]
        ct_e  = row["total_ct_energy"]
        cc_co = row["total_cc"]
        ct_co = row["total_ct"]
        cc_e_uwh = cc_e  * 1e6 if not (np.isnan(cc_e)  if isinstance(cc_e,  float) else False) else float("nan")
        ct_e_uwh = ct_e  * 1e6 if not (np.isnan(ct_e)  if isinstance(ct_e,  float) else False) else float("nan")
        cc_co_mg = cc_co * 1e6 if not (np.isnan(cc_co) if isinstance(cc_co, float) else False) else float("nan")
        ct_co_mg = ct_co * 1e6 if not (np.isnan(ct_co) if isinstance(ct_co, float) else False) else float("nan")
        cc_int_s = f"{(cc_co / cc_e) * 1000:.1f}" if (cc_e > 0 and cc_co > 0 and np.isfinite(cc_e) and np.isfinite(cc_co)) else "n/a"
        ct_int_s = f"{(ct_co / ct_e) * 1000:.1f}" if (ct_e > 0 and ct_co > 0 and np.isfinite(ct_e) and np.isfinite(ct_co)) else "n/a"
        cc_e_s  = f"{cc_e_uwh:.2f}"  if np.isfinite(cc_e_uwh)  else "n/a"
        ct_e_s  = f"{ct_e_uwh:.2f}"  if np.isfinite(ct_e_uwh)  else "n/a"
        cc_co_s = f"{cc_co_mg:.3f}"  if np.isfinite(cc_co_mg)  else "n/a"
        ct_co_s = f"{ct_co_mg:.3f}"  if np.isfinite(ct_co_mg)  else "n/a"
        print(f"  {m:<10} {cc_e_s:>10} {ct_e_s:>10} {cc_co_s:>12} {ct_co_s:>12} {cc_int_s:>10} {ct_int_s:>10}")

    # Also pull NV emissions
    for col in ["fit_nv_emissions_kg", "eval_nv_emissions_kg"]:
        if col not in df.columns:
            df[col] = float("nan")
    nv_sum = (df.groupby("model")[["fit_nv_emissions_kg", "eval_nv_emissions_kg"]]
                .first().reset_index().set_index("model").reindex(models))
    summary["total_nv"] = (nv_sum["fit_nv_emissions_kg"].fillna(0)
                           + nv_sum["eval_nv_emissions_kg"].fillna(0))

    # Convert kg → mg
    cc_vals = summary["total_cc"].values * 1e6
    ct_vals = summary["total_ct"].values * 1e6
    nv_vals = summary["total_nv"].values * 1e6
    colors  = [_color(m) for m in models]

    backends = [("CodeCarbon",    cc_vals, BACKEND_HATCH["CC"], 0.92)]
    if np.any(np.nan_to_num(ct_vals) > 0):
        backends.append(("CarbonTracker", ct_vals, BACKEND_HATCH["CT"], 0.75))
    if np.any(np.nan_to_num(nv_vals) > 0):
        backends.append(("nvidia_smi",    nv_vals, BACKEND_HATCH["NV"], 0.60))
    n_b = len(backends)

    x = np.arange(len(models))
    w = 0.8 / n_b
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 5.6))

    max_v = 0.0
    for bi, (name, vals, hatch, alpha) in enumerate(backends):
        xs = x + (bi - (n_b - 1) / 2.0) * w
        for i, (v, c) in enumerate(zip(vals, colors)):
            if not np.isfinite(v) or v <= 0:
                continue
            ax.bar(xs[i], v, w, color=c, alpha=alpha,
                   hatch=hatch, edgecolor="white", linewidth=0.4, zorder=3)
            max_v = max(max_v, v)
            ax.text(xs[i], v * 1.18, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                    color=ANNOT_COLOR)

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4, top=max_v * 6 if max_v > 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("CO₂eq emissions (mg) — log scale", fontsize=9)
    ax.set_title("CO₂ equivalent emissions — CodeCarbon vs CarbonTracker vs nvidia_smi\n"
                 "(different carbon-intensity sources and PUE assumptions)", fontsize=11)

    from matplotlib.patches import Patch as _Patch2
    legend_els = [_Patch2(facecolor="lightgray", alpha=alpha, hatch=hatch,
                          edgecolor="#444", label=name)
                  for name, _, hatch, alpha in backends]
    ax.legend(handles=legend_els, framealpha=0.85, fontsize=8,
              title="backend", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    _savefig(fig, out_dir, "04_co2_bar.png", dpi)


# --- Figure 5 — Energy per prediction ---

def plot_energy_per_prediction(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Normalise eval energy by n_predictions → nWh per prediction.
    3 bars per model: CC / CT / NV. Default-zero NaN to keep bars where data missing."""
    from matplotlib.patches import Patch as _Patch
    df = _drop_stride_mismatched(df)
    # Sort ascending by mean energy-per-prediction across populated backends
    h_min = df["horizon_steps"].min()
    def _per_pred_mean(m):
        sub = df[(df["model"] == m) & (df["horizon_steps"] == h_min)]
        if sub.empty: return float("inf")
        n_pred = float(sub["n_predictions"].iloc[0]) or 1.0
        vals = []
        for cc_f, cc_e in [("fit_cc_energy_kwh","eval_cc_energy_kwh"),
                            ("fit_ct_energy_kwh","eval_ct_energy_kwh")]:
            if cc_f not in df.columns or cc_e not in df.columns:
                continue
            fit = float(df[df["model"]==m][cc_f].iloc[0] or 0.0) if not df[df["model"]==m].empty else 0.0
            ev  = float(sub[cc_e].iloc[0] or 0.0)
            tot = (fit + ev) / n_pred * 1e9
            if np.isfinite(tot) and tot > 0:
                vals.append(tot)
        return float(np.mean(vals)) if vals else float("inf")
    models = _models_sorted_by(df, _per_pred_mean)
    h1 = df[df["horizon_steps"] == df["horizon_steps"].min()].copy().set_index("model").reindex(models)
    n = h1["n_predictions"].fillna(1).values

    def _per(col_fit: str, col_eval: str) -> np.ndarray:
        for c in (col_fit, col_eval):
            if c not in df.columns:
                df[c] = float("nan")
        fit = df.groupby("model")[col_fit].first().reindex(models).fillna(0).values
        ev  = h1[col_eval].fillna(0).values
        return (fit + ev) / n * 1e9  # kWh → nWh per prediction

    cc_per = _per("fit_cc_energy_kwh", "eval_cc_energy_kwh")
    ct_per = _per("fit_ct_energy_kwh", "eval_ct_energy_kwh")
    nv_per = _per("fit_nv_energy_kwh", "eval_nv_energy_kwh")

    backends = [("CodeCarbon",    cc_per, BACKEND_HATCH["CC"], 0.92)]
    if np.any(ct_per > 0):
        backends.append(("CarbonTracker", ct_per, BACKEND_HATCH["CT"], 0.75))
    if np.any(nv_per > 0):
        backends.append(("nvidia_smi",    nv_per, BACKEND_HATCH["NV"], 0.60))
    n_b = len(backends)

    colors = [_color(m) for m in models]
    x = np.arange(len(models))
    w = 0.8 / n_b
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.4), 5.6))

    max_v = 0.0
    for bi, (name, vals, hatch, alpha) in enumerate(backends):
        xs = x + (bi - (n_b - 1) / 2.0) * w
        for i, (v, c) in enumerate(zip(vals, colors)):
            if not np.isfinite(v) or v <= 0:
                continue
            ax.bar(xs[i], v, w, color=c, alpha=alpha,
                   hatch=hatch, edgecolor="white", linewidth=0.4, zorder=3)
            max_v = max(max_v, v)
            ax.text(xs[i], v * 1.18, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                    color=ANNOT_COLOR)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.001, top=max_v * 6 if max_v > 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Energy per prediction (nWh) — log scale", fontsize=9)
    ax.set_title(f"Energy efficiency per prediction (fit+eval)\n"
                 f"(h=min, n = {int(n[0]):,} origins)", fontsize=11)

    legend_els = [_Patch(facecolor="lightgray", alpha=alpha, hatch=hatch,
                         edgecolor="#444", label=name)
                  for name, _, hatch, alpha in backends]
    ax.legend(handles=legend_els, framealpha=0.85, fontsize=8,
              title="backend", title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()
    _savefig(fig, out_dir, "05_energy_per_pred.png", dpi)


# --- Figure 6 — Accuracy–energy trade-off scatter ---

def plot_tradeoff(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Accuracy–energy scatter per horizon.

    Models are identified by color + marker shape (legend on the right) instead
    of inline labels, which used to overlap whenever models clustered. ARIMA is
    annotated with a single italic note if its energy is constant across horizons.
    """
    from matplotlib.lines import Line2D
    df = _drop_stride_mismatched(df)
    models   = _models(df)
    horizons = _horizons(df)
    n_h = len(horizons)

    fig, axes = plt.subplots(1, n_h, figsize=(5.4 * n_h + 2, 4.8), sharey=False)
    if n_h == 1:
        axes = [axes]

    energy_col = "eval_cc_energy_kwh"

    # ARIMA's eval energy may be the same across horizons (it's logged once and
    # repeated). Flag that with a discreet note on the lowest-h panel.
    arima_e_vals = []
    if "arima" in models:
        for h_check in horizons:
            sub_h = df[(df["horizon_steps"] == h_check) & (df["model"] == "arima")]
            if not sub_h.empty:
                v = sub_h.iloc[0][energy_col]
                if pd.notna(v):
                    arima_e_vals.append(v)
    arima_energy_constant = (
        len(arima_e_vals) >= 2 and
        (max(arima_e_vals) - min(arima_e_vals)) / max(arima_e_vals) < 0.05
    )

    # Track which models actually appear so the side legend stays compact.
    seen: List[str] = []
    for ax, h in zip(axes, horizons):
        sub = df[df["horizon_steps"] == h].set_index("model").reindex(models)
        all_energies = []
        for m in models:
            row = sub.loc[m]
            e = row[energy_col]
            mape = row["MAPE_pct"]
            if pd.isna(e) or pd.isna(mape):
                continue
            e_uwh = e * 1e6
            all_energies.append(e_uwh)
            s_size  = 160 if m == "naive" else 120
            pt_alpha = 0.7 if m == "naive" else 1.0
            marker  = "^" if m in TSFM_KEYS else "o"
            ax.scatter(e_uwh, mape, color=_color(m), s=s_size, zorder=5,
                       marker=marker, edgecolors="white", linewidths=0.8,
                       alpha=pt_alpha)
            if m not in seen:
                seen.append(m)
            if m == "arima" and arima_energy_constant and h == horizons[0]:
                ax.annotate("ARIMA energy: total, not per-horizon",
                            xy=(e_uwh, mape), xytext=(8, -14),
                            textcoords="offset points", fontsize=6.5,
                            color="gray", style="italic")

        ax.set_xlabel("Eval energy per horizon — CodeCarbon (μWh)", fontsize=9)
        ax.set_ylabel("MAPE (%)", fontsize=9)
        ax.set_title(f"Trade-off — {HORIZON_LABEL.get(h, f'h={h}')}", fontsize=10)
        ax.set_xscale("log")
        if all_energies:
            x_min = min(all_energies) * 0.3
            x_max = max(all_energies) * 3
            ax.set_xlim(x_min, x_max)

    # Side legend (rightmost subplot) — model dots in MODEL_ORDER, then a
    # tiny key explaining marker shape.
    legend_els = [
        Line2D([0], [0], marker=("^" if m in TSFM_KEYS else "o"),
               color="none", markerfacecolor=_color(m),
               markeredgecolor="white", markersize=8,
               label=_label(m))
        for m in seen
    ]
    legend_els.append(
        Line2D([0], [0], marker="None", color="none", label="—")  # spacer
    )
    legend_els.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
               markersize=7, label="Classical")
    )
    legend_els.append(
        Line2D([0], [0], marker="^", color="none", markerfacecolor="gray",
               markersize=7, label="Foundation (zero-shot)")
    )
    axes[-1].legend(handles=legend_els, fontsize=7.5, framealpha=0.85,
                    loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    title="model", title_fontsize=8)

    fig.suptitle("Accuracy–energy trade-off by horizon  (lower-left = better)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "06_tradeoff.png", dpi)


# --- Figure 7 — Wall-clock duration ---

def plot_duration(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Wall-clock time: model color, fit/load=solid or dotted / eval=hatched.

    The pre-eval phase is *fit (training)* for classical models but *model load*
    (weights download + GPU transfer) for the zero-shot foundation models — these
    are visually separated via different fill patterns so the legend has three
    entries instead of conflating them as 'Fit / model load'.
    """
    from matplotlib.patches import Patch
    df = _drop_stride_mismatched(df)
    tmp = (df.groupby("model")[["fit_duration_s", "eval_duration_s"]]
             .first().reset_index())
    tmp["total"] = tmp["fit_duration_s"].fillna(0) + tmp["eval_duration_s"].fillna(0)
    # Sort ascending by total wall-clock time
    def _tot(m):
        r = tmp[tmp["model"] == m]
        return float(r["total"].iloc[0]) if not r.empty else float("inf")
    models = _models_sorted_by(df, _tot)
    summary = tmp.set_index("model").reindex(models)

    fit_s   = summary["fit_duration_s"].fillna(0).values
    eval_s  = summary["eval_duration_s"].fillna(0).values
    colors  = [_color(m) for m in models]

    x = np.arange(len(models))
    w = 0.55
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.3), 5.4))

    max_total = 0.0
    for i, (m, f, e, c) in enumerate(zip(models, fit_s, eval_s, colors)):
        is_fm = m in TSFM_KEYS
        if f > 0:
            # FM pre-eval = model load (dotted); classical pre-eval = fit (solid)
            ax.bar(x[i], f, w, color=c, alpha=0.92,
                   hatch="..." if is_fm else "",
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.bar(x[i], e, w, color=c, alpha=0.55,
               bottom=f, hatch="///", edgecolor="white", linewidth=0.4, zorder=3)
        total = f + e
        if total > 0:
            max_total = max(max_total, total)
            lbl = f"{total:.0f}s" if total < 120 else f"{total/60:.1f}m"
            ax.text(x[i], total * 1.18, lbl, ha="center", va="bottom",
                    fontsize=7, rotation=90, color=ANNOT_COLOR)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5, top=max_total * 6 if max_total > 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Wall-clock time (s) — log scale", fontsize=9)
    ax.set_title("Computation time: fit / model load vs walk-forward evaluation", fontsize=11)
    legend_els = [
        Patch(facecolor="lightgray", alpha=0.92, edgecolor="#444",
              label="Fit (classical training)"),
        Patch(facecolor="lightgray", alpha=0.92, hatch="...", edgecolor="#444",
              label="Model load (FM weights + GPU)"),
        Patch(facecolor="lightgray", alpha=0.55, hatch="///",
              edgecolor="#444", label="Eval (walk-forward)"),
    ]
    ax.legend(handles=legend_els, framealpha=0.85, fontsize=8,
              loc="upper left", bbox_to_anchor=(1.005, 1.0))

    fig.tight_layout()
    _savefig(fig, out_dir, "07_duration_bar.png", dpi)


def plot_duration_tradeoff(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Accuracy vs total duration — the only tradeoff plot that includes TSFM.

    Inline labels removed; model identity comes from a side legend (color +
    marker shape), which guarantees no label-on-label overlap.
    """
    from matplotlib.lines import Line2D
    df = _drop_stride_mismatched(df)
    models   = _models(df)
    horizons = sorted(_horizons(df))
    n_h      = len(horizons)

    dur_first = df.groupby("model")[["fit_duration_s", "eval_duration_s"]].first()

    fig, axes = plt.subplots(1, n_h, figsize=(5.5 * n_h + 2, 5.0), sharey=False)
    if n_h == 1:
        axes = [axes]

    seen: List[str] = []
    for ax, h in zip(axes, horizons):
        for m in models:
            row = df[(df["model"] == m) & (df["horizon_steps"] == h)]
            if row.empty:
                continue
            mape = row.iloc[0]["MAPE_pct"]
            if pd.isna(mape):
                continue
            fit_d  = float(dur_first.loc[m, "fit_duration_s"]  or 0) if m in dur_first.index else 0
            eval_d = float(dur_first.loc[m, "eval_duration_s"] or 0) if m in dur_first.index else 0
            total_s = fit_d + eval_d
            if total_s <= 0:
                continue

            c       = _color(m)
            marker  = "^" if m in TSFM_KEYS else "o"
            ax.scatter(total_s, mape, color=c, s=130, marker=marker,
                       zorder=5, edgecolors="white", linewidths=0.8)
            if m not in seen:
                seen.append(m)

        ax.set_xlabel("Total wall-clock time (s, log scale)", fontsize=9)
        ax.set_ylabel("MAPE (%)", fontsize=9)
        ax.set_title(f"Accuracy–duration trade-off — {HORIZON_LABEL.get(h, f'h={h}')}", fontsize=10)
        ax.set_xscale("log")

    legend_els = [
        Line2D([0], [0], marker=("^" if m in TSFM_KEYS else "o"),
               color="none", markerfacecolor=_color(m),
               markeredgecolor="white", markersize=8,
               label=_label(m))
        for m in seen
    ]
    legend_els.append(Line2D([0], [0], marker="None", color="none", label="—"))
    legend_els.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
               markersize=7, label="Classical")
    )
    legend_els.append(
        Line2D([0], [0], marker="^", color="none", markerfacecolor="gray",
               markersize=7, label="Foundation (zero-shot)")
    )
    axes[-1].legend(handles=legend_els, fontsize=7.5, framealpha=0.85,
                    loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    title="model", title_fontsize=8)

    fig.suptitle("Accuracy–computation time trade-off (all models — lower-left = better)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "08_duration_tradeoff.png", dpi)


# --- Summary table ---

def print_summary(df: pd.DataFrame) -> None:
    models   = _models(df)
    horizons = _horizons(df)

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Model':<12} {'h':>4} {'h(min)':>6}  {'MAE':>8}  {'MAPE%':>7}  "
          f"{'CC_kWh':>12}  {'CT_kWh':>12}  {'duration':>9}")
    print("-" * 90)

    h1 = df.groupby("model")[
        ["fit_cc_energy_kwh", "eval_cc_energy_kwh",
         "fit_ct_energy_kwh", "eval_ct_energy_kwh",
         "fit_duration_s", "eval_duration_s"]
    ].first()

    for m in models:
        for h in horizons:
            row = df[(df["model"] == m) & (df["horizon_steps"] == h)]
            if row.empty:
                continue
            r = row.iloc[0]
            e_row = h1.loc[m] if m in h1.index else {}
            cc = (e_row.get("fit_cc_energy_kwh", 0) or 0) + (e_row.get("eval_cc_energy_kwh", 0) or 0)
            ct = (e_row.get("fit_ct_energy_kwh", 0) or 0) + (e_row.get("eval_ct_energy_kwh", 0) or 0)
            dur = (e_row.get("fit_duration_s", 0) or 0) + (e_row.get("eval_duration_s", 0) or 0)
            dur_str = f"{dur:.0f}s" if dur < 120 else f"{dur/60:.1f}m"
            cc_str = f"{cc*1e6:.3f}μWh" if cc > 0 else "n/a"
            ct_str = f"{ct*1e6:.3f}μWh" if ct > 0 else "n/a"
            print(f"{m:<12} {h:>4} {h*15:>6}  "
                  f"{r['MAE']:>8.1f}  {r['MAPE_pct']:>7.2f}  "
                  f"{cc_str:>12}  {ct_str:>12}  {dur_str:>9}")
        print()

    print("=" * 90)


# --- CLI ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Energy + accuracy visualisation for the walk-forward benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default="results",
                   help="Directory containing wf_*/metrics.csv files.")
    p.add_argument("--run_id", default=None,
                   help="Specific run ID (e.g. wf_20260407_122418). "
                        "Defaults to the most recent run with energy data.")
    p.add_argument("--out_dir", default=None,
                   help="Output directory. Defaults to figures/energy_<timestamp>/.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or os.path.join(
        "figures", "energy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput folder: {out_dir}")

    print("\nLoading metrics …")
    raw = _load_run(args.results_dir, args.run_id)
    df  = _prep(raw)

    print_summary(df)

    print("\nGenerating figures …")
    plot_accuracy(df, out_dir, args.dpi)
    plot_energy_cc_vs_ct(df, out_dir, args.dpi)
    plot_energy_breakdown(df, out_dir, args.dpi)
    plot_co2(df, out_dir, args.dpi)
    plot_energy_per_prediction(df, out_dir, args.dpi)
    plot_tradeoff(df, out_dir, args.dpi)
    plot_duration(df, out_dir, args.dpi)
    plot_duration_tradeoff(df, out_dir, args.dpi)

    print(f"\nDone. {len(os.listdir(out_dir))} figures in {out_dir}/")


if __name__ == "__main__":
    main()
