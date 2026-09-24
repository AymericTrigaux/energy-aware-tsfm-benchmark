"""Side-by-side comparison of model accuracy and energy across two datasets.

Reads metrics.csv from run folders and produces RMSE, skill-score, energy,
and Pareto plots. Also supports a compact MASE+energy table mode (--table).

Usage:
    python compare_datasets.py \\
        --dataset_a Elia results/classical/wf_energy_full_20260512 \\
        --dataset_b ETTh1 results/classical/wf_20260514_171100 results/foundation_models/tsfm_20260514_171217
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.palette import (
    MODEL_COLOR as MODEL_COLORS,
    MODEL_LABEL as MODEL_DISPLAY,
    MODEL_ORDER,
)


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
    "grid.alpha":        0.30,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
})


# --- Metrics loading ---

def load_metrics(run_dirs: List[str]) -> pd.DataFrame:
    """Concatenate the metrics.csv files in `run_dirs` into one DataFrame."""
    frames = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "metrics.csv"
        if not path.exists():
            print(f"  [warn] {path} not found — skipping")
            continue
        df = pd.read_csv(path)
        df["__source"] = str(run_dir)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics.csv found in {run_dirs}")
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].astype(str).str.strip().str.lower()
    return df


def _energy_wh(row: pd.Series) -> float:
    """Sum total energy for one (model × horizon) row in Wh.

    Prefers CodeCarbon when available, falls back to CarbonTracker otherwise.
    The metrics.csv stores energy in kWh, so multiply by 1000.
    """
    fit_kwh = row.get("fit_cc_energy_kwh")
    eval_kwh = row.get("eval_cc_energy_kwh")
    if pd.isna(fit_kwh) and pd.isna(eval_kwh):
        fit_kwh = row.get("fit_ct_energy_kwh")
        eval_kwh = row.get("eval_ct_energy_kwh")
    total = 0.0
    for v in (fit_kwh, eval_kwh):
        if v is not None and not pd.isna(v):
            total += float(v)
    return total * 1000.0


def _duration_s(row: pd.Series) -> float:
    """Total fit + eval wall-clock in seconds."""
    s = 0.0
    for col in ("fit_duration_s", "eval_duration_s"):
        v = row.get(col)
        if v is not None and not pd.isna(v):
            s += float(v)
    return s


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add energy_wh, duration_s, n_predictions (typed)."""
    out = df.copy()
    out["energy_wh"]   = out.apply(_energy_wh, axis=1)
    out["duration_s"]  = out.apply(_duration_s, axis=1)
    out["n_predictions"] = pd.to_numeric(out.get("n_predictions"), errors="coerce")
    out["RMSE"]        = pd.to_numeric(out["RMSE"], errors="coerce")
    out["MAE"]         = pd.to_numeric(out["MAE"], errors="coerce")
    return out


def per_model_energy(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model. energy_wh, duration_s, n_predictions (per horizon)."""
    g = df.groupby("model").agg(
        energy_wh=("energy_wh", "first"),
        duration_s=("duration_s", "first"),
        n_predictions=("n_predictions", "first"),
    )
    return g.reset_index()


def skill_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return (model, horizon_steps, skill) with skill = 1 - RMSE_m / RMSE_naive."""
    rows = []
    for h, sub in df.groupby("horizon_steps"):
        naive_row = sub[sub["model"] == "naive"]
        if naive_row.empty:
            continue
        rmse_naive = float(naive_row["RMSE"].iloc[0])
        for _, r in sub.iterrows():
            skill = 1.0 - float(r["RMSE"]) / rmse_naive
            rows.append({"model": r["model"], "horizon_steps": int(h),
                         "skill": skill, "rmse": float(r["RMSE"])})
    return pd.DataFrame(rows)


# --- Common helpers ---

def order_models(models: List[str]) -> List[str]:
    """Return models sorted by MODEL_ORDER, unknowns appended alphabetically."""
    known = [m for m in MODEL_ORDER if m in models]
    extra = sorted(m for m in models if m not in MODEL_ORDER)
    return known + extra


def common_models(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    """Intersection of models present in both datasets, in canonical order."""
    return order_models(list(set(df_a["model"]) & set(df_b["model"])))


def common_horizons_by_minute(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> List[Tuple[int, int, int]]:
    """Pair horizons by their physical duration in minutes.

    Returns a list of ``(minutes, h_steps_a, h_steps_b)`` for every duration
    that exists in both datasets.
    """
    a = {int(m): int(h) for m, h in
         zip(df_a["horizon_minutes"], df_a["horizon_steps"])}
    b = {int(m): int(h) for m, h in
         zip(df_b["horizon_minutes"], df_b["horizon_steps"])}
    shared = sorted(set(a) & set(b))
    return [(m, a[m], b[m]) for m in shared]


def horizon_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} min"
    if minutes < 24 * 60:
        return f"{minutes // 60} h"
    days = minutes / (24 * 60)
    if abs(days - round(days)) < 0.01:
        return f"{int(round(days))} d"
    return f"{days:.1f} d"


# --- Plot 1 — Raw RMSE per model × horizon ---

def plot_rmse_per_model(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    name_a: str, name_b: str,
    horizon_pairs: List[Tuple[int, int, int]],
    models: List[str],
    out_path: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"RMSE per model × horizon — {name_a} vs {name_b}", fontsize=13)

    n_h = len(horizon_pairs)
    bar_w = 0.8 / max(1, n_h)
    HATCHES = ["", "///", "xxx", "...", "\\\\\\"]

    for ax, df, dataset_name, h_key in zip(
        axes,
        [df_a, df_b],
        [name_a, name_b],
        [1, 2],  # 1 = use h_a, 2 = use h_b from the pair
    ):
        x = np.arange(len(models))
        for hi, (minutes, h_a, h_b) in enumerate(horizon_pairs):
            h = h_a if h_key == 1 else h_b
            vals = []
            for mk in models:
                row = df[(df["model"] == mk) & (df["horizon_steps"] == h)]
                vals.append(float(row["RMSE"].iloc[0]) if not row.empty else np.nan)
            xs = x + (hi - (n_h - 1) / 2.0) * bar_w
            for xi, v in enumerate(vals):
                if np.isnan(v):
                    continue
                ax.bar(xs[xi], v, width=bar_w,
                       color=MODEL_COLORS.get(models[xi], "#555"),
                       hatch=HATCHES[hi % len(HATCHES)],
                       edgecolor="white", linewidth=0.5,
                       alpha=0.92, zorder=3)
        ax.set_title(f"{dataset_name}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m) for m in models],
            rotation=30, ha="right", fontsize=8,
        )
        for tick, mk in zip(ax.get_xticklabels(), models):
            tick.set_color(MODEL_COLORS.get(mk, "#555"))
            tick.set_fontweight("bold")
        ax.set_ylabel("RMSE (native units)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

    from matplotlib.patches import Patch as _Patch
    legend_els = [
        _Patch(facecolor="lightgray", hatch=HATCHES[hi % len(HATCHES)],
               edgecolor="white", label=horizon_label(m))
        for hi, (m, _, _) in enumerate(horizon_pairs)
    ]
    fig.legend(handles=legend_els, loc="upper center", ncol=len(horizon_pairs),
               fontsize=9, bbox_to_anchor=(0.5, 0.96), framealpha=0.9,
               title="horizon (pattern)", title_fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# --- Plot 2 — Skill score relative to naive ---

def plot_skill_score(
    skill_a: pd.DataFrame, skill_b: pd.DataFrame,
    name_a: str, name_b: str,
    horizon_pairs: List[Tuple[int, int, int]],
    models: List[str],
    out_path: str,
    dpi: int,
) -> None:
    """Grouped bars: x = model, hue = dataset, one panel per horizon."""
    n_h = len(horizon_pairs)
    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 5), sharey=True)
    if n_h == 1:
        axes = [axes]
    fig.suptitle(
        f"Skill score vs Naive baseline (higher = better) — {name_a} vs {name_b}",
        fontsize=13,
    )

    # Drop naive itself (skill is 0 by construction)
    models_no_naive = [m for m in models if m != "naive"]

    for ax, (minutes, h_a, h_b) in zip(axes, horizon_pairs):
        x = np.arange(len(models_no_naive))
        bar_w = 0.38
        vals_a = [
            float(skill_a[(skill_a["model"] == m) &
                          (skill_a["horizon_steps"] == h_a)]["skill"].iloc[0])
            if not skill_a[(skill_a["model"] == m) &
                           (skill_a["horizon_steps"] == h_a)].empty
            else np.nan
            for m in models_no_naive
        ]
        vals_b = [
            float(skill_b[(skill_b["model"] == m) &
                          (skill_b["horizon_steps"] == h_b)]["skill"].iloc[0])
            if not skill_b[(skill_b["model"] == m) &
                           (skill_b["horizon_steps"] == h_b)].empty
            else np.nan
            for m in models_no_naive
        ]
        for i, mk in enumerate(models_no_naive):
            c = MODEL_COLORS.get(mk, "#555")
            if not np.isnan(vals_a[i]):
                ax.bar(x[i] - bar_w / 2, vals_a[i], width=bar_w,
                       color=c, edgecolor="white", linewidth=0.5,
                       alpha=0.95, zorder=3)
            if not np.isnan(vals_b[i]):
                ax.bar(x[i] + bar_w / 2, vals_b[i], width=bar_w,
                       color=c, edgecolor="white", linewidth=0.5,
                       hatch="///", alpha=0.95, zorder=3)
        ax.axhline(0, color="black", lw=0.8, ls="-", zorder=2)
        ax.set_title(f"horizon = {horizon_label(minutes)}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m) for m in models_no_naive],
            rotation=30, ha="right", fontsize=8,
        )
        for tick, mk in zip(ax.get_xticklabels(), models_no_naive):
            tick.set_color(MODEL_COLORS.get(mk, "#555"))
            tick.set_fontweight("bold")
        ax.set_ylabel("Skill score  (1 − RMSE/RMSE_naive)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

    from matplotlib.patches import Patch as _Patch
    handles = [
        _Patch(facecolor="lightgray", edgecolor="black", label=name_a),
        _Patch(facecolor="lightgray", edgecolor="black", hatch="///", label=name_b),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.96), framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# --- Plot 3 — Total energy per model ---

def plot_energy_per_model(
    ema_a: pd.DataFrame, ema_b: pd.DataFrame,
    name_a: str, name_b: str,
    models: List[str],
    out_path: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle(f"Total energy per model (Wh, log scale) — {name_a} vs {name_b}",
                 fontsize=13)

    for ax, ema, dataset_name in zip(axes, [ema_a, ema_b], [name_a, name_b]):
        x = np.arange(len(models))
        vals = []
        for mk in models:
            row = ema[ema["model"] == mk]
            vals.append(float(row["energy_wh"].iloc[0]) if not row.empty else 0.0)
        for i, (mk, v) in enumerate(zip(models, vals)):
            c = MODEL_COLORS.get(mk, "#555")
            # Tiny floor so log-scale shows zero/near-zero bars
            display = max(v, 1e-3)
            ax.bar(x[i], display, color=c, edgecolor="white",
                   linewidth=0.5, alpha=0.95, zorder=3)
            ax.text(x[i], display, f"{v:.2g}", ha="center", va="bottom",
                    fontsize=7)
        ax.set_yscale("log")
        ax.set_title(dataset_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m) for m in models],
            rotation=30, ha="right", fontsize=8,
        )
        for tick, mk in zip(ax.get_xticklabels(), models):
            tick.set_color(MODEL_COLORS.get(mk, "#555"))
            tick.set_fontweight("bold")
        ax.set_ylabel("Total energy (Wh, log)", fontsize=9)
        ax.grid(True, axis="y", which="both", alpha=0.20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# --- Plot 4 — Energy per forecasted point ---

def plot_energy_per_prediction(
    ema_a: pd.DataFrame, ema_b: pd.DataFrame,
    name_a: str, name_b: str,
    models: List[str],
    out_path: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle(
        f"Energy per forecasted point  (Wh / origin × horizon, log scale) — "
        f"{name_a} vs {name_b}",
        fontsize=13,
    )

    for ax, ema, dataset_name in zip(axes, [ema_a, ema_b], [name_a, name_b]):
        x = np.arange(len(models))
        for i, mk in enumerate(models):
            row = ema[ema["model"] == mk]
            if row.empty:
                continue
            wh = float(row["energy_wh"].iloc[0])
            n  = float(row["n_predictions"].iloc[0])
            if n <= 0:
                continue
            per_pred = wh / n
            c = MODEL_COLORS.get(mk, "#555")
            display = max(per_pred, 1e-6)
            ax.bar(x[i], display, color=c, edgecolor="white",
                   linewidth=0.5, alpha=0.95, zorder=3)
            ax.text(x[i], display, f"{per_pred:.2g}", ha="center", va="bottom",
                    fontsize=7)
        ax.set_yscale("log")
        ax.set_title(dataset_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m) for m in models],
            rotation=30, ha="right", fontsize=8,
        )
        for tick, mk in zip(ax.get_xticklabels(), models):
            tick.set_color(MODEL_COLORS.get(mk, "#555"))
            tick.set_fontweight("bold")
        ax.set_ylabel("Wh per prediction (log)", fontsize=9)
        ax.grid(True, axis="y", which="both", alpha=0.20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# --- Plot 5 — Pareto: skill vs total energy ---

def plot_pareto(
    skill_a: pd.DataFrame, skill_b: pd.DataFrame,
    ema_a: pd.DataFrame, ema_b: pd.DataFrame,
    name_a: str, name_b: str,
    horizon_pairs: List[Tuple[int, int, int]],
    models: List[str],
    out_path: str,
    dpi: int,
) -> None:
    """Energy (x) vs skill score (y).  One marker shape per dataset, one
    sub-marker color per model, one per horizon (annotated). Linked points for
    same model are connected with a thin line.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(
        f"Accuracy–energy trade-off — {name_a} ●  vs  {name_b} ◆",
        fontsize=13,
    )

    def _plot_set(skill_df, ema_df, marker: str, label: str):
        for mk in models:
            if mk == "naive":
                continue
            row_e = ema_df[ema_df["model"] == mk]
            if row_e.empty:
                continue
            wh = float(row_e["energy_wh"].iloc[0])
            wh = max(wh, 1e-3)
            c = MODEL_COLORS.get(mk, "#555")

            xs, ys = [], []
            for minutes, h_a, h_b in horizon_pairs:
                h = h_a if label == name_a else h_b
                s_row = skill_df[(skill_df["model"] == mk) &
                                 (skill_df["horizon_steps"] == h)]
                if s_row.empty:
                    continue
                xs.append(wh)
                ys.append(float(s_row["skill"].iloc[0]))
            if xs:
                ax.plot(xs, ys, color=c, lw=0.6, alpha=0.5, zorder=2)
                ax.scatter(xs, ys, s=70, color=c, marker=marker,
                           edgecolor="black", linewidth=0.5, alpha=0.95,
                           zorder=5,
                           label=f"{MODEL_DISPLAY.get(mk, mk)} ({label})")

    _plot_set(skill_a, ema_a, marker="o", label=name_a)
    _plot_set(skill_b, ema_b, marker="D", label=name_b)

    ax.set_xscale("log")
    ax.set_xlabel("Total energy (Wh, log)", fontsize=10)
    ax.set_ylabel("Skill score  (1 − RMSE/RMSE_naive)", fontsize=10)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.7)
    ax.grid(True, alpha=0.25)

    # Build a clean two-column legend: model color (left) + dataset marker (right)
    from matplotlib.lines import Line2D
    color_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=MODEL_COLORS.get(m, "#555"),
               markersize=10, label=MODEL_DISPLAY.get(m, m))
        for m in models if m != "naive"
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor="lightgray",
               markersize=10, lw=0, label=name_a),
        Line2D([0], [0], marker="D", color="black", markerfacecolor="lightgray",
               markersize=10, lw=0, label=name_b),
    ]
    leg1 = ax.legend(handles=color_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=8,
                     title="Model", title_fontsize=8, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="upper left",
              bbox_to_anchor=(1.02, 0.30), fontsize=8,
              title="Dataset", title_fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path}")


# --- Summary table ---

def print_summary(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    name_a: str, name_b: str,
    horizon_pairs: List[Tuple[int, int, int]],
    models: List[str],
) -> None:
    print()
    print("=" * 92)
    print(f"  CROSS-DATASET SUMMARY  —  {name_a} vs {name_b}")
    print("=" * 92)
    header = f"{'Model':<20} {'Horizon':<10}"
    for n in (name_a, name_b):
        header += f"{n+' RMSE':>14}{n+' Skill':>14}"
    print(header)
    print("-" * 92)

    skill_a = skill_score(df_a)
    skill_b = skill_score(df_b)

    for mk in models:
        for minutes, h_a, h_b in horizon_pairs:
            row_a = df_a[(df_a["model"] == mk) & (df_a["horizon_steps"] == h_a)]
            row_b = df_b[(df_b["model"] == mk) & (df_b["horizon_steps"] == h_b)]
            sa = skill_a[(skill_a["model"] == mk) & (skill_a["horizon_steps"] == h_a)]
            sb = skill_b[(skill_b["model"] == mk) & (skill_b["horizon_steps"] == h_b)]
            rmse_a = float(row_a["RMSE"].iloc[0]) if not row_a.empty else float("nan")
            rmse_b = float(row_b["RMSE"].iloc[0]) if not row_b.empty else float("nan")
            ssa    = float(sa["skill"].iloc[0])  if not sa.empty   else float("nan")
            ssb    = float(sb["skill"].iloc[0])  if not sb.empty   else float("nan")
            print(f"{MODEL_DISPLAY.get(mk, mk):<20} {horizon_label(minutes):<10}"
                  f"{rmse_a:>14.3f}{ssa:>14.3f}"
                  f"{rmse_b:>14.3f}{ssb:>14.3f}")
        print()
    print("=" * 92)


# --- Compact table mode ---

def _tab_total_energy_wh(row: pd.Series) -> float:
    cc_fit  = row.get("fit_cc_energy_kwh")
    cc_eval = row.get("eval_cc_energy_kwh")
    if pd.isna(cc_fit) and pd.isna(cc_eval):
        cc_fit  = row.get("fit_ct_energy_kwh")
        cc_eval = row.get("eval_ct_energy_kwh")
    e = 0.0
    for v in (cc_fit, cc_eval):
        if v is not None and not pd.isna(v):
            e += float(v)
    return e * 1000.0


def _tab_mase_scale(data_path: str, target_col: str, freq: str,
                    naive_s: int, test_years: float,
                    timestamp_col: str = "datetime") -> float:
    from src.data import load_timeseries, clean_timeseries, split_last_n_years
    df = load_timeseries(data_path, timestamp_col=timestamp_col,
                         target_col=target_col, freq=freq)
    df = clean_timeseries(df, impute_method="interpolate")
    df_train, _ = split_last_n_years(df, test_years)
    y = df_train["y"].to_numpy(dtype=float)
    if len(y) <= naive_s:
        raise ValueError(f"Train set too short ({len(y)}) for seasonal_s={naive_s}.")
    return float(np.mean(np.abs(y[naive_s:] - y[:-naive_s])))


def _tab_h_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} min"
    if minutes < 24 * 60:
        return f"{minutes // 60} h"
    days = minutes / (24 * 60)
    return f"{int(round(days))} d" if abs(days - round(days)) < 0.01 else f"{days:.1f} d"


def _tab_cell(df: pd.DataFrame, model: str, h_minutes: int,
              mase_denom: float) -> dict:
    sub = df[(df["model"] == model) & (df["horizon_minutes"] == h_minutes)]
    if sub.empty:
        return {"MAE": float("nan"), "MASE": float("nan"),
                "energy_per_pred_wh": float("nan"), "n": 0}
    row = sub.iloc[0]
    e_wh = _tab_total_energy_wh(row)
    n    = float(row["n_predictions"])
    mae  = float(row["MAE"])
    return {
        "MAE":  mae,
        "MASE": mae / mase_denom if mase_denom > 0 else float("nan"),
        "energy_per_pred_wh": e_wh / n if n > 0 else float("nan"),
        "n":    int(n),
    }


def _tab_load(run_dirs: list) -> pd.DataFrame:
    frames = []
    for d in run_dirs:
        path = Path(d) / "metrics.csv"
        if not path.exists():
            print(f"  [warn] {path} not found — skipping")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"No metrics.csv in {run_dirs}")
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["model"].str.strip().str.lower()
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    df["n_predictions"] = pd.to_numeric(df["n_predictions"], errors="coerce")
    return df


def _main_table(args: argparse.Namespace) -> None:
    name_a, *runs_a = args.dataset_a
    name_b, *runs_b = args.dataset_b
    if not runs_a or not runs_b:
        raise SystemExit("Each --dataset_X needs a name and at least one run dir.")

    print(f"\nLoading {name_a} from {runs_a}")
    df_a = _tab_load(runs_a)
    print(f"Loading {name_b} from {runs_b}")
    df_b = _tab_load(runs_b)

    print("\nComputing MASE scales …")
    scale_a = _tab_mase_scale(args.data_a, args.target_a, args.freq_a,
                               args.naive_s_a, args.test_years_a)
    scale_b = _tab_mase_scale(args.data_b, args.target_b, args.freq_b,
                               args.naive_s_b, args.test_years_b)
    print(f"  {name_a}: scale = {scale_a:.4f}  (s={args.naive_s_a})")
    print(f"  {name_b}: scale = {scale_b:.4f}  (s={args.naive_s_b})")

    models          = args.models
    horizons_minutes = args.horizons_minutes

    # Print table
    rows = []
    width = 108
    print(); print("=" * width)
    print(f"  Cross-dataset robustness check  —  {name_a} vs {name_b}")
    print(f"  MASE scales: {name_a}={scale_a:.4f}   {name_b}={scale_b:.4f}")
    print("=" * width)
    header = (f"{'Model':<20}{'Horizon':<10}"
              f"{name_a+' MASE':>14}{name_b+' MASE':>14}"
              f"{name_a+' Wh/pred':>17}{name_b+' Wh/pred':>17}")
    print(header); print("-" * width)
    for mk in models:
        for h in horizons_minutes:
            ca = _tab_cell(df_a, mk, h, scale_a)
            cb = _tab_cell(df_b, mk, h, scale_b)
            hl = _tab_h_label(h)
            print(f"{MODEL_DISPLAY.get(mk, mk):<20}{hl:<10}"
                  f"{ca['MASE']:>14.4f}{cb['MASE']:>14.4f}"
                  f"{ca['energy_per_pred_wh']:>17.3e}{cb['energy_per_pred_wh']:>17.3e}")
            rows.append({
                "model": mk, "horizon_minutes": h,
                f"{name_a}_MAE": ca["MAE"], f"{name_b}_MAE": cb["MAE"],
                f"{name_a}_MASE": ca["MASE"], f"{name_b}_MASE": cb["MASE"],
                f"{name_a}_Wh_per_pred": ca["energy_per_pred_wh"],
                f"{name_b}_Wh_per_pred": cb["energy_per_pred_wh"],
            })
    print("=" * width)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"robustness_{name_a}_vs_{name_b}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "summary_table.csv", index=False)
    print(f"  saved {out_dir / 'summary_table.csv'}")

    # Figure
    from matplotlib.patches import Patch as _Patch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Robustness check — {name_a} vs {name_b}", fontsize=13)
    labels  = [f"{MODEL_DISPLAY.get(m, m)}\n{_tab_h_label(h)}"
               for m in models for h in horizons_minutes]
    colors  = [MODEL_COLORS.get(m, "#555") for m in models for h in horizons_minutes]
    x = np.arange(len(labels))
    bar_w = 0.38
    for ax, key, ylabel, ylog in [
        (axes[0], "MASE",               "MASE  (MAE / in-sample seasonal-naive MAE)", False),
        (axes[1], "energy_per_pred_wh", "Energy per prediction (Wh)",                  True),
    ]:
        vals_a, vals_b = [], []
        for m in models:
            for h in horizons_minutes:
                vals_a.append(_tab_cell(df_a, m, h, scale_a)[key])
                vals_b.append(_tab_cell(df_b, m, h, scale_b)[key])
        for i, (va, vb, c) in enumerate(zip(vals_a, vals_b, colors)):
            if not np.isnan(va):
                ax.bar(x[i] - bar_w / 2, va, width=bar_w,
                       color=c, edgecolor="white", linewidth=0.5, zorder=3)
                ax.text(x[i] - bar_w / 2, va, f"{va:.2g}",
                        ha="center", va="bottom", fontsize=7)
            if not np.isnan(vb):
                ax.bar(x[i] + bar_w / 2, vb, width=bar_w,
                       color=c, hatch="///", edgecolor="white", linewidth=0.5, zorder=3)
                ax.text(x[i] + bar_w / 2, vb, f"{vb:.2g}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center")
        for tick, c in zip(ax.get_xticklabels(), colors):
            tick.set_color(c); tick.set_fontweight("bold")
        ax.set_ylabel(ylabel, fontsize=9)
        if ylog:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25, which="both" if ylog else "major")
    handles = [_Patch(facecolor="lightgray", edgecolor="black", label=name_a),
               _Patch(facecolor="lightgray", edgecolor="black", hatch="///", label=name_b)]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.95), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / "robustness.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'robustness.png'}")
    print(f"\nDone. Output in {out_dir}/\n")


# --- CLI + main ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset_a", nargs="+", required=True,
                   metavar="NAME RUN_DIR [RUN_DIR ...]",
                   help="Display name followed by one or more run folders.")
    p.add_argument("--dataset_b", nargs="+", required=True,
                   metavar="NAME RUN_DIR [RUN_DIR ...]",
                   help="Display name followed by one or more run folders.")
    p.add_argument("--out", default="figures",
                   help="Base output directory. A timestamped sub-folder is created.")
    p.add_argument("--dpi", type=int, default=150)

    # Compact table mode (absorbs compare_datasets_table.py)
    p.add_argument("--table", action="store_true",
                   help="Run compact MASE + energy-per-prediction table instead of full plots.")
    p.add_argument("--models", nargs="+", default=["naive", "lgbm", "chronos_bolt_mini"],
                   help="Models to include in table mode.")
    p.add_argument("--horizons_minutes", type=int, nargs="+", default=[60, 1440],
                   help="Horizons in minutes to include in table mode.")
    p.add_argument("--data_a",       default="data/processed/elia_load_15min.csv")
    p.add_argument("--target_a",     default="totalload")
    p.add_argument("--freq_a",       default="15min")
    p.add_argument("--naive_s_a",    type=int,   default=96)
    p.add_argument("--test_years_a", type=float, default=1.0)
    p.add_argument("--data_b",       default="data/processed/etth1.csv")
    p.add_argument("--target_b",     default="ot")
    p.add_argument("--freq_b",       default="h")
    p.add_argument("--naive_s_b",    type=int,   default=24)
    p.add_argument("--test_years_b", type=float, default=0.4)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.table:
        _main_table(args)
        return

    name_a, *runs_a = args.dataset_a
    name_b, *runs_b = args.dataset_b
    if not runs_a or not runs_b:
        raise SystemExit("Each --dataset_X needs a name followed by ≥1 run folder.")

    print(f"\nLoading {name_a} metrics from: {runs_a}")
    df_a = enrich(load_metrics(runs_a))
    print(f"Loading {name_b} metrics from: {runs_b}")
    df_b = enrich(load_metrics(runs_b))

    models = common_models(df_a, df_b)
    pairs  = common_horizons_by_minute(df_a, df_b)
    if not models:
        raise SystemExit("No models in common between the two datasets.")
    if not pairs:
        raise SystemExit("No horizon (in minutes) common between the two datasets.")

    print(f"\n  Models   : {models}")
    print(f"  Horizons : {[(m, ha, hb) for m, ha, hb in pairs]}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"compare_{name_a}_vs_{name_b}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output   : {out_dir}\n")

    skill_a = skill_score(df_a)
    skill_b = skill_score(df_b)
    ema_a   = per_model_energy(df_a)
    ema_b   = per_model_energy(df_b)

    print("Generating plots ...")
    plot_rmse_per_model(df_a, df_b, name_a, name_b, pairs, models,
                        str(out_dir / "rmse_per_model.png"), args.dpi)
    plot_skill_score(skill_a, skill_b, name_a, name_b, pairs, models,
                     str(out_dir / "skill_score_vs_naive.png"), args.dpi)
    plot_energy_per_model(ema_a, ema_b, name_a, name_b, models,
                          str(out_dir / "energy_per_model.png"), args.dpi)
    plot_energy_per_prediction(ema_a, ema_b, name_a, name_b, models,
                                str(out_dir / "energy_per_prediction.png"), args.dpi)
    plot_pareto(skill_a, skill_b, ema_a, ema_b, name_a, name_b, pairs, models,
                str(out_dir / "pareto_energy_vs_skill.png"), args.dpi)

    print_summary(df_a, df_b, name_a, name_b, pairs, models)
    print(f"\nDone. Figures saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
