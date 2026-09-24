"""Shared plot palette — colors, labels, ordering, hatches for all figures."""
from __future__ import annotations

from typing import Dict, List, Set


MODEL_COLOR: Dict[str, str] = {
    # Classical
    "naive":             "#6E6E6E",
    "arima":             "#C62828",
    "sarima":            "#EF6C00",
    "linear":            "#1565C0",
    "lgbm":              "#2E7D32",

    # Chronos T5
    "chronos_mini":      "#AB47BC",
    "chronos_large":     "#4A148C",

    # Chronos Bolt
    "chronos_bolt_mini": "#FBC02D",
    "chronos_bolt_base": "#827717",

    # TimesFM
    "timesfm_200m":      "#26C6DA",
    "timesfm_500m":      "#006064",

    # Moirai-1.0-R
    "moirai_small":      "#A1887F",
    "moirai_base":       "#6D4C41",
    "moirai_large":      "#3E2723",

    # Moirai-2.0-R
    "moirai2_small":     "#F9A825",

    # Lag-Llama
    "lag_llama":         "#EC407A",
}


MODEL_LABEL: Dict[str, str] = {
    "naive":             "Seasonal Naive",
    "arima":             "ARIMA",
    "sarima":            "SARIMA",
    "linear":            "Ridge",
    "lgbm":              "LightGBM",
    "chronos_mini":      "Chronos T5-mini",
    "chronos_large":     "Chronos T5-large",
    "chronos_bolt_mini": "Chronos-Bolt mini",
    "chronos_bolt_base": "Chronos-Bolt base",
    "timesfm_200m":      "TimesFM 200M",
    "timesfm_500m":      "TimesFM 500M",
    "moirai_small":      "Moirai-small",
    "moirai_base":       "Moirai-base",
    "moirai_large":      "Moirai-large",
    "moirai2_small":     "Moirai-2.0-small",
    "lag_llama":         "Lag-Llama",
}


MODEL_ORDER: List[str] = [
    "naive", "arima", "sarima", "linear", "lgbm",
    "chronos_mini", "chronos_large",
    "chronos_bolt_mini", "chronos_bolt_base",
    "timesfm_200m", "timesfm_500m",
    "moirai_small", "moirai_base", "moirai_large",
    "moirai2_small",
    "lag_llama",
]


# Hatch per forecast horizon (used by accuracy bars to encode horizon).
HORIZON_HATCH: Dict[int, str] = {1: "", 4: "///", 96: "xxx"}
HORIZON_LABEL: Dict[int, str] = {1: "15 min", 4: "1 hour", 96: "24 hours"}


BACKEND_HATCH: Dict[str, str] = {
    "CC": "...",   # dots
    "CT": "///",   # diagonal
    "NV": "xxx",   # cross
}


TSFM_KEYS: Set[str] = {
    "chronos_mini", "chronos_large",
    "chronos_bolt_mini", "chronos_bolt_base",
    "timesfm_200m", "timesfm_500m",
    "moirai_small", "moirai_base", "moirai_large",
    "moirai2_small",
    "lag_llama",
}


ANNOT_COLOR: str = "#424242"
