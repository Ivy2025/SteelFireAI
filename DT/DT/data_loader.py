from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


TARGET_COLUMNS = ["fy_reduction", "fu_reduction", "E_reduction"]
FEATURE_COLUMNS = [
    "N",
    "HR",
    "TMCP",
    "TMCP_T",
    "QT",
    "CF",
    "C",
    "Mn",
    "Si",
    "Cr",
    "Mo",
    "Nb",
    "V",
    "Ti",
    "Ni",
    "Al",
    "Cu",
    "P",
    "S",
    "temperature",
]
GROUP_COLUMN = "diff_steel"


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    raw_subset: pd.DataFrame


def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(csv_path, na_values=["", " ", "  "])


def _coerce_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert selected columns to numeric, coercing invalid text to NaN."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def prepare_target_data(df: pd.DataFrame, target: str) -> PreparedData:
    """Prepare feature matrix, target vector and groups for a single target."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target: {target}")

    required = [GROUP_COLUMN] + FEATURE_COLUMNS + TARGET_COLUMNS
    _validate_columns(df, required)

    # Ensure model inputs/targets are numeric before imputation and fitting.
    subset_all = _coerce_numeric_columns(df.copy(), FEATURE_COLUMNS + TARGET_COLUMNS)

    subset = subset_all.dropna(subset=[target]).copy()
    if subset.empty:
        raise ValueError(f"No samples left after dropping missing values for target: {target}")

    X = subset[FEATURE_COLUMNS].copy()
    y = subset[target].copy()
    groups = subset[GROUP_COLUMN].copy()

    return PreparedData(X=X, y=y, groups=groups, raw_subset=subset)
