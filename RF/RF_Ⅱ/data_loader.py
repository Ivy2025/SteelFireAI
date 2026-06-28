from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TARGET_COLUMNS: List[str] = ["fy_reduction", "fu_reduction", "E_reduction"]
FEATURE_COLUMNS: List[str] = [
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
AUXILIARY_COLUMNS: List[str] = ["diff_steel"]


def _validate_columns(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + TARGET_COLUMNS + AUXILIARY_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load CSV and add sample_id based on original row index."""
    df = pd.read_csv(csv_path)
    _validate_columns(df)

    # Coerce numeric fields so blanks like " " become NaN.
    numeric_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use raw row position as a stable sample identifier.
    df = df.copy()
    df["sample_id"] = np.arange(len(df), dtype=int)
    return df, FEATURE_COLUMNS.copy(), TARGET_COLUMNS.copy()


def get_target_data(
    df: pd.DataFrame,
    target: str,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Drop missing target rows and return X, y, sample_id, diff_steel."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target: {target}")

    target_df = df.dropna(subset=[target]).copy()
    if target_df.empty:
        raise ValueError(f"No valid rows available after dropping NaN in target '{target}'")

    x = target_df[feature_columns].copy()
    y = target_df[target].astype(float).copy()
    sample_id = target_df["sample_id"].copy()
    diff_steel = target_df["diff_steel"].copy()
    return x, y, sample_id, diff_steel
