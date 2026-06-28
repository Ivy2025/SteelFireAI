from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


PROCESS_FEATURES = ["N", "HR", "TMCP", "TMCP_T", "QT", "CF"]
CHEM_FEATURES = ["C", "Mn", "Si", "Cr", "Mo", "Nb", "V", "Ti", "Ni", "Al", "Cu", "P", "S"]
TEMP_FEATURES = ["temperature"]
FEATURE_COLUMNS = PROCESS_FEATURES + CHEM_FEATURES + TEMP_FEATURES
TARGET_COLUMNS = ["fy_reduction", "fu_reduction", "E_reduction"]
GROUP_COLUMN = "diff_steel"


@dataclass
class TargetData:
    target: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    sample_id: pd.Series
    dataframe: pd.DataFrame


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load csv and add 1-based sample_id based on original row order."""
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True).copy()

    # Convert blank strings to NaN first, then coerce numeric columns safely.
    numeric_cols = FEATURE_COLUMNS + TARGET_COLUMNS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sample_id"] = df.index + 1
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_cols = set(FEATURE_COLUMNS + TARGET_COLUMNS + [GROUP_COLUMN, "sample_id"])
    missing = required_cols - set(df.columns)
    if missing:
        missing_sorted = sorted(missing)
        raise ValueError(f"Missing required columns: {missing_sorted}")


def prepare_target_data(df: pd.DataFrame, target: str) -> TargetData:
    """
    Keep rows with non-null target and return feature matrix, labels, groups and ids.
    diff_steel is only kept as grouping info and never enters model features.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported target: {target}")

    target_df = df[df[target].notna()].copy()
    if target_df.empty:
        raise ValueError(f"No available samples for target: {target}")

    X = target_df[FEATURE_COLUMNS].copy()
    y = target_df[target].astype(float).copy()
    groups = target_df[GROUP_COLUMN].copy()
    sample_id = target_df["sample_id"].astype(int).copy()

    return TargetData(
        target=target,
        X=X,
        y=y,
        groups=groups,
        sample_id=sample_id,
        dataframe=target_df,
    )


def get_feature_columns() -> List[str]:
    return FEATURE_COLUMNS.copy()


def get_target_columns() -> List[str]:
    return TARGET_COLUMNS.copy()
