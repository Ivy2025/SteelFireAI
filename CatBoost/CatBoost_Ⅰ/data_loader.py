from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

GROUP_COLUMN = "diff_steel"
TARGET_COLUMNS = ["fy_reduction", "fu_reduction", "E_reduction"]
PROCESS_COLUMNS = ["N", "HR", "TMCP", "TMCP_T", "QT", "CF"]
CHEMICAL_COLUMNS = [
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
]
TEMPERATURE_COLUMN = "temperature"
FEATURE_COLUMNS = PROCESS_COLUMNS + CHEMICAL_COLUMNS + [TEMPERATURE_COLUMN]


@dataclass
class TargetData:
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    sample_id: pd.Series
    feature_columns: List[str]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        na_values=["", " ", "  ", "   "],
        keep_default_na=True,
    )
    # sample_id uses original row order and starts from 1.
    df.insert(0, "sample_id", range(1, len(df) + 1))
    return df


def prepare_target_data(df: pd.DataFrame, target: str) -> TargetData:
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")

    required_columns = ["sample_id", GROUP_COLUMN] + FEATURE_COLUMNS + [target]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for target {target}: {missing}")

    work_df = df[required_columns].copy()
    work_df[target] = pd.to_numeric(work_df[target], errors="coerce")
    work_df = work_df.dropna(subset=[target]).reset_index(drop=True)

    X = work_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = work_df[target].astype(float)
    groups = work_df[GROUP_COLUMN].astype(str)
    sample_id = work_df["sample_id"].astype(int)

    return TargetData(
        df=work_df,
        X=X,
        y=y,
        groups=groups,
        sample_id=sample_id,
        feature_columns=FEATURE_COLUMNS,
    )
