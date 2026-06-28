from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

GROUP_COLUMN = "diff_steel"
TARGET_COLUMNS = ["fy_reduction", "fu_reduction", "E_reduction"]
PROCESS_ONEHOT_COLUMNS = ["N", "HR", "TMCP", "TMCP_T", "QT", "CF"]
PROCESS_COLUMN = "process_type"
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
NUMERIC_FEATURE_COLUMNS = CHEMICAL_COLUMNS + [TEMPERATURE_COLUMN]
FEATURE_COLUMNS = [PROCESS_COLUMN] + NUMERIC_FEATURE_COLUMNS


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

    # Backward compatibility for historical files using one-hot process columns.
    if PROCESS_COLUMN not in df.columns and all(col in df.columns for col in PROCESS_ONEHOT_COLUMNS):
        process_flags = (
            df[PROCESS_ONEHOT_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        onehot_sum = process_flags.sum(axis=1)
        process_type = pd.Series("UNKNOWN", index=df.index, dtype="string")
        one_mask = onehot_sum.eq(1)
        process_type.loc[one_mask] = process_flags.loc[one_mask].idxmax(axis=1).astype("string")
        process_type.loc[onehot_sum.eq(0)] = "NONE"
        process_type.loc[onehot_sum.gt(1)] = "MULTI"
        df[PROCESS_COLUMN] = process_type

    # sample_id uses original row order and starts from 1.
    df.insert(0, "sample_id", range(1, len(df) + 1))
    return df


def prepare_target_data(df: pd.DataFrame, target: str) -> TargetData:
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")

    required_columns = ["sample_id"] + FEATURE_COLUMNS + [target]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for target {target}: {missing}")

    selected_columns = required_columns + ([GROUP_COLUMN] if GROUP_COLUMN in df.columns else [])
    work_df = df[selected_columns].copy()
    work_df[target] = pd.to_numeric(work_df[target], errors="coerce")
    work_df = work_df.dropna(subset=[target]).reset_index(drop=True)

    X = work_df[FEATURE_COLUMNS].copy()
    X[PROCESS_COLUMN] = X[PROCESS_COLUMN].fillna("UNKNOWN").astype(str)
    for col in NUMERIC_FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = work_df[target].astype(float)
    if GROUP_COLUMN in work_df.columns:
        groups = work_df[GROUP_COLUMN].astype(str)
    else:
        groups = pd.Series(["NA"] * len(work_df), index=work_df.index, dtype="string")
    sample_id = work_df["sample_id"].astype(int)

    return TargetData(
        df=work_df,
        X=X,
        y=y,
        groups=groups,
        sample_id=sample_id,
        feature_columns=FEATURE_COLUMNS,
    )
