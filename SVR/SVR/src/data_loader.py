# src/data_loader.py
from pathlib import Path
from typing import Tuple

import pandas as pd


PROCESS_ONEHOT_COLS = ["HR", "N", "TMCP", "TMCP_T", "QT", "CF"]
CHEM_COLS = ["C", "Mn", "Si", "Cr", "Mo", "Nb", "V", "Ti", "Ni", "Al", "Cu", "P", "S"]
GROUP_COL = "diff_steel"


def _resolve_temperature_col(columns) -> str:
    lower_map = {c.strip().lower(): c for c in columns}
    if "temperature" in lower_map:
        return lower_map["temperature"]
    if "temp" in lower_map:
        return lower_map["temp"]

    for c in columns:
        if c.strip().lower().startswith("temp"):
            return c

    raise ValueError("未找到温度列，请确认列名包含 temperature 或 temp。")


def load_target_data(
    data_path: str | Path, target: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    data_path = Path(data_path)
    df = pd.read_csv(data_path, na_values=["", " "], keep_default_na=True)

    if GROUP_COL not in df.columns:
        raise ValueError(f"缺少分组列: {GROUP_COL}")
    if target not in df.columns:
        raise ValueError(f"缺少目标列: {target}")

    temp_col = _resolve_temperature_col(df.columns)
    if temp_col != "temperature":
        df = df.rename(columns={temp_col: "temperature"})

    required_feature_cols = PROCESS_ONEHOT_COLS + CHEM_COLS + ["temperature"]
    missing_feature_cols = [c for c in required_feature_cols if c not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"缺少输入特征列: {missing_feature_cols}")

    # 仅删除当前 target 缺失样本；feature 缺失交给 Pipeline 中位数填补
    df = df.dropna(subset=[target]).copy()

    for col in required_feature_cols + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # target 转换后若出现 NaN，继续剔除
    df = df.dropna(subset=[target]).copy()

    X = df[required_feature_cols].copy()
    y = df[target].astype(float).copy()
    groups = df[GROUP_COL].astype(str).copy()
    meta = df[[GROUP_COL, "temperature"]].copy()

    return X, y, groups, meta