from pathlib import Path
from typing import Tuple

import pandas as pd

GROUP_COL = "diff_steel"

PROCESS_FEATURES = ["N", "HR", "TMCP", "TMCP_T", "QT", "CF"]
CHEM_FEATURES = ["C", "Mn", "Si", "Cr", "Mo", "Nb", "V", "Ti", "Ni", "Al", "Cu", "P", "S"]
TEMP_FEATURES = ["temperature"]

TARGET_COLS = ["fy_reduction", "fu_reduction", "E_reduction"]
FEATURE_COLS = PROCESS_FEATURES + CHEM_FEATURES + TEMP_FEATURES


def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    df = pd.read_csv(path)
    return df


def _check_required_columns(df: pd.DataFrame) -> None:
    required_cols = set([GROUP_COL] + FEATURE_COLS + TARGET_COLS)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")


def prepare_target_data(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    返回:
    X: 当前目标对应的特征矩阵（不含 diff_steel 和任意目标列）
    y: 当前目标
    groups: 分组列 diff_steel
    diff_steel: 与样本对齐的标识列（用于保存预测）
    """
    if target not in TARGET_COLS:
        raise ValueError(f"不支持的目标列: {target}")

    _check_required_columns(df)

    # 当前目标或分组缺失的样本无法用于建模/分组
    data = df.dropna(subset=[target, GROUP_COL]).copy()

    # 仅保留本任务明确的特征列，避免泄漏其他目标列
    X = data[FEATURE_COLS].copy()

    # 将特征转为数值，非数值强制转 NaN，后续由 Pipeline 中位数填补
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    y = pd.to_numeric(data[target], errors="coerce")
    valid_y_mask = y.notna()

    X = X.loc[valid_y_mask].reset_index(drop=True)
    y = y.loc[valid_y_mask].reset_index(drop=True)
    groups = data.loc[valid_y_mask, GROUP_COL].reset_index(drop=True)
    diff_steel = data.loc[valid_y_mask, GROUP_COL].reset_index(drop=True)

    return X, y, groups, diff_steel
