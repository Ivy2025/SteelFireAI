from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def p20_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    non_zero_mask = np.abs(y_true) > 1e-12
    if not np.any(non_zero_mask):
        return float("nan")

    rel_error = np.abs((y_pred[non_zero_mask] - y_true[non_zero_mask]) / y_true[non_zero_mask])
    return float(np.mean(rel_error <= 0.20))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    results = {
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_MSE": float(mse),
        f"{prefix}_RMSE": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}_P20": p20_score(y_true, y_pred),
    }

    # 新增非保守性相关指标
    rel_err = np.full_like(y_pred, np.inf, dtype=float)
    non_zero = (y_true != 0)
    rel_err[non_zero] = (y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero]
    non_conservative_mask = (rel_err > 0.2)
    non_conservative_p20 = np.mean(non_conservative_mask)
    results[f"{prefix}_non_conservative_P20"] = non_conservative_p20
    results[f"{prefix}_one_minus_non_conservative_P20"] = 1 - non_conservative_p20
    results[f"{prefix}_non_conservative_P20_count"] = int(np.sum(non_conservative_mask))

    # 针对温度<800的子集
    # 这里假设X在全局不可用，需由调用方传入。为兼容性，先尝试全局变量X，否则跳过
    try:
        X = globals().get('X', None)
        if X is not None and hasattr(X, 'columns') and 'temperature' in X.columns:
            t_mask = (X['temperature'] < 800).values
            if np.sum(t_mask) > 0:
                results[f"{prefix}_non_conservative_P20_800"] = float(np.mean(non_conservative_mask[t_mask]))
            else:
                results[f"{prefix}_non_conservative_P20_800"] = 0.0
    except Exception:
        results[f"{prefix}_non_conservative_P20_800"] = np.nan

    return results