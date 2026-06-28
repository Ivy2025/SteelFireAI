from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def p20_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    non_zero_mask = np.abs(y_true) > 1e-12
    if not np.any(non_zero_mask):
        return float("nan")

    rel_error = np.abs((y_pred[non_zero_mask] - y_true[non_zero_mask]) / y_true[non_zero_mask])
    return float(np.mean(rel_error <= 0.20))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": rmse,
        f"{prefix}_P20": p20_score(y_true, y_pred),
    }
