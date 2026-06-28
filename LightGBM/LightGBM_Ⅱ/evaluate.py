from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def p20_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Defensive check: dataset currently has no zero targets, but keep safe behavior.
    non_zero = y_true != 0
    if not np.any(non_zero):
        return float("nan")

    rel_err = np.abs((y_pred[non_zero] - y_true[non_zero]) / y_true[non_zero])
    return float(np.mean(rel_err <= 0.20))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(root_mean_squared_error(y_true, y_pred)),
        f"{prefix}_P20": p20_score(y_true, y_pred),
    }