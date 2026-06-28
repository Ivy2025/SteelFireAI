from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_p20(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate P20 with safe handling for near-zero denominators."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    non_zero_mask = np.abs(y_true) > 1e-12
    rel_ok = np.zeros_like(y_true, dtype=bool)
    rel_ok[non_zero_mask] = (
        np.abs((y_pred[non_zero_mask] - y_true[non_zero_mask]) / y_true[non_zero_mask]) <= 0.20
    )

    zero_mask = ~non_zero_mask
    if np.any(zero_mask):
        rel_ok[zero_mask] = np.abs(y_pred[zero_mask] - y_true[zero_mask]) <= 0.20

    return float(np.mean(rel_ok))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """Return unified metric dictionary for regression."""
    return {
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        f"{prefix}_P20": calculate_p20(y_true, y_pred),
    }
