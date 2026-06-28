from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def p20_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return fraction of samples where relative error is within 20%."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return float("nan")

    denom = np.where(np.abs(y_true) > 1e-12, np.abs(y_true), np.nan)
    rel_err = np.abs(y_pred - y_true) / denom
    valid_mask = ~np.isnan(rel_err)

    if not np.any(valid_mask):
        return float("nan")

    return float(np.mean(rel_err[valid_mask] <= 0.20))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_P20": p20_score(y_true, y_pred),
    }
