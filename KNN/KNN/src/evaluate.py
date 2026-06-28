from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def p20_score(y_true, y_pred) -> float:
    """
    P20 = abs((y_pred - y_true) / y_true) <= 0.20 的样本占比
    对 y_true == 0 的样本进行忽略（不计入分母）
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = y_true != 0
    if valid.sum() == 0:
        return float("nan")

    rel_err = np.abs((y_pred[valid] - y_true[valid]) / y_true[valid])
    return float(np.mean(rel_err <= 0.20))


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "P20": p20_score(y_true, y_pred),
    }
