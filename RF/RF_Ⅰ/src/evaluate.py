# src/evaluate.py

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluate a model on test set and optionally training set.

    Returns a dictionary containing R2/MAE for test data and, if
    training data are provided, also for the training set.  Keys are
    prefixed with "test_" or "train_" accordingly.
    """
    results = {}

    # helper to compute metrics given X and y and a prefix

    def _compute(prefix, X, y):
        y_pred = model.predict(X)
        results[f"{prefix}_R2"] = r2_score(y, y_pred)
        results[f"{prefix}_MAE"] = mean_absolute_error(y, y_pred)
        # 新增RMSE
        results[f"{prefix}_RMSE"] = mean_squared_error(y, y_pred, squared=False)

        # 计算 P20 和 P25：预测在真实值±百分比范围内的比例
        # 避免除以零的情况，当真实值为0时只要求预测也为0即可
        pct_error = None
        # 我们用绝对误差除以真实值的绝对值（处理0）
        abs_err = abs(y_pred - y)
        with_zero = (y == 0)
        if any(with_zero):
            # 对于真实值为0的样本，用预测等于0视为 0% 误差，其余视为无穷大
            pct_error = abs_err.copy()
            pct_error[with_zero] = np.where(y_pred[with_zero] == 0, 0, np.inf)
            pct_error[~with_zero] = abs_err[~with_zero] / np.abs(y[~with_zero])
        else:
            pct_error = abs_err / np.abs(y)

        results[f"{prefix}_P20"] = np.mean(pct_error <= 0.20)
        results[f"{prefix}_P25"] = np.mean(pct_error <= 0.25)

        # 新增：满足 (预测值-真实值)/真实值 > 0.2 的样本占全部样本比例
        # 对 y=0 的样本将相对误差置为 +inf，避免除零并排除该样本
        rel_err = np.full_like(y_pred, np.inf, dtype=float)
        non_zero = (y != 0)
        rel_err[non_zero] = (y_pred[non_zero] - y[non_zero]) / y[non_zero]
        non_conservative_mask = (rel_err > 0.2)
        non_conservative_p20 = np.mean(non_conservative_mask)
        results[f"{prefix}_non_conservative_P20"] = non_conservative_p20
        results[f"{prefix}_one_minus_non_conservative_P20"] = 1 - non_conservative_p20
        results[f"{prefix}_non_conservative_P20_count"] = int(np.sum(non_conservative_mask))

        if hasattr(X, 'columns') and 'temperature' in X.columns:
            t_mask = (X['temperature'] < 800).values
            if np.sum(t_mask) > 0:
                results[f"{prefix}_non_conservative_P20_800"] = float(np.mean(non_conservative_mask[t_mask]))
            else:
                results[f"{prefix}_non_conservative_P20_800"] = 0.0

    _compute("test", X_test, y_test)

    if X_train is not None and y_train is not None:
        _compute("train", X_train, y_train)

    return results
