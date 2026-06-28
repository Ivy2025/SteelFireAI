from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def p20_score(y_true, y_pred) -> float:
	y_true_arr = np.asarray(y_true, dtype=float)
	y_pred_arr = np.asarray(y_pred, dtype=float)

	denominator = np.where(np.abs(y_true_arr) < 1e-12, np.nan, np.abs(y_true_arr))
	rel_error = np.abs(y_pred_arr - y_true_arr) / denominator
	valid_mask = np.isfinite(rel_error)
	if not np.any(valid_mask):
		return float("nan")
	return float(np.mean(rel_error[valid_mask] <= 0.2))


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
	mse = float(mean_squared_error(y_true, y_pred))
	return {
		"R2": float(r2_score(y_true, y_pred)),
		"MAE": float(mean_absolute_error(y_true, y_pred)),
		"MSE": mse,
		"RMSE": float(np.sqrt(mse)),
		"P20": p20_score(y_true, y_pred),
	}

