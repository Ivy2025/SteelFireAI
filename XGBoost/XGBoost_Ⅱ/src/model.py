from typing import Dict

from xgboost import XGBRegressor


def build_xgb_regressor(random_state: int) -> XGBRegressor:
	return XGBRegressor(
		objective="reg:squarederror",
		random_state=random_state,
		n_jobs=1,
	)


def get_param_grid(target_name: str | None = None) -> Dict:
	if target_name == "E_reduction":
		# Stage-2 local refinement around the current E_reduction best region.
		return {
			"n_estimators": [160, 180, 200, 220, 260, 300],
			"max_depth": [3],
			"learning_rate": [0.015, 0.02, 0.025, 0.03],
			"subsample": [0.8, 0.85, 0.9],
			"colsample_bytree": [0.75, 0.8, 0.85],
			"reg_alpha": [0.0],
			"reg_lambda": [0.5, 0.8, 1.0, 1.2],
		}

	return {
		"n_estimators": [200, 400],
		"max_depth": [4, 6],
		"learning_rate": [0.03, 0.08],
		"subsample": [0.8, 1.0],
		"colsample_bytree": [0.8, 1.0],
		"reg_alpha": [0.0, 0.1],
		"reg_lambda": [1.0, 3.0],
	}

