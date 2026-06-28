from collections import Counter
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from src.config import Settings
from src.cv import generate_random_folds
from src.data_loader import (
	coerce_numeric_columns,
	fill_missing_with_median,
	infer_feature_info,
	load_dataset,
	load_feature_info,
	sanitize_feature_info,
	save_feature_info,
	validate_columns,
)
from src.features import split_features_targets
from src.metrics import evaluate_regression
from src.model import build_xgb_regressor, get_param_grid
from src.plotter import plot_feature_importance, plot_fold_metrics_bar, plot_pred_vs_true
from src.utils import ensure_dirs, save_json, setup_logger


def _build_or_load_feature_info(settings: Settings, df: pd.DataFrame) -> Dict:
	excluded_feature_cols = list(settings.excluded_feature_cols)
	target_cols = list(settings.target_cols)

	if settings.feature_info_path.exists():
		feature_info = load_feature_info(settings.feature_info_path)
		feature_info = sanitize_feature_info(feature_info, excluded_feature_cols=excluded_feature_cols)
		try:
			validate_columns(df, feature_info, allow_missing_id_col=True)
		except ValueError:
			feature_info = infer_feature_info(
				df,
				id_col=settings.id_col,
				group_col=settings.group_col,
				target_cols=target_cols,
				excluded_feature_cols=excluded_feature_cols,
			)
	else:
		feature_info = infer_feature_info(
			df,
			id_col=settings.id_col,
			group_col=settings.group_col,
			target_cols=target_cols,
			excluded_feature_cols=excluded_feature_cols,
		)

	feature_info = sanitize_feature_info(feature_info, excluded_feature_cols=excluded_feature_cols)
	save_feature_info(feature_info, settings.feature_info_path)
	validate_columns(df, feature_info, allow_missing_id_col=True)
	return feature_info


def _select_representative_params(best_params_records: List[Dict]) -> Dict:
	if not best_params_records:
		return {}
	normalized = [tuple(sorted(item.items(), key=lambda kv: kv[0])) for item in best_params_records]
	chosen_tuple = Counter(normalized).most_common(1)[0][0]
	return dict(chosen_tuple)


def _summarize_param_combinations(best_params_records: List[Dict]) -> pd.DataFrame:
	if not best_params_records:
		return pd.DataFrame(columns=["rank", "count", "proportion"])

	normalized = [tuple(sorted(item.items(), key=lambda kv: kv[0])) for item in best_params_records]
	combo_counts = Counter(normalized)
	total = sum(combo_counts.values())
	rows = []
	for rank, (combo, count) in enumerate(combo_counts.most_common(), start=1):
		row = {"rank": rank, "count": int(count), "proportion": float(count / total)}
		row.update(dict(combo))
		rows.append(row)

	return pd.DataFrame(rows)


def _run_fold_grid_search(
	x_train: pd.DataFrame,
	y_train: pd.Series,
	target_name: str,
	settings: Settings,
) -> Tuple[object, Dict]:
	n_train = int(len(x_train))
	if n_train < 2:
		model = build_xgb_regressor(random_state=settings.random_state)
		model.fit(x_train, y_train)
		return model, model.get_params()

	inner_splits = min(settings.inner_cv_splits, n_train)
	if inner_splits < 2:
		model = build_xgb_regressor(random_state=settings.random_state)
		model.fit(x_train, y_train)
		return model, model.get_params()

	inner_splitter = KFold(
		n_splits=inner_splits,
		shuffle=True,
		random_state=settings.random_state,
	)

	estimator = build_xgb_regressor(random_state=settings.random_state)
	grid_search = GridSearchCV(
		estimator=estimator,
		param_grid=get_param_grid(target_name=target_name),
		scoring=settings.scoring,
		cv=inner_splitter,
		n_jobs=settings.n_jobs,
		refit=True,
		verbose=0,
	)
	# numpy + sklearn may emit this benign warning while formatting masked CV results.
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message="invalid value encountered in cast",
			category=RuntimeWarning,
		)
		grid_search.fit(x_train, y_train)
	return grid_search.best_estimator_, grid_search.best_params_


def _split_train_test_random(
	n_samples: int,
	test_size: float,
	random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
	if n_samples < 2:
		raise ValueError("At least 2 samples are required for train/test split.")

	test_size = float(np.clip(test_size, 0.05, 0.5))
	indices = np.arange(n_samples, dtype=int)
	train_idx, test_idx = train_test_split(
		indices,
		test_size=test_size,
		random_state=random_state,
		shuffle=True,
	)
	if len(train_idx) == 0 or len(test_idx) == 0:
		raise ValueError("Failed to create non-empty random train/test split.")

	return train_idx, test_idx


def run_training(settings: Settings) -> None:
	result_suffix = settings.result_suffix
	keep_test_only = settings.keep_test_summary_and_scatter_only

	def _with_suffix(name: str) -> str:
		if not result_suffix:
			return name
		base, dot, ext = name.rpartition(".")
		if not dot:
			return f"{name}{result_suffix}"
		return f"{base}{result_suffix}.{ext}"

	ensure_dirs(
		[
			settings.metrics_dir,
			settings.params_dir,
			settings.predictions_dir,
			settings.figures_dir,
			settings.logs_dir,
			settings.feature_info_path.parent,
		]
	)
	logger = setup_logger(settings.training_log_path)
	logger.info("Training started.")

	df = load_dataset(settings.data_raw_path)
	logger.info("Loaded data: shape=%s", df.shape)

	feature_info = _build_or_load_feature_info(settings, df)
	logger.info("Feature info loaded. features=%d targets=%d", len(feature_info["feature_cols"]), len(feature_info["target_cols"]))

	numeric_cols = feature_info["feature_cols"] + feature_info["target_cols"]
	df = coerce_numeric_columns(df, numeric_cols)
	missing_before_fill = int(df[feature_info["feature_cols"]].isna().sum().sum())
	if settings.skip_feature_imputation:
		missing_after_fill = missing_before_fill
		if missing_before_fill > 0:
			logger.info(
				"Feature missing values kept for XGBoost native missing handling. count=%d",
				missing_before_fill,
			)
	else:
		df = fill_missing_with_median(df, feature_info["feature_cols"])
		missing_after_fill = int(df[feature_info["feature_cols"]].isna().sum().sum())
		if missing_before_fill > 0:
			logger.info(
				"Feature missing values handled by median imputation. before=%d after=%d",
				missing_before_fill,
				missing_after_fill,
			)

	required_non_feature_cols = [
		feature_info["id_col"],
		feature_info["group_col"],
		*feature_info["target_cols"],
	]
	n_before_drop = len(df)
	df = df.dropna(subset=required_non_feature_cols).reset_index(drop=True)
	dropped_rows = n_before_drop - len(df)
	if dropped_rows > 0:
		logger.warning(
			"Dropped %d rows with missing key columns/targets after preprocessing.",
			dropped_rows,
		)

	x_all, y_all, groups, ids = split_features_targets(df, feature_info)
	target_cols = feature_info["target_cols"]

	if settings.use_test_holdout:
		train_idx, test_idx = _split_train_test_random(
			n_samples=len(df),
			test_size=settings.test_size,
			random_state=settings.random_state,
		)
		x_train_all = x_all.iloc[train_idx].reset_index(drop=True)
		x_test = x_all.iloc[test_idx].reset_index(drop=True)
		ids_test = ids.iloc[test_idx].reset_index(drop=True)
		groups_test = groups.iloc[test_idx].reset_index(drop=True)
		logger.info(
			"Random holdout split created. train_rows=%d test_rows=%d",
			len(train_idx),
			len(test_idx),
		)
	else:
		x_train_all = x_all
		x_test = pd.DataFrame(columns=x_all.columns)
		ids_test = pd.Series(dtype=ids.dtype)
		groups_test = pd.Series(dtype=groups.dtype)
		train_idx = np.arange(len(df), dtype=int)
		test_idx = np.array([], dtype=int)
		logger.info("Holdout test is disabled. Only cross-validation metrics will be produced.")

	cv_summary_rows = []
	test_summary_rows = []
	combined_oof = pd.DataFrame({
		feature_info["id_col"]: ids,
		feature_info["group_col"]: groups,
	})
	combined_test = pd.DataFrame({
		feature_info["id_col"]: ids_test,
		feature_info["group_col"]: groups_test,
	})

	outer_splits = min(settings.n_splits, int(len(x_train_all)))
	if outer_splits < 2:
		raise ValueError("At least 2 training samples are required for KFold CV.")

	for target_name in target_cols:
		logger.info("Start target: %s", target_name)
		y_target = y_all[target_name]
		y_train_target = y_target.iloc[train_idx].reset_index(drop=True)
		y_test_target = y_target.iloc[test_idx].reset_index(drop=True) if len(test_idx) > 0 else pd.Series(dtype=float)

		oof_pred = pd.Series(index=df.index, dtype=float)
		oof_fold = pd.Series(index=df.index, dtype="Int64")

		fold_metrics = []
		fold_params = []
		fold_importances = []

		for fold_idx, fold_train_idx, val_idx in generate_random_folds(
			x_train_all,
			outer_splits,
			random_state=settings.random_state,
			shuffle=True,
		):
			x_train = x_train_all.iloc[fold_train_idx]
			y_train = y_train_target.iloc[fold_train_idx]
			x_val = x_train_all.iloc[val_idx]
			y_val = y_train_target.iloc[val_idx]

			model, best_params = _run_fold_grid_search(
				x_train,
				y_train,
				target_name,
				settings,
			)
			y_pred = model.predict(x_val)

			global_val_idx = train_idx[val_idx]
			oof_pred.iloc[global_val_idx] = y_pred
			oof_fold.iloc[global_val_idx] = fold_idx

			metric_row = {"fold": fold_idx, **evaluate_regression(y_val, y_pred)}
			fold_metrics.append(metric_row)
			fold_params.append({"fold": fold_idx, **best_params})

			if hasattr(model, "feature_importances_"):
				fold_importances.append(
					pd.DataFrame(
						{
							"fold": fold_idx,
							"feature": feature_info["feature_cols"],
							"importance": model.feature_importances_,
						}
					)
				)

			logger.info(
				"target=%s fold=%d R2=%.4f MAE=%.4f MSE=%.4f RMSE=%.4f P20=%.4f",
				target_name,
				fold_idx,
				metric_row["R2"],
				metric_row["MAE"],
				metric_row["MSE"],
				metric_row["RMSE"],
				metric_row["P20"],
			)

		fold_metrics_df = pd.DataFrame(fold_metrics)
		if not keep_test_only:
			fold_metrics_path = settings.metrics_dir / _with_suffix(f"cv_fold_metrics_{target_name}.csv")
			fold_metrics_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")

		params_df = pd.DataFrame(fold_params)
		params_csv_path = settings.params_dir / _with_suffix(f"best_params_{target_name}.csv")
		params_df.to_csv(params_csv_path, index=False, encoding="utf-8-sig")

		param_combo_df = _summarize_param_combinations(
			[
				{k: v for k, v in row.items() if k != "fold"}
				for row in fold_params
			]
		)
		param_combo_path = settings.params_dir / _with_suffix(f"best_params_combo_{target_name}.csv")
		param_combo_df.to_csv(param_combo_path, index=False, encoding="utf-8-sig")

		representative_params = _select_representative_params(
			[
				{k: v for k, v in row.items() if k != "fold"}
				for row in fold_params
			]
		)
		params_json_path = settings.params_dir / _with_suffix(f"best_params_{target_name}.json")
		save_json(
			{
				"target": target_name,
				"representative_best_params": representative_params,
				"best_param_combinations": param_combo_df.to_dict(orient="records"),
				"fold_best_params": fold_params,
			},
			params_json_path,
		)

		test_metric_row = {
			"target": target_name,
			"test_R2": float("nan"),
			"test_MAE": float("nan"),
			"test_MSE": float("nan"),
			"test_RMSE": float("nan"),
			"test_P20": float("nan"),
			"test_abs_err_le_0p1_count": 0,
			"test_abs_err_le_0p1_pct": float("nan"),
			"test_samples": int(len(test_idx)),
		}

		if len(test_idx) > 0:
			final_model = build_xgb_regressor(random_state=settings.random_state)
			if representative_params:
				final_model.set_params(**representative_params)
			final_model.fit(x_train_all, y_train_target)
			y_test_pred = final_model.predict(x_test)

			test_scores = evaluate_regression(y_test_target, y_test_pred)
			abs_error = np.abs(np.asarray(y_test_pred, dtype=float) - np.asarray(y_test_target, dtype=float))
			abs_err_le_0p1_count = int(np.sum(abs_error <= 0.1))
			abs_err_le_0p1_pct = float(abs_err_le_0p1_count / len(test_idx))
			test_metric_row.update(
				{
					"test_R2": test_scores["R2"],
					"test_MAE": test_scores["MAE"],
					"test_MSE": test_scores["MSE"],
					"test_RMSE": test_scores["RMSE"],
					"test_P20": test_scores["P20"],
					"test_abs_err_le_0p1_count": abs_err_le_0p1_count,
					"test_abs_err_le_0p1_pct": abs_err_le_0p1_pct,
				}
			)

			if not keep_test_only:
				test_pred_df = pd.DataFrame(
					{
						feature_info["id_col"]: ids_test,
						feature_info["group_col"]: groups_test,
						f"y_true_{target_name}": y_test_target,
						f"y_pred_{target_name}": y_test_pred,
					}
				)
				test_pred_df.to_csv(
					settings.predictions_dir / _with_suffix(f"test_predictions_{target_name}.csv"),
					index=False,
					encoding="utf-8-sig",
				)

			if not keep_test_only:
				combined_test[f"y_true_{target_name}"] = y_test_target
				combined_test[f"y_pred_{target_name}"] = y_test_pred

			plot_pred_vs_true(
				y_true=y_test_target,
				y_pred=y_test_pred,
				target_name=f"test_{target_name}",
				save_path=settings.figures_dir / _with_suffix(f"scatter_test_pred_vs_true_{target_name}.png"),
			)
			if not keep_test_only:
				plot_pred_vs_true(
					y_true=y_test_target,
					y_pred=y_test_pred,
					target_name=f"test_{target_name}_abs_err_0.1",
					save_path=settings.figures_dir / _with_suffix(f"scatter_test_pred_vs_true_abs_err_0p1_{target_name}.png"),
					abs_error_band=0.1,
					show_percent_band=False,
				)
				plot_pred_vs_true(
					y_true=y_test_target,
					y_pred=y_test_pred,
					target_name=f"test_{target_name}_perfect_only",
					save_path=settings.figures_dir / _with_suffix(f"scatter_test_pred_vs_true_perfect_only_{target_name}.png"),
					show_percent_band=False,
				)

			logger.info(
				"target=%s test_R2=%.4f test_MAE=%.4f test_MSE=%.4f test_RMSE=%.4f test_P20=%.4f abs_err<=0.1_pct=%.4f",
				target_name,
				test_scores["R2"],
				test_scores["MAE"],
				test_scores["MSE"],
				test_scores["RMSE"],
				test_scores["P20"],
				abs_err_le_0p1_pct,
			)

		if not keep_test_only:
			oof_df = pd.DataFrame(
				{
					feature_info["id_col"]: ids,
					feature_info["group_col"]: groups,
					"fold": oof_fold,
					f"y_true_{target_name}": y_target,
					f"y_pred_{target_name}": oof_pred,
				}
			)
			oof_df.to_csv(
				settings.predictions_dir / _with_suffix(f"oof_predictions_{target_name}.csv"),
				index=False,
				encoding="utf-8-sig",
			)

		if not keep_test_only:
			combined_oof[f"y_true_{target_name}"] = y_target
			combined_oof[f"y_pred_{target_name}"] = oof_pred

		if not keep_test_only:
			plot_pred_vs_true(
				y_true=y_target.iloc[train_idx],
				y_pred=oof_pred.iloc[train_idx],
				target_name=target_name,
				save_path=settings.figures_dir / _with_suffix(f"scatter_pred_vs_true_{target_name}.png"),
			)

		if fold_importances and not keep_test_only:
			all_importance_df = pd.concat(fold_importances, ignore_index=True)
			mean_importance_df = all_importance_df.groupby("feature", as_index=False)["importance"].mean()
			mean_importance_df.to_csv(
				settings.params_dir / _with_suffix(f"feature_importance_{target_name}.csv"),
				index=False,
				encoding="utf-8-sig",
			)
			plot_feature_importance(
				mean_importance_df,
				target_name=target_name,
				save_path=settings.figures_dir / _with_suffix(f"feature_importance_{target_name}.png"),
			)

		if not keep_test_only:
			plot_fold_metrics_bar(
				fold_metrics_df,
				target_name=target_name,
				save_path=settings.figures_dir / _with_suffix(f"fold_metrics_{target_name}.png"),
			)

		summary_row = {
			"target": target_name,
			"R2_mean": float(fold_metrics_df["R2"].mean()),
			"R2_std": float(fold_metrics_df["R2"].std(ddof=1)),
			"MAE_mean": float(fold_metrics_df["MAE"].mean()),
			"MAE_std": float(fold_metrics_df["MAE"].std(ddof=1)),
			"MSE_mean": float(fold_metrics_df["MSE"].mean()),
			"MSE_std": float(fold_metrics_df["MSE"].std(ddof=1)),
			"RMSE_mean": float(fold_metrics_df["RMSE"].mean()),
			"RMSE_std": float(fold_metrics_df["RMSE"].std(ddof=1)),
			"P20_mean": float(fold_metrics_df["P20"].mean()),
			"P20_std": float(fold_metrics_df["P20"].std(ddof=1)),
			"test_R2": test_metric_row["test_R2"],
			"test_MAE": test_metric_row["test_MAE"],
			"test_MSE": test_metric_row["test_MSE"],
			"test_RMSE": test_metric_row["test_RMSE"],
			"test_P20": test_metric_row["test_P20"],
			"test_abs_err_le_0p1_count": test_metric_row["test_abs_err_le_0p1_count"],
			"test_abs_err_le_0p1_pct": test_metric_row["test_abs_err_le_0p1_pct"],
			"test_samples": test_metric_row["test_samples"],
		}
		cv_summary_rows.append(summary_row)
		test_summary_rows.append(test_metric_row)
		logger.info("Finished target: %s", target_name)

	if not keep_test_only:
		pd.DataFrame(cv_summary_rows).to_csv(
			settings.metrics_dir / _with_suffix("cv_summary_all_targets.csv"),
			index=False,
			encoding="utf-8-sig",
		)
	pd.DataFrame(test_summary_rows).to_csv(
		settings.metrics_dir / _with_suffix("test_summary_all_targets.csv"),
		index=False,
		encoding="utf-8-sig",
	)
	if not keep_test_only:
		combined_oof.to_csv(
			settings.predictions_dir / _with_suffix("oof_predictions_all_targets.csv"),
			index=False,
			encoding="utf-8-sig",
		)

	if not keep_test_only and len(test_idx) > 0 and not combined_test.empty:
		combined_test.to_csv(
			settings.predictions_dir / _with_suffix("test_predictions_all_targets.csv"),
			index=False,
			encoding="utf-8-sig",
		)

	logger.info("Training completed successfully.")

