from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pred_vs_true(
	y_true,
	y_pred,
	target_name: str,
	save_path: Path,
	abs_error_band: float | None = None,
	show_percent_band: bool = True,
) -> None:
	save_path.parent.mkdir(parents=True, exist_ok=True)

	y_true_arr = np.asarray(y_true, dtype=float)
	y_pred_arr = np.asarray(y_pred, dtype=float)
	min_v = float(min(np.nanmin(y_true_arr), np.nanmin(y_pred_arr)))
	max_v = float(max(np.nanmax(y_true_arr), np.nanmax(y_pred_arr)))

	fig, ax = plt.subplots(figsize=(7, 6))
	ax.scatter(y_true_arr, y_pred_arr, alpha=0.7, edgecolors="none")

	x_line = np.linspace(min_v, max_v, 200)
	ax.plot(x_line, x_line, linestyle="-", linewidth=1.5, label="y=x")
	if show_percent_band:
		ax.plot(x_line, 1.2 * x_line, linestyle="--", linewidth=1.2, label="+20%")
		ax.plot(x_line, 0.8 * x_line, linestyle="--", linewidth=1.2, label="-20%")
	if abs_error_band is not None and abs_error_band > 0:
		ax.plot(
			x_line,
			x_line + abs_error_band,
			linestyle=":",
			linewidth=1.2,
			label=f"+{abs_error_band:g} abs err",
		)
		ax.plot(
			x_line,
			x_line - abs_error_band,
			linestyle=":",
			linewidth=1.2,
			label=f"-{abs_error_band:g} abs err",
		)

	ax.set_title(f"Predicted vs True | {target_name}")
	ax.set_xlabel("True")
	ax.set_ylabel("Predicted")
	ax.legend()
	ax.grid(alpha=0.25)

	fig.tight_layout()
	fig.savefig(save_path, dpi=300)
	plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, target_name: str, save_path: Path) -> None:
	save_path.parent.mkdir(parents=True, exist_ok=True)
	if importance_df.empty:
		return

	sorted_df = importance_df.sort_values("importance", ascending=True).tail(20)

	fig, ax = plt.subplots(figsize=(8, 7))
	ax.barh(sorted_df["feature"], sorted_df["importance"])
	ax.set_title(f"Top Feature Importance | {target_name}")
	ax.set_xlabel("Importance")
	ax.set_ylabel("Feature")
	ax.grid(axis="x", alpha=0.25)

	fig.tight_layout()
	fig.savefig(save_path, dpi=300)
	plt.close(fig)


def plot_fold_metrics_bar(fold_metrics_df: pd.DataFrame, target_name: str, save_path: Path) -> None:
	save_path.parent.mkdir(parents=True, exist_ok=True)
	if fold_metrics_df.empty:
		return

	metrics = [col for col in fold_metrics_df.columns if col != "fold"]
	if not metrics:
		return

	x = np.arange(len(fold_metrics_df))
	width = 0.8 / max(len(metrics), 1)
	offset_center = (len(metrics) - 1) / 2

	fig, ax = plt.subplots(figsize=(9, 6))
	for i, metric_name in enumerate(metrics):
		ax.bar(
			x + (i - offset_center) * width,
			fold_metrics_df[metric_name].values,
			width=width,
			label=metric_name,
		)

	ax.set_xticks(x)
	ax.set_xticklabels([f"Fold {int(f)}" for f in fold_metrics_df["fold"].values])
	ax.set_title(f"Per-Fold Metrics | {target_name}")
	ax.set_ylabel("Metric Value")
	ax.legend()
	ax.grid(axis="y", alpha=0.25)

	fig.tight_layout()
	fig.savefig(save_path, dpi=300)
	plt.close(fig)

