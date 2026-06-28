import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings


def _sqrt_safe(value):
	if pd.isna(value):
		return float("nan")
	return float(math.sqrt(float(value)))


def _backfill_for_file(file_path: Path) -> bool:
	if not file_path.exists():
		return False

	df = pd.read_csv(file_path)
	updated = False

	if "MSE" in df.columns:
		df["RMSE"] = df["MSE"].apply(_sqrt_safe)
		updated = True

	if "test_MSE" in df.columns:
		df["test_RMSE"] = df["test_MSE"].apply(_sqrt_safe)
		updated = True

	if updated:
		df.to_csv(file_path, index=False, encoding="utf-8-sig")

	return updated


def _backfill_cv_summary(settings: Settings) -> bool:
	summary_path = settings.metrics_dir / "cv_summary_all_targets.csv"
	if not summary_path.exists():
		return False

	df = pd.read_csv(summary_path)
	if "target" not in df.columns:
		return False

	updated = False
	for idx, row in df.iterrows():
		target_name = row["target"]
		fold_path = settings.metrics_dir / f"cv_fold_metrics_{target_name}.csv"
		if not fold_path.exists():
			continue

		fold_df = pd.read_csv(fold_path)
		if "RMSE" not in fold_df.columns:
			if "MSE" not in fold_df.columns:
				continue
			fold_df["RMSE"] = fold_df["MSE"].apply(_sqrt_safe)
			fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
			updated = True

		df.loc[idx, "RMSE_mean"] = float(fold_df["RMSE"].mean())
		df.loc[idx, "RMSE_std"] = float(fold_df["RMSE"].std(ddof=1))
		updated = True

	if updated:
		df.to_csv(summary_path, index=False, encoding="utf-8-sig")

	return updated


def backfill_rmse(settings: Settings) -> list[Path]:
	metrics_dir = settings.metrics_dir
	targets = [
		metrics_dir / "test_summary_all_targets.csv",
		metrics_dir / "test_summary_all_targets_planB.csv",
	]
	targets.extend(sorted(metrics_dir.glob("cv_fold_metrics_*.csv")))

	updated_files = []
	for file_path in targets:
		if _backfill_for_file(file_path):
			updated_files.append(file_path)

	if _backfill_cv_summary(settings):
		updated_files.append(metrics_dir / "cv_summary_all_targets.csv")
	return updated_files


def main() -> None:
	settings = Settings()
	updated_files = backfill_rmse(settings)
	if not updated_files:
		print("No metrics files were updated.")
		return

	print("Updated RMSE columns in:")
	for path in updated_files:
		print(f"- {path}")


if __name__ == "__main__":
	main()