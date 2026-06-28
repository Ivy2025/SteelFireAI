import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings
from src.trainer import run_training


def main() -> None:
	# PlanA: median imputation control group.
	settings_plan_a = Settings(
		skip_feature_imputation=False,
		keep_test_summary_and_scatter_only=True,
		result_suffix="_planA",
	)
	run_training(settings_plan_a)

	# PlanB: XGBoost native missing-value handling.
	settings_plan_b = Settings(
		skip_feature_imputation=True,
		keep_test_summary_and_scatter_only=True,
		result_suffix="_planB",
	)
	run_training(settings_plan_b)


if __name__ == "__main__":
	main()

