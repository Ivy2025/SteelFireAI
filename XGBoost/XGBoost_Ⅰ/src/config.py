from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
	project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
	raw_data_file: str = "steel_v2_cleaned_r2.csv"
	feature_info_file: str = "feature_info.json"
	excluded_feature_cols: tuple[str, ...] = ("Ceq",)
	target_cols: tuple[str, ...] = ("fy_reduction", "fu_reduction", "E_reduction")

	group_col: str = "diff_steel"
	id_col: str = "diff_steel"

	n_splits: int = 5
	inner_cv_splits: int = 3
	use_test_holdout: bool = True
	test_size: float = 0.2
	random_state: int = 42
	n_jobs: int = -1
	scoring: str = "r2"

	# Plan-B experiment controls.
	skip_feature_imputation: bool = False
	keep_test_summary_and_scatter_only: bool = False
	result_suffix: str = ""

	@property
	def data_raw_path(self) -> Path:
		return self.project_root / "data" / self.raw_data_file

	@property
	def feature_info_path(self) -> Path:
		return self.project_root / "data" / self.feature_info_file

	@property
	def results_dir(self) -> Path:
		return self.project_root / "results"

	@property
	def metrics_dir(self) -> Path:
		return self.results_dir / "metrics"

	@property
	def params_dir(self) -> Path:
		return self.results_dir / "params"

	@property
	def predictions_dir(self) -> Path:
		return self.results_dir / "predictions"

	@property
	def figures_dir(self) -> Path:
		return self.results_dir / "figures"

	@property
	def logs_dir(self) -> Path:
		return self.results_dir / "logs"

	@property
	def training_log_path(self) -> Path:
		return self.logs_dir / "training_log.txt"

