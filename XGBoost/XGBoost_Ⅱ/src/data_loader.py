import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"Data file not found: {csv_path}")
	return pd.read_csv(csv_path)


def coerce_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
	converted_df = df.copy()
	for col in numeric_cols:
		converted_df[col] = pd.to_numeric(converted_df[col], errors="coerce")
	return converted_df


def fill_missing_with_median(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
	filled_df = df.copy()
	for col in cols:
		if col not in filled_df.columns:
			continue
		median_value = filled_df[col].median(skipna=True)
		if pd.isna(median_value):
			continue
		filled_df[col] = filled_df[col].fillna(median_value)
	return filled_df


def infer_feature_info(
	df: pd.DataFrame,
	id_col: str,
	group_col: str,
	target_cols: List[str] | None = None,
	excluded_feature_cols: List[str] | None = None,
) -> Dict:
	if target_cols is None:
		target_cols = [c for c in df.columns if c.endswith("_reduction")]
		if len(target_cols) < 3:
			target_cols = list(df.columns[-3:])
	if excluded_feature_cols is None:
		excluded_feature_cols = []

	feature_cols = [
		c
		for c in df.columns
		if c not in set([id_col, group_col] + list(target_cols) + list(excluded_feature_cols))
	]

	return {
		"id_col": id_col,
		"group_col": group_col,
		"target_cols": list(target_cols),
		"feature_cols": feature_cols,
	}


def sanitize_feature_info(feature_info: Dict, excluded_feature_cols: List[str] | None = None) -> Dict:
	if excluded_feature_cols is None:
		excluded_feature_cols = []

	excluded_set = set(excluded_feature_cols)
	cleaned_feature_cols = [c for c in feature_info["feature_cols"] if c not in excluded_set]
	feature_info["feature_cols"] = cleaned_feature_cols
	return feature_info


def save_feature_info(feature_info: Dict, feature_info_path: Path) -> None:
	feature_info_path.parent.mkdir(parents=True, exist_ok=True)
	with feature_info_path.open("w", encoding="utf-8") as f:
		json.dump(feature_info, f, ensure_ascii=False, indent=2)


def load_feature_info(feature_info_path: Path) -> Dict:
	if not feature_info_path.exists():
		raise FileNotFoundError(f"feature_info.json not found: {feature_info_path}")
	with feature_info_path.open("r", encoding="utf-8") as f:
		return json.load(f)


def validate_columns(df: pd.DataFrame, feature_info: Dict, allow_missing_id_col: bool = False) -> None:
	required = [
		feature_info["id_col"],
		feature_info["group_col"],
		*feature_info["target_cols"],
		*feature_info["feature_cols"],
	]
	if allow_missing_id_col:
		required = [c for c in required if c != feature_info["id_col"]]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in dataset: {missing}")

