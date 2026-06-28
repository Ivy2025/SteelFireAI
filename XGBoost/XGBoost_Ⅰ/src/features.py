from typing import Dict, Tuple

import pandas as pd


def split_features_targets(
	df: pd.DataFrame,
	feature_info: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	feature_cols = feature_info["feature_cols"]
	target_cols = feature_info["target_cols"]
	group_col = feature_info["group_col"]
	id_col = feature_info["id_col"]

	x = df[feature_cols].copy()
	y = df[target_cols].copy()
	groups = df[group_col].copy()
	if id_col in df.columns:
		ids = df[id_col].copy()
	else:
		# Keep output schema stable even when no explicit ID column is provided.
		ids = pd.Series(range(1, len(df) + 1), index=df.index, name=id_col)
	return x, y, groups, ids

