from typing import Generator, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold


def build_group_kfold(n_splits: int) -> GroupKFold:
	return GroupKFold(n_splits=n_splits)


def generate_group_folds(
	x,
	groups,
	n_splits: int,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
	splitter = build_group_kfold(n_splits=n_splits)
	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, groups=groups), start=1):
		yield fold_idx, train_idx, val_idx

