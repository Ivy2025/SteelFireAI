from typing import Generator, Tuple

import numpy as np
from sklearn.model_selection import KFold


def build_kfold(n_splits: int, random_state: int, shuffle: bool = True) -> KFold:
	return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def generate_random_folds(
	x,
	n_splits: int,
	random_state: int,
	shuffle: bool = True,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
	splitter = build_kfold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
	for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x), start=1):
		yield fold_idx, train_idx, val_idx

