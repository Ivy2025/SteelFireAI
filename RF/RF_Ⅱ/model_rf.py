from __future__ import annotations

from typing import Dict, List, Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_rf_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
        ]
    )


def get_rf_param_grid() -> Dict[str, List[Union[int, float, str, None]]]:
    return {
        "rf__n_estimators": [300, 600],
        "rf__max_depth": [None, 10, 20, 30],
        "rf__min_samples_split": [2, 4, 8],
        "rf__min_samples_leaf": [1, 2, 3],
        "rf__max_features": ["sqrt", 0.5, None],
    }
