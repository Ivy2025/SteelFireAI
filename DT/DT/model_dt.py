from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def build_dt_pipeline(random_state: int = 42) -> Pipeline:
    """Build a robust decision tree regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("dt", DecisionTreeRegressor(random_state=random_state)),
        ]
    )


def get_param_grid() -> dict:
    """Get a compact and stable hyperparameter grid for decision tree."""
    return {
        "dt__max_depth": [3, 5, 8, 12, None],
        "dt__min_samples_split": [2, 5, 10, 20],
        "dt__min_samples_leaf": [1, 2, 4, 8],
        "dt__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
    }
