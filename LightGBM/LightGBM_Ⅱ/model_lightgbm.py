from __future__ import annotations

from typing import Dict, Tuple

from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_base_model(random_state: int = 42) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=random_state,
        verbosity=-1,
    )


def get_param_grid() -> Dict[str, list]:
    return {
        "n_estimators": [300, 600],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [15, 31],
        "max_depth": [4, 6, -1],
        "min_child_samples": [10, 20],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }


def get_estimator_and_param_grid(plan: str, random_state: int = 42) -> Tuple[object, Dict[str, list]]:
    """Return estimator and matching grid keys for supported plans."""
    model = get_base_model(random_state=random_state)
    grid = get_param_grid()

    if plan == "plan1":
        return model, grid

    if plan == "plan2":
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        pipeline_grid = {f"model__{k}": v for k, v in grid.items()}
        return pipeline, pipeline_grid

    if plan == "plan3":
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        pipeline_grid = {f"model__{k}": v for k, v in grid.items()}
        return pipeline, pipeline_grid

    if plan == "planB":
        # Baseline control: let LightGBM handle missing values directly.
        return model, grid

    raise ValueError(f"Unsupported plan: {plan}")
