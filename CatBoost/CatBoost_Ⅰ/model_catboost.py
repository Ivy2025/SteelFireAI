from __future__ import annotations

from typing import Dict

from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_param_grid() -> Dict[str, list]:
    return {
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__iterations": [200, 500, 800],
        "model__l2_leaf_reg": [1, 3, 5],
    }


def build_pipeline(plan: str, random_state: int = 42) -> Pipeline:
    if plan not in {"plan1", "plan2"}:
        raise ValueError(f"Unknown plan: {plan}")

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_state=random_state,
        verbose=0,
    )

    if plan == "plan1":
        return Pipeline([
            ("model", model),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])
