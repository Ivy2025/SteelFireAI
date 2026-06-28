from __future__ import annotations

import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split

from data_loader import get_target_data
from evaluate import regression_metrics
from model_rf import build_rf_pipeline, get_rf_param_grid


def train_single_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, object]:
    """Train one RF model for one target and return all artifacts."""
    try:
        x, y, sample_id, diff_steel = get_target_data(df=df, target=target, feature_columns=feature_columns)

        (
            x_train,
            x_test,
            y_train,
            y_test,
            sid_train,
            sid_test,
            diff_train,
            diff_test,
        ) = train_test_split(
            x,
            y,
            sample_id,
            diff_steel,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        pipeline = build_rf_pipeline(random_state=random_state)
        param_grid = get_rf_param_grid()

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="r2",
            cv=kfold,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        grid.fit(x_train, y_train)

        best_model = grid.best_estimator_
        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)

        train_metrics = regression_metrics(y_train.values, y_train_pred, prefix="train")
        test_metrics = regression_metrics(y_test.values, y_test_pred, prefix="test")

        # Use fold-specific fitted models under the same best hyperparameters.
        oof_pred = cross_val_predict(
            estimator=best_model,
            X=x_train,
            y=y_train,
            cv=kfold,
            n_jobs=-1,
            method="predict",
        )

        fold_ids = np.empty(len(x_train), dtype=int)
        for fold, (_, valid_idx) in enumerate(kfold.split(x_train, y_train), start=1):
            fold_ids[valid_idx] = fold

        oof_metrics_raw = regression_metrics(y_train.values, oof_pred, prefix="oof")

        oof_df = pd.DataFrame(
            {
                "sample_id": sid_train.values,
                "diff_steel": diff_train.values,
                "target": target,
                "y_true": y_train.values,
                "y_pred": oof_pred,
                "split": "train_oof",
                "fold": fold_ids,
            }
        )

        test_df = pd.DataFrame(
            {
                "sample_id": sid_test.values,
                "diff_steel": diff_test.values,
                "target": target,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
                "split": "test",
            }
        )

        feature_importance = pd.Series(
            best_model.named_steps["rf"].feature_importances_,
            index=feature_columns,
            name="importance",
        )

        result = {
            "target": target,
            "best_params": grid.best_params_,
            "train_sample_count": int(len(x_train)),
            "test_sample_count": int(len(x_test)),
            "cv_R2_mean": float(grid.cv_results_["mean_test_score"][grid.best_index_]),
            "cv_R2_std": float(grid.cv_results_["std_test_score"][grid.best_index_]),
            **train_metrics,
            **test_metrics,
        }

        return {
            "success": True,
            "result": result,
            "model": best_model,
            "oof_predictions": oof_df,
            "test_predictions": test_df,
            "feature_importance": feature_importance,
            "oof_metrics": {
                "R2": oof_metrics_raw["oof_R2"],
                "MAE": oof_metrics_raw["oof_MAE"],
                "RMSE": oof_metrics_raw["oof_RMSE"],
                "P20": oof_metrics_raw["oof_P20"],
            },
            "error": None,
        }

    except Exception as exc:  # Keep pipeline running for remaining targets.
        return {
            "success": False,
            "result": {
                "target": target,
                "best_params": "{}",
                "train_sample_count": 0,
                "test_sample_count": 0,
                "cv_R2_mean": np.nan,
                "cv_R2_std": np.nan,
                "train_R2": np.nan,
                "train_MAE": np.nan,
                "train_RMSE": np.nan,
                "train_P20": np.nan,
                "test_R2": np.nan,
                "test_MAE": np.nan,
                "test_RMSE": np.nan,
                "test_P20": np.nan,
                "error": str(exc),
            },
            "model": None,
            "oof_predictions": None,
            "test_predictions": None,
            "feature_importance": None,
            "oof_metrics": None,
            "error": f"{exc}\n{traceback.format_exc()}",
        }
