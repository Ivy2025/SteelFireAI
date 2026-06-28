from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV

from data_loader import FEATURE_COLUMNS, GROUP_COLUMN, PreparedData
from evaluate import regression_metrics
from model_dt import build_dt_pipeline, get_param_grid
from visualize import (
    ensure_dir,
    plot_decision_tree_structure,
    plot_feature_importance,
    plot_true_vs_pred_scatter,
)


@dataclass
class TrainResult:
    success: bool
    target: str
    metrics: Optional[Dict] = None
    best_params: Optional[Dict] = None
    oof_df: Optional[pd.DataFrame] = None
    test_pred_df: Optional[pd.DataFrame] = None
    model: Optional[object] = None
    error: Optional[str] = None


def _determine_n_splits(groups_train: pd.Series, preferred: int = 5) -> int:
    unique_group_count = int(groups_train.nunique())
    if unique_group_count < 3:
        raise ValueError(
            f"Available training groups ({unique_group_count}) < 3, cannot run GroupKFold safely."
        )
    return min(preferred, unique_group_count)


def _metrics_text(r2: float, mae: float, rmse: float, p20: float) -> str:
    return f"R2={r2:.4f}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nP20={p20:.4f}"


def train_single_target(
    prepared: PreparedData,
    target: str,
    prefix: str,
    random_state: int,
    results_root: str,
) -> TrainResult:
    try:
        X = prepared.X
        y = prepared.y
        groups = prepared.groups
        raw_subset = prepared.raw_subset

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        groups_train = groups.iloc[train_idx].copy()
        raw_train = raw_subset.iloc[train_idx].copy()

        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()
        groups_test = groups.iloc[test_idx].copy()
        raw_test = raw_subset.iloc[test_idx].copy()

        n_splits_used = _determine_n_splits(groups_train, preferred=5)
        cv = GroupKFold(n_splits=n_splits_used)

        pipeline = build_dt_pipeline(random_state=random_state)
        param_grid = get_param_grid()

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        grid.fit(X_train, y_train, groups=groups_train)

        best_model = grid.best_estimator_

        oof_predictions = np.zeros(len(X_train), dtype=float)
        oof_folds = np.full(len(X_train), -1, dtype=int)

        for fold_id, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train), start=1):
            fold_model = clone(best_model)
            fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            oof_predictions[val_idx] = fold_model.predict(X_train.iloc[val_idx])
            oof_folds[val_idx] = fold_id

        if np.any(oof_folds == -1):
            raise RuntimeError("OOF prediction failed: some training samples were not assigned to folds.")

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metrics = regression_metrics(y_train.to_numpy(), y_train_pred, prefix="train")
        test_metrics = regression_metrics(y_test.to_numpy(), y_test_pred, prefix="test")

        oof_r2_scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
            fold_model = clone(best_model)
            fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            fold_pred = fold_model.predict(X_train.iloc[val_idx])
            oof_r2_scores.append(r2_score(y_train.iloc[val_idx], fold_pred))

        metrics = {
            "target": target,
            "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
            "train_sample_count": int(len(X_train)),
            "test_sample_count": int(len(X_test)),
            "train_group_count": int(groups_train.nunique()),
            "test_group_count": int(groups_test.nunique()),
            "n_splits_used": int(n_splits_used),
            "cv_R2_mean": float(np.mean(oof_r2_scores)),
            "cv_R2_std": float(np.std(oof_r2_scores, ddof=0)),
        }
        metrics.update(train_metrics)
        metrics.update(test_metrics)

        oof_df = pd.DataFrame(
            {
                "row_index": raw_train.index.to_numpy(),
                "diff_steel": raw_train[GROUP_COLUMN].to_numpy(),
                "target": target,
                "fold": oof_folds,
                "y_true": y_train.to_numpy(),
                "y_pred": oof_predictions,
            }
        )

        test_pred_df = pd.DataFrame(
            {
                "row_index": raw_test.index.to_numpy(),
                "diff_steel": raw_test[GROUP_COLUMN].to_numpy(),
                "target": target,
                "y_true": y_test.to_numpy(),
                "y_pred": y_test_pred,
                "split": "test",
            }
        )

        metrics_dir = os.path.join(s_root, "metrics")
        pred_dir = os.path.join(results_root, "predictions")
        model_dir = os.path.join(results_root, "models")
        tuning_dir = os.path.join(results_root, "tuning")
        fig_dir = os.path.join(results_root, "figures")
        for d in [metrics_dir, pred_dir, model_dir, tuning_dir, fig_dir]:
            ensure_dir(d)

        model_path = os.path.join(model_dir, f"{prefix}_{target}.pkl")
        joblib.dump(best_model, model_path)

        best_param_path = os.path.join(tuning_dir, f"{prefix}_{target}_best_params.json")
        with open(best_param_path, "w", encoding="utf-8") as f:
            json.dump(grid.best_params_, f, ensure_ascii=False, indent=2)

        test_text = _metrics_text(
            test_metrics["test_R2"],
            test_metrics["test_MAE"],
            test_metrics["test_RMSE"],
            test_metrics["test_P20"],
        )
        plot_true_vs_pred_scatter(
            y_true=y_test.to_numpy(),
            y_pred=y_test_pred,
            title=f"{target} Test: y_true vs y_pred",
            metrics_text=test_text,
            save_path=os.path.join(fig_dir, f"{prefix}_{target}_test_scatter.png"),
        )

        oof_metrics = regression_metrics(y_train.to_numpy(), oof_predictions, prefix="oof")
        oof_text = _metrics_text(
            oof_metrics["oof_R2"],
            oof_metrics["oof_MAE"],
            oof_metrics["oof_RMSE"],
            oof_metrics["oof_P20"],
        )
        plot_true_vs_pred_scatter(
            y_true=y_train.to_numpy(),
            y_pred=oof_predictions,
            title=f"{target} OOF: y_true vs y_pred",
            metrics_text=oof_text,
            save_path=os.path.join(fig_dir, f"{prefix}_{target}_oof_scatter.png"),
        )

        dt_model = best_model.named_steps["dt"]
        importances = dt_model.feature_importances_
        plot_feature_importance(
            feature_names=FEATURE_COLUMNS,
            importances=importances,
            title=f"{target} Feature Importance",
            save_path=os.path.join(fig_dir, f"{prefix}_{target}_feature_importance.png"),
        )

        plot_decision_tree_structure(
            fitted_pipeline=best_model,
            feature_names=FEATURE_COLUMNS,
            title=f"{target} Decision Tree (max_depth=3)",
            save_path=os.path.join(fig_dir, f"{prefix}_{target}_tree.png"),
            max_depth=3,
        )

        return TrainResult(
            success=True,
            target=target,
            metrics=metrics,
            best_params=grid.best_params_,
            oof_df=oof_df,
            test_pred_df=test_pred_df,
            model=best_model,
        )
    except Exception as exc:
        return TrainResult(success=False, target=target, error=str(exc))
