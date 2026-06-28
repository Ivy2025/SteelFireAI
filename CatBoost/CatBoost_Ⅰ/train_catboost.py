from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

from evaluate import regression_metrics
from model_catboost import build_pipeline, get_param_grid
from visualize import plot_feature_importance, plot_scatter_with_bands


def split_group_holdout(
    groups: pd.Series,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.array(sorted(groups.astype(str).unique()))
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for outer split.")

    rng = np.random.default_rng(random_state)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_test_groups = int(round(n_groups * test_ratio))
    n_test_groups = max(1, min(n_test_groups, n_groups - 1))

    test_groups = set(shuffled[:n_test_groups])
    train_mask = ~groups.astype(str).isin(test_groups).to_numpy()
    test_mask = ~train_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Outer split produced empty train or test set.")

    return train_idx, test_idx, groups.iloc[train_idx].unique(), groups.iloc[test_idx].unique()


def _extract_model_params(best_params: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in best_params.items():
        if k.startswith("model__"):
            out[k.replace("model__", "", 1)] = v
    return out


def train_one_target_one_plan(
    target_data,
    target: str,
    plan: str,
    output_prefix: str,
    output_dirs: Dict[str, str],
    random_state: int = 42,
) -> Dict[str, object]:
    X = target_data.X
    y = target_data.y
    groups = target_data.groups
    sample_id = target_data.sample_id
    raw_df = target_data.df

    train_idx, test_idx, train_groups, test_groups = split_group_holdout(
        groups=groups,
        test_ratio=0.2,
        random_state=random_state,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)
    sample_id_train = sample_id.iloc[train_idx].reset_index(drop=True)
    sample_id_test = sample_id.iloc[test_idx].reset_index(drop=True)

    n_train_groups = len(np.unique(groups_train))
    if n_train_groups < 3:
        raise ValueError(
            f"Target {target} has only {n_train_groups} train groups; need at least 3 for GroupKFold."
        )

    n_splits = min(5, n_train_groups)

    base_pipeline = build_pipeline(plan=plan, random_state=random_state)
    param_grid = get_param_grid()
    inner_cv = GroupKFold(n_splits=n_splits)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train, groups=groups_train)

    best_estimator = grid.best_estimator_
    best_params_raw = grid.best_params_
    best_params_model = _extract_model_params(best_params_raw)

    oof_pred = np.empty(len(X_train), dtype=float)
    oof_fold = np.empty(len(X_train), dtype=int)

    oof_cv = GroupKFold(n_splits=n_splits)
    for fold_id, (tr_idx, va_idx) in enumerate(
        oof_cv.split(X_train, y_train, groups_train),
        start=1,
    ):
        fold_model = clone(base_pipeline)
        fold_model.set_params(**best_params_raw)
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_pred[va_idx] = fold_model.predict(X_train.iloc[va_idx])
        oof_fold[va_idx] = fold_id

    train_pred = best_estimator.predict(X_train)
    test_pred = best_estimator.predict(X_test)

    train_metrics = regression_metrics(y_train.to_numpy(), train_pred, prefix="train")
    test_metrics = regression_metrics(y_test.to_numpy(), test_pred, prefix="test")
    oof_metrics = regression_metrics(y_train.to_numpy(), oof_pred, prefix="oof")

    best_index = int(grid.best_index_)
    cv_r2_mean = float(grid.cv_results_["mean_test_score"][best_index])
    cv_r2_std = float(grid.cv_results_["std_test_score"][best_index])

    metrics_row = {
        "plan": plan,
        "target": target,
        "best_params": json.dumps(best_params_model, ensure_ascii=True),
        "train_sample_count": int(len(X_train)),
        "test_sample_count": int(len(X_test)),
        "train_group_count": int(len(train_groups)),
        "test_group_count": int(len(test_groups)),
        "n_splits_used": int(n_splits),
        "cv_R2_mean": cv_r2_mean,
        "cv_R2_std": cv_r2_std,
        **train_metrics,
        **test_metrics,
    }

    oof_df = pd.DataFrame(
        {
            "sample_id": sample_id_train,
            "diff_steel": groups_train,
            "plan": plan,
            "target": target,
            "fold": oof_fold,
            "y_true": y_train,
            "y_pred": oof_pred,
        }
    )

    test_df = pd.DataFrame(
        {
            "sample_id": sample_id_test,
            "diff_steel": groups_test,
            "plan": plan,
            "target": target,
            "y_true": y_test,
            "y_pred": test_pred,
            "split": "test",
        }
    )

    model_path = os.path.join(output_dirs["models"], f"{output_prefix}_{plan}_{target}.pkl")
    params_path = os.path.join(output_dirs["tuning"], f"{output_prefix}_{plan}_{target}_best_params.json")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(params_path), exist_ok=True)

    joblib.dump(best_estimator, model_path)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_model, f, ensure_ascii=True, indent=2)

    test_scatter_path = os.path.join(
        output_dirs["figures"],
        f"{output_prefix}_{plan}_{target}_test_scatter.png",
    )
    oof_scatter_path = os.path.join(
        output_dirs["figures"],
        f"{output_prefix}_{plan}_{target}_oof_scatter.png",
    )
    fi_path = os.path.join(
        output_dirs["figures"],
        f"{output_prefix}_{plan}_{target}_feature_importance.png",
    )

    plot_scatter_with_bands(
        y_true=y_test,
        y_pred=test_pred,
        title=f"{plan} {target} test scatter",
        metrics={
            "R2": test_metrics["test_R2"],
            "MAE": test_metrics["test_MAE"],
            "P20": test_metrics["test_P20"],
        },
        save_path=test_scatter_path,
    )

    plot_scatter_with_bands(
        y_true=y_train,
        y_pred=oof_pred,
        title=f"{plan} {target} oof scatter",
        metrics={
            "R2": oof_metrics["oof_R2"],
            "MAE": oof_metrics["oof_MAE"],
            "P20": oof_metrics["oof_P20"],
        },
        save_path=oof_scatter_path,
    )

    final_model = best_estimator.named_steps["model"]
    importances = final_model.get_feature_importance()
    plot_feature_importance(
        feature_names=target_data.feature_columns,
        importances=importances,
        title=f"{plan} {target} feature importance",
        save_path=fi_path,
    )

    return {
        "metrics_row": metrics_row,
        "oof_df": oof_df,
        "test_df": test_df,
    }
