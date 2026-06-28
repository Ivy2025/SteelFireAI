import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

from data_loader import prepare_target_data
from evaluate import regression_metrics
from model_knn import build_knn_pipeline, get_param_grid


def _safe_cv_splits(train_groups: pd.Series, max_splits: int = 5) -> int:
    group_count = int(train_groups.nunique())
    if group_count < 2:
        raise ValueError("训练集可用分组数小于2，无法进行 GroupKFold 交叉验证。")
    return min(max_splits, group_count)


def _build_file_stem(prefix: str, target: str) -> str:
    if prefix:
        return f"{prefix}_{target}"
    return target


def train_one_target(
    df: pd.DataFrame,
    target: str,
    seed: int,
    test_size: float,
    output_dirs: Dict[str, Path],
    prefix: str,
) -> Dict[str, object]:
    X, y, groups, diff_steel = prepare_target_data(df, target=target)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)
    diff_train = diff_steel.iloc[train_idx].reset_index(drop=True)
    diff_test = diff_steel.iloc[test_idx].reset_index(drop=True)

    n_splits = _safe_cv_splits(groups_train, max_splits=5)
    cv = GroupKFold(n_splits=n_splits)

    pipeline = build_knn_pipeline()
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

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    train_m = regression_metrics(y_train, train_pred)
    test_m = regression_metrics(y_test, test_pred)

    best_idx = int(grid.best_index_)
    cv_mean = float(grid.cv_results_["mean_test_score"][best_idx])
    cv_std = float(grid.cv_results_["std_test_score"][best_idx])

    metrics_row = {
        "target": target,
        "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
        "train_sample_count": int(len(X_train)),
        "test_sample_count": int(len(X_test)),
        "train_group_count": int(groups_train.nunique()),
        "test_group_count": int(groups_test.nunique()),
        "cv_fold_count": int(n_splits),
        "cv_R2_mean": cv_mean,
        "cv_R2_std": cv_std,
        "train_R2": train_m["R2"],
        "train_MAE": train_m["MAE"],
        "train_RMSE": train_m["RMSE"],
        "train_P20": train_m["P20"],
        "test_R2": test_m["R2"],
        "test_MAE": test_m["MAE"],
        "test_RMSE": test_m["RMSE"],
        "test_P20": test_m["P20"],
    }

    stem = _build_file_stem(prefix, target)

    model_path = output_dirs["models"] / f"{stem}.pkl"
    params_path = output_dirs["tuning"] / f"{stem}_best_params.json"

    joblib.dump(best_model, model_path)

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(grid.best_params_, f, ensure_ascii=False, indent=2)

    test_pred_df = pd.DataFrame(
        {
            "diff_steel": diff_test,
            "target": target,
            "y_true": y_test,
            "y_pred": test_pred,
            "split": "test",
        }
    )

    _ = diff_train  # 保留变量便于后续扩展训练集预测输出

    return {
        "metrics_row": metrics_row,
        "test_predictions": test_pred_df,
    }
