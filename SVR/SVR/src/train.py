# src/train.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    GridSearchCV,
    cross_val_predict,
)

from .data_loader import load_target_data
from .evaluate import regression_metrics, save_test_scatter_plot
from .model_svr import (
    build_svr_pipeline,
    get_svr_param_grid,
    build_svr_pca_pipeline,
    get_svr_pca_param_grid,
)


DEFAULT_TARGETS = ["fy_reduction", "fu_reduction", "E_reduction"]


def _build_pred_frame(
    target: str,
    split_name: str,
    y_true,
    y_pred,
    group_values,
    temperature_values,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, 1.0, np.abs(y_true))
    rel_err = np.abs((y_pred - y_true) / denom)

    return pd.DataFrame(
        {
            "target": target,
            "split": split_name,  # train_fit / test_holdout / cv_oof_train
            "diff_steel": group_values,
            "temperature": temperature_values,
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_rel_error": rel_err,
            "within_20pct": (rel_err <= 0.20).astype(int),
        }
    )


def train_single_target_svr(
    data_path: str | Path,
    target: str,
    pipeline_builder=build_svr_pipeline,
    param_grid_getter=get_svr_param_grid,
    random_state: int = 42,
    test_size: float = 0.2,
    n_splits: int = 5,
):
    X, y, groups, meta = load_target_data(data_path, target)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
    m_train, m_test = meta.iloc[train_idx], meta.iloc[test_idx]

    pipeline = pipeline_builder()
    param_grid = param_grid_getter()
    cv = GroupKFold(n_splits=n_splits)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train, groups=g_train)

    best_model = search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # 训练集 OOF 预测（用于输出 cv 预测文件）
    oof_model = pipeline_builder().set_params(**search.best_params_)
    y_train_oof = cross_val_predict(
        oof_model,
        X_train,
        y_train,
        groups=g_train,
        cv=cv,
        n_jobs=-1,
        method="predict",
    )

    train_metrics = regression_metrics(y_train, y_train_pred, prefix="train")
    test_metrics = regression_metrics(y_test, y_test_pred, prefix="test")
    cv_r2_mean = float(search.cv_results_["mean_test_score"][search.best_index_])
    cv_r2_std = float(search.cv_results_["std_test_score"][search.best_index_])

    metrics_row = {
        "test_R2": test_metrics["test_R2"],
        "test_MAE": test_metrics["test_MAE"],
        "test_RMSE": test_metrics["test_RMSE"],
        "test_P20": test_metrics["test_P20"],
        "train_R2": train_metrics["train_R2"],
        "train_MAE": train_metrics["train_MAE"],
        "train_RMSE": train_metrics["train_RMSE"],
        "train_P20": train_metrics["train_P20"],
        "cv_R2_mean": cv_r2_mean,
        "cv_R2_std": cv_r2_std,
        "target": target,
    }

    pred_train_fit = _build_pred_frame(
        target=target,
        split_name="train_fit",
        y_true=y_train.values,
        y_pred=y_train_pred,
        group_values=g_train.values,
        temperature_values=m_train["temperature"].values,
    )
    pred_test = _build_pred_frame(
        target=target,
        split_name="test_holdout",
        y_true=y_test.values,
        y_pred=y_test_pred,
        group_values=g_test.values,
        temperature_values=m_test["temperature"].values,
    )
    pred_oof = _build_pred_frame(
        target=target,
        split_name="cv_oof_train",
        y_true=y_train.values,
        y_pred=y_train_oof,
        group_values=g_train.values,
        temperature_values=m_train["temperature"].values,
    )

    pred_all = pd.concat([pred_train_fit, pred_test, pred_oof], axis=0, ignore_index=True)

    return {
        "best_model": best_model,
        "best_params": search.best_params_,
        "metrics_row": metrics_row,
        "predictions": pred_all,
    }


def run_svr_experiment(
    data_path: str | Path,
    output_root: str | Path = "results",
    targets: list[str] | None = None,
    random_state: int = 42,
    file_suffix: str = "",
    pipeline_builder=build_svr_pipeline,
    param_grid_getter=get_svr_param_grid,
):
    if targets is None:
        targets = DEFAULT_TARGETS

    output_root = Path(output_root)
    metrics_dir = output_root / "metrics"
    pred_dir = output_root / "predictions"
    model_dir = output_root / "models"
    tuning_dir = output_root / "tuning"
    fig_dir = output_root / "figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_preds = []
    scatter_paths = {}

    for target in targets:
        result = train_single_target_svr(
            data_path=data_path,
            target=target,
            pipeline_builder=pipeline_builder,
            param_grid_getter=param_grid_getter,
            random_state=random_state,
            test_size=0.2,
            n_splits=5,
        )

        all_metrics.append(result["metrics_row"])
        all_preds.append(result["predictions"])

        model_path = model_dir / f"svr_{target}{file_suffix}.pkl"
        joblib.dump(result["best_model"], model_path)

        params_path = tuning_dir / f"svr_{target}_best_params{file_suffix}.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(result["best_params"], f, ensure_ascii=False, indent=2)

        test_df = result["predictions"].loc[result["predictions"]["split"] == "test_holdout"]
        scatter_path_basic = fig_dir / f"svr_{target}_test_scatter_perfect{file_suffix}.png"
        save_test_scatter_plot(
            y_true=test_df["y_true"].values,
            y_pred=test_df["y_pred"].values,
            target=target,
            save_path=scatter_path_basic,
            add_p20_band=False,
        )

        scatter_path_with_p20 = fig_dir / f"svr_{target}_test_scatter_perfect_p20{file_suffix}.png"
        save_test_scatter_plot(
            y_true=test_df["y_true"].values,
            y_pred=test_df["y_pred"].values,
            target=target,
            save_path=scatter_path_with_p20,
            add_p20_band=True,
        )

        scatter_paths[target] = {
            "perfect_only": str(scatter_path_basic),
            "perfect_and_p20": str(scatter_path_with_p20),
        }

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, axis=0, ignore_index=True)

    metrics_path = metrics_dir / f"svr_metrics{file_suffix}.csv"
    preds_path = pred_dir / f"svr_cv_predictions{file_suffix}.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")

    return {
        "metrics_path": str(metrics_path),
        "predictions_path": str(preds_path),
        "scatter_paths": scatter_paths,
        "metrics_df": metrics_df,
    }


def run_svr_planb_pca_experiment(
    data_path: str | Path,
    output_root: str | Path = "results",
    targets: list[str] | None = None,
    random_state: int = 42,
):
    return run_svr_experiment(
        data_path=data_path,
        output_root=output_root,
        targets=targets,
        random_state=random_state,
        file_suffix="_planB_pca",
        pipeline_builder=build_svr_pca_pipeline,
        param_grid_getter=get_svr_pca_param_grid,
    )