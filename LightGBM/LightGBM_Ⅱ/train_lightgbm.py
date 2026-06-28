from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from evaluate import regression_metrics
from model_lightgbm import get_estimator_and_param_grid
from visualize import plot_feature_importance, plot_scatter


def _compact_grid(param_grid: Dict[str, list]) -> Dict[str, list]:
    """Fallback tiny grid for memory-constrained retry."""
    return {k: [v[0]] for k, v in param_grid.items() if isinstance(v, list) and len(v) > 0}


@dataclass
class TrainResult:
    metric_row: Dict
    oof_df: pd.DataFrame
    test_pred_df: pd.DataFrame
    error: Optional[str] = None


def _ensure_dirs(base_result_dir: str) -> Dict[str, str]:
    dirs = {
        "metrics": os.path.join(base_result_dir, "metrics"),
        "predictions": os.path.join(base_result_dir, "predictions"),
        "models": os.path.join(base_result_dir, "models"),
        "tuning": os.path.join(base_result_dir, "tuning"),
        "figures": os.path.join(base_result_dir, "figures"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _determine_n_splits(train_sample_count: int) -> int:
    if train_sample_count < 3:
        raise ValueError(f"Not enough train samples for KFold: {train_sample_count} < 3")
    return min(5, train_sample_count)


def _get_model_from_estimator(estimator):
    # For plan2 pipeline, trained LightGBM model lives in named_steps["model"].
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps["model"]
    return estimator


def _enforce_non_negative(y_pred: np.ndarray) -> np.ndarray:
    """Apply physical constraint: target predictions cannot be negative."""
    return np.clip(np.asarray(y_pred, dtype=float), a_min=0.0, a_max=None)


def train_one_target_one_plan(
    data_df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    plan: str,
    train_sample_ids: List,
    test_sample_ids: List,
    output_prefix: str,
    result_dir: str,
    random_state: int = 42,
    n_jobs: int = 1,
) -> TrainResult:
    dirs = _ensure_dirs(result_dir)

    try:
        df = data_df.copy()
        df = df[df[target].notna()].copy()
        if df.empty:
            raise ValueError(f"No available samples after dropping NA target for {target}")

        df["is_train"] = df["sample_id"].isin(train_sample_ids)
        train_df = df[df["is_train"]].copy()
        test_df = df[df["sample_id"].isin(test_sample_ids)].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Empty split for target={target}, plan={plan}. "
                f"train={len(train_df)}, test={len(test_df)}"
            )

        X_train = train_df[feature_cols].copy()
        y_train = train_df[target].astype(float).values
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target].astype(float).values

        n_splits = _determine_n_splits(len(train_df))

        estimator, param_grid = get_estimator_and_param_grid(plan=plan, random_state=random_state)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scorer = make_scorer(r2_score)

        used_fallback_grid = False
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            verbose=0,
            return_train_score=False,
        )
        try:
            grid.fit(X_train, y_train)
        except MemoryError:
            used_fallback_grid = True
            fallback_grid = _compact_grid(param_grid)
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=fallback_grid,
                scoring=scorer,
                cv=cv,
                refit=True,
                n_jobs=1,
                verbose=0,
                return_train_score=False,
            )
            grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        best_params = grid.best_params_

        train_pred = _enforce_non_negative(best_estimator.predict(X_train))
        test_pred = _enforce_non_negative(best_estimator.predict(X_test))

        train_m = regression_metrics(y_train, train_pred, prefix="train")
        test_m = regression_metrics(y_test, test_pred, prefix="test")

        # OOF predictions from CV with best params
        oof_pred = np.full(shape=len(train_df), fill_value=np.nan, dtype=float)
        fold_arr = np.full(shape=len(train_df), fill_value=-1, dtype=int)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            fold_model = clone(best_estimator)
            fold_model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            oof_pred[va_idx] = _enforce_non_negative(fold_model.predict(X_train.iloc[va_idx]))
            fold_arr[va_idx] = fold_idx

        if np.isnan(oof_pred).any() or (fold_arr < 1).any():
            raise RuntimeError("OOF prediction was not fully populated.")

        oof_m = regression_metrics(y_train, oof_pred, prefix="oof")

        metric_row = {
            "plan": plan,
            "target": target,
            "best_params": json.dumps(best_params, ensure_ascii=False),
            "train_sample_count": int(len(train_df)),
            "test_sample_count": int(len(test_df)),
            "train_group_count": int(train_df["diff_steel"].nunique()),
            "test_group_count": int(test_df["diff_steel"].nunique()),
            "n_splits_used": int(n_splits),
            "cv_R2_mean": float(grid.best_score_),
            "cv_R2_std": float(grid.cv_results_["std_test_score"][grid.best_index_]),
            "fallback_grid_used": used_fallback_grid,
            "train_R2": train_m["train_R2"],
            "train_MAE": train_m["train_MAE"],
            "train_P20": train_m["train_P20"],
            "test_R2": test_m["test_R2"],
            "test_MAE": test_m["test_MAE"],
            "test_P20": test_m["test_P20"],
            "oof_R2": oof_m["oof_R2"],
            "oof_MAE": oof_m["oof_MAE"],
            "oof_P20": oof_m["oof_P20"],
        }

        oof_df = pd.DataFrame(
            {
                "sample_id": train_df["sample_id"].astype(int).values,
                "diff_steel": train_df["diff_steel"].values,
                "plan": plan,
                "target": target,
                "fold": fold_arr,
                "y_true": y_train,
                "y_pred": oof_pred,
            }
        )

        test_pred_df = pd.DataFrame(
            {
                "sample_id": test_df["sample_id"].astype(int).values,
                "diff_steel": test_df["diff_steel"].values,
                "plan": plan,
                "target": target,
                "y_true": y_test,
                "y_pred": test_pred,
                "split": "test",
            }
        )

        model_path = os.path.join(dirs["models"], f"{output_prefix}_{plan}_{target}.pkl")
        joblib.dump(best_estimator, model_path)

        params_path = os.path.join(dirs["tuning"], f"{output_prefix}_{plan}_{target}_best_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        test_scatter_path = os.path.join(
            dirs["figures"], f"{output_prefix}_{plan}_{target}_test_scatter.png"
        )
        plot_scatter(
            y_true=y_test,
            y_pred=test_pred,
            metrics={"R2": test_m["test_R2"], "MAE": test_m["test_MAE"], "P20": test_m["test_P20"]},
            title=f"{plan} | {target} | Test",
            save_path=test_scatter_path,
        )

        oof_scatter_path = os.path.join(
            dirs["figures"], f"{output_prefix}_{plan}_{target}_oof_scatter.png"
        )
        plot_scatter(
            y_true=y_train,
            y_pred=oof_pred,
            metrics={"R2": oof_m["oof_R2"], "MAE": oof_m["oof_MAE"], "P20": oof_m["oof_P20"]},
            title=f"{plan} | {target} | OOF",
            save_path=oof_scatter_path,
        )

        model_obj = _get_model_from_estimator(best_estimator)
        importances = model_obj.feature_importances_
        fi_path = os.path.join(
            dirs["figures"], f"{output_prefix}_{plan}_{target}_feature_importance.png"
        )
        plot_feature_importance(
            feature_names=feature_cols,
            importances=importances,
            title=f"{plan} | {target} | Feature Importance",
            save_path=fi_path,
        )

        return TrainResult(metric_row=metric_row, oof_df=oof_df, test_pred_df=test_pred_df, error=None)

    except Exception as exc:  # noqa: BLE001
        err_text = f"target={target}, plan={plan}, error={type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        metric_row = {
            "plan": plan,
            "target": target,
            "best_params": "",
            "train_sample_count": 0,
            "test_sample_count": 0,
            "train_group_count": 0,
            "test_group_count": 0,
            "n_splits_used": 0,
            "cv_R2_mean": np.nan,
            "cv_R2_std": np.nan,
            "train_R2": np.nan,
            "train_MAE": np.nan,
            "train_P20": np.nan,
            "test_R2": np.nan,
            "test_MAE": np.nan,
            "test_P20": np.nan,
            "oof_R2": np.nan,
            "oof_MAE": np.nan,
            "oof_P20": np.nan,
            "error": err_text,
        }
        return TrainResult(metric_row=metric_row, oof_df=pd.DataFrame(), test_pred_df=pd.DataFrame(), error=err_text)


def make_random_split_for_target(
    data_df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List, List]:
    target_df = data_df[data_df[target].notna()].copy()
    sample_ids = pd.Series(target_df["sample_id"].astype(int).unique())

    n_samples = len(sample_ids)
    if n_samples < 3:
        raise ValueError(f"Target {target} has less than 3 samples: {n_samples}")

    shuffled = sample_ids.sample(frac=1.0, random_state=random_state).tolist()

    n_test_samples = max(1, int(round(n_samples * test_size)))
    n_test_samples = min(n_test_samples, n_samples - 1)

    test_sample_ids = shuffled[:n_test_samples]
    train_sample_ids = shuffled[n_test_samples:]

    if len(set(train_sample_ids)) < 3:
        raise ValueError(
            f"Target {target} train samples after split < 3. "
            f"train_samples={len(set(train_sample_ids))}, total_samples={n_samples}"
        )

    return train_sample_ids, test_sample_ids
