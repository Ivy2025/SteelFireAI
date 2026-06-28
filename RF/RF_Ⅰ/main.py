# main.py
import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import GroupKFold, cross_val_score

from src.train import train_model, save_model
from src.evaluate import evaluate_model
from src.visualize import (
    plot_chemical_feature_importance,
    plot_prediction_vs_actual,
    plot_prediction_vs_actual_abs_error,
)


if __name__ == "__main__":

    # ===== 数据集路径（你刚刚给的）=====
    csv_path = "data/processed/steel_v2_cleaned_r2.csv"

    # ===== 要预测的强度指标 =====
    targets = [
        "fy_reduction",
        "fu_reduction",
        "E_reduction"
    ]

    # ===== 化学成分特征 =====
    chemical_features = [
        "C", "N", "Si", "Mn", "P", "S",
        "Cr", "Ni", "Mo", "V",
        "Nb", "Ti", "Al", "Cu"
    ]

    # ===== 输出目录 =====
    fig_dir = "results/figures"
    metrics_dir = "results/metrics"
    models_dir = "results/models"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    results = []
    abs_error_results = []

    # 可以通过这个字典调整随机森林超参数以控制过拟合。
    rf_params = {
        # 默认设置，若不调参则使用这些参数
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
    }

    # 是否对每个目标进行网格搜索调参
    tune = True
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 0.5, None]
    }

    for target in targets:

        print(f"\nTraining model for {target} ...")

        model, X_train, X_test, y_train, y_test, feature_names, groups_train = train_model(
            csv_path=csv_path,
            target_col=target,
            targets=targets,
            rf_params=rf_params,
            tune=tune,
            param_grid=param_grid if tune else None,
            cv=5
        )

        # 计算并记录 CV R2（均值与标准差）
        if groups_train is not None:
            cv_splitter = GroupKFold(n_splits=5)
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_splitter,
                groups=groups_train,
                scoring='r2',
                n_jobs=-1,
            )
            print(f"GroupKFold CV R2 scores: {cv_scores}")
        else:
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring='r2',
                n_jobs=-1,
            )
            print(f"KFold CV R2 scores: {cv_scores}")

        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"CV mean R2: {cv_mean:.4f}, std: {cv_std:.4f}\n")


        # —— 学习曲线 —— #
        curve_path = os.path.join(fig_dir, f"{target}_learning_curve.png")
        from src.visualize import plot_learning_curve
        plot_learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            title=f"Learning Curve - {target}",
            save_path=curve_path
        )

        # —— 评价 —— #
        metrics = evaluate_model(
            model,
            X_test, y_test,
            X_train=X_train, y_train=y_train
        )
        metrics["cv_R2_mean"] = cv_mean
        metrics["cv_R2_std"] = cv_std
        metrics["target"] = target
        results.append(metrics)

        y_test_pred = model.predict(X_test)
        abs_error = np.abs(np.asarray(y_test) - np.asarray(y_test_pred))
        abs_error_results.append({
            "target": target,
            "test_sample_count": int(len(y_test)),
            "test_abs_error_le_0p1_count": int(np.sum(abs_error <= 0.1)),
            "test_abs_error_le_0p1_ratio": float(np.mean(abs_error <= 0.1)),
            "test_abs_error_le_0p15_count": int(np.sum(abs_error <= 0.15)),
            "test_abs_error_le_0p15_ratio": float(np.mean(abs_error <= 0.15)),
            "test_abs_error_le_0p2_count": int(np.sum(abs_error <= 0.2)),
            "test_abs_error_le_0p2_ratio": float(np.mean(abs_error <= 0.2)),
        })

        # —— 预测值与真实值散点图 —— #
        scatter_path = os.path.join(
            fig_dir,
            f"{target}_pred_vs_actual.png"
        )

        plot_prediction_vs_actual(
            model=model,
            X_data=X_test,
            y_data=y_test,
            target_name=target,
            save_path=scatter_path
        )

        # —— 只保留温度<800的散点图 —— #
        if hasattr(X_test, 'columns') and 'temperature' in X_test.columns:
            temp_mask = X_test['temperature'] < 800
            if temp_mask.any():
                scatter_path_800 = os.path.join(
                    fig_dir,
                    f"{target}_pred_vs_actual_800.png"
                )
                plot_prediction_vs_actual(
                    model=model,
                    X_data=X_test[temp_mask],
                    y_data=y_test[temp_mask],
                    target_name=target,
                    save_path=scatter_path_800
                )

        scatter_abs_err_01_path = os.path.join(
            fig_dir,
            f"{target}_pred_vs_actual_abs_error_0p1.png"
        )
        plot_prediction_vs_actual_abs_error(
            model=model,
            X_data=X_test,
            y_data=y_test,
            target_name=target,
            save_path=scatter_abs_err_01_path,
            abs_error=0.1
        )

        scatter_abs_err_015_path = os.path.join(
            fig_dir,
            f"{target}_pred_vs_actual_abs_error_0p15.png"
        )
        plot_prediction_vs_actual_abs_error(
            model=model,
            X_data=X_test,
            y_data=y_test,
            target_name=target,
            save_path=scatter_abs_err_015_path,
            abs_error=0.15
        )

        scatter_abs_err_02_path = os.path.join(
            fig_dir,
            f"{target}_pred_vs_actual_abs_error_0p2.png"
        )
        plot_prediction_vs_actual_abs_error(
            model=model,
            X_data=X_test,
            y_data=y_test,
            target_name=target,
            save_path=scatter_abs_err_02_path,
            abs_error=0.2
        )

        # —— 化学成分重要性 —— #
        fig_path = os.path.join(
            fig_dir,
            f"{target}_chemical_importance.png"
        )

        plot_chemical_feature_importance(
            model=model,
            feature_names=feature_names,
            chemical_features=chemical_features,
            target_name=target,
            save_path=fig_path,
            top_k=10
        )

        # —— 保存模型 —— #
        model_path = os.path.join(
            models_dir,
            f"{target}_model.pkl"
        )
        save_model(model, model_path)
        print(f"✓ Model saved to {model_path}")

    print("\nModel performance summary:")
    for r in results:
        print(r)

    # ===== 保存指标 =====
    # 保存为 JSON 格式
    json_path = os.path.join(metrics_dir, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Metrics saved to {json_path}")

    # 保存为 CSV 格式
    csv_path = os.path.join(metrics_dir, "metrics.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✓ Metrics saved to {csv_path}")

    abs_error_csv_path = os.path.join(metrics_dir, "test_abs_error_coverage.csv")
    abs_error_df = pd.DataFrame(abs_error_results)
    abs_error_df.to_csv(abs_error_csv_path, index=False)
    print(f"✓ Absolute error coverage saved to {abs_error_csv_path}")

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)