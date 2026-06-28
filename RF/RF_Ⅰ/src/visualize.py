# src/visualize.py
import matplotlib
matplotlib.use("Agg")   # ⭐ 禁用 Tk

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve


def plot_prediction_vs_actual(
    model,
    X_data,
    y_data,
    target_name,
    save_path,
    dataset_label="Test"
):
    """
    绘制预测值与真实值的散点图
    """
    y_pred = model.predict(X_data)
    print(f"{dataset_label}样本数量:", len(y_data))
    
    plt.figure(figsize=(8, 6))
    
    min_val = min(y_data.min(), y_pred.min())
    max_val = max(y_data.max(), y_pred.max())
    
    color = 'blue' if dataset_label == 'Test' else 'gray'
    alpha = 0.6 if dataset_label == 'Test' else 0.5
    plt.scatter(y_data, y_pred, alpha=alpha, s=50, c=color, label=dataset_label, marker='o')
    
    # 绘制完美预测线（45度线）
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    # 添加 ±20% 偏差线
    # y = 1.2x 和 y = 0.8x
    plt.plot([min_val, max_val], [min_val * 1.2, max_val * 1.2], 'b--', lw=1, label='+20% deviation')
    plt.plot([min_val, max_val], [min_val * 0.8, max_val * 0.8], 'b--', lw=1, label='-20% deviation')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual ({dataset_label}) - {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_vs_actual_log(
    model,
    X_data,
    y_data,
    target_name,
    save_path,
    dataset_label="Test"
):
    """
    绘制对数坐标散点图，参考线定义与常规图一致（1:1 与 ±20%）。
    """
    y_pred = model.predict(X_data)

    # Log 坐标只能显示正值，过滤非正样本。
    y_true_arr = np.asarray(y_data)
    y_pred_arr = np.asarray(y_pred)
    positive_mask = (y_true_arr > 0) & (y_pred_arr > 0)

    if not np.any(positive_mask):
        raise ValueError(f"{dataset_label} 数据中无正值样本，无法绘制 Log 坐标散点图。")

    y_true_pos = y_true_arr[positive_mask]
    y_pred_pos = y_pred_arr[positive_mask]

    plt.figure(figsize=(8, 6))

    min_val = min(y_true_pos.min(), y_pred_pos.min())
    max_val = max(y_true_pos.max(), y_pred_pos.max())

    color = 'blue' if dataset_label == 'Test' else 'gray'
    alpha = 0.6 if dataset_label == 'Test' else 0.5
    plt.scatter(y_true_pos, y_pred_pos, alpha=alpha, s=50, c=color, label=dataset_label, marker='o')

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    plt.plot([min_val, max_val], [min_val * 1.2, max_val * 1.2], 'b--', lw=1, label='+20% deviation')
    plt.plot([min_val, max_val], [min_val * 0.8, max_val * 0.8], 'b--', lw=1, label='-20% deviation')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Values (log scale)')
    plt.ylabel('Predicted Values (log scale)')
    plt.title(f'Predicted vs Actual in Log Scale ({dataset_label}) - {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_vs_actual_abs_error(
    model,
    X_data,
    y_data,
    target_name,
    save_path,
    abs_error=0.1,
    dataset_label="Test"
):
    """
    绘制散点图，保留 1:1 线，新增单一绝对误差参考线：±abs_error。
    """
    y_pred = model.predict(X_data)

    plt.figure(figsize=(8, 6))

    min_val = min(y_data.min(), y_pred.min())
    max_val = max(y_data.max(), y_pred.max())

    color = 'blue' if dataset_label == 'Test' else 'gray'
    alpha = 0.6 if dataset_label == 'Test' else 0.5
    plt.scatter(y_data, y_pred, alpha=alpha, s=50, c=color, label=dataset_label, marker='o')

    x_line = np.array([min_val, max_val])

    plt.plot(x_line, x_line, 'r--', lw=2, label='Perfect prediction')
    plt.plot(x_line, x_line + abs_error, 'g--', lw=1.5, label=f'+{abs_error} abs error')
    plt.plot(x_line, x_line - abs_error, 'g--', lw=1.5, label=f'-{abs_error} abs error')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual with ±{abs_error} Absolute Error ({dataset_label}) - {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# 新增：只含有perfect prediction参考线的散点图
def plot_perfect_prediction_scatter(
    model, 
    X_data, 
    y_data, 
    target_name, 
    save_path, 
    dataset_label="Test"
):
     """
     只绘制散点和完美预测线（45度线）
     """
     y_pred = model.predict(X_data)
     plt.figure(figsize=(8, 6))
     
     min_val = min(y_data.min(), y_pred.min())
     max_val = max(y_data.max(), y_pred.max())
     
     color = 'blue' if dataset_label == 'Test' else 'gray'
     alpha = 0.6 if dataset_label == 'Test' else 0.5
     plt.scatter(y_data, y_pred, alpha=alpha, s=50, c=color, label=dataset_label, marker='o')
     
     plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
     plt.xlabel('True Values')
     plt.ylabel('Predicted Values')
     plt.title(f'Perfect Prediction Reference ({dataset_label}) - {target_name}')
     plt.legend()
     plt.grid(True, alpha=0.3)
     plt.tight_layout()
     plt.savefig(save_path, dpi=300)
     plt.close()


def plot_cv_aggregate_scatter(y_true_cv, y_pred_cv, target_name, save_path):
    """
    绘制CV所有折测试集样本汇总散点图（真实值 vs 预测值）。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_cv, y_pred_cv, alpha=0.6, s=45)

    min_val = min(np.min(y_true_cv), np.min(y_pred_cv))
    max_val = max(np.max(y_true_cv), np.max(y_pred_cv))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    plt.xlabel('True Values (CV test folds)')
    plt.ylabel('Predicted Values (CV test folds)')
    plt.title(f'CV Aggregated Predicted vs Actual - {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_chemical_feature_importance(
    model,
    feature_names,
    chemical_features,
    target_name,
    save_path,
    top_k=10
):
    """
    只可视化化学成分特征的重要性
    """

    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # 只保留化学成分
    df = df[df["feature"].isin(chemical_features)]

    # 排序
    df = df.sort_values("importance", ascending=False).head(top_k)

    # 作图
    plt.figure(figsize=(6, 0.35 * len(df)))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()

    plt.xlabel("Feature importance")
    plt.title(f"Chemical feature importance ({target_name})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return df


def plot_learning_curve(
    estimator,
    X,
    y,
    title,
    save_path,
    cv=5,
    scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10),
):
    """
    绘制学习曲线：训练集大小 vs 训练/验证分数。

    estimator: 需实现 fit/predict 的模型
    X, y: 数据
    title: 图表标题
    save_path: 图片保存路径
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross‑val score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    plt.title(title)
    plt.xlabel('Training set size')
    plt.ylabel(scoring)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
