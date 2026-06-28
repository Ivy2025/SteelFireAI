# src/train.py
from src.data_loader import load_data
from src.model_rf import build_rf_model
import joblib
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score




def tune_rf_model(X_train, y_train, param_grid, groups=None, cv=5):
    """
    在训练集上运行网格搜索以找到最优随机森林参数。

    如果提供了 groups，会使用 GroupKFold(cv) 作为交叉验证策略，
    并在 fit 时传递 groups，以保证相同组不被拆分到不同折中。

    返回最佳估计器、参数和 cv 结果。
    """
    base = build_rf_model(random_state=42)
    if groups is not None:
        splitter = GroupKFold(n_splits=cv)
        cv_strategy = splitter
    else:
        cv_strategy = cv

    gs = GridSearchCV(
        base,
        param_grid,
        cv=cv_strategy,
        scoring='r2',
        n_jobs=-1,
    )
    if groups is not None:
        gs.fit(X_train, y_train, groups=groups)
    else:
        gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.cv_results_


def train_model(csv_path, target_col, targets=None, rf_params=None, tune=False, param_grid=None, cv=5):
    """
    训练单指标随机森林模型
    
    参数:
        csv_path: 数据 CSV 路径
        target_col: 当前训练的目标列名
        targets: 可选，所有目标列名列表，用于剔除其他目标作为特征
        rf_params: 可选，传给 RandomForestRegressor 的超参数字典
        tune: 是否执行网格搜索调参
        param_grid: 调参用的参数字典
        cv: 若进行交叉验证/调参，使用的折数（GroupKFold）

    返回:
        model: 训练好的 RandomForestRegressor 模型
        X_train, X_test, y_train, y_test: 划分好的数据集
        feature_names: 特征列名列表
        groups_train: 训练集的分组信息（若存在 source_ref 或 steel_ID）
    """
    X_train, X_test, y_train, y_test, feature_names, groups_train, groups_test = load_data(
        csv_path,
        target_col,
        targets=targets
    )

    # 如果需要调参并且提供了参数网格，则先在训练集上执行网格搜索
    if tune and param_grid is not None:
        print(f"调参中：{target_col} 使用参数网格 {param_grid}")
        best_model, best_params, cv_results = tune_rf_model(
            X_train, y_train, param_grid,
            groups=groups_train,
            cv=cv,
        )
        print(f"最佳参数: {best_params}")
        model = best_model
    else:
        model = build_rf_model(random_state=42, **(rf_params or {}))
        model.fit(X_train, y_train)

    # 将特征名附加到模型，方便 GUI 等后续使用
    try:
        model.feature_names_ = feature_names
    except Exception:
        pass

    return model, X_train, X_test, y_train, y_test, feature_names, groups_train


def save_model(model, model_path):
    """
    保存训练好的模型到文件
    
    参数:
        model: 训练好的 RandomForestRegressor 模型
        model_path: 模型文件保存路径
    """
    joblib.dump(model, model_path)