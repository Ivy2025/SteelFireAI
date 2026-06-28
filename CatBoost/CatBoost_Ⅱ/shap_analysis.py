# -*- coding: utf-8 -*-
"""
CatBoost SHAP 分析脚本
功能：
1. 使用 CatBoost 训练时的处理后数据集 processed_dataset.csv
2. 输入特征为：化学成分 + temperature + process_type
3. process_type 作为 CatBoost 类别特征
4. 使用 CatBoost 原生 SHAP 方法计算 SHAP 值
5. 最终只输出化学成分变量的 SHAP summary 图和 mean_abs_shap 重要性表

依赖库：
pandas, numpy, matplotlib, shap, joblib, catboost

如缺少依赖，请运行：
pip install pandas numpy matplotlib shap joblib catboost
"""

import matplotlib
matplotlib.use("Agg")  # 避免 Tkinter 报错

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import Pool
from sklearn.pipeline import Pipeline


# =========================
# 1. 路径配置
# =========================

MODEL_PATHS = {
    "fy_reduction": Path("results/models/catboost_plan3_fy_reduction.pkl"),
    "fu_reduction": Path("results/models/catboost_plan3_fu_reduction.pkl"),
    "E_reduction": Path("results/models/catboost_plan3_E_reduction.pkl"),
}

DATA_PATH = Path("data/processed_dataset.csv")

FIGURE_DIR = Path("results/figures/shap")
METRIC_DIR = Path("results/metrics/shap")


# =========================
# 2. 变量配置
# =========================

COMPOSITION_VARS = [
    "C", "Mn", "Si", "Cr", "Mo", "Nb", "V",
    "Ti", "Ni", "Al", "Cu", "P", "S"
]

NUMERIC_VARS = COMPOSITION_VARS + ["temperature"]

CAT_FEATURES = ["process_type"]

FEATURE_COLS = COMPOSITION_VARS + ["temperature"] + CAT_FEATURES

TARGETS = ["fy_reduction", "fu_reduction", "E_reduction"]


# =========================
# 3. 工具函数
# =========================

def ensure_dir(path: Path) -> None:
    """确保输出文件夹存在"""
    path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """读取处理后数据集，并进行基础清洗"""
    if not DATA_PATH.exists():
        sys.exit(f"【错误】数据文件不存在：{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 将空字符串、纯空格字符串替换为 NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    required_cols = FEATURE_COLS + TARGETS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        sys.exit(f"【错误】数据缺少必要列：{missing_cols}")

    # 数值特征转为数值型
    for col in NUMERIC_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 目标变量转为数值型
    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 类别变量保持字符串类型，不能转为数值
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


def load_model(model_path: Path):
    """加载模型"""
    if not model_path.exists():
        sys.exit(f"【错误】模型文件不存在：{model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        sys.exit(f"【错误】模型加载失败：{model_path}\n错误信息：{e}")

    return model


def get_final_estimator(model):
    """
    如果保存的是 Pipeline，则取最后一步 estimator；
    如果保存的是 CatBoostRegressor，则直接返回。
    """
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def get_model_feature_names(estimator) -> list:
    """
    优先从 CatBoost 模型中读取训练时的特征名和顺序。
    如果读取失败，则使用 FEATURE_COLS。
    """
    if hasattr(estimator, "feature_names_") and estimator.feature_names_:
        return list(estimator.feature_names_)

    if hasattr(estimator, "get_feature_names"):
        try:
            names = estimator.get_feature_names()
            if names:
                return list(names)
        except Exception:
            pass

    return FEATURE_COLS.copy()


def prepare_X_for_model(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    按模型训练时的特征名和顺序构造 X。
    注意：
    1. 不加入 diff_steel
    2. 不加入目标变量
    3. process_type 保持字符串类别变量
    4. 数值变量转为 numeric
    """
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集中缺少模型需要的特征列：{missing_cols}")

    X = df[feature_names].copy()

    for col in X.columns:
        if col in CAT_FEATURES:
            X[col] = X[col].fillna("Unknown").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def prepare_X_for_plot(X: pd.DataFrame) -> pd.DataFrame:
    """
    绘图用 X。
    为避免 SHAP summary 图大量灰色点，绘图时对化学成分变量做中位数填补。
    注意：这只影响颜色显示，不改变模型 SHAP 计算输入。
    """
    X_plot = X.copy()

    for col in COMPOSITION_VARS:
        if col in X_plot.columns:
            X_plot[col] = pd.to_numeric(X_plot[col], errors="coerce")
            median_value = X_plot[col].median()
            X_plot[col] = X_plot[col].fillna(median_value)

    return X_plot


def get_cat_features_in_X(X: pd.DataFrame) -> list:
    """返回当前 X 中存在的类别特征"""
    return [col for col in CAT_FEATURES if col in X.columns]


def compute_catboost_shap(estimator, X: pd.DataFrame):
    """
    使用 CatBoost 原生方法计算 SHAP 值。
    CatBoost 返回的 SHAP 结果最后一列是 expected value，需要去掉。
    """
    cat_features = get_cat_features_in_X(X)

    pool = Pool(
        data=X,
        cat_features=cat_features
    )

    shap_values = estimator.get_feature_importance(
        pool,
        type="ShapValues"
    )

    # 最后一列是 base value / expected value，不是特征 SHAP
    shap_arr = shap_values[:, :-1]

    feature_names = list(X.columns)

    if shap_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP 列数与特征数量不一致："
            f"SHAP={shap_arr.shape[1]}, features={len(feature_names)}"
        )

    return shap_arr, feature_names


def plot_chemical_shap_summary(
    shap_arr: np.ndarray,
    X: pd.DataFrame,
    feature_names: list,
    target: str
) -> None:
    """绘制只包含化学成分变量的 SHAP summary 图"""
    ensure_dir(FIGURE_DIR)

    chemical_indices = [
        i for i, name in enumerate(feature_names)
        if name in COMPOSITION_VARS
    ]

    if not chemical_indices:
        raise ValueError(f"未找到化学成分特征。当前 feature_names={feature_names}")

    chemical_names = [feature_names[i] for i in chemical_indices]

    shap_chemical = shap_arr[:, chemical_indices]

    X_plot_all = prepare_X_for_plot(X)
    X_chemical = X_plot_all[chemical_names].copy()

    # 诊断：检查绘图数据是否还有缺失值
    missing_for_plot = X_chemical.isna().sum()
    if missing_for_plot.sum() > 0:
        print("【提示】绘图用化学成分仍存在缺失值：")
        print(missing_for_plot[missing_for_plot > 0])

    plt.figure(figsize=(8, 6))

    shap.summary_plot(
        shap_chemical,
        X_chemical,
        feature_names=chemical_names,
        show=False,
        plot_type="dot",
        max_display=len(chemical_names)
    )

    plt.title(f"Chemical SHAP summary ({target})", fontsize=12)
    plt.tight_layout()

    fig_path = FIGURE_DIR / f"{target}_shap_summary_chemical_features.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"【完成】SHAP 图已保存：{fig_path}")


def save_chemical_shap_importance(
    shap_arr: np.ndarray,
    feature_names: list,
    target: str
) -> None:
    """保存只包含化学成分变量的 mean_abs_shap 重要性表"""
    ensure_dir(METRIC_DIR)

    chemical_indices = [
        i for i, name in enumerate(feature_names)
        if name in COMPOSITION_VARS
    ]

    if not chemical_indices:
        raise ValueError(f"未找到化学成分特征。当前 feature_names={feature_names}")

    chemical_names = [feature_names[i] for i in chemical_indices]

    shap_chemical = shap_arr[:, chemical_indices]

    mean_abs_shap = np.abs(shap_chemical).mean(axis=0)

    df_imp = pd.DataFrame({
        "feature": chemical_names,
        "mean_abs_shap": mean_abs_shap
    })

    df_imp = df_imp.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    df_imp["rank"] = df_imp.index + 1

    out_path = METRIC_DIR / f"{target}_shap_importance_chemical_features.csv"
    df_imp.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"【完成】SHAP 重要性表已保存：{out_path}")


def print_diagnostics(target: str, X: pd.DataFrame, shap_arr: np.ndarray, feature_names: list) -> None:
    """打印诊断信息，防止特征错位"""
    print(f"\n【诊断】目标变量：{target}")
    print(f"【诊断】X shape: {X.shape}")
    print(f"【诊断】SHAP shape: {shap_arr.shape}")
    print(f"【诊断】feature_names 数量: {len(feature_names)}")

    print("【诊断】模型输入特征顺序：")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")

    chemical_names = [name for name in feature_names if name in COMPOSITION_VARS]
    print(f"【诊断】用于展示的化学成分特征：{chemical_names}")

    if shap_arr.shape[1] != X.shape[1]:
        raise ValueError(
            f"SHAP 列数与 X 列数不一致："
            f"SHAP={shap_arr.shape[1]}, X={X.shape[1]}"
        )

    if shap_arr.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP 列数与 feature_names 数量不一致："
            f"SHAP={shap_arr.shape[1]}, feature_names={len(feature_names)}"
        )


def analyze_target(df: pd.DataFrame, target: str) -> None:
    """对单个目标变量进行 SHAP 分析"""
    print(f"\n正在分析目标变量：{target}")

    model_path = MODEL_PATHS[target]
    loaded_model = load_model(model_path)

    estimator = get_final_estimator(loaded_model)

    feature_names = get_model_feature_names(estimator)

    # 如果模型保存的特征名为空或异常，则使用默认特征
    if not feature_names:
        feature_names = FEATURE_COLS.copy()

    # 只保留模型真正需要的特征，并严格按照模型训练时顺序排列
    X = prepare_X_for_model(df, feature_names)

    shap_arr, feature_names = compute_catboost_shap(estimator, X)

    print_diagnostics(target, X, shap_arr, feature_names)

    plot_chemical_shap_summary(shap_arr, X, feature_names, target)

    save_chemical_shap_importance(shap_arr, feature_names, target)

    print(f"【完成】{target} SHAP 分析完成。")


def main() -> None:
    """主函数"""
    ensure_dir(FIGURE_DIR)
    ensure_dir(METRIC_DIR)

    df = load_data()

    print("【诊断】数据列名：")
    print(list(df.columns))

    print("\n【诊断】数值特征缺失值数量：")
    print(df[NUMERIC_VARS].isna().sum().sort_values(ascending=False))

    for target in TARGETS:
        try:
            analyze_target(df, target)
        except Exception as e:
            print(f"【错误】{target} 分析失败：{e}")


if __name__ == "__main__":
    main()