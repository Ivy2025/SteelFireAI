# -*- coding: utf-8 -*-
"""
shap_analysis.py
钢材高温力学性能折减因子——化学成分特征SHAP分析
"""
import matplotlib
matplotlib.use("Agg")  # 避免Tkinter报错
import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# 需要的库：pandas, numpy, matplotlib, shap, joblib, sklearn
# 如果缺少shap，请运行：pip install shap

# 模型路径配置
MODEL_PATHS = {
    "fy_reduction": Path("results/models/fy_reduction_model.pkl"),
    "fu_reduction": Path("results/models/fu_reduction_model.pkl"),
    "E_reduction": Path("results/models/E_reduction_model.pkl"),
}

# 化学成分变量
COMPOSITION_VARS = [
    "C", "Mn", "Si", "Cr", "Mo", "Nb", "V",
    "Ti", "Ni", "Al", "Cu", "P", "S"
]
# 工艺变量
BINARY_VARS = ["HR", "N", "TMCP", "TMCP_T", "QT", "CF"]
# 训练时完整特征
FEATURE_COLS = BINARY_VARS + COMPOSITION_VARS + ["temperature"]

# 数据路径
DATA_PATH = Path("data/processed/steel_v2_cleaned_r2.csv")
# 输出路径
FIGURE_DIR = Path("results/figures/shap")
METRIC_DIR = Path("results/metrics/shap")

EXPLAIN_DATA = "all"  # 默认用全量数据


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def load_model(target: str):
    model_path = MODEL_PATHS[target]
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    return joblib.load(model_path)


def get_feature_names(model, X: pd.DataFrame):
    # 尽量获取pipeline/模型的特征名
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    if isinstance(model, Pipeline):
        # 尝试获取最后一步estimator的特征名
        est = model.steps[-1][1]
        if hasattr(est, "feature_names_in_"):
            return list(est.feature_names_in_)
        if hasattr(est, "feature_names_"):
            return list(est.feature_names_)
    # 否则用X的列
    return list(X.columns)


def prepare_X(df: pd.DataFrame):
    # 保证输入特征顺序和训练一致
    X = df.copy()
    # 只保留训练时用到的特征
    missing_cols = [col for col in FEATURE_COLS if col not in X.columns]
    if missing_cols:
        raise ValueError(f"数据缺少以下特征列: {missing_cols}")
    X = X[FEATURE_COLS]
    return X


def fill_missing(X: pd.DataFrame):
    # 工艺变量用0填补，化学成分和温度用中位数，先强制转为float
    X_filled = X.copy()
    for col in X_filled.columns:
        if col in BINARY_VARS:
            X_filled[col] = X_filled[col].fillna(0)
        elif col in COMPOSITION_VARS or col == "temperature":
            # 先转为数值型
            X_filled[col] = pd.to_numeric(X_filled[col], errors='coerce')
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())
    return X_filled


def get_shap_explainer(model, X):
    # 兼容Pipeline和树模型
    if isinstance(model, Pipeline):
        # 尝试直接解释Pipeline
        try:
            explainer = shap.Explainer(model, X)
            return explainer
        except Exception:
            # 拆分pipeline，先预处理再解释estimator
            pre = Pipeline(model.steps[:-1])
            est = model.steps[-1][1]
            X_proc = pre.transform(X)
            # 尝试用TreeExplainer
            explainer = shap.TreeExplainer(est)
            return explainer, X_proc
    # 直接是树模型
    try:
        explainer = shap.Explainer(model, X)
        return explainer
    except Exception:
        explainer = shap.TreeExplainer(model)
        return explainer


def get_original_feature_names(model, X):
    # 处理pipeline特征名映射
    if isinstance(model, Pipeline):
        pre = Pipeline(model.steps[:-1])
        try:
            X_proc = pre.transform(X)
            if hasattr(X_proc, 'columns'):
                proc_names = list(X_proc.columns)
            elif hasattr(pre, 'get_feature_names_out'):
                proc_names = list(pre.get_feature_names_out())
            else:
                proc_names = [str(i) for i in range(X_proc.shape[1])]
            # 尝试映射回原始化学成分名
            mapping = {}
            for name in proc_names:
                for c in COMPOSITION_VARS:
                    if name.endswith(f"__{c}") or name == c:
                        mapping[name] = c
            return mapping
        except Exception:
            pass
    # 直接返回化学成分名
    return {c: c for c in COMPOSITION_VARS}


def plot_shap_summary(shap_values, X, target, mapping):
    ensure_dir(FIGURE_DIR)
    plt.figure(figsize=(7, 5))
    # 只保留化学成分变量
    comp_cols = [col for col in X.columns if mapping.get(col, col) in COMPOSITION_VARS]
    shap.summary_plot(
        shap_values[:, [X.columns.get_loc(c) for c in comp_cols]],
        X[comp_cols],
        feature_names=[mapping.get(c, c) for c in comp_cols],
        show=False,
        plot_type="dot",
        color_bar=True
    )
    plt.title(f"Chemical SHAP summary ({target})", fontsize=13)
    plt.tight_layout()
    out_path = FIGURE_DIR / f"{target}_shap_summary_chemical_features.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存SHAP图: {out_path}")


def save_shap_importance(shap_values, X, target, mapping):
    ensure_dir(METRIC_DIR)
    # 只保留化学成分变量
    comp_cols = [col for col in X.columns if mapping.get(col, col) in COMPOSITION_VARS]
    mean_abs = np.abs(shap_values[:, [X.columns.get_loc(c) for c in comp_cols]]).mean(axis=0)
    df_imp = pd.DataFrame({
        "feature": [mapping.get(c, c) for c in comp_cols],
        "mean_abs_shap": mean_abs
    })
    df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1
    out_path = METRIC_DIR / f"{target}_shap_importance_chemical_features.csv"
    df_imp.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存SHAP重要性表: {out_path}")


def main():
    df = load_data()
    for target in ["fy_reduction", "fu_reduction", "E_reduction"]:
        print(f"\n==== {target} SHAP分析 ====")
        model = load_model(target)
        X = prepare_X(df)
        # Pipeline不额外填补，非Pipeline需填补
        if isinstance(model, Pipeline):
            X_input = X
        else:
            X_input = fill_missing(X)
        # 获取shap explainer
        explainer = get_shap_explainer(model, X_input)
        if isinstance(explainer, tuple):
            explainer, X_proc = explainer
            shap_values = explainer(X_proc)
            mapping = get_original_feature_names(model, X)
            X_for_plot = pd.DataFrame(X_proc, columns=list(mapping.keys()))
        else:
            shap_values = explainer(X_input)
            mapping = {c: c for c in X_input.columns}
            X_for_plot = X_input
        # SHAP值为shap.Explanation对象时取values
        if hasattr(shap_values, 'values'):
            shap_arr = shap_values.values
        else:
            shap_arr = shap_values
        plot_shap_summary(shap_arr, X_for_plot, target, mapping)
        save_shap_importance(shap_arr, X_for_plot, target, mapping)

if __name__ == "__main__":
    main()
