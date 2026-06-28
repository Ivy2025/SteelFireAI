# -*- coding: utf-8 -*-
"""
shap_analysis.py
钢材高温力学性能折减因子——化学成分特征SHAP分析
需要的库：pandas, numpy, matplotlib, shap, joblib, sklearn
如缺少shap，请运行：pip install shap
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

# 模型路径配置
MODEL_PATHS = {
    "fy_reduction": Path("results/models/catboost_plan1_fy_reduction.pkl"),
    "fu_reduction": Path("results/models/catboost_plan1_fu_reduction.pkl"),
    "E_reduction": Path("results/models/catboost_plan1_E_reduction.pkl"),
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
DATA_PATH = Path("data/steel_v2_cleaned_r2.csv")
# 输出路径
FIGURE_DIR = Path("results/figures/shap")
METRIC_DIR = Path("results/metrics/shap")

EXPLAIN_DATA = "all"  # 默认用全量数据

def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)

def load_data():
    """读取数据，检查列名"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    for col in FEATURE_COLS + ["fy_reduction", "fu_reduction", "E_reduction"]:
        if col not in df.columns:
            raise ValueError(f"数据缺少必要列: {col}")
    return df

def load_model(target):
    """加载模型，支持Pipeline和单模型"""
    model_path = MODEL_PATHS[target]
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {model_path}\n{e}")
    return model

def fill_missing(X: pd.DataFrame) -> pd.DataFrame:
    """对输入特征做缺失值填补"""
    X_filled = X.copy()
    for col in COMPOSITION_VARS + ["temperature"]:
        if col in X_filled.columns:
            X_filled[col] = pd.to_numeric(X_filled[col], errors="coerce")
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())
    for col in BINARY_VARS:
        if col in X_filled.columns:
            X_filled[col] = X_filled[col].fillna(0)
    # 保证列顺序
    X_filled = X_filled[[c for c in FEATURE_COLS if c in X_filled.columns]]
    return X_filled

def get_shap_explainer(model, X):
    """获取shap解释器，兼容Pipeline和树模型"""
    if isinstance(model, Pipeline) and len(model.steps) > 0:
        if len(model.steps) > 1:
            pre = Pipeline(model.steps[:-1])
            est = model.steps[-1][1]
        else:
            pre = None
            est = model.steps[0][1]
        # 预处理
        if pre is not None and hasattr(pre, 'feature_names_in_'):
            X_aligned = X[list(pre.feature_names_in_)]
            X_proc = pre.transform(X_aligned)
        elif pre is not None:
            X_proc = pre.transform(X)
        else:
            X_proc = X
        try:
            explainer = shap.Explainer(est, X_proc)
        except Exception:
            explainer = shap.TreeExplainer(est)
        return explainer, X_proc
    # 直接是树模型或Pipeline为空
    try:
        explainer = shap.Explainer(model, X)
        return explainer
    except Exception:
        explainer = shap.TreeExplainer(model)
        return explainer

def get_original_feature_names(model, X):
    """尝试将预处理后特征名映射回原始化学成分名"""
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        pre = Pipeline(model.steps[:-1])
        try:
            X_proc = pre.transform(X)
            if hasattr(X_proc, 'columns'):
                proc_names = list(X_proc.columns)
            elif hasattr(pre, 'get_feature_names_out'):
                proc_names = list(pre.get_feature_names_out())
            else:
                proc_names = [str(i) for i in range(X_proc.shape[1])]
            mapping = {}
            for name in proc_names:
                for c in COMPOSITION_VARS:
                    if name.endswith(f"__{c}") or name == c:
                        mapping[name] = c
                        break
            return mapping
        except Exception:
            pass
    # 默认直接返回原名
    return {c: c for c in COMPOSITION_VARS}

def plot_shap_summary(shap_values, X_for_plot, target, mapping):
    """绘制SHAP summary图，只保留化学成分特征"""
    ensure_dir(FIGURE_DIR)

    # SHAP值转为数组
    if hasattr(shap_values, "values"):
        shap_arr = shap_values.values
    else:
        shap_arr = shap_values

    # 如果是三维，取最后一维前先报错提示
    if shap_arr.ndim != 2:
        raise ValueError(f"当前SHAP值维度为 {shap_arr.shape}，不是二维数组，请检查模型类型或SHAP输出。")

    # 找到化学成分列及其在完整特征矩阵中的位置
    comp_cols = []
    comp_indices = []

    for i, col in enumerate(X_for_plot.columns):
        original_name = mapping.get(col, col)
        if original_name in COMPOSITION_VARS:
            comp_cols.append(col)
            comp_indices.append(i)

    if not comp_cols:
        raise ValueError("没有找到可用于绘图的化学成分列，请检查特征名映射。")

    # 检查 SHAP 总列数是否和完整 X_for_plot 对应
    if shap_arr.shape[1] != X_for_plot.shape[1]:
        raise ValueError(
            f"完整SHAP特征数和X_for_plot列数不一致："
            f"shap_arr.shape={shap_arr.shape}, X_for_plot.shape={X_for_plot.shape}。"
            f"请检查Pipeline预处理后的特征名和列数是否匹配。"
        )

    # 同步筛选化学成分的 X 和 SHAP 值
    X_plot = X_for_plot[comp_cols].copy()
    shap_plot = shap_arr[:, comp_indices]

    # 保证绘图用特征为数值型并填补缺失
    for col in X_plot.columns:
        X_plot[col] = pd.to_numeric(X_plot[col], errors="coerce")
        X_plot[col] = X_plot[col].fillna(X_plot[col].median())

    feature_names = [mapping.get(c, c) for c in comp_cols]

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_plot,
        X_plot,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
        color_bar=True
    )

    plt.title(f"Chemical SHAP summary ({target})")
    plt.tight_layout()

    out_path = FIGURE_DIR / f"{target}_shap_summary_chemical_features.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"已保存SHAP图: {out_path}")

def save_shap_importance(shap_values, X_for_plot, target, mapping):
    """保存化学成分特征的SHAP重要性表"""
    ensure_dir(METRIC_DIR)

    if hasattr(shap_values, "values"):
        shap_arr = shap_values.values
    else:
        shap_arr = shap_values

    if shap_arr.ndim != 2:
        raise ValueError(f"当前SHAP值维度为 {shap_arr.shape}，不是二维数组，请检查模型类型或SHAP输出。")

    comp_cols = []
    comp_indices = []

    for i, col in enumerate(X_for_plot.columns):
        original_name = mapping.get(col, col)
        if original_name in COMPOSITION_VARS:
            comp_cols.append(col)
            comp_indices.append(i)

    if not comp_cols:
        raise ValueError("没有找到可用于保存重要性表的化学成分列，请检查特征名映射。")

    if shap_arr.shape[1] != X_for_plot.shape[1]:
        raise ValueError(
            f"完整SHAP特征数和X_for_plot列数不一致："
            f"shap_arr.shape={shap_arr.shape}, X_for_plot.shape={X_for_plot.shape}。"
            f"请检查Pipeline预处理后的特征名和列数是否匹配。"
        )

    shap_plot = shap_arr[:, comp_indices]

    mean_abs_shap = np.abs(shap_plot).mean(axis=0)

    df_imp = pd.DataFrame({
        "feature": [mapping.get(c, c) for c in comp_cols],
        "mean_abs_shap": mean_abs_shap
    })

    df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1)

    out_path = METRIC_DIR / f"{target}_shap_importance_chemical_features.csv"
    df_imp.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"已保存SHAP重要性表: {out_path}")

def main():
    try:
        df = load_data()
    except Exception as e:
        print(f"【错误】{e}")
        return
    for target in ["fy_reduction", "fu_reduction", "E_reduction"]:
        print(f"\n==== {target} SHAP分析 ====")
        try:
            model = load_model(target)
        except Exception as e:
            print(f"【错误】{e}")
            continue
        # 构造输入特征
        X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
        # Pipeline不额外填补，非Pipeline需填补
        if isinstance(model, Pipeline):
            X_input = fill_missing(X)
        else:
            X_input = fill_missing(X)
        # 获取shap explainer
        explainer = get_shap_explainer(model, X_input)
        if isinstance(explainer, tuple):
            explainer, X_proc = explainer
            shap_values = explainer(X_proc)
            mapping = get_original_feature_names(model, X)
            if hasattr(X, 'columns'):
                X_for_plot = pd.DataFrame(X_proc, columns=X.columns)
            else:
                X_for_plot = pd.DataFrame(X_proc)
        else:
            shap_values = explainer(X_input)
            mapping = {c: c for c in X_input.columns}
            X_for_plot = X_input.copy()
        # 只保留化学成分特征并补缺
        for col in COMPOSITION_VARS:
            if col in X_for_plot.columns:
                X_for_plot[col] = pd.to_numeric(X_for_plot[col], errors="coerce")
                X_for_plot[col] = X_for_plot[col].fillna(X_for_plot[col].median())
        try:
            plot_shap_summary(shap_values, X_for_plot, target, mapping)
            save_shap_importance(shap_values, X_for_plot, target, mapping)
            print(f"【完成】{target} SHAP分析结果已保存。")
        except Exception as e:
            print(f"【错误】{target} SHAP输出失败：{e}")

if __name__ == "__main__":
    main()