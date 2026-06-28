# -*- coding: utf-8 -*-
"""
correlation_analysis.py
钢材高温力学性能折减因子——化学成分特征Spearman相关性分析
"""
import matplotlib
matplotlib.use("Agg")  # 避免Tkinter报错
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 化学成分变量
COMPOSITION_VARS = [
    "C", "Mn", "Si", "Cr", "Mo", "Nb", "V",
    "Ti", "Ni", "Al", "Cu", "P", "S"
]

# 数据路径
DATA_PATH = Path("data/steel_v2_cleaned_r2.csv")
# 输出路径
METRIC_DIR = Path("results/metrics/correlation")
FIGURE_DIR = Path("results/figures/correlation")
ABS_CORR_THRESHOLD = 0.7


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def get_chemical_df(df: pd.DataFrame):
    missing_cols = [col for col in COMPOSITION_VARS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少以下化学成分列: {missing_cols}")
    chem_df = df[COMPOSITION_VARS].copy()
    # 强制转为float，防止有空字符串或异常
    for col in chem_df.columns:
        chem_df[col] = pd.to_numeric(chem_df[col], errors='coerce')
    return chem_df


def save_missing_count(df: pd.DataFrame):
    ensure_dir(METRIC_DIR)
    missing = df.isna().sum().reset_index()
    missing.columns = ["feature", "missing_count"]
    out_path = METRIC_DIR / "chemical_missing_count.csv"
    missing.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存缺失值统计: {out_path}")


def save_spearman_corr(df: pd.DataFrame):
    ensure_dir(METRIC_DIR)
    corr = df.corr(method="spearman")
    out_path = METRIC_DIR / "chemical_spearman_corr.csv"
    corr.to_csv(out_path, encoding="utf-8-sig")
    print(f"已保存Spearman相关性矩阵: {out_path}")
    return corr


def plot_heatmap(corr: pd.DataFrame):
    ensure_dir(FIGURE_DIR)
    plt.figure(figsize=(8, 7))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=11)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=11)
    # 在格子中标注相关系数
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=10)
    plt.title("Spearman correlation among chemical composition features", fontsize=14)
    plt.tight_layout()
    out_path = FIGURE_DIR / "chemical_spearman_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存相关性热力图: {out_path}")


def save_strong_corr_pairs(corr: pd.DataFrame):
    ensure_dir(METRIC_DIR)
    records = []
    n = len(corr.columns)
    for i in range(n):
        for j in range(i+1, n):
            f1 = corr.columns[i]
            f2 = corr.columns[j]
            r = corr.iloc[i, j]
            if abs(r) >= ABS_CORR_THRESHOLD:
                records.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "spearman_corr": r,
                    "abs_corr": abs(r)
                })
    df_pairs = pd.DataFrame(records)
    out_path = METRIC_DIR / "chemical_strong_corr_pairs.csv"
    df_pairs.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已保存强相关成分对: {out_path}")


def main():
    df = load_data()
    chem_df = get_chemical_df(df)
    save_missing_count(chem_df)
    corr = save_spearman_corr(chem_df)
    plot_heatmap(corr)
    save_strong_corr_pairs(corr)

if __name__ == "__main__":
    main()
