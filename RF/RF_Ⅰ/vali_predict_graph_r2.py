from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import make_interp_spline


def load_model_and_predict(models_dir: Path, target: str, feature_data: pd.DataFrame):
    """加载模型并对特征数据进行预测"""
    model_path = models_dir / f"{target}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = joblib.load(model_path)
    
    # 获取模型特征
    if not hasattr(model, "feature_names_"):
        raise ValueError(f"模型 {target} 没有 feature_names_ 属性")
    
    feature_names = list(model.feature_names_)
    
    # 准备特征矩阵
    x = feature_data.reindex(columns=feature_names)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(0.0).astype(float)
    
    # 预测
    predictions = model.predict(x)
    return predictions


def create_standard_temperature_features(original_df: pd.DataFrame, diff_steel: str, 
                                        standard_temps: list) -> pd.DataFrame:
    """
    为指定钢种创建标准温度点的特征数据
    
    Args:
        original_df: 原始验证数据集
        diff_steel: 钢种标识
        standard_temps: 标准温度列表 [100, 200, ..., 800]
    
    Returns:
        包含标准温度点特征的DataFrame
    """
    # 筛选指定钢种的数据
    steel_data = original_df[original_df["diff_steel"] == diff_steel].copy()
    
    if steel_data.empty:
        return pd.DataFrame()
    
    # 获取该钢种的第一个样本作为模板（假设同一钢种的其他特征相同）
    template_row = steel_data.iloc[0].copy()
    
    # 创建标准温度点的数据
    new_rows = []
    for temp in standard_temps:
        new_row = template_row.copy()
        new_row["temperature"] = temp
        
        # 添加其他必要的列（如果存在）
        for col in ["steel_ID", "diff_steel", "HR", "TMCP", "TMCP_T", "QT", "CF", 
                   "C", "Mn", "Si", "Cr", "Mo", "Nb", "V", "Ti", "Ni", "Al", "Cu", "P", "S"]:
            if col in template_row.index:
                new_row[col] = template_row[col]
        
        new_rows.append(new_row)
    
    new_df = pd.DataFrame(new_rows)
    return new_df


def main() -> None:
    # 配置路径
    base_dir = Path(__file__).resolve().parent
    original_data_path = Path("data/validation_data.CSV")  # 原始数据集
    prediction_data_path = Path("results/validation_prediction.csv")  # 已有预测结果
    models_dir = Path("results/models")  # 模型目录
    output_dir = Path("results/figures/combined_validation_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否存在
    if not original_data_path.exists():
        raise FileNotFoundError(f"找不到原始数据文件: {original_data_path}")
    
    if not prediction_data_path.exists():
        raise FileNotFoundError(f"找不到预测数据文件: {prediction_data_path}")
    
    if not models_dir.exists():
        raise FileNotFoundError(f"找不到模型目录: {models_dir}")
    
    # 1. 读取原始验证数据（用于生成新预测）
    try:
        original_df = pd.read_csv(original_data_path, encoding="utf-8")
    except UnicodeDecodeError:
        original_df = pd.read_csv(original_data_path, encoding="gbk")
    
    # 2. 读取已有预测结果（用于获取真实值）
    try:
        pred_df = pd.read_csv(prediction_data_path, encoding="utf-8")
    except UnicodeDecodeError:
        pred_df = pd.read_csv(prediction_data_path, encoding="gbk")
    
    # 清理列名
    original_df.columns = original_df.columns.str.strip()
    pred_df.columns = pred_df.columns.str.strip()
    
    # 转换数值列
    numeric_cols = ["temperature"] + [col for col in original_df.columns 
                                     if any(x in col for x in ["reduction", "C", "Mn", "Si", "Cr", "Mo", "Nb", "V", "Ti", "Ni", "Al", "Cu", "P", "S"])]
    for col in numeric_cols:
        if col in original_df.columns:
            original_df[col] = pd.to_numeric(original_df[col], errors="coerce")
    
    # 转换预测数据中的数值列
    pred_numeric_cols = ["temperature"] + [col for col in pred_df.columns if "reduction" in col]
    for col in pred_numeric_cols:
        if col in pred_df.columns:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    
    # 处理字符串列
    for col in ["diff_steel", "steel_ID"]:
        if col in original_df.columns:
            original_df[col] = original_df[col].astype(str).str.strip()
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].astype(str).str.strip()
    
    # 定义标准温度点
    standard_temps = list(range(100, 801, 100))
    
    # 定义目标变量配置
    target_configs = [
        {
            "name": "fy",
            "model_target": "fy_reduction",
            "true_col": "fy_reduction_true",
            "pred_col": "fy_reduction_pred",
            "color": "#FF6B6B",  # 红色
            "marker": "o",  # 圆形
            "label": "fy (屈服强度)"
        },
        {
            "name": "fu", 
            "model_target": "fu_reduction",
            "true_col": "fu_reduction_true", 
            "pred_col": "fu_reduction_pred",
            "color": "#4ECDC4",  # 青色
            "marker": "s",  # 方形
            "label": "fu (抗拉强度)"
        },
        {
            "name": "E",
            "model_target": "E_reduction",
            "true_col": "E_reduction_true",
            "pred_col": "E_reduction_pred", 
            "color": "#95E77E",  # 绿色
            "marker": "^",  # 三角形
            "label": "E (弹性模量)"
        }
    ]
    
    # 获取所有唯一的钢种
    diff_steels = pred_df["diff_steel"].dropna().unique().tolist()
    
    if not diff_steels:
        raise ValueError("没有可用于绘图的 diff_steel 数据。")
    
    print(f"开始处理 {len(diff_steels)} 个钢种...")
    
    for diff_steel in diff_steels:
        print(f"处理钢种: {diff_steel}")
        
        # 筛选当前钢种的已有预测数据（用于真实值）
        steel_pred_df = pred_df[pred_df["diff_steel"] == diff_steel].copy()
        
        if steel_pred_df.empty:
            print(f"  {diff_steel}: 预测数据为空，跳过")
            continue
        
        # 创建标准温度点的特征数据
        standard_features = create_standard_temperature_features(
            original_df, diff_steel, standard_temps
        )
        
        if standard_features.empty:
            print(f"  {diff_steel}: 无法创建标准温度点特征，跳过")
            continue
        
        # 存储标准温度点的预测结果
        standard_predictions = {}
        
        # 为每个目标变量生成标准温度点的预测
        for config in target_configs:
            try:
                # 使用模型预测标准温度点
                predictions = load_model_and_predict(
                    models_dir, config["model_target"], standard_features
                )
                
                # 存储预测结果
                standard_predictions[config["name"]] = {
                    "temperatures": standard_temps,
                    "values": predictions
                }
                
            except Exception as e:
                print(f"  {config['name']} 预测失败: {e}")
                standard_predictions[config["name"]] = {
                    "temperatures": [],
                    "values": []
                }
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 用于存储所有y值以便设置y轴范围
        all_y_values = []
        
        # 为每个目标变量绘制图形
        for config in target_configs:
            true_col = config["true_col"]
            pred_col = config["pred_col"]
            
            # 1. 绘制真实值（散点图）- 从已有预测数据中获取
            true_mask = ~steel_pred_df[true_col].isna()
            if true_mask.any():
                true_points = steel_pred_df[true_mask]
                
                # 确保温度是数值类型
                true_points = true_points.copy()
                true_points["temperature"] = pd.to_numeric(true_points["temperature"], errors="coerce")
                true_points = true_points.dropna(subset=["temperature", true_col])
                
                plt.scatter(
                    true_points["temperature"],
                    true_points[true_col],
                    s=80,  # 点的大小
                    marker=config["marker"],
                    color=config["color"],
                    alpha=0.8,
                    edgecolors=config["color"],
                    linewidths=0.5,
                    zorder=5,  # 确保点在最上层
                    label=f"{config['label']} (真实值)"
                )
                all_y_values.extend(true_points[true_col].tolist())
            
            # 2. 绘制预测值（折线图）- 使用新生成的标准温度点预测
            if config["name"] in standard_predictions:
                pred_data = standard_predictions[config["name"]]
                
                if len(pred_data["temperatures"]) > 0 and len(pred_data["values"]) > 0:
                    # 确保温度和预测值数量匹配
                    if len(pred_data["temperatures"]) == len(pred_data["values"]):
                        # 按温度排序
                        temp_indices = np.argsort(pred_data["temperatures"])
                        sorted_temps = [pred_data["temperatures"][i] for i in temp_indices]
                        sorted_values = [pred_data["values"][i] for i in temp_indices]
                        
                        # 绘制预测值折线
                        if len(sorted_temps) > 1:
                        # 创建更多的点来平滑曲线
                            temp_array = np.array(sorted_temps)
                            value_array = np.array(sorted_values)
    
                        # 如果数据点数量足够，使用样条插值
                            if len(temp_array) >= 3:
                                # 生成更多的温度点
                                temp_smooth = np.linspace(temp_array.min(), temp_array.max(), 300)
        
                                # 使用样条插值
                                spl = make_interp_spline(temp_array, value_array, k=3)  # 3次样条
                                value_smooth = spl(temp_smooth)
        
                                plt.plot(
                                    temp_smooth,
                                    value_smooth,
                                    color=config["color"],
                                    linestyle='-',
                                    linewidth=2.5,
                                    marker='',
                                    zorder=4,
                                    label=f"{config['label']} (预测值)"
                                )
                            else:
                            # 如果点太少，用直线连接
                                plt.plot(
                                    sorted_temps,
                                    sorted_values,
                                    color=config["color"],
                                    linestyle='-',
                                    linewidth=2.5,
                                    marker='',
                                    zorder=4,
                                    label=f"{config['label']} (预测值)"
                                )
                        all_y_values.extend(sorted_values)
        
        # 设置图形属性
        plt.xlabel("温度 (°C)", fontsize=14, fontweight='bold')
        plt.ylabel("折减系数", fontsize=14, fontweight='bold')
        plt.title(f"钢种: {diff_steel} - 高温性能折减系数", fontsize=16, fontweight='bold')
        
        # 设置x轴
        x_ticks = list(range(0, 801, 100))
        plt.xticks(x_ticks, fontsize=12)
        plt.xlim(0, 850)  # 稍微扩展x轴范围
        
        # 设置y轴
        if all_y_values:
            y_values = [y for y in all_y_values if not np.isnan(y)]
            if y_values:
                y_max = max(y_values)
                y_min = min(y_values)
                
                # 计算y轴上限：至少到1.0，如果最大值超过1.0，则向上取整到0.1的倍数
                y_upper = max(1.0, np.ceil(y_max * 10) / 10)
                
                # 确保y轴下限为0或略低于最小值
                y_lower = max(0, np.floor(y_min * 10) / 10 - 0.1)
                y_lower = max(0, y_lower)  # 确保不为负
                
                # 生成y轴刻度
                y_interval = 0.1
                y_ticks = np.arange(y_lower, y_upper + y_interval, y_interval)
                y_ticks = [round(tick, 1) for tick in y_ticks]
                
                plt.yticks(y_ticks, fontsize=12)
                plt.ylim(y_lower - 0.05, y_upper + 0.05)  # 稍微扩展y轴范围
        
        plt.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 添加参考线
        plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = output_dir / f"diff_steel_{diff_steel}_combined.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close("all")
        
        print(f"  已保存: {output_file}")
    
    print(f"\n全部完成！共生成 {len(diff_steels)} 张图表")
    print(f"图片保存路径: {output_dir.resolve()}")


if __name__ == "__main__":
    main()