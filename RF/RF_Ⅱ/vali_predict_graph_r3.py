from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    """生成真实值vs预测值的散点图，每个目标一张图"""
    
    # 配置路径
    base_dir = Path(__file__).resolve().parent
    data_path = Path("results/validation_prediction.csv")
    output_dir = Path("results/scatter_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    
    # 读取数据
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(data_path, encoding="gbk")
        except:
            # 尝试自动检测编码
            df = pd.read_csv(data_path)
    
    # 清理列名
    df.columns = df.columns.str.strip()
    
    # 确保温度列是数值类型
    if "temperature" in df.columns:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    
    # 过滤掉温度≥800的数据点
    if "temperature" in df.columns:
        original_count = len(df)
        df = df[df["temperature"] < 800].copy()
        filtered_count = len(df)
        print(f"数据过滤: {original_count} → {filtered_count} (移除温度≥800°C的数据)")
    
    # 定义目标变量配置
    target_configs = [
        {
            "name": "fy",  # 修改：改为基础名称
            "true_col": "fy_reduction_true",  # 修改：直接指定真实值列名
            "pred_col": "fy_reduction_pred",  # 修改：直接指定预测值列名
            "label": "Predicted vs Actual (Test) - fy_reduction",
            "color": "#0000FF",  
            "marker": "o"
        },
        {
            "name": "fu",  # 修改：改为基础名称
            "true_col": "fu_reduction_true",  # 修改：直接指定真实值列名
            "pred_col": "fu_reduction_pred",  # 修改：直接指定预测值列名
            "label": "Predicted vs Actual (Test) - fu_reduction",
            "color": "#FF0000",  
            "marker": "o"
        },
        {
            "name": "E",  # 修改：改为基础名称
            "true_col": "E_reduction_true",  # 修改：直接指定真实值列名
            "pred_col": "E_reduction_pred",  # 修改：直接指定预测值列名
            "label": "Predicted vs Actual (Test) - E_reduction",
            "color": "#00CC00",  
            "marker": "o"
        }
    ]
    
    # 用于存储所有目标变量的数据，用于组合图
    all_data = []
    
    # 为每个目标变量生成散点图
    for target in target_configs:
        true_col = target["true_col"]  # 修改：直接使用配置的真实值列名
        pred_col = target["pred_col"]  # 修改：直接使用配置的预测值列名
        
        # 检查列是否存在
        if true_col not in df.columns or pred_col not in df.columns:
            print(f"警告: 缺少 {true_col} 或 {pred_col} 列，跳过 {target['name']}")
            continue
        
        # 提取有效数据
        plot_data = df[[true_col, pred_col]].copy()
        
        # 转换为数值类型
        plot_data[true_col] = pd.to_numeric(plot_data[true_col], errors="coerce")
        plot_data[pred_col] = pd.to_numeric(plot_data[pred_col], errors="coerce")
        
        # 删除缺失值
        plot_data = plot_data.dropna()
        
        if plot_data.empty:
            print(f"警告: {target['name']} 无有效数据，跳过")
            continue
        
        x = plot_data[true_col]
        y = plot_data[pred_col]
        
        # 存储数据用于组合图
        all_data.append({
            "x": x,
            "y": y,
            "config": target
        })
        
        print(f"{target['name']}: 有效数据点 = {len(x)}")
        
        # 创建图形
        plt.figure(figsize=(8, 7))
        
        # 绘制散点图
        plt.scatter(
            x, y,
            s=100,
            alpha=0.6,
            color=target["color"],
            marker=target["marker"],
            edgecolors=target["color"],
            linewidths=0.3,
            label="Test"
        )
        
        # 计算数据范围来确定坐标轴
        all_values = pd.concat([x, y])
        data_min = 0
        data_max = all_values.max()
        
        # 根据图片信息，设置坐标轴范围：
        x_max = max(1.0, np.ceil(data_max * 5) / 5)  # 向上取整到0.2的倍数
        y_max = max(1.2, np.ceil(data_max * 5) / 5)  # 向上取整到0.2的倍数
        
        # 设置固定的坐标轴刻度间隔为0.2
        tick_interval = 0.2
        
        # 生成x轴和y轴刻度
        x_ticks = np.arange(0, x_max + 0.01, tick_interval)
        y_ticks = np.arange(0, y_max + 0.01, tick_interval)
        
        # 设置坐标轴范围和刻度
        plt.xlim(0, x_max)
        plt.ylim(0, y_max)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        
        # 绘制45°参考线
        plt.plot([0, x_max], [0, x_max], 
                'r--',
                linewidth=1.5,
                alpha=0.8,
                label="Perfect prediction")
        
        # 绘制20%相对误差线
        x_line = np.linspace(0, x_max, 100)
        y_upper = 1.2 * x_line
        y_lower = 0.8 * x_line
        
        # 只绘制在坐标轴范围内的部分
        mask_upper = y_upper <= y_max
        mask_lower = y_lower >= 0
        
        plt.plot(x_line[mask_upper], y_upper[mask_upper], 
                'b--',
                linewidth=1.2,
                alpha=0.6,
                label="+20% deviation")
        
        plt.plot(x_line[mask_lower], y_lower[mask_lower], 
                'b--',
                linewidth=1.2,
                alpha=0.6,
                label="-20% deviation")
        
        # 计算落在+20%线之下的点占全部点的比例
        # 即 y <= 1.2*x 的点
        if len(x) > 0 and len(y) > 0:
            # 计算相对误差
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_error = np.abs((y - x) / np.where(x != 0, x, np.nan))
            
            valid_errors = rel_error[np.isfinite(rel_error)]
            if len(valid_errors) > 0:
                # 计算落在+20%线之下的点
                # 即预测值 <= 1.2 * 真实值的点
                within_upper_20 = (y <= 1.2 * x).sum()
                total_points = len(x)
                within_pct = (within_upper_20 / total_points * 100)
                
                # 添加统计信息
                stats_text = f"Points under 20% line: {within_pct:.1f}%"

                # 将统计信息放在图的右下角
                plt.text(0.95, 0.05, stats_text,
                        transform=plt.gca().transAxes,
                        fontsize=16,
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 设置图形属性
        plt.xlabel("True Values", fontsize=16)
        plt.ylabel("Predicted Values", fontsize=16)
        plt.title(f"{target['label']}", fontsize=16)
        
        # 设置网格
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 添加图例
        plt.legend(loc='upper left', fontsize=14)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = output_dir / f"scatter_{target['name']}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close("all")
        
        print(f"  已保存: {output_file}")
    
    # 新增：创建组合图，显示所有目标变量的点
    if all_data:
        print("\n生成组合图...")
        
        # 创建组合图形
        plt.figure(figsize=(10, 8))
        
        # 用于存储所有点的坐标，以便计算坐标轴范围
        all_x_values = []
        all_y_values = []
        
        # 为每个目标变量绘制散点
        for data in all_data:
            x = data["x"]
            y = data["y"]
            config = data["config"]
            
            # 绘制散点图
            plt.scatter(
                x, y,
                s=100,
                alpha=0.6,
                color=config["color"],
                marker=config["marker"],
                edgecolors=config["color"],
                linewidths=0.3,
                label=f"{config['name']} (Test)"
            )
            
            # 收集所有点的坐标
            all_x_values.extend(x.tolist())
            all_y_values.extend(y.tolist())
        
        # 计算组合图的数据范围
        if all_x_values and all_y_values:
            all_values = pd.Series(all_x_values + all_y_values)
            data_max = all_values.max()
            
            # 设置坐标轴范围
            x_max = max(1.0, np.ceil(data_max * 5) / 5)
            y_max = max(1.2, np.ceil(data_max * 5) / 5)
            
            # 设置固定的坐标轴刻度间隔为0.2
            tick_interval = 0.2
            
            # 生成x轴和y轴刻度
            x_ticks = np.arange(0, x_max + 0.01, tick_interval)
            y_ticks = np.arange(0, y_max + 0.01, tick_interval)
            
            # 设置坐标轴范围和刻度
            plt.xlim(0, x_max)
            plt.ylim(0, y_max)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
            
            # 绘制45°参考线
            plt.plot([0, x_max], [0, x_max], 
                    'r--',
                    linewidth=1.5,
                    alpha=0.8,
                    label="Perfect prediction")
            
            # 绘制20%相对误差线
            x_line = np.linspace(0, x_max, 100)
            y_upper = 1.2 * x_line
            y_lower = 0.8 * x_line
            
            # 只绘制在坐标轴范围内的部分
            mask_upper = y_upper <= y_max
            mask_lower = y_lower >= 0
            
            plt.plot(x_line[mask_upper], y_upper[mask_upper], 
                    'b--',
                    linewidth=1.2,
                    alpha=0.6,
                    label="+20% deviation")
            
            plt.plot(x_line[mask_lower], y_lower[mask_lower], 
                    'b--',
                    linewidth=1.2,
                    alpha=0.6,
                    label="-20% deviation")
            
            # 计算组合图中落在+20%线之下的点占全部点的比例
            # 合并所有目标变量的数据
            combined_x = []
            combined_y = []
            
            for data in all_data:
                combined_x.extend(data["x"].tolist())
                combined_y.extend(data["y"].tolist())
            
            if len(combined_x) > 0 and len(combined_y) > 0:
                combined_x_array = np.array(combined_x)
                combined_y_array = np.array(combined_y)
                
                # 计算落在+20%线之下的点
                within_upper_20 = (combined_y_array <= 1.2 * combined_x_array).sum()
                total_points = len(combined_x_array)
                within_pct = (within_upper_20 / total_points * 100)
                
                # 添加统计信息
                stats_text = f"Points under 20% line: {within_pct:.1f}%"
                
                plt.text(0.95, 0.05, stats_text,
                        transform=plt.gca().transAxes,
                        fontsize=16,
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # 设置图形属性
        plt.xlabel("True Values", fontsize=16)
        plt.ylabel("Predicted Values", fontsize=16)
        plt.title("Predicted vs Actual (Test) - All Targets", fontsize=16)
        
        # 设置网格
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 添加图例
        plt.legend(loc='upper left', fontsize=14)
        
        plt.tight_layout()
        
        # 保存组合图
        output_file = output_dir / f"scatter_all_targets.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close("all")
        
        print(f"  已保存组合图: {output_file}")
    
    print(f"\n全部完成！图片保存在: {output_dir.resolve()}")


if __name__ == "__main__":
    main()