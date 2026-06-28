# XGBoost_v4 使用说明

本项目用于对数据集最后三个目标值进行回归预测：

- fy_reduction
- fu_reduction
- E_reduction

关键规则：

- diff_steel 作为分组字段，仅用于 GroupKFold 与组内约束，不作为特征。
- 当前默认将 diff_steel 作为输出中的样本标识列（id_col）。
- Ceq 不作为建模特征（即使原始数据中存在该列，也会在训练前自动排除）。
- 特征、目标、分组信息统一由 data/feature_info.json 读取。

## 1. 环境准备

在项目根目录执行：

建议先激活你的训练环境（例如 xgboostv1）：

```powershell
conda activate xgboostv1
```

```powershell
pip install -r requirements.txt
```

## 2. 运行顺序

1. 准备原始数据：
	 - 将数据文件放在 data/steel_v2_cleaned_r2.csv
2. 检查配置：
	 - data/feature_info.json 中已定义特征列、目标列、分组列
3. 启动训练：

```powershell
python -m src.run_train
```

## 3. 训练流程说明

对每个目标列单独训练：

1. 使用 GroupKFold 按 diff_steel 做外层交叉验证。
2. 在每个 fold 的训练子集内执行 GridSearchCV（同样基于分组 CV）寻找最优参数。
3. 用该 fold 的最佳模型预测验证集并计算指标：R2、MAE、MSE、RMSE、P20。
4. 汇总每个目标的 fold 指标，并输出整体 CV 均值与标准差。

### Plan_B（缺失值原生处理）说明

Plan_B 用于对照实验，核心目的是验证“不做特征中位数填充”时的效果。

- 含义：保留特征缺失值（NaN），交由 XGBoost 原生缺失值分裂机制处理。
- 命名规则：Plan_B 输出统一追加 `_planB` 后缀，避免覆盖常规实验结果。
- 输出范围：仅保留
	- `results/metrics/test_summary_all_targets_planB.csv`
	- `results/figures/scatter_test_pred_vs_true_fy_reduction_planB.png`
	- `results/figures/scatter_test_pred_vs_true_fu_reduction_planB.png`
	- `results/figures/scatter_test_pred_vs_true_E_reduction_planB.png`
- 当前状态：`src/run_train.py` 默认按 Plan_B 配置运行。

## 4. 输出结果目录


- results/metrics/
	- cv_fold_metrics_xxx.csv：每个目标每个 fold 的 R2、MAE、MSE、RMSE、P20
	- cv_summary_all_targets.csv：所有目标的 CV 均值与标准差
- results/params/
	- best_params_xxx.csv：每个目标每个 fold 的最佳参数
	- best_params_xxx.json：每个目标代表性最佳参数 + 各 fold 参数
	- feature_importance_xxx.csv：每个目标按 fold 平均后的特征重要性
- results/predictions/
	- oof_predictions_xxx.csv：每个目标的 OOF 预测
	- oof_predictions_all_targets.csv：三目标合并 OOF 结果
- results/figures/
	- scatter_pred_vs_true_xxx.png：预测值 vs 真实值散点图（含 y=x 与 ±20% 偏移线）
	- feature_importance_xxx.png：特征重要性图
	- fold_metrics_xxx.png：各 fold 指标柱状图
- results/logs/
	- training_log.txt：训练日志

## 5. 代码结构

- src/config.py：统一配置
- src/data_loader.py：数据读取、feature_info 读取与字段检查
- src/features.py：特征、目标、分组拆分
- src/metrics.py：R2、MAE、P20 指标
- src/cv.py：分组交叉验证
- src/model.py：XGBoost 模型与参数网格
- src/trainer.py：训练主流程（每目标独立训练）
- src/plotter.py：画图模块
- src/utils.py：日志与通用工具
- src/run_train.py：训练入口

