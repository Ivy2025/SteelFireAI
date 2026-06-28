# LightGBM 钢材高温力学性能退化系数回归

本项目使用 `lightgbm.LGBMRegressor` 对以下 3 个目标分别训练独立回归模型：
- `fy_reduction`
- `fu_reduction`
- `E_reduction`

当前实验方案：
- `plan3`: 样本随机划分（不使用 `diff_steel` 作为分组依据）+ 中位数填补 + LightGBM
- `planB`: 样本随机划分（不使用 `diff_steel` 作为分组依据）+ 不填补（由 LightGBM 原生处理缺失值）

## 1. 环境准备

建议先创建并激活 Conda 环境：

```powershell
conda create -n lightgbm python=3.11.0 -y
conda activate lightgbm
```

安装依赖：

```powershell
pip install -r requirements.txt
```

## 2. 数据路径

默认数据路径为：

- `data/steel_v2_cleaned_r2.csv`

如果你传入 `steel_v2_cleaned_r2.csv` 且当前目录找不到，程序会自动尝试回退到 `data/steel_v2_cleaned_r2.csv`。

## 3. 运行指令

### 3.1 默认运行（plan3 + 三目标）

```powershell
python main.py
```

### 3.2 指定数据路径和输出前缀

```powershell
python main.py --data_path steel_v2_cleaned_r2.csv --output_prefix lightgbm
```

### 3.3 指定方案

```powershell
python main.py --plans plan3
```

运行不填补的对照组（LightGBM 原生缺失值处理）：

```powershell
python main.py --plans planB
```

### 3.4 只跑某个目标（加速迭代）

只跑 `E_reduction`：

```powershell
python main.py --targets E_reduction
```

同时只跑 `plan3 + E_reduction`：

```powershell
python main.py --plans plan3 --targets E_reduction
```

### 3.5 自定义随机种子

```powershell
python main.py --random_state 42
```

### 3.6 查看全部参数帮助

```powershell
python main.py -h
```

## 4. 输出目录

运行后结果默认保存到 `results/`：

- `results/metrics/`
- `results/predictions/`
- `results/models/`
- `results/tuning/`
- `results/figures/`

可通过参数修改输出根目录：

```powershell
python main.py --result_dir results
```

## 5. 说明

- 输出后缀由 `--plans` 决定（例如 `lightgbm_plan3_metrics.csv`、`lightgbm_planB_metrics.csv`）。
- 输出文件名前缀由 `--output_prefix` 控制，默认前缀为 `lightgbm`。
- 数据划分采用样本随机划分（不再使用 `diff_steel` 分组划分）。
- 物理约束：目标值不可能为负，因此模型预测后统一施加非负约束；当预测值小于 0 时取 0（即 `y_pred = max(y_pred, 0)`）。
- 该约束已在训练集预测、测试集预测和 OOF 预测上统一生效，因此指标计算、图表和导出预测文件均基于非负约束后的结果。
