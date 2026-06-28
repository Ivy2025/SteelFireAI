# LightGBM 钢材高温力学性能退化系数回归

本项目使用 `lightgbm.LGBMRegressor` 对以下 3 个目标分别训练独立回归模型：
- `fy_reduction`
- `fu_reduction`
- `E_reduction`

并支持两套实验方案：
- `plan1`: 不做缺失值填补，使用 LightGBM 原生缺失值处理
- `plan2`: 先做中位数填补，再训练 LightGBM

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

### 3.1 默认运行（两方案 + 三目标）

```powershell
python main.py
```

### 3.2 指定数据路径和输出前缀

```powershell
python main.py --data_path steel_v2_cleaned_r2.csv --output_prefix lightgbm
```

### 3.3 只跑某个方案（加速排错）

只跑 `plan2`：

```powershell
python main.py --plans plan2
```

只跑 `plan1`：

```powershell
python main.py --plans plan1
```

### 3.4 只跑某个目标（加速迭代）

只跑 `E_reduction`：

```powershell
python main.py --targets E_reduction
```

同时只跑 `plan2 + E_reduction`：

```powershell
python main.py --plans plan2 --targets E_reduction
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

- 只运行单个方案时，不会生成 `plan1_vs_plan2` 对比文件，这是预期行为。
- 输出文件名前缀由 `--output_prefix` 控制，默认前缀为 `lightgbm`。
- 模型训练采用分组划分（`diff_steel`），避免同组样本泄漏。
