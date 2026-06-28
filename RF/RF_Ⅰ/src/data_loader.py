# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(
    csv_path,
    target_col,
    targets=None,
    test_size=0.2,
    random_state=42
):
    """
    从 CSV 读取数据并返回划分后的训练/测试集。

    `targets` 可选地传入当前所有的预测列名列表，
    这样函数在构建特征矩阵时会自动删除其他目标列，
    避免模型使用未来/泄露信息作为特征。
    """
    """
    从 CSV 读取数据并返回划分后的训练/测试集。

    在循环不同预测指标时，我们希望 **只删除当前目标列为空的行**，
    而保留其它列即使它们含有缺失值。这样每次训练时的样本集
    只基于对应目标有效即可。

    参数:
        csv_path: 数据 CSV 路径
        target_col: 当前训练的目标列名（只检查该列的缺失）
        test_size: 测试集占比
        random_state: 随机种子

    返回:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(csv_path)

    # 统一氮元素列名：历史数据可能使用 N_pro，新数据使用 N。
    if 'N' not in df.columns and 'N_pro' in df.columns:
        df = df.rename(columns={'N_pro': 'N'})
    elif 'N' in df.columns and 'N_pro' in df.columns:
        # 若两列同时存在，优先保留 N，并用 N_pro 补齐缺失后删除旧列。
        df['N'] = df['N'].fillna(df['N_pro'])
        df = df.drop(columns=['N_pro'])

    # 替换空格字符串为 NaN（方便后续判断）
    df = df.replace(r'^\s+$', pd.NA, regex=True)
    print("原始数据总数:", len(df))

    # 如果提供了完整目标列表，则删除除当前 target 之外的其它列，
    # 以免模型看到其他指标作为特征。
    if targets is not None:
        other_targets = [t for t in targets if t != target_col]
        if other_targets:
            print(f"从数据中删除其他目标列：{other_targets}")
            df = df.drop(columns=other_targets, errors='ignore')

    # —— 只删除当前 target 为空的行 —— #
    df = df.dropna(subset=[target_col])
    print("当前 target:", target_col)
    print("当前 target 有效样本数:", len(df))

    # 针对其它特征的缺失，采用简单的填充策略而不是丢弃行，
    # 保证样本数足够。我们先把所有特征转为数值型（非数值转换为 NaN），
    # 然后计算中位数进行填充。
    if df.isna().any().any():
        missing_before = df.isna().sum()
        print("发现以下列有缺失，将使用中位数填充：")
        print(missing_before[missing_before > 0])

        # 将所有列（除目标）转换为 numeric 类型
        for col in df.columns:
            if col != target_col:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算各列中位数
        medians = df.median(numeric_only=True)
        # 删除完全为空的列，因为中位数也为 NaN
        all_missing = medians[medians.isna()].index.tolist()
        if all_missing:
            print(f"以下列全部缺失，将被删除：{all_missing}")
            df = df.drop(columns=all_missing)
            medians = medians.drop(all_missing)

        # 填充其余缺失值
        df = df.fillna(medians)
        missing_after = df.isna().sum()
        print("填充后缺失情况：")
        print(missing_after[missing_after > 0])

    # 先把目标列也转换为数值，coerce 会把无法转换的项置为 NaN。
    y = pd.to_numeric(df[target_col], errors='coerce')
    # 如果转换后又出现了 NaN，则将这些行剔除
    if y.isna().any():
        count = y.isna().sum()
        print(f"目标列转换为数值后出现 {count} 个 NaN，将从数据中删除")
        df = df[~y.isna()]
        y = y[~y.isna()]

    # 将 diff_steel 用作分组变量，避免来自同一
    # 来源/批次的样本同时出现在训练集和测试集中。注意：
    # 这些列不会作为特征输入模型。
    groups = None
    if 'diff_steel' in df.columns:
        groups = df['diff_steel']

    X = df.drop(columns=[target_col])

    # 删除不应作为预测变量的标识列
    for col in ('steel_ID', 'source_ref', 'diff_steel'):
        if col in X.columns:
            X = X.drop(columns=[col])

    # 确保所有数据都是数值类型
    X = X.astype(float)
    y = y.astype(float)

    groups_train = groups_test = None
    if groups is not None:
        # 使用 GroupShuffleSplit 进行分层切分
        from sklearn.model_selection import GroupShuffleSplit
        splitter = GroupShuffleSplit(test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    # 返回训练/测试集、特征名以及训练/测试分组信息（如果使用了
    # source_ref/steel_ID 作为分组变量）。group 信息可传给交叉
    # 验证函数避免混合来源样本。
    return X_train, X_test, y_train, y_test, X.columns.tolist(), groups_train, groups_test
