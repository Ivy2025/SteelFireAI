# src/model.py
from sklearn.ensemble import RandomForestRegressor


def build_rf_model(random_state=42, **rf_params):
    """
    构建随机森林回归器。

    rf_params 可以传入任意 RandomForestRegressor 支持的参数，
    例如 max_depth、min_samples_leaf、max_features 等。
    如果未在 rf_params 中指定，以下默认值将用于抑制过拟合：
      - n_estimators=500
      - max_depth=10
      - min_samples_split=4
      - min_samples_leaf=5
      - max_features='sqrt'
    """

    defaults = {
        'n_estimators': 500,
        'max_depth': 10,
        'min_samples_split': 4,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
    }
    # 将用户提供的参数覆盖默认值
    defaults.update(rf_params)

    model = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        **defaults
    )
    return model
