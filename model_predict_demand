import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import lightgbm as lgb

FEATURES = [
    'population',
    'pct_pensioners',
    'avg_income',
    'has_shop',
    'dist_to_city_km'
]

def load_data():
    villages = pd.read_csv('data/villages.csv')
    revenue = pd.read_csv('data/village_revenue.csv')
    data = villages.merge(revenue, on='village')
    X = data[FEATURES]
    y = data['avg_revenue']
    return data, X, y

def print_data_info(y):
    print(f"Средняя выручка: {y.mean():.0f} ₽")
    print(f"Медиана: {y.median():.0f} ₽")
    print(f"Диапазон: {y.min():.0f} – {y.max():.0f} ₽")
    print(f"Перекос распределения: {y.skew():.1f}")
    print()

def create_model():
    return lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        random_state=42,
        verbose=-1
    )

def check_baseline(X, y):
    dummy = DummyRegressor(strategy='median')
    mae = -cross_val_score(dummy, X, y, cv=5, scoring='neg_mean_absolute_error')
    r2 = cross_val_score(dummy, X, y, cv=5, scoring='r2')
    print('Baseline:')
    print(f'MAE: {mae.mean():.0f} ± {mae.std():.0f} ₽')
    print(f'R квадрат: {r2.mean():.3f}')
    print()
    return mae.mean()

def cross_validate_model(X, y):
    model = create_model()
    y_log = np.log1p(y)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_list = []
    r2_list = []

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_log.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        pred = np.clip(np.expm1(pred_log), 0, None)
        mae_list.append(mean_absolute_error(y_test, pred))
        r2_list.append(r2_score(y_test, pred))

    print('LightGBM cross-validation:')
    print(f'MAE: {np.mean(mae_list):.0f} ± {np.std(mae_list):.0f} ₽')
    print(f'R²:  {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}')
    print()
    return np.mean(mae_list)

def check_holdout(X, y):
    model = create_model()
    y_log = np.log1p(y)
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.25, random_state=42
    )
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    pred = np.clip(np.expm1(pred_log), 0, None)
    y_test = np.expm1(y_test_log)
    print('Holdout test:')
    print(f'MAE: {mean_absolute_error(y_test, pred):.0f} ₽')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.0f} ₽')
    print(f'R квадрат: {r2_score(y_test, pred):.3f}')
    print()
    return model

def print_feature_importance(model):
    imp = pd.Series(model.feature_importances_, index=FEATURES)
    print('Важность признаков:')
    print(imp.sort_values(ascending=False))
    print()

def save_forecast(df, X, y):
    model = create_model()
    y_log = np.log1p(y)
    model.fit(X, y_log)
    pred_log = model.predict(X)
    pred = np.clip(np.expm1(pred_log), 0, None)
    df['predicted_revenue'] = pred.round(0)
    df['error_pct'] = ((df['predicted_revenue'] - df['avg_revenue']) / df['avg_revenue'] * 100).round(1)

    output = df[[
        'village',
        'population',
        'avg_income',
        'avg_revenue',
        'predicted_revenue',
        'error_pct'
    ]]

    output = output.sort_values('predicted_revenue', ascending=False)
    output.to_csv('data/demand_forecast.csv', index=False)
    print('Топ 10 деревень по прогнозируемой выручке:')
    print(output.head(10).to_string(index=False))


df, X, y = load_data()
print_data_info(y)

base_mae = check_baseline(X, y)
model_mae = cross_validate_model(X, y)
improve = 1 - model_mae / base_mae

print(f'Улучшение MAE: {base_mae - model_mae:.0f} ₽')
print(f'Улучшение в процентах: {improve * 100:.0f}%')
final_model = check_holdout(X, y)
print_feature_importance(final_model)

save_forecast(df, X, y)
