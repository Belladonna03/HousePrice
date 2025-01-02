from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json
from eda import read_csv_file, check_dir

# Функция для разделения данных на обучающую и тестовую выборки с предварительной обработкой
def train_test(df):
    df = df.reset_index(drop=True)
    df['House_Price'] = np.log1p(df['House_Price'])  # Логарифмирование целевой переменной
    X = df.drop('House_Price', axis=1)
    y = df['House_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, shuffle=True)

    # Масштабирование признаков
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns.tolist()

# Функция для построения графика "Фактические vs Предсказанные"
def plot_actual(y_test, y_pred, model_name):
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted for {model_name}')
    filename = f"results/actual_vs_predicted_{model_name}.png"
    plt.savefig(filename)
    plt.show()

# Функция для построения графика остатков
def plot_tails(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')  # Горизонтальная линия на уровне 0
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model_name}')
    filename = f"results/residual_plot_{model_name}.png"
    plt.savefig(filename)
    plt.show()

# Построение важности признаков
def plot_feature_importance(model, feature_names, model_name):
    importances = model.feature_importances_

    # Визуализация важности признаков
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), feature_names)  # Используем список feature_names
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance for {model_name}")
    plt.tight_layout()
    filename = f"results/feature_importance_{model_name}.png"
    plt.savefig(filename)
    plt.show()

"""Linear Regression"""
def Linear_Regression_with_cv(X_train, X_test, y_train):
    # Линейная регрессия с кросс-валидацией
    model = LinearRegression()
    # Настраиваем кросс-валидацию с 5 фолдами
    kf = KFold(n_splits=5, shuffle=True, random_state=101)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -cv_scores

    # Выводим результаты кросс-валидации
    print(f'MSE for each fold: {mse_scores}')
    print(f'Mean MSE (Cross-Validation): {mse_scores.mean()}')
    print(f'Standard Deviation of MSE (Cross-Validation): {mse_scores.std()}')

    # Обучаем модель на полной обучающей выборке
    model.fit(X_train, y_train)
    # Делаем предсказания на тестовой выборке
    y_pred = model.predict(X_test)

    return y_pred

"""XGBoost"""
def XGBoost_with_cv(X_train, X_test, y_train, feature_names):
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=101)
    # Задаем сетку гиперпараметров
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    # Выполняем GridSearchCV для поиска лучших параметров
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error',
                               cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    # Строим график важности признаков
    plot_feature_importance(best_model, feature_names, 'XGBoost')
    return y_pred

"""Random Forest"""
def RandomForest_with_cv(X_train, X_test, y_train, feature_names):
    model = RandomForestRegressor(random_state=101)
    # Сетка гиперпараметров
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # GridSearchCV для выбора оптимальных параметров
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error',
                               cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    # Строим график важности признаков
    plot_feature_importance(best_model, feature_names, 'Random Forest')
    return y_pred

def sklearn_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results = {
        'MSE': f'{mse:.3f}',
        'RMSE': f'{rmse:.3f}',
        'MAE': f'{mae:.3f}',
        'R2': f'{r2:.3f}',
        'MAPE': f'{mape:.3f}'
    }

    return results


def manual_metrics(y_test, y_pred):
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    y_mean = np.mean(y_test)
    r2 = 1 - np.mean((y_test - y_pred)**2) / np.mean((y_test - y_mean)**2)

    if np.any(y_test == 0):
        raise ValueError("y_test contains zero values, MAPE is undefined.")
    mape = np.mean(np.abs((y_test - y_pred) / y_test))

    results = {
        'MSE': f'{mse:.3f}',
        'RMSE': f'{rmse:.3f}',
        'MAE': f'{mae:.3f}',
        'R2': f'{r2:.3f}',
        'MAPE': f'{mape:.3f}'
    }

    return results


if __name__ == "__main__":
    # Загрузка данных
    filepath = 'data/cleaned_dataset.csv'
    df = read_csv_file(filepath)

    # Разделение данных на тестовую и обучающую выборки
    X_train, X_test, y_train, y_test, feature_names = train_test(df)

    # Проверка существования директории
    check_dir('results')

    data = {}

    # Linear Regression
    model_name = 'LinearRegression'
    y_pred = Linear_Regression_with_cv(X_train, X_test, y_train)
    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)

    # Сохранение метрик для текущей модели
    data[model_name] = {
            'sklearn':
                sklearn_metrics(y_test, y_pred),
            'manual':
                manual_metrics(y_test, y_pred)
    }

    plot_actual(y_test, y_pred, model_name)
    plot_tails(y_test, y_pred, model_name)

    # Random Forest
    model_name = 'RandomForest'
    y_pred = RandomForest_with_cv(X_train, X_test, y_train, feature_names)
    y_pred = np.expm1(y_pred)

    # Сохранение метрик для текущей модели
    data[model_name] = {
            'sklearn':
                sklearn_metrics(y_test, y_pred),
            'manual':
                manual_metrics(y_test, y_pred)
    }

    plot_actual(y_test, y_pred, model_name)
    plot_tails(y_test, y_pred, model_name)

    # XGBoost
    model_name = 'XGBoost'
    y_pred = XGBoost_with_cv(X_train, X_test, y_train, feature_names)
    y_pred = np.expm1(y_pred)

    # Сохранение метрик для текущей модели
    data[model_name] = {
            'sklearn':
                sklearn_metrics(y_test, y_pred),
            'manual':
                manual_metrics(y_test, y_pred)
    }

    plot_actual(y_test, y_pred, model_name)
    plot_tails(y_test, y_pred, model_name)

    filename = 'results/metrics.json'
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)