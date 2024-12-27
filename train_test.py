from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from eda import read_csv_file

def train_test(df):
    X = df.drop('House_Price', axis=1)
    y = df['House_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, shuffle=True)

    # Только один тип масштабирования (StandardScaler)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def plot_actual(y_test, y_pred):
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red',
             linestyle='--')  # Линия идеального предсказания
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

def plot_tails(y_test, y_pred):
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')  # Горизонтальная линия на уровне 0
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

"""Linear Regression"""
def Linear_Regression_with_cv(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=101)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -cv_scores

    # Выводим результаты кросс-валидации
    print(f'MSE for each fold: {mse_scores}')
    print(f'Mean MSE (Cross-Validation): {mse_scores.mean()}')
    print(f'Standard Deviation of MSE (Cross-Validation): {mse_scores.std()}')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred

def XGBoost_with_cv(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=101)

    # Сетка гиперпараметров для поиска
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # Поиск лучших гиперпараметров
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_

    # Предсказание на тестовых данных
    y_pred = best_model.predict(X_test)

    # Возвращаем предсказания
    return y_pred

def RandomForest_with_cv(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=101)

    # Сетка гиперпараметров
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Настройка GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # Поиск лучших гиперпараметров
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_

    # Предсказания
    y_pred = best_model.predict(X_test)

    return y_pred


if __name__== "__main__":
    filepath = 'data/cleaned_dataset.csv'
    df = read_csv_file(filepath)

    X_train, X_test, y_train, y_test = train_test(df)

    y_pred = Linear_Regression_with_cv(X_train, X_test, y_train, y_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Вывод результатов
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape:.2f}%")

    plot_actual(y_test, y_pred)

    plot_tails(y_test, y_pred)

    y_pred = XGBoost_with_cv(X_train, X_test, y_train, y_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Вывод результатов
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape:.2f}%")

    plot_actual(y_test, y_pred)

    plot_tails(y_test, y_pred)

    y_pred = RandomForest_with_cv(X_train, X_test, y_train, y_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Вывод результатов
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape:.2f}%")

    plot_actual(y_test, y_pred)

    plot_tails(y_test, y_pred)