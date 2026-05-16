# src/modeling.py
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DATA_PATH
from src.preprocessing import run_preprocessing_pipeline

SEED = 42

def train_and_evaluate():
    # 1. Если обработанного файла нет, запускаем пайплайн предобработки
    if not os.path.exists(PROCESSED_DATA_PATH):
        df = run_preprocessing_pipeline()
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH)

    # 2. Разделение на X и y
    X = df.drop(columns=['Quantity Sold'])
    y = df['Quantity Sold']

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Масштабирование признаков (критично для Ridge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Конфигурация моделей для экспериментов из ТЗ
    configs = {
        "Ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
        "RandomForest": (RandomForestRegressor(random_state=SEED), {"n_estimators": [50, 100]}),
        "ExtraTrees": (ExtraTreesRegressor(random_state=SEED), {"n_estimators": [50, 100]}),
        "XGBoost": (XGBRegressor(random_state=SEED), {"n_estimators": [50, 100], "learning_rate": [0.1]}),
        "LightGBM": (LGBMRegressor(random_state=SEED, verbose=-1), {"n_estimators": [50, 100]})
    }

    best_overall_mae = float('inf')
    best_pipeline_model = None
    best_model_name = ""

    print("\n🏋️‍♂️ Начинаем процесс обучения моделей...")
    for name, (model, params) in configs.items():
        # Перебор гиперпараметров
        grid = GridSearchCV(model, params, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)

        # Ridge обучаем на масштабированных данных, древесные — на обычных
        if name == "Ridge":
            grid.fit(X_train_scaled, y_train)
            preds = grid.best_estimator_.predict(X_test_scaled)
        else:
            grid.fit(X_train, y_train)
            preds = grid.best_estimator_.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        print(f"📊 Модель: {name:14} | Лучшие параметры: {grid.best_params_} | Тест MAE: {mae:.4f}")

        # Выбор абсолютно лучшей модели
        if mae < best_overall_mae:
            best_overall_mae = mae
            best_pipeline_model = grid.best_estimator_
            best_model_name = name

    # 4. Вывод Feature Importance для лучшей модели (если это деревья/бустинг) или весов (если Ridge)
    print("\n💡 АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ДЛЯ ЛУЧШЕЙ МОДЕЛИ:")
    if best_model_name == "Ridge":
        importances = np.abs(best_pipeline_model.coef_)
    elif hasattr(best_pipeline_model, 'feature_importances_'):
        importances = best_pipeline_model.feature_importances_
    else:
        importances = None

    if importances is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(importance_df.head(10).to_string(index=False))

    # 5. Сохранение победителя
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Если победила Ridge, сохраняем вместе со скалером в кортеже, чтобы пайплайн не падал при инференсе
    if best_model_name == "Ridge":
        joblib.dump((scaler, best_pipeline_model), MODEL_PATH)
    else:
        joblib.dump(best_pipeline_model, MODEL_PATH)

    print(f"\n🥇 Победитель: {best_model_name} с MAE: {best_overall_mae:.4f}")
    print(f"💾 Модель успешно сохранена в: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
