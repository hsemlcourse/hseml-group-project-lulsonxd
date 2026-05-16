# src/preprocessing.py
import os
import pandas as pd
import kagglehub
from src.config import KAGGLE_DATASET, RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH

def download_data():
    """Скачивает датасет с Kaggle через kagglehub и сохраняет в data/raw"""
    print("🚀 Загрузка датасета с Kaggle...")
    # Скачиваем последнюю версию датасета
    cache_path = kagglehub.dataset_download(KAGGLE_DATASET)

    # Ищем CSV файл в скачанной папке
    files = [f for f in os.listdir(cache_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("CSV файл не найден в скачанном датасете.")

    full_cache_path = os.path.join(cache_path, files[0])
    
    # Создаем локальную папку data/raw и копируем файл туда
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df = pd.read_csv(full_cache_path)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Исходные данные сохранены в: {RAW_DATA_PATH}")
    return df

def clean_data(df):
    """Базовая очистка типов данных и пропусков"""
    df = df.copy()
    # Преобразование дат
    df['Inward Date'] = pd.to_datetime(df['Inward Date'])
    df['Dispatch Date'] = pd.to_datetime(df['Dispatch Date'])

    # Парсинг числовых значений из строк (RAM, ROM, SSD)
    for col in ['RAM', 'ROM', 'SSD']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

    # Заполнение пропусков в специфичных характеристиках
    df['Core Specification'] = df['Core Specification'].fillna('None')
    df['Processor Specification'] = df['Processor Specification'].fillna('None')

    return df

def feature_engineering(df):
    """Генерация новых признаков и удаление лишнего текста"""
    df = df.copy()
    # 1. Время хранения на складе
    df['Days_In_Stock'] = (df['Dispatch Date'] - df['Inward Date']).dt.days

    # 2. Месяц продажи (сезонность)
    df['Month_Sold'] = df['Dispatch Date'].dt.month

    # 3. Флаг премиум-бренда
    premium_brands = ['Apple', 'Samsung']
    df['Is_Premium'] = df['Brand'].apply(lambda x: 1 if str(x) in premium_brands else 0)

    # Удаляем неинформативные текстовые колонки для обучения
    to_drop = ['Product Code', 'Product Specification', 'Customer Name',
               'Inward Date', 'Dispatch Date', 'Customer Location']
    
    # Оставляем только те колонки, которые реально есть в датасете
    existing_drops = [col for col in to_drop if col in df.columns]
    return df.drop(columns=existing_drops)

def run_preprocessing_pipeline():
    """Полный цикл предобработки данных"""
    # 1. Скачивание/загрузка
    df = download_data()

    # 2. Очистка и фичи
    df = clean_data(df)
    df = feature_engineering(df)

    # 3. One-Hot Encoding для оставшихся категориальных признаков перед обучением
    # Выделяем таргет, чтобы он не закодировался случайно
    target_col = 'Quantity Sold'
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X = pd.get_dummies(X, drop_first=True)
        df_final = pd.concat([X, y], axis=1)
    else:
        df_final = pd.get_dummies(df, drop_first=True)

    # 4. Сохранение финального результата
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✨ Очищенные и закодированные данные сохранены в: {PROCESSED_DATA_PATH}")
    return df_final

if __name__ == "__main__":
    run_preprocessing_pipeline()