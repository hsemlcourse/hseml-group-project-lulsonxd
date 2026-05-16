# src/config.py
import os

# Идентификатор датасета на Kaggle
KAGGLE_DATASET = "vinothkannaece/mobiles-and-laptop-sales-data"

# Пути к данным
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "sales_data.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_sales.csv")

# Путь для сохранения модели
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
