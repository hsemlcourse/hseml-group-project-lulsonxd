import pandas as pd
from sklearn.preprocessing import StandardScaler

def pipeline_preprocessing(raw_data_path: str):
    """Функция загружает сырые данные, чистит их и масштабирует"""
    # 1. Загрузка
    df = pd.read_csv(raw_data_path)
    
    # 2. Твой код очистки (выделение фич, перевод категорий в числа, заполнение пропусков)
    # Пример:
    X = df.drop(columns=['Quantity Sold'])
    y = df['Quantity Sold']
    
    # 3. Масштабирование (обязательно для Ridge и PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y