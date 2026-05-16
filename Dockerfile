FROM python:3.10-slim

WORKDIR /app

# 1. СНАЧАЛА копируем только файл с зависимостями
COPY requirements.txt .

# 2. Устанавливаем библиотеки. 
# Этот шаг закэшируется и НЕ БУДЕТ перезапускаться, пока ты не изменишь сам requirements.txt!
RUN pip install --no-cache-dir -r requirements.txt

# 3. И ТОЛЬКО ПОТОМ копируем файлы исходного кода (src, pyproject.toml)
COPY src/ ./src/
COPY pyproject.toml .

# 4. Запуск линтера и пайплайна
CMD ["sh", "-c", "ruff check src/ && python -m src.modeling"]