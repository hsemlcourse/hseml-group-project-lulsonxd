FROM python:3.10-slim

WORKDIR /app

# Системная зависимость для компиляции LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Команда сначала проверяет код линтером ruff, а затем запускает долгое обучение
CMD ["sh", "-c", "ruff check src/ && python src/train.py"]