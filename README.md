[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Прогнозирование рыночной стоимости и спроса на мобильные устройства и ноутбуки.

**Студент:** Сметанин Л. В. 

**Группа:** БИВ236


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Регрессия

**Датасет:** [Mobiles & laptop Sales Data](https://www.kaggle.com/datasets/vinothkannaece/mobiles-and-laptop-sales-data)

**Целевая метрика:** MSE


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-lulsonxd.git
cd hseml-group-project-lulsonxd

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Пройтись линтером (опционально)
ruff check src/

# 5. Запуск
python -m src.modeling
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты

Модель	Ошибка (MAE)	Лучшие параметры
0	Ridge	2.4939	{'alpha': 10.0}
1	RandomForest	2.5439	{'n_estimators': 100}
2	ExtraTrees	2.6165	{'n_estimators': 100}
3	XGBoost	2.4967	{'learning_rate': 0.1, 'n_estimators': 50}
4	LightGBM	2.4961	{'n_estimators': 50}

Baseline: Ridge, MAE: 2.4939
Лучшая модель: Ridge


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
