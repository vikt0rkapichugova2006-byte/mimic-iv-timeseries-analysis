# Анализ временных рядов MIMIC-IV v3.1

## Описание проекта
Курсовая работа по первичному анализу многомерных временных рядов медицинских показателей пациентов ОИТ.
Цель: прогнозирование жизненно важных показателей (HeartRate, SysBP, DiaBP, RespRate, Temperature, SpO2).

## Набор данных
- Название: MIMIC-IV (Medical Information Mart for Intensive Care), v3.1
- Источник: PhysioNet (https://physionet.org/content/mimiciv/3.1/)
- Тип данных: Многомерный временной ряд с часовой дискретизацией
- Конфиденциальность: Данные полностью анонимизированы

## Структура репозитория
- `data_ts/` — исходные данные (CSV)
- `images_ts/` — 7 визуализаций первичного анализа
- `notebooks/` — Jupyter Notebook (опционально)
- `analysis_timeseries.py` — скрипт генерации графиков
- `requirements.txt` — зависимости
