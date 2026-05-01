#!/bin/bash

echo "Запуск Lakehouse Pipeline..."

# Останавливаем скрипт, если хоть один шаг упадет с ошибкой
set -e 

echo "Шаг 1: Bronze Layer"
python src/01_bronze.py

echo "Шаг 2: Silver Layer"
python src/02_silver.py

echo "Шаг 3: Gold Layer"
python src/03_gold.py

echo "Шаг 4: ML Training"
python src/04_ml_pipeline.py

echo "Шаг 5: Maintenance "
python src/05_optimize.py

echo "Пайплайн успешно завершен!"