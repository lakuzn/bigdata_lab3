import os
import glob
import polars as pl

# Настройки для подключения к MinIO 
storage_options = {
    "aws_access_key_id": "admin",
    "aws_secret_access_key": "password",
    "aws_endpoint_url": "http://minio:9000",
    "aws_region": "us-east-1",
    "aws_allow_http": "true",
}

# Путь в нашем бакете MinIO, куда будем сохранять Delta-таблицу
BRONZE_TABLE_PATH = "s3://lakehouse/bronze/flights"

def load_to_bronze():
    
    csv_files = sorted(glob.glob("data/raw/*.csv"))

    print(f"Найдено файлов для загрузки: {len(csv_files)}")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"Обработка батча (файла): {file_name}...")

        # Читаем CSV через Polars.
        # infer_schema_length=10000 полезно, так как
        # типы колонок в первых строках могут быть неочевидны
        # ignore_errors=True помогает пропустить битые строки, если они есть
        df = pl.read_csv(
            file_path, 
            infer_schema_length=10000, 
            ignore_errors=True,
            null_values=["NA", "NaN", ""] # Сразу говорим, что считать за NULL
        )
        
        # Записываем в Delta Lake в режиме append
        df.write_delta(
            BRONZE_TABLE_PATH,
            mode="append",
            storage_options=storage_options,
            delta_write_options={"schema_mode": "merge"} # Разрешаем schema evolution
        )
        
        print(f"Файл {file_name} успешно добавлен в Bronze.")

    print("\nЗагрузка в Bronze layer завершена.")

if __name__ == "__main__":
    load_to_bronze()