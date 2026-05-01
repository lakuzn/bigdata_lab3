from deltalake import DeltaTable

storage_options = {
    "aws_access_key_id": "admin",
    "aws_secret_access_key": "password",
    "aws_endpoint_url": "http://minio:9000",
    "aws_region": "us-east-1",
    "aws_allow_http": "true",
}

SILVER_TABLE_PATH = "s3://lakehouse/silver/flights"

def run_maintenance():
    print("Подключение к Silver таблице для обслуживания...")
    dt = DeltaTable(SILVER_TABLE_PATH, storage_options=storage_options)
    
    # 1. OPTIMIZE (Compaction) - собирает мелкие Parquet-файлы в крупные для скорости
    # Z-ORDER - физически сортирует данные внутри файлов по колонке Origin, 
    # чтобы запросы по аэропортам работали мгновенно
    print("Запуск OPTIMIZE + Z-ORDER по колонке 'Origin'...")
    dt.optimize.z_order(["Origin"])
    print("Оптимизация завершена!")

    # 2. VACUUM - удаляет старые физические файлы, которые больше не нужны 
    # (по умолчанию старше 7 дней, но для теста можно поставить retention_hours=0)
    print("Запуск VACUUM...")
    dt.vacuum(retention_hours=168, enforce_retention_duration=False)
    print("Очистка завершена!")

if __name__ == "__main__":
    run_maintenance()