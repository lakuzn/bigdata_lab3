import polars as pl
from deltalake import DeltaTable

storage_options = {
    "aws_access_key_id": "admin",
    "aws_secret_access_key": "password",
    "aws_endpoint_url": "http://minio:9000",
    "aws_region": "us-east-1",
    "aws_allow_http": "true",
}

BRONZE_TABLE_PATH = "s3://lakehouse/bronze/flights"
SILVER_TABLE_PATH = "s3://lakehouse/silver/flights"

def process_silver():
    print("Инициализация LazyFrame из Bronze...")
    # 1. Lazy Scan
    lf = pl.scan_delta(BRONZE_TABLE_PATH, storage_options=storage_options)

    # 2. Pipeline
    silver_lf = (
        lf
        # переименовываем колонку с кодом авиакомпании в Airline для удобства
        .rename({"IATA_Code_Marketing_Airline": "Airline"})
            
        # Фильтрация
        .filter(
            (pl.col("Cancelled") == False) | (pl.col("Cancelled") == 0) & 
            pl.col("ArrDelay").is_not_null()
        )
            # Приводим строковую дату в тип Date
        .with_columns(
            pl.col("FlightDate").str.to_date("%Y-%m-%d").alias("date")
        )

        # Генерация производных признаков
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.weekday().alias("day_of_week"),
            # CRSDepTime обычно записан как 1530
            (pl.col("CRSDepTime") // 100).cast(pl.Int32).alias("hour"),
            # Сезон
            ((pl.col("date").dt.month() % 12) // 3 + 1).alias("season"),
            # Маршрут
            (pl.col("Origin") + "-" + pl.col("Dest")).alias("route"),
            
            # Генерируем суррогатный ключ (ID) для каждого рейса, чтобы делать MERGE
            # Собираем его из даты, авиакомпании, маршрута и времени
            (pl.col("FlightDate").cast(pl.String) + "_" + 
             pl.col("Airline") + "_" + 
             pl.col("Origin") + "_" + 
             pl.col("Dest") + "_" + 
             pl.col("CRSDepTime").cast(pl.String)).alias("flight_id")
        ])
        # Оставляем только нужные колонки
        .select([
            "flight_id", "year", "month", "day_of_week", "hour", "season", "date",
            "Airline", "Origin", "Dest", "route", "DepDelay", "ArrDelay"
        ])
    )

    # Вывод физического плана для README
    print("\n=== Физический план запроса (.explain) для README ===")
    print(silver_lf.explain())
    print("=====================================================\n")

    # 3. Выполнение вычислений
    print("Запуск вычислений графа и сбор данных...")
    df = silver_lf.collect()
    print(f"Обработано строк: {len(df)}")

    # 4. Запись в Delta Lake (с партицированием и логикой MERGE)
    try:
        dt = DeltaTable(SILVER_TABLE_PATH, storage_options=storage_options)
        print("Silver-таблица найдена. Запуск операции MERGE (UPSERT)...")
        (
            dt.merge(
                source=df.to_arrow(),
                predicate="target.flight_id = source.flight_id",
                source_alias="source",
                target_alias="target"
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
        print("MERGE успешно завершен!")
        
    except Exception as e:
        print("Silver-таблица не найдена. Выполняем OVERWRITE...")
        df.write_delta(
            SILVER_TABLE_PATH,
            mode="overwrite",
            storage_options=storage_options,
            delta_write_options={"partition_by": ["year", "month"]}
        )
        print("Таблица создана и партицирована.")

if __name__ == "__main__":
    process_silver()