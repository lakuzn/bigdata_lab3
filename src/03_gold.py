import polars as pl

storage_options = {
    "aws_access_key_id": "admin",
    "aws_secret_access_key": "password",
    "aws_endpoint_url": "http://minio:9000",
    "aws_region": "us-east-1",
    "aws_allow_http": "true",
}

SILVER_TABLE_PATH = "s3://lakehouse/silver/flights"
GOLD_AGGS_PATH = "s3://lakehouse/gold/aggregates"
GOLD_FEATURES_PATH = "s3://lakehouse/gold/features"

def process_gold():
    print("Чтение данных из Silver слоя...")
    # Используем lazy scan
    silver_lf = pl.scan_delta(SILVER_TABLE_PATH, storage_options=storage_options)

    print("Расчет аналитических агрегатов...")
    aggs_lf = (
        silver_lf
        .group_by(["Airline", "Origin", "season", "hour"])
        .agg([
            pl.col("ArrDelay").mean().alias("avg_arr_delay"),
            pl.col("DepDelay").mean().alias("avg_dep_delay"),
            pl.col("flight_id").count().alias("total_flights")
        ])
        # Оставляем только те срезы, где было больше 10 рейсов
        .filter(pl.col("total_flights") > 10)
        .sort(["avg_arr_delay"], descending=True)
    )
    
    aggs_df = aggs_lf.collect()
    print(f"Строк в агрегатах: {len(aggs_df)}")
    
    aggs_df.write_delta(
        GOLD_AGGS_PATH, 
        mode="overwrite", 
        storage_options=storage_options
    )

    print("Подготовка Feature Table для ML...")
    features_lf = (
        silver_lf
        # Убираем рейсы, где нет данных по задержке прибытия
        .filter(pl.col("ArrDelay").is_not_null())
        .with_columns([
            # Создаем целевую переменную для классификации (задержка > 15 минут)
            (pl.col("ArrDelay") > 15).cast(pl.Int32).alias("is_delayed")
        ])
        .select([
            "Airline", "Origin", "Dest", "month", "day_of_week", 
            "hour", "season", "DepDelay", "ArrDelay", "is_delayed"
        ])
        .drop_nulls()
    )

    features_df = features_lf.collect()
    print(f"Готово. Строк в ML Feature Table: {len(features_df)}")
    
    features_df.write_delta(
        GOLD_FEATURES_PATH, 
        mode="overwrite", 
        storage_options=storage_options
    )
    
    print("\nСлой Gold успешно сформирован.")

if __name__ == "__main__":
    process_gold()