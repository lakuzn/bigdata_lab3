import os
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from deltalake import DeltaTable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import OrdinalEncoder

# Экспортируем переменные окружения для boto3
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Настройки MinIO
storage_options = {
    "aws_access_key_id": "admin",
    "aws_secret_access_key": "password",
    "aws_endpoint_url": "http://minio:9000",
    "aws_region": "us-east-1",
    "aws_allow_http": "true",
}

GOLD_FEATURES_PATH = "s3://lakehouse/gold/features"

# Настраиваем MLflow на наш локальный сервер
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Flight_Delay_Prediction")

def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_imp["Feature"], df_imp["Importance"], color='skyblue')
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig

def run_ml_pipeline():
    print("Подключение к Gold таблице...")
    dt = DeltaTable(GOLD_FEATURES_PATH, storage_options=storage_options)
    gold_version = dt.version()
    print(f"Используется версия данных (Time Travel): v{gold_version}")

    df = dt.to_pandas()
    
    # Для ускорения лабы можно взять случайную подвыборку (100k строк). Если нужно, можно раскомментить строку внизу
    # df = df.sample(n=100000, random_state=42) 

    # Кодируем строковые категории в числа
    cat_cols = ["Airline", "Origin", "Dest"]
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Определяем фичи (X) и таргеты (y)
    feature_cols = ["Airline", "Origin", "Dest", "month", "day_of_week", "hour", "season", "DepDelay"]
    X = df[feature_cols]
    
    y_reg = df["ArrDelay"]      # Для регрессии 
    y_clf = df["is_delayed"]    # Для классификации 

    # Разбиваем на train/test
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # 1. Модель Регрессии (предсказание минут)
    print("\nОбучение модели регрессии...")
    with mlflow.start_run(run_name="RandomForest_Regression"):
        # Логируем версию данных
        mlflow.log_param("gold_table_version", gold_version)
        
        reg_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        reg_model.fit(X_train, y_train_reg)
        
        preds_reg = reg_model.predict(X_test)
        rmse = root_mean_squared_error(y_test_reg, preds_reg)
        mae = mean_absolute_error(y_test_reg, preds_reg)
        
        print(f"Регрессия -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        mlflow.log_params({"n_estimators": 50, "max_depth": 10})
        mlflow.log_metrics({"rmse": rmse, "mae": mae})
        mlflow.sklearn.log_model(reg_model, "model_regression")
        
        # График важности признаков
        fig_reg = plot_feature_importance(reg_model, feature_cols, "Feature Importance (Regression)")
        mlflow.log_figure(fig_reg, "feature_importance_reg.png")

    # 2. Модель Классификации (предсказание факта задержки > 15 мин)
    print("\nОбучение модели классификации...")
    with mlflow.start_run(run_name="RandomForest_Classification"):
        mlflow.log_param("gold_table_version", gold_version)
        
        clf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        clf_model.fit(X_train, y_train_clf)
        
        preds_clf = clf_model.predict(X_test)
        acc = accuracy_score(y_test_clf, preds_clf)
        f1 = f1_score(y_test_clf, preds_clf)
        
        print(f"Классификация -> Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")
        
        mlflow.log_params({"n_estimators": 50, "max_depth": 10})
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        mlflow.sklearn.log_model(clf_model, "model_classification")
        
        fig_clf = plot_feature_importance(clf_model, feature_cols, "Feature Importance (Classification)")
        mlflow.log_figure(fig_clf, "feature_importance_clf.png")

    print("\nОбучение завершено! Все данные успешно отправлены в MLflow.")

if __name__ == "__main__":
    run_ml_pipeline()