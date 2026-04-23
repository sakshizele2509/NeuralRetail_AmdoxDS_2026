# -----------------------------------------------------
# Baseline Demand Forecast using Prophet
# -----------------------------------------------------

import pandas as pd
import mlflow
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from src.config import SILVER_DATA_PATH


def run_forecast_model():

    df = pd.read_csv(SILVER_DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Aggregate daily revenue
    daily = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index()
    daily.columns = ["ds", "y"]

    train = daily[:-30]
    test = daily[-30:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    preds = forecast["yhat"][-30:]
    mape = mean_absolute_percentage_error(test["y"], preds)

    mlflow.set_experiment("NeuralRetail_Week1")

    with mlflow.start_run():

        mlflow.log_param("model_type", "Prophet")
        mlflow.log_metric("MAPE", mape)
        mlflow.prophet.log_model(model, name="forecast_model")

    print("Forecast MAPE:", round(mape, 4))