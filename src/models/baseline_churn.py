# -----------------------------------------------------
# Baseline Churn Model (Leakage-Free)
# -----------------------------------------------------

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from src.config import FEATURE_DATA_PATH


def run_churn_model(feature_path):

    df = pd.read_csv(feature_path)

    # ----------------------------------------
    # Define churn realistically
    # ----------------------------------------
    # If Recency > 120 days → churn
    # BUT remove Recency from feature set
    # ----------------------------------------

    df["Churn"] = (df["Recency"] > 120).astype(int)

    X = df[["Frequency", "Monetary"]]  # remove Recency
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    # ----------------------------------------
    # MLflow Logging
    # ----------------------------------------

    mlflow.set_experiment("NeuralRetail_Week1")

    with mlflow.start_run():

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("AUC", auc)

        mlflow.sklearn.log_model(
            model,
            name="churn_model",
            input_example=X_train.iloc[:5]
        )

    print("Leakage-Free Churn Model AUC:", round(auc, 4))