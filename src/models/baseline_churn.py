# -----------------------------------------------------
# Baseline Churn Model
# Logistic Regression
# -----------------------------------------------------

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

def run_churn_model(feature_path):

    df = pd.read_csv(feature_path)

    # Fake churn label for baseline
    df["Churn"] = (df["Recency"] > 90).astype(int)

    X = df[["Recency", "Frequency", "Monetary"]]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)

    mlflow.set_experiment("NeuralRetail_Week1")

    with mlflow.start_run():
        mlflow.log_metric("AUC", auc)
        mlflow.sklearn.log_model(model, "churn_model")

    print("Churn Model AUC:", auc)