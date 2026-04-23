# -----------------------------------------------------
# Feature Engineering Script
# Week-1: RFM + Time-based Features
# -----------------------------------------------------

import os
import pandas as pd
from src.config import FEATURE_DATA_PATH


def build_features(df):
    """
    Creates customer-level RFM features.

    R = Recency   (days since last purchase)
    F = Frequency (number of invoices)
    M = Monetary  (total spending)

    Saves final feature dataset into data/features/
    """

    # -------------------------------------------------
    # Safety Check
    # -------------------------------------------------
    required_columns = [
        "CustomerID",
        "InvoiceDate",
        "InvoiceNo",
        "TotalPrice"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # -------------------------------------------------
    # Ensure datetime format
    # -------------------------------------------------
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # -------------------------------------------------
    # Snapshot date (for Recency calculation)
    # -------------------------------------------------
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # -------------------------------------------------
    # RFM Aggregation
    # -------------------------------------------------
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",     # count unique invoices
        "TotalPrice": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # -------------------------------------------------
    # Time-based Features (transaction-level)
    # (Not used in churn yet, but useful later)
    # -------------------------------------------------
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Month"] = df["InvoiceDate"].dt.month

    # -------------------------------------------------
    # Ensure output folder exists
    # -------------------------------------------------
    os.makedirs(os.path.dirname(FEATURE_DATA_PATH), exist_ok=True)

    # -------------------------------------------------
    # Save feature dataset
    # -------------------------------------------------
    rfm.to_csv(FEATURE_DATA_PATH, index=False)

    print("Feature engineering completed successfully.")
    print("Feature dataset shape:", rfm.shape)

    return rfm