# -----------------------------------------------------
# Data Cleaning Module
# Cleans raw retail dataset
# -----------------------------------------------------

import pandas as pd
from src.config import SILVER_DATA_PATH


def clean_data(df):
    """
    Cleans raw retail data:
    - Fix column encoding issues
    - Remove null customers
    - Remove negative quantities
    - Create TotalPrice column
    """

    # 🔥 Fix BOM / hidden characters in column names
    df.columns = df.columns.str.encode('utf-8').str.decode('utf-8-sig')

    # Strip spaces
    df.columns = df.columns.str.strip()

    print("Columns after fixing encoding:", df.columns)

    # Remove rows without CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove negative quantities (returns)
    df = df[df["Quantity"] > 0]

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Create TotalPrice column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Save cleaned dataset
    df.to_csv(SILVER_DATA_PATH, index=False)

    print("Cleaned data saved.")

    return df