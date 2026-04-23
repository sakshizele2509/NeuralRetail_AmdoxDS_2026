# -----------------------------------------------------
# Data Ingestion Module
# Loads raw retail dataset
# -----------------------------------------------------

import pandas as pd
from src.config import RAW_DATA_PATH


def load_raw_data():
    """
    Loads raw dataset with correct encoding handling.
    Fixes BOM issue at source.
    """

    # 🔥 IMPORTANT: Use utf-8-sig to remove BOM
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")

    print("Raw Data Loaded:", df.shape)
    print("Columns after loading:", df.columns)

    return df