# ---------------------------------------------------------
# EDA Script
# Generates automated HTML profiling report
# ---------------------------------------------------------

import os
import pandas as pd
from ydata_profiling import ProfileReport
from src.config import SILVER_DATA_PATH

def generate_eda_report():

    print("Loading cleaned dataset for EDA...")

    df = pd.read_csv(SILVER_DATA_PATH)

    print("Generating profiling report...")

    profile = ProfileReport(
        df,
        title="NeuralRetail Week-1 EDA Report",
        explorative=True
    )

    # Ensure reports folder exists
    os.makedirs("reports", exist_ok=True)

    profile.to_file("reports/eda_report.html")

    print("EDA report generated successfully!")

if __name__ == "__main__":
    generate_eda_report()