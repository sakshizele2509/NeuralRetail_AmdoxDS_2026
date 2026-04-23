# -----------------------------------------------------
# Main Pipeline Runner
# Executes full Week-1 workflow
# -----------------------------------------------------

from src.data.ingest import load_raw_data
from src.data.clean import clean_data
from src.features.build_features import build_features
from src.models.baseline_churn import run_churn_model
from src.config import FEATURE_DATA_PATH


def run_pipeline():

    df = load_raw_data()

    clean_df = clean_data(df)

    # 🔎 DEBUG: Check columns after cleaning
    print("Columns after cleaning:", clean_df.columns)

    features = build_features(clean_df)

    run_churn_model(FEATURE_DATA_PATH)


if __name__ == "__main__":
    run_pipeline()