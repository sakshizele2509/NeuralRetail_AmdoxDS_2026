# -----------------------------------------------------
# Project Configuration File
# Central place for paths and constants
# -----------------------------------------------------

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "online_retail.csv")
SILVER_DATA_PATH = os.path.join(BASE_DIR, "data", "silver", "retail_clean.csv")
FEATURE_DATA_PATH = os.path.join(BASE_DIR, "data", "features", "retail_features.csv")

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "NeuralRetail_Week1"