"""
House Price Prediction - Baseline Training Script
Student ID: 24L-8012
"""

import os
import pandas as pd

STUDENT_ID = "24L-8012"

DATA_PATH = os.path.join("data", "house_prices.csv")
SCALING_METHOD = "normalization"


def load_data():
    """Load the house price dataset."""
    print(f"Student ID: {STUDENT_ID}")
    print(f"Loading dataset from: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")

    return data


if __name__ == "__main__":
    load_data()