"""
House Price Prediction Training Pipeline
Student ID: 24L-8012
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


STUDENT_ID = "24L-8012"

DATA_PATH = os.path.join("data", "house_prices.csv")
MODEL_PATH = os.path.join("model", "house_price_model_24L-8012.pkl")

TARGET_COLUMN = "price"


def main():
    print("=" * 60)
    print("House Price Prediction Training Pipeline")
    print(f"Student ID: {STUDENT_ID}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Identify column types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Numeric preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Create output directory if needed
    os.makedirs("model", exist_ok=True)

    # Save trained model
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nModel saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    main()