import pandas as pd
from src.models.linear_regression import LinearRegression
from src.config import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    COLUMNS_TO_DROP,
    ORDINAL_COLUMNS,
    NOMINAL_COLUMNS
)
from src.data.loader import load_raw_data
from src.data.cleaning import clean_data
from src.analysis.eda import run_eda
from src.features.preprocessing import build_design_matrix
from src.features.splitting import split_data

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_processed = build_design_matrix(
        df_clean, COLUMNS_TO_DROP, ORDINAL_COLUMNS, NOMINAL_COLUMNS
    )
    X_train, X_test, y_train, y_test = split_data(
        df_processed, NUMERIC_COLUMNS, TARGET_COLUMN
    )

    model = LinearRegression()
    model.fit(X_train.values, y_train.values)

    predictions = model.predict(X_test.values)
    print(predictions[:10])
    print(f"First 10 actual: {y_test.values[:10]}")
