import pandas as pd
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
    df_design = build_design_matrix(df_clean, COLUMNS_TO_DROP, ORDINAL_COLUMNS, NOMINAL_COLUMNS)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # run_eda(df_clean, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(
        df_design,
        NUMERIC_COLUMNS,
        TARGET_COLUMN
    )

    print(X_train.head())
    print(X_train.shape)
    print(y_train.head())