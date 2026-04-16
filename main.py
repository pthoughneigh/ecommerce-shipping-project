import pandas as pd
from src.config import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN
)
from src.data.loader import load_raw_data
from src.data.cleaning import clean_data
from src.analysis.eda import run_eda

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    run_eda(df_clean, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, TARGET_COLUMN)
