import pandas as pd
from src.data.loader import load_raw_data
from src.data.cleaning import clean_data

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    print(df_clean.head())
    print(df_clean.dtypes)