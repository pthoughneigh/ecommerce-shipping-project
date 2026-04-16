import numpy as np
from typing import Sequence, Tuple
import pandas as pd
from src.logger import get_logger
from src.features.preprocessing import fit_numerical_scaler, transform_numerical_columns

logger = get_logger(__name__)

def split_data(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into train and test sets and apply numerical scaling.

    Shuffles the data, splits by test_size ratio, fits a numerical scaler
    on the training set only, and applies it to both sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target.
    numerical_cols : Sequence[str]
        Numerical columns to scale.
    target_col : str
        Name of the target column to predict.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Default 0.2.
    random_state : int, optional
        Seed for reproducibility. Default 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test

    Notes
    -----
    - Does NOT modify the original DataFrame
    - Scaler is fitted on the training set only to prevent data leakage
    """
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}, rows={len(df)}")

    df = df.copy()
    rng = np.random.default_rng(random_state)

    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_test = int(len(df) * test_size)
    test_indices  = indices[:n_test]
    train_indices = indices[n_test:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df  = df.iloc[test_indices].reset_index(drop=True)

    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    means, stds = fit_numerical_scaler(X_train, numerical_cols)
    X_train = transform_numerical_columns(X_train, numerical_cols, means, stds)
    X_test  = transform_numerical_columns(X_test,  numerical_cols, means, stds)

    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    logger.info("Data split and scaling complete.")

    return X_train, X_test, y_train, y_test