from typing import Sequence, Dict, Tuple

import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


def drop_unnecessary_columns(
        df: pd.DataFrame,
        columns: Sequence[str],
) -> pd.DataFrame:
    """
    Clean the DataFrame by dropping unnecessary columns

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        columns (Sequence[str]): A list of columns to be dropped.
    Returns:
        pd.DataFrame: A cleaned DataFrame without unnecessary columns.
    """
    df = df.copy()

    cols_to_drop = [col for col in columns if col in df.columns]
    missing = set(columns) - set(cols_to_drop)

    for col in missing:
        logger.warning(f"Column '{col}' not found in DataFrame, skipping.")

    return df.drop(columns=cols_to_drop)


def encode_ordinal_columns(
    df: pd.DataFrame,
    ordinal_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Encode ordinal categorical columns into numeric values.

    Ordinal features have a natural order, so they are mapped to integers
    while preserving category order.

    Missing values are replaced with "unknown", which is mapped to -1.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    ordinal_cols : Sequence[str]
        List of ordinal column names to encode.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with encoded ordinal columns.

    Notes
    -----
    - Encoding preserves ordering relationships between categories
    - "unknown" values are explicitly handled and mapped to -1
    - Each column has its own mapping dictionary
    - Does NOT modify the original DataFrame
    """
    df = df.copy()

    ordinal_mappings: Dict[str, Dict[str, int]] = {
        "product_importance": {
            "unknown": -1,
            "low": 0,
            "medium": 1,
            "high": 2
        }
    }

    for col in ordinal_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame.")
            continue

        if col not in ordinal_mappings:
            logger.warning(f"No ordinal mapping defined for column '{col}'.")
            continue

        df[col] = df[col].fillna("unknown")
        df[col] = df[col].map(ordinal_mappings[col])

        unexpected = df[col].isna()
        if unexpected.any():
            logger.warning(
                f"Column '{col}' contains unmapped values after encoding."
            )
            df[col] = df[col].fillna(-1)

        df[col] = df[col].astype(int)

    return df


def encode_nominal_columns(
    df: pd.DataFrame,
    nominal_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Encode nominal categorical columns using one-hot encoding.

    Each category is converted into a binary indicator column.
    One category per feature is dropped to avoid multicollinearity
    in linear models.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    nominal_cols : Sequence[str]
        List of nominal column names to encode.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with one-hot encoded columns.

    Notes
    -----
    - Uses pd.get_dummies with drop_first=True
    - Avoids dummy-variable redundancy for linear models
    - Creates an implicit baseline category
    - Does NOT modify the original DataFrame
    """
    df = df.copy()

    missing = [col for col in nominal_cols if col not in df.columns]
    for col in missing:
        logger.warning(f"Column '{col}' not found in DataFrame, skipping.")

    nominal_cols = [col for col in nominal_cols if col in df.columns]

    df = pd.get_dummies(df, columns=list(nominal_cols), drop_first=True, dtype=int)

    return df


def fit_numerical_scaler(
    df: pd.DataFrame,
    numerical_cols: Sequence[str]
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute scaling parameters for numerical columns.
    Standardization uses mean and standard deviation.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame used to estimate scaling parameters.
    numerical_cols : Sequence[str]
        Numerical columns to scale.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        means : pd.Series
            Mean values of numerical training columns.
        stds : pd.Series
            Standard deviations of numerical training columns, with near-zero
            values protected against division by zero.

    Notes
    -----
    This function should be fitted on the training set only.
    """
    logger.info(f"Fitting numerical scaler on columns: {list(numerical_cols)}")

    numerical_cols = [col for col in numerical_cols if col in df.columns]

    means = df[list(numerical_cols)].mean()
    stds = df[list(numerical_cols)].std()
    stds = stds.mask(stds < 1e-12, 1.0)

    logger.info(f"Scaler fitted. Means: {means.to_dict()}, Stds: {stds.to_dict()}")

    return means, stds


def transform_numerical_columns(
    df: pd.DataFrame,
    numerical_cols: Sequence[str],
    means: pd.Series,
    stds: pd.Series
) -> pd.DataFrame:
    """
    Apply numerical scaling using precomputed parameters.

    Each column is transformed as:
        (value - mean) / standard deviation

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame to transform.
    numerical_cols : Sequence[str]
        Numerical columns to scale.
    means : pd.Series
        Means computed on the training set.
    stds : pd.Series
        Standard deviations computed on the training set.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with scaled numerical columns.

    Notes
    -----
    - Does NOT modify the original DataFrame
    - Uses training statistics only
    - Suitable for application to both train and test sets
    """
    df = df.copy()

    logger.info(f"Transforming numerical columns: {list(numerical_cols)}")

    for col in numerical_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame.")
            continue

        df[col] = (df[col] - means[col]) / stds[col]

    return df


def build_design_matrix(
        df: pd.DataFrame,
        unnecessary_cols: Sequence[str],
        ordinal_columns: Sequence[str],
        nominal_cols: Sequence[str],
) -> pd.DataFrame:
    """

    This function performs only transformations that do not require
    fitting statistics from the full dataset:
    1. Drop unnecessary columns.
    2. Encode ordinal features
    3. Encode nominal features

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    unnecessary_cols: Sequence[str]
        List of unnecessary columns.
    ordinal_columns : Sequence[str]
        List of ordinal columns.
    nominal_cols : Sequence[str]
        List of nominal columns.

    Returns
    -------
    pd.DataFrame
        Processed dataset ready for train-test split and later scaling.

    Notes
    -----
    Scaling is intentionally excluded from this function to avoid
    data leakage. Numerical scaling should be fitted on the training
    set only and then applied to both train and test sets.
    """
    logger.info("Building design_matrix...")

    df = df.copy()

    df = (
        df
        .pipe(drop_unnecessary_columns, unnecessary_cols)
        .pipe(encode_ordinal_columns, ordinal_columns)
        .pipe(encode_nominal_columns, nominal_cols)
    )

    logger.info("Design matrix built.")
    return df