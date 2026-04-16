import pandas as pd
from typing import (
    Sequence,
)
from src.logger import get_logger

logger = get_logger(__name__)

# =========================
# BASIC OVERVIEW
# =========================

def print_shape(
        df: pd.DataFrame
) -> None:
    """
    Print the number of rows and columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    None
        This function prints the shape of the DataFrame.
    """
    print("\n=== SHAPE ===")
    rows, columns = df.shape
    print(f"rows: {rows}, columns: {columns}")


def print_column_types(
        df: pd.DataFrame
) -> None:
    """
       Print the types of columns in the DataFrame.

       Parameters
       ----------
       df : pandas.DataFrame
           Input DataFrame.

       Returns
       -------
       None
           This function prints the types of columns in the DataFrame.
       """
    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)


def print_missing_values(
        df: pd.DataFrame
) -> None:
    """
    Print the number of missing values for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    None
        This function prints the number of missing values per column.
    """
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())


# =========================
# TARGET ANALYSIS
# =========================
def print_target_distribution(
    df: pd.DataFrame,
    target_col: str = "cost_of_the_product"
) -> None:
    """
    Print target value descriptive statistics

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str, default="cost_of_the_product"
        Name of the target column.

    Returns
    -------
    None
        This function prints target value descriptive statistics
    """

    print(f"\n=== TARGET DISTRIBUTION: {target_col} ===")
    print("\n*** Descriptive Statistics ***:")
    print(df[target_col].describe())


def print_target_skewness(
        df: pd.DataFrame,
        target_col: str='cost_of_the_product'
) -> None:
    """
    Print target skewness.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str, default="cost_of_the_product"
        Name of the target column.

    Returns
    -------
    None
        This function prints target skewness.
    """
    _skewness(df, target_col)


# =========================
# NUMERICAL FEATURES
# =========================
def print_numerical_summary(
        df: pd.DataFrame,
        numeric_cols: Sequence[str]
) -> None:
    """
        Print descriptive statistics and skewness for numeric columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.

        numeric_cols : Sequence[str]
            Names of numeric columns.

        Returns
        -------
        None
            This function prints descriptive statistics and skewness for the numeric columns.
        """
    print("\n=== NUMERIC COLUMNS SUMMARY ===")
    for col in numeric_cols:
        print(f"\n* {col} *")
        print(df[col].describe())
        _skewness(df, col)


def print_skewness_summary(
        df: pd.DataFrame,
        numeric_cols: Sequence[str]
) -> None:
    """
    Print skewness for all numeric columns sorted by absolute value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : Sequence[str]
        List of numeric column names.

    Returns
    -------
    None
        This function prints skewness values for all numeric columns.
    """
    print("\n=== SKEWNESS SUMMARY ===\n")

    def get_label(skew_val):
        if abs(skew_val) < 0.5:
            return "symmetric"
        elif abs(skew_val) < 1.0:
            return "moderate"
        else:
            return "high"

    skewness_dict = {}
    for col in numeric_cols:
        skew_val = _compute_skewness(df[col])
        skewness_dict[col] = {
            "skewness": skew_val,
            "label": get_label(skew_val)
        }

    skew_df = pd.DataFrame(skewness_dict).T.sort_values(by="skewness", key=abs, ascending=False)

    print(skew_df.to_string())


def print_correlations_with_target(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    target_col: str = "cost_of_the_product",
) -> None:
    """
    Print Pearson correlation coefficients for each numeric predictor vs target.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_cols : Sequence[str]
        List of numeric column names to correlate against the target.
    target_col : str, default="cost_of_the_product"
        Name of the target column.

    Returns
    -------
    None
        This function prints Pearson correlation coefficients for each numeric predictor vs target.
    """
    print(f"\n=== CORRELATIONS WITH TARGET: {target_col} ===")

    correlations = pd.Series(
        {col: _pearson_r(df[col], df[target_col]) for col in numeric_cols}
    ).sort_values(key=abs, ascending=False)

    print(f"\n{'Predictor':<35} {'Pearson r':>10}")
    print("-" * 47)

    for col, r in correlations.items():
        print(f"{col:<35} {r:>10.4f}")


# =========================
# CATEGORICAL FEATURES
# =========================
def print_categorical_summary(
        df: pd.DataFrame,
        categorical_col: Sequence[str],
        max_display: int = 10
) -> None:
    """
    Print descriptive statistics and skewness for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    categorical_col : Sequence[str]
        Names of numeric columns.
    max_display : int, default=10
        Maximum number of unique values to display for a column.
    Returns
    -------
    None
        This function prints descriptive statistics for the categorical columns.
    """
    print("\n=== CATEGORICAL COLUMNS SUMMARY ===")
    for col in categorical_col:
        print(f"\nColumn: {col}")
        print(df[col].describe())

        unique_values = df[col].dropna().unique()
        num_unique = len(unique_values)

        if num_unique > max_display:
            values_to_print = "Too many unique values to display."
        else:
            values_to_print = sorted(unique_values, key=str)


        print(f"Values: {values_to_print}\n")


def print_group_means_by_category(
        df: pd.DataFrame,
        categorical_cols: Sequence[str],
        target_col: str
) -> None:
    """
     Print mean target value for each category in each categorical column.

     Parameters
     ----------
     df : pd.DataFrame
         Input DataFrame.
     categorical_cols : Sequence[str]
         List of categorical column names to group by.
     target_col : str
         Name of the target column.

     Returns
     -------
     None
         This function prints group means for each categorical column.
     """
    print(f"\n=== GROUP MEANS BY CATEGORY: {target_col} ===")

    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print("-" * 40)

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            logger.warning(f"Target '{target_col}' is not numeric — skipping group means.")
            return

        group_means = df.groupby(col)[target_col].mean().sort_values(ascending=False)

        for category, mean_val in group_means.items():
            print(f"  {str(category):<25} {mean_val:>10.2f}")


def print_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: Sequence[str]
) -> None:
    """
        Print Pearson correlation matrix for all numeric column pairs.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        numeric_cols : Sequence[str]
            List of numeric column names.

        Returns
        -------
        None
            This function prints the full correlation matrix.
        """
    print("\n=== CORRELATION MATRIX ===\n")
    if not numeric_cols:
        logger.warning("No numeric columns provided — skipping correlation matrix.")
        return
    matrix = {}
    for row_col in numeric_cols:
        matrix[row_col] = {}
        for col in numeric_cols:
            matrix[row_col][col] = _pearson_r(df[row_col], df[col])

    corr_df = pd.DataFrame(matrix).T

    print(corr_df.round(4))


# =========================
# INTERNAL HELPERS
# =========================
def _compute_skewness(x: pd.Series) -> float:
    """
    Compute skewness manually without pandas .skew().

    Parameters
    ----------
    x : pd.Series
        Numeric series.

    Returns
    -------
    float
        Skewness value.
    """
    n = len(x)
    mean = x.mean()
    std = x.std()

    if std == 0:
        return float('nan')

    cubed_devs = ((x - mean) ** 3).sum()

    return cubed_devs / (n * std ** 3)


def _skewness(df: pd.DataFrame, col: str) -> None:
    """
    Calculate, describe and print numerical column skewness.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Name of the column.

    Returns
    -------
    None
        This function calculates, describes and prints numerical column skewness.
    """
    skew_val = _compute_skewness(df[col])

    if skew_val is None:
        print(f"\n[!] '{col}' is not numeric — skipping skewness.")
        return

    print(f"\nSkewness: {skew_val:.4f}")

    if abs(skew_val) < 0.5:
        skew_label = "approximately symmetric"
    elif abs(skew_val) < 1.0:
        skew_label = "moderately skewed"
    else:
        skew_label = "highly skewed"

    direction = "right (positive)" if skew_val > 0 else "left (negative)"
    print(f"→ Distribution is {skew_label}, {direction}")


def _pearson_r(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Pearson correlation coefficient between two Series.

    Parameters
    ----------
    x : pd.Series
        First numeric variable (predictor).
    y : pd.Series
        Second numeric variable (target).

    Returns
    -------
    float
        Pearson r in the range [-1, 1].
    """
    if len(x) != len(y):
        logger.error(f"Series have different lengths: {len(x)} vs {len(y)}")
        return float('nan')

    x_mean = x.mean()
    y_mean = y.mean()

    # How much each value deviates from its mean
    x_dev = x - x_mean
    y_dev = y - y_mean

    # Numerator: how much x and y move together
    numerator = (x_dev * y_dev).sum()

    # Denominator: how much each variable varies on its own
    x_std = (x_dev ** 2).sum() ** 0.5
    y_std = (y_dev ** 2).sum() ** 0.5
    if x_std == 0 or y_std == 0:
        logger.warning(f"Zero variance detected — cannot compute Pearson r.")
        return float('nan')

    return numerator / (x_std * y_std)


def run_eda(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str
) -> None:
    logger.info("Starting full EDA...")

    print("\n*************** EDA ***************")

    print("######### BASIC OVERVIEW #########")
    print_shape(df)
    print_column_types(df)
    print_missing_values(df)
    logger.info("Basic overview complete.")

    print("\n\n######### TARGET ANALYSIS #########")
    print_target_distribution(df, target_col)
    print_target_skewness(df, target_col)
    logger.info("Target analysis complete.")

    print("\n\n######### NUMERICAL FEATURES #########")
    print_numerical_summary(df, numeric_cols)
    print_correlations_with_target(df, numeric_cols)
    print_skewness_summary(df, numeric_cols)
    logger.info("Numerical features complete.")

    print("\n\n######### CATEGORICAL FEATURES #########")
    print_categorical_summary(df, categorical_cols)
    print_group_means_by_category(df, categorical_cols, target_col)
    logger.info("Categorical features complete.")

    print("\n\n######### MULTICOLLINEARITY #########")
    print_correlation_matrix(df, numeric_cols)
    logger.info("Multicollinearity analysis complete.")



    logger.info("Full EDA completed.")