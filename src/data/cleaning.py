import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

def clean_data(
        df: pd.DataFrame
) -> pd.DataFrame:
    """Clean the raw DataFrame by standardizing column names, dropping unnecessary
    columns, and removing duplicate rows.

    Args:
        df (pd.DataFrame): The raw input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: A cleaned DataFrame with standardized column names,
        no 'id' column, and no duplicate rows.
    """

    df = df.copy()

    logger.info(f"Starting cleaning. Initial shape: {df.shape}")

    df = df.rename(columns=lambda x: x.lower().strip().replace(".", "_").replace("_y_n", ""))
    logger.info("Renamed columns...")
    logger.info(f"Columns after renaming: {df.columns.tolist()}")

    logger.info("Removing unnecessary columns...")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        logger.info("Dropped 'id' column...")

    num_duplicates = df.duplicated().sum()
    if  num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} duplicates, removing....")
        df = df.drop_duplicates(keep="first")
        logger.info("Duplicates removed.")
    else:
        logger.info("No duplicates found.")

    logger.info(f"Final columns: {df.columns.tolist()}")
    logger.info(f"Cleaning completed. Final shape: {df.shape}")

    return df