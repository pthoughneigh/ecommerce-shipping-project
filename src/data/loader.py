import pandas as pd
from src.config import RAW_DATA_FILE
from src.logger import get_logger

logger = get_logger(__name__)

def load_raw_data(
) -> pd.DataFrame:
    """Load the raw dataset from the configured CSV path.

       Returns:
           pd.DataFrame: The raw dataset with all original columns intact.

       Raises:
           FileNotFoundError: If RAW_DATA_FILE does not exist.
       """

    if not RAW_DATA_FILE.exists():
        logger.error(f"Raw data file not found: {RAW_DATA_FILE}")
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_FILE}")

    df = pd.read_csv(str(RAW_DATA_FILE))
    logger.info(f"Successfully loaded {len(df)} rows from {RAW_DATA_FILE.name}")
    return df