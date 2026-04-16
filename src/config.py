from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# ── Data paths
DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"

# ── Output paths
FIGURES_DIR: Path = ROOT_DIR / "outputs" / "figures"
REPORTS_DIR: Path = ROOT_DIR / "outputs" / "reports"

# ── Logs file path
LOGS_DIR: Path = ROOT_DIR / "outputs" / "logs"

# ── Raw data file
RAW_DATA_FILE: Path = DATA_RAW_DIR / "e_commerce_shipping_data.csv"

# ── Log level
LOG_LEVEL = "DEBUG"

# Columns
TARGET_COLUMN = 'cost_of_the_product'
CATEGORICAL_COLUMNS = ['warehouse_block', 'mode_of_shipment', 'gender', 'reached_on_time','product_importance']
NUMERIC_COLUMNS = ['prior_purchases', 'discount_offered', 'weight_in_gms', 'customer_care_calls', 'customer_rating']

ORDINAL_COLUMNS = ['product_importance']
NOMINAL_COLUMNS = ['warehouse_block', 'mode_of_shipment', 'gender']

