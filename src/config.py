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