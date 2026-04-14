import logging
from src.config import LOGS_DIR
from src.config import LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    """Create and return a configured logger that writes to file and console."""

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL))

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    log_file = LOGS_DIR / "project.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger