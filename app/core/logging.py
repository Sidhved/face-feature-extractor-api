# ---------------------------------------------Logging Configuration-------------------------------------------- #

import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logging():
    """Configure logging for the application."""
    logger = logging.getLogger("facial_feature_api")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(levelname)s | %(asctime)s | %(name)s | %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(logs_dir / "app.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(levelname)s | %(asctime)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

# ---------------------------------------------------------------------------------------------------- #