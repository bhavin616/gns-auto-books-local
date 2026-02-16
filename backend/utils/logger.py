"""
Centralized logging module for the application.

Provides a configurable logger that writes to both a rotating log file and
the console. Includes a convenience function to log messages at different
levels.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
LOG_FILE = "bank_rec.log"
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 5                      # Keep last 5 log files


def setup_logger() -> logging.Logger:
    """
    Sets up the application logger.

    Creates a logger that logs messages to a rotating file and the console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("benk_rec_logger")
    logger.setLevel(logging.INFO)  # Minimum log level

    # Avoid adding multiple handlers if setup_logger is called multiple times
    if not logger.handlers:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(LOG_FILE) or "."
            os.makedirs(log_dir, exist_ok=True)

            # Rotating file handler
            file_handler = RotatingFileHandler(
                LOG_FILE, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT
            )
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(file_formatter)

            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            logger.info("Logger initialized. Logging to file '%s' and console.", LOG_FILE)

        except Exception as e:
            # Fallback to basicConfig in case of handler setup failure
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger("benk_rec_logger")
            logger.warning("Logger setup failed: %s. Using basicConfig instead.", e)

    return logger


# Initialize logger
logger = setup_logger()


def log_message(level: str, message: str):
    """
    Logs a message at the specified level.

    Args:
        level (str): Logging level ('info', 'warning', 'error', 'critical', 'debug').
        message (str): The message to log.
    """
    level = level.lower()
    try:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "critical":
            logger.critical(message)
        elif level == "debug":
            logger.debug(message)
        else:
            logger.info(message)
    except Exception as e:
        # Avoid crashing if logging fails
        print(f"Logging failed: {e}. Original message: {message}")
