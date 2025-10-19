"""
Centralized Logging Utility for Image Alignment Pipeline
Provides unified logging to both console and log.txt file.

All scripts in the pipeline use this module to ensure consistent logging
with timestamps, script names, and persistent file output.

Usage:
    from logging_utils import setup_logger

    logger = setup_logger('ScriptName')
    logger.info("This goes to both console and log.txt")
    logger.error("Errors are also captured")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import threading

# Global lock for thread-safe file writing
_file_lock = threading.Lock()

# Track if log file has been initialized
_log_initialized = False


def setup_logger(script_name: str, log_dir: str = '../scripts_output',
                 log_filename: str = 'log.txt') -> logging.Logger:
    """
    Setup a logger that writes to both console and log file.

    Args:
        script_name: Name of the script (e.g., 'Step1_ParticleSelection')
        log_dir: Directory where log.txt will be saved (default: '../scripts_output')
        log_filename: Name of log file (default: 'log.txt')

    Returns:
        logging.Logger instance configured for dual output
    """
    global _log_initialized

    # Create logger with script name
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / log_filename

    # Write separator on first initialization
    if not _log_initialized:
        with _file_lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"NEW PIPELINE RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
        _log_initialized = True

    # Custom formatter with timestamp and script name
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler (append mode)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler (same format as file)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_to_file_only(message: str, log_dir: str = '../scripts_output',
                      log_filename: str = 'log.txt'):
    """
    Write a message directly to log file without console output.
    Useful for worker processes that need to log without interfering with progress bars.

    Args:
        message: Message to write to log file
        log_dir: Directory where log.txt is located
        log_filename: Name of log file
    """
    log_path = Path(log_dir)
    log_file = log_path / log_filename

    timestamp = datetime.now().strftime('%H:%M:%S')

    with _file_lock:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance from setup_logger()
        exception: Exception object to log
        context: Optional context string (e.g., "Error processing image X")
    """
    import traceback

    if context:
        logger.error(f"{context}: {type(exception).__name__}: {exception}")
    else:
        logger.error(f"{type(exception).__name__}: {exception}")

    # Log full traceback
    tb_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    logger.debug(f"Traceback:\n{tb_str}")


def log_worker_message(message: str, log_dir: str = '../scripts_output',
                        log_filename: str = 'log.txt', also_print: bool = True):
    """
    Log a message from a worker process.
    Writes to both console (with flush) and log file.

    Args:
        message: Message to log
        log_dir: Directory where log.txt is located
        log_filename: Name of log file
        also_print: If True, also print to console (default: True)
    """
    if also_print:
        print(message, flush=True)

    log_to_file_only(message, log_dir, log_filename)
