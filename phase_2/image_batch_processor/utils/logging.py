"""Logging utilities with configurable log levels."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with configurable log level.
    
    Args:
        name: Name of the logger
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Only set up if not already configured
    if not logger.handlers:
        return setup_logger(name, level)
    
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger for the application.
    
    Args:
        level: Logging level for the root logger (default: INFO)
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def add_file_logging(
    log_path: Union[str, Path],
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> Path:
    """
    Attach a file handler to the root logger so all log output is also written
    to a file (in addition to the console). Useful for monitoring unattended
    runs via `tail`/`Get-Content -Wait`.

    Args:
        log_path: Path to the log file (parent directories are created).
        level: Logging level for the file handler.
        format_string: Custom format string for log messages.

    Returns:
        The resolved log file path.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            format_string or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    logging.getLogger().addHandler(file_handler)
    return log_path
