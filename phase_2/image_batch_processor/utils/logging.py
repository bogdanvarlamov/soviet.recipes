"""Logging utilities with configurable log levels."""

import logging
import sys
from typing import Optional


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
