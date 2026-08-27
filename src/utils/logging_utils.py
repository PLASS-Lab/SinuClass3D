"""
Logging utilities for training and evaluation.

Follows Python standard library and Google best practices:
- Use module-level loggers with logging.getLogger(__name__)
- Application code configures handlers, not library code
- Standard format with timestamps, module names, and line numbers

Example usage in modules:
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

Example output:
    2024-11-24 10:15:30 INFO     [src.training.trainer:42] Starting training
    2024-11-24 10:15:31 WARNING  [src.data.loading:121] No validation data
    2024-11-24 10:15:32 ERROR    [src.models.resnet:45] CUDA OOM
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Optional[Path] = None,
                  log_file: str = "train.log",
                  level: int = logging.INFO,
                  console: bool = True) -> None:
    """
    Setup logging configuration following Python best practices.
    
    Format:
        TIMESTAMP LEVEL [module:line] message
    
    This should only be called by application entry points (train.py, evaluate.py),
    NOT by library modules.
    
    Args:
        log_dir: Directory to save log files (optional)
        log_file: Log filename (default: train.log)
        level: Logging level (default: INFO)
        console: Whether to log to console (default: True)
    
    Example:
        >>> setup_logging(log_dir=Path("experiments/run1"), level=logging.INFO)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Training started")
        2024-11-24 10:15:30 INFO [__main__:42] Training started
    """
    # Standard format with module name (not filename)
    # %(name)s gives module hierarchy like src.training.trainer
    log_format = '%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger (only acceptable at application level)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
