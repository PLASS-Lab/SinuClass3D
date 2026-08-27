"""Initialize the utils package.

This package provides various utility functions and helpers, including logging
configuration, environment setup, checkpointing, and miscellaneous tools
shared across the project.
"""

from src.utils.checkpointing import ModelCheckpoint
from src.utils.early_stopping import EarlyStopping
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import set_seed
from src.utils.serialization import convert_to_serializable
from src.utils.wandb_utils import (
    # Core utilities
    log_system_info,
    save_model_artifact,
    finish_run,
    # Results & summary
    log_cv_results_table,
    log_dataset_info,
    log_config_summary,
)

__all__ = [
    'ModelCheckpoint',
    'EarlyStopping',
    'setup_logging',
    'set_seed',
    'convert_to_serializable',
    # WandB utilities
    'log_system_info',
    'save_model_artifact',
    'finish_run',
    'log_cv_results_table',
    'log_dataset_info',
    'log_config_summary',
]
