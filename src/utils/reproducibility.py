"""
Reproducibility utilities for deterministic training.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    This function configures all supported random number generators to use a fixed
    seed, allowing reproducible results. Optionally enforces deterministic behavior
    in PyTorch operations.
    
    Args:
        seed: Random seed value (must be non-negative)
        deterministic: Whether to enable deterministic operations in PyTorch.
                      If False, allows cudnn.benchmark for faster training.
    
    Raises:
        ValueError: If seed is negative
    """
    # Validate input
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got: {seed}")

    logger.info(
        f"Setting random seed to {seed} (deterministic={deterministic})")

    # Python random
    logger.debug("Setting Python random seed to %d", seed)
    random.seed(seed)

    # Numpy
    logger.debug("Setting NumPy random seed to %d", seed)
    np.random.seed(seed)

    # PyTorch CPU
    logger.debug("Setting PyTorch random seed to %d", seed)
    torch.manual_seed(seed)

    # PyTorch CUDA (only if available)
    if torch.cuda.is_available():
        logger.debug("Setting PyTorch CUDA random seed to %d", seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic behavior
    if deterministic:
        logger.debug("Enabling deterministic CuDNN behavior")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.debug("Enabling CuDNN benchmark (faster but non-deterministic)")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Environment variable for Python hash seed
    logger.debug("Setting PYTHONHASHSEED to %s", seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info("Random seed configured successfully")
