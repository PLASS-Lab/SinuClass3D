"""
Classification Training Module

This module contains training utilities for medical image classification.

Components:
- losses: Loss functions (WeightedCrossEntropyLoss, FocalLoss)
- optimizer: Optimizers (AdamW, Adam, SGD)
- scheduler: Learning rate schedulers with warmup support
- trainer: ClassificationTrainer with early stopping and model selection

For usage examples and integration patterns, see the README or individual
component docstrings.
"""

# Loss functions
from src.training.losses import (
    FocalLoss,
    WeightedCrossEntropyLoss,
    get_loss,
)

# Optimizers
from src.training.optimizer import (
    get_adam_optimizer,
    get_adamw_optimizer,
    get_sgd_optimizer,
    get_optimizer,
)

# Schedulers
from src.training.scheduler import (
    get_cosine_annealing_lr_scheduler,
    get_reduce_on_plateau_scheduler,
    get_step_lr_scheduler,
    get_exponential_lr_scheduler,
    get_polynomial_decay_lr_scheduler,
    PolynomialDecayLR,
    WarmupScheduler,
    get_scheduler,
)

# Trainer
from src.training.trainer import ClassificationTrainer

__all__ = [
    # Loss functions
    'FocalLoss',
    'WeightedCrossEntropyLoss',
    'get_loss',
    # Optimizers
    'get_adam_optimizer',
    'get_adamw_optimizer',
    'get_sgd_optimizer',
    'get_optimizer',
    # Schedulers
    'get_cosine_annealing_lr_scheduler',
    'get_reduce_on_plateau_scheduler',
    'get_step_lr_scheduler',
    'get_exponential_lr_scheduler',
    'get_polynomial_decay_lr_scheduler',
    'PolynomialDecayLR',
    'WarmupScheduler',
    'get_scheduler',
    # Trainer
    'ClassificationTrainer',
]
