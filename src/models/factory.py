"""Factory functions for creating classification models."""

import logging
from typing import Dict

import torch.nn as nn

from src.models.sinus3dcnn import Sinus3DCNN
from src.models.resnet import ResNet3D
from src.models.densenet import DenseNet3D
from src.models.efficientnet import EfficientNet3D

logger = logging.getLogger(__name__)


def get_model(config: Dict) -> nn.Module:
    """
    Factory function to create model based on config.

    Args:
        config: Full configuration (must include 'classification' and 'model'
            sections)

    Returns:
        PyTorch model

    Note:
        num_classes is read from config['classification']['num_classes']
        to ensure single source of truth for task definition.

    Example:
        >>> config = {
        ...     'classification': {'num_classes': 2},
        ...     'model': {
        ...         'name': 'sinus3dcnn',
        ...         'in_channels': 3,
        ...         'dropout': 0.3,
        ...     }
        ... }
        >>> model = get_model(config)
    """
    if 'model' not in config:
        raise KeyError(
            f"Cannot find 'model' in config. "
            f"Available keys: {list(config.keys())}. "
            f"Ensure model config is under 'model' key."
        )

    model_config = config['model']
    model_name = model_config['name']
    logger.info(f"Creating model: {model_name}")

    num_classes = config['classification']['num_classes']
    logger.info(f"Using num_classes={num_classes} from classification config")

    if model_name == "sinus3dcnn":
        model = Sinus3DCNN(
            in_channels=model_config['in_channels'],
            num_classes=num_classes,
            dropout=model_config['dropout'],
        )

    elif model_name == "resnet3d":
        model = ResNet3D(
            in_channels=model_config['in_channels'],
            num_classes=num_classes,
            spatial_dims=model_config['spatial_dims'],
            architecture=model_config['architecture'],
            dropout=model_config['dropout'],
        )

    elif model_name == "densenet3d":
        model = DenseNet3D(
            in_channels=model_config['in_channels'],
            num_classes=num_classes,
            spatial_dims=model_config['spatial_dims'],
            dropout=model_config['dropout'],
        )

    elif model_name == "efficientnet3d":
        model = EfficientNet3D(
            in_channels=model_config['in_channels'],
            num_classes=num_classes,
            spatial_dims=model_config['spatial_dims'],
            architecture=model_config['architecture'],
            dropout=model_config['dropout'],
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Supported: sinus3dcnn, resnet3d, "
            f"densenet3d, efficientnet3d."
        )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    return model
