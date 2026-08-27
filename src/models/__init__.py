"""3D CNN models for ECRS endotype classification.

The primary architecture reported in the paper is the lightweight three-channel
``Sinus3DCNN``. The higher-capacity architectures used for the capacity
comparison (Table 6) are also included. Use :func:`get_model` to construct any
of them from config.

Usage:
    >>> from src.models import get_model
    >>> config = {'classification': {'num_classes': 2},
    ...           'model': {'name': 'sinus3dcnn', 'in_channels': 3,
    ...                     'dropout': 0.3}}
    >>> model = get_model(config)
"""

from src.models.sinus3dcnn import Sinus3DCNN
from src.models.resnet import ResNet3D
from src.models.densenet import DenseNet3D
from src.models.efficientnet import EfficientNet3D
from src.models.factory import get_model

__all__ = [
    'Sinus3DCNN',
    'ResNet3D',
    'DenseNet3D',
    'EfficientNet3D',
    'get_model',
]
