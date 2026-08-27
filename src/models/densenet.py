"""3D DenseNet for medical image classification."""

import logging
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

logger = logging.getLogger(__name__)


class DenseNet3D(nn.Module):
    """3D DenseNet for medical image classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        spatial_dims: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
            dropout_prob=dropout,
        )

        logger.info(
            f"Created DenseNet121 with {in_channels} input channels, {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
