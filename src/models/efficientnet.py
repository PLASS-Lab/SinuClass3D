"""3D EfficientNet for medical image classification."""

import logging
import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBN

logger = logging.getLogger(__name__)


class EfficientNet3D(nn.Module):
    """3D EfficientNet for medical image classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        spatial_dims: int = 3,
        architecture: str = "efficientnet-b0",
        dropout: float = 0.3,
    ):
        super().__init__()

        self.model = EfficientNetBN(
            model_name=architecture,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
        )

        logger.info(
            f"Created {architecture} with {in_channels} input channels, {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
