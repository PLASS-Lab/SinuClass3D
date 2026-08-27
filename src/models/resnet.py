"""3D ResNet for medical image classification."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResNet3D(nn.Module):
    """3D ResNet for medical image classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        spatial_dims: int = 3,
        architecture: str = "resnet18",
        dropout: float = 0.3,
    ):
        super().__init__()
        from monai.networks.nets import resnet18, resnet34, resnet50

        # Map architecture names to MONAI functions
        arch_map = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
        }

        if architecture not in arch_map:
            raise ValueError(
                f"Unknown architecture: {architecture}. Supported: {list(arch_map.keys())}"
            )

        # Create ResNet using MONAI's convenience function
        self.model = arch_map[architecture](
            pretrained=False,
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=num_classes,
        )

        # Add dropout before final FC
        if dropout > 0:
            # Replace final FC with dropout + FC
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Dropout(p=dropout),
                                          nn.Linear(in_features, num_classes))
            logger.info(f"Added dropout ({dropout}) before final FC layer")

        # Log model info
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Created {architecture}")
        logger.info(f"  - Input: {in_channels} channels, {num_classes} classes")
        logger.info(
            f"  - Parameters: {n_params:,} total, {n_trainable:,} trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
