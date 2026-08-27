"""Sinus3DCNN: the lightweight 3D CNN used for ECRS endotype classification."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Sinus3DCNN(nn.Module):
    """
    Lightweight 3D CNN for sinus CT endotype classification.
    
    Architecture:
        Conv3D(in, 32) -> BN -> ReLU -> MaxPool
        Conv3D(32, 64) -> BN -> ReLU -> MaxPool
        Conv3D(64, 128) -> BN -> ReLU -> MaxPool
        Conv3D(128, 256) -> BN -> ReLU -> AdaptiveAvgPool
        Dropout -> FC(256, num_classes)
    
    Args:
        in_channels: Number of input channels (default: 1)
        num_classes: Number of output classes (default: 2)
        dropout: Dropout rate before final FC (default: 0.3)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: in -> 32
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 2: 32 -> 64
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 3: 64 -> 128
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Block 4: 128 -> 256
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling (handles any input size)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Log model info
        n_params = sum(p.numel() for p in self.parameters())
        logger.info("Created Sinus3DCNN")
        logger.info(f"  - Input: {in_channels} channels, {num_classes} classes")
        logger.info(f"  - Parameters: {n_params:,} total")
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
