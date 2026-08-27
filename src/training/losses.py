"""
Loss Functions for Imbalanced Medical Image Classification

This module implements standard loss functions used in medical imaging classification:
- WeightedCrossEntropyLoss: Standard approach for imbalanced data
- FocalLoss: Alternative for severe imbalance (Lin et al. 2017)

Both losses support automatic class weight computation from training data.

References:
- Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
- King & Zeng "Logistic Regression in Rare Events Data" 2001
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for class imbalance.
    Computes class weights from training data.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            weight: Class weights tensor [num_classes]
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] unnormalized scores
            targets: [B] class indices
            
        Returns:
            loss: Scalar loss value
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
        https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Class weights [num_classes] or None
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] unnormalized scores
            targets: [B] class indices
            
        Returns:
            loss: Focal loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Get probability of true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # [B]

        # Compute focal weight
        focal_weight = (1 - pt)**self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss(config: Dict,
             class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to create loss function based on config.
    
    Args:
        config: Loss configuration from YAML
        class_weights: Pre-computed class weights [num_classes]
        
    Returns:
        Loss module
    """
    # Get loss config from training hierarchy (REQUIRED)
    if 'training' not in config or 'loss' not in config['training']:
        raise KeyError(
            f"Cannot find 'training.loss' in config. "
            f"Available keys: {list(config.keys())}. "
            f"Ensure loss config is under 'training.loss' hierarchy.")

    loss_config = config['training']['loss']

    # Required field - direct access (fail fast if missing)
    loss_name = loss_config['name']
    logger.info(f"Creating loss function: {loss_name}")

    if loss_name == "weighted_ce":
        # Weighted Cross-Entropy
        cfg = loss_config['weighted_ce']

        # Use provided weights or compute from config
        if class_weights is not None:
            weight = class_weights
        elif cfg['class_weights'] is not None:
            weight = torch.FloatTensor(cfg['class_weights'])
        else:
            weight = None
            logger.warning("No class weights provided for weighted CE!")

        loss_fn = WeightedCrossEntropyLoss(
            weight=weight,
            label_smoothing=cfg['label_smoothing'],
        )

        logger.info(f"Using weighted CE with weights: {weight}")

    elif loss_name == "focal":
        # Focal Loss
        cfg = loss_config['focal']

        alpha = class_weights if class_weights is not None else None
        if cfg['alpha'] is not None:
            alpha = torch.FloatTensor(cfg['alpha'])

        loss_fn = FocalLoss(
            alpha=alpha,
            gamma=cfg['gamma'],
            reduction=cfg['reduction'],
        )

        logger.info(
            f"Using Focal Loss with gamma={cfg['gamma']}, alpha={alpha}")

    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                         f"Supported: 'weighted_ce', 'focal'")

    return loss_fn
