"""Optimizers for medical image classification training.

This module provides standard optimizers used in medical imaging classification:
- AdamW: Most common in modern medical imaging (MONAI default, ImageNet standard)
- Adam: Alternative to AdamW (no weight decay decoupling)
- SGD with momentum: Classical optimizer (ResNet original training)

AdamW with lr=1e-4 and weight_decay=1e-4 is common for 3D medical classification.
For fine-tuning pretrained models, lower lr (1e-5 to 1e-4) is typical.

References:
- Loshchilov & Hutter "Decoupled Weight Decay Regularization" (ICLR 2019)
  https://arxiv.org/abs/1711.05101
- He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
  Uses SGD with momentum for ImageNet classification
"""

import torch
from torch import optim
from typing import Iterable


def get_adam_optimizer(params: Iterable[torch.nn.Parameter],
                       lr: float = 1e-4,
                       weight_decay: float = 1e-4,
                       betas=(0.9, 0.999),
                       eps: float = 1e-8,
                       amsgrad: bool = False) -> optim.Optimizer:
    """
    Returns an Adam optimizer.
    
    Adam is widely used in medical image classification, though AdamW is
    generally preferred in modern implementations.

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4, standard for medical imaging)
        weight_decay: L2 penalty (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        amsgrad: Whether to use AMSGrad variant (default: False)

    Returns:
        optim.Optimizer: A PyTorch Adam optimizer
    """
    return optim.Adam(params,
                      lr=lr,
                      weight_decay=weight_decay,
                      betas=betas,
                      eps=eps,
                      amsgrad=amsgrad)


def get_adamw_optimizer(params: Iterable[torch.nn.Parameter],
                        lr: float = 1e-4,
                        weight_decay: float = 1e-4,
                        betas=(0.9, 0.999),
                        eps: float = 1e-8,
                        amsgrad: bool = False) -> optim.Optimizer:
    """
    Returns an AdamW optimizer (RECOMMENDED for medical image classification).
    
    AdamW decouples weight decay from the gradient-based update, providing
    better regularization than standard Adam. This is the default optimizer
    in MONAI and most modern medical imaging classification frameworks.

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-4, standard for medical imaging)
        weight_decay: Weight decay factor (default: 1e-4, standard)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        amsgrad: Whether to use AMSGrad variant (default: False)

    Returns:
        optim.Optimizer: A PyTorch AdamW optimizer
    """
    return optim.AdamW(params,
                       lr=lr,
                       weight_decay=weight_decay,
                       betas=betas,
                       eps=eps,
                       amsgrad=amsgrad)


def get_sgd_optimizer(params: Iterable[torch.nn.Parameter],
                      lr: float = 0.001,
                      momentum: float = 0.9,
                      weight_decay: float = 1e-4,
                      nesterov: bool = True) -> optim.Optimizer:
    """
    Returns an SGD optimizer with momentum.
    
    SGD with momentum is the classical optimizer used in the original
    ResNet paper (He et al. 2016) for ImageNet classification. It's still
    competitive for many classification tasks, especially with proper
    learning rate scheduling.

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3, higher than Adam for SGD)
        momentum: Momentum factor (default: 0.9, standard from He et al. 2016)
        weight_decay: L2 regularization (default: 1e-4)
        nesterov: Whether to use Nesterov momentum (default: True, recommended)

    Returns:
        optim.Optimizer: A PyTorch SGD optimizer
    """
    return optim.SGD(params,
                     lr=lr,
                     momentum=momentum,
                     weight_decay=weight_decay,
                     nesterov=nesterov)


def get_optimizer(optimizer_name: str, params: Iterable[torch.nn.Parameter],
                  **kwargs) -> optim.Optimizer:
    """
    Factory function to retrieve an optimizer by name.
    
    Standard optimizers for medical image classification:
    - adamw: AdamW (RECOMMENDED - MONAI default, best for 3D medical imaging)
    - adam: Adam (alternative to AdamW)
    - sgd: SGD with momentum (classical baseline, ResNet original)

    Args:
        optimizer_name: The name of the optimizer ('adamw', 'adam', 'sgd')
        params: Model parameters
        **kwargs: Additional keyword arguments for the optimizer
                 For AdamW/Adam: lr, weight_decay, betas, eps, amsgrad
                 For SGD: lr, momentum, weight_decay, nesterov

    Returns:
        optim.Optimizer: The requested optimizer initialized with provided params and kwargs

    Raises:
        ValueError: If the given optimizer_name is not supported
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        return get_adam_optimizer(params, **kwargs)
    elif optimizer_name == 'adamw':
        return get_adamw_optimizer(params, **kwargs)
    elif optimizer_name == 'sgd':
        return get_sgd_optimizer(params, **kwargs)
    else:
        raise ValueError(
            f"Unsupported optimizer name: {optimizer_name}. "
            f"Choose from 'adamw' (recommended), 'adam', or 'sgd'.")
