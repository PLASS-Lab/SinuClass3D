"""
Learning rate schedulers for medical image classification training.

This module defines standard learning rate schedulers used in medical imaging:
- CosineAnnealingLR (common for classification, ImageNet standard)
- ReduceLROnPlateau (validation metric-based, practical for medical imaging)
- StepLR (classical step decay, ResNet original)
- PolynomialDecayLR (alternative smooth decay)
- LR Warmup (stabilizes early training)

For classification, CosineAnnealingLR is very popular (used in modern vision
transformers and ResNets). ReduceLROnPlateau is practical for medical imaging
where validation metrics are critical.

References:
- Loshchilov & Hutter "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017)
  https://arxiv.org/abs/1608.03983
- He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
  Uses step decay for ImageNet classification
- Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
  Introduces linear warmup for large batch training
"""

import logging
import torch
from typing import Literal
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR, ExponentialLR)

logger = logging.getLogger(__name__)


def get_cosine_annealing_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        t_max: int,
        eta_min: float = 0.0) -> CosineAnnealingLR:
    """
    CosineAnnealingLR follows a cosine curve from initial LR down to eta_min.
    
    This is very popular for classification tasks (ImageNet, medical imaging).
    Provides smooth decay and often better than step decay for deep networks.

    Args:
        optimizer: The optimizer
        t_max: Maximum number of iterations (usually number of epochs)
        eta_min: Minimum learning rate (default: 0.0)

    Returns:
        CosineAnnealingLR: A PyTorch CosineAnnealingLR scheduler
        
    Examples:
        >>> # Train for 100 epochs with cosine annealing
        >>> scheduler = get_cosine_annealing_lr_scheduler(optimizer, t_max=100, eta_min=1e-6)
    """
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def get_reduce_on_plateau_scheduler(optimizer: torch.optim.Optimizer,
                                    mode: Literal['min', 'max'] = 'min',
                                    factor: float = 0.1,
                                    patience: int = 10,
                                    threshold: float = 0.0001,
                                    min_lr: float = 0.0) -> ReduceLROnPlateau:
    """
    ReduceLROnPlateau reduces LR when a metric stops improving.
    
    Very practical for medical imaging classification where validation
    performance is the key metric. Automatically adapts to training dynamics.

    Args:
        optimizer: The optimizer
        mode: 'min' or 'max'. In 'min' mode, LR reduces when metric stops decreasing
        factor: Factor by which the LR will be reduced (new_lr = lr * factor)
        patience: Number of epochs with no improvement after which LR will be reduced
        threshold: Threshold for measuring improvement
        min_lr: Minimum learning rate

    Returns:
        ReduceLROnPlateau: A PyTorch ReduceLROnPlateau scheduler
        
    Examples:
        >>> # Reduce LR when validation loss plateaus
        >>> scheduler = get_reduce_on_plateau_scheduler(optimizer, mode='min', patience=10)
        >>> 
        >>> # In training loop:
        >>> my_val_loss = 0.05  # from your validation
        >>> scheduler.step(my_val_loss)
    """
    return ReduceLROnPlateau(optimizer,
                             mode=mode,
                             factor=factor,
                             patience=patience,
                             threshold=threshold,
                             min_lr=min_lr)


def get_step_lr_scheduler(optimizer: torch.optim.Optimizer,
                          step_size: int = 30,
                          gamma: float = 0.1) -> StepLR:
    """
    StepLR decays the learning rate by gamma every step_size epochs.
    
    Classical scheduler used in original ResNet training (He et al. 2016).
    Simple and interpretable, but step decay can be abrupt.

    Args:
        optimizer: The optimizer
        step_size: Period of learning rate decay (epochs)
        gamma: Multiplicative factor of learning rate decay

    Returns:
        StepLR: A PyTorch StepLR scheduler
        
    Examples:
        >>> # Reduce LR by 10x every 30 epochs (ResNet-style)
        >>> scheduler = get_step_lr_scheduler(optimizer, step_size=30, gamma=0.1)
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def get_exponential_lr_scheduler(optimizer: torch.optim.Optimizer,
                                 gamma: float = 0.95) -> ExponentialLR:
    """
    ExponentialLR decays the learning rate exponentially every epoch.
    
    Smooth decay: lr = lr * gamma each epoch

    Args:
        optimizer: The optimizer
        gamma: Multiplicative factor of learning rate decay

    Returns:
        ExponentialLR: A PyTorch ExponentialLR scheduler
        
    Examples:
        >>> # Decay by 5% each epoch
        >>> scheduler = get_exponential_lr_scheduler(optimizer, gamma=0.95)
    """
    return ExponentialLR(optimizer, gamma=gamma)


# noinspection PyProtectedMember
class PolynomialDecayLR(torch.optim.lr_scheduler._LRScheduler):  # noqa: SLF001
    # Note: Accessing _LRScheduler is the official way to subclass PyTorch schedulers
    """
    Polynomial decay of learning rate:
    lr = initial_lr * (1 - epoch/max_epoch)^power
    
    Alternative to cosine annealing, provides smooth decay.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 max_epochs: int,
                 power: float = 0.9,
                 last_epoch: int = -1):
        """
        Initialize polynomial decay scheduler.
        
        Args:
            optimizer: The optimizer
            max_epochs: Total number of training epochs
            power: Power factor for polynomial decay (default: 0.9)
            last_epoch: The index of last epoch (default: -1)
        """
        self.max_epochs = max_epochs
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (1 - self.last_epoch / self.max_epochs)**self.power
            for base_lr in self.base_lrs
        ]


def get_polynomial_decay_lr_scheduler(optimizer: torch.optim.Optimizer,
                                      max_epochs: int,
                                      power: float = 0.9) -> PolynomialDecayLR:
    """
    Returns a PolynomialDecayLR scheduler.

    Args:
        optimizer: The optimizer
        max_epochs: The total number of epochs to train
        power: Power factor for polynomial decay (default: 0.9)

    Returns:
        PolynomialDecayLR: Custom polynomial decay LR scheduler
        
    Examples:
        >>> # Polynomial decay over 200 epochs
        >>> scheduler = get_polynomial_decay_lr_scheduler(optimizer, max_epochs=200, power=0.9)
    """
    return PolynomialDecayLR(optimizer, max_epochs=max_epochs, power=power)


# noinspection PyProtectedMember
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):  # noqa: SLF001
    # Note: Accessing _LRScheduler is the official way to subclass PyTorch schedulers
    """
    Learning rate warmup scheduler.
    
    Gradually increases learning rate from warmup_start_lr to base_lr over
    warmup_epochs, then switches to the main scheduler.
    
    This stabilizes early training, especially important for:
    - Transfer learning with pretrained weights
    - Large batch sizes
    - Medical imaging
    
    References:
    - Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
    - Common in modern vision papers (ViT, DINO, etc.)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        main_scheduler: torch.optim.lr_scheduler.
        _LRScheduler,  # noqa: SLF001  # noinspection PyProtectedMember
        warmup_epochs: int,
        warmup_start_lr: float = 1e-6,
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: The optimizer
            main_scheduler: Main scheduler (e.g., CosineAnnealingLR) to use after warmup
            warmup_epochs: Number of epochs for warmup
            warmup_start_lr: Starting learning rate for warmup
        """
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0

        # Store base learning rates from optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer)

        logger.info(f"Warmup scheduler: {warmup_epochs} epochs, "
                    f"{warmup_start_lr:.2e} → {self.base_lrs[0]:.2e}")

    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Use main scheduler
            return self.main_scheduler.get_last_lr()

    def step(self, *args, **kwargs):
        """Step the scheduler."""
        if self.current_epoch < self.warmup_epochs:
            # During warmup
            super().step()
        else:
            # After warmup, use main scheduler
            self.main_scheduler.step(*args, **kwargs)

        self.current_epoch += 1

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'main_scheduler': self.main_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.main_scheduler.load_state_dict(state_dict['main_scheduler'])


def get_scheduler(scheduler_name: str, optimizer: torch.optim.Optimizer,
                  **kwargs):
    """
    Factory function to retrieve a learning rate scheduler by name.
    
    Standard schedulers for medical image classification:
    - cosine: CosineAnnealingLR (RECOMMENDED - modern standard, smooth decay)
    - plateau: ReduceLROnPlateau (practical for medical imaging, metric-based)
    - step: StepLR (classical, ResNet original)
    - exponential: ExponentialLR (smooth exponential decay)
    - polynomial: PolynomialDecayLR (alternative smooth decay)
    
    Warmup Support:
    - Pass warmup_enabled=True to enable warmup
    - warmup_epochs: Number of warmup epochs (default: 5)
    - warmup_start_lr: Starting LR for warmup (default: 1e-6)

    Args:
        scheduler_name: One of "cosine", "plateau", "step", "exponential", "polynomial"
        optimizer: The optimizer to schedule
        **kwargs: Additional arguments for the specific scheduler
                 For cosine: t_max (int), eta_min (float, default 0.0)
                 For plateau: mode (str), factor (float), patience (int), min_lr (float)
                 For step: step_size (int), gamma (float)
                 For exponential: gamma (float)
                 For polynomial: max_epochs (int), power (float, default 0.9)
                 For warmup: warmup_enabled (bool), warmup_epochs (int), warmup_start_lr (float)

    Returns:
        _LRScheduler: The requested learning rate scheduler (optionally wrapped with warmup)

    Raises:
        ValueError: If an invalid scheduler_name is provided
        
    Examples:
        >>> # Cosine annealing with warmup
        >>> cosine_sched = get_scheduler("cosine", optimizer, t_max=150, eta_min=1e-6,
        ...                              warmup_enabled=True, warmup_epochs=5, warmup_start_lr=1e-6)
        >>> 
        >>> # ReduceLROnPlateau (practical for medical imaging)
        >>> plateau_sched = get_scheduler("plateau", optimizer, mode="min", patience=10)
        >>> 
        >>> # Step decay (ResNet-style)
        >>> step_sched = get_scheduler("step", optimizer, step_size=30, gamma=0.1)
    """
    scheduler_name = scheduler_name.lower()

    # Extract warmup parameters
    warmup_enabled = kwargs.pop('warmup_enabled', False)
    warmup_epochs = kwargs.pop('warmup_epochs', 5)
    warmup_start_lr = kwargs.pop('warmup_start_lr', 1e-6)

    # Create main scheduler
    if scheduler_name == "cosine":
        main_scheduler = get_cosine_annealing_lr_scheduler(optimizer, **kwargs)
    elif scheduler_name == "plateau":
        main_scheduler = get_reduce_on_plateau_scheduler(optimizer, **kwargs)
    elif scheduler_name == "step":
        main_scheduler = get_step_lr_scheduler(optimizer, **kwargs)
    elif scheduler_name == "exponential" or scheduler_name == "exp":
        main_scheduler = get_exponential_lr_scheduler(optimizer, **kwargs)
    elif scheduler_name in ["polynomial", "poly"]:
        main_scheduler = get_polynomial_decay_lr_scheduler(optimizer, **kwargs)
    else:
        raise ValueError(
            f"Unsupported scheduler name: {scheduler_name}. "
            f"Choose from 'cosine' (recommended), 'plateau', 'step', 'exponential', 'polynomial'."
        )

    # Wrap with warmup if enabled
    if warmup_enabled and warmup_epochs > 0:
        logger.info(
            f"✓ Wrapping {scheduler_name} scheduler with {warmup_epochs}-epoch warmup"
        )
        return WarmupScheduler(
            optimizer,
            main_scheduler,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
        )
    else:
        return main_scheduler
