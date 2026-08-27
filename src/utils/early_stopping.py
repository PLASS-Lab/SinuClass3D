"""
Early Stopping Utility

Reusable early stopping logic for any training loop.
Can be used by classification, segmentation, or any other task.
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to stop training when metric stops improving.
    
    Supports both minimization (e.g., loss) and maximization (e.g., accuracy, AUC).
    
    Args:
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: "min" for metrics to minimize (loss), "max" for metrics to maximize (accuracy)
        verbose: Whether to log messages
        
    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='max')
        >>> for epoch in range(100):
        ...     # Train and get validation metric
        ...     val_metric = 0.95  # from your validation loop
        ...     if early_stop.step(val_metric):
        ...         print(f"Stopping at epoch {epoch}")
        ...         break
        ...     if early_stop.is_best():
        ...         # Save your best model checkpoint here
        ...         pass
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        if mode == "min":
            self.best_score = float('inf')
            self.is_improvement = lambda current, best: current < (best -
                                                                   min_delta)
        elif mode == "max":
            self.best_score = float('-inf')
            self.is_improvement = lambda current, best: current > (best +
                                                                   min_delta)
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.counter = 0
        self.early_stop = False
        self._is_best = False

        if verbose:
            logger.info(f"EarlyStopping initialized: patience={patience}, "
                        f"min_delta={min_delta}, mode={mode}")

    def step(self, metric: float) -> bool:
        """
        Update early stopping state with new metric value.
        
        Args:
            metric: Current metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        self._is_best = False

        if self.is_improvement(metric, self.best_score):
            # Improvement detected
            self.best_score = metric
            self.counter = 0
            self._is_best = True

            if self.verbose:
                logger.info(f"EarlyStopping: Metric improved to {metric:.4f}")

        else:
            # No improvement
            self.counter += 1

            if self.verbose:
                logger.debug(
                    f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training (best: {self.best_score:.4f})"
                    )
                return True

        return False

    def is_best(self) -> bool:
        """Check if last metric was the best so far."""
        return self._is_best

    def get_best_score(self) -> float:
        """Get the best score seen so far."""
        return self.best_score

    def reset(self):
        """Reset early stopping state."""
        if self.mode == "min":
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
        self.counter = 0
        self.early_stop = False
        self._is_best = False

        if self.verbose:
            logger.info("EarlyStopping: State reset")
