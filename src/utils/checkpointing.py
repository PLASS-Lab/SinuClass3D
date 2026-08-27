"""
Model Checkpointing Utility

Handles saving and loading model checkpoints during training.
Supports saving best models, periodic checkpoints, and top-K checkpoint management.

Features:
- Save best model based on monitored metric
- Save latest checkpoint for resuming training
- Periodic checkpoint saving every N epochs
- Automatic cleanup of old checkpoints (top-K management)
- Compatible with optimizer and scheduler state saving
"""

import os
import torch
import logging
import heapq

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """
    Handles model checkpointing for saving and resuming training.

    Automatically saves:
    - Best model(s) based on monitored metric (e.g., val_loss, val_auc)
    - Latest checkpoint for resuming interrupted training
    - Periodic checkpoints every N epochs

    Args:
        dirpath: Directory where checkpoints are saved
        monitor: Metric to monitor for best model (e.g., 'val_loss', 'val_auc')
        mode: 'min' for loss metrics, 'max' for accuracy/AUC metrics
        save_top_k: Number of the best checkpoints to keep (older ones are deleted)
        save_last: Whether to always save the latest checkpoint
        save_every_n_epochs: Frequency of periodic checkpoint saving

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     dirpath='checkpoints/',
        ...     monitor='val_auc',
        ...     mode='max',
        ...     save_top_k=3
        ... )
        >>> # Save checkpoint after training epoch (assume model, optimizer exist)
        >>> checkpoint.save_checkpoint(
        ...     epoch=10, model=..., optimizer=..., val_score=0.95
        ... )
    """

    def __init__(
        self,
        dirpath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
        save_every_n_epochs: int = 5,
        min_epoch_for_best: int = 0,
    ):
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.min_epoch_for_best = min_epoch_for_best

        # Initialize best score based on mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_checkpoints = []  # Min-heap to track top-K checkpoints
        # Note: Initialization logging handled by caller for better context

    def save_checkpoint(self,
                        epoch,
                        model,
                        optimizer,
                        val_score,
                        scheduler=None):
        """
        Save a checkpoint and manage the top-K best checkpoints.

        Performs three types of saving:
        1. Latest checkpoint (for resuming training)
        2. Best checkpoint(s) based on validation metric
        3. Periodic checkpoint every N epochs

        Args:
            epoch: Current epoch number
            model: Model instance to save
            optimizer: Optimizer state to save
            val_score: Validation score for the monitored metric
            scheduler: Optional scheduler state to save
        """
        # Prepare checkpoint dictionary with all training state
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_score": val_score,
        }
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # 1. Save the latest checkpoint (always overwrites)
        if self.save_last:
            last_ckpt_path = os.path.join(self.dirpath, "last_checkpoint.pth")
            torch.save(checkpoint, last_ckpt_path)
            logger.info("Saved last checkpoint at epoch %d to %s", epoch,
                         last_ckpt_path)

        # 2. Save periodic checkpoint (every N epochs) — always, regardless of min_epoch
        if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
            periodic_ckpt_path = os.path.join(self.dirpath,
                                              f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, periodic_ckpt_path)
            logger.info("Saved periodic checkpoint at epoch %d to %s", epoch,
                         periodic_ckpt_path)

        # 3. Save the best checkpoint based on validation score
        # Skip best-model saving before min_epoch_for_best (avoids untrained checkpoints)
        if epoch < self.min_epoch_for_best:
            if epoch == 1 or epoch == self.min_epoch_for_best - 1:
                logger.info(
                    "Skipping best-model check at epoch %d (min_epoch_for_best=%d)",
                    epoch, self.min_epoch_for_best)
            return

        is_best = (val_score < self.best_score) if self.mode == "min" else (
            val_score > self.best_score)
        if is_best:
            self.best_score = val_score
            best_ckpt_path = os.path.join(self.dirpath,
                                          f"best_checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, best_ckpt_path)
            logger.info(
                "New best model saved at epoch %d with score %.4f to %s", epoch,
                val_score, best_ckpt_path)

            # Create/update the best_model.pth symlink used by evaluation
            # This ensures THE best model is always identifiable without ambiguity
            best_model_link = os.path.join(self.dirpath, "best_model.pth")
            # Remove old symlink/file if exists
            if os.path.exists(best_model_link) or os.path.islink(best_model_link):
                os.remove(best_model_link)
            # Create symlink to the best checkpoint
            try:
                os.symlink(os.path.basename(best_ckpt_path), best_model_link)
                logger.info("Updated best_model.pth -> %s", os.path.basename(best_ckpt_path))
            except OSError:
                # Fallback to copy if symlink not supported (e.g., Windows)
                import shutil
                shutil.copy2(best_ckpt_path, best_model_link)
                logger.info("Copied best model to best_model.pth")

            # Manage top-K checkpoints using min-heap
            # For mode="min" (e.g., val_loss), negate scores so the min-heap
            # pops the worst (highest loss) checkpoint, not the best.
            # For mode="max", raw scores work: min-heap pops the worst (lowest AUC).
            heap_score = -val_score if self.mode == "min" else val_score
            heapq.heappush(self.best_checkpoints, (heap_score, best_ckpt_path))
            if len(self.best_checkpoints) > self.save_top_k:
                # Remove the worst checkpoint when exceeding top-K
                _, removed_ckpt = heapq.heappop(self.best_checkpoints)
                if os.path.exists(removed_ckpt):
                    os.remove(removed_ckpt)
                    logger.info("Removed old checkpoint: %s", removed_ckpt)
