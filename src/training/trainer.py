"""
Classification Trainer Module

This module provides the ClassificationTrainer class for training 3D medical
image classification models with comprehensive support for:
- Full-volume 3D classification
- Mixed precision training (AMP)
- Early stopping and checkpointing
- Comprehensive metrics evaluation
- Learning rate scheduling
- Gradient clipping

Compatible with the classification training scripts and YAML configuration.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training import get_optimizer, get_scheduler, WarmupScheduler
from src.core import create_metrics_evaluator
from src.utils import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """
    Trainer for 3D medical image classification.
    
    Handles the complete training loop including:
    - Training and validation epochs
    - Mixed precision training
    - Early stopping
    - Model checkpointing
    - Comprehensive metrics
    
    Args:
        model: Classification model (3D CNN)
        config: Configuration dictionary from YAML
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Torch device (cuda or cpu)
        fold: Cross-validation fold index (default: 0)
    
    Example:
        >>> # Assume you have already created:
        >>> # - model: Your 3D CNN (nn.Module)
        >>> # - config: Configuration dict from YAML
        >>> # - train_loader, val_loader: PyTorch DataLoaders
        >>> # - loss_fn: Loss function (e.g., nn.BCEWithLogitsLoss())
        >>> import torch
        >>> trainer = ClassificationTrainer(
        ...     model=...,           # Your model instance
        ...     config=...,          # Your config dict
        ...     train_loader=...,    # Your train DataLoader
        ...     val_loader=...,      # Your val DataLoader
        ...     criterion=...,       # Your loss function
        ...     device=torch.device('cuda')
        ... )
        >>> trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        fold: int = 0,
        full_training: bool = False,
        wandb_run=None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.fold = fold
        self.full_training = full_training
        self.wandb_run = wandb_run

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Extract training config once (cleaner pattern)
        training_config = config['training']

        # Mixed precision training (optional)
        self.use_amp = training_config['mixed_precision']
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training enabled")

        # Model selection (which checkpoint to use for testing)
        # Must be parsed before best_val_metric init (needs selection_mode)
        self.model_selection_config = training_config['model_selection']
        self.selection_criterion = self.model_selection_config['criterion']
        self.selection_mode = self.model_selection_config['mode']
        logger.info(
            f"Model selection criterion: {self.selection_criterion} ({self.selection_mode})"
        )

        # Early stopping
        self.best_val_metric = np.inf if self.selection_mode == 'min' else -np.inf
        self.best_train_loss = np.inf  # For full training mode
        self._full_train_patience_counter = 0  # For full training early stopping
        self.patience_counter = 0
        self.early_stop_config = training_config['early_stopping']

        # Checkpointing (using utility class)
        core_config = config['core']
        experiment_dir = Path(core_config['experiment_dir'])
        
        # Full training mode: save to full_train/ directory
        if full_training:
            checkpoint_subdir = config['data'].get('full_training', {}).get('checkpoint_subdir', 'full_train')
            self.checkpoint_dir = experiment_dir / checkpoint_subdir / "checkpoints"
        else:
            self.checkpoint_dir = experiment_dir / f"fold_{fold + 1}" / "checkpoints"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        # Checkpoint manager config (full training uses minimal checkpointing)
        full_training_config = config['data'].get('full_training', {})
        if full_training:
            save_top_k = full_training_config.get('save_top_k', 1)
            save_last = full_training_config.get('save_last', False)
            checkpoint_every_n = 0 if not full_training_config.get('save_periodic', False) else training_config['checkpoint_every_n_epochs']
        else:
            save_top_k = training_config['save_top_k']
            save_last = training_config['save_last']
            checkpoint_every_n = training_config['checkpoint_every_n_epochs']

        min_epoch_for_best = self.model_selection_config.get('min_epoch_for_best', 0)
        self.checkpoint_manager = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            monitor=self.selection_criterion,
            mode=self.selection_mode,
            save_top_k=save_top_k,
            save_last=save_last,
            save_every_n_epochs=checkpoint_every_n,
            min_epoch_for_best=min_epoch_for_best,
        )
        logger.info(
            f"ModelCheckpoint initialized: monitor={self.selection_criterion}, mode={self.selection_mode}, "
            f"save_top_k={save_top_k}, save_last={save_last}, checkpoint_every_n_epochs={checkpoint_every_n}, "
            f"min_epoch_for_best={min_epoch_for_best}"
        )

        # Early stopping (using utility class)
        early_stop_patience = self.early_stop_config['patience']
        early_stop_min_delta = self.early_stop_config.get('min_delta', 0.0)
        self.early_stop_metric = self.early_stop_config.get('metric', self.selection_criterion)
        early_stop_mode = self.early_stop_config.get('mode', self.selection_mode)
        self.early_stopping = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            mode=early_stop_mode,
            verbose=True,
        )
        logger.info(
            f"EarlyStopping initialized: patience={early_stop_patience}, "
            f"metric={self.early_stop_metric}, mode={early_stop_mode}, min_delta={early_stop_min_delta}"
        )

        # Metrics evaluator (training mode: fast validation, no bootstrap CIs)
        self.metrics_evaluator = create_metrics_evaluator(config, is_training_mode=True)
        
        # Store predictions from best epoch for OOF pooled metrics
        self.best_predictions = None

        # Gradient clipping config (parse once, not per batch)
        # Config is under training.optimizer.grad_clip
        opt_config = config['training']['optimizer']
        grad_clip_config = opt_config['grad_clip']

        self.grad_clip_enabled = grad_clip_config['enabled']
        self.grad_clip_max_norm = grad_clip_config['max_norm']
        self.grad_clip_norm_type = grad_clip_config['norm_type']
        if self.grad_clip_enabled:
            logger.info(
                f"Gradient clipping enabled: max_norm={self.grad_clip_max_norm}, norm_type={self.grad_clip_norm_type}"
            )

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        # Get optimizer config from training hierarchy (REQUIRED)
        if 'training' not in self.config or 'optimizer' not in self.config[
                'training']:
            raise KeyError(
                f"Cannot find 'training.optimizer' in config. "
                f"Ensure optimizer config is under 'training.optimizer' hierarchy."
            )

        opt_config = self.config['training']['optimizer']

        # Build optimizer kwargs - only pass parameters that exist in config
        # This allows different optimizers to have different parameters
        # (e.g., AdamW has betas/eps, SGD has momentum/nesterov)
        optimizer_kwargs = {
            'lr': opt_config['lr'],  # Required
            'weight_decay': opt_config['weight_decay'],
        }

        # Add optimizer-specific parameters if they exist
        if 'betas' in opt_config:
            optimizer_kwargs['betas'] = opt_config['betas']
        if 'eps' in opt_config:
            optimizer_kwargs['eps'] = opt_config['eps']
        if 'momentum' in opt_config:
            optimizer_kwargs['momentum'] = opt_config['momentum']
        if 'nesterov' in opt_config:
            optimizer_kwargs['nesterov'] = opt_config['nesterov']
        if 'amsgrad' in opt_config:
            optimizer_kwargs['amsgrad'] = opt_config['amsgrad']

        optimizer = get_optimizer(
            opt_config['name'],  # Required
            self.model.parameters(),
            **optimizer_kwargs)

        logger.info(
            f"Created optimizer: {opt_config['name']} (lr={opt_config['lr']})")
        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler from config."""
        # Get scheduler config from training hierarchy (OPTIONAL)
        if 'training' not in self.config or 'scheduler' not in self.config[
                'training']:
            logger.warning(
                "No 'training.scheduler' found in config. "
                "Training will proceed without learning rate scheduling.")
            return None

        sch_config = self.config['training']['scheduler']

        # Warmup config
        warmup_config = sch_config['warmup']
        warmup_enabled = warmup_config['enabled']
        warmup_epochs = warmup_config['warmup_epochs']
        warmup_start_lr = warmup_config['warmup_start_lr']

        # Build scheduler kwargs - only pass parameters that exist in config
        # This allows different schedulers to have different parameters
        # (e.g., cosine has T_max/eta_min, reduce_on_plateau has mode/patience)
        scheduler_kwargs = {}

        # Add scheduler-specific parameters if they exist
        # Note: Config uses T_max but function uses t_max (lowercase)
        if 'T_max' in sch_config:
            scheduler_kwargs['t_max'] = sch_config['T_max']
        if 'eta_min' in sch_config:
            scheduler_kwargs['eta_min'] = sch_config['eta_min']
        if 'mode' in sch_config:
            scheduler_kwargs['mode'] = sch_config['mode']
        if 'factor' in sch_config:
            scheduler_kwargs['factor'] = sch_config['factor']
        if 'patience' in sch_config:
            scheduler_kwargs['patience'] = sch_config['patience']
        if 'min_lr' in sch_config:
            scheduler_kwargs['min_lr'] = sch_config['min_lr']
        if 'step_size' in sch_config:
            scheduler_kwargs['step_size'] = sch_config['step_size']
        if 'gamma' in sch_config:
            scheduler_kwargs['gamma'] = sch_config['gamma']
        if 'max_epochs' in sch_config:
            scheduler_kwargs['max_epochs'] = sch_config['max_epochs']
        if 'power' in sch_config:
            scheduler_kwargs['power'] = sch_config['power']
        if 'threshold' in sch_config:
            scheduler_kwargs['threshold'] = sch_config['threshold']

        # Always add warmup parameters
        scheduler_kwargs['warmup_enabled'] = warmup_enabled
        scheduler_kwargs['warmup_epochs'] = warmup_epochs
        scheduler_kwargs['warmup_start_lr'] = warmup_start_lr

        scheduler = get_scheduler(
            sch_config['name'],  # Required
            self.optimizer,
            **scheduler_kwargs)

        logger.info(f"Created scheduler: {sch_config['name']}")
        if warmup_enabled:
            logger.info(
                f"  ✓ LR warmup enabled: {warmup_epochs} epochs ({warmup_start_lr:.2e} → base_lr)"
            )
        return scheduler

    def train_one_epoch(self, epoch: int) -> Dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics (loss, accuracy, etc.)
        """
        self.model.train()
        self.current_epoch = epoch

        running_loss = 0.0

        if self.full_training:
            pbar = tqdm(self.train_loader, desc=f"Full Train Epoch {epoch}")
        else:
            pbar = tqdm(self.train_loader, desc=f"Fold {self.fold + 1} Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            batch: Dict[str, Any]  # Type hint for PyCharm
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip_enabled:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_max_norm,
                        norm_type=self.grad_clip_norm_type)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.grad_clip_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_max_norm,
                        norm_type=self.grad_clip_norm_type)

                self.optimizer.step()

            # Update global step
            self.global_step += 1

            # Collect metrics
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Epoch metrics
        avg_loss = running_loss / len(self.train_loader)

        results = {
            'train_loss': avg_loss,
            'epoch': epoch,
        }

        logger.info(f"Train Epoch {epoch}: Loss={avg_loss:.4f}")

        return results

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        running_loss = 0.0
        all_labels = []
        all_probs = []
        all_logit_diffs = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch: Dict[str, Any]  # Type hint for PyCharm
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Collect metrics
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            # Raw logit diff: z1 - z0 (before softmax)
            logit_diff = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            all_logit_diffs.extend(logit_diff)

        # Compute minimal metrics for fast validation
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_logit_diffs = np.array(all_logit_diffs)

        avg_loss = running_loss / len(self.val_loader)

        # Primary metrics only (for model selection and monitoring)
        # Note: Comprehensive metrics with bootstrap CIs computed in final evaluation only
        from sklearn.metrics import roc_auc_score, average_precision_score

        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        pr_auc = average_precision_score(all_labels, all_probs[:, 1])

        # Raw logit diff range (z1 - z0) — direct model output before softmax
        logit_diff_min, logit_diff_max = float(all_logit_diffs.min()), float(all_logit_diffs.max())

        # Calibration slope — detect degenerate models early
        # A well-calibrated model has slope ~1.0; >>1 means underconfident, <<1 means overconfident
        prob_pos = all_probs[:, 1]
        prob_min, prob_max = float(prob_pos.min()), float(prob_pos.max())
        cal_slope = float('nan')
        try:
            from sklearn.linear_model import LogisticRegression
            eps = 1e-7
            clipped = np.clip(prob_pos, eps, 1.0 - eps)
            logit_vals = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            # Only compute if there is variance in logits and both classes present
            if logit_vals.std() > 1e-6 and len(np.unique(all_labels)) == 2:
                cal_lr = LogisticRegression(C=np.inf, solver='lbfgs', max_iter=1000)
                cal_lr.fit(logit_vals, all_labels)
                cal_slope = float(cal_lr.coef_[0, 0])
        except Exception:
            pass  # Non-critical monitoring — don't break training

        metrics = {
            'val_loss': avg_loss,
            'val_roc_auc': roc_auc,
            'val_pr_auc': pr_auc,
            'val_cal_slope': cal_slope,
            'val_prob_min': prob_min,
            'val_prob_max': prob_max,
            'val_logit_diff_min': logit_diff_min,
            'val_logit_diff_max': logit_diff_max,
            'epoch': epoch,
            'predictions': {
                'y_true': all_labels.tolist(),
                'y_prob': all_probs[:, 1].tolist(),  # Positive class probability
                'y_pred': (all_probs[:, 1] >= 0.5).astype(int).tolist()  # Binary predictions
            }
        }

        # Log key metrics (for WandB and console)
        prob_range_str = f"[{prob_min:.4f}, {prob_max:.4f}]"
        logit_diff_str = f"[{logit_diff_min:.2f}, {logit_diff_max:.2f}]"
        cal_slope_str = f"{cal_slope:.2f}" if not np.isnan(cal_slope) else "N/A"
        logger.info(f"Val Epoch {epoch}: "
                    f"Loss={avg_loss:.4f}, "
                    f"ROC-AUC={roc_auc:.4f}, "
                    f"PR-AUC={pr_auc:.4f}, "
                    f"Prob={prob_range_str}, "
                    f"LogitDiff={logit_diff_str}, "
                    f"CalSlope={cal_slope_str}")

        return metrics

    def train(self, num_epochs: int) -> Dict:
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Dictionary with final training results
        """
        if self.full_training:
            logger.info(f"Starting FULL training for {num_epochs} epochs (all samples, no validation)")
        else:
            logger.info(f"Starting training for {num_epochs} epochs (Fold {self.fold + 1})")

        best_metrics = {}

        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_one_epoch(epoch)

            if self.full_training:
                # Full training mode: no validation, save by training loss
                current_loss = train_metrics['train_loss']
                
                # Learning rate scheduling (step-based, no metric needed)
                if self.scheduler is not None:
                    _is_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    if not _is_plateau and isinstance(self.scheduler, WarmupScheduler):
                        _is_plateau = isinstance(self.scheduler.main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    if not _is_plateau:
                        self.scheduler.step()
                
                # Save checkpoint by training loss (lower is better)
                # Invert loss for checkpoint manager (which expects higher=better)
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    val_score=-current_loss,  # Negate so lower loss = higher score
                    scheduler=self.scheduler,
                )
                
                # Track best training loss
                if current_loss < self.best_train_loss:
                    self.best_train_loss = current_loss
                    best_metrics = train_metrics.copy()
                    self._full_train_patience_counter = 0  # Reset patience
                else:
                    self._full_train_patience_counter += 1
                
                # Log to W&B (train only)
                self._log_metrics_to_wandb(epoch, train_metrics, {})
                
                # Early stopping for full training mode (based on training loss)
                if self.early_stop_config.get('enabled', True):
                    patience = self.early_stop_config.get('patience', 30)
                    if self._full_train_patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                        logger.info(f"Best training loss: {self.best_train_loss:.4f}")
                        break
            else:
                # Standard mode with validation
                val_metrics = self.validate(epoch)

                # Learning rate scheduling
                if self.scheduler is not None:
                    # Check if scheduler (or wrapped scheduler) is ReduceLROnPlateau
                    _is_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    if not _is_plateau and isinstance(self.scheduler, WarmupScheduler):
                        _is_plateau = isinstance(self.scheduler.main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    if _is_plateau:
                        # ReduceLROnPlateau needs a metric — use the early stopping metric
                        monitor_metric = val_metrics[self.early_stop_metric]
                        self.scheduler.step(monitor_metric)
                    else:
                        self.scheduler.step()

                # Checkpointing (using the model_selection criterion)
                current_metric = val_metrics[self.selection_criterion]

                # Save checkpoint using utility (handles best/latest/periodic automatically)
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    val_score=current_metric,
                    scheduler=self.scheduler,
                )

                # Track the best metrics (for final reporting)
                # Model selection is ALWAYS by val metric, independent of early stopping metric
                is_better = (current_metric > self.best_val_metric if self.selection_mode == 'max'
                             else current_metric < self.best_val_metric)
                if is_better:
                    best_metrics = val_metrics.copy()
                    self.best_val_metric = current_metric
                    # Store predictions from best epoch for OOF pooling
                    self.best_predictions = val_metrics['predictions'].copy()

                # Log to W&B
                self._log_metrics_to_wandb(epoch, train_metrics, val_metrics)

                # Early stopping (may monitor a different metric than model selection)
                if self.early_stop_metric == 'train_loss':
                    early_stop_value = train_metrics['train_loss']
                else:
                    early_stop_value = val_metrics.get(self.early_stop_metric, current_metric)

                if 'enabled' in self.early_stop_config and not self.early_stop_config[
                        'enabled']:
                    # Early stopping disabled, continue training
                    pass
                elif self.early_stopping.step(early_stop_value):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    logger.info(
                        f"Best {self.early_stop_metric}: {self.early_stopping.get_best_score():.4f}"
                    )
                    break

        logger.info("Training completed!")
        
        if self.full_training:
            logger.info(f"Best training loss: {self.best_train_loss:.4f}")
        else:
            # Log metrics summary (without predictions - too verbose for console)
            metrics_summary = {k: v for k, v in best_metrics.items() if k != 'predictions'}
            logger.info(f"Best validation metrics: {metrics_summary}")
            
            # Ensure predictions from best epoch are included for OOF pooling
            if self.best_predictions is not None and 'predictions' not in best_metrics:
                best_metrics['predictions'] = self.best_predictions
        
        # Finish W&B run for this fold
        if self.wandb_run is not None:
            self.wandb_run.finish()
            logger.info("✓ W&B run finished")

        return best_metrics

    def _log_metrics_to_wandb(self, epoch: int, train_metrics: Dict,
                              val_metrics: Dict) -> None:
        """
        Log metrics to Weights & Biases.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        if not self.wandb_run:
            return

        # Prepare metrics for logging
        log_dict = {
            'epoch': epoch,
            'fold': self.fold,
        }

        # Training metrics
        for key, value in train_metrics.items():
            if key != 'epoch' and isinstance(value, (int, float)):
                # Add prefix if not already present
                metric_name = key if key.startswith(
                    'train_') else f'train/{key}'
                log_dict[metric_name] = value

        # Validation metrics
        for key, value in val_metrics.items():
            if key != 'epoch' and isinstance(value, (int, float)):
                # Add prefix if not already present (e.g., 'val_pr_auc' stays as is)
                metric_name = key if key.startswith('val_') else f'val/{key}'
                log_dict[metric_name] = value

        # Learning rate
        if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']

        # Log to W&B (use epoch as x-axis for standard monitoring)
        self.wandb_run.log(log_dict, step=epoch)
