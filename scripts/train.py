"""
3D Classification Training Script

Supports both single-fold training and K-fold cross-validation via configuration.

Features:
- Single-fold OR K-fold stratified cross-validation (config-controlled)
- CT-specific preprocessing (HU windowing, RAS, isotropic resampling)
- ROI cropping using segmentation masks
- Class imbalance handling (weighted CE or Focal loss)
- AdamW optimizer with cosine annealing
- Early stopping on PR-AUC
- Comprehensive evaluation (ROC-AUC, PR-AUC, calibration, DCA)
- Grad-CAM explainability
- Bootstrap confidence intervals

Usage:
    # Single-fold training (faster, for quick experiments)
    python scripts/classification/train.py data.cross_validation.enabled=false
    
    # 5-fold cross-validation (recommended for final results)
    python scripts/classification/train.py data.cross_validation.enabled=true data.cross_validation.n_folds=5
    
    # Custom experiment name
    python scripts/classification/train.py experiment_name=my_experiment
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, cast

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.core import CrossValidator
from src.data import create_data_module, prepare_csv_for_training
from src.training import get_loss, ClassificationTrainer
from src.models import get_model
from src.utils import setup_logging, set_seed

logger = logging.getLogger(__name__)
_wandb_init_lock = threading.Lock()


def _initialize_wandb(cfg: DictConfig, notes: str = None, fold: int = None):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        cfg: Hydra configuration
        notes: Optional notes for the run
        fold: Optional fold number for cross-validation
        
    Returns:
        wandb.Run object if enabled, None otherwise
    """
    # Extract config sections
    wandb_config = cfg['wandb']

    if not wandb_config['enabled']:
        logger.info("W&B logging disabled")
        return None

    # Convert config to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create run name
    core_config = cfg['core']
    experiment_name = core_config['experiment_name']
    experiment_version = wandb_config['version']

    # Add version to name if specified
    if experiment_version and fold is None:
        run_name = f"{experiment_name}_{experiment_version}"
    elif fold is not None:
        run_name = f"{experiment_name}_fold{fold + 1}"
        tags = ['cross_validation', f'fold_{fold + 1}']
    else:
        run_name = experiment_name
        tags = ['single_fold']

    # Initialize tags for single fold (if not already set)
    if fold is None:
        tags = ['single_fold']

    # Add version tag if specified
    if experiment_version:
        tags.append(experiment_version)

    # Add user tags
    if 'wandb' in cfg and 'tags' in cfg['wandb']:
        tags.extend(cfg['wandb']['tags'])

    # Initialize W&B - serialize with lock for thread safety (parallel folds)
    with _wandb_init_lock:
        run = wandb.init(
            project=cfg['wandb']['project'],  # Required
            entity=wandb_config['entity'],  # Optional
            name=run_name,
            config=config_dict,
            notes=notes or wandb_config['notes'],
            tags=tags,
            reinit=True  # Allow multiple runs in same script (for CV)
        )
        # Detach from global singleton so the next wandb.init(reinit=True)
        # doesn't call finish() on this run. Each fold trainer stores its
        # own run reference via self.wandb_run.
        wandb.run = None
        wl = wandb.setup()
        wl._active_runs.clear()

    logger.info(f"✓ W&B initialized: {cfg['wandb']['project']}/{run_name}")
    return run


def create_trainer_from_config(config: Dict,
                               fold: int = 0,
                               full_training: bool = False,
                               wandb_run=None) -> ClassificationTrainer:
    """
    Factory function to create ClassificationTrainer from config.
    
    Constructs all necessary components (model, data loaders, loss, etc.)
    and returns a configured ClassificationTrainer instance.
    
    Args:
        config: Configuration dictionary
        fold: Fold index for cross-validation
        full_training: If True, train on all samples (no validation split)
        
    Returns:
        Configured ClassificationTrainer instance
    """
    # Device
    device = torch.device(
        config['compute']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data
    data_module = create_data_module(config)
    
    if full_training:
        # Full training mode: use all samples, no validation split
        train_loader = data_module.get_full_train_loader()
        val_loader = None  # No validation in full training mode
        logger.info("Full training mode: using all samples, no validation split")
    else:
        train_loader, val_loader = data_module.get_train_loaders(fold)

    # Model - Full volume 3D classification
    logger.info("Standard full-volume 3D classification")
    model = get_model(config)

    # Loss
    class_weights = data_module.class_weights.to(device)
    criterion = get_loss(config, class_weights)

    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        fold=fold,
        full_training=full_training,
        wandb_run=wandb_run,
    )

    return trainer


@hydra.main(version_base=None,
            config_path="pkg://configs",
            config_name="config_3d")
def main(cfg: DictConfig):
    """
    Main training function.
    
    Supports two modes via config:
    1. Single-fold training (data.cross_validation.enabled=false)
       - Faster training for quick experiments and hyperparameter tuning
       - Uses fixed train/val split
       
    2. K-fold cross-validation (data.cross_validation.enabled=true)
       - Recommended for final model evaluation and publication
       - Provides robust performance estimates with confidence intervals
       - Default: 5 folds (configurable via data.cross_validation.n_folds)
    """
    
    # Setup logging
    logging_config = cfg['logging']
    log_dir = Path(
        cfg.core.experiment_dir) / "logs" if logging_config['file'] else None

    # Convert string level to logging constant
    log_level_str = logging_config['level']
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    setup_logging(log_dir=log_dir,
                  log_file="train.log",
                  level=log_level,
                  console=logging_config['console'])
    logger.info("=" * 80)
    logger.info("3D CLASSIFICATION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Experiment: {cfg['core']['experiment_name']}")
    logger.info(f"Output directory: {cfg['core']['experiment_dir']}")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    seed = cfg['reproducibility']['random_seed']
    deterministic = cfg['reproducibility']['deterministic']
    set_seed(seed, deterministic)
    logger.info(f"Random seed: {seed}")
    logger.info(f"Deterministic mode: {deterministic}")

    # Preprocess CSV: Convert specific image type column to generic 'image_path'
    # This handles the mapping from full_head_ct_path, mask_bbox_margin_path, etc.
    # to the generic 'image_path' column expected by the data loader
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    preprocessed_csv_path = prepare_csv_for_training(cfg_dict)

    # Update config to use preprocessed CSV
    cfg['data']['paths']['csv_path'] = preprocessed_csv_path
    logger.info(f"Using preprocessed CSV: {preprocessed_csv_path}")
    logger.info(f"Image type: {cfg_dict['data']['image_type']}")

    # ══════════════════════════════════════════════════════════════════════
    # Full Volume 3D Classification
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("🎯 TRAINING MODE: Full Volume 3D Classification")
    logger.info("=" * 80)
    logger.info("Model: 3D CNN (ResNet/DenseNet/EfficientNet)")
    logger.info(f"Training: Full volume (160³ @ 1mm)")
    logger.info("No cropping - entire volume processed")
    logger.info("=" * 80)

    # Extract cross-validation config (used throughout this section)
    data_config = cfg['data']
    cv_config = data_config['cross_validation']
    cv_enabled = cv_config['enabled']
    cv_mode = cv_config['mode']

    # Convert config to dict for CrossValidator
    config_dict: Dict[str,
                      Any] = cast(Dict[str, Any],
                                  OmegaConf.to_container(cfg, resolve=True))

    if cv_enabled:
        # Initialize CrossValidator
        cv = CrossValidator(config_dict)

        # Create wrapper for trainer creation with W&B per fold
        def create_trainer_with_wandb(config: Dict, fold: int):
            # Initialize W&B for this fold
            wandb_run = _initialize_wandb(cfg, notes=f"Fold {fold}", fold=fold)

            # Create trainer (pass wandb_run so each fold has its own run)
            trainer = create_trainer_from_config(config, fold=fold, wandb_run=wandb_run)

            return trainer


        if cv_mode == 'standard':
            # ═══════════════════════════════════════════════════════════════
            # STANDARD K-FOLD CROSS-VALIDATION MODE
            # ═══════════════════════════════════════════════════════════════
            n_folds = cfg['data']['cross_validation']['n_folds']
            results = cv.run_standard_cv(
                trainer_class=create_trainer_with_wandb, n_folds=n_folds)

            logger.info("=" * 80)
            logger.info(f"FINAL CV RESULTS ({n_folds} folds)")
            logger.info("=" * 80)
            logger.info(
                f"Mean ± Std: {results['mean']:.4f} ± {results['std']:.4f}")
            logger.info(f"Range: [{results['min']:.4f}, {results['max']:.4f}]")

            # Log CV summary to W&B if enabled and there's an active run
            wandb_config = cfg['wandb']
            if wandb_config['enabled'] and wandb.run is not None:
                wandb.log({
                    'cv/mean': results['mean'],
                    'cv/std': results['std'],
                    'cv/min': results['min'],
                    'cv/max': results['max']
                })
            elif wandb_config['enabled']:
                # Initialize a summary run for aggregated CV results
                summary_run = wandb.init(
                    project=cfg['wandb']['project'],
                    entity=wandb_config.get('entity'),
                    name=f"{cfg['core']['experiment_name']}_cv_summary",
                    tags=['cross_validation', 'summary'],
                    config=OmegaConf.to_container(cfg, resolve=True),
                    reinit=True
                )
                # Log per-fold aggregated metrics
                wandb.log({
                    'cv/mean': results['mean'],
                    'cv/std': results['std'],
                    'cv/min': results['min'],
                    'cv/max': results['max']
                })
                
                # Log pooled OOF metrics if available
                if results.get('pooled_metrics_available', False) and 'pooled_oof_metrics' in results:
                    pooled = results['pooled_oof_metrics']
                    wandb.log({
                        'oof/roc_auc': pooled['roc_auc'],
                        'oof/pr_auc': pooled['pr_auc'],
                        'oof/f1_score': pooled['f1_score'],
                        'oof/sensitivity': pooled['sensitivity'],
                        'oof/specificity': pooled['specificity'],
                        'oof/accuracy': pooled['accuracy'],
                        'oof/balanced_accuracy': pooled['balanced_accuracy'],
                        'oof/n_samples': pooled['n_samples']
                    })
                    logger.info("✓ Pooled OOF metrics logged to W&B")
                
                # Log fold results table
                if 'fold_results' in results:
                    fold_data = [[r['fold'] + 1, r['best_metric']] 
                                 for r in results['fold_results']]
                    table = wandb.Table(
                        columns=['Fold', 'Best Metric'],
                        data=fold_data
                    )
                    wandb.log({'cv_results_table': table})
                wandb.finish()
                logger.info("✓ CV summary logged to W&B")


        else:
            raise ValueError(
                f"Unknown CV mode: {cv_mode}. Choose 'standard' or 'disabled'.")

    else:
        # Check for full training mode
        full_training_config = data_config.get('full_training', {})
        full_training_enabled = full_training_config.get('enabled', False)
        
        if full_training_enabled:
            # ═══════════════════════════════════════════════════════════════
            # FULL TRAINING MODE (Train on all samples, no validation)
            # ═══════════════════════════════════════════════════════════════
            logger.info("=" * 80)
            logger.info("🎯 FULL TRAINING MODE (ALL SAMPLES)")
            logger.info("=" * 80)
            logger.info("Training on 100% of data - no validation split")
            logger.info("Model selection by training loss (lowest loss = best)")
            logger.info("Use this AFTER CV to train final deployment model")
            logger.info("=" * 80)

            # Initialize W&B
            wandb_run = _initialize_wandb(cfg, notes="Full training (all samples)")

            trainer = create_trainer_from_config(config_dict, fold=0, full_training=True, wandb_run=wandb_run)
            num_epochs = config_dict['training']['num_epochs']
            trainer.train(num_epochs=num_epochs)

            logger.info(f"Final training loss: {trainer.best_train_loss:.4f}")
            logger.info(f"Checkpoint saved to: {trainer.checkpoint_dir}")
        else:
            # Single-fold training mode (with validation split)
            logger.info("=" * 80)
            logger.info("🎯 SINGLE-FOLD TRAINING (NO CV)")
            logger.info("=" * 80)
            logger.info(
                "⚠️  Note: For rigorous evaluation, use cross-validation (data.cross_validation.enabled=true)"
            )
            logger.info("=" * 80)

            # Initialize W&B for single-fold training
            wandb_run = _initialize_wandb(cfg, notes="Single-fold training")

            trainer = create_trainer_from_config(config_dict, fold=0, wandb_run=wandb_run)
            num_epochs = config_dict['training']['num_epochs']
            trainer.train(num_epochs=num_epochs)

            logger.info(f"Best validation metric: {trainer.best_val_metric:.4f}")

    logger.info("=" * 80)
    logger.info("✅ Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
