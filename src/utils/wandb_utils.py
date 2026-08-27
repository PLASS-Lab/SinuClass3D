"""Utility module for Weights & Biases (WandB) integration.

This module provides reusable WandB utilities for experiment tracking.
For initialization, see scripts/train.py.

Core Utilities:
- log_system_info: Log GPU, CUDA, and Python environment info
- save_model_artifact: Save model checkpoints as WandB artifacts
  (Control via force parameter, defaults to True)
- finish_run: Clean WandB shutdown with optional summary metrics

Results & Summary:
- log_cv_results_table: Log cross-validation results as table
- log_dataset_info: Log dataset statistics
- log_config_summary: Log key configuration settings

Note:
- Training metrics (loss, accuracy, AUC per epoch) are logged in trainer.py
- Publication-quality figures use evaluation/visualization/ package
  (ROC curves, calibration plots, confusion matrices, etc.)
"""

import wandb


def log_system_info():
    """Log system information to WandB."""
    import torch
    import sys
    import platform

    if wandb.run is None:
        return

    wandb.log({
        'system/gpu_count':
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'system/cuda_available':
            torch.cuda.is_available(),
        'system/cuda_version':
            torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'system/pytorch_version':
            torch.__version__,
        'system/python_version':
            sys.version.split()[0],
        'system/platform':
            platform.platform(),
    })


def save_model_artifact(model_path: str,
                        name: str,
                        metadata: dict = None,
                        force: bool = True):
    """
    Save model as WandB artifact for versioning.
    
    Args:
        model_path: Path to the saved model checkpoint
        name: Name for the artifact
        metadata: Optional metadata dictionary
        force: If True, upload artifact (default: True)
               Set to False to skip artifact upload
    
    Returns:
        True if artifact was uploaded, False otherwise
    
    Note:
        To control artifact uploads globally, use the force parameter
        when calling this function. Defaults to True for convenience.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Check if WandB run is active
    if wandb.run is None:
        logger.debug("WandB run not initialized, skipping artifact upload")
        return False

    # Check force parameter
    if not force:
        logger.info(f"⏭️  Skipping model artifact upload (force=False): {name}")
        return False

    # Upload artifact
    logger.info(f"📤 Uploading model artifact to W&B: {name}")

    artifact = wandb.Artifact(name=name, type='model', metadata=metadata or {})
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    logger.info(f"✅ Model artifact uploaded: {name}")
    return True


def finish_run(summary_metrics: dict = None):
    """
    Finish WandB run with optional summary metrics.
    
    Args:
        summary_metrics: Dictionary of final summary metrics
    """
    if wandb.run is None:
        return

    if summary_metrics:
        for key, value in summary_metrics.items():
            wandb.run.summary[key] = value

    wandb.finish()


def log_cv_results_table(cv_results: list):
    """
    Log cross-validation results as WandB table.
    
    Args:
        cv_results: List of dicts with fold results
            Example: [
                {'fold': 0, 'roc_auc': 0.88, 'sensitivity': 0.91, ...},
                {'fold': 1, 'roc_auc': 0.86, 'sensitivity': 0.89, ...},
            ]
    """
    if wandb.run is None:
        return

    if not cv_results:
        return

    # Extract columns from first result
    columns = list(cv_results[0].keys())

    # Create table data
    data = [[result[col] for col in columns] for result in cv_results]

    # Add mean row
    import numpy as np
    numeric_cols = [col for col in columns if col != 'fold']
    mean_row = ['mean'] + [
        np.mean([r[col]
                 for r in cv_results
                 if col in r]) if col in numeric_cols else '-'
        for col in columns[1:]
    ]
    data.append(mean_row)

    table = wandb.Table(columns=columns, data=data)
    wandb.log({'cv_results': table})


def log_dataset_info(dataset_stats: dict):
    """
    Log dataset statistics to WandB.
    
    Args:
        dataset_stats: Dictionary with dataset info
            Example: {
                'total_samples': 184,
                'train_samples': 147,
                'val_samples': 37,
                'class_0_count': 75,
                'class_1_count': 109,
                'class_balance': '59/41',
            }
    """
    if wandb.run is None:
        return

    wandb.log({f'dataset/{key}': value for key, value in dataset_stats.items()})


def log_config_summary(config: dict, prefix='config'):
    """
    Log important config settings to WandB.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for logged keys
    """
    if wandb.run is None:
        return

    # Extract key settings
    summary = {
        f'{prefix}/learning_rate': config['optimizer']['lr'],
        f'{prefix}/batch_size': config['data']['batch_size'],
        f'{prefix}/num_epochs': config['training']['num_epochs'],
        f'{prefix}/model': config['model']['name'],
        f'{prefix}/loss': config['loss']['type'],
    }

    # Remove None values
    summary = {k: v for k, v in summary.items() if v is not None}

    wandb.log(summary)
