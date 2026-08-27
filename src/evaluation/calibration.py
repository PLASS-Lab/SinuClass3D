"""
Shared Platt Scaling Calibration Pipeline.

Extracts the common calibration logic used by:
  - CrossValidator._fit_pooled_calibration() (single-model OOF during CV training)
  - evaluate.py.evaluate_all_folds() (pooled single-model OOF during evaluation)

Both pool leak-free single-model OOF predictions (each sample predicted only
by the model that never saw it). Once (y_true, y_prob, patient_ids) are
assembled, the calibration fitting + threshold selection + artifact saving
is identical. This module provides that shared pipeline.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from src.core.metrics import select_youden_threshold

logger = logging.getLogger(__name__)


def fit_platt_calibration(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """
    Fit Platt scaling (logistic regression on logits).
    
    Args:
        y_true: Ground truth binary labels
        y_prob_pos: Positive-class probabilities (raw/uncalibrated)
        
    Returns:
        (calibrator, calibrated_probs, logits)
    """
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob_pos, eps, 1 - eps)
    logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.simplefilter("ignore")
        calibrator = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        calibrator.fit(logits.reshape(-1, 1), y_true)

    calibrated_probs = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]

    logger.info(f"Raw probabilities range: [{y_prob_pos.min():.4f}, {y_prob_pos.max():.4f}]")
    logger.info(f"Calibrated probabilities range: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]")

    return calibrator, calibrated_probs, logits


def save_calibration_artifacts(
    calibrator: LogisticRegression,
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    calibrated_probs: np.ndarray,
    patient_ids: List,
    youden_threshold: Dict,
    output_dir: Path,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Save calibration artifacts (calibrator pkl, config YAML, predictions CSV).
    
    Args:
        calibrator: Fitted Platt scaling calibrator
        y_true: Ground truth labels
        y_prob_pos: Raw positive-class probabilities
        calibrated_probs: Calibrated probabilities
        patient_ids: Patient identifiers
        youden_threshold: Dict with 'threshold', 'sensitivity', 'specificity'
        output_dir: Directory to save artifacts to
        metadata: Optional extra metadata to include in config YAML
        
    Returns:
        Dictionary with paths and summary info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrated_threshold = youden_threshold['threshold']

    # Save calibrator model
    calibrator_path = output_dir / 'calibrator.pkl'
    joblib.dump(calibrator, calibrator_path)
    logger.info(f"Saved calibrator to {calibrator_path}")

    # Build calibration config
    calibration_config = {
        'calibrated_threshold': calibrated_threshold,
        'youden_threshold': youden_threshold,
        'calibrator_path': str(calibrator_path),
        'n_samples': len(y_true),
        'class_distribution': {
            'negative': int(np.sum(y_true == 0)),
            'positive': int(np.sum(y_true == 1)),
        },
        'raw_prob_range': [float(y_prob_pos.min()), float(y_prob_pos.max())],
        'calibrated_prob_range': [float(calibrated_probs.min()), float(calibrated_probs.max())],
        'threshold_method': 'youden',
        'created_at': datetime.now().isoformat(),
    }

    # Merge extra metadata
    if metadata:
        calibration_config.update(metadata)

    config_path = output_dir / 'calibration_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(calibration_config, f, default_flow_style=False)
    logger.info(f"Saved calibration config to {config_path}")

    # Save OOF predictions CSV
    oof_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': y_true,
        'prob_raw': y_prob_pos,
        'prob_calibrated': calibrated_probs,
    })
    oof_path = output_dir / 'pooled_oof_predictions.csv'
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"Saved OOF predictions to {oof_path}")

    # Log summary
    logger.info("=" * 80)
    logger.info(f"✅ Calibration complete!")
    logger.info(f"   Youden's J threshold = {calibrated_threshold:.4f} "
                f"(sens={youden_threshold['sensitivity']:.4f}, "
                f"spec={youden_threshold['specificity']:.4f})")
    logger.info("=" * 80)

    return {
        'calibrated_threshold': calibrated_threshold,
        'youden_threshold': youden_threshold,
        'calibrator_path': str(calibrator_path),
        'config_path': str(config_path),
        'n_samples': len(y_true),
    }


def fit_and_save_calibration(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    patient_ids: List,
    output_dir: Path,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Full calibration pipeline: fit Platt scaling, select Youden threshold, save artifacts.
    
    This is the single entry point that replaces duplicated calibration code in
    both cross_validation.py and evaluate.py.
    
    Args:
        y_true: Ground truth binary labels
        y_prob_pos: Positive-class probabilities (raw/uncalibrated)
        patient_ids: Patient identifiers matching y_true
        output_dir: Directory to save calibration artifacts
        metadata: Optional extra metadata (e.g. n_folds, method description)
        
    Returns:
        Dictionary with calibration results (threshold, paths, etc.)
    """
    logger.info(f"Pooled {len(y_true)} predictions")
    logger.info(f"Class distribution: {np.sum(y_true == 0)} negative, {np.sum(y_true == 1)} positive")

    # Step 1: Fit Platt scaling
    calibrator, calibrated_probs, _ = fit_platt_calibration(y_true, y_prob_pos)

    # Step 2: Select Youden's J threshold on calibrated probabilities
    youden_threshold = select_youden_threshold(y_true, calibrated_probs)

    # Step 3: Save all artifacts
    return save_calibration_artifacts(
        calibrator=calibrator,
        y_true=y_true,
        y_prob_pos=y_prob_pos,
        calibrated_probs=calibrated_probs,
        patient_ids=patient_ids,
        youden_threshold=youden_threshold,
        output_dir=output_dir,
        metadata=metadata,
    )
