"""
Cross-Validation Module for Classification

This module provides unified cross-validation utilities for classification models.

Features:
- Standard k-fold cross-validation
- Patient-level or scan-level splitting
- Results aggregation with confidence intervals
- Compatible with YAML configuration
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src.data import create_data_module
from src.models import get_model

# Note: set_seed is handled by application code (train.py), not library code

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Cross-validation for classification models.
    
    Handles patient-level vs scan-level splitting automatically.
    
    Args:
        config: Configuration dictionary from YAML
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cv_config = config['data']['cross_validation']
        self.eval_config = config['evaluation']
        self.experiment_dir = Path(config['core']['experiment_dir'])

        # Get seed from config hierarchy (reproducibility > cv_config > default)
        # Note: Global seed is set by application (train.py), not library code
        self.seed = self._get_seed_from_config()
        logger.debug(f"Using random seed: {self.seed}")

        # Log evaluation mode
        eval_mode = self.eval_config['mode']
        compute_pooled = self.eval_config['compute_pooled_metrics']
        per_fold = self.eval_config['per_fold_metrics']

        logger.info("Initialized CrossValidator")
        logger.info(f"  - Experiment dir: {self.experiment_dir}")
        logger.info(f"  - Evaluation mode: {eval_mode}")
        logger.info(f"  - Compute pooled metrics: {compute_pooled}")
        logger.info(f"  - Per-fold metrics: {per_fold}")

    def _run_cv_folds(self, trainer_class: Callable, n_folds: int,
                       log_prefix: str = "") -> Dict:
        """
        Core fold-training loop.
        
        Trains n_folds models, evaluates each, collects OOF predictions,
        fits pooled calibration, and aggregates results.
        
        Args:
            trainer_class: Trainer class or factory function
            n_folds: Number of folds
            log_prefix: Optional prefix for log messages (e.g. "Repeat 3, ")
            
        Returns:
            Dictionary with aggregated results, calibration, and raw OOF predictions
        """
        all_fold_results = []
        all_oof_predictions = []

        for fold in range(n_folds):
            logger.info("=" * 80)
            logger.info(f"{log_prefix}FOLD {fold + 1}/{n_folds}")
            logger.info("=" * 80)

            # Create trainer for this fold
            trainer = trainer_class(self.config, fold=fold)

            # Train
            num_epochs = self.config['training']['num_epochs']
            best_metrics = trainer.train(num_epochs)

            # Evaluate this fold and collect OOF predictions
            logger.info(f"{log_prefix}Evaluating fold {fold + 1}...")
            eval_metrics, oof_preds = self._evaluate_fold(fold, return_predictions=True)
            
            if oof_preds is not None:
                all_oof_predictions.append({
                    'fold': fold,
                    'labels': oof_preds['labels'],
                    'probabilities': oof_preds['probabilities'],
                    'patient_ids': oof_preds['patient_ids'],
                })

            fold_results = {
                'fold': fold,
                'best_metric': trainer.best_val_metric,
                'metrics': best_metrics,
                'eval_metrics': eval_metrics,
            }

            all_fold_results.append(fold_results)
            logger.info(f"{log_prefix}Fold {fold + 1} completed: {trainer.best_val_metric:.4f}")

        # Pooled calibration
        calibration_result = self._fit_pooled_calibration(all_oof_predictions, n_folds)

        # Aggregate results
        aggregated = self._aggregate_results(all_fold_results)
        aggregated['calibration'] = calibration_result

        return {
            'aggregated': aggregated,
            'oof_predictions': all_oof_predictions,
            'calibration': calibration_result,
        }

    def run_standard_cv(self,
                        trainer_class: Callable,
                        n_folds: Optional[int] = None) -> Dict:
        """
        Run standard k-fold cross-validation.
        
        Args:
            trainer_class: Trainer class or factory function
            n_folds: Number of folds (uses config if None)
            
        Returns:
            Dictionary with aggregated results and per-fold metrics
        """
        if n_folds is None:
            n_folds = self.cv_config['n_folds']

        logger.info("=" * 80)
        logger.info(f"🔬 STANDARD {n_folds}-FOLD CROSS-VALIDATION")
        logger.info("=" * 80)

        result = self._run_cv_folds(trainer_class, n_folds)
        aggregated_results = result['aggregated']

        # Save results
        self._save_cv_results(aggregated_results,
                              mode='standard',
                              n_folds=n_folds)

        # Log summary
        self._log_cv_summary(aggregated_results, n_folds)

        return aggregated_results

    def _evaluate_fold(self, fold: int, return_predictions: bool = False):
        """
        Evaluate a single fold after training completes.
        
        Args:
            fold: Fold number (0-indexed)
            return_predictions: If True, return (metrics, predictions) tuple for pooled calibration
            
        Returns:
            Dictionary of evaluation metrics, or tuple (metrics, predictions) if return_predictions=True
        """
        # Get checkpoint path
        exp_dir = Path(self.config['core']['experiment_dir'])
        checkpoint_path = exp_dir / f'fold_{fold + 1}' / 'checkpoints' / 'best_model.pth'
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return ({}, None) if return_predictions else {}
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Create evaluator (lazy import to avoid circular dependency)
        from src.evaluation.evaluator import ClassificationEvaluator
        fold_output_dir = exp_dir / f'fold_{fold + 1}' / 'evaluation'
        self.config['evaluation']['output']['output_dir'] = str(fold_output_dir)
        
        evaluator = ClassificationEvaluator(
            model=model,
            device=device,
            config=self.config,
            is_mil_mode=False,
            checkpoint_path=str(checkpoint_path)
        )
        
        # Create data module and get validation loader
        data_module = create_data_module(self.config)
        val_loader = data_module.get_val_loader(fold=fold, batch_size=1)
        
        # Generate predictions and evaluate
        predictions = evaluator.predict(val_loader)
        metrics = evaluator.evaluate(predictions, save_results=True)
        
        # Save per-fold threshold YAML (uncalibrated, for reference)
        threshold = metrics.get('threshold', 0.5)
        self._save_fold_threshold(fold, threshold, metrics, exp_dir)
        
        logger.info(f"Fold {fold + 1} evaluation: ROC-AUC={metrics.get('roc_auc', 'N/A'):.4f}, Threshold={threshold:.4f}")
        
        if return_predictions:
            return metrics, predictions
        return metrics

    def _save_fold_threshold(self, fold: int, threshold: float, metrics: Dict, exp_dir: Path):
        """
        Save threshold YAML file for this fold.
        
        Args:
            fold: Fold number (0-indexed)
            threshold: Optimal threshold for this fold
            metrics: Evaluation metrics
            exp_dir: Experiment directory
        """
        import yaml
        from datetime import datetime
        
        threshold_config = {
            'fold': fold + 1,
            'threshold': float(threshold),
            'method': self.config.get('evaluation', {}).get('operating_point', {}).get('method', 'youden'),
            'metrics': {
                'roc_auc': float(metrics.get('roc_auc', 0)),
                'pr_auc': float(metrics.get('pr_auc', 0)),
                'sensitivity': float(metrics.get('sensitivity', 0)),
                'specificity': float(metrics.get('specificity', 0)),
            },
            'created_at': datetime.now().isoformat(),
            'experiment_dir': str(exp_dir),
        }
        
        # Save to fold directory
        fold_dir = exp_dir / f'fold_{fold + 1}'
        threshold_path = fold_dir / 'threshold.yaml'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        with open(threshold_path, 'w') as f:
            yaml.dump(threshold_config, f, default_flow_style=False)
        
        logger.info(f"Saved threshold {threshold:.4f} to {threshold_path}")

    def _fit_pooled_calibration(self, all_oof_predictions: List[Dict], n_folds: int) -> Dict:
        """
        Fit a single calibrator on pooled OOF predictions and select one threshold.
        
        This implements the proper calibration workflow:
        1. Pool all OOF predictions (every training patient has an "honest" prediction)
        2. Fit one Platt scaling calibrator on pooled predictions
        3. Pick one threshold on calibrated predictions
        4. Save calibrator + threshold for external evaluation
        
        Args:
            all_oof_predictions: List of dicts with 'labels', 'probabilities', 'patient_ids' per fold
            n_folds: Number of folds
            
        Returns:
            Dictionary with calibration results
        """
        from src.evaluation.calibration import fit_and_save_calibration
        
        if not all_oof_predictions:
            logger.warning("No OOF predictions available for pooled calibration")
            return {}
        
        logger.info("=" * 80)
        logger.info("POOLED CALIBRATION")
        logger.info("=" * 80)
        
        # Pool all OOF predictions
        y_true = np.concatenate([p['labels'] for p in all_oof_predictions])
        y_prob_raw = np.concatenate([p['probabilities'] for p in all_oof_predictions])
        patient_ids = []
        for p in all_oof_predictions:
            patient_ids.extend(p['patient_ids'])
        
        # Extract positive class probability (binary classification)
        if y_prob_raw.ndim == 2 and y_prob_raw.shape[1] == 2:
            y_prob_pos = y_prob_raw[:, 1]
        else:
            y_prob_pos = y_prob_raw
        
        # Calibrate, select Youden threshold, save artifacts (shared pipeline)
        exp_dir = Path(self.config['core']['experiment_dir'])
        calibration_dir = exp_dir / 'calibration'
        
        return fit_and_save_calibration(
            y_true=y_true,
            y_prob_pos=y_prob_pos,
            patient_ids=patient_ids,
            output_dir=calibration_dir,
            metadata={'n_folds': n_folds, 'calibration_method': 'platt_scaling_single_model_oof'},
        )

    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate standard CV results based on evaluation config.
        
        Args:
            fold_results: List of per-fold results
            
        Returns:
            Aggregated results dictionary
        """
        metrics = [r['best_metric'] for r in fold_results]

        aggregated: Dict[str, Any] = {
            'mode': 'standard',
            'n_folds': len(fold_results),
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics)),
            'min': float(np.min(metrics)),
            'max': float(np.max(metrics)),
        }

        # Add per-fold results if enabled
        if self.eval_config['per_fold_metrics']:
            aggregated['fold_results'] = fold_results
            logger.info("✓ Included per-fold results")

        # Compute pooled metrics if enabled
        if self.eval_config['compute_pooled_metrics']:
            # Check if we have predictions available for pooling
            if all('predictions' in r['metrics'] for r in fold_results):
                logger.info("✓ Computing pooled OOF metrics...")
                
                # Concatenate predictions from all folds (Out-Of-Fold)
                all_y_true = []
                all_y_prob = []
                all_y_pred = []
                
                for r in fold_results:
                    all_y_true.extend(r['metrics']['predictions']['y_true'])
                    all_y_prob.extend(r['metrics']['predictions']['y_prob'])
                    all_y_pred.extend(r['metrics']['predictions']['y_pred'])
                
                all_y_true = np.array(all_y_true)
                all_y_prob = np.array(all_y_prob)
                all_y_pred = np.array(all_y_pred)
                
                # Compute pooled metrics using sklearn
                from sklearn.metrics import (
                    roc_auc_score, 
                    average_precision_score,
                    accuracy_score,
                    balanced_accuracy_score,
                    f1_score,
                    recall_score,
                    precision_score
                )
                
                pooled_metrics = {
                    'n_samples': len(all_y_true),
                    'roc_auc': float(roc_auc_score(all_y_true, all_y_prob)),
                    'pr_auc': float(average_precision_score(all_y_true, all_y_prob)),
                    'accuracy': float(accuracy_score(all_y_true, all_y_pred)),
                    'balanced_accuracy': float(balanced_accuracy_score(all_y_true, all_y_pred)),
                    'f1_score': float(f1_score(all_y_true, all_y_pred)),
                    'sensitivity': float(recall_score(all_y_true, all_y_pred)),
                    'specificity': float(recall_score(all_y_true, all_y_pred, pos_label=0)),
                    'precision': float(precision_score(all_y_true, all_y_pred)),
                }
                
                aggregated['pooled_oof_metrics'] = pooled_metrics
                aggregated['pooled_metrics_available'] = True
                
                logger.info("✓ Pooled OOF metrics computed:")
                logger.info(f"  Samples: {pooled_metrics['n_samples']}")
                logger.info(f"  ROC-AUC: {pooled_metrics['roc_auc']:.4f}")
                logger.info(f"  PR-AUC: {pooled_metrics['pr_auc']:.4f}")
                logger.info(f"  F1-Score: {pooled_metrics['f1_score']:.4f}")
                logger.info(f"  Sensitivity: {pooled_metrics['sensitivity']:.4f}")
                logger.info(f"  Specificity: {pooled_metrics['specificity']:.4f}")
            else:
                logger.warning(
                    "⚠ Pooled metrics requested but predictions not available in fold results"
                )
                aggregated['pooled_metrics_available'] = False

        return aggregated

    def _save_cv_results(self, cv_results: Dict, mode: str, n_folds: int):
        """Save CV results to JSON."""
        # n_folds is kept for API consistency but not currently used in filename
        results_file = self.experiment_dir / f"{mode}_cv_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return str(obj)

        with open(results_file, 'w') as f:
            json.dump(cv_results, f, indent=2, default=convert_numpy)

        logger.info(f"Saved CV results to: {results_file}")

    @staticmethod
    def _log_cv_summary(cv_results: Dict, n_folds: int):
        """Log standard CV summary."""
        logger.info("=" * 80)
        logger.info(f"CROSS-VALIDATION RESULTS ({n_folds} folds)")
        logger.info("=" * 80)

        # Log per-fold results if available
        if 'fold_results' in cv_results:
            for r in cv_results['fold_results']:
                logger.info(f"  Fold {r['fold'] + 1}: {r['best_metric']:.4f}")

        # Always log aggregate statistics (per-fold averages)
        logger.info(f"Per-Fold Averages:")
        logger.info(
            f"  Mean ± Std: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
        logger.info(
            f"  Range: [{cv_results['min']:.4f}, {cv_results['max']:.4f}]")

        # Log pooled OOF metrics if computed
        if cv_results.get('pooled_metrics_available', False) and 'pooled_oof_metrics' in cv_results:
            logger.info("Pooled OOF Metrics (All Patients):")
            pooled = cv_results['pooled_oof_metrics']
            logger.info(f"  ROC-AUC: {pooled['roc_auc']:.4f}")
            logger.info(f"  PR-AUC: {pooled['pr_auc']:.4f}")
            logger.info(f"  F1-Score: {pooled['f1_score']:.4f}")
            logger.info(f"  Sensitivity: {pooled['sensitivity']:.4f}")
            logger.info(f"  Specificity: {pooled['specificity']:.4f}")
            logger.info(f"  Accuracy: {pooled['accuracy']:.4f}")
            logger.info(f"  Balanced Accuracy: {pooled['balanced_accuracy']:.4f}")

    def _get_seed_from_config(self) -> int:
        """
        Get random seed from config with fallback hierarchy.
        
        Priority:
        1. reproducibility.random_seed (preferred)
        2. data.cross_validation.random_seed
        3. core.seed (legacy)
        4. Default: 42
        
        Returns:
            Random seed value
        """
        # First try reproducibility config (preferred)
        seed = self.config.get('reproducibility', {}).get('random_seed')
        if seed is not None:
            logger.debug(f"Using seed from reproducibility.random_seed: {seed}")
            return seed

        # Fall back to CV-specific config
        seed = self.cv_config['random_seed']
        if seed is not None:
            logger.debug(
                f"Using seed from data.cross_validation.random_seed: {seed}")
            return seed

        # Fall back to core.seed (legacy)
        seed = self.config['core']['seed']
        if seed is not None:
            logger.debug(f"Using seed from core.seed: {seed}")
            return seed

        # Default fallback
        logger.debug("Using default seed: 42")
        return 42
