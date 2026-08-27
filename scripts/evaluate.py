"""
3D Classification Evaluation Script

Supports two evaluation modes:
1. Internal CV evaluation (all folds or specific fold)
2. External validation (ensemble of all folds)

Two-Phase Evaluation:
Phase 1 - Core Metrics (src.evaluation):
    - ROC curve with 95% CI (bootstrap)
    - PR curve with 95% CI
    - Calibration plot (reliability diagram)
    - Confusion matrix
    - Basic metrics table (CSV + LaTeX)
    - Grad-CAM explanations (optional)
    - Test-Time Augmentation (optional)

Usage:
    # Mode 1: Held-out test cohort evaluation (ensemble of all folds)
    python scripts/evaluate.py \
        evaluation.held_out_test_cohort=true \
        evaluation.experiment_dir=outputs/experiment/ \
        data.paths.csv_path=csv/patient_ids_held_out_test.csv \
        evaluation/operating_point=deployment_threshold \
    
    # Mode 2a: Internal CV evaluation (all folds)
    python scripts/evaluate.py \
        evaluation.evaluate_all_folds=true \
        evaluation.experiment_dir=outputs/experiment/
    
    # Mode 2b: Internal CV evaluation (specific fold)
    python scripts/evaluate.py \
        evaluation.evaluate_all_folds=false \
        evaluation.fold=2 \
        evaluation.experiment_dir=outputs/experiment/
    
    # With comprehensive report
    python scripts/evaluate.py \
        evaluation.fold=0 \
        evaluation.experiment_dir=outputs/experiment/ \
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from src.data import create_data_module, prepare_csv_for_training
from src.evaluation import ClassificationEvaluator
from src.models import get_model
from src.utils import convert_to_serializable

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(cfg: Dict, checkpoint_path: str):
    """
    Load model and restore weights from checkpoint.
    
    Args:
        cfg: Configuration dictionary
        checkpoint_path: Path to model checkpoint (.pth file)
        
    Returns:
        tuple: (model, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = get_model(cfg)
    # Model name is logged by get_model factory

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=False)

    # Load state dict with error handling
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"✓ Loaded checkpoint from {checkpoint_path}")
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")
        logger.warning("Attempting to load with strict=False...")
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")
        logger.info(
            f"✓ Loaded checkpoint from {checkpoint_path} (with warnings)")

    return model, device


def evaluate_single_fold(cfg: Dict,
                         fold: int = 0,
                         save_to_fold_dir: bool = False,
                         return_predictions: bool = False):
    """
    Evaluate a single model checkpoint.
    
    Args:
        cfg: Configuration dictionary
        fold: Fold number (for CV models)
        save_to_fold_dir: If True, save results to fold-specific directory
        return_predictions: If True, return (metrics, predictions) tuple for pooled calibration
        
    Returns:
        Dictionary of evaluation metrics, or (metrics, predictions) if return_predictions=True
    """
    # Extract evaluation config
    evaluation_config = cfg['evaluation']

    checkpoint_path = evaluation_config['checkpoint_path']
    if not checkpoint_path:
        # Try to find checkpoint in experiment directory
        exp_dir = Path(evaluation_config['experiment_dir'])
        checkpoints_dir = exp_dir / f'fold_{fold + 1}' / 'checkpoints'

        # Look for best checkpoint (priority order)
        checkpoint_candidates = [
            checkpoints_dir /
            'best_model.pth',  # Priority 1: Symlink to THE best checkpoint
        ]

        # Fallback: Look for best_checkpoint_epoch_*.pth files
        if checkpoints_dir.exists():
            best_epoch_checkpoints = sorted(
                checkpoints_dir.glob('best_checkpoint_epoch_*.pth'))
            if best_epoch_checkpoints:
                # With save_top_k=1, there should be only 1, but pick most recent if multiple
                checkpoint_candidates.append(best_epoch_checkpoints[-1])

        # Last resort: use last checkpoint (not ideal for evaluation)
        checkpoint_candidates.append(checkpoints_dir / 'last_checkpoint.pth')

        # Find first existing checkpoint
        for candidate in checkpoint_candidates:
            if candidate.exists():
                checkpoint_path = candidate
                logger.info(f"Auto-discovered checkpoint: {checkpoint_path}")
                break
        else:
            raise ValueError(
                f"No checkpoint found in {checkpoints_dir}. "
                f"Tried: {[c.name for c in checkpoint_candidates]}")

    logger.info(f"Evaluating fold {fold + 1}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Override output directory for fold-specific saving
    if save_to_fold_dir:
        exp_dir = Path(evaluation_config['experiment_dir'])
        fold_output_dir = exp_dir / f'fold_{fold + 1}' / 'evaluation'
        cfg['evaluation']['output']['output_dir'] = str(
            fold_output_dir)
        logger.info(f"Results will be saved to: {fold_output_dir}")

    # Load model from checkpoint
    model, device = load_model_from_checkpoint(cfg, str(checkpoint_path))

    # Create evaluator with loaded model
    evaluator = ClassificationEvaluator(model=model,
                                        device=device,
                                        config=cfg,
                                        is_mil_mode=False,
                                        checkpoint_path=str(checkpoint_path))

    # Create data module
    data_module = create_data_module(cfg)

    # Get validation loader for evaluation (efficient - skips train data)
    # Force batch_size=1 to avoid OOM with full-volume validation images
    test_loader = data_module.get_val_loader(fold=fold, batch_size=1)

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = evaluator.predict(test_loader)

    # Phase 1: Evaluate and compute basic metrics
    logger.info("Computing performance metrics...")
    metrics = evaluator.evaluate(predictions, save_results=True)


    if return_predictions:
        return metrics, predictions
    return metrics


def evaluate_all_folds(cfg: Dict):
    """
    Evaluate all folds from cross-validation and aggregate results.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with aggregated metrics and per-fold results
    """
    cv_config = cfg['data']['cross_validation']
    n_folds = 1 if not cv_config['enabled'] else cv_config['n_folds']
    exp_dir = Path(cfg['evaluation']['experiment_dir'])

    logger.info("=" * 80)
    logger.info(f"Evaluating all {n_folds} folds from cross-validation")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info("=" * 80)

    all_fold_metrics = []
    all_oof_predictions = []  # Collect OOF predictions for pooled calibration

    for fold in range(n_folds):
        logger.info("=" * 80)
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info("=" * 80)

        # Update config with fold-specific checkpoint (will be auto-discovered)
        fold_cfg = cfg.copy()
        fold_cfg['evaluation'][
            'checkpoint_path'] = ''  # Let evaluate_single_fold auto-discover

        # Evaluate this fold with fold-specific output directory
        try:
            metrics, predictions = evaluate_single_fold(fold_cfg,
                                                        fold=fold,
                                                        save_to_fold_dir=True,
                                                        return_predictions=True)
        except ValueError as e:
            logger.warning(f"Skipping fold {fold + 1}: {e}")
            continue
        all_fold_metrics.append({'fold': fold, 'metrics': metrics})
        all_oof_predictions.append({'fold': fold, 'predictions': predictions})

    # Aggregate results
    logger.info("=" * 80)
    logger.info("CROSS-VALIDATION AGGREGATED RESULTS")
    logger.info("=" * 80)

    # Compute mean and std for each metric
    aggregated = {}
    metric_names = [
        'roc_auc', 'pr_auc', 'f1_score', 'sensitivity', 'specificity',
        'accuracy', 'balanced_accuracy', 'brier_score', 'ece'
    ]

    for metric_name in metric_names:
        values = [
            fold['metrics'][metric_name]
            for fold in all_fold_metrics
            if metric_name in fold['metrics'] and
            fold['metrics'][metric_name] is not None
        ]

        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregated[metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'values': [float(v) for v in values]
            }

            logger.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    # ============================================================================
    # AUTO-FREEZE THRESHOLD FOR EXTERNAL VALIDATION
    # ============================================================================
    # Collect per-fold thresholds (for reference / variability analysis)
    per_fold_thresholds = [
        fold['metrics']['threshold']
        for fold in all_fold_metrics
        if 'threshold' in fold['metrics'] and
        fold['metrics']['threshold'] is not None
    ]

    if per_fold_thresholds:
        threshold_median = float(np.median(per_fold_thresholds))
        threshold_mean = float(np.mean(per_fold_thresholds))
        threshold_std = float(np.std(per_fold_thresholds))
        threshold_min = float(np.min(per_fold_thresholds))
        threshold_max = float(np.max(per_fold_thresholds))

        logger.info("=" * 80)
        logger.info("PER-FOLD THRESHOLD ANALYSIS (Reference)")
        logger.info("=" * 80)
        logger.info(f"Per-fold thresholds: {[f'{t:.4f}' for t in per_fold_thresholds]}")
        logger.info(f"Mean: {threshold_mean:.4f} ± {threshold_std:.4f}")
        logger.info(f"Range: [{threshold_min:.4f}, {threshold_max:.4f}]")
        logger.info(f"{'='*80}")

    # ============================================================================
    # POOLED OOF CALIBRATION (Youden's J)
    # ============================================================================
    # Pool single-model OOF predictions (each sample predicted by the ONE model
    # that never saw it during training — no data leakage).
    # Fit Platt scaling on these pooled predictions and select Youden threshold.
    from src.evaluation.calibration import fit_and_save_calibration

    logger.info("=" * 80)
    logger.info("POOLED OOF CALIBRATION (Platt Scaling)")
    logger.info("=" * 80)

    calibration_dir = exp_dir / 'calibration'
    youden_threshold = None

    if all_oof_predictions:
        # Pool per-fold single-model OOF predictions
        y_true = np.concatenate([p['predictions']['labels'] for p in all_oof_predictions])
        y_prob_raw = np.concatenate([p['predictions']['probabilities'] for p in all_oof_predictions])
        patient_ids = []
        fold_ids = []
        for p in all_oof_predictions:
            pids = p['predictions']['patient_ids']
            patient_ids.extend(pids)
            fold_ids.extend([p['fold'] + 1] * len(pids))

        # Extract positive class probability (binary classification)
        if y_prob_raw.ndim == 2 and y_prob_raw.shape[1] == 2:
            y_prob_pos = y_prob_raw[:, 1]
        else:
            y_prob_pos = y_prob_raw

        logger.info(f"Pooled {len(y_true)} single-model OOF predictions (leak-free)")
        logger.info(f"Class distribution: {np.bincount(y_true.astype(int))}")
        logger.info(f"OOF prob range: [{y_prob_pos.min():.4f}, {y_prob_pos.max():.4f}]")

        cal_result = fit_and_save_calibration(
            y_true=y_true,
            y_prob_pos=y_prob_pos,
            patient_ids=patient_ids,
            output_dir=calibration_dir,
            metadata={
                'n_folds': n_folds,
                'calibration_method': 'platt_scaling',
                'threshold_selection_method': 'youden_on_pooled_oof',
                'oof_method': 'single_model_per_fold',
            },
        )

        youden_threshold = cal_result.get('youden_threshold')

        # Save OOF predictions CSV with fold info
        import pandas as pd
        from src.evaluation.calibration import fit_platt_calibration
        _, calibrated_probs, _ = fit_platt_calibration(y_true, y_prob_pos)
        oof_df = pd.DataFrame({
            'patient_id': patient_ids,
            'fold': fold_ids,
            'label': y_true,
            'oof_probability': y_prob_pos,
            'calibrated_probability': calibrated_probs,
        })
        oof_path = calibration_dir / 'pooled_oof_predictions_with_folds.csv'
        oof_df.to_csv(oof_path, index=False)
        logger.info(f"✓ Saved OOF predictions (with fold info) to {oof_path}")
    else:
        logger.warning("⚠️  No OOF predictions available for calibration.")

    # Generate deployment threshold YAML
    project_root = Path(__file__).parent.parent
    deployment_yaml_path = project_root / 'configs' / 'evaluation' / 'operating_point' / 'deployment_threshold.yaml'
    deployment_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    if youden_threshold:
        clean_yaml_lines = [
            '# ============================================================================',
            '# AUTO-GENERATED DEPLOYMENT THRESHOLD CONFIGURATION',
            '# ============================================================================',
            f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Source: Internal {n_folds}-fold pooled OOF calibration (single-model, leak-free)",
            f"# Experiment: {exp_dir}",
            "# Selection method: Youden's J on Platt-calibrated pooled OOF predictions",
            '# DO NOT EDIT MANUALLY - Regenerate by running internal CV evaluation',
            '# ============================================================================',
            '',
            '# Frozen threshold from internal CV (on calibrated pooled OOF predictions)',
            'use_prespecified_threshold: true',
            f"prespecified_threshold_value: {youden_threshold['threshold']:.4f}",
            '',
            "# Youden's J operating point (LOCKED)",
            'youden_threshold:',
            f"  threshold: {youden_threshold['threshold']:.4f}",
            f"  sensitivity: {youden_threshold['sensitivity']:.4f}",
            f"  specificity: {youden_threshold['specificity']:.4f}",
            '',
            '# Metadata',
            'frozen_threshold_metadata:',
            '  source: internal_pooled_oof_calibration',
            '  selection_method: youden',
            f'  n_folds: {n_folds}',
        ]

        if per_fold_thresholds:
            clean_yaml_lines.extend([
                f'  per_fold_youden_values: {[float(t) for t in per_fold_thresholds]}',
                '  per_fold_statistics:',
                f'    median: {threshold_median:.4f}',
                f'    mean: {threshold_mean:.4f}',
                f'    std: {threshold_std:.4f}',
                f'    min: {threshold_min:.4f}',
                f'    max: {threshold_max:.4f}',
            ])

        clean_yaml_lines.extend([
            f"  created_at: '{datetime.now().isoformat()}'",
            f"  experiment_dir: '{exp_dir}'",
            '',
            '# Sensitivity analysis on external data',
            'compute_posthoc_optimal: true',
            'report_all_thresholds: true',
            '',
            '# These are ignored when using prespecified threshold',
            'method: prespecified',
            'selection_set: none',
            'per_fold_threshold: false',
        ])

        with open(deployment_yaml_path, 'w') as f:
            f.write('\n'.join(clean_yaml_lines))

        # Also save JSON version for programmatic access
        threshold_config = {
            'youden_threshold': youden_threshold,
            'aggregated_performance': aggregated
        }

        logger.info(f"\n✅ Frozen threshold config saved:")
        logger.info(f"   YAML: {deployment_yaml_path}")
        logger.info(f"   Youden's J: {youden_threshold['threshold']:.4f} "
                     f"(sens={youden_threshold['sensitivity']:.4f}, "
                     f"spec={youden_threshold['specificity']:.4f})")
        logger.info(f"\n📋 For held-out test cohort evaluation, run:")
        logger.info(f"   python scripts/evaluate.py \\")
        logger.info(f"     evaluation.held_out_test_cohort=true \\")
        logger.info(f"     evaluation.experiment_dir={exp_dir}")
        logger.info(f"\n⚠️  Remember: These are pre-specified - do NOT re-optimize on external data!\n")

        # Add threshold info to aggregated results
        aggregated['youden_threshold'] = youden_threshold
    else:
        logger.warning(
            "⚠️  No Youden threshold available from OOF-ensemble calibration."
        )

    # Save aggregated results
    output_dir = exp_dir / 'evaluation_aggregated'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results (convert numpy types to native Python)
    with open(output_dir / 'cv_evaluation_results.json', 'w') as f:
        json.dump(
            convert_to_serializable({
                'n_folds': n_folds,
                'aggregated_metrics': aggregated,
                'per_fold_results': all_fold_metrics
            }),
            f,
            indent=2)

    # Save JSON threshold config (convert numpy types to native Python)
    if youden_threshold:
        with open(output_dir / 'deployment_config.json', 'w') as f:
            json.dump(convert_to_serializable(threshold_config), f, indent=2)

    logger.info(f"\nAggregated results saved to: {output_dir}")

    return aggregated


def evaluate_single_model_held_out_test(cfg: Dict) -> Dict:
    """
    External validation: Evaluate single model on held-out test cohort (no ensemble).
    
    Use this for evaluating the full-training model on a held-out test cohort.
    Auto-discovers checkpoint from: {experiment_dir}/full_train/checkpoints/best_model.pth
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary of evaluation metrics
    """
    exp_dir = Path(cfg['evaluation']['experiment_dir'])
    
    logger.info("=" * 80)
    logger.info("SINGLE MODEL HELD-OUT TEST COHORT EVALUATION")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info("=" * 80)
    
    # Auto-discover checkpoint from full_train directory
    checkpoint_path = cfg['evaluation'].get('checkpoint_path', '')
    if not checkpoint_path:
        # Default to full_train checkpoint
        checkpoint_path = exp_dir / 'full_train' / 'checkpoints' / 'best_model.pth'
        if not checkpoint_path.exists():
            raise ValueError(
                f"No checkpoint found at {checkpoint_path}. "
                f"Run full training first: python scripts/train.py data/cross_validation=full_training"
            )
        checkpoint_path = str(checkpoint_path)
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Load model
    model, device = load_model_from_checkpoint(cfg, checkpoint_path)
    
    # Create data loader for held-out test cohort
    data_module = create_data_module(cfg)
    test_loader = data_module.get_test_loader(batch_size=1)
    
    logger.info(f"Held-out test cohort: {len(test_loader.dataset)} samples")
    
    # Create evaluator
    output_dir = exp_dir / 'held_out_test_evaluation'
    cfg_eval = cfg.copy()
    cfg_eval['evaluation']['output'] = cfg_eval['evaluation'].get('output', {})
    cfg_eval['evaluation']['output']['output_dir'] = str(output_dir)
    
    evaluator = ClassificationEvaluator(
        model=model,
        device=device,
        config=cfg_eval,
        is_mil_mode=False,
        checkpoint_path=checkpoint_path
    )
    
    # Generate predictions
    predictions = evaluator.predict(test_loader)
    
    # Extract probabilities and labels
    y_prob = predictions['probabilities']
    y_prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    y_true = predictions['labels']
    
    # Compute only threshold-free discrimination metrics (no Brier/ECE/calibration)
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    roc_auc = roc_auc_score(y_true, y_prob_pos)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob_pos)
    pr_auc = auc(recall_vals, precision_vals)
    
    # Bootstrap CIs for AUC metrics
    def _pr_auc_score(yt, yp):
        p, r, _ = precision_recall_curve(yt, yp)
        return auc(r, p)
    
    bootstrap_config = cfg.get('evaluation', {}).get('bootstrap', {})
    n_boot = bootstrap_config.get('n_iterations', 1000)
    rng = np.random.RandomState(42)
    roc_scores, pr_scores = [], []
    for _ in range(n_boot):
        # Stratified bootstrap
        indices = []
        for c in np.unique(y_true):
            c_idx = np.where(y_true == c)[0]
            indices.extend(rng.choice(c_idx, size=len(c_idx), replace=True))
        indices = np.array(indices)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            roc_scores.append(roc_auc_score(y_true[indices], y_prob_pos[indices]))
            pr_scores.append(_pr_auc_score(y_true[indices], y_prob_pos[indices]))
        except Exception:
            continue
    
    roc_ci = (np.percentile(roc_scores, 2.5), np.percentile(roc_scores, 97.5)) if roc_scores else (None, None)
    pr_ci = (np.percentile(pr_scores, 2.5), np.percentile(pr_scores, 97.5)) if pr_scores else (None, None)
    
    def _fmt_ci(ci):
        if ci[0] is not None and ci[1] is not None:
            return f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        return "N/A"
    
    # Save per-sample predictions CSV
    import pandas as pd
    output_dir = Path(cfg['evaluation']['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        'patient_id': predictions['patient_ids'],
        'true_label': y_true,
        'prob_class_0': y_prob[:, 0] if y_prob.ndim == 2 else 1 - y_prob,
        'prob_class_1': y_prob_pos,
    })
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Log results
    logger.info("=" * 80)
    logger.info("SINGLE MODEL HELD-OUT TEST RESULTS (threshold-free metrics only)")
    logger.info("=" * 80)
    logger.info(f"Dataset: Held-out test cohort  |  N={len(y_true)}")
    logger.info(f"ECRS: {int(np.sum(y_true == 1))}  |  non-ECRS: {int(np.sum(y_true == 0))}")
    logger.info("-" * 80)
    logger.info(f"ROC-AUC:  {roc_auc:.4f}  (95% CI: {_fmt_ci(roc_ci)})")
    logger.info(f"PR-AUC:   {pr_auc:.4f}  (95% CI: {_fmt_ci(pr_ci)})")
    logger.info("-" * 80)
    logger.info("⚠ No pre-specified threshold available for single model (uncalibrated).")
    logger.info("  Threshold-dependent metrics not reported.")
    logger.info("  Use the 5-fold ensemble for full evaluation.")
    logger.info("=" * 80)
    
    # Save JSON
    results_path = output_dir / 'held_out_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable({
            'threshold_free_metrics': {
                'roc_auc': roc_auc,
                'roc_auc_ci': [roc_ci[0], roc_ci[1]],
                'pr_auc': pr_auc,
                'pr_auc_ci': [pr_ci[0], pr_ci[1]],
            },
            'note': 'Single model trained on all data — no valid pre-specified threshold. Use ensemble for threshold-dependent metrics.',
            'n_samples': len(y_true),
            'class_distribution': {'ecrs': int(np.sum(y_true == 1)), 'non_ecrs': int(np.sum(y_true == 0))},
        }), f, indent=2)
    logger.info(f"Saved held-out test results JSON to {results_path}")
    logger.info(f"Saved predictions CSV to {output_dir / 'predictions.csv'}")
    
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}


def evaluate_ensemble(cfg: Dict) -> Dict:
    """
    External validation: Evaluate ensemble of all fold models with pooled calibration.
    
    Workflow:
    1. Load calibrator and calibrated threshold from internal CV
    2. Generate predictions from each fold model
    3. Average predictions (ensemble)
    4. Apply calibration to ensemble probabilities
    5. Evaluate using the calibrated threshold (LOCKED from internal CV)
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dictionary of evaluation metrics for ensemble
    """
    import joblib
    
    cv_config = cfg['data']['cross_validation']
    n_folds = 1 if not cv_config['enabled'] else cv_config['n_folds']
    exp_dir = Path(cfg['evaluation']['experiment_dir'])
    
    logger.info("=" * 80)
    logger.info(f"EXTERNAL VALIDATION: {n_folds} fold models + Calibrated Ensemble")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info("=" * 80)
    
    # Load calibrator and Youden threshold from internal CV
    calibration_dir = exp_dir / 'calibration'
    calibrator_path = calibration_dir / 'calibrator.pkl'
    calibration_config_path = calibration_dir / 'calibration_config.yaml'
    
    calibrator = None
    calibrated_threshold = 0.5  # Default
    youden_threshold = None
    
    if calibrator_path.exists() and calibration_config_path.exists():
        calibrator = joblib.load(calibrator_path)
        with open(calibration_config_path) as f:
            cal_config = yaml.safe_load(f)
            calibrated_threshold = cal_config['calibrated_threshold']
            youden_threshold = cal_config.get('youden_threshold')
        logger.info(f"✓ Loaded calibrator from {calibrator_path}")
        logger.info(f"✓ Youden's J threshold (LOCKED from internal CV): {calibrated_threshold:.4f}")
    else:
        logger.warning("⚠ No calibrator found. Run internal CV training first to generate calibration.")
        logger.warning("  Falling back to uncalibrated evaluation with threshold=0.5")
    
    # Find all fold checkpoints
    fold_info = []
    for fold in range(n_folds):
        checkpoint_path = exp_dir / f'fold_{fold + 1}' / 'checkpoints' / 'best_model.pth'
        if checkpoint_path.exists():
            fold_info.append({
                'fold': fold,
                'checkpoint_path': str(checkpoint_path),
            })
            logger.info(f"  ✓ Fold {fold + 1}: checkpoint found")
        else:
            logger.warning(f"  ✗ Missing fold {fold + 1}: {checkpoint_path}")
    
    if not fold_info:
        raise ValueError(f"No checkpoints found in {exp_dir}")
    
    logger.info(f"\nUsing {len(fold_info)}/{n_folds} fold checkpoints")
    
    # Create data loader for external test set
    data_module = create_data_module(cfg)
    test_loader = data_module.get_test_loader(batch_size=1)
    
    # Collect raw predictions from all folds
    all_fold_raw_probs = []
    
    for info in fold_info:
        fold_idx = info['fold']
        checkpoint_path = info['checkpoint_path']
        
        logger.info(f"Generating predictions from fold {fold_idx + 1}...")
        
        # Load model
        model, device = load_model_from_checkpoint(cfg, checkpoint_path)
        
        # Create evaluator
        evaluator = ClassificationEvaluator(
            model=model,
            device=device,
            config=cfg,
            is_mil_mode=False,
            checkpoint_path=checkpoint_path
        )
        
        # Generate predictions
        predictions = evaluator.predict(test_loader)
        
        # Extract positive class probability
        probs = predictions['probabilities']
        if probs.ndim == 2 and probs.shape[1] == 2:
            prob_pos = probs[:, 1]
        else:
            prob_pos = probs
        
        all_fold_raw_probs.append(prob_pos)
        logger.info(f"  Fold {fold_idx + 1}: raw prob range [{prob_pos.min():.4f}, {prob_pos.max():.4f}]")
    
    # Store labels/patient_ids from last fold (same for all)
    first_predictions = {'labels': predictions['labels'], 'patient_ids': predictions.get('patient_ids', [])}
    
    # Ensemble: Average RAW probabilities FIRST, then calibrate ONCE
    # This matches how calibrator was trained on OOF-ensemble predictions
    logger.info("=" * 80)
    logger.info("ENSEMBLE RESULTS (Average-then-Calibrate)")
    logger.info("=" * 80)
    
    # Average raw probabilities across folds
    ensemble_raw_prob = np.mean(all_fold_raw_probs, axis=0)
    logger.info(f"Raw ensemble prob range: [{ensemble_raw_prob.min():.4f}, {ensemble_raw_prob.max():.4f}]")
    
    # Apply calibration ONCE to ensemble (matches training procedure)
    user_threshold = cfg['evaluation'].get('threshold', None)
    
    if calibrator is not None:
        eps = 1e-15
        prob_clipped = np.clip(ensemble_raw_prob, eps, 1 - eps)
        logits = np.log(prob_clipped / (1 - prob_clipped))
        ensemble_calibrated_prob = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        
        logger.info(f"Calibrated ensemble prob range: [{ensemble_calibrated_prob.min():.4f}, {ensemble_calibrated_prob.max():.4f}]")
        final_prob_pos = ensemble_calibrated_prob
    else:
        final_prob_pos = ensemble_raw_prob
        logger.info("No calibrator available — using raw ensemble probabilities")
    
    # Reconstruct 2-class probability array for evaluator
    final_probs = np.column_stack([1 - final_prob_pos, final_prob_pos])
    y_true = first_predictions['labels']
    
    # ---- Determine primary operating point (Youden's J) ----
    if youden_threshold:
        primary_threshold = youden_threshold['threshold']
    else:
        primary_threshold = calibrated_threshold
    
    # User override
    if user_threshold is not None:
        primary_threshold = user_threshold
        logger.info(f"Threshold: {primary_threshold:.4f} (user-specified override)")
    else:
        logger.info(f"Threshold: {primary_threshold:.4f} (Youden's J, LOCKED from internal CV)")
    
    # Create dedicated evaluator for ensemble
    ensemble_cfg = cfg.copy()
    ensemble_output_dir = exp_dir / 'ensemble_evaluation'
    ensemble_cfg['evaluation']['output']['output_dir'] = str(ensemble_output_dir)
    
    ensemble_evaluator = ClassificationEvaluator(
        model=model,
        device=device,
        config=ensemble_cfg,
        is_mil_mode=False,
        checkpoint_path='ensemble'
    )
    
    # ---- Evaluate at primary threshold (with full reporting) ----
    primary_predictions = {
        'labels': y_true,
        'patient_ids': first_predictions['patient_ids'],
        'probabilities': final_probs,
        'predictions': (final_prob_pos >= primary_threshold).astype(int),
    }
    primary_metrics = ensemble_evaluator.evaluate(primary_predictions, save_results=True, threshold=primary_threshold)
    
    # ---- Helper: compute metrics at a given threshold ----
    def _metrics_at_threshold(t):
        preds = (final_prob_pos >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return {
            'threshold': round(t, 4), 'sensitivity': round(sens, 4),
            'specificity': round(spec, 4), 'ppv': round(ppv, 4),
            'npv': round(npv, 4), 'accuracy': round(acc, 4), 'f1_score': round(f1, 4),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }
    
    # ---- Sensitivity analysis: fixed thresholds from 0.30 to 0.70 ----
    sensitivity_thresholds = [round(x * 0.05 + 0.30, 2) for x in range(9)]  # 0.30, 0.35, ..., 0.70
    # Ensure primary (Youden) threshold is included
    if primary_threshold not in sensitivity_thresholds:
        sensitivity_thresholds.append(round(primary_threshold, 4))
        sensitivity_thresholds.sort()
    
    sensitivity_analysis = []
    for t in sensitivity_thresholds:
        row = _metrics_at_threshold(t)
        row['is_primary'] = abs(t - primary_threshold) < 1e-6
        sensitivity_analysis.append(row)
    
    # ---- Log primary results ----
    logger.info("=" * 80)
    logger.info("HELD-OUT TEST RESULTS — PRIMARY OPERATING POINT (Youden's J)")
    logger.info("=" * 80)
    logger.info(f"Dataset: Held-out test cohort  |  N={len(y_true)}  |  Ensemble: {len(fold_info)} folds")
    logger.info(f"ROC-AUC: {primary_metrics.get('roc_auc', 'N/A'):.4f}  |  PR-AUC: {primary_metrics.get('pr_auc', 'N/A'):.4f}")
    logger.info(f"Brier Score: {primary_metrics.get('brier_score', 'N/A'):.4f}  |  ECE: {primary_metrics.get('ece', 'N/A'):.4f}")
    pm = _metrics_at_threshold(primary_threshold)
    logger.info(f"Threshold: {primary_threshold:.4f}  |  Sens: {pm['sensitivity']:.4f}  |  Spec: {pm['specificity']:.4f}")
    logger.info(f"PPV: {pm['ppv']:.4f}  |  NPV: {pm['npv']:.4f}  |  F1: {pm['f1_score']:.4f}  |  Acc: {pm['accuracy']:.4f}")
    logger.info("=" * 80)
    
    # ---- Log sensitivity analysis table ----
    logger.info("")
    logger.info("SENSITIVITY ANALYSIS — Performance Across Thresholds")
    logger.info("=" * 96)
    logger.info(f"{'Threshold':>10} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'NPV':>8} {'F1':>8} {'Acc':>8}   {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    logger.info("-" * 96)
    
    for row in sensitivity_analysis:
        marker = " ◀ Youden" if row['is_primary'] else ""
        logger.info(
            f"{row['threshold']:>10.4f} {row['sensitivity']:>8.4f} {row['specificity']:>8.4f} "
            f"{row['ppv']:>8.4f} {row['npv']:>8.4f} {row['f1_score']:>8.4f} {row['accuracy']:>8.4f}   "
            f"{row['tp']:>4} {row['tn']:>4} {row['fp']:>4} {row['fn']:>4}{marker}"
        )
    
    logger.info("-" * 96)
    logger.info(f"N = {len(y_true)} (ECRS: {int(np.sum(y_true == 1))}, non-ECRS: {int(np.sum(y_true == 0))})")
    logger.info("=" * 96)
    
    # ---- Save results ----
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sensitivity analysis CSV (paper-ready)
    import pandas as pd
    sa_df = pd.DataFrame(sensitivity_analysis)
    sa_csv_path = ensemble_output_dir / 'sensitivity_analysis.csv'
    sa_df.to_csv(sa_csv_path, index=False)
    logger.info(f"Saved sensitivity analysis CSV to {sa_csv_path}")
    
    # Save per-patient held-out test predictions CSV
    held_out_test_pred_df = pd.DataFrame({
        'patient_id': first_predictions['patient_ids'],
        'true_label': y_true,
        'predicted_label': (final_prob_pos >= primary_threshold).astype(int),
        'prob_class_0': 1 - final_prob_pos,
        'prob_class_1': final_prob_pos,
    })
    held_out_test_pred_path = ensemble_output_dir / 'held_out_test_predictions.csv'
    held_out_test_pred_df.to_csv(held_out_test_pred_path, index=False)
    logger.info(f"Saved held-out test predictions CSV to {held_out_test_pred_path}")
    
    # Save comprehensive JSON
    results_path = ensemble_output_dir / 'held_out_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable({
            'threshold_free_metrics': {
                'roc_auc': primary_metrics.get('roc_auc'),
                'pr_auc': primary_metrics.get('pr_auc'),
                'average_precision': primary_metrics.get('average_precision'),
                'brier_score': primary_metrics.get('brier_score'),
                'ece': primary_metrics.get('ece'),
            },
            'primary_operating_point': {
                'method': 'youden_j',
                **pm,
            },
            'sensitivity_analysis': sensitivity_analysis,
            'n_samples': len(y_true),
            'class_distribution': {'ecrs': int(np.sum(y_true == 1)), 'non_ecrs': int(np.sum(y_true == 0))},
            'ensemble_size': len(fold_info),
            'calibrated': calibrator is not None,
        }), f, indent=2)
    logger.info(f"Saved held-out test results JSON to {results_path}")
    

    # Add info to primary metrics for backward compatibility
    primary_metrics['calibrated'] = calibrator is not None
    primary_metrics['calibrated_threshold'] = primary_threshold
    primary_metrics['sensitivity_analysis'] = sensitivity_analysis
    
    return primary_metrics


@hydra.main(version_base=None,
            config_path="pkg://configs",
            config_name="config_3d")
def main(cfg: DictConfig):
    """
    Main evaluation function.
    
    Supports two modes:
    1. Internal CV evaluation (all folds or specific fold)
    2. External validation (ensemble of all folds on held-out test cohort)
    """

    logger.info("=" * 80)
    logger.info("3D CLASSIFICATION EVALUATION")
    logger.info("=" * 80)

    cfg_dict: Dict[str,
                   Any] = OmegaConf.to_container(cfg,
                                                 resolve=True)  # type: ignore

    # Preprocess CSV: Convert specific image type column to generic 'image_path'
    # This handles the mapping from full_head_ct_path, mask_bbox_margin_path, etc.
    preprocessed_csv_path = prepare_csv_for_training(cfg_dict)
    cfg_dict['data']['paths']['csv_path'] = preprocessed_csv_path
    logger.info(f"Using preprocessed CSV: {preprocessed_csv_path}")
    logger.info(f"Image type: {cfg_dict['data']['image_type']}")

    # Check evaluation mode
    evaluation_config = cfg_dict['evaluation']
    held_out_test_cohort = evaluation_config.get('held_out_test_cohort', False)
    use_single_model = evaluation_config.get('use_single_model', False)
    evaluate_all = evaluation_config.get('evaluate_all_folds', False)

    # For held-out test cohort evaluation: swap CSV
    if held_out_test_cohort:
        ext_test_cfg = cfg_dict['data'].get('external_test', {})
        ext_csv = ext_test_cfg.get('csv_path', '')
        if ext_csv:
            logger.info(f"Held-out test cohort evaluation: switching CSV to {ext_csv}")
            cfg_dict['data']['paths']['csv_path'] = ext_csv
            preprocessed_csv_path = prepare_csv_for_training(cfg_dict)
            cfg_dict['data']['paths']['csv_path'] = preprocessed_csv_path
            logger.info(f"Using preprocessed external CSV: {preprocessed_csv_path}")
        else:
            logger.warning("⚠ held_out_test_cohort=true but no data.external_test.csv_path configured!")

    if held_out_test_cohort and use_single_model:
        # Mode 0: Single model held-out test evaluation (for full training)
        logger.info("Mode: SINGLE MODEL HELD-OUT TEST EVALUATION")

        if 'experiment_dir' not in evaluation_config:
            raise ValueError(
                "Please specify evaluation.experiment_dir when use_single_model=true"
            )

        metrics = evaluate_single_model_held_out_test(cfg_dict)

        # Summary already printed by evaluate_single_model_held_out_test

    elif held_out_test_cohort:
        # Mode 1: Held-out test cohort evaluation (ensemble of all folds)
        logger.info("Mode: HELD-OUT TEST COHORT EVALUATION (ensemble)")

        if 'experiment_dir' not in evaluation_config:
            raise ValueError(
                "Please specify evaluation.experiment_dir when held_out_test_cohort=true"
            )

        metrics = evaluate_ensemble(cfg_dict)

        # Summary already printed by evaluate_ensemble

    elif evaluate_all:
        # Mode 2a: Internal CV evaluation (all folds)
        logger.info("Mode: INTERNAL CV EVALUATION (all folds)")

        if 'experiment_dir' not in evaluation_config:
            raise ValueError(
                "Please specify evaluation.experiment_dir when evaluate_all_folds=true"
            )

        aggregated_metrics = evaluate_all_folds(cfg_dict)

        # Print aggregated summary
        logger.info("=" * 80)
        logger.info("INTERNAL CV EVALUATION SUMMARY")
        logger.info("=" * 80)
        for metric_name, values in aggregated_metrics.items():
            if not isinstance(values, dict) or 'mean' not in values:
                continue
            mean = values['mean']
            std = values['std']
            logger.info(f"{metric_name}: {mean:.4f} ± {std:.4f}")
        logger.info("=" * 80)

    else:
        # Mode 2b: Internal CV evaluation (specific fold)
        fold_num = evaluation_config.get('fold', 0)
        logger.info(f"Mode: INTERNAL CV EVALUATION (fold {fold_num + 1})")

        checkpoint_path = evaluation_config.get('checkpoint_path', '')
        if not checkpoint_path:
            logger.info(
                "No checkpoint_path specified - will auto-discover from experiment_dir"
            )

        metrics = evaluate_single_fold(cfg_dict, fold=fold_num)

        # Print summary
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        logger.info(f"PR-AUC: {metrics.get('pr_auc', 'N/A')}")
        logger.info(f"Sensitivity: {metrics.get('sensitivity', 'N/A')}")
        logger.info(f"Specificity: {metrics.get('specificity', 'N/A')}")
        logger.info(f"F1 Score: {metrics.get('f1_score', 'N/A')}")
        logger.info(f"Brier Score: {metrics.get('brier_score', 'N/A')}")
        logger.info("=" * 80)

    logger.info("=" * 80)
    logger.info("✅ Evaluation completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
