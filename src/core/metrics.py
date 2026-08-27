"""
Comprehensive Evaluation Metrics for Medical Image Classification
Metrics for binary clinical prediction models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def select_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """
    Select the optimal operating point via Youden's J statistic.

    Youden's J = max(sensitivity + specificity - 1) over all thresholds.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary with 'threshold', 'sensitivity', 'specificity'.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1 - fpr

    j_scores = tpr - fpr
    youden_idx = np.argmax(j_scores)

    result = {
        'threshold': float(thresholds[youden_idx]),
        'sensitivity': float(tpr[youden_idx]),
        'specificity': float(spec[youden_idx]),
    }

    logger.info(
        f"Youden's J threshold → {result['threshold']:.4f} "
        f"(sens={result['sensitivity']:.4f}, spec={result['specificity']:.4f})"
    )

    return result


class ComprehensiveMetrics:
    """
    Comprehensive evaluation metrics for medical image classification.
    
    Includes:
    - Primary: ROC-AUC, PR-AUC, Average Precision (with 95% CI via bootstrapping)
    - Secondary: Sensitivity, Specificity, F1, Balanced Accuracy
    - Calibration: Brier Score, Expected Calibration Error (ECE)
    
    for publication-quality figures. This module focuses on metric computation only.
    """

    def __init__(self, config: Dict, is_training_mode: bool = None):
        """
        Args:
            config: Metrics configuration from YAML
            is_training_mode: Override for evaluation mode.
                            - True: Fast mode (no bootstrap CIs) for training validation
                            - False: Full mode (with CIs) for final evaluation
                            - None: Auto-detect from config or context
        """
        self.config = config

        # Auto-detect evaluation mode
        if is_training_mode is not None:
            # Explicit override from caller (trainer or evaluator)
            self.is_training_mode = is_training_mode
            self.evaluation_mode = 'training' if is_training_mode else 'final'
            logger.info(
                f"📊 Evaluation mode: {'TRAINING' if is_training_mode else 'FINAL'} "
                f"(auto-detected from caller context)")
        else:
            # Read from config (backward compatible)
            self.evaluation_mode = config.get('evaluation_mode', 'auto')

            if self.evaluation_mode == 'auto':
                # Auto-detect: assume 'final' mode for standalone evaluation
                # (training explicitly passes is_training_mode=True)
                self.is_training_mode = False
                logger.info(
                    "📊 Evaluation mode: AUTO → FINAL (bootstrap CIs enabled)")
            else:
                self.is_training_mode = (self.evaluation_mode == 'training')
                if self.is_training_mode:
                    logger.info(
                        "📊 Evaluation mode: TRAINING (bootstrap CIs disabled for speed)"
                    )
                else:
                    logger.info(
                        "📊 Evaluation mode: FINAL (bootstrap CIs enabled for publication)"
                    )

        # Get evaluation config
        if 'evaluation' not in config:
            raise KeyError(f"Cannot find 'evaluation' in config. "
                           f"Available keys: {list(config.keys())}. "
                           f"Ensure evaluation config is present.")

        self.metrics_config = config['evaluation']

        # Get metrics subconfig if it exists (from metrics.yaml)
        if 'metrics' in config:
            self.metrics_config.update(config['metrics'])

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True labels [N]
            y_prob: Predicted probabilities [N, C] or [N] for binary
            threshold: Classification threshold (computed if None)
            
        Returns:
            Dictionary of all metrics with confidence intervals
        """
        results = {}

        # Validate binary classification (only supported for now)
        if y_prob.ndim == 2:
            if y_prob.shape[1] != 2:
                raise ValueError(f"Only binary classification supported. Got {y_prob.shape[1]} classes. "
                                 "For multi-class, implement one-vs-rest strategy.")
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob

        # Held-out test cohort guardrail: enforce frozen threshold
        is_external = self.config.get('evaluation', {}).get('held_out_test_cohort', False)
        if is_external and threshold is None:
            op_config = self.metrics_config.get('operating_point', {})
            if isinstance(op_config, dict) and not op_config.get('use_prespecified_threshold', False):
                raise ValueError(
                    "External validation requires use_prespecified_threshold=true "
                    "(do not optimize threshold on external data). "
                    "Either pass threshold= argument or set operating_point.use_prespecified_threshold=true"
                )

        # Select threshold if not provided
        if threshold is None:
            # Check if using pre-specified threshold (for external validation)
            op_config = self.metrics_config['operating_point']
            
            # Handle case where op_config is a string (config name) vs dict (resolved config)
            if isinstance(op_config, str):
                # Load the config file from configs/evaluation/operating_point/{name}.yaml
                # Use Hydra's original CWD if available (handles Hydra run directories)
                try:
                    import hydra
                    original_cwd = hydra.utils.get_original_cwd()
                except (ImportError, ValueError):
                    original_cwd = '.'
                config_path = Path(original_cwd) / 'configs/evaluation/operating_point' / f'{op_config}.yaml'
                if config_path.exists():
                    with open(config_path) as f:
                        op_config = yaml.safe_load(f)
                    logger.info(f"Loaded operating_point config from {config_path}")
                else:
                    logger.warning(f"Operating point config not found: {config_path}")
                    op_config = {}
            
            use_prespecified = op_config.get('use_prespecified_threshold', False)

            if use_prespecified:
                threshold = op_config['prespecified_threshold_value']
                if threshold is None:
                    logger.warning(
                        "⚠️ use_prespecified_threshold=true but no threshold value provided. "
                        "Falling back to threshold selection on current data.")
                    threshold = self._select_threshold(y_true, y_prob_pos)
                    threshold_source = 'selected_fallback'
                else:
                    logger.info("=" * 80)
                    logger.info("FROZEN THRESHOLD (Pre-specified from Internal CV)")
                    logger.info("=" * 80)
                    logger.info(f"Using frozen threshold: {threshold:.4f}")
                    logger.info("⚠️  This is PROSPECTIVE validation - no threshold optimization on this data")
                    logger.info("=" * 80)
                    threshold_source = 'prespecified'
            else:
                threshold = self._select_threshold(y_true, y_prob_pos)
                threshold_source = 'selected'
        else:
            # Threshold was provided as argument
            threshold_source = 'provided'

        results['threshold'] = threshold
        results['threshold_source'] = threshold_source

        # Update predictions based on threshold
        y_pred = (y_prob_pos >= threshold).astype(int)

        # Primary metrics
        results.update(self._compute_primary_metrics(y_true, y_prob_pos))

        # Secondary metrics
        results.update(self._compute_secondary_metrics(y_true, y_pred))

        # Calibration metrics
        results.update(self._compute_calibration_metrics(y_true, y_prob_pos))

        # Per-class metrics
        # Per-class metrics
        if self.metrics_config['discrimination']['per_class']['enabled']:
            results['per_class'] = self._compute_per_class_metrics(
                y_true, y_pred)
        else:
            results['per_class'] = {}

        # Confusion matrix
        results['confusion_matrix'] = self._compute_confusion_matrix(
            y_true, y_pred)

        logger.info(f"Computed metrics at threshold={threshold:.4f}")

        return results

    def _select_threshold(self, y_true: np.ndarray,
                          y_prob: np.ndarray) -> float:
        """
        Select optimal threshold using configured method.
        
        Reads from: evaluation.operating_point
        """
        # Read from operating_point config (single source of truth)
        threshold_config = self.metrics_config['operating_point']

        # Handle case where threshold_config is a string (config name) vs dict
        if isinstance(threshold_config, str):
            # Use Hydra's original CWD if available (handles Hydra run directories)
            try:
                import hydra
                original_cwd = hydra.utils.get_original_cwd()
            except (ImportError, ValueError):
                original_cwd = '.'
            config_path = Path(original_cwd) / 'configs/evaluation/operating_point' / f'{threshold_config}.yaml'
            if config_path.exists():
                with open(config_path) as f:
                    threshold_config = yaml.safe_load(f)
                logger.info(f"Loaded operating_point config from {config_path}")
            else:
                threshold_config = {}
        
        method = threshold_config.get('method', 'youden')

        if method == "youden":
            # Youden's J statistic: max(sensitivity + specificity - 1)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = thresholds[optimal_idx]
            logger.info(f"Youden's index: {j_scores[optimal_idx]:.4f}")

        elif method == "f1_optimal":
            # Maximize F1 score
            precision, recall, thresholds = precision_recall_curve(
                y_true, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]
            logger.info(f"Optimal F1 score: {f1_scores[optimal_idx]:.4f}")

        elif method == "sens_at_spec":
            # Maximize sensitivity at target specificity
            target_spec = threshold_config['target_specificity']
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            spec = 1 - fpr
            # Find threshold where specificity >= target
            valid_idx = np.where(spec >= target_spec)[0]
            if len(valid_idx) > 0:
                optimal_idx = valid_idx[np.argmax(tpr[valid_idx])]
                threshold = thresholds[optimal_idx]
                actual_sens = tpr[optimal_idx]
                actual_spec = spec[optimal_idx]
                logger.info(
                    f"Target spec: {target_spec:.2f}, Achieved: sens={actual_sens:.4f}, spec={actual_spec:.4f}"
                )
            else:
                threshold = 0.5
                logger.warning(
                    f"Could not achieve target specificity {target_spec:.2f}, using default 0.5"
                )

        elif method == "spec_at_sens":
            # Maximize specificity at target sensitivity
            target_sens = threshold_config['target_sensitivity']
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            spec = 1 - fpr
            # Find threshold where sensitivity >= target
            valid_idx = np.where(tpr >= target_sens)[0]
            if len(valid_idx) > 0:
                optimal_idx = valid_idx[np.argmax(spec[valid_idx])]
                threshold = thresholds[optimal_idx]
                actual_sens = tpr[optimal_idx]
                actual_spec = spec[optimal_idx]
                logger.info(
                    f"Target sens: {target_sens:.2f}, Achieved: sens={actual_sens:.4f}, spec={actual_spec:.4f}"
                )
            else:
                threshold = 0.5
                logger.warning(
                    f"Could not achieve target sensitivity {target_sens:.2f}, using default 0.5"
                )
        else:
            threshold = 0.5
            logger.warning(
                f"Unknown threshold method '{method}', using default 0.5")

        logger.info(
            f"Selected threshold: {threshold:.4f} using method: {method}")

        # Validate selection_set
        selection_set = threshold_config.get('selection_set')
        if selection_set and selection_set != 'validation':
            logger.warning(
                f"⚠️  Threshold selected on '{selection_set}' set. "
                "For unbiased evaluation, use 'validation' (not 'test'!).")

        return threshold

    def _compute_primary_metrics(self, y_true: np.ndarray,
                                 y_prob: np.ndarray) -> Dict:
        """Compute primary metrics with bootstrap confidence intervals."""
        results = {}

        # ROC-AUC
        # ROC-AUC
        if self.metrics_config['discrimination']['primary']['roc_auc'][
                'enabled']:
            roc_auc = roc_auc_score(y_true, y_prob)
            results['roc_auc'] = roc_auc

            # Only compute bootstrap CI in final evaluation mode
            if not self.is_training_mode and self.metrics_config[
                    'discrimination']['primary']['roc_auc']['bootstrap_ci']:
                ci_lower, ci_upper = self._bootstrap_ci(y_true, y_prob,
                                                        roc_auc_score)
                results['roc_auc_ci_lower'] = ci_lower
                results['roc_auc_ci_upper'] = ci_upper
                logger.info(
                    f"ROC-AUC: {roc_auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )

        # PR-AUC
        # PR-AUC
        if self.metrics_config['discrimination']['primary']['pr_auc'][
                'enabled']:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            results['pr_auc'] = pr_auc

            # Only compute bootstrap CI in final evaluation mode
            if not self.is_training_mode and self.metrics_config[
                    'discrimination']['primary']['pr_auc']['bootstrap_ci']:

                def pr_auc_score(y_true, y_prob):
                    precision, recall, _ = precision_recall_curve(
                        y_true, y_prob)
                    return auc(recall, precision)

                ci_lower, ci_upper = self._bootstrap_ci(y_true, y_prob,
                                                        pr_auc_score)
                results['pr_auc_ci_lower'] = ci_lower
                results['pr_auc_ci_upper'] = ci_upper
                logger.info(
                    f"PR-AUC: {pr_auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )

        # Average Precision (AP) - commonly reported in medical ML papers
        ap = average_precision_score(y_true, y_prob)
        results['average_precision'] = ap

        return results

    def _compute_secondary_metrics(self, y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict:
        """
        Compute secondary metrics at selected threshold with bootstrap CIs.
        
        Reported at the selected threshold:
        - Sensitivity (True Positive Rate)
        - Specificity (True Negative Rate)
        - PPV (Positive Predictive Value)
        - NPV (Negative Predictive Value)
        """
        results = {}

        # Confusion matrix components (force label order to handle missing classes)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        secondary_config = self.metrics_config['discrimination']['secondary']

        # Sensitivity (True Positive Rate)
        if secondary_config['sensitivity']['enabled']:
            sensitivity = recall_score(y_true, y_pred)
            results['sensitivity'] = sensitivity
            results['recall'] = sensitivity  # Alias for ML compatibility
        else:
            sensitivity = recall_score(y_true, y_pred)  # Compute for logging

        if self._should_compute_bootstrap_ci('secondary', 'sensitivity'):
            ci_lower, ci_upper = self._bootstrap_ci_for_predictions(
                y_true, y_pred,
                lambda yt, yp: recall_score(yt, yp, zero_division=0))
            results['sensitivity_ci_lower'] = ci_lower
            results['sensitivity_ci_upper'] = ci_upper
            results['recall_ci_lower'] = ci_lower  # Alias
            results['recall_ci_upper'] = ci_upper  # Alias

        # Specificity (True Negative Rate)
        # Specificity (True Negative Rate)
        if secondary_config['specificity']['enabled']:
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            results['specificity'] = specificity
        else:
            specificity = tn / (tn + fp) if (
                tn + fp) > 0 else 0.0  # Compute for logging

        if self._should_compute_bootstrap_ci('secondary', 'specificity'):

            def spec_fn(yt, yp):
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0

            ci_lower, ci_upper = self._bootstrap_ci_for_predictions(
                y_true, y_pred, spec_fn)
            results['specificity_ci_lower'] = ci_lower
            results['specificity_ci_upper'] = ci_upper

        # PPV (Positive Predictive Value)
        # PPV (Positive Predictive Value)
        if secondary_config['ppv']['enabled']:
            ppv = precision_score(y_true, y_pred, zero_division=0)
            results['ppv'] = ppv
            results['precision'] = ppv  # Alias for ML compatibility
        else:
            ppv = precision_score(y_true, y_pred,
                                  zero_division=0)  # Compute for logging

        if self._should_compute_bootstrap_ci('secondary', 'ppv'):
            ci_lower, ci_upper = self._bootstrap_ci_for_predictions(
                y_true, y_pred,
                lambda yt, yp: precision_score(yt, yp, zero_division=0))
            results['ppv_ci_lower'] = ci_lower
            results['ppv_ci_upper'] = ci_upper
            results['precision_ci_lower'] = ci_lower  # Alias
            results['precision_ci_upper'] = ci_upper  # Alias

        # NPV (Negative Predictive Value)
        # NPV (Negative Predictive Value)
        if secondary_config['npv']['enabled']:
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            results['npv'] = npv
        else:
            npv = tn / (tn + fn) if (tn +
                                     fn) > 0 else 0.0  # Compute for logging

        if self._should_compute_bootstrap_ci('secondary', 'npv'):

            def npv_fn(yt, yp):
                tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
                return tn / (tn + fn) if (tn + fn) > 0 else 0.0

            ci_lower, ci_upper = self._bootstrap_ci_for_predictions(
                y_true, y_pred, npv_fn)
            results['npv_ci_lower'] = ci_lower
            results['npv_ci_upper'] = ci_upper

        # F1 Score
        # F1 Score
        if secondary_config['f1_score']['enabled']:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            results['f1_score'] = f1
        else:
            f1 = f1_score(y_true, y_pred,
                          zero_division=0)  # Compute for logging

        if self._should_compute_bootstrap_ci('secondary', 'f1_score'):
            ci_lower, ci_upper = self._bootstrap_ci_for_predictions(
                y_true, y_pred,
                lambda yt, yp: f1_score(yt, yp, zero_division=0))
            results['f1_score_ci_lower'] = ci_lower
            results['f1_score_ci_upper'] = ci_upper

        # Balanced Accuracy
        # Balanced Accuracy
        if secondary_config['balanced_accuracy']['enabled']:
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            results['balanced_accuracy'] = balanced_acc
        else:
            balanced_acc = 0.0  # Not computed

        if self._should_compute_bootstrap_ci('secondary', 'balanced_accuracy'):
            ci_lower, ci_upper = self._bootstrap_ci_for_predictions(
                y_true, y_pred, balanced_accuracy_score)
            results['balanced_accuracy_ci_lower'] = ci_lower
            results['balanced_accuracy_ci_upper'] = ci_upper

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        results['accuracy'] = accuracy

        logger.info(
            f"Medical metrics - Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, PPV: {ppv:.4f}, NPV: {npv:.4f}, F1: {f1:.4f}"
        )

        return results

    def _should_compute_bootstrap_ci(self, category: str, metric: str) -> bool:
        """
        Check if bootstrap CI should be computed for a metric.
        
        Returns False during training mode (for speed).
        Only computes CIs in final evaluation mode.
        """
        # Skip bootstrap CIs during training
        if self.is_training_mode:
            return False

        metric_config = self.metrics_config['discrimination'][category][metric]
        return isinstance(metric_config, dict) and metric_config['bootstrap_ci']

    def _bootstrap_ci_for_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_fn,
        n_iterations: Optional[int] = None,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for metrics based on predictions.
        Similar to _bootstrap_ci but works with y_pred instead of y_prob.
        """
        bootstrap_config = self.metrics_config['bootstrap']
        if n_iterations is None:
            n_iterations = bootstrap_config['n_iterations']

        stratified = bootstrap_config['stratified']
        # Use bootstrap seed if specified, otherwise fall back to global seed
        random_seed = bootstrap_config['random_seed']
        if random_seed is None:
            random_seed = self.config['reproducibility']['random_seed']
        method = bootstrap_config['method']

        rng = np.random.RandomState(random_seed)
        scores = []

        n_samples = len(y_true)

        for _ in range(n_iterations):
            if stratified:
                indices = []
                for class_label in np.unique(y_true):
                    class_indices = np.where(y_true == class_label)[0]
                    class_sample = rng.choice(class_indices,
                                              size=len(class_indices),
                                              replace=True)
                    indices.extend(class_sample)
                indices = np.array(indices)
                rng.shuffle(indices)
            else:
                indices = rng.choice(n_samples, size=n_samples, replace=True)

            if len(np.unique(y_true[indices])) < 2:
                continue

            try:
                score = metric_fn(y_true[indices], y_pred[indices])
                scores.append(score)
            except Exception:
                continue

        # Guard against empty scores (can happen with small/imbalanced data)
        if len(scores) == 0:
            logger.warning("Bootstrap produced no valid resamples; returning NaNs for CI")
            return (float("nan"), float("nan"))

        # Compute confidence interval using specified method
        alpha = (1 - confidence_level) / 2

        if method == 'percentile':
            ci_lower = np.percentile(scores, alpha * 100)
            ci_upper = np.percentile(scores, (1 - alpha) * 100)
        elif method == 'bca':
            # Bias-corrected and accelerated (BCa) method
            logger.warning("BCa method not fully implemented, using percentile")
            ci_lower = np.percentile(scores, alpha * 100)
            ci_upper = np.percentile(scores, (1 - alpha) * 100)
        else:
            # Default to percentile
            ci_lower = np.percentile(scores, alpha * 100)
            ci_upper = np.percentile(scores, (1 - alpha) * 100)

        return ci_lower, ci_upper

    def _compute_calibration_metrics(self, y_true: np.ndarray,
                                     y_prob: np.ndarray) -> Dict:
        """Compute calibration metrics."""
        results = {}

        # Get calibration config (check both locations for backward compatibility)
        # Primary: evaluation.calibration (analysis.yaml)
        # Fallback: evaluation.metrics.calibration (old location)
        # Get calibration config (check both locations for backward compatibility)
        # Primary: evaluation.calibration (analysis.yaml)
        # Fallback: evaluation.metrics.calibration (old location)
        calib_config = self.metrics_config['calibration']

        # Brier Score
        if calib_config['brier_score']['enabled']:
            brier = brier_score_loss(y_true, y_prob)
            results['brier_score'] = brier

            # Bootstrap CI for Brier score (only in final evaluation mode)
            if not self.is_training_mode and calib_config['brier_score'][
                    'bootstrap_ci']:
                ci_lower, ci_upper = self._bootstrap_ci(y_true, y_prob,
                                                        brier_score_loss)
                results['brier_score_ci_lower'] = ci_lower
                results['brier_score_ci_upper'] = ci_upper
                logger.info(
                    f"Brier Score: {brier:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )
            else:
                logger.info(f"Brier Score: {brier:.4f}")

        # Expected Calibration Error (ECE)
        if calib_config['expected_calibration_error']['enabled']:
            n_bins = calib_config['expected_calibration_error']['n_bins']
            ece = self._compute_ece(y_true, y_prob, n_bins)
            results['ece'] = ece

            # Bootstrap CI for ECE (only in final evaluation mode)
            if not self.is_training_mode and calib_config[
                    'expected_calibration_error']['bootstrap_ci']:

                def ece_fn(y_true, y_prob):
                    return self._compute_ece(y_true, y_prob, n_bins)

                ci_lower, ci_upper = self._bootstrap_ci(y_true, y_prob, ece_fn)
                results['ece_ci_lower'] = ci_lower
                results['ece_ci_upper'] = ci_upper
                logger.info(
                    f"Expected Calibration Error: {ece:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )
            else:
                logger.info(f"Expected Calibration Error: {ece:.4f}")

        # Calibration Slope and Intercept (reported per TRIPOD)
        if calib_config['calibration_slope']['enabled']:
            slope, intercept = self._compute_calibration_slope_intercept(
                y_true, y_prob)
            results['calibration_slope'] = slope
            results['calibration_intercept'] = intercept

            # Bootstrap CI for slope and intercept (only in final evaluation mode)
            if not self.is_training_mode and calib_config['calibration_slope'][
                    'bootstrap_ci']:

                def slope_fn(y_true, y_prob):
                    s, _ = self._compute_calibration_slope_intercept(
                        y_true, y_prob)
                    return s

                def intercept_fn(y_true, y_prob):
                    _, i = self._compute_calibration_slope_intercept(
                        y_true, y_prob)
                    return i

                slope_ci_lower, slope_ci_upper = self._bootstrap_ci(
                    y_true, y_prob, slope_fn)
                intercept_ci_lower, intercept_ci_upper = self._bootstrap_ci(
                    y_true, y_prob, intercept_fn)

                results['calibration_slope_ci_lower'] = slope_ci_lower
                results['calibration_slope_ci_upper'] = slope_ci_upper
                results['calibration_intercept_ci_lower'] = intercept_ci_lower
                results['calibration_intercept_ci_upper'] = intercept_ci_upper

                logger.info(
                    f"Calibration Slope: {slope:.4f} (95% CI: [{slope_ci_lower:.4f}, {slope_ci_upper:.4f}])"
                )
                logger.info(
                    f"Calibration Intercept: {intercept:.4f} (95% CI: [{intercept_ci_lower:.4f}, {intercept_ci_upper:.4f}])"
                )
            else:
                logger.info(
                    f"Calibration Slope: {slope:.4f}, Intercept: {intercept:.4f}"
                )

        return results

    def _compute_ece(self,
                     y_true: np.ndarray,
                     y_prob: np.ndarray,
                     n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted probabilities and actual outcomes.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

            if np.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = np.mean(y_prob[in_bin])
                # Average accuracy in bin
                avg_accuracy = np.mean(y_true[in_bin])
                # Weight by proportion of samples
                weight = np.sum(in_bin) / len(y_prob)

                ece += weight * np.abs(avg_confidence - avg_accuracy)

        return ece

    def _compute_calibration_slope_intercept(
            self, y_true: np.ndarray,
            y_prob: np.ndarray) -> Tuple[float, float]:
        """
        Compute calibration slope and intercept via logistic regression.
        
        Perfect calibration: slope = 1.0, intercept = 0.0
        Slope < 1: Model is overconfident
        Slope > 1: Model is underconfident
        
        Args:
            y_true: True labels [N]
            y_prob: Predicted probabilities [N]
            
        Returns:
            (slope, intercept)
        """
        from sklearn.linear_model import LogisticRegression
        import warnings

        # Convert probabilities to logits
        # Use larger epsilon to avoid extreme logit values that cause overflow
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Check for infinity or extreme values (can happen with perfect predictions)
        if not np.all(np.isfinite(logits)):
            logger.warning("Calibration slope/intercept: logits contain infinity (perfect predictions)")
            return float('nan'), float('nan')

        # Fit logistic regression: y_true ~ logits
        # Slope and intercept measure calibration (no regularization)
        # Handle sklearn version differences for penalty parameter
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
                lr.fit(logits.reshape(-1, 1), y_true)
            except (TypeError, ValueError) as e:
                # Check if it's an infinity error
                if "infinity" in str(e).lower():
                    logger.warning("Calibration slope/intercept fit failed (extreme logit values)")
                    return float('nan'), float('nan')
                # Fallback for older sklearn: use very weak regularization
                lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=1000)
                lr.fit(logits.reshape(-1, 1), y_true)
            except Exception:
                # Perfect separation or other issues - return NaN
                logger.warning("Calibration slope/intercept fit failed (possible perfect separation)")
                return float('nan'), float('nan')

        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]

        return slope, intercept

    def _compute_per_class_metrics(self, y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict:
        """Compute per-class metrics based on config."""
        unique_classes = np.unique(y_true)
        per_class = {}

        # Get list of metrics to compute from config
        per_class_config = self.metrics_config['discrimination']['per_class']
        metrics_to_compute = per_class_config['metrics']

        for cls in unique_classes:
            cls_mask = (y_true == cls)
            cls_pred = (y_pred == cls)

            tp = np.sum(cls_mask & cls_pred)
            fp = np.sum(~cls_mask & cls_pred)
            fn = np.sum(cls_mask & ~cls_pred)

            cls_metrics = {}

            if 'precision' in metrics_to_compute:
                cls_metrics['precision'] = tp / (tp + fp) if (tp +
                                                              fp) > 0 else 0.0

            if 'recall' in metrics_to_compute:
                cls_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if 'f1_score' in metrics_to_compute:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                cls_metrics['f1_score'] = 2 * (precision * recall) / (
                    precision + recall) if (precision + recall) > 0 else 0.0

            if 'support' in metrics_to_compute:
                cls_metrics['support'] = np.sum(cls_mask)

            per_class[f"class_{cls}"] = cls_metrics

        return per_class

    def _compute_confusion_matrix(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix with safe normalization."""
        # Force label order to handle missing classes in small datasets
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Normalize if configured (figures.confusion_matrix.normalize)
        cm_config = self.metrics_config['figures']['confusion_matrix']
        normalize = cm_config['normalize']
        
        # Accept bool + common strings robustly
        if normalize is True or str(normalize).lower() == 'true':
            denom = cm.sum(axis=1, keepdims=True)
            denom = np.where(denom == 0, 1, denom)
            cm = cm.astype(float) / denom
        elif str(normalize).lower() == 'pred':
            denom = cm.sum(axis=0, keepdims=True)
            denom = np.where(denom == 0, 1, denom)
            cm = cm.astype(float) / denom
        elif str(normalize).lower() == 'all':
            denom = cm.sum()
            cm = cm.astype(float) / (denom if denom > 0 else 1.0)

        return cm

    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric_fn,
        n_iterations: Optional[int] = None,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals for a metric.
        
        Uses stratified bootstrap if configured (maintains class distribution).
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric_fn: Function to compute metric
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            (ci_lower, ci_upper): Confidence interval bounds
        """
        bootstrap_config = self.metrics_config['bootstrap']
        if n_iterations is None:
            n_iterations = bootstrap_config['n_iterations']

        stratified = bootstrap_config['stratified']
        # Use bootstrap seed if specified, otherwise fall back to global seed
        random_seed = bootstrap_config['random_seed']
        if random_seed is None:
            random_seed = self.config['reproducibility']['random_seed']
        method = bootstrap_config['method']

        rng = np.random.RandomState(random_seed)
        scores = []

        n_samples = len(y_true)

        for _ in range(n_iterations):
            if stratified:
                # Stratified resampling: maintains class distribution
                indices = []
                for class_label in np.unique(y_true):
                    class_indices = np.where(y_true == class_label)[0]
                    class_sample = rng.choice(class_indices,
                                              size=len(class_indices),
                                              replace=True)
                    indices.extend(class_sample)
                indices = np.array(indices)
                # Shuffle to avoid class ordering
                rng.shuffle(indices)
            else:
                # Standard bootstrap: random resampling with replacement
                indices = rng.choice(n_samples, size=n_samples, replace=True)

            # Skip if only one class present (shouldn't happen with stratified)
            if len(np.unique(y_true[indices])) < 2:
                continue

            try:
                score = metric_fn(y_true[indices], y_prob[indices])
                scores.append(score)
            except Exception:
                continue

        # Guard against empty scores (can happen with small/imbalanced data)
        if len(scores) == 0:
            logger.warning("Bootstrap produced no valid resamples; returning NaNs for CI")
            return (float("nan"), float("nan"))

        # Compute confidence interval using specified method
        alpha = (1 - confidence_level) / 2

        if method == 'percentile':
            # Percentile method (standard)
            ci_lower = np.percentile(scores, alpha * 100)
            ci_upper = np.percentile(scores, (1 - alpha) * 100)
        elif method == 'bca':
            # Bias-corrected and accelerated (BCa) method
            # Note: Simplified implementation, full BCa requires jackknife
            logger.warning("BCa method not fully implemented, using percentile")
            ci_lower = np.percentile(scores, alpha * 100)
            ci_upper = np.percentile(scores, (1 - alpha) * 100)
        else:
            # Default to percentile
            ci_lower = np.percentile(scores, alpha * 100)
            ci_upper = np.percentile(scores, (1 - alpha) * 100)

        return ci_lower, ci_upper


def create_metrics_evaluator(config: Dict,
                             is_training_mode: bool = None
                            ) -> ComprehensiveMetrics:
    """
    Factory function to create metrics evaluator.
    
    Args:
        config: Metrics configuration from YAML
        is_training_mode: Override for evaluation mode
            - True: Fast mode for training validation (no CIs)
            - False: Full mode for final evaluation (with CIs)
            - None: Auto-detect from config
    
    Returns:
        ComprehensiveMetrics instance
    """
    return ComprehensiveMetrics(config, is_training_mode=is_training_mode)
