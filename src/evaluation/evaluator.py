"""
Classification Evaluator Module

This module provides the ClassificationEvaluator class for comprehensive
evaluation of trained 3D medical image classification models.

Features:
- Prediction generation
- Comprehensive metrics computation
- Visualization generation (ROC, PR, calibration, confusion matrix)
- Results export (CSV, LaTeX tables, JSON)
- Support for standard and MIL models
- Bootstrap confidence intervals

Compatible with evaluation scripts and YAML configuration.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.metrics import create_metrics_evaluator

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """
    Evaluator for trained classification models.
    
    Performs comprehensive evaluation including:
    - Prediction generation with optional Test-Time Augmentation
    - Metrics computation (ROC-AUC, PR-AUC, accuracy, etc.)
    - Visualization generation (ROC curve, PR curve, calibration, confusion matrix)
    - Results export in multiple formats
    
    Args:
        model: Trained classification model
        device: Torch device (cuda or cpu)
        config: Configuration dictionary from YAML
        is_mil_mode: Whether model uses MIL (default: False)
        checkpoint_path: Optional path to model checkpoint (for logging)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict,
        is_mil_mode: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.is_mil_mode = is_mil_mode
        self.checkpoint_path = checkpoint_path

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Metrics evaluator (final mode: full evaluation with bootstrap CIs)
        # Note: Trainer passes is_training_mode=True for fast validation
        self.metrics_evaluator = create_metrics_evaluator(config, is_training_mode=False)

        # Explainability (optional, auto-detects architecture)

        # Output directory
        eval_config = config['evaluation']
        output_config = eval_config['output']
        output_dir = output_config['output_dir']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Evaluator initialized. Results will be saved to: {self.output_dir}"
        )

    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> Dict:
        """
        Generate predictions on dataset.
        
        Args:
            data_loader: PyTorch DataLoader with test/validation data
            
        Returns:
            Dictionary with predictions, probabilities, labels, and patient_ids
        """
        # Get dataset size if available
        try:
            dataset_size = len(data_loader.dataset)  # type: ignore
            logger.info(f"Generating predictions on {dataset_size} samples")
        except TypeError:
            logger.info("Generating predictions...")

        all_preds = []
        all_probs = []
        all_labels = []
        all_patient_ids = []

        for batch in tqdm(data_loader, desc="Predictions"):
            images = batch['image'].to(self.device)
            labels = batch['label']
            patient_ids = batch['patient_id']

            # Forward pass
            if self.is_mil_mode:
                # MIL model returns (logits, attention, features)
                logits, _, _ = self.model(images,
                                          return_attention=False,
                                          return_features=False)
            else:
                logits = self.model(images)

            # Convert to predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Convert patient_ids to strings (handle tensors)
            for pid in patient_ids:
                if isinstance(pid, torch.Tensor):
                    all_patient_ids.append(str(pid.item()))
                else:
                    all_patient_ids.append(str(pid))

        predictions = {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs),
            'labels': np.array(all_labels),
            'patient_ids': all_patient_ids,
        }

        logger.info(
            f"Generated predictions for {len(predictions['predictions'])} samples"
        )

        return predictions

    def evaluate(self, predictions: Dict, save_results: bool = True, threshold: float = None) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Dictionary from predict() method
            save_results: Whether to save results and plots (default: True)
            threshold: Classification threshold (if None, computed from config)
            
        Returns:
            Dictionary of all computed metrics
        """
        y_true = predictions['labels']
        y_prob = predictions['probabilities']

        logger.info("Computing comprehensive metrics...")

        # Compute all metrics (y_pred is computed internally from threshold)
        metrics = self.metrics_evaluator.compute_all_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
        )

        if save_results:
            # Save results
            self._save_results(metrics, predictions)

        # Log summary (disabled - script-level summary is more complete)
        # self._log_metrics_summary(metrics)

        return metrics

    def _save_results(self, metrics: Dict, predictions: Dict):
        """Save results in multiple formats (CSV, LaTeX, JSON)."""
        output_config = self.config['evaluation']['output']
        formats = output_config['formats']

        # CSV format
        if formats['csv']:
            # Metrics summary
            metrics_flat = self._flatten_metrics(metrics)
            df_metrics = pd.DataFrame([metrics_flat])
            df_metrics.to_csv(self.output_dir / 'metrics_summary.csv',
                              index=False)
            logger.info("✓ Metrics summary saved (CSV)")

            # Per-sample predictions
            num_classes = predictions['probabilities'].shape[1]
            pred_dict = {
                'patient_id': predictions['patient_ids'],
                'true_label': predictions['labels'],
                'predicted_label': predictions['predictions'],
            }

            # Add probability columns for each class
            for i in range(num_classes):
                pred_dict[f'prob_class_{i}'] = predictions['probabilities'][:,
                                                                            i]

            df_preds = pd.DataFrame(pred_dict)
            df_preds.to_csv(self.output_dir / 'predictions.csv', index=False)
            logger.info("✓ Predictions saved (CSV)")

        # LaTeX table
        if formats['latex']:
            self._save_latex_table(metrics)
            logger.info("✓ LaTeX table saved")

        # JSON format
        if formats.get('json', False):
            import json
            from src.utils import convert_to_serializable
            json_path = self.output_dir / 'metrics.json'
            with open(json_path, 'w') as f:
                json.dump(convert_to_serializable(metrics), f, indent=2)
            logger.info("✓ Metrics saved (JSON)")

        logger.info(f"All results saved to {self.output_dir}")

    @staticmethod
    def _flatten_metrics(metrics: Dict) -> Dict:
        """Flatten nested metrics dictionary for CSV export."""
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, (int, float, str, bool)):
                flat[key] = value
            elif isinstance(value, np.ndarray):
                # Skip arrays (like confusion matrix) in flattened version
                continue
        return flat

    def _save_latex_table(self, metrics: Dict):
        """Generate LaTeX table for publication."""
        precision = self.config['evaluation']['output']['precision']

        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Classification Performance Metrics}",
            "\\begin{tabular}{lc}",
            "\\hline",
            "Metric & Value (95\\% CI) \\\\",
            "\\hline",
        ]

        # Primary metrics
        primary_metrics = [
            'roc_auc', 'pr_auc', 'accuracy', 'sensitivity', 'specificity',
            'ppv', 'npv'
        ]

        for metric_name in primary_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                ci_low = metrics.get(f"{metric_name}_ci_lower", None)
                ci_high = metrics.get(f"{metric_name}_ci_upper", None)

                metric_label = metric_name.replace('_', ' ').title().replace(
                    'Auc', 'AUC').replace('Ppv', 'PPV').replace('Npv', 'NPV')

                if ci_low is not None and ci_high is not None:
                    latex_lines.append(
                        f"{metric_label} & {value:.{precision}f} ({ci_low:.{precision}f}-{ci_high:.{precision}f}) \\\\"
                    )
                else:
                    latex_lines.append(
                        f"{metric_label} & {value:.{precision}f} \\\\")

        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])

        latex_path = self.output_dir / 'metrics_table.tex'
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

    @staticmethod
    def _log_metrics_summary(metrics: Dict):
        """Log summary of key metrics."""
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)

        key_metrics = [
            'accuracy', 'roc_auc', 'pr_auc', 'sensitivity', 'specificity'
        ]

        for metric_name in key_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                logger.info(f"{metric_name.upper()}: {value:.4f}")

        logger.info("=" * 50)
