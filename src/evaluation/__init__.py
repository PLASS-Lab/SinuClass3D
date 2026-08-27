"""Model Evaluation Module

This module provides core model evaluation capabilities for trained 3D medical
image classification models.

Core Components:
- ClassificationEvaluator: Main evaluator class for computing performance metrics

The evaluator focuses on:
- Prediction generation
- Performance metrics computation (ROC-AUC, PR-AUC, accuracy, etc.)
- Basic visualization (ROC curve, PR curve, calibration, confusion matrix)
- Results export (CSV, LaTeX tables)
"""

from src.evaluation.evaluator import ClassificationEvaluator
from src.evaluation.calibration import fit_and_save_calibration, fit_platt_calibration

__all__ = ['ClassificationEvaluator', 'fit_and_save_calibration', 'fit_platt_calibration']
