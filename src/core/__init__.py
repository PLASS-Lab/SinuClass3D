"""
Core classification utilities.

High-level interface:
- CrossValidator: Unified CV interface (use this!)

Modules:
- cross_validation: Standard k-fold cross-validation
- metrics: Comprehensive metrics with bootstrap CIs
"""

from src.core.cross_validation import CrossValidator
from src.core.metrics import ComprehensiveMetrics, create_metrics_evaluator, select_youden_threshold

__all__ = [
    'CrossValidator',
    'ComprehensiveMetrics',
    'create_metrics_evaluator',
    'select_youden_threshold',
]
