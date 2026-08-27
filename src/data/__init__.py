"""
Data loading and preprocessing for classification.

This module handles:
- 3D medical image loading (NIfTI, DICOM)
- MONAI-based preprocessing and augmentation
- Patient-level stratified K-fold CV
- Data module creation
- CSV preprocessing for flexible image type selection
"""

from .cv_splitter import (
    CrossValidationSplitter,
    create_cv_splitter,
)
from .data_loading import (
    ClassificationDataModule,
    create_data_module,
)
from .csv_preprocessing import (
    preprocess_classification_csv,
    prepare_csv_for_training,
)
from .transforms import build_transforms

__all__ = [
    'ClassificationDataModule',
    'create_data_module',
    'CrossValidationSplitter',
    'create_cv_splitter',
    'preprocess_classification_csv',
    'prepare_csv_for_training',
    'build_transforms',
]
