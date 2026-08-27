"""
CSV Preprocessing Utilities

Handles column selection and preprocessing for classification CSV files.
This logic is kept separate from the data loading module to maintain
separation of concerns - scripts handle domain-specific column names,
while the data loader stays generic.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_classification_csv(
        csv_path: str,
        image_type: str = 'full_head_ct_path',
        output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocess classification CSV to create generic 'image_path' column.
    
    This function handles the mapping from specific column names
    (full_head_ct_path, mask_bbox_margin_path, etc.) to the generic
    'image_path' column expected by the data loader.
    
    Args:
        csv_path: Path to the input CSV file
        image_type: Which image column to use. Options:
            - 'full_head_ct_path': Full head CT scan (default)
            - 'mask_bbox_margin_path': CT cropped to mask bbox with margin
            - 'dilated_mask_crop_path': CT with dilated mask applied
            - 'no_dilated_mask_crop_path': CT with non-dilated mask applied
        output_path: Optional path to save preprocessed CSV
    
    Returns:
        Preprocessed DataFrame with 'image_path' column
    
    Raises:
        ValueError: If image_type is invalid or column doesn't exist
    """
    # Valid image types
    valid_image_types = [
        'full_head_ct_path', 'mask_bbox_margin_path', 'dilated_mask_crop_path',
        'no_dilated_mask_crop_path'
    ]

    if image_type not in valid_image_types:
        raise ValueError(f"Invalid image_type: '{image_type}'. "
                         f"Must be one of {valid_image_types}")

    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV from {csv_path} with {len(df)} samples")
    logger.info(f"Available columns: {list(df.columns)}")

    # Verify the image_type column exists
    if image_type not in df.columns:
        raise ValueError(f"Column '{image_type}' not found in CSV. "
                         f"Available columns: {list(df.columns)}")

    # Create 'image_path' column by copying the selected image type
    df['image_path'] = df[image_type]
    logger.info(f"Created 'image_path' column from '{image_type}'")

    # Verify we have required columns
    required_columns = ['patient_id', 'label', 'image_path']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        # Try to handle 'class' vs 'label'
        if 'label' in missing_columns and 'class' in df.columns:
            df['label'] = df['class']
            logger.info("Renamed 'class' column to 'label'")
            missing_columns.remove('label')

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. "
                             f"CSV must have: {required_columns}")

    # Log statistics
    logger.info(f"Preprocessed CSV ready:")
    logger.info(f"  - Total samples: {len(df)}")
    logger.info(f"  - Image type: {image_type}")
    logger.info(
        f"  - Label distribution: {df['label'].value_counts().to_dict()}")

    # Save preprocessed CSV if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed CSV to {output_path}")

    return df


def prepare_csv_for_training(config: dict) -> str:
    """
    Prepare CSV for training by preprocessing it based on config.
    
    This is the main entry point for scripts (train.py, evaluate.py).
    It reads the config, preprocesses the CSV, and returns the path
    to the preprocessed CSV (which can be a temp file or the original
    if already in correct format).
    
    Args:
        config: Configuration dictionary with data.image_type and data.paths.csv_path
    
    Returns:
        Path to preprocessed CSV file
    """
    csv_path = config['data']['paths']['csv_path']
    image_type = config['data']['image_type']

    # Check if CSV already has 'image_path' column
    df = pd.read_csv(csv_path)

    if 'image_path' in df.columns:
        # CSV already has generic 'image_path' column - use as is
        logger.info(f"CSV already has 'image_path' column - using as is")
        return csv_path
    else:
        # Need to preprocess - create temp CSV
        import tempfile
        temp_dir = Path(config['data']['paths']['cache_dir'])
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_csv_path = temp_dir / f"preprocessed_{image_type}.csv"

        # Preprocess and save
        preprocess_classification_csv(csv_path=csv_path,
                                      image_type=image_type,
                                      output_path=str(temp_csv_path))

        return str(temp_csv_path)
