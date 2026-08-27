"""
Cross-validation splitting logic for medical image classification.

Supports:
- Stratified K-fold cross-validation
- Patient-level splitting (prevents data leakage)
- Scan-level splitting
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class CrossValidationSplitter:
    """
    Handles stratified K-fold cross-validation splitting.
    
    Features:
    - Patient-level splitting to prevent data leakage when patients have multiple scans
    - Stratification to maintain class balance across folds
    - Detailed logging of split statistics
    """

    def __init__(self, df: pd.DataFrame, cv_config: Dict, global_config: Dict):
        """
        Initialize cross-validation splitter.
        
        Args:
            df: DataFrame with columns: 'label', optionally 'patient_id'
            cv_config: Cross-validation config from config['data']['cross_validation']
            global_config: Full config (for accessing reproducibility settings)
        """
        self.df = df
        self.cv_config = cv_config
        self.global_config = global_config
        self.n_folds = cv_config['n_folds']
        self.patient_level = cv_config['patient_level']

        # Use CV seed if specified, otherwise fall back to global seed
        self.random_state = cv_config['random_state']
        if self.random_state is None:
            self.random_state = (global_config['reproducibility']['random_seed'])

        logger.info(
            f"CrossValidationSplitter initialized: {self.n_folds} folds, "
            f"patient_level={self.patient_level}, random_state={self.random_state}"
        )

    def split(self, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform cross-validation split for a specific fold.
        
        Args:
            fold: Fold index (0 to n_folds-1)
            
        Returns:
            (train_idx, val_idx): Arrays of DataFrame row indices
            
        Raises:
            ValueError: If fold index is out of range
        """
        if fold < 0 or fold >= self.n_folds:
            raise ValueError(
                f"Fold index {fold} out of range [0, {self.n_folds})")


        # Check if patient-level splitting is enabled and possible
        if self.patient_level and 'patient_id' in self.df.columns:
            return self._perform_patient_level_split(fold)
        else:
            return self._perform_scan_level_split(fold)

    def _perform_patient_level_split(
            self, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform patient-level stratified K-fold split.
        
        Groups all scans from the same patient together to prevent data leakage.
        
        Args:
            fold: Fold index
            
        Returns:
            (train_idx, val_idx): Arrays of DataFrame row indices
        """
        # Group by patient - assume all scans from same patient have same label
        patient_df = self.df.groupby('patient_id').agg({
            'label': 'first'
        }).reset_index()

        X_patients = patient_df['patient_id'].values
        y_patients = patient_df['label'].values

        # Create stratified K-fold splitter
        skf = StratifiedKFold(n_splits=self.n_folds,
                              shuffle=True,
                              random_state=self.random_state)

        # Split patients into folds
        splits = list(skf.split(X_patients, y_patients))
        train_patient_idx, val_patient_idx = splits[fold]

        # Get patient IDs for this fold
        train_patients = set(patient_df.iloc[train_patient_idx]['patient_id'])
        val_patients = set(patient_df.iloc[val_patient_idx]['patient_id'])
        
        logger.info(f"Patient-level CV: Fold {fold + 1}")

        # Get all scan indices for these patients
        train_idx = self.df[self.df['patient_id'].isin(
            train_patients)].index.values
        val_idx = self.df[self.df['patient_id'].isin(val_patients)].index.values
        logger.info(
            f"  Train patients: {len(train_patients)}, scans: {len(train_idx)}")
        logger.info(
            f"  Val patients: {len(val_patients)}, scans: {len(val_idx)}")

        self._log_split_distributions(train_idx, val_idx, fold)

        return train_idx, val_idx

    def _perform_scan_level_split(self,
                                  fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform scan-level stratified K-fold split.
        
        Standard splitting where each scan is treated independently.
        
        Args:
            fold: Fold index
            
        Returns:
            (train_idx, val_idx): Arrays of DataFrame row indices
        """
        X = self.df.index.values
        y = self.df['label'].values

        # Create stratified K-fold splitter
        skf = StratifiedKFold(n_splits=self.n_folds,
                              shuffle=True,
                              random_state=self.random_state)

        # Get train/val indices for this fold
        splits = list(skf.split(X, y))
        train_idx, val_idx = splits[fold]

        logger.info(f"Scan-level CV: Fold {fold + 1}")
        self._log_split_distributions(train_idx, val_idx, fold)

        return train_idx, val_idx

    def _log_split_distributions(self,
                                 train_idx: np.ndarray,
                                 val_idx: np.ndarray,
                                 fold: Optional[int] = None):
        """
        Log statistics about the train/val split.
        
        Args:
            train_idx: Training indices
            val_idx: Validation indices
            fold: Optional fold number for logging
        """
        fold_str = f"Fold {fold + 1}: " if fold is not None else ""

        logger.info(
            f"{fold_str}{len(train_idx)} train, {len(val_idx)} val indices")

        train_labels = self.df.iloc[train_idx]['label'].values
        val_labels = self.df.iloc[val_idx]['label'].values

        logger.info(f"Train class distribution: {np.bincount(train_labels)}")
        logger.info(f"Val class distribution: {np.bincount(val_labels)}")


def create_cv_splitter(df: pd.DataFrame, cv_config: Dict,
                       global_config: Dict) -> CrossValidationSplitter:
    """
    Factory function to create a cross-validation splitter.
    
    Args:
        df: DataFrame with sample data
        cv_config: Cross-validation configuration
        global_config: Full configuration dictionary
        
    Returns:
        CrossValidationSplitter instance
    """
    return CrossValidationSplitter(df, cv_config, global_config)
