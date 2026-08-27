"""
3D Classification Data Loading with CT-Specific Preprocessing
Leak-safe transforms following medical imaging best practices
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.transforms import Compose
from sklearn.utils.class_weight import compute_class_weight

from src.data.cv_splitter import CrossValidationSplitter
from src.data.transforms import build_transforms

logger = logging.getLogger(__name__)


class ClassificationDataModule:
    """
    Data module for 3D medical image classification with CT-specific preprocessing.
    
    Features:
    - HU windowing and z-score normalization (per-volume, leak-safe)
    - RAS reorientation and isotropic resampling
    - Optional ROI cropping using segmentation masks
    - Light augmentation (flips, small affine, mild noise)
    - 5-fold stratified cross-validation
    - Class imbalance handling
    """

    def __init__(self, config: Dict):
        """
        Initialize data module.
        
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.csv_path = Path(config['data']['paths']['csv_path'])
        self.root_dir = Path(config['data']['paths']['root_dir'])

        # Get image type selection (which column to use as image_path)
        self.image_type = config['data']['image_type']
        self.dual_window_mode = (self.image_type == 'dual_window')
        self.multi_channel_mode = (self.image_type == 'multi_channel')
        
        # Get multi-channel config if applicable
        if self.multi_channel_mode:
            self.multi_channel_config = config['data'].get('multi_channel', {})
            num_ch = sum(1 for i in range(10) if f'channel_{i}' in self.multi_channel_config)
            logger.info(f"Multi-channel mode ({num_ch}ch): {list(self.multi_channel_config.values())}")
        elif self.dual_window_mode:
            self.dual_window_config = config['data'].get('dual_window', {})
            logger.info(f"Dual window mode: channel_0={self.dual_window_config.get('channel_0')}, "
                       f"channel_1={self.dual_window_config.get('channel_1')}")

        # Load data
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} samples from {self.csv_path}")

        # Handle image_type column mapping (skip for multi-channel modes)
        if not self.dual_window_mode and not self.multi_channel_mode:
            if 'image_path' not in self.df.columns and self.image_type in self.df.columns:
                self.df['image_path'] = self.df[self.image_type]
                logger.info(f"Created 'image_path' column from '{self.image_type}'")
            elif self.image_type != 'image_path' and self.image_type in self.df.columns:
                # Override image_path with selected type
                self.df['image_path'] = self.df[self.image_type]
                logger.info(
                    f"Overriding 'image_path' with '{self.image_type}' column")
            elif self.image_type not in [
                    'image_path'
            ] and self.image_type not in self.df.columns:
                logger.warning(
                    f"Specified image_type '{self.image_type}' not found in CSV, using 'image_path'"
                )

        # Rename 'class' column to 'label' if it exists for consistency
        if 'class' in self.df.columns and 'label' not in self.df.columns:
            self.df['label'] = self.df['class']
            logger.info("Renamed 'class' column to 'label'")

        # Convert class names to numeric labels if needed
        if self.df['label'].dtype == 'object' or pd.api.types.is_string_dtype(self.df['label']):
            # Validate CSV classes match config
            csv_classes = sorted(self.df['label'].unique())
            config_classes = config['classification']['class_names']
            num_classes = config['classification']['num_classes']

            # Check number of classes
            if len(csv_classes) != num_classes:
                raise ValueError(
                    f"Class mismatch! CSV has {len(csv_classes)} classes "
                    f"but config expects {num_classes}.\n"
                    f"CSV classes: {csv_classes}\n"
                    f"Config classes: {config_classes}")

            # Check class names match (order-independent)
            if set(csv_classes) != set(config_classes):
                raise ValueError(f"Class names mismatch!\n"
                                 f"CSV classes: {csv_classes}\n"
                                 f"Config classes: {config_classes}")

            logger.info(f"✓ Validated classes match config: {config_classes}")

            # Map ECRS/non-ECRS to 0/1
            label_map = {name: idx for idx, name in enumerate(config_classes)}
            self.df['label'] = self.df['label'].map(label_map).astype(int)
            logger.info(f"Mapped class names to labels: {label_map}")

        # Cross-validation splitter
        cv_config = config['data']['cross_validation']
        self.cv_splitter = CrossValidationSplitter(self.df, cv_config, config)
        self.n_folds = cv_config['n_folds']  # Keep for backward compatibility

        # Class weights for imbalanced data
        self.class_weights = self._compute_class_weights()

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced dataset."""
        labels = self.df['label'].values
        unique_labels = np.unique(labels)

        weights = compute_class_weight(class_weight='balanced',
                                       classes=unique_labels,
                                       y=labels)

        logger.info(f"Class distribution: {np.bincount(labels.astype(int))}")
        logger.info(f"Class weights: {weights}")

        return torch.FloatTensor(weights)

    def get_transforms(self, mode: str = "train") -> Compose:
        """
        Get transforms for train/val/test.
        
        Delegates to build_transforms() function in transforms.py module.
        
        Args:
            mode: "train", "val", or "test"
            
        Returns:
            MONAI Compose transform
        """
        return build_transforms(self.config, mode)

    def _perform_cv_split(self, fold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform cross-validation split and return indices.
        
        Delegates to CrossValidationSplitter for the actual splitting logic.
        
        Args:
            fold: Fold index (0 to n_folds-1)
            
        Returns:
            (train_idx, val_idx): Arrays of DataFrame row indices
        """
        return self.cv_splitter.split(fold)

    def prepare_data_dicts(self, fold: int = 0, split: Optional[str] = None):
        """
        Prepare data dictionaries for a specific fold with lazy loading support.
        
        Args:
            fold: Fold index (0 to n_folds-1)
            split: Optional split to return:
                   - None: Return both (train_data, val_data) [default, backward compatible]
                   - 'train': Return only train_data
                   - 'val': Return only val_data
            
        Returns:
            If split is None: (train_data, val_data) tuple
            If split is 'train' or 'val': data for that split only
        """
        # Get train/val indices
        train_idx, val_idx = self._perform_cv_split(fold)

        # Lazy loading: only create requested split(s)
        if split == 'train':
            train_data = self._create_data_dicts(train_idx)
            logger.info(f"Created {len(train_data)} train samples")
            return train_data
        elif split == 'val':
            val_data = self._create_data_dicts(val_idx)
            logger.info(f"Created {len(val_data)} val samples")
            return val_data
        else:
            # Default: return both (backward compatible)
            train_data = self._create_data_dicts(train_idx)
            val_data = self._create_data_dicts(val_idx)
            logger.info(
                f"Created {len(train_data)} train, {len(val_data)} val samples")
            return train_data, val_data

    def _create_data_dicts(self, indices: np.ndarray) -> List[Dict]:
        """
        Create list of data dictionaries from DataFrame indices.
        
        Expects CSV to have 'image_path' column (or it should be created by preprocessing).
        Handles optional labels (for inference mode).
        Supports dual_window mode for 2-channel input.
        """
        data_dicts = []

        for idx in indices:
            row = self.df.iloc[idx]

            if self.multi_channel_mode:
                # Multi-channel mode: load N channels dynamically from config
                ch_cols = []
                for i in range(10):  # Support up to 10 channels
                    key = f'channel_{i}'
                    if key in self.multi_channel_config:
                        ch_cols.append(self.multi_channel_config[key])
                    else:
                        break
                
                data_dict = {"patient_id": row['patient_id']}
                skip_patient = False
                for i, col in enumerate(ch_cols):
                    ch_path = row[col]
                    if self.root_dir and not Path(ch_path).is_absolute():
                        ch_path = str(self.root_dir / ch_path)
                    else:
                        ch_path = str(ch_path)
                    # Validate file exists
                    if not Path(ch_path).exists():
                        logger.warning(f"File not found for {row['patient_id']}: {ch_path}")
                        skip_patient = True
                        break
                    data_dict[f"image_ch{i}"] = ch_path
                
                if skip_patient:
                    continue
                    
            elif self.dual_window_mode:
                # Dual window mode: load both channels
                ch0_col = self.dual_window_config['channel_0']
                ch1_col = self.dual_window_config['channel_1']
                
                ch0_path = row[ch0_col]
                ch1_path = row[ch1_col]
                
                # Handle both absolute and relative paths
                if self.root_dir and not Path(ch0_path).is_absolute():
                    ch0_path = str(self.root_dir / ch0_path)
                    ch1_path = str(self.root_dir / ch1_path)
                else:
                    ch0_path = str(ch0_path)
                    ch1_path = str(ch1_path)
                
                data_dict = {
                    "image_ch0": ch0_path,  # Soft-tissue window
                    "image_ch1": ch1_path,  # Bone window
                    "patient_id": row['patient_id'],
                }
            else:
                # Single channel mode
                image_path = row['image_path']

                # Handle both absolute and relative paths
                if self.root_dir and not Path(image_path).is_absolute():
                    image_path = str(self.root_dir / image_path)
                else:
                    image_path = str(image_path)

                data_dict = {
                    "image": image_path,
                    "patient_id": row['patient_id'],
                }

            # Add label if available (for evaluation with ground truth)
            if 'label' in row and pd.notna(row['label']):
                data_dict["label"] = int(row['label'])

            data_dicts.append(data_dict)

        return data_dicts

    def _get_all_data_dicts(self) -> List[Dict]:
        """
        Get data dictionaries for ALL samples (no CV split).
        
        Internal helper for test/inference dataloaders.
        
        Returns:
            List of data dictionaries for all samples in CSV
        """
        all_indices = np.arange(len(self.df))
        data_dicts = self._create_data_dicts(all_indices)
        logger.info(f"Created {len(data_dicts)} data dicts (no CV split)")
        return data_dicts

    def _create_dataset(self, data_dicts: List[Dict], mode: str,
                        num_workers: int) -> CacheDataset:
        """
        Create a MONAI CacheDataset with transforms.
        
        Args:
            data_dicts: List of data dictionaries
            mode: 'train' or 'val' (determines transforms)
            num_workers: Number of workers for caching
            
        Returns:
            CacheDataset with appropriate transforms
        """
        transforms = self.get_transforms(mode)
        return CacheDataset(
            data=data_dicts,
            transform=transforms,
            cache_rate=1.0,
            num_workers=num_workers,
        )

    def _create_loader(self, dataset: CacheDataset, mode: str,
                       batch_size: Optional[int],
                       num_workers: int) -> DataLoader:
        """
        Create a DataLoader with appropriate settings.
        
        Args:
            dataset: Dataset to load from
            mode: 'train' or 'val' (determines shuffle and batch size)
            batch_size: Batch size (None uses config defaults)
            num_workers: Number of workers
            
        Returns:
            Configured DataLoader
        """
        # Determine batch size
        if batch_size is None:
            if mode == 'val':
                batch_size = self.config['compute']['val_batch_size']
            else:
                batch_size = self.config['compute']['batch_size']

        # Train shuffles, val/test does not
        shuffle = (mode == 'train')

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config['data']['pin_memory'],
            persistent_workers=self.config['data']['persistent_workers']
            if num_workers > 0 else False,
            prefetch_factor=self.config['data']['prefetch_factor']
            if num_workers > 0 else None,
            collate_fn=pad_list_data_collate,
        )

    def get_train_loaders(
            self,
            fold: int = 0,
            batch_size: Optional[int] = None,
            num_workers: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation dataloaders for training.
        
        Handles both CV and normal splits (config determines split type).
        
        Args:
            fold: Fold index (for CV) or split index (for normal split)
            batch_size: Batch size (overrides config)
            num_workers: Number of workers (overrides config)
            
        Returns:
            (train_loader, val_loader)
            
        Example:
            # Works for both CV and normal split
            train_loader, val_loader = data_module.get_train_loaders(fold=0)
        """
        # Get data dicts for both splits
        train_data, val_data = self.prepare_data_dicts(fold)

        # Set num_workers
        num_workers = num_workers or self.config['compute']['num_workers']

        # Create datasets using helper
        train_dataset = self._create_dataset(train_data, 'train', num_workers)
        val_dataset = self._create_dataset(val_data, 'val', num_workers)

        # Create dataloaders using helper
        train_loader = self._create_loader(train_dataset, 'train', batch_size,
                                           num_workers)
        val_loader = self._create_loader(val_dataset, 'val', batch_size,
                                         num_workers)

        return train_loader, val_loader

    def get_val_loader(self,
                       fold: int = 0,
                       batch_size: Optional[int] = None,
                       num_workers: Optional[int] = None) -> DataLoader:
        """
        Get validation dataloader for evaluation.
        
        Handles both CV and normal splits (config determines split type).
        Uses lazy loading - only creates validation data, skips training data entirely.
        
        Efficiency gains:
        - Skips train data dict creation
        - Skips train dataset creation and caching
        - Saves ~3-4GB RAM for typical 3D medical volumes
        
        Args:
            fold: Fold index (for CV) or split index (for normal split)
            batch_size: Batch size (overrides config, defaults to val_batch_size)
            num_workers: Number of workers (overrides config)
            
        Returns:
            val_loader
            
        Example:
            # Evaluate on validation split
            val_loader = data_module.get_val_loader(fold=0)
        """
        # Lazy loading: Get ONLY validation data (skips train completely)
        val_data = self.prepare_data_dicts(fold, split='val')

        # Set num_workers
        num_workers = num_workers or self.config['compute']['num_workers']

        # Create validation dataset using helper
        val_dataset = self._create_dataset(val_data, 'val', num_workers)

        # Create validation dataloader using helper
        val_loader = self._create_loader(val_dataset, 'val', batch_size,
                                         num_workers)

        return val_loader

    def get_full_train_loader(
            self,
            batch_size: Optional[int] = None,
            num_workers: Optional[int] = None) -> DataLoader:
        """
        Get dataloader for full training (all samples, no validation split).
        
        Use this for training final deployment model on ALL available data
        after cross-validation has been used to estimate performance.
        
        Args:
            batch_size: Batch size (overrides config)
            num_workers: Number of workers (overrides config)
            
        Returns:
            DataLoader for full training (all samples with train augmentations)
            
        Example:
            # Full training mode
            train_loader = data_module.get_full_train_loader()
            for batch in train_loader:
                loss = train_step(model, batch)
        """
        # Get all data (no CV split)
        data_dicts = self._get_all_data_dicts()
        
        # Set num_workers
        num_workers = num_workers or self.config['compute']['num_workers']
        
        # Create dataset with TRAIN transforms (with augmentation)
        dataset = self._create_dataset(data_dicts, 'train', num_workers)
        
        # Create dataloader with shuffle=True for training
        loader = self._create_loader(dataset, 'train', batch_size, num_workers)
        
        logger.info(f"Created full training loader: {len(data_dicts)} samples")
        return loader

    def get_test_loader(self,
                        batch_size: Optional[int] = None,
                        num_workers: Optional[int] = None) -> DataLoader:
        """
        Get dataloader for external test set evaluation.
        
        Use this for evaluating on external/held-out test cohorts (with labels).
        Uses ALL samples in CSV (no CV split).
        
        Args:
            batch_size: Batch size (overrides config, defaults to val_batch_size)
            num_workers: Number of workers (overrides config)
            
        Returns:
            DataLoader for external test set
            
        Example:
            # External test set evaluation
            test_config = config.copy()
            test_config['data']['paths']['csv_path'] = '/path/to/external_test.csv'
            test_module = create_data_module(test_config)
            test_loader = test_module.get_test_loader()
            
            # Evaluate
            for batch in test_loader:
                predictions = model(batch['image'])
                labels = batch['label']  # Has ground truth
                metrics = compute_metrics(predictions, labels)
        """
        # Get all data (no CV split for external test sets)
        data_dicts = self._get_all_data_dicts()

        # Set num_workers
        num_workers = num_workers or self.config['compute']['num_workers']

        # Create dataset with val transforms (no augmentation)
        dataset = self._create_dataset(data_dicts, 'val', num_workers)

        # Create dataloader
        loader = self._create_loader(dataset, 'val', batch_size, num_workers)

        return loader

    @staticmethod
    def _create_single_sample_dict(image_path: str,
                                   patient_id: Optional[str] = None,
                                   label: Optional[int] = None) -> Dict:
        """
        Create a data dictionary for a single sample.
        
        Internal helper for get_inference_loader.
        
        Args:
            image_path: Path to NIfTI image
            patient_id: Optional patient identifier
            label: Optional ground truth label
            
        Returns:
            Data dictionary compatible with MONAI transforms
        """
        data_dict = {
            "image": str(image_path),
            "patient_id": patient_id or Path(image_path).stem,
        }

        if label is not None:
            data_dict["label"] = int(label)

        return data_dict

    def get_inference_loader(self,
                             image_path: str,
                             patient_id: Optional[str] = None,
                             label: Optional[int] = None,
                             num_workers: Optional[int] = None) -> DataLoader:
        """
        Get dataloader for single image inference (no labels expected).
        
        Args:
            image_path: Path to NIfTI image
            patient_id: Optional patient identifier
            label: Optional ground truth label (for evaluation)
            num_workers: Number of workers (overrides config)
            
        Returns:
            DataLoader with single sample (batch_size=1)
            
        Example:
            # Single image inference
            loader = data_module.get_inference_loader('/path/to/scan.nii.gz')
            
            for batch in loader:
                predictions = model(batch['image'])
                print(f"Prediction: {predictions.argmax()}")
        """
        # Create data dict for single sample
        data_dict = self._create_single_sample_dict(image_path, patient_id,
                                                    label)
        data_dicts = [data_dict]

        # Set num_workers
        num_workers = num_workers or self.config['compute']['num_workers']

        # Create dataset and loader (use val transforms - no augmentation)
        dataset = self._create_dataset(data_dicts, 'val', num_workers)
        loader = self._create_loader(dataset,
                                     'val',
                                     batch_size=1,
                                     num_workers=num_workers)

        logger.info(f"Created inference loader for: {image_path}")
        return loader


def create_data_module(config: Dict) -> ClassificationDataModule:
    """Factory function to create data module."""
    return ClassificationDataModule(config)
