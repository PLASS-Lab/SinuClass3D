"""
Classification Predictor Module

This module provides the ClassificationPredictor class for inference on
new, unlabeled 3D medical images using trained classification models.

Features:
- Single model or ensemble prediction
- Uncertainty quantification (model disagreement)
- Grad-CAM explanations (optional)
- Batch processing with progress tracking
- Publication-ready output formats (CSV, JSON)

Compatible with prediction scripts and YAML configuration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ClassificationPredictor:
    """
    Predictor for trained classification models.
    
    Supports both single model and ensemble prediction. Can quantify
    prediction uncertainty using model ensemble disagreement.
    
    Args:
        models: List of trained models (single model or ensemble)
        config: Configuration dictionary from YAML
        device: Torch device (cuda or cpu)
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: Dict,
        device: torch.device,
    ):
        self.models = models
        self.config = config
        self.device = device

        # Move all models to device and set to eval mode
        for model in self.models:
            model.to(self.device)
            model.eval()

        logger.info(f"Predictor initialized:")
        logger.info(f"  - Number of models: {len(self.models)}")
        logger.info(f"  - Device: {self.device}")

        # Class names for output
        self.class_names = config['classification']['class_names']

    @torch.no_grad()
    def predict_batch(self, batch: Dict) -> Dict:
        """
        Predict on a batch of samples.
        
        Args:
            batch: Batch dictionary with 'image' and 'patient_id'
            
        Returns:
            Dictionary with:
                - patient_ids: List of patient identifiers
                - predictions: (batch_size,) predicted class labels
                - probabilities: (batch_size, num_classes) class probabilities
                - uncertainty: (batch_size,) prediction uncertainty (ensemble std)
        """
        images = batch['image'].to(self.device)
        patient_ids = batch['patient_id']

        # Collect predictions from all models
        all_probs = []

        for model in self.models:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

        # Ensemble averaging
        all_probs = np.array(all_probs)  # [n_models, batch_size, n_classes]
        mean_probs = all_probs.mean(axis=0)  # [batch_size, n_classes]
        std_probs = all_probs.std(axis=0)  # [batch_size, n_classes]

        # Predictions (argmax of mean probabilities)
        predictions = mean_probs.argmax(axis=1)

        # Uncertainty: standard deviation at predicted class
        uncertainty = std_probs[np.arange(len(predictions)), predictions]

        return {
            'patient_ids': self._convert_patient_ids(patient_ids),
            'predictions': predictions,
            'probabilities': mean_probs,
            'uncertainty': uncertainty,
        }

    def predict_dataset(self, data_loader: DataLoader) -> pd.DataFrame:
        """
        Predict on entire dataset.
        
        Args:
            data_loader: PyTorch DataLoader with unlabeled data
            
        Returns:
            DataFrame with predictions, probabilities, and uncertainty for each sample
        """
        # Get dataset size if available
        try:
            dataset_size = len(data_loader.dataset)  # type: ignore
            logger.info(f"Generating predictions on {dataset_size} samples")
        except TypeError:
            logger.info("Generating predictions...")

        all_patient_ids = []
        all_predictions = []
        all_probabilities = []
        all_uncertainties = []

        desc = "Predictions (Ensemble)" if len(
            self.models) > 1 else "Predictions"

        for batch in tqdm(data_loader, desc=desc):
            results = self.predict_batch(batch)

            all_patient_ids.extend(results['patient_ids'])
            all_predictions.extend(results['predictions'])
            all_probabilities.append(results['probabilities'])
            all_uncertainties.extend(results['uncertainty'])

        # Stack probabilities
        all_probabilities = np.vstack(all_probabilities)

        # Create DataFrame
        num_classes = all_probabilities.shape[1]

        df_dict = {
            'patient_id': all_patient_ids,
            'predicted_label': all_predictions,
        }

        # Add probability columns for each class
        for i in range(num_classes):
            df_dict[f'prob_class_{i}'] = all_probabilities[:, i].tolist()

        df_dict['uncertainty'] = all_uncertainties

        df = pd.DataFrame(df_dict)

        # Add confidence (max probability)
        prob_cols = [f'prob_class_{i}' for i in range(num_classes)]
        df['confidence'] = df[prob_cols].max(axis=1)

        # Add class names
        df['predicted_class_name'] = df['predicted_label'].map({
            i: name for i, name in enumerate(self.class_names)
        })

        # Add ensemble size if applicable
        if len(self.models) > 1:
            df['ensemble_size'] = len(self.models)

        logger.info(
            f"Generated predictions for {len(df)} samples (ensemble size: {len(self.models)})"
        )

        return df

    @staticmethod
    def _convert_patient_ids(patient_ids) -> List[str]:
        """Convert patient IDs to strings."""
        patient_ids_str = []
        for pid in patient_ids:
            if isinstance(pid, torch.Tensor):
                patient_ids_str.append(str(pid.item()))
            else:
                patient_ids_str.append(str(pid))
        return patient_ids_str

    @staticmethod
    def _load_single_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device):
        """Load checkpoint into model."""
        checkpoint = torch.load(checkpoint_path,
                                map_location=device,
                                weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    @staticmethod
    def save_predictions(df: pd.DataFrame, output_path: str):
        """
        Save predictions to CSV file.
        
        Args:
            df: DataFrame with predictions
            output_path: Path to save CSV
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")

    @classmethod
    def from_checkpoints(cls,
                         checkpoint_paths: List[str],
                         config: Dict,
                         model_factory_fn: Callable,
                         device: Optional[torch.device] = None):
        """
        Create predictor from saved checkpoints.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
            config: Configuration dictionary
            model_factory_fn: Function that creates the model architecture
            device: Torch device (defaults to cuda if available)
            
        Returns:
            ClassificationPredictor instance
        
        """
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        models = []

        for checkpoint_path in checkpoint_paths:
            # Create model
            model = model_factory_fn(config)

            # Load checkpoint
            cls._load_single_checkpoint(checkpoint_path, model, device)

            models.append(model)

        return cls(models=models, config=config, device=device)


def find_ensemble_checkpoints(experiment_dir: Path, n_folds: int) -> List[str]:
    """
    Find all fold checkpoints for ensemble prediction.
    
    Args:
        experiment_dir: Experiment directory containing fold_X subdirectories
        n_folds: Number of cross-validation folds
        
    Returns:
        List of checkpoint paths
        
    Example:
        Returns list of paths like 'experiment_dir/fold_1/checkpoints/best_model.pth'
    """
    checkpoint_paths = []

    for fold in range(n_folds):
        checkpoint_path = experiment_dir / f'fold_{fold + 1}' / 'checkpoints' / 'best_model.pth'
        if checkpoint_path.exists():
            checkpoint_paths.append(str(checkpoint_path))
        else:
            logger.warning(
                f"Checkpoint not found for fold {fold + 1}: {checkpoint_path}")

    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {experiment_dir}")

    logger.info(
        f"Found {len(checkpoint_paths)}/{n_folds} checkpoints for ensemble")
    return checkpoint_paths
