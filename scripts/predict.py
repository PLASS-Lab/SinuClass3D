"""
3D Classification Inference Script

Performs inference on new, unlabeled CT scans using trained models.

Features:
- Single model or ensemble (multiple folds) prediction
- Uncertainty quantification (prediction variance)
- Batch processing with progress tracking
- Output as CSV

predicted_label is the argmax of the (mean) raw softmax probabilities. In
ensemble mode the script also applies the paper's operating point, as
scripts/evaluate.py does for the held-out cohort: the averaged raw probability
is Platt-calibrated with the calibrator fitted on pooled out-of-fold
predictions and thresholded at the Youden's J value locked during
cross-validation (calibrated_prob_class_1, threshold,
calibrated_predicted_label, calibrated_predicted_class_name). Single-model
mode has no matching calibrator and reports raw probabilities only.

Usage (the prediction.* keys are not in the default config, hence the "+"):
    # Single model prediction
    python scripts/predict.py \
        +prediction.use_ensemble=false \
        +prediction.checkpoint_path=experiments/<experiment_name>/fold_1/checkpoints/best_model.pth \
        +prediction.input_csv=data/new_patients.csv \
        +prediction.output_dir=predictions/

    # Ensemble prediction (all CV folds)
    python scripts/predict.py \
        +prediction.use_ensemble=true \
        +prediction.experiment_dir=experiments/<experiment_name> \
        +prediction.input_csv=data/new_patients.csv \
        +prediction.output_dir=predictions/
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import hydra
import joblib
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from src.data import create_data_module, prepare_csv_for_training
from src.inference import ClassificationPredictor
from src.models import get_model
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def find_ensemble_checkpoints(experiment_dir: Path, n_folds: int) -> List[str]:
    """
    Find all fold checkpoints for ensemble prediction.

    Args:
        experiment_dir: Experiment directory containing fold_X subdirectories
        n_folds: Number of folds

    Returns:
        List of checkpoint paths
    """
    # Import to avoid duplication
    from src.inference import find_ensemble_checkpoints as find_checkpoints
    return find_checkpoints(experiment_dir, n_folds)


def add_calibrated_predictions(predictions_df, experiment_dir: Path,
                               class_names: List[str]):
    """
    Add the calibrated operating point to ensemble predictions.

    Mirrors scripts/evaluate.py: Platt-calibrate the averaged raw probability
    of the positive class once, then threshold it at the Youden's J value
    locked from internal cross-validation.
    """
    calibration_dir = experiment_dir / 'calibration'
    calibrator_path = calibration_dir / 'calibrator.pkl'
    config_path = calibration_dir / 'calibration_config.yaml'
    if not (calibrator_path.exists() and config_path.exists()):
        logger.warning(f"No calibrator in {calibration_dir}; "
                       "reporting raw probabilities only")
        return predictions_df

    calibrator = joblib.load(calibrator_path)
    with open(config_path) as f:
        threshold = yaml.safe_load(f)['calibrated_threshold']

    eps = 1e-15
    raw = np.clip(predictions_df['prob_class_1'].to_numpy(), eps, 1 - eps)
    logits = np.log(raw / (1 - raw))
    calibrated = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    labels = (calibrated >= threshold).astype(int)

    predictions_df['calibrated_prob_class_1'] = calibrated
    predictions_df['threshold'] = threshold
    predictions_df['calibrated_predicted_label'] = labels
    predictions_df['calibrated_predicted_class_name'] = [class_names[i] for i in labels]
    logger.info(f"Calibrated with {calibrator_path} at locked threshold {threshold:.4f}")
    return predictions_df


@hydra.main(version_base=None, config_path="pkg://configs", config_name="config_3d")
def main(cfg: DictConfig):
    """
    Main prediction function.

    Supports two modes:
    1. Single model prediction (prediction.checkpoint_path specified)
    2. Ensemble prediction (prediction.use_ensemble=true)
    """

    logger.info("=" * 80)
    logger.info("3D CLASSIFICATION INFERENCE")
    logger.info("=" * 80)

    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    # Set seed for reproducibility
    reproducibility_config = cfg_dict['reproducibility']
    seed = reproducibility_config['random_seed']
    deterministic = reproducibility_config['deterministic']
    set_seed(seed, deterministic)
    logger.info(f"Random seed: {seed}")
    logger.info(f"Deterministic mode: {deterministic}")

    # Get prediction config
    pred_config = cfg_dict['prediction']

    # Input CSV (required for prediction)
    input_csv = pred_config['input_csv']
    if not input_csv or not Path(input_csv).exists():
        raise ValueError(
            f"Input CSV not found: {input_csv}. "
            "Please specify prediction.input_csv"
        )

    logger.info(f"Input CSV: {input_csv}")

    # Output directory
    output_dir = Path(pred_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Determine checkpoint paths
    use_ensemble = pred_config['use_ensemble']

    if use_ensemble:
        # Ensemble mode: use all folds
        logger.info("Mode: ENSEMBLE (all folds)")

        experiment_dir = Path(pred_config['experiment_dir'])
        if not experiment_dir.exists():
            raise ValueError(f"Experiment directory not found: {experiment_dir}")

        n_folds = cfg_dict['data']['cross_validation']['n_folds']
        checkpoint_paths = find_ensemble_checkpoints(experiment_dir, n_folds)

    else:
        # Single model mode
        logger.info("Mode: SINGLE MODEL")

        checkpoint_path = pred_config['checkpoint_path']
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Please specify prediction.checkpoint_path"
            )

        checkpoint_paths = [checkpoint_path]

    # Create predictor from checkpoints
    predictor = ClassificationPredictor.from_checkpoints(
        checkpoint_paths=checkpoint_paths,
        config=cfg_dict,
        model_factory_fn=get_model,
    )

    # Create data loader for inference
    # Note: For prediction, we need a special data loader that:
    # 1. Loads from input_csv (without labels)
    # 2. Returns patient_id with each sample

    # Update config to use inference CSV
    cfg_dict['data']['paths']['csv_path'] = input_csv  # type: ignore
    cfg_dict['data']['inference_mode'] = True  # type: ignore # No labels expected

    # Preprocess CSV: Convert specific image type column to generic 'image_path'
    preprocessed_csv_path = prepare_csv_for_training(cfg_dict)
    cfg_dict['data']['paths']['csv_path'] = preprocessed_csv_path  # type: ignore
    logger.info(f"Using preprocessed CSV: {preprocessed_csv_path}")
    logger.info(f"Image type: {cfg_dict['data']['image_type']}")

    data_module = create_data_module(cfg_dict)

    # Get inference data loader (loads all samples from CSV, no CV split)
    inference_loader = data_module.get_test_loader()

    # Run predictions
    logger.info("\nGenerating predictions...")
    predictions_df = predictor.predict_dataset(inference_loader)
    if use_ensemble:
        predictions_df = add_calibrated_predictions(
            predictions_df, experiment_dir, cfg_dict['classification']['class_names'])

    # Save results
    output_csv = output_dir / 'predictions.csv'
    predictions_df.to_csv(output_csv, index=False)
    logger.info(f"\nPredictions saved to: {output_csv}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(predictions_df)}")
    logger.info(f"Class distribution:")
    for label, count in predictions_df['predicted_label'].value_counts().items():
        class_name = predictions_df[predictions_df['predicted_label'] == label]['predicted_class_name'].iloc[0]
        logger.info(f"  {class_name}: {count} ({count/len(predictions_df)*100:.1f}%)")

    logger.info(f"\nMean confidence: {predictions_df['confidence'].mean():.4f}")
    logger.info(f"Mean uncertainty: {predictions_df['uncertainty'].mean():.4f}")

    # Flag uncertain predictions
    high_uncertainty_threshold = 0.1  # Can be configured
    uncertain_samples = predictions_df[predictions_df['uncertainty'] > high_uncertainty_threshold]

    if len(uncertain_samples) > 0:
        logger.warning(f"\n⚠️  {len(uncertain_samples)} samples with high uncertainty (>{high_uncertainty_threshold}):")
        logger.warning(f"  Consider manual review: {list(uncertain_samples['patient_id'])}")

        # Save uncertain samples separately
        uncertain_csv = output_dir / 'predictions_uncertain.csv'
        uncertain_samples.to_csv(uncertain_csv, index=False)
        logger.info(f"  Uncertain samples saved to: {uncertain_csv}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Inference completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
