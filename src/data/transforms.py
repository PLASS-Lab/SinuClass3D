"""
3D Classification Transforms for CT Imaging

CT-specific transforms:
- HU windowing and per-volume z-score normalization (leak-safe)
- RAS reorientation and isotropic resampling
- Anatomically valid spatial augmentation (no SI flips)
- Mild intensity augmentation (CT-specific)

References:
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- MONAI: https://docs.monai.io/en/stable/transforms.html
"""

import logging
from typing import Dict, List

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    Transform,
)

logger = logging.getLogger(__name__)


class ConcatChannelsd(MapTransform):
    """
    Concatenate multiple image channels into a single multi-channel image.
    
    Used for dual-window CT input where soft-tissue and bone windows
    are loaded separately and need to be combined.
    """
    
    def __init__(self, keys: List[str], output_key: str = "image"):
        """
        Args:
            keys: List of keys to concatenate (e.g., ["image_ch0", "image_ch1"])
            output_key: Key for the concatenated output (default: "image")
        """
        super().__init__(keys)
        self.output_key = output_key
    
    def __call__(self, data):
        d = dict(data)
        # Stack channels along channel dimension (dim 0)
        channels = [d[key] for key in self.keys]
        d[self.output_key] = np.concatenate(channels, axis=0)
        # Remove original keys
        for key in self.keys:
            if key != self.output_key:
                del d[key]
        return d


def build_transforms(config: Dict, mode: str = "train") -> Compose:
    """
    Build MONAI transform pipeline for 3D CT classification.
    
    Pipeline stages:
    1. Load & channel handling
    2. HU windowing (CT-specific)
    3. Spatial standardization (RAS + resampling)
    4. Intensity normalization (per-volume z-score, leak-safe)
    5. Spatial augmentation (train only, anatomically valid)
    6. Intensity augmentation (train only, CT-specific)
    7. Convert to tensor
    
    Args:
        config: Configuration dictionary from YAML
        mode: "train", "val", or "test"
               - train: Full pipeline with augmentation
               - val: Preprocessing only (validation during training)
               - test: Preprocessing only (evaluation and inference)
        
    Returns:
        MONAI Compose transform
        
    Raises:
        ValueError: If mode is not one of the valid options
        KeyError: If required config keys are missing
        
    Note:
        Use mode="test" for both evaluation (with labels) and inference (without labels).
        The transform pipeline is identical - only the downstream usage differs.
        
    Example:
        >>> from src.data.transforms import build_transforms
        >>> train_transforms = build_transforms(config, mode="train")
        >>> val_transforms = build_transforms(config, mode="val")
        >>> test_transforms = build_transforms(config, mode="test")  # For eval & inference
    """
    # Validate mode
    valid_modes = ["train", "val", "test"]
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of {valid_modes}. "
            f"Note: Use 'test' for both evaluation and inference.")

    logger.info(f"Building transforms for mode: {mode}")
    # =========================================================================
    # CONFIG PARSING (Handle nested structure)
    # =========================================================================
    if 'transforms' in config:
        cfg_transforms = config['transforms']
    elif 'data' in config and 'transforms' in config['data']:
        cfg_transforms = config['data']['transforms']
    else:
        raise KeyError(
            f"Cannot find 'transforms' in config. Available keys: {list(config.keys())}"
        )

    cfg_preproc = cfg_transforms['preprocessing']
    
    # Check for multi-channel modes
    image_type = config.get('data', {}).get('image_type', '')
    multi_channel_mode = (image_type == 'multi_channel')
    dual_window_mode = (image_type == 'dual_window')

    # =========================================================================
    # 1. BASE PREPROCESSING TRANSFORMS (Always Applied, Leak-Safe)
    # =========================================================================
    if multi_channel_mode:
        # Multi-channel: load N channels dynamically from config
        multi_ch_config = config.get('data', {}).get('multi_channel', {})
        num_channels = sum(1 for i in range(10) if f'channel_{i}' in multi_ch_config)
        load_keys = [f"image_ch{i}" for i in range(num_channels)]
        keys = ["image"]  # After concatenation
        logger.info(f"Multi-channel mode ({num_channels}ch): {', '.join(load_keys)}")
        
        transforms: List[Transform] = [
            LoadImaged(keys=load_keys, image_only=False),
            EnsureChannelFirstd(keys=load_keys),
            ConcatChannelsd(keys=load_keys, output_key="image"),  # Concat to N-channel
        ]
    elif dual_window_mode:
        # Dual window: load both channels separately, then concatenate
        load_keys = ["image_ch0", "image_ch1"]
        keys = ["image"]  # After concatenation
        logger.info("Dual window mode: loading image_ch0 (soft-tissue) + image_ch1 (bone)")
        
        transforms: List[Transform] = [
            LoadImaged(keys=load_keys, image_only=False),
            EnsureChannelFirstd(keys=load_keys),
            ConcatChannelsd(keys=load_keys, output_key="image"),  # Concat to 2-channel
        ]
    else:
        # Single channel mode
        keys = ["image"]
        transforms: List[Transform] = [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
        ]

    # =========================================================================
    # 2. HU WINDOWING (CT-Specific)
    # =========================================================================
    # REPORT: HU [-1000, 1000] scaled to [-1, 1]
    # Note: Skip for multi-channel modes (windowing already applied during preprocessing)
    if cfg_preproc['hu_windowing']['enabled'] and not dual_window_mode and not multi_channel_mode:
        transforms.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=cfg_preproc['hu_windowing']['window_min'],
                a_max=cfg_preproc['hu_windowing']['window_max'],
                b_min=cfg_preproc['hu_windowing']['scale_min'],
                b_max=cfg_preproc['hu_windowing']['scale_max'],
                clip=True,
            ))
    elif multi_channel_mode or dual_window_mode:
        # For multi-channel/dual-window, normalize each channel independently
        # Tissue-specific channels already have isolated HU values
        transforms.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=False,
                channel_wise=True,  # Normalize each channel independently
            ))

    # =========================================================================
    # 3. SPATIAL STANDARDIZATION
    # =========================================================================
    # 3a. Reorient to RAS (Right-Anterior-Superior)
    if cfg_preproc['spatial']['reorient_to_ras']:
        transforms.append(
            Orientationd(
                keys=keys,
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")
                       )  # Explicit labels to avoid MONAI warning
            ))

    # 3b. Resample to isotropic spacing
    if cfg_preproc['spatial']['resample']['enabled']:
        target_spacing = cfg_preproc['spatial']['resample']['target_spacing']
        # Set interpolation mode based on number of keys
        interp_mode = ["bilinear"] * len(keys) if len(keys) == 1 else [
            "bilinear", "nearest"
        ]
        transforms.append(
            Spacingd(
                keys=keys,
                pixdim=target_spacing,
                mode=interp_mode,  # Renamed to avoid conflict with dataset mode
            ))

    # =========================================================================
    # 4. INTENSITY NORMALIZATION (Per-Volume Z-Score, Leak-Safe!)
    # =========================================================================
    # Note: Full volume processing - no cropping needed (volumes fit in GPU)
    logger.info(f"{mode} mode: Loading full volume (no cropping needed)")
    if cfg_preproc['normalization']['method'] == 'zscore_per_volume':
        transforms.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,  # Only normalize non-zero voxels
                channel_wise=True,  # LEAK-SAFE: per volume, not per dataset
            ))

    # =========================================================================
    # 5. SPATIAL AUGMENTATION (Training Only, Anatomically Valid)
    # =========================================================================
    # Patient-level, no slice mixing
    # Only apply augmentation during training
    if mode != "train":
        logger.debug(f"{mode} mode: Skipping spatial augmentation")
    elif (cfg_transforms['train_augmentation']['enabled']):
        cfg_aug = cfg_transforms['train_augmentation']
        logger.debug("Applying spatial augmentation")

        # 5a. Random Flips: LR & AP only (NOT SI)
        if (cfg_aug['random_flip']['enabled']):
            transforms.append(
                RandFlipd(
                    keys=["image"],
                    spatial_axis=cfg_aug['random_flip']
                    ['axes'],  # [0, 1] = LR, AP only
                    prob=cfg_aug['random_flip']['prob'],  # p=0.5
                ))

        # 5b. Random Affine: modest rotation/scale following nnU-Net
        if (cfg_aug['random_affine']['enabled']):
            # Separate in-plane and through-plane translation
            rotate_deg = cfg_aug['random_affine']['rotate_range']
            scale_range = cfg_aug['random_affine']['scale_range']

            # Build translate range: [x_min, x_max, y_min, y_max, z_min, z_max]
            translate_xy = cfg_aug['random_affine']['translate_range_xy']
            translate_z = cfg_aug['random_affine']['translate_range_z']

            transforms.append(
                RandAffined(
                    keys=["image"],
                    rotate_range=[
                        np.deg2rad(rotate_deg[0]),
                        np.deg2rad(rotate_deg[1])
                    ] * 3,
                    scale_range=[scale_range[0] - 1, scale_range[1] - 1],
                    translate_range=[
                        -translate_xy[0],
                        translate_xy[1],  # X in-plane
                        -translate_xy[0],
                        translate_xy[1],  # Y in-plane
                        -translate_z[0],
                        translate_z[1],  # Z through-plane
                    ],
                    prob=cfg_aug['random_affine']['prob'],  # p=0.25
                    mode=cfg_aug['random_affine']['mode'],
                    padding_mode=cfg_aug['random_affine']['padding_mode'],
                ))

    # =========================================================================
    # 6. INTENSITY AUGMENTATION (Training Only, CT-Specific, Mild)
    # =========================================================================
    # Only apply augmentation during training
    if mode != "train":
        logger.debug(f"{mode} mode: Skipping intensity augmentation")
    elif (cfg_transforms['intensity_augmentation']['enabled']):
        cfg_intensity = cfg_transforms['intensity_augmentation']
        logger.debug("Applying intensity augmentation")

        # 6a. Gaussian Noise (simulates acquisition noise)
        if (cfg_intensity['random_gaussian_noise']['enabled']):
            transforms.append(
                RandGaussianNoised(
                    keys=["image"],
                    mean=cfg_intensity['random_gaussian_noise']['mean'],
                    std=cfg_intensity['random_gaussian_noise']['std'],  # σ=0.02
                    prob=cfg_intensity['random_gaussian_noise']
                    ['prob'],  # p=0.15
                ))

        # 6b. Gaussian Smoothing (simulates kernel variance)
        if (cfg_intensity['random_gaussian_smooth']['enabled']):
            sigma_range = cfg_intensity['random_gaussian_smooth']['sigma_range']
            transforms.append(
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=tuple(sigma_range),
                    sigma_y=tuple(sigma_range),
                    sigma_z=tuple(sigma_range),
                    prob=cfg_intensity['random_gaussian_smooth']
                    ['prob'],  # p=0.2
                ))

        # 6c. Scale Intensity (multiplicative)
        if (cfg_intensity['random_scale_intensity']['enabled']):
            factors = cfg_intensity['random_scale_intensity']['factors']
            transforms.append(
                RandScaleIntensityd(
                    keys=["image"],
                    factors=tuple(factors),  # 0.9-1.1
                    prob=cfg_intensity['random_scale_intensity']
                    ['prob'],  # p=0.2
                ))

        # 6d. Shift Intensity (additive)
        if (cfg_intensity['random_shift_intensity']['enabled']):
            offsets = cfg_intensity['random_shift_intensity']['offsets']
            transforms.append(
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=tuple(offsets),  # ±0.05
                    prob=cfg_intensity['random_shift_intensity']
                    ['prob'],  # p=0.2
                ))

    # =========================================================================
    # 7. FINAL CONVERSION
    # =========================================================================
    transforms.append(ToTensord(keys=["image"]))

    return Compose(transforms)
