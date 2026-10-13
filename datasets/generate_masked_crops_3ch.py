#!/usr/bin/env python3
"""
Generate 3-channel masked crops for classification dataset.

Creates cropped CT volumes with:
- 3 channels: Air, Soft tissue, Bone (HU thresholds)
- Resampled to 1mm isotropic spacing
- Non-ROI regions blacked out using the mask
- Visualize with plotly

Output columns (1mm isotropic):
- bbox_5mm_1mm_iso_mask_path
- bbox_5mm_1mm_iso_masked_air_path
- bbox_5mm_1mm_iso_masked_soft_path  
- bbox_5mm_1mm_iso_masked_bone_path
- masked_hu_1ch_path (single-channel masked HU = sum of the three channels,
  the input of the single-channel ablation)
"""

import argparse
import csv
import logging
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# Optional plotly for visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_bbox_with_margin(mask, voxel_spacing, margin_mm=5):
    """Get bounding box of foreground with margin in mm."""
    foreground = np.where(mask > 0)
    
    if len(foreground[0]) == 0:
        logger.warning("No foreground found in mask!")
        return None
    
    mins = [np.min(foreground[i]) for i in range(3)]
    maxs = [np.max(foreground[i]) for i in range(3)]
    
    margin_voxels = [int(np.ceil(margin_mm / voxel_spacing[i])) for i in range(3)]
    
    bbox_slices = []
    for i in range(3):
        start = max(0, mins[i] - margin_voxels[i])
        end = min(mask.shape[i], maxs[i] + margin_voxels[i] + 1)
        bbox_slices.append(slice(start, end))
    
    return tuple(bbox_slices)


def resample_volume(volume, original_spacing, target_spacing, order=1):
    """
    Resample volume to target spacing.
    
    Args:
        volume: 3D numpy array
        original_spacing: Original voxel spacing (x, y, z)
        target_spacing: Target voxel spacing (x, y, z)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    """
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    resampled = zoom(volume, zoom_factors, order=order)
    return resampled


def process_patient(patient_id, image_path, label_path, output_dir, margin_mm=5, target_spacing=(1.0, 1.0, 1.0)):
    """
    Process a single patient to generate 3-channel masked crops.
    
    Pipeline (the order used to build the paper's inputs):
    1. Load image and mask
    2. Resample the whole volume to target spacing (1mm isotropic)
    3. Crop a bounding box with margin around the resampled mask
    4. Create 3 tissue channels with mask applied (non-ROI = 0)
    """
    try:
        logger.info(f"Processing patient {patient_id}")
        
        # Load image and label
        img_nii = nib.load(image_path)
        label_nii = nib.load(label_path)
        
        img_data = img_nii.get_fdata()
        label_data = label_nii.get_fdata()
        
        # Get original voxel spacing from affine
        voxel_spacing = np.abs(np.diag(img_nii.affine)[:3])
        logger.info(f"  Native spacing: {voxel_spacing}")
        logger.info(f"  Native shape: {img_data.shape}")
        
        # Resample the whole volume to target spacing (1mm isotropic) first
        img_iso = resample_volume(img_data, voxel_spacing, target_spacing, order=1)
        mask_iso = resample_volume((label_data > 0).astype(np.float64), voxel_spacing,
                                   target_spacing, order=0)  # Nearest for mask

        # Bounding box with margin around the resampled mask, then crop
        bbox = get_bbox_with_margin(mask_iso, target_spacing, margin_mm=margin_mm)
        if bbox is None:
            logger.error(f"Patient {patient_id}: No foreground found")
            return False, None, None

        img_resampled = img_iso[bbox]
        mask_resampled = (mask_iso[bbox] > 0.5).astype(np.float32)
        
        logger.info(f"  Resampled shape (1mm iso): {img_resampled.shape}")
        
        # Create new affine for resampled image (1mm isotropic)
        affine_iso = np.diag([target_spacing[0], target_spacing[1], target_spacing[2], 1.0])
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Output paths - 3 channels + mask (1mm isotropic)
        output_paths = {
            'bbox_5mm_1mm_iso_masked_air_path': os.path.join(output_dir, f"{patient_id}_bbox_5mm_1mm_iso_masked_air.nii.gz"),
            'bbox_5mm_1mm_iso_masked_soft_path': os.path.join(output_dir, f"{patient_id}_bbox_5mm_1mm_iso_masked_soft.nii.gz"),
            'bbox_5mm_1mm_iso_masked_bone_path': os.path.join(output_dir, f"{patient_id}_bbox_5mm_1mm_iso_masked_bone.nii.gz"),
            'bbox_5mm_1mm_iso_mask_path': os.path.join(output_dir, f"{patient_id}_bbox_5mm_1mm_iso_mask.nii.gz"),
            'masked_hu_1ch_path': os.path.join(output_dir, f"{patient_id}_bbox_5mm_1mm_iso_masked_hu.nii.gz"),
        }
        
        # 1. Air channel: ≤ -150 HU, masked
        air_crop = np.where(img_resampled <= -150, img_resampled, 0) * mask_resampled
        nib.save(nib.Nifti1Image(air_crop.astype(np.float32), affine_iso), output_paths['bbox_5mm_1mm_iso_masked_air_path'])
        
        # 2. Soft tissue channel: -150 to 250 HU, masked
        soft_crop = np.where((img_resampled > -150) & (img_resampled < 250), img_resampled, 0) * mask_resampled
        nib.save(nib.Nifti1Image(soft_crop.astype(np.float32), affine_iso), output_paths['bbox_5mm_1mm_iso_masked_soft_path'])
        
        # 3. Bone channel: ≥ 250 HU, masked
        bone_crop = np.where(img_resampled >= 250, img_resampled, 0) * mask_resampled
        nib.save(nib.Nifti1Image(bone_crop.astype(np.float32), affine_iso), output_paths['bbox_5mm_1mm_iso_masked_bone_path'])
        
        # 4. Resampled mask
        nib.save(nib.Nifti1Image(mask_resampled.astype(np.float32), affine_iso), output_paths['bbox_5mm_1mm_iso_mask_path'])

        # 5. Single-channel masked HU (the thresholds partition HU, so this
        #    equals air + soft + bone)
        hu_crop = air_crop + soft_crop + bone_crop
        nib.save(nib.Nifti1Image(hu_crop.astype(np.float32), affine_iso), output_paths['masked_hu_1ch_path'])
        
        logger.info(f"  ✅ Saved 3 channels + mask (1mm iso): {img_resampled.shape}")
        
        return True, {
            'air': air_crop,
            'soft': soft_crop,
            'bone': bone_crop,
            'mask': mask_resampled,
            'shape': img_resampled.shape
        }, output_paths
        
    except Exception as e:
        logger.error(f"Patient {patient_id}: Error - {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


def visualize_3d_volume_plotly(data_dict, patient_id, output_path=None, max_points=64):
    """
    Visualize 3-channel masked crops in 3D with toggle buttons for each channel.
    Uses sparse sampling but preserves original coordinate space.
    """
    air = data_dict['air']
    soft = data_dict['soft']
    bone = data_dict['bone']
    orig_shape = air.shape
    
    # Sparse sampling: use fewer points but keep original coordinate range
    step = [max(1, s // max_points) for s in orig_shape]
    if any(s > 1 for s in step):
        air = air[::step[0], ::step[1], ::step[2]]
        soft = soft[::step[0], ::step[1], ::step[2]]
        bone = bone[::step[0], ::step[1], ::step[2]]
        logger.info(f"  Sparse sampling for viz: {orig_shape} -> {air.shape} (step={step})")
    
    # Create coordinate grids with ORIGINAL size range
    x, y, z = np.mgrid[0:orig_shape[0]:step[0], 0:orig_shape[1]:step[1], 0:orig_shape[2]:step[2]]
    
    fig = go.Figure()
    
    # Calculate thresholds
    air_nonzero = air[air < 0]  # Air is negative HU
    air_threshold = np.percentile(air_nonzero, 50) if len(air_nonzero) > 0 else -500
    
    soft_nonzero = soft[soft > 0]
    soft_threshold = np.percentile(soft_nonzero, 20) if len(soft_nonzero) > 0 else 0
    
    bone_nonzero = bone[bone > 0]
    bone_threshold = np.percentile(bone_nonzero, 30) if len(bone_nonzero) > 0 else 250
    
    # Air isosurface (visible first)
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=air.flatten(),
        isomin=air.min(),
        isomax=air_threshold,
        surface_count=3,
        colorscale='Purples',
        opacity=0.7,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Air (≤-150 HU)',
        visible=True
    ))
    
    # Soft tissue isosurface (hidden initially)
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=soft.flatten(),
        isomin=soft_threshold,
        isomax=soft.max(),
        surface_count=3,
        colorscale='Blues',
        opacity=0.7,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Soft (-150~250 HU)',
        visible=False
    ))
    
    # Bone isosurface (hidden initially)
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=bone.flatten(),
        isomin=bone_threshold,
        isomax=bone.max(),
        surface_count=3,
        colorscale='Hot',
        opacity=0.7,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Bone (≥250 HU)',
        visible=False
    ))
    
    # Create toggle buttons
    fig.update_layout(
        title=f'Patient {patient_id} - 3D Masked Crops (Shape: {data_dict["shape"]})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data'
        ),
        height=800,
        width=1000,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.5,
                y=1.15,
                xanchor="center",
                buttons=[
                    dict(
                        label="Air (≤-150 HU)",
                        method="update",
                        args=[{"visible": [True, False, False]}]
                    ),
                    dict(
                        label="Soft (-150~250 HU)",
                        method="update",
                        args=[{"visible": [False, True, False]}]
                    ),
                    dict(
                        label="Bone (≥250 HU)",
                        method="update",
                        args=[{"visible": [False, False, True]}]
                    ),
                    dict(
                        label="All",
                        method="update",
                        args=[{"visible": [True, True, True]}]
                    ),
                ]
            )
        ]
    )
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"  📊 Saved 3D visualization to {output_path}")
    
    return fig


def visualize_3ch_slices_plotly(data_dict, patient_id, output_path=None):
    """
    Create interactive slider visualization for all slices.
    """
    air = data_dict['air']
    soft = data_dict['soft']
    bone = data_dict['bone']
    
    n_slices = air.shape[2]
    
    # Create frames for each slice
    frames = []
    for z in range(n_slices):
        frame = go.Frame(
            data=[
                go.Heatmap(z=air[:, :, z].T, colorscale='Blues', showscale=False, name='Air'),
                go.Heatmap(z=soft[:, :, z].T, colorscale='Greys', showscale=False, name='Soft'),
                go.Heatmap(z=bone[:, :, z].T, colorscale='Hot', showscale=False, name='Bone'),
            ],
            name=str(z)
        )
        frames.append(frame)
    
    # Initial data (middle slice)
    z_mid = n_slices // 2
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Air (≤-150 HU)', 'Soft (-150~250 HU)', 'Bone (≥250 HU)'),
        horizontal_spacing=0.05
    )
    
    fig.add_trace(
        go.Heatmap(z=air[:, :, z_mid].T, colorscale='Blues', showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=soft[:, :, z_mid].T, colorscale='Greys', showscale=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=bone[:, :, z_mid].T, colorscale='Hot', showscale=False),
        row=1, col=3
    )
    
    # Add slider
    sliders = [{
        'active': z_mid,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'prefix': 'Slice: ',
            'visible': True,
            'xanchor': 'right'
        },
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.05,
        'y': 0,
        'steps': [
            {'args': [[str(z)], {'frame': {'duration': 0}, 'mode': 'immediate'}],
             'label': str(z),
             'method': 'animate'}
            for z in range(n_slices)
        ]
    }]
    
    fig.update_layout(
        title=f'Patient {patient_id} - 3-Channel Masked (Shape: {data_dict["shape"]})',
        height=500,
        width=1200,
        sliders=sliders
    )
    fig.frames = frames
    
    # Update axes
    for j in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=1, col=j)
        fig.update_yaxes(showticklabels=False, scaleanchor=f"x{j}", row=1, col=j)
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"  📊 Saved interactive visualization to {output_path}")
    
    return fig


def main():
    """Main function - process patients and update CSV with new columns."""
    parser = argparse.ArgumentParser(description="Generate 3-channel masked crops with visualization")
    parser.add_argument("--csv", type=str, default="csv/patient_ids_classification.csv")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV path (default: overwrite input)")
    parser.add_argument("--patient-id", type=str, default=None, help="Process specific patient")
    parser.add_argument("--n", type=int, default=None, help="Number of patients to process (default: all)")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="visualization_output")
    args = parser.parse_args()
    
    csv_path = args.csv
    output_csv_path = args.output_csv or csv_path
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
    
    logger.info(f"Found {len(all_rows)} patients in CSV")
    
    # Add new columns if not present (1mm isotropic)
    new_columns = [
        'bbox_5mm_1mm_iso_masked_air_path',
        'bbox_5mm_1mm_iso_masked_soft_path',
        'bbox_5mm_1mm_iso_masked_bone_path',
        'bbox_5mm_1mm_iso_mask_path',
        'masked_hu_1ch_path'
    ]
    for col in new_columns:
        if col not in fieldnames:
            fieldnames = list(fieldnames) + [col]
    
    # Determine which rows to process
    if args.patient_id:
        process_ids = {args.patient_id}
    elif args.n:
        process_ids = {r['patient_id'] for r in all_rows[:args.n]}
    else:
        process_ids = {r['patient_id'] for r in all_rows}
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    for row in all_rows:
        patient_id = row['patient_id']
        
        if patient_id not in process_ids:
            continue
            
        image_path = row['image_path']
        mask_path = row['mask_path']
        output_dir = os.path.dirname(image_path)
        
        logger.info(f"\nProcessing {patient_id}")
        
        success, data_dict, output_paths = process_patient(
            patient_id,
            image_path,
            mask_path,
            output_dir,
            margin_mm=5
        )
        
        if success:
            # Update row with new paths
            row.update(output_paths)
            success_count += 1
            
            if args.visualize:
                if not HAS_PLOTLY:
                    logger.warning("Plotly not installed. Skipping visualization. Install with: pip install plotly")
                else:
                    # Generate 3D visualization
                    viz_3d_path = os.path.join(args.output_dir, f"{patient_id}_3d_volume.html")
                    visualize_3d_volume_plotly(data_dict, patient_id, viz_3d_path)
                    
                    # Also generate slice slider view
                    viz_path = os.path.join(args.output_dir, f"{patient_id}_3ch_slices.html")
                    visualize_3ch_slices_plotly(data_dict, patient_id, viz_path)
    
    # Write updated CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    logger.info(f"\n✅ Done! Processed {success_count}/{len(process_ids)} patients")
    logger.info(f"📄 Updated CSV: {output_csv_path}")
    if args.visualize:
        logger.info(f"📊 Visualizations: {args.output_dir}/")


if __name__ == '__main__':
    main()
