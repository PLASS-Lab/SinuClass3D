# Dataset Structure

## Overview
Medical imaging dataset with DICOM to NIfTI conversion for sinus CT scans.

## Server Location
`/opt/datasets/sinus/`

## Directory Structure

```
/opt/datasets/sinus/
├── classification/                    # 184 patients
│   ├── {patient_id}/
│   │   ├── {patient_id}.nii.gz       # CT volume (int16)
│   │   └── dcm/original/             # Source DICOM files
│   └── ...
├── segmentation/                      # 20 patients  
│   ├── {patient_id}/
│   │   ├── {patient_id}.nii.gz       # CT volume (int16)
│   │   ├── {patient_id}_gt.nii.gz    # Binary mask (uint8)
│   │   └── dcm/
│   │       ├── original/             # Source DICOM files
│   │       └── label/                # Annotation DICOM files
│   └── ...
├── patient_ids_classification.csv     # Patient list for classification
└── patient_ids_segmentation.csv       # Patient list for segmentation
```

## File Details

### Classification Dataset
- **Patients**: 184 total
- **Files per patient**: 1 NIfTI volume
- **Data type**: int16 (CT imaging standard)
- **Average size**: ~58.6 MB per volume

### Segmentation Dataset
- **Patients**: 20 total
- **Files per patient**: 2 NIfTI files (original + ground truth)
- **Original data type**: int16 (CT imaging standard)
- **Ground truth data type**: uint8 (binary mask optimized)
- **Average sizes**: 
  - Original: ~56.5 MB
  - Ground truth: ~0.27 MB

## Conversion Status
- ✅ Classification: 184/184 patients (100% success)
- ✅ Segmentation: 20/20 patients (100% success)
- ✅ Total NIfTI files: 224 files
- ✅ Data types optimized for medical imaging workflows
