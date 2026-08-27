#!/usr/bin/env python3
"""
Segmentation DICOM to NIfTI Converter

A professional-grade medical imaging converter that transforms segmentation DICOM datasets
from /opt/datasets/sinus_test/segmentation/{label,original} into properly formatted 3D NIfTI 
volumes at /opt/datasets/sinus with binary ground truth masks.
"""

import csv
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import nibabel as nib
import numpy as np
import pydicom

from dcm2niix_converter import Dcm2niixConverter


class SegmentationDicomConverter:
    """
    Professional DICOM to NIfTI converter for segmentation datasets.
    
    Converts segmentation DICOM files from sinus_test structure to unified sinus dataset
    with validated RGB-to-binary conversion logic (R==G & R!=B) that captures
    all 3,710+ yellow annotation color variations.
    """

    def __init__(self,
                 patient_ids_file: Optional[str] = None,
                 debug: bool = False) -> None:
        """
        Initialize the segmentation converter.
        
        Args:
            patient_ids_file: Path to patient IDs file (CSV or TXT). 
                            Defaults to /opt/datasets/sinus/patient_ids_segmentation.csv
            debug: Enable debug logging for detailed GT DICOM processing logs
        """
        # Set default CSV file if not specified
        if patient_ids_file is None:
            patient_ids_file = "/opt/datasets/sinus/patient_ids_segmentation.csv"

        self.patient_ids_file = Path(patient_ids_file)
        self.input_base = Path("/opt/datasets/sinus_test/segmentation")
        self.output_base = Path("/opt/datasets/sinus/segmentation")

        self._setup_logging(debug)
        self._validate_paths()

        # Initialize dcm2niix converter utility
        self.dcm2niix_converter = Dcm2niixConverter(enable_json=False)

        self.logger.info("Segmentation DICOM to NIfTI Converter initialized")
        self.logger.info(f"Patient IDs file: {self.patient_ids_file}")
        self.logger.info(f"Input base: {self.input_base}")
        self.logger.info(f"Output base: {self.output_base}")

    def _setup_logging(self, debug: bool = False) -> None:
        """Setup logging with Google Code Style compliant format."""
        self.logger = logging.getLogger('SegmentationConverter')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)

        # Custom formatter with timestamp
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _validate_paths(self) -> None:
        """Validate that all required paths exist."""
        if not self.patient_ids_file.exists():
            raise FileNotFoundError(
                f"Patient IDs file not found: {self.patient_ids_file}")

        original_dir = self.input_base / "original"
        label_dir = self.input_base / "label"

        if not original_dir.exists():
            raise FileNotFoundError(
                f"Original directory not found: {original_dir}")

        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

    def _load_patient_ids(self) -> List[str]:
        """Load patient IDs from CSV or TXT file."""
        try:
            patient_ids = []

            # Check if file is CSV based on extension
            if self.patient_ids_file.suffix.lower() == '.csv':
                # Parse CSV file and extract patient_id column
                with open(self.patient_ids_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'patient_id' in row and row['patient_id'].strip():
                            patient_ids.append(row['patient_id'].strip())

                self.logger.info(
                    f"Loaded {len(patient_ids)} patient IDs from CSV file")
            else:
                # Parse TXT file (one patient ID per line)
                with open(self.patient_ids_file, 'r') as f:
                    patient_ids = [line.strip() for line in f if line.strip()]

                self.logger.info(
                    f"Loaded {len(patient_ids)} patient IDs from TXT file")

            if not patient_ids:
                raise ValueError("No patient IDs found in the file")

            return patient_ids

        except Exception as e:
            self.logger.error(f"Failed to load patient IDs: {e}")
            raise

    def _create_output_directory(self, patient_id: str) -> Path:
        """Create output directory for a specific patient with dcm subfolder structure."""
        patient_output_dir = self.output_base / patient_id
        dcm_dir = patient_output_dir / "dcm"
        dcm_original_dir = dcm_dir / "original"
        dcm_gt_dir = dcm_dir / "gt"

        # Create all directories
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        dcm_dir.mkdir(parents=True, exist_ok=True)
        dcm_original_dir.mkdir(parents=True, exist_ok=True)
        dcm_gt_dir.mkdir(parents=True, exist_ok=True)

        return patient_output_dir

    def _copy_dicom_files(self, patient_id: str,
                          output_dir: Path) -> Tuple[bool, Optional[str]]:
        """
        Copy DICOM files according to the new workflow:
        1. Copy original *.dcm files to dcm/original/
        2. Copy original *.dcm files to dcm/gt/ (for GT processing)
        3. Copy label *.dcm files to temp_folder/ (for RGB processing)
        
        Args:
            patient_id: Patient identifier
            output_dir: Output directory for the patient
            
        Returns:
            Tuple of (success, temp_folder_path)
        """
        try:
            # Source directories
            original_source = self.input_base / "original" / patient_id
            label_source = self.input_base / "label" / patient_id

            # Destination directories
            dcm_original_dest = output_dir / "dcm" / "original"
            dcm_gt_dest = output_dir / "dcm" / "gt"

            # Create temporary folder for label processing
            temp_folder = tempfile.mkdtemp(prefix=f"temp_labels_{patient_id}_")
            temp_folder_path = Path(temp_folder)

            # Copy original DICOM files to original folder
            if original_source.exists():
                for dicom_file in original_source.glob("*.dcm"):
                    shutil.copy2(dicom_file, dcm_original_dest)
                for dicom_file in original_source.glob("*.DCM"):
                    shutil.copy2(dicom_file, dcm_original_dest)
                original_count = len(list(
                    dcm_original_dest.glob("*.dcm"))) + len(
                        list(dcm_original_dest.glob("*.DCM")))
                self.logger.info(
                    f"Copied {original_count} original DICOM files to original/"
                )
            else:
                self.logger.error(
                    f"Original source directory not found: {original_source}")
                return False, None

            # Copy original DICOM files to GT folder (will be modified later)
            if original_source.exists():
                for dicom_file in original_source.glob("*.dcm"):
                    shutil.copy2(dicom_file, dcm_gt_dest)
                for dicom_file in original_source.glob("*.DCM"):
                    shutil.copy2(dicom_file, dcm_gt_dest)
                gt_count = len(list(dcm_gt_dest.glob("*.dcm"))) + len(
                    list(dcm_gt_dest.glob("*.DCM")))
                self.logger.info(
                    f"Copied {gt_count} original DICOM files to gt/ (for processing)"
                )

            # Copy label DICOM files to temporary folder
            if label_source.exists():
                for dicom_file in label_source.glob("*.dcm"):
                    shutil.copy2(dicom_file, temp_folder_path)
                for dicom_file in label_source.glob("*.DCM"):
                    shutil.copy2(dicom_file, temp_folder_path)
                label_count = len(list(temp_folder_path.glob("*.dcm"))) + len(
                    list(temp_folder_path.glob("*.DCM")))
                self.logger.info(
                    f"Copied {label_count} label DICOM files to temp folder")
            else:
                self.logger.error(
                    f"Label source directory not found: {label_source}")
                return False, None

            return True, temp_folder

        except Exception as e:
            self.logger.error(f"Failed to copy DICOM files: {e}")
            return False, None

    def _stage_axial_slices(self, source_dir: Path,
                            staging_root: Path) -> Path:
        """
        Stage only the slices sharing the series' dominant orientation.

        Scanners file the scout / localizer image under the same
        SeriesInstanceUID as the axial stack. Mixed orientations leave dcm2niix
        without a consistent geometry, and it then writes a volume with a
        degenerate affine rather than failing.

        Args:
            source_dir: Directory holding the DICOM files to filter
            staging_root: Directory to stage the filtered series in

        Returns:
            Path to the staged directory, or the untouched source directory if
            no orientation could be read
        """
        by_orientation: dict = {}
        dicom_files = sorted(source_dir.glob("*.dcm")) + sorted(
            source_dir.glob("*.DCM"))

        for dicom_file in dicom_files:
            try:
                header = pydicom.dcmread(dicom_file, stop_before_pixels=True)
                orientation = tuple(
                    round(float(v)) for v in header.ImageOrientationPatient)
            except Exception:
                continue
            by_orientation.setdefault(orientation, []).append(dicom_file)

        if not by_orientation:
            self.logger.warning(
                "No slice orientation could be read; converting unfiltered")
            return source_dir

        dominant = max(by_orientation, key=lambda k: len(by_orientation[k]))
        dropped = sum(len(files) for orientation, files in by_orientation.items()
                      if orientation != dominant)

        staged = staging_root / source_dir.name
        staged.mkdir(parents=True, exist_ok=True)
        for dicom_file in by_orientation[dominant]:
            shutil.copy2(dicom_file, staged)

        if dropped:
            self.logger.warning(
                f"Dropped {dropped} slice(s) from {source_dir.name} not "
                f"matching the dominant orientation {dominant}; these are "
                f"scout or localizer images and would corrupt the "
                f"reconstructed geometry")

        return staged

    def _convert_original_dicom(self, patient_id: str,
                                output_dir: Path) -> bool:
        """
        Convert original DICOM files using dcm2niix on the dcm/original/ folder.
        
        Args:
            patient_id: Patient identifier
            output_dir: Output directory for converted files
            
        Returns:
            True if conversion successful, False otherwise
        """
        original_dicom_dir = output_dir / "dcm" / "original"

        with tempfile.TemporaryDirectory() as staging_root:
            staged_dir = self._stage_axial_slices(
                original_dicom_dir, Path(staging_root))

            # Use dcm2niix converter utility with int16 data type
            conversion_result = self.dcm2niix_converter.convert_dicom_directory(
                input_dir=staged_dir,
                output_dir=output_dir,
                output_filename=patient_id,
                data_type="int16")

        if conversion_result.success:
            self.logger.info("Original DICOM conversion successful")
            return True
        else:
            self.logger.error(
                f"Original DICOM conversion failed: {conversion_result.error_message}"
            )
            return False

    def _apply_rgb_to_binary_logic(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Apply validated R==G & R!=B logic with contour filling.
        
        This logic captures all 3,710+ yellow annotation color variations as unified ROI,
        including anti-aliased edges and gradient boundaries from professional medical
        annotation software.
        
        Args:
            pixel_array: RGB pixel array from DICOM
            
        Returns:
            Binary mask with 0/1 values, with filled contours
        """
        if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
            # Create binary mask using validated color detection logic
            mask = np.zeros_like(pixel_array[..., 0], dtype=np.float32)

            # R==G & R!=B captures all yellow family annotation colors
            condition = ((pixel_array[..., 0] == pixel_array[..., 1]) &
                         (pixel_array[..., 0] != pixel_array[..., 2]))

            mask[condition] = 1

            # Apply contour filling to match PyTorch dataset processing
            if np.any(mask):
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mask, contours, -1, (1), thickness=cv2.FILLED)

            return mask.astype(np.uint8)
        else:
            # Already grayscale, convert to binary
            return (pixel_array > 0).astype(np.uint8)

    def _process_rgb_annotations_inplace(self, temp_folder: str,
                                         output_dir: Path) -> bool:
        """
        Process RGB annotations in-place according to the new workflow:
        1. Read RGB data from temp_folder
        2. Apply R==G & R!=B logic + contour filling
        3. Update corresponding DICOM files in dcm/gt/ with binary pixel data
        
        Args:
            temp_folder: Temporary folder containing label DICOM files
            output_dir: Output directory containing dcm/gt/ folder
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            temp_folder_path = Path(temp_folder)
            gt_dicom_dir = output_dir / "dcm" / "gt"

            # Get all label DICOM files from temp folder
            label_files = list(temp_folder_path.glob("*.dcm")) + list(
                temp_folder_path.glob("*.DCM"))

            if not label_files:
                self.logger.error("No label DICOM files found in temp folder")
                return False

            self.logger.info(
                f"Processing {len(label_files)} RGB annotation files")
            processed_count = 0

            for label_file in label_files:
                try:
                    # Read RGB data from temp label file
                    label_ds = pydicom.dcmread(label_file)
                    rgb_pixel_array = label_ds.pixel_array

                    # Apply RGB-to-binary conversion logic
                    binary_mask = self._apply_rgb_to_binary_logic(
                        rgb_pixel_array)

                    if np.any(binary_mask):
                        # Convert binary mask to 0/255 values
                        binary_mask_255 = (binary_mask * 255).astype(np.uint8)
                    else:
                        # No annotations, create zero mask
                        binary_mask_255 = np.zeros_like(binary_mask,
                                                        dtype=np.uint8)

                    # Find corresponding GT DICOM file (handle case-insensitive matching)
                    # Extract base filename without extension
                    base_name = label_file.stem

                    # Try both .dcm and .DCM extensions for GT file
                    gt_file = None
                    for ext in ['.dcm', '.DCM']:
                        potential_gt_file = gt_dicom_dir / f"{base_name}{ext}"
                        if potential_gt_file.exists():
                            gt_file = potential_gt_file
                            break

                    if gt_file and gt_file.exists():
                        self.logger.info(f"Processing GT DICOM: {gt_file.name}")

                        # Load GT DICOM file and update pixel data
                        gt_ds = pydicom.dcmread(gt_file)
                        self.logger.debug(
                            f"Original GT DICOM metadata - PhotometricInterpretation: {getattr(gt_ds, 'PhotometricInterpretation', 'N/A')}"
                        )

                        # Log binary mask statistics
                        annotation_pixels = np.sum(binary_mask_255 > 0)
                        total_pixels = binary_mask_255.size
                        annotation_percentage = (annotation_pixels /
                                                 total_pixels) * 100
                        self.logger.info(
                            f"Binary mask stats - Annotation pixels: {annotation_pixels:,}/{total_pixels:,} ({annotation_percentage:.2f}%)"
                        )

                        # Update pixel data and metadata
                        self.logger.debug(
                            "Updating GT DICOM pixel data and metadata...")
                        gt_ds.PixelData = binary_mask_255.tobytes()
                        gt_ds.PhotometricInterpretation = 'MONOCHROME2'
                        gt_ds.SamplesPerPixel = 1
                        gt_ds.BitsAllocated = 8
                        gt_ds.BitsStored = 8
                        gt_ds.HighBit = 7
                        gt_ds.PixelRepresentation = 0
                        gt_ds.Rows, gt_ds.Columns = binary_mask_255.shape
                        self.logger.debug(
                            f"Updated dimensions: {binary_mask_255.shape}")

                        # Remove CT-specific scaling attributes to prevent Hounsfield unit conversion
                        self.logger.debug(
                            "Removing CT-specific scaling attributes...")
                        ct_scaling_attrs = [
                            'RescaleSlope', 'RescaleIntercept', 'RescaleType',
                            'WindowCenter', 'WindowWidth',
                            'WindowCenterWidthExplanation',
                            'PlanarConfiguration'
                        ]
                        removed_attrs = []
                        for attr in ct_scaling_attrs:
                            if hasattr(gt_ds, attr):
                                delattr(gt_ds, attr)
                                removed_attrs.append(attr)

                        if removed_attrs:
                            self.logger.debug(
                                f"Removed CT scaling attributes: {', '.join(removed_attrs)}"
                            )

                        # Set proper scaling for binary data (no scaling)
                        gt_ds.RescaleSlope = 1.0
                        gt_ds.RescaleIntercept = 0.0
                        self.logger.debug(
                            "Set binary scaling: RescaleSlope=1.0, RescaleIntercept=0.0"
                        )

                        # Save modified GT DICOM
                        self.logger.debug(
                            f"Saving modified GT DICOM to: {gt_file}")
                        gt_ds.save_as(gt_file)
                        self.logger.info(
                            f"✓ Successfully updated GT DICOM: {gt_file.name}")
                        processed_count += 1
                    else:
                        self.logger.warning(
                            f"Corresponding GT file not found for {label_file.name}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error processing {label_file.name}: {e}")
                    continue

            self.logger.info(
                f"Successfully processed {processed_count} annotation files")

            # Clean up temp folder
            try:
                shutil.rmtree(temp_folder)
                self.logger.info("Cleaned up temporary folder")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp folder: {e}")

            return processed_count > 0

        except Exception as e:
            self.logger.error(f"Failed to process RGB annotations: {e}")
            return False

    def _verify_conversion(self, patient_id: str,
                           output_dir: Path) -> Tuple[bool, Tuple, Tuple]:
        """
        Verify that conversion was successful and dimensions match.
        
        Args:
            patient_id: Patient identifier
            output_dir: Output directory containing converted files
            
        Returns:
            Tuple of (success, original_shape, gt_shape)
        """
        original_file = output_dir / f"{patient_id}.nii.gz"
        gt_file = output_dir / f"{patient_id}_gt.nii.gz"

        try:
            if not original_file.exists() or not gt_file.exists():
                return False, None, None

            # Load both files
            original_img = nib.load(original_file)
            gt_img = nib.load(gt_file)

            original_shape = original_img.shape
            gt_shape = gt_img.shape

            # A zero voxel spacing means the series geometry could not be
            # reconstructed. Downstream stages divide by this, so reject it
            # here rather than emitting a volume that fails silently later.
            voxel_spacing = np.abs(np.diag(original_img.affine)[:3])
            if not np.all(voxel_spacing > 0):
                self.logger.error(
                    f"Degenerate affine (voxel spacing {voxel_spacing}); "
                    f"the series geometry could not be reconstructed")
                return False, None, None

            self.logger.info(f"Original: {original_shape}")
            self.logger.info(f"GT: {gt_shape}")

            if original_shape == gt_shape:
                self.logger.info("✓ Slice counts match")
                return True, original_shape, gt_shape
            else:
                self.logger.error("✗ Slice counts do not match")
                return False, original_shape, gt_shape

        except Exception as e:
            self.logger.error(f"Error during verification: {e}")
            return False, None, None

    def _convert_gt_dicom(self, patient_id: str, output_dir: Path) -> bool:
        """
        Convert GT DICOM files using dcm2niix with uint8 output.
        
        Args:
            patient_id: Patient identifier
            output_dir: Output directory for converted files
            
        Returns:
            True if conversion successful, False otherwise
        """
        gt_dicom_dir = output_dir / "dcm" / "gt"

        # The annotation is filtered the same way as the volume it labels, so
        # the two keep the same slice count for _verify_conversion to compare.
        with tempfile.TemporaryDirectory() as staging_root:
            staged_dir = self._stage_axial_slices(
                gt_dicom_dir, Path(staging_root))

            conversion_result = self.dcm2niix_converter.convert_dicom_directory(
                input_dir=staged_dir,
                output_dir=output_dir,
                output_filename=f"{patient_id}_gt",
                data_type="uint8")

        if conversion_result.success:
            self.logger.info("GT DICOM conversion successful")
            return True
        else:
            self.logger.error(
                f"GT DICOM conversion failed: {conversion_result.error_message}"
            )
            return False

    def convert_patient(self, patient_id: str) -> bool:
        """
        Convert a single patient's DICOM data to NIfTI format using the new workflow.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            True if conversion successful, False otherwise
        """
        self.logger.info(f"--- Processing Patient: {patient_id} ---")

        try:
            # Step 1: Create output directory
            output_dir = self._create_output_directory(patient_id)

            # Step 2: Copy DICOM files according to new workflow
            self.logger.info("Copying DICOM files according to new workflow...")
            copy_success, temp_folder = self._copy_dicom_files(
                patient_id, output_dir)
            if not copy_success:
                self.logger.error("DICOM file copying failed")
                return False

            # Step 3: Process RGB annotations in-place
            self.logger.info("Processing RGB annotations in-place...")
            if not self._process_rgb_annotations_inplace(
                    temp_folder, output_dir):
                self.logger.error("RGB annotation processing failed")
                return False

            # Step 4: Convert original DICOM files
            self.logger.info("Converting original DICOM files...")
            if not self._convert_original_dicom(patient_id, output_dir):
                self.logger.error("Original DICOM conversion failed")
                return False

            # Step 5: Convert GT DICOM files
            self.logger.info("Converting GT DICOM files...")
            if not self._convert_gt_dicom(patient_id, output_dir):
                self.logger.error("GT DICOM conversion failed")
                return False

            # Step 6: Verify conversion
            self.logger.info("Verifying dimensions...")
            success, original_shape, gt_shape = self._verify_conversion(
                patient_id, output_dir)

            if success:
                self.logger.info(
                    f"✓ Patient {patient_id} converted successfully")
                return True
            else:
                self.logger.error("Dimension verification failed")
                return False

        except Exception as e:
            self.logger.error(
                f"Unexpected error processing patient {patient_id}: {e}")
            return False

    def convert_all_patients(self) -> List[bool]:
        """
        Convert all patients listed in the patient IDs file.
        
        Returns:
            List of boolean results for each patient conversion
        """
        self.logger.info("Starting segmentation DICOM to NIfTI conversion")

        # Load patient IDs
        patient_ids = self._load_patient_ids()
        self.logger.info(f"Found {len(patient_ids)} patients to process")

        # Convert each patient
        results = []
        for i, patient_id in enumerate(patient_ids, 1):
            self.logger.info(
                f"--- Processing Patient {i}/{len(patient_ids)}: {patient_id} ---"
            )
            result = self.convert_patient(patient_id)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r)
        failed = len(results) - successful

        self.logger.info("=== Conversion Summary ===")
        self.logger.info(
            f"Total: {len(results)}, Success: {successful}, Failed: {failed}")

        if failed == 0:
            self.logger.info("🎉 All conversions completed successfully!")
        else:
            self.logger.warning(f"{failed} conversions failed")
            # Failed patient details are already logged in convert_patient method

        return results


def main() -> bool:
    """Main function to run the segmentation conversion pipeline."""
    try:
        # Initialize converter with default CSV file
        # Uses /opt/datasets/sinus/patient_ids_segmentation.csv by default
        converter = SegmentationDicomConverter(debug=True)

        # Run conversion
        conversion_results = converter.convert_all_patients()

        # Final summary
        successful = sum(1 for r in conversion_results if r)
        total = len(conversion_results)

        logger = logging.getLogger('Main')
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Conversion success rate: {successful}/{total}")

        if successful == total:
            logger.info("🎉 All segmentation patients successfully converted!")
            logger.info("Dataset ready for machine learning training!")
        else:
            logger.warning(
                "Some conversions failed - please review errors above")

        return successful == total

    except Exception as e:
        logger = logging.getLogger('Main')
        logger.error(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
