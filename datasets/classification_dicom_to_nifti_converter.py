#!/usr/bin/env python3
"""
Professional Classification DICOM to NIfTI Converter

A specialized medical imaging converter that transforms classification DICOM datasets
from /opt/datasets/sinus_test/classification/{patient_id} into properly formatted 3D NIfTI 
volumes at /opt/datasets/sinus/classification with organized DICOM file structure.
Only processes original CT volumes - no label/segmentation processing.
"""

import logging
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from dcm2niix_converter import Dcm2niixConverter


@dataclass
class ClassificationResult:
    """Data class to store classification conversion results for a single patient."""
    patient_id: str
    success: bool
    volume_shape: Optional[tuple] = None
    slice_count: int = 0
    error_message: Optional[str] = None


class ClassificationDicomConverter:
    """
    Professional classification DICOM to NIfTI converter.
    
    Converts classification DICOM datasets with only original CT volumes,
    organizing files with proper directory structure and comprehensive logging.
    Follows SOLID principles with single responsibility for classification conversion.
    """

    def __init__(self, patient_ids_file: str) -> None:
        """
        Initialize the classification converter.
        
        Args:
            patient_ids_file: Path to patient_ids_classification.csv file
        """
        self.patient_ids_file = Path(patient_ids_file)
        self.input_base = Path("/opt/datasets/sinus_test/classification")
        self.output_base = Path("/opt/datasets/sinus/classification")

        self._setup_logging()
        self._validate_paths()
        
        # Initialize dcm2niix converter utility
        self.dcm2niix_converter = Dcm2niixConverter(enable_json=False)

        self.logger.info("Classification DICOM to NIfTI Converter initialized")
        self.logger.info(f"Input base: {self.input_base}")
        self.logger.info(f"Output base: {self.output_base}")

    def _setup_logging(self) -> None:
        """Setup logging with Google Code Style compliant format."""
        self.logger = logging.getLogger('ClassificationConverter')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler with timestamp format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

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

        if not self.input_base.exists():
            raise FileNotFoundError(
                f"Classification directory not found: {self.input_base}")

    def _load_patient_ids(self) -> List[str]:
        """Load patient IDs from the classification CSV file."""
        try:
            # Read CSV file and extract patient IDs from the first column
            df = pd.read_csv(self.patient_ids_file)
            patient_ids = df.iloc[:, 0].astype(str).tolist()
            
            # Remove any empty or NaN values
            patient_ids = [pid.strip() for pid in patient_ids if pd.notna(pid) and pid.strip()]

            self.logger.info(
                f"Loaded {len(patient_ids)} classification patient IDs from CSV")
            return patient_ids

        except Exception as e:
            self.logger.error(f"Failed to load patient IDs from CSV: {e}")
            raise

    def _create_output_directory(self, patient_id: str) -> Path:
        """Create output directory for a specific patient with dcm subfolder structure."""
        patient_output_dir = self.output_base / patient_id
        dcm_dir = patient_output_dir / "dcm"
        dcm_original_dir = dcm_dir / "original"

        # Create all directories
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        dcm_dir.mkdir(parents=True, exist_ok=True)
        dcm_original_dir.mkdir(parents=True, exist_ok=True)

        return patient_output_dir


    def _copy_dicom_files(self, patient_id: str, output_dir: Path) -> bool:
        """
        Copy original DICOM files to organized dcm folder structure.
        
        Args:
            patient_id: Patient identifier
            output_dir: Output directory for the patient
            
        Returns:
            True if copying successful, False otherwise
        """
        try:
            # Source directory (classification has direct patient folders)
            original_source = self.input_base / patient_id
            
            # Destination directory
            dcm_original_dest = output_dir / "dcm" / "original"

            # Copy original DICOM files
            if original_source.exists():
                for dicom_file in original_source.glob("*.dcm"):
                    shutil.copy2(dicom_file, dcm_original_dest)
                
                original_count = len(list(dcm_original_dest.glob("*.dcm")))
                self.logger.info(
                    f"Copied {original_count} original DICOM files")
            else:
                self.logger.error(
                    f"Original source directory not found: {original_source}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to copy DICOM files: {e}")
            return False

    def _stage_axial_slices(self, patient_id: str,
                            staging_root: Path) -> Path:
        """
        Stage only the slices sharing the series' dominant orientation.

        Scanners file the scout / localizer image under the same
        SeriesInstanceUID as the axial stack. Mixed orientations leave dcm2niix
        without a consistent geometry, and it then writes a volume with a
        degenerate affine rather than failing, so the slices are removed here.

        Args:
            patient_id: Patient identifier
            staging_root: Directory to stage the filtered series in

        Returns:
            Path to the staged directory, or the untouched source directory if
            no orientation could be read
        """
        source = self.input_base / patient_id
        by_orientation: dict = {}

        for dicom_file in sorted(source.glob("*.dcm")):
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
            return source

        dominant = max(by_orientation, key=lambda k: len(by_orientation[k]))
        dropped = sum(len(files) for orientation, files in by_orientation.items()
                      if orientation != dominant)

        staged = staging_root / patient_id
        staged.mkdir(parents=True, exist_ok=True)
        for dicom_file in by_orientation[dominant]:
            shutil.copy2(dicom_file, staged)

        if dropped:
            self.logger.warning(
                f"Dropped {dropped} slice(s) not matching the dominant "
                f"orientation {dominant}; these are scout or localizer images "
                f"and would corrupt the reconstructed geometry")

        return staged

    def _convert_original_dicom(self, patient_id: str,
                                output_dir: Path) -> bool:
        """
        Convert original DICOM files using the dcm2niix converter utility.
        
        Args:
            patient_id: Patient identifier
            output_dir: Output directory for converted files
            
        Returns:
            True if conversion successful, False otherwise
        """
        with tempfile.TemporaryDirectory() as staging_root:
            original_dicom_dir = self._stage_axial_slices(
                patient_id, Path(staging_root))

            # Use the extracted dcm2niix converter utility with int16 data type
            conversion_result = self.dcm2niix_converter.convert_dicom_directory(
                input_dir=original_dicom_dir,
                output_dir=output_dir,
                output_filename=patient_id,
                data_type="int16"
            )

        if conversion_result.success:
            self.logger.info("Original conversion successful")
            return True
        else:
            self.logger.error(f"Original conversion failed: {conversion_result.error_message}")
            return False

    def _verify_conversion(self, patient_id: str,
                           output_dir: Path) -> tuple:
        """
        Verify that conversion was successful and get volume information.
        
        Args:
            patient_id: Patient identifier
            output_dir: Output directory containing converted files
            
        Returns:
            Tuple of (success, volume_shape, slice_count)
        """
        original_file = output_dir / f"{patient_id}.nii.gz"

        try:
            if not original_file.exists():
                return False, None, 0

            # Load and verify the file
            import nibabel as nib
            original_img = nib.load(original_file)
            original_shape = original_img.shape
            slice_count = original_shape[2] if len(original_shape) >= 3 else 1

            # A zero voxel spacing means the series geometry could not be
            # reconstructed. Downstream stages divide by this, so reject it
            # here rather than emitting a volume that fails silently later.
            voxel_spacing = np.abs(np.diag(original_img.affine)[:3])
            if not np.all(voxel_spacing > 0):
                self.logger.error(
                    f"Degenerate affine (voxel spacing {voxel_spacing}); "
                    f"the series geometry could not be reconstructed")
                return False, None, 0

            self.logger.info(f"Volume shape: {original_shape}")
            self.logger.info(f"Slice count: {slice_count}")
            self.logger.info("✓ Classification volume verified")
            
            return True, original_shape, slice_count

        except Exception as e:
            self.logger.error(f"Error during verification: {e}")
            return False, None, 0

    def convert_patient(self, patient_id: str) -> ClassificationResult:
        """
        Convert a single patient's classification DICOM data to NIfTI format.
        
        Args:
            patient_id: Patient identifier to convert
            
        Returns:
            ClassificationResult object with conversion details
        """
        self.logger.info(f"--- Processing Patient: {patient_id} ---")

        try:
            # Create output directory
            output_dir = self._create_output_directory(patient_id)

            # Copy DICOM files to organized structure
            self.logger.info("Copying DICOM files to organized structure...")
            if not self._copy_dicom_files(patient_id, output_dir):
                return ClassificationResult(
                    patient_id=patient_id,
                    success=False,
                    error_message="DICOM file copying failed")

            # Convert original DICOM files
            self.logger.info("Converting original DICOM files...")
            if not self._convert_original_dicom(patient_id, output_dir):
                return ClassificationResult(
                    patient_id=patient_id,
                    success=False,
                    error_message="Original DICOM conversion failed")

            # Verify conversion
            self.logger.info("Verifying conversion...")
            success, volume_shape, slice_count = self._verify_conversion(
                patient_id, output_dir)

            if success:
                self.logger.info(
                    f"✓ Patient {patient_id} converted successfully")
                return ClassificationResult(patient_id=patient_id,
                                            success=True,
                                            volume_shape=volume_shape,
                                            slice_count=slice_count)
            else:
                return ClassificationResult(
                    patient_id=patient_id,
                    success=False,
                    volume_shape=volume_shape,
                    slice_count=slice_count,
                    error_message="Volume verification failed")

        except Exception as e:
            self.logger.error(
                f"Unexpected error processing patient {patient_id}: {e}")
            return ClassificationResult(patient_id=patient_id,
                                        success=False,
                                        error_message=str(e))

    def convert_all_patients(self) -> List[ClassificationResult]:
        """
        Convert all patients listed in the patient IDs file.
        
        Returns:
            List of ClassificationResult objects
        """
        self.logger.info("Starting classification DICOM to NIfTI conversion")

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
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        self.logger.info("=== Conversion Summary ===")
        self.logger.info(
            f"Total: {len(results)}, Success: {successful}, Failed: {failed}")

        if failed == 0:
            self.logger.info("🎉 All conversions completed successfully!")
        else:
            self.logger.error("Some conversions failed:")
            for result in results:
                if not result.success:
                    self.logger.error(
                        f"Failed: {result.patient_id} - {result.error_message}")

        return results


def main() -> bool:
    """Main function to run the classification conversion pipeline."""
    # Configuration
    patient_ids_file = "/opt/datasets/sinus/patient_ids_classification.csv"

    try:
        # Initialize converter
        converter = ClassificationDicomConverter(patient_ids_file)

        # Run conversion
        conversion_results = converter.convert_all_patients()

        # Final summary
        successful = sum(1 for r in conversion_results if r.success)
        total = len(conversion_results)

        logger = logging.getLogger('Main')
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Conversion success rate: {successful}/{total}")

        if successful == total:
            logger.info("🎉 All classification patients successfully converted!")
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
