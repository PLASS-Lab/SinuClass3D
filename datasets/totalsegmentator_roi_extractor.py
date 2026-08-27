#!/usr/bin/env python3
"""
TotalSegmentator ROI Extractor for Sinus Medical Imaging

This script uses TotalSegmentator to extract specific anatomical regions of interest
from NIfTI CT volumes for medical imaging analysis.
"""

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import nibabel as nib
import numpy as np
from tqdm import tqdm


@dataclass
class ROIExtractionResult:
    """Result of ROI extraction operation."""
    success: bool
    patient_id: str
    input_file: Path
    output_directory: Path
    extracted_rois: List[str]
    failed_rois: List[str]
    error_message: Optional[str] = None


class TotalSegmentatorROIExtractor:
    """
    Extracts anatomical regions of interest using TotalSegmentator.
    
    This class handles the extraction of skull and brain structures
    from CT NIfTI volumes using the TotalSegmentator deep learning model.
    """

    # Target ROIs - skull and brain
    # TotalSegmentator will output these as separate files
    TARGET_ROIS = ["skull", "brain"]

    # TotalSegmentator output file names
    # Using CLI will generate individual structure files
    BRAIN_FILE = "brain.nii.gz"
    SKULL_FILE = "skull.nii.gz"

    def __init__(self,
                 patient_ids_file: Optional[str] = None,
                 input_base: str = "/opt/datasets/sinus/segmentation",
                 output_base: str = "/opt/datasets/sinus/segmentation",
                 python_path: str = sys.executable,
                 fast: bool = False):
        """
        Initialize TotalSegmentator ROI extractor.
        
        Args:
            patient_ids_file: Path to patient IDs file (CSV or TXT).
                            Defaults to /opt/datasets/sinus/patient_ids_segmentation.csv
            input_base: Base directory containing patient NIfTI files
            output_base: Base directory for ROI output files
            python_path: Path to Python executable with TotalSegmentator installed
            fast: Use fast mode (lower resolution, 3mm). Default False for most accurate results.
        """
        # Set default CSV file if not specified
        if patient_ids_file is None:
            patient_ids_file = "/opt/datasets/sinus/patient_ids_segmentation.csv"

        self.patient_ids_file = Path(patient_ids_file)
        self.input_base = Path(input_base)
        self.output_base = Path(output_base)
        self.python_path = python_path
        self.fast = fast
        self.logger = self._setup_logging()

        self.logger.info("TotalSegmentator ROI Extractor initialized")
        self.logger.info(f"Patient IDs file: {self.patient_ids_file}")
        self.logger.info(f"Input base: {self.input_base}")
        self.logger.info(f"Output base: {self.output_base}")
        self.logger.info(f"Fast mode: {self.fast}")

    def _setup_logging(self) -> logging.Logger:
        """
        Configure logging with timestamps in ISO format.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # ISO timestamp format as specified
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _load_patient_ids(self) -> List[str]:
        """Load patient IDs from CSV or TXT file."""
        try:
            if not self.patient_ids_file.exists():
                raise FileNotFoundError(
                    f"Patient IDs file not found: {self.patient_ids_file}")

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

    def _check_totalsegmentator_installation(self) -> bool:
        """
        Verify TotalSegmentator CLI is installed and accessible.
        
        Returns:
            True if TotalSegmentator is available, False otherwise
        """
        try:
            import subprocess
            result = subprocess.run(["TotalSegmentator", "--help"],
                                    capture_output=True,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                self.logger.info("TotalSegmentator CLI verified")
                return True
            else:
                self.logger.error("TotalSegmentator CLI not working")
                return False
        except FileNotFoundError:
            self.logger.error("TotalSegmentator CLI not found")
            return False
        except Exception as e:
            self.logger.error(f"Error checking TotalSegmentator: {e}")
            return False

    def _run_totalsegmentator(
            self, input_file: Path,
            output_dir: Path) -> Tuple[bool, str, Optional[Path]]:
        """
        Execute TotalSegmentator CLI to extract all anatomical structures.
        
        Args:
            input_file: Path to input NIfTI file
            output_dir: Directory for TotalSegmentator output (temp directory)
            
        Returns:
            Tuple of (success_status, error_message, output_directory)
        """
        import subprocess

        try:
            self.logger.info(f"Running TotalSegmentator on {input_file.name}")
            self.logger.info(f"Input: {input_file}")
            self.logger.info(f"Output directory: {output_dir}")

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run TotalSegmentator CLI with "total" task
            # This saves individual structure files to output directory
            self.logger.info(
                "Running TotalSegmentator with 'total' task (all structures)")

            # Build command
            # Note: Do NOT use --ml flag as it creates one multilabel file instead of individual structure files
            cmd = [
                "TotalSegmentator",
                "-i",
                str(input_file),
                "-o",
                str(output_dir),
                "--task",
                "total",
                "--device",
                "gpu:0",  # Use GPU #0 specifically
            ]

            # Add fast flag if requested (lower resolution, 3mm)
            if self.fast:
                cmd.append("--fast")
                self.logger.info("Using fast mode (3mm resolution)")
            else:
                self.logger.info("Using full resolution mode (most accurate)")

            # Run command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode != 0:
                error_msg = f"TotalSegmentator failed: {result.stderr}"
                self.logger.error(error_msg)
                return False, error_msg, None

            self.logger.info("TotalSegmentator completed successfully")
            self.logger.info(f"Output files saved to: {output_dir}")

            return True, "", output_dir

        except subprocess.TimeoutExpired:
            error_msg = "TotalSegmentator timed out (>5 minutes)"
            self.logger.error(error_msg)
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Error running TotalSegmentator: {e}"
            self.logger.error(error_msg)
            return False, error_msg, None

    def _extract_target_rois(self, ts_output_dir: Path, final_output_dir: Path,
                             patient_id: str) -> Tuple[List[str], List[str]]:
        """
        Copy brain and skull ROI files from TotalSegmentator CLI output directory.
        
        Args:
            ts_output_dir: TotalSegmentator output directory with individual structure files
            final_output_dir: Final output directory (same as patient directory)
            patient_id: Patient identifier for filename prefix
            
        Returns:
            Tuple of (successfully_extracted_rois, failed_rois)
        """
        import shutil

        extracted_rois = []
        failed_rois = []

        # Ensure output directory exists
        final_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # List all available files in TotalSegmentator output
            available_files = sorted(
                [f.name for f in ts_output_dir.glob("*.nii.gz")])
            self.logger.info(
                f"TotalSegmentator output contains {len(available_files)} files"
            )
            if len(available_files) <= 20:
                self.logger.info(
                    f"Available files: {', '.join(available_files)}")
            else:
                self.logger.info(
                    f"First 20 files: {', '.join(available_files[:20])}")
                self.logger.info(
                    f"... and {len(available_files) - 20} more files")

            # Copy brain.nii.gz
            self.logger.info("Copying brain.nii.gz...")
            brain_source = ts_output_dir / self.BRAIN_FILE
            brain_dest = final_output_dir / f"{patient_id}_brain.nii.gz"

            if brain_source.exists():
                shutil.copy2(brain_source, brain_dest)
                # Get voxel count for logging
                brain_img = nib.load(str(brain_dest))
                brain_voxels = np.sum(brain_img.get_fdata() > 0)
                extracted_rois.append("brain")
                self.logger.info(
                    f"Copied ROI: brain -> {brain_dest.name} ({brain_voxels:,} voxels)"
                )
            else:
                self.logger.warning(f"Brain file not found: {brain_source}")
                failed_rois.append("brain")

            # Copy skull.nii.gz
            self.logger.info("Copying skull.nii.gz...")
            skull_source = ts_output_dir / self.SKULL_FILE
            skull_dest = final_output_dir / f"{patient_id}_skull.nii.gz"

            if skull_source.exists():
                shutil.copy2(skull_source, skull_dest)
                # Get voxel count for logging
                skull_img = nib.load(str(skull_dest))
                skull_voxels = np.sum(skull_img.get_fdata() > 0)
                extracted_rois.append("skull")
                self.logger.info(
                    f"Copied ROI: skull -> {skull_dest.name} ({skull_voxels:,} voxels)"
                )
            else:
                self.logger.warning(
                    f"Skull file not found: {skull_source} - may be outside scan coverage"
                )
                failed_rois.append("skull")

        except Exception as e:
            self.logger.error(f"Error copying ROI files: {e}")
            failed_rois = self.TARGET_ROIS.copy()

        return extracted_rois, failed_rois

    def _clean_existing_roi_files(self, output_directory: Path,
                                  patient_id: str) -> None:
        """
        Delete existing brain and skull files if they exist.

        Args:
            output_directory: Directory containing ROI files
            patient_id: Patient identifier for filename prefix
        """
        brain_file = output_directory / f"{patient_id}_brain.nii.gz"
        skull_file = output_directory / f"{patient_id}_skull.nii.gz"

        deleted_files = []
        if brain_file.exists():
            brain_file.unlink()
            deleted_files.append("brain")
            self.logger.info(f"Deleted existing file: {brain_file.name}")

        if skull_file.exists():
            skull_file.unlink()
            deleted_files.append("skull")
            self.logger.info(f"Deleted existing file: {skull_file.name}")

        if deleted_files:
            self.logger.info(
                f"Cleaned up {len(deleted_files)} existing ROI file(s): {', '.join(deleted_files)}"
            )

    def extract_rois_from_nifti(self, input_file: Path,
                                output_directory: Path) -> ROIExtractionResult:
        """
        Extract anatomical ROIs from a NIfTI file using TotalSegmentator.
        
        Args:
            input_file: Path to input NIfTI file
            output_directory: Directory to save extracted ROI files
            
        Returns:
            ROIExtractionResult with extraction details
        """
        # Extract patient ID from filename
        patient_id = input_file.stem.replace('.nii', '')

        self.logger.info(f"Starting ROI extraction for patient {patient_id}")
        self.logger.info(f"Input file: {input_file}")
        self.logger.info(f"Output directory: {output_directory}")

        # Verify input file exists
        if not input_file.exists():
            error_msg = f"Input file does not exist: {input_file}"
            self.logger.error(error_msg)
            return ROIExtractionResult(success=False,
                                       patient_id=patient_id,
                                       input_file=input_file,
                                       output_directory=output_directory,
                                       extracted_rois=[],
                                       failed_rois=self.TARGET_ROIS.copy(),
                                       error_message=error_msg)

        # Check TotalSegmentator installation
        if not self._check_totalsegmentator_installation():
            error_msg = "TotalSegmentator is not properly installed"
            self.logger.error(error_msg)
            return ROIExtractionResult(success=False,
                                       patient_id=patient_id,
                                       input_file=input_file,
                                       output_directory=output_directory,
                                       extracted_rois=[],
                                       failed_rois=self.TARGET_ROIS.copy(),
                                       error_message=error_msg)

        # Clean up existing brain and skull files
        self._clean_existing_roi_files(output_directory, patient_id)

        # Create temporary directory for TotalSegmentator output
        with tempfile.TemporaryDirectory(
                prefix=f"totalseg_{patient_id}_") as temp_dir:
            temp_output_dir = Path(temp_dir)

            # Run TotalSegmentator CLI (saves individual files)
            success, error_msg, ts_output_dir = self._run_totalsegmentator(
                input_file, temp_output_dir)

            if not success or ts_output_dir is None:
                return ROIExtractionResult(success=False,
                                           patient_id=patient_id,
                                           input_file=input_file,
                                           output_directory=output_directory,
                                           extracted_rois=[],
                                           failed_rois=self.TARGET_ROIS.copy(),
                                           error_message=error_msg)

            # Copy brain and skull files from TotalSegmentator output
            extracted_rois, failed_rois = self._extract_target_rois(
                ts_output_dir, output_directory, patient_id)

        # Determine overall success
        overall_success = len(extracted_rois) > 0

        self.logger.info(f"ROI extraction completed for patient {patient_id}")
        self.logger.info(
            f"Successfully extracted: {len(extracted_rois)}/{len(self.TARGET_ROIS)} ROIs"
        )

        if failed_rois:
            self.logger.warning(
                f"Failed to extract ROIs: {', '.join(failed_rois)}")

        return ROIExtractionResult(success=overall_success,
                                   patient_id=patient_id,
                                   input_file=input_file,
                                   output_directory=output_directory,
                                   extracted_rois=extracted_rois,
                                   failed_rois=failed_rois,
                                   error_message=None if overall_success else
                                   "Some ROIs could not be extracted")

    def extract_rois_for_all_patients(self) -> List[ROIExtractionResult]:
        """
        Extract ROIs for all patients listed in the patient IDs file.
        
        Returns:
            List of ROIExtractionResult objects for each patient
        """
        # Load patient IDs
        patient_ids = self._load_patient_ids()

        self.logger.info(
            f"Starting ROI extraction for {len(patient_ids)} patients")

        results = []
        successful_extractions = 0

        # Progress bar for batch processing
        with tqdm(total=len(patient_ids),
                  desc="Extracting ROIs",
                  unit="patient") as pbar:
            for i, patient_id in enumerate(patient_ids, 1):
                pbar.set_description(f"Processing {patient_id}")

                self.logger.info(
                    f"Processing patient {i}/{len(patient_ids)}: {patient_id}")

                # Construct input file path
                input_file = self.input_base / patient_id / f"{patient_id}.nii.gz"
                output_directory = self.output_base / patient_id

                # Extract ROIs for this patient
                result = self.extract_rois_from_nifti(input_file,
                                                      output_directory)
                results.append(result)

                if result.success:
                    successful_extractions += 1
                    self.logger.info(
                        f"Patient {patient_id}: Successfully extracted {len(result.extracted_rois)} ROIs"
                    )
                else:
                    self.logger.error(
                        f"Patient {patient_id}: ROI extraction failed - {result.error_message}"
                    )

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "success": successful_extractions,
                    "failed": i - successful_extractions
                })

        # Summary
        self.logger.info(f"ROI extraction completed for all patients")
        self.logger.info(
            f"Success rate: {successful_extractions}/{len(patient_ids)} "
            f"({successful_extractions/len(patient_ids)*100:.1f}%)")

        return results


def main():
    """
    Main function to extract ROIs for all patients from CSV file.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract brain and skull ROIs using TotalSegmentator")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=
        "Use fast mode (3mm resolution). Default: full resolution (most accurate)"
    )
    parser.add_argument(
        "--patient-ids-file",
        type=str,
        default=None,
        help=
        "Path to patient IDs CSV file. Default: /opt/datasets/sinus/patient_ids_segmentation.csv"
    )
    parser.add_argument("--input-base",
                        type=str,
                        default="/opt/datasets/sinus/segmentation",
                        help="Base directory containing patient NIfTI files")
    parser.add_argument("--output-base",
                        type=str,
                        default="/opt/datasets/sinus/segmentation",
                        help="Base directory for ROI output files")

    args = parser.parse_args()

    # Initialize extractor with arguments
    extractor = TotalSegmentatorROIExtractor(
        patient_ids_file=args.patient_ids_file,
        input_base=args.input_base,
        output_base=args.output_base,
        fast=args.fast)

    # Extract ROIs for all patients
    results = extractor.extract_rois_for_all_patients()

    # Print summary results
    print(f"\n{'='*80}")
    print(f"TOTALSEGMENTATOR ROI EXTRACTION SUMMARY")
    print(f"{'='*80}")

    successful_patients = [r for r in results if r.success]
    failed_patients = [r for r in results if not r.success]

    print(f"Total patients processed: {len(results)}")
    print(f"Successful extractions: {len(successful_patients)}")
    print(f"Failed extractions: {len(failed_patients)}")
    print(f"Success rate: {len(successful_patients)/len(results)*100:.1f}%")

    if successful_patients:
        print(f"\nSuccessful patients:")
        for result in successful_patients:
            print(
                f"  - {result.patient_id}: {len(result.extracted_rois)} ROIs extracted"
            )

    if failed_patients:
        print(f"\nFailed patients:")
        for result in failed_patients:
            print(f"  - {result.patient_id}: {result.error_message}")

    print(f"{'='*80}")

    # Exit with appropriate code
    sys.exit(0 if len(failed_patients) == 0 else 1)


if __name__ == "__main__":
    main()
