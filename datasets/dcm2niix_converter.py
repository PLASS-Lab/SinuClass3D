#!/usr/bin/env python3
"""
dcm2niix Conversion Utility

Typed wrapper around the dcm2niix command-line tool. The dataset converters in
this package depend on it for the DICOM-to-NIfTI step and rely on the result
landing at a predictable path with a predictable data type: CT volumes as
int16, binary annotation masks as uint8.

dcm2niix decides its own output filenames and may emit companion files, so
conversion runs in a temporary directory and only the selected volume is moved
into place under the requested name.
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np

# Data types the dataset converters ask for, mapped to their numpy equivalents.
SUPPORTED_DATA_TYPES = {
    "int16": np.int16,
    "uint8": np.uint8,
    "float32": np.float32,
}


@dataclass
class ConversionResult:
    """Data class to store the outcome of a single dcm2niix invocation."""
    success: bool
    output_path: Optional[Path] = None
    slice_count: int = 0
    error_message: Optional[str] = None


class Dcm2niixConverter:
    """
    Professional dcm2niix wrapper.

    Converts a directory of DICOM files into a single compressed NIfTI volume
    at a caller-chosen path, casting to the requested data type. Follows SOLID
    principles with single responsibility for the conversion step itself; slice
    selection and dataset layout are the caller's concern.
    """

    EXECUTABLE = "dcm2niix"
    SEARCH_DEPTH = "0"  # do not recurse; the caller stages exactly what it wants

    def __init__(self, enable_json: bool = True,
                 compress: bool = True) -> None:
        """
        Initialize the converter.

        Args:
            enable_json: Whether to let dcm2niix write its BIDS sidecar JSON
            compress: Whether to write gzip-compressed .nii.gz output
        """
        self.enable_json = enable_json
        self.compress = compress
        self.logger = logging.getLogger(__name__)

        self.executable_path = shutil.which(self.EXECUTABLE)
        if self.executable_path is None:
            self.logger.warning(
                f"{self.EXECUTABLE} was not found on PATH; conversion will "
                f"fail until it is installed")

    @property
    def suffix(self) -> str:
        """File extension produced for the configured compression setting."""
        return ".nii.gz" if self.compress else ".nii"

    def _build_command(self, input_dir: Path, staging_dir: Path,
                       output_filename: str) -> List[str]:
        """
        Assemble the dcm2niix command line.

        Args:
            input_dir: Directory holding the DICOM files to convert
            staging_dir: Temporary directory to receive dcm2niix output
            output_filename: Base name, without extension, for the volume

        Returns:
            Command as a list of arguments suitable for subprocess.run
        """
        return [
            self.executable_path,
            "-z", "y" if self.compress else "n",
            "-b", "y" if self.enable_json else "n",
            "-d", self.SEARCH_DEPTH,
            "-f", output_filename,
            "-o", str(staging_dir),
            str(input_dir),
        ]

    def _select_volume(self, staging_dir: Path) -> Optional[Path]:
        """
        Pick the converted volume from whatever dcm2niix produced.

        A DICOM directory holding more than one distinct geometry makes
        dcm2niix split its output. The largest volume is the intended one; the
        split itself is reported so it is never silent.

        Args:
            staging_dir: Directory dcm2niix wrote into

        Returns:
            Path to the selected volume, or None if nothing was produced
        """
        volumes = sorted(staging_dir.glob(f"*{self.suffix}"))
        if not volumes:
            return None

        if len(volumes) > 1:
            self.logger.warning(
                f"dcm2niix produced {len(volumes)} volumes from one directory, "
                f"which means the input held more than one geometry; keeping "
                f"the largest")
            volumes.sort(key=lambda p: p.stat().st_size, reverse=True)

        return volumes[0]

    def _cast_data_type(self, volume_path: Path, data_type: str) -> int:
        """
        Rewrite a volume in the requested data type, preserving its geometry.

        Args:
            volume_path: Volume to rewrite in place
            data_type: Key into SUPPORTED_DATA_TYPES

        Returns:
            Number of slices along the third axis
        """
        image = nib.load(volume_path)
        array = np.asanyarray(image.dataobj).astype(SUPPORTED_DATA_TYPES[data_type])

        recast = nib.Nifti1Image(array, image.affine, image.header)
        recast.set_data_dtype(SUPPORTED_DATA_TYPES[data_type])
        # The header carries the pre-cast scaling; clear it so the stored
        # integers are read back at face value.
        recast.header.set_slope_inter(None, None)
        nib.save(recast, volume_path)

        return array.shape[2] if array.ndim >= 3 else 1

    def convert_dicom_directory(self, input_dir: Path, output_dir: Path,
                                output_filename: str,
                                data_type: str = "int16") -> ConversionResult:
        """
        Convert one directory of DICOM files to a single NIfTI volume.

        Args:
            input_dir: Directory holding the DICOM files to convert
            output_dir: Directory to write the finished volume into
            output_filename: Base name, without extension, for the volume
            data_type: One of SUPPORTED_DATA_TYPES

        Returns:
            ConversionResult describing the outcome
        """
        input_dir, output_dir = Path(input_dir), Path(output_dir)

        if self.executable_path is None:
            return ConversionResult(
                success=False,
                error_message=f"{self.EXECUTABLE} not found on PATH")

        if data_type not in SUPPORTED_DATA_TYPES:
            return ConversionResult(
                success=False,
                error_message=f"Unsupported data type '{data_type}'; expected "
                              f"one of {sorted(SUPPORTED_DATA_TYPES)}")

        if not input_dir.is_dir():
            return ConversionResult(
                success=False,
                error_message=f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{output_filename}{self.suffix}"

        try:
            with tempfile.TemporaryDirectory() as staging:
                staging_dir = Path(staging)
                command = self._build_command(
                    input_dir, staging_dir, output_filename)

                completed = subprocess.run(
                    command, capture_output=True, text=True, check=False)

                if completed.returncode != 0:
                    return ConversionResult(
                        success=False,
                        error_message=f"{self.EXECUTABLE} exited with "
                                      f"{completed.returncode}: "
                                      f"{completed.stderr.strip()[:400]}")

                volume = self._select_volume(staging_dir)
                if volume is None:
                    return ConversionResult(
                        success=False,
                        error_message=f"{self.EXECUTABLE} produced no volume "
                                      f"from {input_dir}")

                shutil.move(str(volume), str(destination))

                if self.enable_json:
                    sidecar = volume.with_suffix("").with_suffix(".json")
                    if sidecar.exists():
                        shutil.move(
                            str(sidecar),
                            str(destination.parent / f"{output_filename}.json"))

            slice_count = self._cast_data_type(destination, data_type)

            self.logger.info(
                f"Converted {input_dir} -> {destination} "
                f"({slice_count} slices, {data_type})")

            return ConversionResult(
                success=True, output_path=destination, slice_count=slice_count)

        except Exception as e:
            return ConversionResult(
                success=False,
                error_message=f"{type(e).__name__}: {e}")
