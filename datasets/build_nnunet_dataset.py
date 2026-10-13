#!/usr/bin/env python3
"""
Build the nnU-Net v2 raw dataset for Stage 1 (sinus segmentation).

Reads the annotated set written by segmentation_dicom_to_nifti_converter.py,
<segmentation-root>/<id>/<id>.nii.gz (CT) and <id>_gt.nii.gz (manual mask),
and writes $nnUNet_raw/Dataset<ID>_<name>/ in the layout nnU-Net expects:

    imagesTr/<id>_0000.nii.gz   CT (symlink)
    labelsTr/<id>.nii.gz        manual mask remapped to {0, 1} uint8
    dataset.json                one CT channel, labels background/sinus

Then plan, train and predict with the nnU-Net v2 CLI (see the README).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Build the nnU-Net v2 dataset for sinus segmentation.")
    parser.add_argument("--segmentation-root",
                        default="/opt/datasets/sinus/segmentation",
                        help="Folder with <id>/<id>.nii.gz and <id>/<id>_gt.nii.gz")
    parser.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"),
                        help="nnU-Net raw data folder (default: $nnUNet_raw)")
    parser.add_argument("--dataset-id", type=int, default=998)
    parser.add_argument("--dataset-name", default="SinusSeg")
    args = parser.parse_args()

    if not args.nnunet_raw:
        parser.error("set --nnunet-raw or the nnUNet_raw environment variable")

    seg_root = Path(args.segmentation_root)
    raw = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr, labels_tr = raw / "imagesTr", raw / "labelsTr"
    for d in (images_tr, labels_tr):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    cases = sorted(p.name for p in seg_root.iterdir()
                   if p.is_dir() and (p / f"{p.name}.nii.gz").exists()
                   and (p / f"{p.name}_gt.nii.gz").exists())
    if not cases:
        raise SystemExit(f"no <id>/<id>.nii.gz + <id>_gt.nii.gz pairs under {seg_root}")

    for case in cases:
        ct = seg_root / case / f"{case}.nii.gz"
        gt = seg_root / case / f"{case}_gt.nii.gz"
        (images_tr / f"{case}_0000.nii.gz").symlink_to(ct.resolve())

        gt_img = nib.load(str(gt))
        if nib.load(str(ct)).shape != gt_img.shape:
            raise SystemExit(f"{case}: CT and mask shapes differ")
        mask = (np.asanyarray(gt_img.dataobj) > 0).astype(np.uint8)
        out = nib.Nifti1Image(mask, gt_img.affine, gt_img.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(labels_tr / f"{case}.nii.gz"))

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "sinus": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": raw.name,
    }
    (raw / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Wrote {raw} with {len(cases)} cases")


if __name__ == "__main__":
    main()
