<div align="center">

# Non-invasive Prediction of Eosinophilic Chronic Rhinosinusitis Endotype from Routine Sinus CT

### A Fully Automated Pipeline with Safety-Aware Deployment Analysis

**Ethan Yu**<sup>1,\*</sup> &nbsp;&nbsp; **Jin Youp Kim**<sup>2,3,\*</sup> &nbsp;&nbsp; **Doo Hee Han**<sup>4</sup> &nbsp;&nbsp; **Tae-Bin Won**<sup>4</sup> <br>
**Hyun Jik Kim**<sup>4</sup> &nbsp;&nbsp; **Dong-Young Kim**<sup>4</sup> &nbsp;&nbsp; **Chae-Seo Rhee**<sup>4,5,†</sup> &nbsp;&nbsp; **Yunsik Son**<sup>1,†</sup>

<sup>1</sup> Department of Computer Science and Artificial Intelligence, Dongguk University, Seoul, Republic of Korea <br>
<sup>2</sup> Department of Otorhinolaryngology–Head and Neck Surgery, Ilsan Hospital, Dongguk University, Goyang, Republic of Korea <br>
<sup>3</sup> Sensory Organ Research Institute, College of Medicine, Dongguk University, Gyeongju, Republic of Korea <br>
<sup>4</sup> Department of Otorhinolaryngology–Head and Neck Surgery, Seoul National University College of Medicine, Seoul National University Hospital, Seoul, Republic of Korea <br>
<sup>5</sup> Department of Otorhinolaryngology, Bundang Jesaeng Hospital, Seongnam, Republic of Korea <br>
<sup>\*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding authors

[![Paper](https://img.shields.io/badge/Paper-under%20review-b31b1b.svg)](#citation)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-ee4c2c.svg)](pyproject.toml)
[![uv](https://img.shields.io/badge/managed%20by-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

</div>

---

## TL;DR

Routine sinus CT already encodes whether chronic rhinosinusitis is eosinophilic.
A two-stage pipeline — nnU-Net v2 segmentation followed by a **1.17 M-parameter**
3D CNN — recovers it at **AUC 0.897** on a held-out test set, with no manual
region selection. Controlled ablations show the segmentation mask is what
carries the performance, not the input representation and not model capacity.

## Abstract

**Purpose.** Eosinophilic and non-eosinophilic chronic rhinosinusitis (CRS)
require different treatments, yet endotype assignment relies on invasive biopsy.
We investigated whether routine sinus computed tomography (CT) alone could
support automated, non-invasive endotype prediction.

**Methods.** In this single-center retrospective study of 184 patients with
histopathologically confirmed labels, we developed a minimal two-stage pipeline:
nnU-Net v2 segments the paranasal sinuses, and a lightweight 3D convolutional
neural network classifies the segmented volume. To identify what determined
performance, we ablated input representation, region-of-interest masking, and
classifier. The development set (n = 147) was evaluated by five-fold stratified
cross-validation; a held-out test set (n = 37) was assessed once, at a Youden's J
threshold pre-specified from Platt-calibrated out-of-fold predictions.

**Results.** The pooled out-of-fold area under the receiver operating
characteristic (ROC) curve (AUC) was 0.825. On the held-out set, the calibrated
five-fold ensemble achieved an AUC of 0.897 (95% CI: 0.776–0.994), sensitivity
of 72.7%, and specificity of 86.7%. In the ablations, neither tissue-channel
decomposition nor added capacity improved discrimination (single-channel AUC
0.903; no improvement with deeper networks), whereas removing the segmentation
mask reduced the AUC to 0.564 (near chance); segmentation (mean Dice 0.920) was
thus the decisive component. Safety-Aware ROC analysis identified rule-out and
rule-in zones meeting negative predictive value (NPV) ≥ 0.95 and positive
predictive value (PPV) ≥ 0.90 simultaneously.

**Conclusions.** Routine sinus CT encodes endotype information that a fully
automated pipeline can extract; performance depends on segmentation rather than
on input representation or capacity. These results motivate external validation
of imaging-based CRS endotype triage.

## Method

<table>
<tr><td width="60%">

**Stage 1 — Segmentation.** nnU-Net v2 (`3d_fullres`) segments the paranasal
sinuses from the CT volume. Trained on a separate 20-patient annotated set,
disjoint from the classification cohort, and frozen before Stage 2.
Mean Dice **0.920 ± 0.055**.

**Stage 2 — Classification.** The masked ROI is cropped with a 5 mm margin,
resampled to 1 mm isotropic, and decomposed into three HU channels. A 3D CNN
with four convolutional blocks (32→256), global average pooling and dropout
outputs the endotype.

</td><td>

| | |
|---|---|
| Parameters | 1.17 M |
| Input | 3 × ROI volume |
| Air channel | ≤ −150 HU |
| Soft tissue | −150 – 250 HU |
| Bone | ≥ 250 HU |
| Spacing | 1 × 1 × 1 mm |

</td></tr>
</table>

## Results

Development set *n* = 147 (five-fold stratified, patient-level CV); held-out
test set *n* = 37, evaluated once at a threshold frozen before the test data
were touched (Youden's *J* on Platt-calibrated out-of-fold predictions,
τ = 0.5849).

| Split | AUC | Sensitivity | Specificity |
|:--|:--:|:--:|:--:|
| Pooled out-of-fold (*n* = 147) | 0.825 | 0.851 | 0.667 |
| **Held-out test (*n* = 37)** | **0.897** <sub>(95% CI 0.776–0.994)</sub> | **0.727** | **0.867** |

### What the performance depends on

| Ablation | CV AUC | Held-out AUC |
|:--|:--:|:--:|
| Three-channel masked (pre-specified) | 0.825 | 0.897 |
| Single-channel masked HU | 0.840 | 0.903 |
| Two-channel, ≥ 250 HU excluded | 0.756 | 0.782 |
| **Segmentation mask removed** | 0.721 | **0.564** |

| Architecture | Params | CV AUC | Held-out AUC |
|:--|:--:|:--:|:--:|
| **Sinus3DCNN (ours)** | **1.17 M** | 0.825 | **0.897** |
| 3D EfficientNet-B0 | 4.69 M | 0.666 | 0.788 |
| 3D DenseNet-121 | 11.29 M | 0.711 | 0.821 |
| 3D ResNet-18 | 33.21 M | 0.852 | 0.861 |

Neither the channel decomposition nor 28× more parameters improves held-out
discrimination. Removing the
mask drops held-out AUC to near chance *while cross-validation stays well above
it* — the unmasked model latches onto background cues that do not generalize.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and, for GPU training, an NVIDIA
driver supporting CUDA 12.8.

```bash
git clone https://github.com/PLASS-Lab/SinuClass3D
cd SinuClass3D
uv sync
```

`uv sync` builds `.venv` from the versions pinned in `uv.lock`, so the
environment is reproducible. Data preparation additionally needs the
[`dcm2niix`](https://github.com/rordenlab/dcm2niix) command-line tool (DICOM
conversion) and [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (Stage 1, the
paper used `nnunetv2` 2.6.4); both are installed separately, outside `uv`.

## Data

**Patient data is not included and cannot be redistributed.** The CT volumes and
labels come from a single center under an IRB protocol that does not permit
public release.

To run on your own cohort, prepare the data in four steps.

**1. DICOM → NIfTI.**
[`classification_dicom_to_nifti_converter.py`](datasets/classification_dicom_to_nifti_converter.py)
converts each classification patient's CT, and
[`segmentation_dicom_to_nifti_converter.py`](datasets/segmentation_dicom_to_nifti_converter.py)
converts the annotated segmentation set (CT plus manual sinus mask). Both take
`--patient-ids`, `--input-base` and `--output-base`;
[`datasets/dataset_structure.md`](datasets/dataset_structure.md) describes the
layout they write.

**2. Stage 1: sinus masks with nnU-Net v2.** With `nnUNet_raw`,
`nnUNet_preprocessed` and `nnUNet_results` set:

```bash
# annotated set -> nnU-Net dataset (CT channel, binary sinus label)
uv run python datasets/build_nnunet_dataset.py --segmentation-root <annotated set> --dataset-id 998
nnUNetv2_plan_and_preprocess -d 998 -c 3d_fullres --verify_dataset_integrity
for f in 0 1 2 3 4; do nnUNetv2_train 998 3d_fullres $f; done

# classification CTs, linked as <patient_id>_0000.nii.gz in one folder
nnUNetv2_predict -i <ct folder> -o <mask folder> -d 998 -c 3d_fullres -f 0 1 2 3 4
```

Each patient's `mask_path` is then `<mask folder>/<patient_id>.nii.gz`. The
segmentation Dice reported in the paper (0.920 ± 0.055) is this five-fold
cross-validation run with `-tr nnUNetTrainer_100epochs`. Segmentation weights
are not distributed.

**3. Crops and tissue channels.**

```bash
uv run python datasets/generate_masked_crops_3ch.py --csv <manifest> --output-csv <manifest with channels>
```

The input manifest needs `patient_id`, `class`, `image_path` and `mask_path`.
For each patient the script crops a box with a 5 mm margin around the mask,
resamples it to 1 mm isotropic, and writes the three masked channels and the
single-channel masked HU next to the CT, adding their path columns. Without
`--output-csv` it overwrites the input manifest.

**4. Manifest.** The output of step 3 matches
[`csv/patient_ids_TEMPLATE.csv`](csv/patient_ids_TEMPLATE.csv): one row per
patient with a `class` column (`ECRS` / `non-ECRS`). Point
`data.paths.csv_path` at the development manifest and
`data.external_test.csv_path` at the held-out one.

## Training

```bash
# five-fold cross-validation on the development set
uv run python scripts/train.py
```

Configuration is [Hydra](https://hydra.cc)-based; anything in `configs/` can be
overridden on the command line.

```bash
# capacity comparison
uv run python scripts/train.py models=resnet3d core.experiment_name=resnet3d_compare

# single-channel ablation
uv run python scripts/train.py model.in_channels=1 data.image_type=masked_hu_1ch_path
```

Weights & Biases logging is **off** by default; enable it with
`wandb.enabled=true` after `wandb login`.

<details>
<summary><b>Training configuration</b></summary>

Batch size 4, mixed precision, up to 1,000 epochs with early stopping (patience
200 on validation loss). AdamW, lr 1e-4, weight decay 1e-4, gradient clipping at
1.0, five-epoch warm-up from 1e-6 followed by plateau decay (×0.5, patience 10).
Weighted cross-entropy with inverse-frequency class weights. Seed 42,
deterministic kernels.

Augmentation: left–right and anterior–posterior flips (*p* = 0.5; no
superior–inferior flips, which are not anatomically plausible for head CT);
affine (*p* = 0.25; ±7°, scale 0.95–1.05, in-plane translation 8–10 mm,
through-plane 4–6 mm); Gaussian noise (σ = 0.02, *p* = 0.15); Gaussian smoothing
(σ = 0.25–0.5, *p* = 0.2); intensity scale and shift (*p* = 0.2 each). No
augmentation and no test-time augmentation at evaluation.

</details>

## Evaluation

```bash
# held-out test cohort, calibrated five-fold ensemble
uv run python scripts/evaluate.py \
    evaluation.held_out_test_cohort=true \
    evaluation.experiment_dir=experiments/<experiment_name> \
    data.paths.csv_path=<development manifest> \
    data.external_test.csv_path=<held-out manifest>
```

`experiments/<experiment_name>` is the directory `scripts/train.py` wrote
(`core.experiment_name`). The script reads the development manifest before it
switches to the held-out one, so pass both.

The ensemble is formed at inference: the five fold models' raw probabilities are
averaged, then the Platt calibrator fitted on pooled out-of-fold predictions is
applied once. There is no separate ensemble checkpoint.

## Pretrained models

| Artifact | Files | Notes |
|:--|:--:|:--|
| Fold checkpoints | 5 × ~13 MB | `fold_{1..5}/checkpoints/best_model.pth`, selected by lowest validation loss |
| Platt calibrator | 1 | needed for the operating point; AUC is rank-based and unaffected |

To be released on the [Releases](https://github.com/PLASS-Lab/SinuClass3D/releases)
page upon acceptance of the paper. Together they reproduce the held-out results
without access to the training data.

## Repository layout

```
configs/     Hydra configuration (core, data, model, training, evaluation, inference)
datasets/    DICOM -> NIfTI conversion, nnU-Net dataset builder, crops and channels
src/         Library code: data, models, training, evaluation, inference, utils
scripts/     Entry points: train, evaluate, predict
csv/         Cohort manifest schema
```

This repository holds the training and evaluation pipeline. Figure generation,
HTML reporting and Grad-CAM++ explainability code are not part of this release.

## Citation

The paper is under review. A BibTeX entry will be added here once it is assigned
a DOI.

```bibtex
@article{yu_sinuclass3d,
  title   = {Non-invasive prediction of eosinophilic chronic rhinosinusitis
             endotype from routine sinus CT: a fully automated pipeline with
             safety-aware deployment analysis},
  author  = {Yu, Ethan and Kim, Jin Youp and Han, Doo Hee and Won, Tae-Bin and
             Kim, Hyun Jik and Kim, Dong-Young and Rhee, Chae-Seo and Son, Yunsik},
  note    = {Submitted to Computerized Medical Imaging and Graphics},
  year    = {2026}
}
```

## Acknowledgements

Segmentation is built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
(Isensee et al., *Nature Methods* 2021). The deployment analysis reported in the
paper uses the [SA-ROC](https://github.com/MGH-LMIC/SA-ROC) framework (Kim et
al., *npj Digital Medicine* 2026), which is distributed by its authors under
their own terms and is not bundled here.

## License

[Apache License 2.0](LICENSE). The license covers the code only; it does not
grant any rights to the clinical data.
