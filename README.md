<div align="center">

# Non-invasive Prediction of Eosinophilic Chronic Rhinosinusitis Endotype from Routine Sinus CT

### A Fully Automated Pipeline with Safety-Aware Deployment Analysis

**Ethan Yu**<sup>1</sup> &nbsp;&nbsp; **Jin Youp Kim**<sup>2</sup> &nbsp;&nbsp; **Yunsik Son**<sup>1,†</sup>

<sup>1</sup> Department of Computer Science and Artificial Intelligence, Dongguk University, Seoul, Korea <br>
<sup>2</sup> Department of Otorhinolaryngology–Head and Neck Surgery, Dongguk University Ilsan Hospital, Goyang, Korea <br>
<sup>†</sup> Corresponding author

[![Paper](https://img.shields.io/badge/Paper-under%20review-b31b1b.svg)](#citation)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-ee4c2c.svg)](pyproject.toml)
[![uv](https://img.shields.io/badge/managed%20by-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

</div>

<!-- Add a teaser here once you export one, e.g.:
<div align="center"><img src="assets/teaser.png" width="90%"></div>
The figure-generation code lives on the `main` branch. -->

---

## TL;DR

Routine sinus CT already encodes whether chronic rhinosinusitis is eosinophilic.
A two-stage pipeline — nnU-Net v2 segmentation followed by a **1.17 M-parameter**
3D CNN — recovers it at **AUC 0.897** on a held-out test set, with no manual
region selection. Controlled ablations show the segmentation mask is what
carries the performance, not the input representation and not model capacity.

## Abstract

Eosinophilic chronic rhinosinusitis (ECRS) and non-eosinophilic CRS require
different treatment strategies, yet endotype assignment still relies on invasive
tissue biopsy. We ask whether routine paranasal sinus CT alone can support
automated, non-invasive endotype prediction. In a single-center retrospective
study of 184 patients with histopathologically confirmed labels, we build a
deliberately minimal two-stage pipeline: nnU-Net v2 segments the paranasal
sinuses, and a lightweight 3D CNN classifies the segmented volume. To identify
what actually drives performance, we ablate the input representation, the ROI
masking, and the classifier. Neither the tissue-channel decomposition nor added
capacity is required; removing the segmentation mask collapses held-out AUC to
near chance, identifying the segmentation stage as the decisive component.

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
| Single raw-HU channel | 0.840 | 0.903 |
| Two-channel, ≥ 250 HU excluded | 0.756 | 0.782 |
| **Segmentation mask removed** | 0.732 | **0.564** |

| Architecture | Params | CV AUC | Held-out AUC |
|:--|:--:|:--:|:--:|
| **Sinus3DCNN (ours)** | **1.17 M** | 0.825 | **0.897** |
| 3D EfficientNet-B0 | 4.69 M | 0.666 | 0.788 |
| 3D DenseNet-121 | 11.29 M | 0.723 | 0.821 |
| 3D ResNet-18 | 33.21 M | 0.847 | 0.861 |

Neither the channel decomposition nor 29× more parameters helps. Removing the
mask drops held-out AUC to near chance *while cross-validation stays well above
it* — the unmasked model latches onto background cues that do not generalize.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and, for GPU training, an NVIDIA
driver supporting CUDA 12.8.

```bash
git clone https://github.com/0x00000024/SinuClass3D
cd SinuClass3D
uv sync
```

`uv sync` builds `.venv` from the versions pinned in `uv.lock`, so the
environment is reproducible. For the dataset-preparation tools
(TotalSegmentator):

```bash
uv sync --extra preprocessing
```

## Data

**Patient data is not included and cannot be redistributed.** The CT volumes and
labels come from a single center under an IRB protocol that does not permit
public release.

To run on your own cohort, build a manifest matching
[`csv/patient_ids_TEMPLATE.csv`](csv/patient_ids_TEMPLATE.csv) — one row per
patient, a `class` column (`ECRS` / `non_ECRS`), and one path column per
preprocessed representation — then point `data.paths.csv_path` at it.
[`datasets/dataset_structure.md`](datasets/dataset_structure.md) describes the
on-disk layout, and the scripts in [`datasets/`](datasets/) cover DICOM → NIfTI
conversion, ROI extraction and channel generation.

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
uv run python scripts/evaluate.py evaluation.held_out_test_cohort=true
```

The ensemble is formed at inference: the five fold models' raw probabilities are
averaged, then the Platt calibrator fitted on pooled out-of-fold predictions is
applied once. There is no separate ensemble checkpoint.

## Pretrained models

| Artifact | Files | Notes |
|:--|:--:|:--|
| Fold checkpoints | 5 × ~13 MB | `fold_{1..5}/checkpoints/best_model.pth`, selected by lowest validation loss |
| Platt calibrator | 1 | needed for the operating point; AUC is rank-based and unaffected |

Released on the [Releases](https://github.com/0x00000024/SinuClass3D/releases)
page. Together they reproduce the held-out results without access to the
training data.

## Repository layout

```
configs/     Hydra configuration (core, data, model, training, evaluation, inference)
datasets/    DICOM -> NIfTI conversion, ROI extraction, channel generation
src/         Library code: data, models, training, evaluation, inference, utils
scripts/     Entry points: train, evaluate, predict
csv/         Cohort manifest schema
```

This branch is the training and evaluation pipeline. Figure generation, HTML
reporting and Grad-CAM++ explainability are kept with the manuscript sources on
the `main` branch rather than in the code release.

## Citation

The paper is under review. A BibTeX entry will be added here once it is assigned
a DOI.

```bibtex
@article{yu_sinuclass3d,
  title   = {Non-invasive prediction of eosinophilic chronic rhinosinusitis
             endotype from routine sinus CT: a fully automated pipeline with
             safety-aware deployment analysis},
  author  = {Yu, Ethan and Kim, Jin Youp and Son, Yunsik},
  note    = {Manuscript under review},
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
