# TotalSegmentator

Tool for segmentation of most major anatomical structures in any CT or MR image. It was trained on a wide range of different CT and MR images (different scanners, institutions, protocols,...) and therefore works well on most images. A large part of the training dataset can be downloaded here: [CT dataset](https://doi.org/10.5281/zenodo.6802613) (1228 subjects) and [MR dataset](https://zenodo.org/doi/10.5281/zenodo.11367004) (616 subjects). You can also try the tool online at [totalsegmentator.com](https://totalsegmentator.com/) or as [3D Slicer extension](https://github.com/lassoan/SlicerTotalSegmentator).

**ANNOUNCEMENT: We created a platform where anyone can help annotate more data to further improve TotalSegmentator: [TotalSegmentator Annotation Platform](https://annotate.totalsegmentator.com).**  
  
**ANNOUNCEMENT: We created web applications for [abdominal organ volume](https://compute.totalsegmentator.com/volume-report/), [Evans index](https://compute.totalsegmentator.com/evans-index/), and [aorta diameter](https://compute.totalsegmentator.com/aorta-report/).**

Main classes for CT and MR:  
![Alt text](resources/imgs/overview_classes_v2.png)

TotalSegmentator supports a lot more structures. See [subtasks](#subtasks) or [here](https://backend.totalsegmentator.com/find-task/) for more details.

Created by the department of [Research and Analysis at University Hospital Basel](https://www.unispital-basel.ch/en/radiologie-nuklearmedizin/forschung-radiologie-nuklearmedizin).
If you use it please cite our [Radiology AI paper](https://pubs.rsna.org/doi/10.1148/ryai.230024) ([free preprint](https://arxiv.org/abs/2208.05868)). If you use it for MR images please cite the [TotalSegmentator MRI Radiology paper](https://pubs.rsna.org/doi/10.1148/radiol.241613) ([free preprint](https://arxiv.org/abs/2405.19492)). Please also cite [nnUNet](https://github.com/MIC-DKFZ/nnUNet) since TotalSegmentator is heavily based on it.

---

## Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  
3. [Enhanced Liver Vessel & Portal/Hepatic Split Pipeline (NEW)](#enhanced-liver-vessel--portalhepatic-split-pipeline-new)  
4. [Subtasks](#subtasks)  
5. [Advanced Settings](#advanced-settings)  
6. [Other Commands](#other-commands)  
7. [Web Applications](#web-applications)  
8. [Docker](#run-via-docker)  
9. [Resource Requirements](#resource-requirements)  
10. [Python API](#python-api)  
11. [Latest Master](#install-latest-master-branch-contains-latest-bug-fixes)  
12. [Typical Problems](#typical-problems)  
13. [Class Details](#class-details)  

---

## Installation

TotalSegmentator works on Ubuntu, Mac, and Windows and on CPU and GPU.

Install dependencies:
* Python >= 3.9
* [PyTorch](http://pytorch.org/) >= 2.0.0 and < 2.6.0 (and <2.4 for Windows)

Optionally:
* if you use the option `--preview` you have to install xvfb (`apt-get install xvfb`) and fury (`pip install fury`)

Install latest release:
```bash
pip install TotalSegmentator
```

### (Optional) Developer / Editable Install (for enhanced pipeline)

```bash
# Clone your fork
git clone https://github.com/<your-username>/TotalSegmentator.git
cd TotalSegmentator

# Create & activate a virtual environment
python -m venv totalseg_env
source totalseg_env/bin/activate   # (Linux/Mac)
# Windows: .\totalseg_env\Scripts\activate

# Upgrade build tooling
pip install --upgrade pip setuptools wheel

# Editable install
pip install -e .

# (Optional) extra tools for enhanced vessels & splitting
pip install scikit-image trimesh
```

If you encounter `error: invalid command 'bdist_wheel'`, ensure `wheel` is installed in the active environment:
```bash
pip install wheel
```

---

## Usage

For CT images:
```bash
TotalSegmentator -i ct.nii.gz -o segmentations
```

For MR images:
```bash
TotalSegmentator -i mri.nii.gz -o segmentations --task total_mr
```

> Input can be a NIfTI file, a directory of DICOM slices, or a zip containing one study.  
> For CPU-only environments consider `--fast` or a focused `--roi_subset` to reduce runtime.  
> Not a medical device; research use only.

---

## Enhanced Liver Vessel & Portal/Hepatic Split Pipeline (NEW)

This repository includes an optional, heuristic post-processing pipeline to:
1. Enhance the raw liver vessel segmentation.
2. Split intrahepatic vessels into Portal Vein branches vs Hepatic Veins.

### Key Components
| File | Purpose |
|------|---------|
| `totalsegmentator/enhanced_liver_vessels.py` | Enhancement & metadata (fallback handling, component pruning) |
| `totalsegmentator/bin/TotalSegmentatorEnhanced.py` | CLI wrapping baseline liver + vessel enhancement + (optional) splitting |
| `totalsegmentator/vessel_split.py` | Heuristic portal vs hepatic classification |

### When to Use
Use on contrast-enhanced abdominal CT (preferably portal venous phase). Non-contrast or atypical phases may reduce split reliability. The splitting is heuristic (no learned classifier yet).

### Basic Enhanced Run
```bash
TotalSegmentatorEnhanced \
  -i data/case01.nii.gz \
  -o results_case01 \
  --mode enhanced_liver \
  --robust_crop
```

Outputs (core):
* `enhanced_liver_vessels.nii.gz` – Enhanced binary liver vessel mask  
* `enhanced_liver_vessels_metadata.json` – Metadata & QC (voxels, volume, fallbacks)

### Portal vs Hepatic Split (requires portal vein & IVC support labels)
Option A: Provide support labels (faster):
```bash
# Generate only needed supporting structures
TotalSegmentator -i data/case01.nii.gz -o split_support \
  --roi_subset portal_vein_and_splenic_vein inferior_vena_cava liver --robust_crop

TotalSegmentatorEnhanced \
  -i data/case01.nii.gz \
  -o results_case01 \
  --mode enhanced_liver \
  --robust_crop \
  --split_portal_hepatic \
  --split_support_dir split_support
```

Option B: Let the enhanced CLI generate them automatically:
```bash
TotalSegmentatorEnhanced \
  -i data/case01.nii.gz \
  -o results_case01_auto \
  --mode enhanced_liver \
  --robust_crop \
  --split_portal_hepatic \
  --generate_split_support
```

Additional split outputs:
* `portal_vein_branches.nii.gz`
* `hepatic_veins.nii.gz`
* `liver_vessels_labeled.nii.gz` (1=portal, 2=hepatic)
* (optional) `liver_vessels_skeleton_labeled.nii.gz` if skeletonization enabled and scikit-image installed  
QC is appended under `"portal_hepatic_split_qc"` in the metadata file.

### Important CLI Flags (Enhanced Mode)

| Flag | Description |
|------|-------------|
| `--no_fallback_full_liver` | Disable full-volume mask fallback if liver segmentation missing (fail instead) |
| `--min_component_size` | Minimum connected component size kept in enhancement |
| `--split_portal_hepatic` | Perform portal vs hepatic classification |
| `--split_support_dir PATH` | Directory containing supporting labels (portal vein, IVC, liver) |
| `--generate_split_support` | Auto-generate support labels if missing |
| `--no_skeleton` | Disable skeleton-based refinement in splitting |
| `--min_split_component_size` | Prune small portal/hepatic components |

### Interpreting QC (Example)
```json
"portal_hepatic_split_qc": {
  "portal_voxels": 14509,
  "hepatic_voxels": 18597,
  "total_vessel_voxels": 35354,
  "portal_fraction": 0.4104,
  "hepatic_fraction": 0.5260,
  "portal_seed_voxels": 5303,
  "hepatic_seed_voxels": 7071,
  "use_skeleton": false
}
```
Typical portal_fraction ~0.3–0.6 (varies with phase & enhancement). Extreme imbalance (<0.05) suggests seed failure.

### 3D Slicer Visualization (Recommended)
1. Load `data/case01.nii.gz` (CT).
2. Load `liver_vessels_labeled.nii.gz` as LabelMap → Import into Segmentations.
3. Rename segments: Label 1 → Portal_Vein, Label 2 → Hepatic_Veins.
4. Color & enable 3D display.
5. Optionally load `portal_vein_branches.nii.gz` & `hepatic_veins.nii.gz` separately.

### Limitations
* Heuristic splitting (not a trained classifier).
* Arterial branches are not isolated; hepatic artery separation not yet implemented.
* Accuracy depends on contrast phase and segmentation quality.

### Roadmap Ideas
* Confidence scoring (seed coverage & balance)
* Hepatic artery detection
* Learned branch classification
* Phase-aware weighting

---

## Subtasks

(Original section retained)

![Alt text](resources/imgs/overview_subclasses_2.png)

Next to the default task (`total`) there are more subtasks with more classes. If the taskname ends with `_mr` it works for MR images, otherwise for CT images.

Openly available for any usage (Apache-2.0 license):
* **total**: default task containing 117 main classes (see [here](https://github.com/wasserth/TotalSegmentator#class-details) for a list of classes)
* **total_mr**: default task containing 50 main classes on MR images (see [here](https://github.com/wasserth/TotalSegmentator#class-details) for a list of classes)
* **lung_vessels**: lung_vessels (cite [paper](https://www.sciencedirect.com/science/article/pii/S0720048X22001097)), lung_trachea_bronchia
* **body**: body, body_trunc, body_extremities, skin
* **body_mr**: body_trunc, body_extremities (for MR images)
* **vertebrae_mr**: sacrum, vertebrae_L5, vertebrae_L4, vertebrae_L3, vertebrae_L2, vertebrae_L1, vertebrae_T12, vertebrae_T11, vertebrae_T10, vertebrae_T9, vertebrae_T8, vertebrae_T7, vertebrae_T6, vertebrae_T5, vertebrae_T4, vertebrae_T3, vertebrae_T2, vertebrae_T1, vertebrae_C7, vertebrae_C6, vertebrae_C5, vertebrae_C4, vertebrae_C3, vertebrae_C2, vertebrae_C1
* **cerebral_bleed**: intracerebral_hemorrhage (cite [paper](https://www.mdpi.com/2077-0383/12/7/2631))*
* **hip_implant**: hip_implant*
* **pleural_pericard_effusion**: pleural_effusion, pericardial_effusion*
* **head_glands_cavities** ...
* *(list truncated for brevity – keep the original full listing here)*

(Keep the rest of this section exactly as in the original README: open tasks, licensed tasks, usage example, etc.)

---

## Advanced settings
(Original content unchanged)

* `--device`: ...
* `--fast`: ...
* `--roi_subset`: ...
* `--robust_crop`: ...
* `--preview`: ...
* `--ml`: ...
* `--statistics`: ...
* `--radiomics`: ...

---

## Other commands
(Original content unchanged – phase prediction, modality, combining masks, Evans index, weights, license, probabilities)

---

## Web applications
(Original content unchanged)

---

## Run via docker
(Original content unchanged)

---

## Resource Requirements
(Original content unchanged)

---

## Python API
(Original content unchanged)

---

## Install latest master branch (contains latest bug fixes)
(Original content unchanged)

---

## Train/validation/test split
(Original content unchanged)

---

## Typical problems
(Original content unchanged)

---

## Running v1
(Original content unchanged)

---

## Other
(Original content unchanged)

---

## Reference
(Original citation block unchanged)

---

## Class details
(Original tables unchanged — retained below)

<details>
<summary>Click to expand Class Details (CT total)</summary>

<!-- Original large class table stays here -->
</details>

<details>
<summary>Click to expand Class Details (MR total_mr)</summary>

<!-- Original MR class table stays here -->
</details>

---

### Changelog (Enhanced Pipeline Addendum)

| Version | Change |
|---------|--------|
| 0.2.0 (dev) | Added enhanced liver vessel pipeline + portal/hepatic heuristic split |
| 0.2.1 (planned) | Confidence scoring & centerline statistics |
| 0.3.0 (planned) | Hepatic artery separation & phase-aware heuristics |

---

### Feedback

Issues & improvement suggestions welcome. For enhanced pipeline ideas (artery separation, learned classifier, etc.) please open an issue with the label `enhanced-liver`.

---

### DISCLAIMER

The enhanced portal/hepatic splitting is heuristic and should be validated against expert-annotated data for clinical or regulatory applications.

---