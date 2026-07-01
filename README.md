# MedJEPA

This repository contains the current working code for the **MedJEPA** project, a master thesis project at **Goethe University Frankfurt**.
The goal of this repository is to make the current training, evaluation, augmentation, and visualization scripts available to collaborators. 
The code is primarily focused on self-supervised representation learning for mammography images using a LeJEPA / SiGReg-style training setup.

## Repository contents

### `train_medjepa_mg_v4.py`

Main training script for mammography-based MedJEPA experiments.

This script trains a ViT-based LeJEPA model on mammography images stored as a CSV file plus a `.bin` memmap file. It supports distributed training with PyTorch DDP, multi-view augmentations, gradient accumulation, checkpoint saving, warmup epochs, cosine learning-rate scheduling, and optional diagnostic logging.

Main features:

* Mammography dataset loading from CSV and raw `uint16` memmap
* Patient-disjoint train / validation / test split support
* Multi-view augmentation training
* ViT backbone via `timm`
* LeJEPA / SiGReg-style self-supervised objective
* Distributed training with `torchrun` / DDP
* Gradient accumulation
* Optional checkpoint saving every `x` epochs
* Warmup epochs and cosine learning-rate decay
* Periodic representation diagnostics such as projection standard deviation and embedding norm

### `mg_lejepa_aug_v4.json`

Augmentation and preprocessing configuration used by the mammography training script.
This file defines the image preprocessing and augmentation pipeline used during training. It includes settings for resizing, foreground cropping, intensity normalization, geometric augmentations, contrast/brightness changes, and masking of possible image-corner watermarks or scanner annotations.
The config is read by `train_medjepa_mg_v4.py` and can be modified without changing the training code.

### `run_medjepa_linear_probe_report.py`

Evaluation script for frozen-representation linear probing.
This script loads a trained MedJEPA checkpoint, extracts frozen backbone embeddings, and trains linear probes for metadata and clinical labels. It is used to measure which factors are linearly encoded in the learned representation.
Typical probe targets include:

* Dataset/source
* Machine / scanner model
* Machine family
* Mammography view
* Laterality
* Collapsed BI-RADS / actionability label

The script writes a JSON report containing accuracy, balanced accuracy, macro F1, weighted F1, class counts, and timing information.

### `create_medjepa_pca_report_v2.py`

PCA visualization script for trained MedJEPA checkpoints.
This script extracts frozen backbone embeddings from a trained model and creates a PDF report showing 2D and 3D PCA projections of the latent space. The same PCA coordinates are colored by different metadata fields to analyze which factors dominate the representation.
Typical colorings include:

* Collapsed BI-RADS
* Numeric BI-RADS
* Age group
* View
* Laterality
* View/laterality combination
* Dataset/source
* Machine family
* Machine
* Segmentation availability

The output is a PDF report intended for qualitative inspection of the learned latent structure.

### `visualize_mg_lejepa_aug_v2.ipynb`

Notebook for inspecting the mammography preprocessing and augmentation pipeline.
This notebook visualizes original images, preprocessed images, and multiple augmented views generated from the augmentation configuration. It is useful for checking whether the augmentations are medically plausible and whether preprocessing steps such as foreground cropping or watermark masking behave as expected.
The notebook can be used to inspect examples by class, dataset, machine family, or view.

## Typical workflow

1. Configure preprocessing and augmentations in `mg_lejepa_aug_v4.json`.
2. Train a model with `train_medjepa_mg_v4.py`.
3. Evaluate frozen embeddings with `run_medjepa_linear_probe_report.py`.
4. Create PCA reports with `create_medjepa_pca_report_v2.py`.
5. Inspect or debug augmentations with `visualize_mg_lejepa_aug_v2.ipynb`.

## Notes

This repository is intended as a working research-code snapshot for collaborators, not as a polished software package. Paths, dataset locations, and cluster-specific commands may need to be adapted to the local environment.
The project is part of a master thesis at Goethe University Frankfurt.
