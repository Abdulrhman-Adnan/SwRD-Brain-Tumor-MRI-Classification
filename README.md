# SwRD — Brain Tumor MRI Classification

Multiclass brain tumor classification on MRI scans using the SwRD architecture — a parallel Swin Transformer + ResNet-50 fusion model.

This implementation is inspired by the research paper **[Advancing Brain Tumor MRI Classification using SwRD: A Parallel Swin Transformer–ResNet Approach](https://www.sciencedirect.com/science/article/pii/S2096579625000348#abs0010)** 

The notebook achieved **99.88% validation accuracy** on the 31,464-image dataset described below.

---

## Features

- SwRD model implementation (Swin-Base + ResNet-50 parallel fusion)
- Combined multi-source MRI dataset with offline augmentation
- Full training pipeline with early stopping and checkpointing
- Stratified 80/20 train/validation split
- Per-class classification report and confusion matrix
- Grad-CAM visualisation on the ResNet branch
- Checkpoint save/load for resuming training after kernel restart
- Deterministic seeding for reproducible runs

---

## Dataset

The training set is built from two publicly available Kaggle datasets:

| Source | Link | Role |
|:---|:---|:---|
| Hashemi | `mohammadhossein77/brain-tumors-dataset` | 21,672 preprocessed scans |
| SARTAJ | `sartajbhuvaji/brain-tumor-classification-mri` | 3,264 raw scans × 3 augmentations = 9,792 |

The SARTAJ images are augmented offline with zoom, Gaussian blur, and random crop. The combined corpus totals **31,464 images** across four classes: glioma, meningioma, no tumor, and pituitary.

---

## Model Architecture

SwRD runs two backbone branches in parallel on the same 224×224 RGB input:

| Branch | Backbone | Output |
|:---|:---|:---|
| CNN branch | ResNet-50 (ImageNet pretrained, head removed) | 2048-d |
| Transformer branch | Swin-Base patch4 window7 (head removed, avg pool) | 1024-d |

The two feature vectors are concatenated (3072-d) and passed to a single linear layer that produces the 4-class logits.

---

## Training

- **Optimizer:** AdamW (lr = 1e-5, weight decay = 1e-2)
- **Loss:** CrossEntropyLoss
- **Epochs:** 40 (early stopping patience = 12)
- **Batch size:** 32
- **Reproducibility:** All random seeds fixed (Python, NumPy, PyTorch, CUDA)

Checkpoints bundle model weights, training history, best accuracy, and metadata so training can be resumed cleanly.

---

## Results

| Metric | Value |
|:---|:---|
| Validation Accuracy | **99.88%** |

The notebook also generates:

- Per-class precision, recall, and F1-score (classification report)
- Confusion matrix heatmap
- Grad-CAM overlays showing which regions the model focuses on

All outputs are saved inside the notebook.

---

## Notebook Contents

1. Install dependencies
2. Imports and reproducibility setup
3. Hyperparameter configuration
4. Dataset download and preparation
5. Transforms and PyTorch Dataset/DataLoader
6. SwRD model definition
7. Training utilities (`run_epoch`, `fit`, checkpoint helpers)
8. Train from scratch or load existing checkpoint
9. Training curve plots (accuracy and loss)
10. Evaluation (classification report + confusion matrix)
11. Reload saved model for inference
12. Grad-CAM visualisation

---

## Project Structure

```
├── SwRD_Training.ipynb          # Main training notebook
├── README_SwRD.md               # This file
```

---

## Requirements

| Package | Purpose |
|:---|:---|
| Python 3.10+ | Runtime |
| PyTorch | Deep learning framework |
| torchvision | ResNet-50 backbone + ImageNet weights |
| timm | Swin Transformer from Hugging Face |
| numpy | Array operations |
| pandas | DataFrame handling |
| matplotlib / seaborn | Plotting |
| scikit-learn | Train/test split, metrics |
| albumentations | Image augmentation |
| kagglehub | Dataset download |
| Pillow | Image I/O |

Install everything with:

```bash
pip install -q torch torchvision timm scikit-learn albumentations matplotlib seaborn tqdm kagglehub Pillow
```

---

## Usage

1. **Open the notebook** in Jupyter or Kaggle.
2. **Run cells 1–3** to install dependencies, set seeds, and configure hyperparameters.
3. **Run cell 4** to download and assemble the dataset (requires Kaggle API credentials for `kagglehub`).
4. **Run cells 5–6** to build the dataloaders and model.
5. **Run cell 8** to train. If `swrd_multiclass_best.pt` already exists, the notebook loads it instead. Set `FORCE_TRAIN = True` to retrain.
6. **Run cells 9–10** to plot training curves and evaluate on the validation set.
7. **Run cell 11** to reload a checkpoint after a kernel restart.
8. **Run cell 12** to generate Grad-CAM visualisations.

---

## Notes

- The notebook uses checkpoint bundles (weights + history + metadata), so you can pick up where you left off without retraining.
- Grad-CAM hooks the last ResNet conv layer because the Swin branch loses spatial resolution after global average pooling.
- Swin-Base is a large model. If you hit VRAM limits, reduce `BATCH_SIZE` in the hyperparameter cell.
- All notebook outputs (training logs, plots, reports) are already saved in the `.ipynb` file.

---

## Acknowledgments

This implementation is inspired by the research paper:

> **[Advancing Brain Tumor MRI Classification using SwRD: A Parallel Swin Transformer–ResNet Approach](https://www.sciencedirect.com/science/article/pii/S2096579625000348#abs0010)**

The Hashemi dataset is hosted by `mohammadhossein77` and the SARTAJ dataset by `sartajbhuvaji`, both on Kaggle.
