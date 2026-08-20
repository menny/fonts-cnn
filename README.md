# Multi-Head Font & Style Recognition with Deep CNNs

A complete, production-ready deep learning pipeline for synthetic dataset generation, model training, and dual-head font recognition (Font Family & Discrete Style classification) using PyTorch and ONNX Runtime.

---

## 📑 Table of Contents
1. [Overview & System Architecture](#overview--system-architecture)
2. [Project Structure](#project-structure)
3. [Installation & Requirements](#installation--requirements)
4. [Scripts & Workflow](#scripts--workflow)
   - [1. `download_google_fonts.py` (Font Downloader)](#1-download_google_fontspy)
   - [2. `generate_dataset.py` (Synthetic Dataset Generator)](#2-generate_datasetpy)
   - [3. `dataset.py` (PyTorch Dataset & Preprocessing Pipeline)](#3-datasetpy)
   - [4. `train.py` (Multi-Head CNN Trainer & ONNX Exporter)](#4-trainpy)
   - [5. `predict.py` (Standalone PyTorch & ONNX Inference)](#5-predictpy)
5. [End-to-End Pipeline Walkthrough](#end-to-end-pipeline-walkthrough)
6. [Model Architecture Details](#model-architecture-details)
7. [Inference & Output Formats](#inference--output-formats)

---

## 🏛 Overview & System Architecture

This repository provides an end-to-end framework for identifying font families and styles from rendered text patches. The architecture uses a multi-task convolutional neural network (`MultiHeadFontCNN`) with shared low/mid-level feature extractors and separate classification heads:

```
                          ┌────────────────────────┐
                          │ Input: (B, 1, 256, 256)│
                          └───────────┬────────────┘
                                      │
                   ┌──────────────────▼──────────────────┐
                   │ Conv Block 1: Conv(1->32) + BN + MP │ (128x128)
                   │ Conv Block 2: Conv(32->64) + BN + MP│ (64x64)
                   │ Conv Block 3: Conv(64->128)+ BN + MP│ (32x32)
                   │ Conv Block 4: Conv(128->256)+BN+MP  │ (16x16) + Dropout2d(0.15)
                   │ Conv Block 5: Conv(256->512)+BN+MP  │ (8x8)   + Dropout2d(0.15)
                   └──────────────────┬──────────────────┘
                                      │
                   ┌──────────────────▼──────────────────┐
                   │ AdaptiveAvgPool2d((4, 4)) -> 8192-D │
                   │ Linear(8192, 512) + BN1d + Drop(0.4)│
                   └─────────┬──────────────────┬────────┘
                             │                  │
               ┌─────────────▼────┐        ┌────▼─────────────┐
               │  Font Head (FC)  │        │  Style Head (FC) │
               │ (512 -> N_fonts) │        │   (512 -> 4)     │
               └──────────────────┘        └──────────────────┘
```

### Discrete Style Classification
Fonts are parsed and classified into 4 canonical style categories:
- `0`: **Regular** (Book, Roman, Medium, Normal)
- `1`: **Bold** (Black, Heavy, SemiBold, ExtraBold, Weight $\ge 600$)
- `2`: **Italic** (Oblique, Slanted, Italic)
- `3`: **Bold-Italic** (Bold + Italic combined)

---

## 📂 Project Structure

```
.
├── download_google_fonts.py  # Automated Google Fonts catalog downloader
├── generate_dataset.py       # High-throughput synthetic image patch generator
├── dataset.py                # PyTorch Dataset, DataLoaders, and image preprocessor
├── train.py                  # Multi-task CNN training loop and ONNX exporter
├── predict.py                # CLI inference runner supporting .pth & .onnx
├── downloaded_fonts/         # Raw font repository (.ttf, .otf)
└── runs/                     # Experiment checkpoints, ONNX graphs & metrics
```

---

## ⚙️ Installation & Requirements

Python 3.10+ is recommended. Install required packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # Or CUDA equivalent
pip install Pillow fonttools pandas numpy tqdm onnx onnxscript onnxruntime requests
```

---

## 🚀 Scripts & Workflow

### 1. `download_google_fonts.py`
Downloads font files directly from the Google Fonts repository without needing a full git clone.

```bash
# Download all fonts to local directory with 8 parallel download threads
python3 download_google_fonts.py --output_dir downloaded_fonts --workers 8
```

---

### 2. `generate_dataset.py`
Generates augmented image patches across diverse text templates (sentences, pangrams, timestamps, dates, and alphanumeric IDs).

**Key Features:**
- Reads TTF/OTF metadata tables (`name`, `OS/2`, `head`) with `fontTools`.
- Performs tight bounding box cropping (`ImageOps.invert` + `getbbox()`).
- Applies uniform aspect-ratio scaling to $256\times 256$.
- Injects minor augmentations: random 2–8px padding offsets, Gaussian blur ($\sigma \in [0.1, 0.4]$), and subtle threshold dithering.
- Multiprocessing pool with `tqdm` progress tracking.

**Arguments:**
| Flag | Default | Description |
|---|---|---|
| `--fonts_dir` | *Required* | Path to folder containing `.ttf` or `.otf` files. |
| `--output_dir` | *Required* | Target directory for patches and metadata. |
| `--image_size` | `256` | Square canvas dimension. |
| `--samples_per_template` | `5` | Augmentation variations per text template. |
| `--workers` | `auto` | Number of worker processes. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--max_fonts` | `None` | Optional limit on font files (useful for test runs). |

**Usage Example:**
```bash
python3 generate_dataset.py \
  --fonts_dir downloaded_fonts \
  --output_dir dataset \
  --image_size 256 \
  --samples_per_template 5 \
  --workers 4 \
  --seed 123456
```

**Generated Artifacts in `--output_dir`:**
- `dataset_manifest.csv`: Manifest with columns `image_path,font_family,font_id,style_name,style_id`.
- `font_map.json`: Bidirectional dictionary (`font_to_id` and `id_to_font`).
- `style_map.json`: Style ID to label mapping (`0: Regular`, `1: Bold`, `2: Italic`, `3: Bold-Italic`).
- Subdirectories per font family containing `<style>_<hash>.png`.

---

### 3. `dataset.py`
Provides reusable preprocessing routines and PyTorch `FontDataset` / `create_dataloaders` utilities.

**Key Components:**
- `preprocess_image(image_input, target_size=(256, 256), normalize_mode="inverted")`:
  - Accepts a file path, PIL Image, or NumPy array.
  - Converts to 1-channel Grayscale.
  - Uniformly scales and center-pads with white pixels into target dimensions.
  - Returns `(1, H, W)` `torch.FloatTensor` with strokes normalized to $1.0$ on $0.0$ background.
- `FontDataset`: Custom `torch.utils.data.Dataset` returning `(image_tensor, font_id, style_id)`.
- `create_dataloaders`: Stratifies dataset across `(font_id, style_id)` pairs for balanced validation splits.

**Standalone Test:**
```bash
# Test single image preprocessing and export visualization
python3 dataset.py --test_image dataset/Actor/Regular_4fc90ecaef11.png --output_debug debug_preprocessed.png
```

---

### 4. `train.py`
Trains `MultiHeadFontCNN` with multi-task loss balancing and exports the best checkpoint to ONNX format.

**Loss Formulation:**
$$\mathcal{L}_{\text{total}} = \text{CrossEntropy}(\text{font\_logits}, \text{font\_targets}, \text{label\_smoothing}=0.05) + 0.5 \times \text{CrossEntropy}(\text{style\_logits}, \text{style\_targets})$$

**Arguments:**
| Flag | Default | Description |
|---|---|---|
| `--manifest_csv` | `dataset/dataset_manifest.csv` | Path to dataset manifest CSV. |
| `--output_dir` | `runs/experiment1` | Destination directory for checkpoints and artifacts. |
| `--epochs` | `25` | Total training epochs. |
| `--batch_size` | `64` | Mini-batch size. |
| `--lr` | `1e-3` | Initial learning rate for `AdamW`. |
| `--weight_decay` | `1e-4` | Weight decay penalty for `AdamW`. |
| `--val_split` | `0.15` | Validation split fraction. |
| `--num_workers` | `4` | DataLoader background workers. |
| `--device` | `auto` | Execution device (`cuda`, `mps`, `cpu`). |

**Usage Example:**
```bash
python3 train.py \
  --manifest_csv dataset/dataset_manifest.csv \
  --output_dir runs/font_cnn_v1 \
  --epochs 25 \
  --batch_size 64 \
  --lr 1e-3 \
  --num_workers 4
```

**Generated Checkpoints in `--output_dir`:**
- `best_model.pth`: Checkpoint selected via highest Validation Joint Accuracy.
- `last_model.pth`: Final epoch weights.
- `model.onnx`: Exported ONNX model graph with dynamic batch size `(batch_size, 1, 256, 256)`.
- `training_metrics.json`: Per-epoch training & validation history (Top-1, Top-3, Joint Accuracy, and Losses).

---

### 5. `predict.py`
High-speed inference script supporting both PyTorch (`.pth`) and ONNX Runtime (`.onnx`) models.

**Arguments:**
| Flag | Default | Description |
|---|---|---|
| `--image` | *Required* | Path to an image file or directory of images. |
| `--model_path` | `best_model.pth` | Path to `.pth` or `.onnx` model checkpoint. |
| `--font_map` | `auto` | Path to `font_map.json`. |
| `--style_map` | `auto` | Path to `style_map.json`. |
| `--top_k` | `3` | Number of top font candidate predictions to display. |
| `--output_json` | `None` | (Optional) Filepath to save predictions in JSON format. |
| `--device` | `auto` | Target inference device (`cpu`, `cuda`, `mps`). |

**Usage Examples:**

#### Single Image Prediction (PyTorch)
```bash
python3 predict.py \
  --image sample.png \
  --model_path runs/font_cnn_v1/best_model.pth \
  --font_map dataset/font_map.json \
  --style_map dataset/style_map.json \
  --top_k 3
```

#### High-Throughput Inference with ONNX & JSON Export
```bash
python3 predict.py \
  --image test_images/ \
  --model_path runs/font_cnn_v1/model.onnx \
  --font_map dataset/font_map.json \
  --style_map dataset/style_map.json \
  --top_k 5 \
  --output_json results.json
```

---

## 📊 Sample Inference Output

```text
────────────────────────────────────────────────────────────────────────
 🔍 Target Image : test_images/sample_text.png
────────────────────────────────────────────────────────────────────────
  ✦ Primary Font   : Roboto (84.2% confidence)
  ✦ Inferred Style : Bold (91.6% confidence)
────────────────────────────────────────────────────────────────────────
  TOP FONT CANDIDATES:
  Rank  | Font Family                | ID   | Confidence | Distribution
  ────────────────────────────────────────────────────────────────────
  #1    | Roboto                     | 142  |  84.2%   | [████████████░░]
  #2    | Open Sans                  | 88   |   9.1%   | [█░░░░░░░░░░░░░]
  #3    | Lato                       | 53   |   3.4%   | [░░░░░░░░░░░░░░]

  STYLE PROBABILITY DISTRIBUTION:
     Regular      :   6.2%  [█░░░░░░░░░░░░░]
   ★ Bold         :  91.6%  [█████████████░]
     Italic       :   1.4%  [░░░░░░░░░░░░░░]
     Bold-Italic  :   0.8%  [░░░░░░░░░░░░░░]
────────────────────────────────────────────────────────────────────────
```

---

## ⚡ Complete End-to-End Execution Recipe

```bash
# 1. Download font catalog
python3 download_google_fonts.py --output_dir downloaded_fonts --workers 8

# 2. Synthesize dataset patches
python3 generate_dataset.py --fonts_dir downloaded_fonts --output_dir dataset --samples_per_template 5 --workers 4

# 3. Train multi-head classifier & export ONNX
python3 train.py --manifest_csv dataset/dataset_manifest.csv --output_dir runs/experiment1 --epochs 25 --batch_size 64

# 4. Run inference
python3 predict.py --image dataset/ABeeZee/Regular_02aa13a2ddfa.png --model_path runs/experiment1/model.onnx --top_k 3
```
