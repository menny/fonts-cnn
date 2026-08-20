# Multi-Head Font & Style Recognition with Deep CNNs

A complete, production-ready deep learning pipeline for synthetic dataset generation, local & system font discovery, model training, and dual-head font recognition (Font Family & Discrete Style classification) using PyTorch and ONNX Runtime.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)]()
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-blueviolet.svg)]()

---

## 📑 Table of Contents
1. [Overview & System Architecture](#-overview--system-architecture)
2. [Project Structure](#-project-structure)
3. [Installation & Requirements](#-installation--requirements)
4. [Scripts & Workflow](#-scripts--workflow)
   - [1. `download_google_fonts.py` (Font Downloader)](#1-download_google_fontspy)
   - [2. `generate_dataset_v2.py` (Enhanced Dataset Generator)](#2-generate_dataset_v2py-recommended)
   - [3. `dataset.py` (PyTorch Dataset & Preprocessing Pipeline)](#3-datasetpy)
   - [4. `train.py` (Multi-Head CNN Trainer & ONNX Exporter)](#4-trainpy)
   - [5. `predict.py` (Standalone PyTorch & ONNX Inference)](#5-predictpy)
5. [Local & System Font Ingestion](#-local--system-font-ingestion)
6. [Model Architecture Details](#-model-architecture-details)
7. [Inference & Sample Outputs](#-inference--sample-outputs)
8. [License](#-license)

---

## 🏛 Overview & System Architecture

This repository provides an end-to-end framework for identifying font families and styles from rendered text and number patches. The architecture uses a multi-task convolutional neural network (`MultiHeadFontCNN`) with shared low/mid-level feature extractors and separate classification heads:

```text
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
Fonts are parsed from OpenType/TrueType metadata tables (`name`, `OS/2`, `head`) and classified into 4 canonical style categories:
- `0`: **Regular** (Book, Roman, Medium, Normal)
- `1`: **Bold** (Black, Heavy, SemiBold, ExtraBold, Weight $\ge 600$)
- `2`: **Italic** (Oblique, Slanted, Italic)
- `3`: **Bold-Italic** (Bold + Italic combined)

---

## 📂 Project Structure

```text
.
├── download_google_fonts.py  # Asynchronous Google Fonts catalog downloader
├── generate_dataset_v2.py    # Enhanced generator (matrices, dates, local/system fonts)
├── generate_dataset.py       # Legacy dataset generator
├── dataset.py                # PyTorch Dataset, DataLoaders, & robust preprocessing
├── train.py                  # Multi-task CNN training loop and ONNX exporter
├── predict.py                # CLI inference runner supporting .pth & .onnx
├── popular_200_fonts.json    # Catalog filter for top 200 popular Google Fonts
├── downloaded_fonts/         # Raw font repository (.ttf, .otf) [gitignored]
├── runs/                     # Checkpoints, ONNX models & metrics [gitignored]
├── LICENSE                   # Apache 2.0 License
└── README.md                 # Project technical documentation
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
# Download all fonts to local directory with 8 parallel worker threads
python3 download_google_fonts.py --output_dir downloaded_fonts --workers 8
```

---

### 2. `generate_dataset_v2.py` (Recommended)
Generates high-quality synthetic image patches across diverse text templates (pangrams, full multi-line digit matrices `0-9`, realistic expiration dates, timestamps, lot numbers, and financial codes).

**Key Features:**
- **Realistic Digit Matrices**: Generates complete multi-line digit grids (`0 1 2 3 4\n5 6 7 8 9`) so the CNN sees full sets of numeric glyphs for each font.
- **Scale-Clamped Rendering**: Prevents short numbers and single dates from blowing up into oversized $180\text{px}$ glyphs, keeping glyph heights consistent with real-world photos ($28\text{pt} - 42\text{pt}$).
- **Multiple Font Directories**: Accepts multiple search paths via `--fonts_dirs`.
- **System Font Auto-Discovery**: Automatically discovers OS fonts across Linux, macOS, and Windows via `--include_system_fonts`.
- **Catalog Filtering**: Easily filter generation to the top 200 popular fonts via `--popular_json popular_200_fonts.json`.

**Arguments:**
| Flag | Default | Description |
|---|---|---|
| `--fonts_dirs`, `--fonts_dir` | `["downloaded_fonts"]` | One or more folders or file paths to scan for fonts. |
| `--include_system_fonts` | `False` | Automatically discover and include OS system font directories. |
| `--popular_json` | `popular_200_fonts.json` | JSON list of target font family names to generate. |
| `--output_dir` | `dataset_top200_v2` | Target destination for image patches and metadata. |
| `--image_size` | `256` | Square canvas dimension. |
| `--samples_per_template` | `5` | Augmentation variations per text template. |
| `--workers` | `auto` | Number of worker processes. |
| `--seed` | `123456` | Random seed for reproducibility. |

**Usage Example:**
```bash
python3 generate_dataset_v2.py \
  --fonts_dirs downloaded_fonts \
  --popular_json popular_200_fonts.json \
  --output_dir dataset_top200_v2 \
  --samples_per_template 5 \
  --workers 4
```

---

### 3. `dataset.py`
Provides PyTorch `FontDataset`, `create_dataloaders` utilities, and the end-to-end `preprocess_image` pipeline.

**Robust Preprocessing (`preprocess_image`):**
- **Any Resolution & Aspect Ratio**: Uniformly scales preserving aspect ratio using high-quality Lanczos resampling and center-pads onto the $256\times 256$ white canvas.
- **Color Depth & Transparency**: Converts RGB, RGBA, CMYK, and 16-bit images to 1-channel grayscale; alpha transparency is automatically composited onto a pure white background.
- **Auto-Cropping Surrounding Margins**: Automatically detects text bounding boxes and crops excess white margins.
- **Dark Mode Detection**: Inverts dark backgrounds with light text into standard black text on white background.
- **Normalized Output**: Returns a `(1, 256, 256)` float tensor with strokes normalized to $1.0$ on $0.0$ background.

**Standalone Preprocessing Test:**
```bash
python3 dataset.py --test_image sample.png --output_debug debug_preprocessed.png
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
  --manifest_csv dataset_top200_v2/dataset_manifest.csv \
  --output_dir runs/top200_v2_12ep \
  --epochs 12 \
  --batch_size 64 \
  --lr 1e-3 \
  --num_workers 4
```

**Generated Checkpoints in `--output_dir`:**
- `best_model.pth`: Saved whenever Validation Joint Accuracy improves.
- `last_model.pth`: Final epoch weights.
- `model.onnx`: Exported ONNX graph with dynamic batch size `(batch_size, 1, 256, 256)`.
- `training_metrics.json`: Full epoch history (Top-1, Top-3, Joint Accuracy, and Losses).

---

### 5. `predict.py`
Standalone inference script supporting both PyTorch (`.pth`) and ONNX Runtime (`.onnx`) models.

**Arguments:**
| Flag | Default | Description |
|---|---|---|
| `--image` | *Required* | Path to an image file or directory of images. |
| `--model_path` | `best_model.pth` | Path to `.pth` or `.onnx` model checkpoint. |
| `--font_map` | `auto` | Path to `font_map.json`. |
| `--style_map` | `auto` | Path to `style_map.json`. |
| `--top_k` | `3` | Number of top font candidate predictions to display. |
| `--output_json` | `None` | Optional path to export predictions to JSON. |
| `--device` | `auto` | Target inference device (`cpu`, `cuda`, `mps`). |

**Usage Examples:**

#### Single Image Prediction (PyTorch `.pth`)
```bash
python3 predict.py \
  --image sample.png \
  --model_path runs/top200_v2_12ep/best_model.pth \
  --font_map dataset_top200_v2/font_map.json \
  --style_map dataset_top200_v2/style_map.json \
  --top_k 5
```

#### High-Throughput Batch Inference with ONNX Runtime & JSON Export
```bash
python3 predict.py \
  --image test_images/ \
  --model_path runs/top200_v2_12ep/model.onnx \
  --font_map dataset_top200_v2/font_map.json \
  --style_map dataset_top200_v2/style_map.json \
  --top_k 5 \
  --output_json results.json
```

---

## 💻 Local & System Font Ingestion

You can include fonts from your local machine (such as *Times New Roman*, *Arial*, *Calibri*, *Helvetica*, or custom corporate fonts):

### Ingesting Specific Local Directories
```bash
python3 generate_dataset_v2.py \
  --fonts_dirs downloaded_fonts /usr/share/fonts/truetype ~/.local/share/fonts \
  --output_dir dataset_custom
```

### Auto-Discovering Standard OS System Fonts
```bash
python3 generate_dataset_v2.py \
  --include_system_fonts \
  --output_dir dataset_with_system_fonts
```
Standard OS directories scanned automatically:
- **Linux**: `/usr/share/fonts`, `/usr/local/share/fonts`, `~/.fonts`, `~/.local/share/fonts`
- **macOS**: `/Library/Fonts`, `/System/Library/Fonts`, `~/Library/Fonts`
- **Windows**: `C:\Windows\Fonts`, `%LOCALAPPDATA%\Microsoft\Windows\Fonts`

---

## 📊 Inference & Sample Outputs

```text
────────────────────────────────────────────────────────────────────────
 🔍 Target Image : sample_receipt_date.png
────────────────────────────────────────────────────────────────────────
  ✦ Primary Font   : Alumni Sans SC (48.1% confidence)
  ✦ Inferred Style : Regular (100.0% confidence)
────────────────────────────────────────────────────────────────────────
  TOP FONT CANDIDATES:
  Rank  | Font Family                | ID   | Confidence | Distribution
  ────────────────────────────────────────────────────────────────────
  #1    | Alumni Sans SC             | 60   |  48.1%   | [███████░░░░░░░]
  #2    | Alumni Sans Pinstripe      | 59   |  19.6%   | [███░░░░░░░░░░░]
  #3    | Alumni Sans                | 56   |   3.8%   | [█░░░░░░░░░░░░░]
  #4    | Abril Fatface              | 6    |   3.4%   | [░░░░░░░░░░░░░░]
  #5    | Anybody                    | 98   |   3.4%   | [░░░░░░░░░░░░░░]

  STYLE PROBABILITY DISTRIBUTION:
   ★ Regular      : 100.0%  [██████████████]
     Bold         :   0.0%  [░░░░░░░░░░░░░░]
     Italic       :   0.0%  [░░░░░░░░░░░░░░]
     Bold-Italic  :   0.0%  [░░░░░░░░░░░░░░]
────────────────────────────────────────────────────────────────────────
```

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).  
Copyright © 2026 Menny Even Danan.
