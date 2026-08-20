#!/usr/bin/env python3
"""
train.py

PyTorch training script for MultiHeadFontCNN:
A multi-head convolutional neural network for simultaneous Font Family and Style classification.

Features:
1. MultiHeadFontCNN Architecture:
   - 5-stage Conv2d + BatchNorm + ReLU + MaxPool2d backbone with Dropout2d(0.15) on later stages.
   - AdaptiveAvgPool2d((4, 4)) feature aggregation + Dense embedding (512-dim).
   - Dual output heads: font_head (num_fonts) and style_head (4 discrete styles).
2. Multi-Task Balancing & Loss:
   - Font Loss: CrossEntropy with label_smoothing=0.05.
   - Style Loss: CrossEntropy.
   - Total Loss: L_total = L_font + 0.5 * L_style.
3. Training & Validation Tracking:
   - Tracks Font Top-1, Font Top-3, Style Top-1, and Joint Accuracy.
   - Model checkpointing: best_model.pth (selected via highest Validation Joint Accuracy).
4. Export:
   - Exports trained model to model.onnx with dynamic batch size.
   - Saves complete epoch history to training_metrics.json.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import create_dataloaders

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MultiHeadFontCNN(nn.Module):
    """
    Multi-Head Convolutional Neural Network for simultaneous Font Family and Style classification.
    """

    def __init__(self, num_fonts: int, num_styles: int = 4) -> None:
        super().__init__()
        self.num_fonts = num_fonts
        self.num_styles = num_styles

        # 5-Stage Convolutional Backbone
        # Input: (B, 1, 256, 256)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (B, 32, 128, 128)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (B, 64, 64, 64)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (B, 128, 32, 32)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (B, 256, 16, 16)
            nn.Dropout2d(0.15),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (B, 512, 8, 8)
            nn.Dropout2d(0.15),
        )

        # Feature Aggregation
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # -> (B, 512, 4, 4)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        # Dual Task Heads
        self.font_head = nn.Linear(512, num_fonts)
        self.style_head = nn.Linear(512, num_styles)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            font_logits: (B, num_fonts)
            style_logits: (B, num_styles)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        embed = self.embedding(x)
        font_logits = self.font_head(embed)
        style_logits = self.style_head(embed)
        return font_logits, style_logits


def calculate_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Calculates top-k accuracy as a percentage float [0, 100]."""
    k = min(k, logits.size(1))
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        correct_total = correct.any(dim=1).float().sum().item()
        return (correct_total / targets.size(0)) * 100.0


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    total_font_loss = 0.0
    total_style_loss = 0.0

    total_samples = 0
    font_correct_top1 = 0
    style_correct_top1 = 0
    joint_correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, font_targets, style_targets in pbar:
        images = images.to(device, non_blocking=True)
        font_targets = font_targets.to(device, non_blocking=True)
        style_targets = style_targets.to(device, non_blocking=True)

        batch_size = images.size(0)
        optimizer.zero_grad()

        font_logits, style_logits = model(images)

        # Multi-task loss with font label smoothing
        loss_font = F.cross_entropy(font_logits, font_targets, label_smoothing=0.05)
        loss_style = F.cross_entropy(style_logits, style_targets)
        loss_total_batch = loss_font + 0.5 * loss_style

        loss_total_batch.backward()
        optimizer.step()

        # Metrics
        total_loss += loss_total_batch.item() * batch_size
        total_font_loss += loss_font.item() * batch_size
        total_style_loss += loss_style.item() * batch_size
        total_samples += batch_size

        with torch.no_grad():
            font_pred = font_logits.argmax(dim=1)
            style_pred = style_logits.argmax(dim=1)

            font_match = font_pred.eq(font_targets)
            style_match = style_pred.eq(style_targets)

            font_correct_top1 += font_match.sum().item()
            style_correct_top1 += style_match.sum().item()
            joint_correct += (font_match & style_match).sum().item()

        pbar.set_postfix({
            "Loss": f"{loss_total_batch.item():.4f}",
            "FontAcc": f"{(font_correct_top1 / total_samples) * 100:.1f}%",
            "JointAcc": f"{(joint_correct / total_samples) * 100:.1f}%",
        })

    return {
        "train_loss": total_loss / total_samples,
        "train_font_loss": total_font_loss / total_samples,
        "train_style_loss": total_style_loss / total_samples,
        "train_font_acc_top1": (font_correct_top1 / total_samples) * 100.0,
        "train_style_acc_top1": (style_correct_top1 / total_samples) * 100.0,
        "train_joint_acc": (joint_correct / total_samples) * 100.0,
    }


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Evaluates model on validation dataset."""
    model.eval()
    total_loss = 0.0
    total_font_loss = 0.0
    total_style_loss = 0.0

    total_samples = 0
    font_correct_top1 = 0
    font_correct_top3 = 0
    style_correct_top1 = 0
    joint_correct = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False)
        for images, font_targets, style_targets in pbar:
            images = images.to(device, non_blocking=True)
            font_targets = font_targets.to(device, non_blocking=True)
            style_targets = style_targets.to(device, non_blocking=True)

            batch_size = images.size(0)
            font_logits, style_logits = model(images)

            loss_font = F.cross_entropy(font_logits, font_targets, label_smoothing=0.05)
            loss_style = F.cross_entropy(style_logits, style_targets)
            loss_total_batch = loss_font + 0.5 * loss_style

            total_loss += loss_total_batch.item() * batch_size
            total_font_loss += loss_font.item() * batch_size
            total_style_loss += loss_style.item() * batch_size
            total_samples += batch_size

            # Top-1 Predictions
            font_pred = font_logits.argmax(dim=1)
            style_pred = style_logits.argmax(dim=1)

            font_match = font_pred.eq(font_targets)
            style_match = style_pred.eq(style_targets)

            font_correct_top1 += font_match.sum().item()
            style_correct_top1 += style_match.sum().item()
            joint_correct += (font_match & style_match).sum().item()

            # Top-3 Font Predictions
            k = min(3, font_logits.size(1))
            _, font_top3_pred = font_logits.topk(k, dim=1, largest=True, sorted=True)
            font_correct_top3 += font_top3_pred.eq(font_targets.view(-1, 1).expand_as(font_top3_pred)).any(dim=1).sum().item()

    return {
        "val_loss": total_loss / total_samples,
        "val_font_loss": total_font_loss / total_samples,
        "val_style_loss": total_style_loss / total_samples,
        "val_font_acc_top1": (font_correct_top1 / total_samples) * 100.0,
        "val_font_acc_top3": (font_correct_top3 / total_samples) * 100.0,
        "val_style_acc_top1": (style_correct_top1 / total_samples) * 100.0,
        "val_joint_acc": (joint_correct / total_samples) * 100.0,
    }


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int = 256,
) -> None:
    """Exports model to ONNX with dynamic batch size."""
    model.eval()
    dummy_input = torch.randn(1, 1, image_size, image_size, device=next(model.parameters()).device)

    # Export with dynamic batch size
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["font_logits", "style_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "font_logits": {0: "batch_size"},
            "style_logits": {0: "batch_size"},
        },
    )
    logging.info(f"ONNX model successfully exported to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MultiHeadFontCNN for simultaneous Font Family and Style recognition."
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="dataset/dataset_manifest.csv",
        help="Path to dataset_manifest.csv.",
    )
    parser.add_argument(
        "--font_map",
        type=str,
        default=None,
        help="Path to font_map.json (defaults to same folder as manifest_csv).",
    )
    parser.add_argument(
        "--style_map",
        type=str,
        default=None,
        help="Path to style_map.json (defaults to same folder as manifest_csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/experiment1",
        help="Output directory for checkpoints, metrics, and ONNX export.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Total training epochs (default: 25).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of data reserved for validation (default: 0.15).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda', 'mps', 'cpu'). Auto-detected if None.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    manifest_path = Path(args.manifest_csv).resolve()
    if not manifest_path.exists():
        logging.error(f"Manifest CSV not found: {manifest_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    logging.info(f"Using execution device: {device}")

    # Load mappings
    dataset_dir = manifest_path.parent
    font_map_path = Path(args.font_map) if args.font_map else dataset_dir / "font_map.json"
    style_map_path = Path(args.style_map) if args.style_map else dataset_dir / "style_map.json"

    font_to_id = {}
    if font_map_path.exists():
        with open(font_map_path, "r", encoding="utf-8") as f:
            fmap = json.load(f)
            font_to_id = fmap.get("font_to_id", fmap)

    # Build DataLoaders
    logging.info(f"Loading dataset from: {manifest_path}")
    train_loader, val_loader, full_dataset = create_dataloaders(
        manifest_csv=manifest_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        normalize_mode="inverted",
    )

    num_fonts = full_dataset.num_fonts
    num_styles = full_dataset.num_styles
    logging.info(f"Dataset stats: {len(full_dataset)} total samples, {num_fonts} fonts, {num_styles} styles")

    # Initialize Model, Optimizer, Scheduler
    model = MultiHeadFontCNN(num_fonts=num_fonts, num_styles=num_styles).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_joint_acc = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    logging.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "duration_sec": epoch_duration,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_record)

        logging.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_duration:.1f}s] "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Font Top-1: {val_metrics['val_font_acc_top1']:.2f}% | "
            f"Val Font Top-3: {val_metrics['val_font_acc_top3']:.2f}% | "
            f"Val Style Top-1: {val_metrics['val_style_acc_top1']:.2f}% | "
            f"Val Joint Acc: {val_metrics['val_joint_acc']:.2f}%"
        )

        # Checkpoint if best Joint Accuracy
        if val_metrics["val_joint_acc"] > best_val_joint_acc:
            best_val_joint_acc = val_metrics["val_joint_acc"]
            best_epoch = epoch

            best_checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_fonts": num_fonts,
                    "num_styles": num_styles,
                    "val_metrics": val_metrics,
                    "font_to_id": font_to_id,
                },
                best_checkpoint_path,
            )
            logging.info(f"--> Saved new BEST checkpoint (Joint Acc: {best_val_joint_acc:.2f}%) to {best_checkpoint_path.name}")

        # Save last checkpoint
        last_checkpoint_path = output_dir / "last_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_fonts": num_fonts,
                "num_styles": num_styles,
                "val_metrics": val_metrics,
                "font_to_id": font_to_id,
            },
            last_checkpoint_path,
        )

    total_training_time = time.time() - start_time
    logging.info(f"Training finished in {total_training_time / 60:.2f} mins. Best Epoch: {best_epoch} with Joint Acc: {best_val_joint_acc:.2f}%")

    # Save training history JSON
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_joint_acc": best_val_joint_acc,
                "total_training_time_sec": total_training_time,
                "history": history,
            },
            f,
            indent=2,
        )
    logging.info(f"Saved training metrics history to: {metrics_path}")

    # Export best model to ONNX
    logging.info("Exporting best checkpoint to ONNX...")
    best_checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(model, onnx_path, image_size=256)

    print("\n" + "=" * 60)
    print("TRAINING & EXPORT COMPLETED")
    print("=" * 60)
    print(f"Output Directory       : {output_dir}")
    print(f"Best Epoch             : {best_epoch}/{args.epochs}")
    print(f"Best Val Joint Acc     : {best_val_joint_acc:.2f}%")
    print(f"Best Model Checkpoint  : {output_dir / 'best_model.pth'}")
    print(f"ONNX Model             : {onnx_path}")
    print(f"Training Metrics JSON  : {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
