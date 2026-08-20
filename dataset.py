#!/usr/bin/env python3
"""
dataset.py

PyTorch Dataset class and reusable image preprocessing functions for multi-head font and style recognition.

Features:
1. preprocess_image:
   - Converts file path, PIL Image, or NumPy array into 1-channel grayscale.
   - Resizes preserving aspect ratio with pure white letterbox padding to target_size (default: 256x256).
   - Converts to PyTorch FloatTensor with shape (1, H, W) normalized to [0.0, 1.0] (inverted text strokes)
     or standard normalized with mean=[0.5], std=[0.5].
2. FontDataset:
   - Inherits from torch.utils.data.Dataset.
   - Parses manifest CSV with image_path, font_family, font_id, style_name, style_id.
   - Returns (image_tensor, font_id, style_id) where labels are torch.long.
3. create_dataloaders:
   - Stratifies / splits dataset into train and validation DataLoaders.
4. Standalone CLI test:
   - Verifies preprocessing on a sample image and saves `debug_preprocessed.png`.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    normalize_mode: str = "inverted",
    auto_crop: bool = True,
) -> torch.Tensor:
    """
    Preprocesses an input image into a standardized 1-channel tensor with white letterbox padding.

    Handles arbitrary image resolutions, aspect ratios, color depths (RGB, RGBA, 16-bit, etc.),
    alpha transparency, and auto-detects dark backgrounds to invert to black-on-white text.

    Args:
        image_input: File path (str/Path), PIL Image, or NumPy array.
        target_size: Target (width, height) tuple (default: (256, 256)).
        normalize_mode:
            - 'inverted' (default): [0.0, 1.0] where text strokes are 1.0 and white background is 0.0.
            - 'standard': Normalized with mean=0.5, std=0.5 -> [-1.0, 1.0].
            - 'unit': [0.0, 1.0] standard (white=1.0, black=0.0).
        auto_crop: If True, crops excess surrounding margins to scale text glyphs properly.

    Returns:
        torch.FloatTensor of shape (1, target_height, target_width).
    """
    # Load or convert to PIL Image
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 1. Handle transparency (RGBA / LA / P-palette transparency composited onto white background)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        white_bg = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, alpha_img).convert("RGB")

    # 2. Convert to 1-channel Grayscale ('L' mode)
    img = img.convert("L")

    # 3. Auto-detect dark background (e.g. white text on black background) and invert
    arr_check = np.array(img)
    corners = [
        int(arr_check[0, 0]),
        int(arr_check[0, -1]),
        int(arr_check[-1, 0]),
        int(arr_check[-1, -1]),
    ]
    if (sum(corners) / 4.0) < 128.0:
        img = ImageOps.invert(img)

    # 4. Auto-crop excess surrounding whitespace margins
    if auto_crop:
        inverted_for_bbox = ImageOps.invert(img)
        bbox = inverted_for_bbox.getbbox()
        if bbox:
            w, h = img.size
            crop_margin = 6
            crop_box = (
                max(0, bbox[0] - crop_margin),
                max(0, bbox[1] - crop_margin),
                min(w, bbox[2] + crop_margin),
                min(h, bbox[3] + crop_margin),
            )
            img = img.crop(crop_box)

    orig_w, orig_h = img.size
    target_w, target_h = target_size

    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img.size}")

    # 5. Aspect-ratio preserving uniform scaling
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6. Pure white (255) letterbox canvas
    canvas = Image.new("L", (target_w, target_h), 255)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    # Convert to float32 NumPy array
    arr = np.array(canvas, dtype=np.float32)

    # Apply Normalization
    if normalize_mode == "inverted":
        # White background (255) -> 0.0, Black text (0) -> 1.0
        normalized = 1.0 - (arr / 255.0)
    elif normalize_mode == "standard":
        # Standard mean=0.5, std=0.5 -> [-1.0, 1.0]
        normalized = (arr / 255.0 - 0.5) / 0.5
    elif normalize_mode == "unit":
        # Standard [0.0, 1.0] where 255 -> 1.0, 0 -> 0.0
        normalized = arr / 255.0
    else:
        raise ValueError(f"Unknown normalize_mode '{normalize_mode}'. Expected 'inverted', 'standard', or 'unit'.")

    # Add channel dimension: (1, target_h, target_w)
    tensor = torch.from_numpy(normalized).unsqueeze(0).to(dtype=torch.float32)
    return tensor


def tensor_to_image(tensor: torch.Tensor, normalize_mode: str = "inverted") -> Image.Image:
    """
    Converts a preprocessed (1, H, W) PyTorch tensor back into a PIL Grayscale Image for visualization.
    """
    t = tensor.detach().cpu().squeeze(0).numpy()
    if normalize_mode == "inverted":
        arr = np.clip((1.0 - t) * 255.0, 0, 255).astype(np.uint8)
    elif normalize_mode == "standard":
        arr = np.clip((t * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


class FontDataset(Dataset):
    """
    PyTorch Dataset for multi-head font family and font style classification.
    """

    def __init__(
        self,
        manifest_csv: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        target_size: Tuple[int, int] = (256, 256),
        normalize_mode: str = "inverted",
    ) -> None:
        """
        Args:
            manifest_csv: Path to dataset_manifest.csv containing image_path, font_family, font_id, style_name, style_id.
            root_dir: Base directory for image relative paths. Defaults to manifest_csv parent directory.
            transform: Optional callable transform on image. If None, uses preprocess_image.
            target_size: Target square canvas dimension (default: (256, 256)).
            normalize_mode: Normalization mode ('inverted', 'standard', or 'unit').
        """
        self.manifest_path = Path(manifest_csv)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV file not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        required_cols = {"image_path", "font_id", "style_id"}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Manifest CSV is missing required columns: {missing_cols}")

        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.transform = transform
        self.target_size = target_size
        self.normalize_mode = normalize_mode

        # Pre-extract numpy arrays for fast indexing in workers
        self.image_paths = self.df["image_path"].values
        self.font_ids = self.df["font_id"].values.astype(np.int64)
        self.style_ids = self.df["style_id"].values.astype(np.int64)

        # Cache metadata properties
        self.num_fonts = int(self.font_ids.max()) + 1 if len(self.font_ids) > 0 else 0
        self.num_styles = 4  # Discrete: 0=Regular, 1=Bold, 2=Italic, 3=Bold-Italic

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_tensor: torch.FloatTensor of shape (1, H, W)
            font_id: torch.LongTensor scalar
            style_id: torch.LongTensor scalar
        """
        rel_path = self.image_paths[idx]
        full_path = self.root_dir / rel_path

        if self.transform is not None:
            img = Image.open(full_path).convert("L")
            image_tensor = self.transform(img)
        else:
            image_tensor = preprocess_image(
                full_path,
                target_size=self.target_size,
                normalize_mode=self.normalize_mode,
            )

        font_label = torch.tensor(self.font_ids[idx], dtype=torch.long)
        style_label = torch.tensor(self.style_ids[idx], dtype=torch.long)

        return image_tensor, font_label, style_label


def create_dataloaders(
    manifest_csv: Union[str, Path],
    batch_size: int = 64,
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    root_dir: Optional[Union[str, Path]] = None,
    target_size: Tuple[int, int] = (256, 256),
    normalize_mode: str = "inverted",
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, FontDataset]:
    """
    Creates stratified or balanced train and validation DataLoaders from a manifest CSV.

    Args:
        manifest_csv: Path to dataset_manifest.csv.
        batch_size: Mini-batch size.
        val_split: Validation split fraction (e.g. 0.15 for 15% validation).
        num_workers: Number of background data-loading subprocesses.
        seed: Random seed for split reproducibility.
        root_dir: Base directory for image relative paths.
        target_size: Canvas size for patches.
        normalize_mode: Normalization strategy ('inverted', 'standard', or 'unit').
        pin_memory: Whether to pin memory for CUDA tensor transfers.

    Returns:
        (train_loader, val_loader, full_dataset)
    """
    dataset = FontDataset(
        manifest_csv=manifest_csv,
        root_dir=root_dir,
        target_size=target_size,
        normalize_mode=normalize_mode,
    )

    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("Dataset is empty.")

    # Stratified or randomized train/validation index split
    df = dataset.df
    rng = np.random.RandomState(seed)

    # Group by (font_id, style_id) to stratify validation set
    train_indices: List[int] = []
    val_indices: List[int] = []

    # Stratified split per (font_id, style_id) group
    groups = df.groupby(["font_id", "style_id"]).indices
    for _, group_idxs in groups.items():
        shuffled = rng.permutation(group_idxs)
        num_val = int(np.ceil(len(shuffled) * val_split))
        val_indices.extend(shuffled[:num_val].tolist())
        train_indices.extend(shuffled[num_val:].tolist())

    # Fallback to random split if stratification yielded empty train set
    if not train_indices:
        shuffled = rng.permutation(total_samples)
        num_val = int(total_samples * val_split)
        val_indices = shuffled[:num_val].tolist()
        train_indices = shuffled[num_val:].tolist()

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(len(train_subset) > batch_size),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logging.info(
        f"Created DataLoaders: Train samples={len(train_subset)}, Val samples={len(val_subset)}, Batch size={batch_size}"
    )

    return train_loader, val_loader, dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset pipeline and standalone preprocessor test for font recognition."
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default=None,
        help="Path to sample image for testing preprocessing pipeline.",
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=None,
        help="Path to dataset_manifest.csv for testing DataLoader construction.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target square canvas size (default: 256).",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="inverted",
        choices=["inverted", "standard", "unit"],
        help="Normalization strategy (default: inverted).",
    )
    parser.add_argument(
        "--output_debug",
        type=str,
        default="debug_preprocessed.png",
        help="Output path for debug preprocessed image (default: debug_preprocessed.png).",
    )

    args = parser.parse_args()

    # Test 1: Single image preprocessing test
    if args.test_image:
        test_path = Path(args.test_image)
        if not test_path.exists():
            logging.error(f"Test image not found: {test_path}")
            return

        logging.info(f"Testing preprocessing on: {test_path}")
        tensor = preprocess_image(
            test_path,
            target_size=(args.target_size, args.target_size),
            normalize_mode=args.normalize_mode,
        )

        logging.info(f"Output Tensor Shape : {tuple(tensor.shape)}")
        logging.info(f"Data Type           : {tensor.dtype}")
        logging.info(f"Min Value           : {tensor.min().item():.4f}")
        logging.info(f"Max Value           : {tensor.max().item():.4f}")
        logging.info(f"Mean Value          : {tensor.mean().item():.4f}")
        logging.info(f"Std Deviation       : {tensor.std().item():.4f}")

        # Convert back and save debug image
        debug_img = tensor_to_image(tensor, normalize_mode=args.normalize_mode)
        debug_img.save(args.output_debug)
        logging.info(f"Saved letterboxed visualization to: {Path(args.output_debug).resolve()}")

    # Test 2: DataLoader construction test
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv)
        if manifest_path.exists():
            logging.info(f"Testing DataLoaders with manifest: {manifest_path}")
            train_loader, val_loader, dataset = create_dataloaders(
                manifest_csv=manifest_path,
                batch_size=16,
                val_split=0.2,
                num_workers=2,
                normalize_mode=args.normalize_mode,
            )
            for images, font_labels, style_labels in train_loader:
                logging.info(f"Sample Batch - Images shape: {images.shape}, Font labels shape: {font_labels.shape}, Style labels shape: {style_labels.shape}")
                break
        else:
            logging.warning(f"Manifest CSV not found: {manifest_path}")

    if not args.test_image and not args.manifest_csv:
        # Create a synthetic sample image to verify end-to-end
        logging.info("No --test_image specified. Generating a synthetic test canvas...")
        dummy_img = Image.new("RGB", (320, 120), (255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(dummy_img)
        draw.text((20, 40), "Font CNN Test", fill=(0, 0, 0))
        dummy_path = "sample_test_input.png"
        dummy_img.save(dummy_path)

        tensor = preprocess_image(dummy_path, target_size=(args.target_size, args.target_size), normalize_mode=args.normalize_mode)
        logging.info(f"Output Tensor Shape : {tuple(tensor.shape)}")
        logging.info(f"Min Value           : {tensor.min().item():.4f}")
        logging.info(f"Max Value           : {tensor.max().item():.4f}")
        logging.info(f"Mean Value          : {tensor.mean().item():.4f}")

        debug_img = tensor_to_image(tensor, normalize_mode=args.normalize_mode)
        debug_img.save(args.output_debug)
        logging.info(f"Saved debug output to: {args.output_debug}")

        if os.path.exists(dummy_path):
            os.remove(dummy_path)


if __name__ == "__main__":
    main()
