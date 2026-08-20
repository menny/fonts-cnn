#!/usr/bin/env python3
"""
predict.py

Standalone inference script for multi-head font recognition and style classification.
Supports both PyTorch (.pth) and ONNX Runtime (.onnx) backends.

Features:
1. Input: Single image (.png, .jpg) or directory of images.
2. Backend:
   - PyTorch model checkpoint (.pth)
   - ONNX Runtime session (.onnx)
3. Output:
   - Formatted terminal tables with confidence scores and ASCII/Unicode progress bars.
   - Top-K font candidate ranking.
   - 4-class discrete style probability distribution (Regular, Bold, Italic, Bold-Italic).
   - Optional JSON export (--output_json).
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from dataset import preprocess_image

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

STYLE_NAMES = ["Regular", "Bold", "Italic", "Bold-Italic"]


def softmax(x: np.ndarray) -> np.ndarray:
    """Calculates numerically stable softmax across the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def make_progress_bar(percent: float, length: int = 16) -> str:
    """Creates a visual progress bar string for confidence display."""
    filled_len = int(round(length * percent / 100.0))
    filled_len = max(0, min(length, filled_len))
    bar = "█" * filled_len + "░" * (length - filled_len)
    return bar


class ModelPredictor:
    """Unified inference wrapper for PyTorch and ONNX models."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        font_map: Optional[Dict[str, Any]] = None,
        style_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.ext = self.model_path.suffix.lower()
        self.device_str = device or ("cuda" if self._has_cuda() else "cpu")
        self.font_map = font_map or {}
        self.style_map = style_map or {}

        # Resolve ID-to-Font mapping
        self.id_to_font: Dict[int, str] = {}
        if "id_to_font" in self.font_map:
            self.id_to_font = {int(k): v for k, v in self.font_map["id_to_font"].items()}
        elif "font_to_id" in self.font_map:
            self.id_to_font = {int(v): k for k, v in self.font_map["font_to_id"].items()}
        elif self.font_map:
            self.id_to_font = {int(k): v for k, v in self.font_map.items() if str(k).isdigit()}

        # Resolve ID-to-Style mapping
        self.id_to_style: Dict[int, str] = {0: "Regular", 1: "Bold", 2: "Italic", 3: "Bold-Italic"}
        if "id_to_style" in self.style_map:
            self.id_to_style.update({int(k): v for k, v in self.style_map["id_to_style"].items()})
        elif "style_to_id" in self.style_map:
            self.id_to_style.update({int(v): k for k, v in self.style_map["style_to_id"].items()})

        if self.ext in (".pth", ".pt"):
            self._init_pytorch()
        elif self.ext == ".onnx":
            self._init_onnx()
        else:
            raise ValueError(f"Unsupported model extension '{self.ext}'. Expected .pth or .onnx")

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _init_pytorch(self) -> None:
        import torch
        from train import MultiHeadFontCNN

        self.torch = torch
        self.device = torch.device(self.device_str)

        logging.info(f"Loading PyTorch checkpoint from: {self.model_path.name} (device={self.device})")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            num_fonts = checkpoint.get("num_fonts", max(len(self.id_to_font), 1))
            num_styles = checkpoint.get("num_styles", 4)
            self.model = MultiHeadFontCNN(num_fonts=num_fonts, num_styles=num_styles)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            raise ValueError("Unrecognized PyTorch checkpoint format.")

        self.model.to(self.device)
        self.model.eval()

    def _init_onnx(self) -> None:
        import onnxruntime as ort

        self.ort = ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device_str == "cuda" else ["CPUExecutionProvider"]
        logging.info(f"Initializing ONNX Runtime session from: {self.model_path.name}")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: Union[str, Path], top_k: int = 3) -> Dict[str, Any]:
        """
        Executes inference on a single image.
        Returns dictionary with top-K font predictions and style distribution.
        """
        tensor = preprocess_image(image_path, target_size=(256, 256), normalize_mode="inverted")
        batch_tensor = tensor.unsqueeze(0)  # (1, 1, 256, 256)

        if self.ext in (".pth", ".pt"):
            with self.torch.no_grad():
                inp = batch_tensor.to(self.device)
                f_logits, s_logits = self.model(inp)
                f_probs = self.torch.softmax(f_logits, dim=1).cpu().numpy()[0]
                s_probs = self.torch.softmax(s_logits, dim=1).cpu().numpy()[0]
        else:
            inp_numpy = batch_tensor.numpy()
            f_out, s_out = self.session.run(None, {self.input_name: inp_numpy})
            f_probs = softmax(f_out)[0]
            s_probs = softmax(s_out)[0]

        # Top-K Font Candidates
        num_fonts = len(f_probs)
        k = min(top_k, num_fonts)
        top_font_indices = np.argsort(f_probs)[::-1][:k]

        font_candidates = []
        for rank, idx in enumerate(top_font_indices, start=1):
            font_id = int(idx)
            font_name = self.id_to_font.get(font_id, f"Font_{font_id}")
            prob = float(f_probs[font_id])
            font_candidates.append({
                "rank": rank,
                "font_id": font_id,
                "font_family": font_name,
                "confidence": prob,
                "confidence_percent": round(prob * 100.0, 2),
            })

        # Style Predictions
        style_distribution = {}
        for s_idx, s_prob in enumerate(s_probs):
            s_name = self.id_to_style.get(s_idx, STYLE_NAMES[s_idx] if s_idx < 4 else f"Style_{s_idx}")
            style_distribution[s_name] = {
                "style_id": s_idx,
                "confidence": float(s_prob),
                "confidence_percent": round(float(s_prob) * 100.0, 2),
            }

        top_style_idx = int(np.argmax(s_probs))
        top_style_name = self.id_to_style.get(top_style_idx, STYLE_NAMES[top_style_idx] if top_style_idx < 4 else f"Style_{top_style_idx}")
        top_style_conf = float(s_probs[top_style_idx])

        return {
            "image_path": str(image_path),
            "predicted_style": top_style_name,
            "predicted_style_confidence": top_style_conf,
            "predicted_style_confidence_percent": round(top_style_conf * 100.0, 2),
            "style_distribution": style_distribution,
            "top_k_fonts": font_candidates,
        }


def print_prediction_result(res: Dict[str, Any]) -> None:
    """Formats and prints prediction results to terminal."""
    img_name = Path(res["image_path"]).name
    top_font = res["top_k_fonts"][0]["font_family"] if res["top_k_fonts"] else "Unknown"
    top_font_conf = res["top_k_fonts"][0]["confidence_percent"] if res["top_k_fonts"] else 0.0
    top_style = res["predicted_style"]
    top_style_conf = res["predicted_style_confidence_percent"]

    print("\n" + "─" * 72)
    print(f" 🔍 Target Image : {res['image_path']}")
    print("─" * 72)
    print(f"  ✦ Primary Font   : \033[1;32m{top_font}\033[0m ({top_font_conf:.1f}% confidence)")
    print(f"  ✦ Inferred Style : \033[1;36m{top_style}\033[0m ({top_style_conf:.1f}% confidence)")
    print("─" * 72)

    # Font Candidates Table
    print("  TOP FONT CANDIDATES:")
    print(f"  {'Rank':<5} | {'Font Family':<26} | {'ID':<4} | {'Confidence':<8} | {'Distribution'}")
    print("  " + "─" * 68)
    for c in res["top_k_fonts"]:
        bar = make_progress_bar(c["confidence_percent"], length=14)
        print(f"  #{c['rank']:<4} | {c['font_family']:<26} | {c['font_id']:<4} | {c['confidence_percent']:>5.1f}%   | [{bar}]")

    print("\n  STYLE PROBABILITY DISTRIBUTION:")
    for s_name, s_data in res["style_distribution"].items():
        bar = make_progress_bar(s_data["confidence_percent"], length=14)
        is_top = (s_name == top_style)
        marker = "★" if is_top else " "
        print(f"   {marker} {s_name:<12} : {s_data['confidence_percent']:>5.1f}%  [{bar}]")
    print("─" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict font family options and style from a PNG image file using PyTorch or ONNX."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input PNG image or directory of images.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/test_10_fonts_run/best_model.pth",
        help="Path to trained PyTorch (.pth) or ONNX (.onnx) model checkpoint.",
    )
    parser.add_argument(
        "--font_map",
        type=str,
        default=None,
        help="Path to font_map.json (auto-detected if None).",
    )
    parser.add_argument(
        "--style_map",
        type=str,
        default=None,
        help="Path to style_map.json (auto-detected if None).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top font candidates to display (default: 3).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to export predictions as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Target inference device ('cpu', 'cuda', 'mps').",
    )

    args = parser.parse_args()

    # Load mappings
    model_path = Path(args.model_path).resolve()
    model_dir = model_path.parent

    font_map_path = (
        Path(args.font_map).resolve()
        if args.font_map
        else (model_dir / "font_map.json" if (model_dir / "font_map.json").exists() else Path("test_10_fonts_dataset/font_map.json"))
    )

    style_map_path = (
        Path(args.style_map).resolve()
        if args.style_map
        else (model_dir / "style_map.json" if (model_dir / "style_map.json").exists() else Path("test_10_fonts_dataset/style_map.json"))
    )

    font_map = {}
    if font_map_path.exists():
        with open(font_map_path, "r", encoding="utf-8") as f:
            font_map = json.load(f)

    style_map = {}
    if style_map_path.exists():
        with open(style_map_path, "r", encoding="utf-8") as f:
            style_map = json.load(f)

    # Initialize Predictor
    predictor = ModelPredictor(
        model_path=model_path,
        device=args.device,
        font_map=font_map,
        style_map=style_map,
    )

    # Discover Target Images
    input_path = Path(args.image).resolve()
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            [p for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG") for p in input_path.rglob(ext)]
        )
        if not image_files:
            logging.error(f"No image files found in directory: {input_path}")
            sys.exit(1)
        logging.info(f"Discovered {len(image_files)} images in directory: {input_path}")
    else:
        logging.error(f"Input path not found: {input_path}")
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []

    for img_p in image_files:
        try:
            res = predictor.predict(img_p, top_k=args.top_k)
            all_results.append(res)
            print_prediction_result(res)
        except Exception as e:
            logging.error(f"Failed to run inference on {img_p}: {e}")

    # Export to JSON if requested
    if args.output_json:
        out_json_path = Path(args.output_json).resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Saved inference results ({len(all_results)} images) to: {out_json_path}")


if __name__ == "__main__":
    main()
