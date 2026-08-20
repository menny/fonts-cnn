#!/usr/bin/env python3
"""
generate_dataset_v2.py

Enhanced synthetic dataset generator with realistic multi-line digit grids,
scale-consistent number rendering, and product date/expiry formats.

Key improvements:
1. Balanced multi-line digit matrices (0-9) to guarantee comprehensive digit representation.
2. Realistic expiry dates, lot codes, prices, and timestamps.
3. Aspect-ratio and scale-clamped rendering: prevents short numbers from blowing up into oversized glyphs.
4. Clean filtering for top 200 popular Google Fonts.
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from fontTools.ttLib import TTFont, TTLibError
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

STYLE_MAP = {
    0: "Regular",
    1: "Bold",
    2: "Italic",
    3: "Bold-Italic",
}
STYLE_TO_ID = {v: k for k, v in STYLE_MAP.items()}

# Enhanced, balanced template catalog
TEMPLATES_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Sphinx of black quartz, judge my vow.",
    "How vexingly quick daft zebras jump!",
    "Pack my box with five dozen liquor jugs.",
    "Woven silk pyjamas exchanged for blue quartz.",
]

TEMPLATES_NUMBER_MATRICES = [
    "0 1 2 3 4\n5 6 7 8 9",
    "9 8 7 6 5\n4 3 2 1 0",
    "123 456 7890\n098 765 4321",
    "0123456789\n9876543210",
    "13579  24680\n02468  13579",
    "42  809  1701\n38291  940284",
    "1048576  58392017\n928374615  012345",
]

TEMPLATES_DATES_EXPIRY = [
    "EXP 18/08/2026\nLOT: 940284",
    "BEST BY: 24 OCT 1985\nPROD: 14:30:15",
    "2026-08-18\n09:45 AM",
    "18.08.2026\n23:59:00",
    "USE BY: 12/2028\nBATCH: #7XYZ890",
    "OCTOBER 24, 1985\nAUGUST 18, 2026",
]

TEMPLATES_FINANCIAL_CODES = [
    "(800) 555-0199\n+1 212 736 5000",
    "TOTAL: $19.99\nTAX: €4.50  #809",
    "REF: 7XYZ-890\nQTY: 42  ORD-94021",
    "NY-882-PL  ABC-1234\nORD-94021  7XYZ890",
    "14:30:15  09:45\n23:59:00  12:00",
]

TEMPLATES_SINGLE_LINE = [
    "2026-08-18",
    "18/08/2026",
    "0123456789",
    "8005550199",
    "ORD-94021",
]

ALL_TEMPLATES: List[Tuple[str, str]] = (
    [("sentence", t) for t in TEMPLATES_SENTENCES]
    + [("matrix", t) for t in TEMPLATES_NUMBER_MATRICES]
    + [("date_exp", t) for t in TEMPLATES_DATES_EXPIRY]
    + [("finance_code", t) for t in TEMPLATES_FINANCIAL_CODES]
    + [("single_line", t) for t in TEMPLATES_SINGLE_LINE]
)


def sanitize_filename(name: str) -> str:
    """Sanitizes string for use in file and directory paths."""
    clean = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    return clean or "unknown_font"


def extract_font_metadata(font_path: str) -> Optional[Tuple[str, str, int]]:
    """Extracts normalized font family name and classifies font style using fontTools."""
    try:
        tt = TTFont(font_path, fontNumber=0, lazy=True)
        if "name" not in tt:
            return None

        name_table = tt["name"]

        # Extract normalized family name (Prefer Name ID 16, fallback ID 1)
        family = None
        for nid in (16, 1):
            for rec in name_table.names:
                if rec.nameID == nid:
                    try:
                        val = rec.toUnicode()
                        if val and val.strip():
                            family = val.strip()
                            break
                    except Exception:
                        pass
            if family:
                break

        if not family:
            family = name_table.getBestFamilyName()
        if not family:
            family = Path(font_path).stem.split("-")[0].replace("_", " ")

        # Extract subfamily / style string (Prefer Name ID 17, fallback ID 2)
        subfamily = ""
        for nid in (17, 2):
            for rec in name_table.names:
                if rec.nameID == nid:
                    try:
                        val = rec.toUnicode()
                        if val and val.strip():
                            subfamily = val.strip()
                            break
                    except Exception:
                        pass
            if subfamily:
                break

        subfamily_lower = subfamily.lower()
        fname_lower = Path(font_path).name.lower()

        # Check OS/2 table flags & weight class
        fs_selection = 0
        weight_class = 400
        if "OS/2" in tt:
            os2 = tt["OS/2"]
            fs_selection = getattr(os2, "fsSelection", 0)
            weight_class = getattr(os2, "usWeightClass", 400)

        mac_style = 0
        if "head" in tt:
            mac_style = getattr(tt["head"], "macStyle", 0)

        # Determine Italic
        is_italic = (
            (fs_selection & 1 != 0)
            or (mac_style & 2 != 0)
            or any(k in subfamily_lower for k in ["italic", "oblique", "slanted", "ital", "obl"])
            or any(k in fname_lower for k in ["italic", "oblique", "ital"])
        )

        # Determine Bold
        is_bold = (
            (fs_selection & 32 != 0)
            or (mac_style & 1 != 0)
            or (weight_class >= 600)
            or any(k in subfamily_lower for k in ["bold", "black", "heavy", "semibold", "demibold", "extrabold", "ultrabold", "w700", "w800", "w900", "700", "800", "900"])
            or any(k in fname_lower for k in ["bold", "black", "heavy", "semibold", "demibold", "extrabold", "ultrabold"])
        )

        if is_bold and is_italic:
            style_id = 3
        elif is_italic:
            style_id = 2
        elif is_bold:
            style_id = 1
        else:
            style_id = 0

        style_name = STYLE_MAP[style_id]
        return family, style_name, style_id

    except (TTLibError, Exception):
        return None


def apply_subtle_threshold_dithering(img: Image.Image) -> Image.Image:
    """Applies subtle threshold jitter and dithering to simulate authentic scan/print nuances."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0.0, 1.2, arr.shape)
    noisy = np.clip(arr + noise, 0.0, 255.0)

    threshold = random.uniform(124.0, 132.0)
    steepness = random.uniform(0.02, 0.04)
    factor = 1.0 / (1.0 + np.exp(-steepness * (noisy - threshold)))
    dithered = np.where(factor < 0.5, noisy * random.uniform(0.92, 1.0), 255.0 - (255.0 - noisy) * random.uniform(0.92, 1.0))
    dithered = np.clip(dithered, 0, 255).astype(np.uint8)
    return Image.fromarray(dithered)


def render_patch(
    font_path: str,
    text: str,
    category: str,
    image_size: int = 256,
    sample_seed: Optional[int] = None,
) -> Optional[Image.Image]:
    """
    Renders text tightly cropped, scaled with realistic glyph constraints, center-padded, and augmented.
    """
    if sample_seed is not None:
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))

    font_size = random.randint(28, 42)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        return None

    # Handle multi-line wrapping for sentences
    if category == "sentence" and len(text) > 20:
        wrap_width = random.randint(20, 26)
        render_text = textwrap.fill(text, width=wrap_width)
    else:
        render_text = text

    # Oversized temporary canvas
    temp_w, temp_h = 1600, 1600
    temp_img = Image.new("RGB", (temp_w, temp_h), (255, 255, 255))
    draw = ImageDraw.Draw(temp_img)

    draw.multiline_text(
        (100, 100),
        render_text,
        font=font,
        fill=(0, 0, 0),
        align="center",
        spacing=random.randint(4, 8),
    )

    # Tight crop around non-white pixels
    inverted = ImageOps.invert(temp_img.convert("L"))
    bbox = inverted.getbbox()
    if not bbox:
        return None

    cropped = temp_img.crop(bbox)
    crop_w, crop_h = cropped.size
    if crop_w <= 0 or crop_h <= 0:
        return None

    # Determine inner target bounds with random padding offset (2-8px)
    pad = random.randint(4, 10)
    max_w = max(1, image_size - 2 * pad)
    max_h = max(1, image_size - 2 * pad)

    # Scale with realistic scaling constraints:
    # Avoid scaling short numbers up to 600%
    raw_scale = min(max_w / crop_w, max_h / crop_h)
    if category in ("single_line", "matrix"):
        # Clamp maximum magnification so glyph heights remain realistic
        scale = min(raw_scale, 1.35)
    else:
        scale = raw_scale

    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center-pad on image_size x image_size white canvas with slight random offset
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    base_x = (image_size - new_w) // 2
    base_y = (image_size - new_h) // 2

    jitter_x = random.randint(-pad // 2, pad // 2) if pad > 2 else 0
    jitter_y = random.randint(-pad // 2, pad // 2) if pad > 2 else 0
    pos_x = max(0, min(image_size - new_w, base_x + jitter_x))
    pos_y = max(0, min(image_size - new_h, base_y + jitter_y))

    canvas.paste(resized, (pos_x, pos_y))

    # Augmentation: Subtle Blur (radius 0.1 to 0.4)
    blur_radius = random.uniform(0.1, 0.4)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Augmentation: Subtle threshold dithering
    canvas = apply_subtle_threshold_dithering(canvas)

    return canvas


def process_font_file(
    args_tuple: Tuple[str, str, int, int, int, str, int, int]
) -> List[Dict[str, Any]]:
    """Worker function to process a single font file across all templates and sample variations."""
    (
        font_path,
        output_dir,
        image_size,
        samples_per_template,
        base_seed,
        font_family,
        font_id,
        style_id,
    ) = args_tuple

    style_name = STYLE_MAP[style_id]
    safe_family_dir = sanitize_filename(font_family)
    family_output_dir = Path(output_dir) / safe_family_dir
    family_output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for template_idx, (cat, tmpl) in enumerate(ALL_TEMPLATES):
        for sample_idx in range(samples_per_template):
            seed = (
                base_seed
                + font_id * 1000003
                + style_id * 50021
                + template_idx * 1009
                + sample_idx * 17
            ) % (2**31 - 1)

            # Deterministic unique hash
            hash_input = f"{font_family}_{style_name}_{template_idx}_{sample_idx}_{seed}".encode("utf-8")
            hash_str = hashlib.md5(hash_input).hexdigest()[:12]

            img_filename = f"{style_name}_{hash_str}.png"
            img_rel_path = f"{safe_family_dir}/{img_filename}"
            img_abs_path = family_output_dir / img_filename

            patch_img = render_patch(
                font_path=font_path,
                text=tmpl,
                category=cat,
                image_size=image_size,
                sample_seed=seed,
            )

            if patch_img is not None:
                patch_img.save(img_abs_path, format="PNG", optimize=True)
                records.append({
                    "image_path": img_rel_path,
                    "font_family": font_family,
                    "font_id": font_id,
                    "style_name": style_name,
                    "style_id": style_id,
                })

    return records


def find_font_files(fonts_dir: Path) -> List[Path]:
    """Recursively finds all .ttf and .otf font files."""
    font_files = []
    for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
        font_files.extend(fonts_dir.rglob(ext))
    return sorted(list(set(font_files)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset v2 with realistic number matrices, product dates, and balanced scale."
    )
    parser.add_argument(
        "--fonts_dir",
        type=str,
        default="downloaded_fonts",
        help="Path to directory containing font files.",
    )
    parser.add_argument(
        "--popular_json",
        type=str,
        default="popular_200_fonts.json",
        help="Path to popular_200_fonts.json (limits generation to top 200 fonts).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_top200_v2",
        help="Target directory for generated PNGs and metadata.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Target square canvas dimension (default: 256).",
    )
    parser.add_argument(
        "--samples_per_template",
        type=int,
        default=5,
        help="Number of augmentations per text template (default: 5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 16),
        help="Number of parallel worker processes (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Random seed for reproducible augmentation (default: 123456).",
    )

    args = parser.parse_args()

    fonts_dir = Path(args.fonts_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load popular 200 list if present
    target_families = None
    if args.popular_json and Path(args.popular_json).exists():
        with open(args.popular_json, "r") as f:
            target_families = set(json.load(f)[:200])
        logging.info(f"Loaded {len(target_families)} target font families from {args.popular_json}")

    logging.info(f"Scanning for font files in: {fonts_dir}")
    all_font_files = find_font_files(fonts_dir)

    # Filter for target families
    parsed_fonts: List[Tuple[str, str, str, int]] = []
    unique_families = set()

    for fp in tqdm(all_font_files, desc="Validating Fonts"):
        meta = extract_font_metadata(str(fp))
        if meta is not None:
            fam, sname, sid = meta
            if target_families is None or fam in target_families:
                parsed_fonts.append((str(fp), fam, sname, sid))
                unique_families.add(fam)

    logging.info(f"Selected {len(parsed_fonts)} font files across {len(unique_families)} unique font families.")

    # Build deterministic mappings
    sorted_families = sorted(list(unique_families))
    font_to_id = {fam: idx for idx, fam in enumerate(sorted_families)}
    id_to_font = {str(idx): fam for idx, fam in enumerate(sorted_families)}

    # Save font_map.json
    font_map_path = output_dir / "font_map.json"
    with open(font_map_path, "w", encoding="utf-8") as f:
        json.dump({"font_to_id": font_to_id, "id_to_font": id_to_font}, f, indent=2, ensure_ascii=False)

    # Save style_map.json
    style_map_path = output_dir / "style_map.json"
    with open(style_map_path, "w", encoding="utf-8") as f:
        json.dump({"style_to_id": STYLE_TO_ID, "id_to_style": {str(k): v for k, v in STYLE_MAP.items()}}, f, indent=2)

    # Render Dataset Patches in Parallel
    total_samples = len(parsed_fonts) * len(ALL_TEMPLATES) * args.samples_per_template
    logging.info(
        f"Generating {total_samples} image patches ({len(ALL_TEMPLATES)} templates x {args.samples_per_template} samples) across {len(parsed_fonts)} fonts using {args.workers} workers..."
    )

    tasks = [
        (
            fpath,
            str(output_dir),
            args.image_size,
            args.samples_per_template,
            args.seed,
            fam,
            font_to_id[fam],
            sid,
        )
        for (fpath, fam, _, sid) in parsed_fonts
    ]

    all_manifest_records: List[Dict[str, Any]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_font_file, t) for t in tasks]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Rendering Patches"):
            try:
                res = f.result()
                all_manifest_records.extend(res)
            except Exception as e:
                logging.error(f"Error processing font: {e}")

    # Save dataset_manifest.csv
    manifest_path = output_dir / "dataset_manifest.csv"
    df = pd.DataFrame(all_manifest_records, columns=["image_path", "font_family", "font_id", "style_name", "style_id"])
    df.to_csv(manifest_path, index=False)
    logging.info(f"Saved dataset manifest with {len(df)} records to {manifest_path}")

    print("\n" + "=" * 60)
    print("DATASET V2 GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output Directory       : {output_dir}")
    print(f"Total Font Families    : {len(unique_families)}")
    print(f"Total Font Files       : {len(parsed_fonts)}")
    print(f"Total Generated Images : {len(df)}")
    print(f"Dataset Manifest       : {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
