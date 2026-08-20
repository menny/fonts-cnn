#!/usr/bin/env python3
"""
generate_dataset.py

Generates a synthetic dataset of font image patches for training a multi-head font and style classifier.

Features:
1. Recursive font discovery (.ttf, .otf) with TTFont metadata extraction (Family Name, Subfamily/Style).
2. 4-class discrete style classification:
   - 0: "Regular"
   - 1: "Bold"
   - 2: "Italic"
   - 3: "Bold-Italic"
3. Text template rendering across Sentences, Numbers, Dates, Times, and Alphanumeric IDs.
4. Canvas rendering with tight bounding box cropping, uniform aspect-ratio scaling, center-padding,
   and subtle augmentations (random padding offset 2-8px, subtle blur, subtle threshold dithering).
5. Manifest CSV generation and bidirectional JSON mappings (font_map.json, style_map.json).
6. Multi-threaded / multi-process execution with CLI progress bars via tqdm.
"""

import argparse
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
import concurrent.futures

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Discrete Style Mapping
STYLE_MAP = {
    0: "Regular",
    1: "Bold",
    2: "Italic",
    3: "Bold-Italic",
}
STYLE_TO_ID = {v: k for k, v in STYLE_MAP.items()}

# Exact Text Templates as per specifications
TEMPLATES_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Sphinx of black quartz, judge my vow.",
    "How vexingly quick daft zebras jump!",
]

TEMPLATES_NUMBERS = [
    "42",
    "809",
    "1701",
    "38291",
    "940284",
    "1048576",
    "58392017",
    "928374615",
    "0123456789",
    "8005550199",
]

TEMPLATES_DATES = [
    "2026-08-18",
    "18/08/2026",
    "October 24, 1985",
]

TEMPLATES_TIMES = [
    "14:30:15",
    "09:45 AM",
    "23:59",
]

TEMPLATES_IDS = [
    "7XYZ890",
    "ABC-1234",
    "NY-882-PL",
    "ORD-94021",
]

ALL_TEMPLATES: List[Tuple[str, str]] = (
    [("sentence", t) for t in TEMPLATES_SENTENCES]
    + [("number", t) for t in TEMPLATES_NUMBERS]
    + [("date", t) for t in TEMPLATES_DATES]
    + [("time", t) for t in TEMPLATES_TIMES]
    + [("id", t) for t in TEMPLATES_IDS]
)


def sanitize_filename(name: str) -> str:
    """Sanitizes string for use in file and directory paths."""
    clean = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    return clean or "unknown_font"


def extract_font_metadata(font_path: str) -> Optional[Tuple[str, str, int]]:
    """
    Extracts normalized font family name and classifies font style using fontTools.
    
    Returns:
        (font_family, style_name, style_id) or None if unsupported/corrupt.
    """
    try:
        tt = TTFont(font_path, fontNumber=0, lazy=True)
        if "name" not in tt:
            logging.warning(f"Skipping {font_path}: No 'name' table found.")
            return None

        name_table = tt["name"]

        # Extract normalized family name (Prefer Name ID 16 - Typographic Family, fallback ID 1)
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

        # Extract subfamily / style string (Prefer Name ID 17 - Typographic Subfamily, fallback ID 2)
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

        # Check head table macStyle
        mac_style = 0
        if "head" in tt:
            mac_style = getattr(tt["head"], "macStyle", 0)

        # Determine Italic
        is_italic = (
            (fs_selection & 1 != 0)  # Bit 0: ITALIC
            or (mac_style & 2 != 0)  # Bit 1: Italic
            or any(k in subfamily_lower for k in ["italic", "oblique", "slanted", "ital", "obl"])
            or any(k in fname_lower for k in ["italic", "oblique", "ital"])
        )

        # Determine Bold
        is_bold = (
            (fs_selection & 32 != 0)  # Bit 5: BOLD
            or (mac_style & 1 != 0)   # Bit 0: Bold
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

    except (TTLibError, Exception) as e:
        logging.warning(f"Failed to parse font metadata for {font_path}: {e}")
        return None


def apply_subtle_threshold_dithering(img: Image.Image) -> Image.Image:
    """Applies subtle threshold jitter and dithering to simulate authentic scan/print nuances."""
    arr = np.array(img, dtype=np.float32)
    # Mild high frequency gaussian noise
    noise = np.random.normal(0.0, 1.2, arr.shape)
    noisy = np.clip(arr + noise, 0.0, 255.0)

    # Subtle contrast modulation around threshold
    threshold = random.uniform(124.0, 132.0)
    steepness = random.uniform(0.02, 0.04)
    # Sigmoidal micro-thresholding response
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
    Renders text tightly cropped, uniformly scaled, center-padded, and subtly augmented.
    """
    if sample_seed is not None:
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))

    font_size = random.randint(28, 44)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logging.debug(f"Unable to load font {font_path} at size {font_size}: {e}")
        return None

    # Handle multi-line wrapping for sentences
    if category == "sentence" and len(text) > 20:
        # Wrap dynamically into 2-3 lines
        wrap_width = random.randint(20, 26)
        render_text = textwrap.fill(text, width=wrap_width)
    else:
        render_text = text

    # Render onto oversized temporary white canvas to prevent any glyph clipping
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
    pad = random.randint(2, 8)
    max_w = max(1, image_size - 2 * pad)
    max_h = max(1, image_size - 2 * pad)

    # Scale uniformly preserving aspect ratio
    scale = min(max_w / crop_w, max_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center-pad on image_size x image_size white canvas with slight random offset
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    base_x = (image_size - new_w) // 2
    base_y = (image_size - new_h) // 2

    # Slight random padding offset jitter (within bounds)
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
    """
    Worker function to process a single font file across all templates and sample variations.
    """
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
        description="Generate a synthetic dataset of font image patches for training font & style classifiers."
    )
    parser.add_argument(
        "--fonts_dir",
        type=str,
        required=True,
        help="Path to directory containing font files (.ttf, .otf).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target directory for generated PNGs and metadata.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Target square canvas dimension as integer (default: 256).",
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
        help="Number of parallel worker threads/processes (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation (default: 42).",
    )
    parser.add_argument(
        "--max_fonts",
        type=int,
        default=None,
        help="Optional maximum number of font files to process (useful for test runs).",
    )

    args = parser.parse_args()

    fonts_dir = Path(args.fonts_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not fonts_dir.exists() or not fonts_dir.is_dir():
        logging.error(f"Fonts directory does not exist or is not a directory: {fonts_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Scanning for font files in: {fonts_dir}")
    all_font_files = find_font_files(fonts_dir)
    logging.info(f"Discovered {len(all_font_files)} font files.")

    if not all_font_files:
        logging.error("No .ttf or .otf font files found.")
        sys.exit(1)

    if args.max_fonts:
        all_font_files = all_font_files[: args.max_fonts]
        logging.info(f"Limiting to first {args.max_fonts} font files.")

    # Phase 1: Parse and validate font metadata
    logging.info("Parsing font metadata and style classifications...")
    parsed_fonts: List[Tuple[str, str, str, int]] = []  # (font_path, font_family, style_name, style_id)
    unique_families = set()

    for fp in tqdm(all_font_files, desc="Parsing Font Metadata"):
        meta = extract_font_metadata(str(fp))
        if meta is not None:
            fam, sname, sid = meta
            parsed_fonts.append((str(fp), fam, sname, sid))
            unique_families.add(fam)

    logging.info(f"Successfully validated {len(parsed_fonts)} font files across {len(unique_families)} unique font families.")

    if not parsed_fonts:
        logging.error("No valid fonts could be parsed.")
        sys.exit(1)

    # Deterministic mapping for font families
    sorted_families = sorted(list(unique_families))
    font_to_id = {fam: idx for idx, fam in enumerate(sorted_families)}
    id_to_font = {str(idx): fam for idx, fam in enumerate(sorted_families)}

    # Save font_map.json
    font_map_path = output_dir / "font_map.json"
    with open(font_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "font_to_id": font_to_id,
                "id_to_font": id_to_font,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logging.info(f"Saved font mapping to {font_map_path}")

    # Save style_map.json
    style_map_path = output_dir / "style_map.json"
    with open(style_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "style_to_id": STYLE_TO_ID,
                "id_to_style": {str(k): v for k, v in STYLE_MAP.items()},
            },
            f,
            indent=2,
        )
    logging.info(f"Saved style mapping to {style_map_path}")

    # Phase 2: Render Dataset Patches in Parallel
    total_samples = len(parsed_fonts) * len(ALL_TEMPLATES) * args.samples_per_template
    logging.info(
        f"Generating {total_samples} image patches ({len(ALL_TEMPLATES)} templates x {args.samples_per_template} samples/template) using {args.workers} workers..."
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
                logging.error(f"Error processing font task: {e}")

    # Phase 3: Write dataset_manifest.csv
    manifest_path = output_dir / "dataset_manifest.csv"
    df = pd.DataFrame(all_manifest_records, columns=["image_path", "font_family", "font_id", "style_name", "style_id"])
    df.to_csv(manifest_path, index=False)
    logging.info(f"Saved dataset manifest with {len(df)} records to {manifest_path}")

    # Summary report
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output Directory       : {output_dir}")
    print(f"Total Font Families    : {len(unique_families)}")
    print(f"Total Font Files       : {len(parsed_fonts)}")
    print(f"Total Generated Images : {len(df)}")
    print(f"Image Canvas Size      : {args.image_size}x{args.image_size}")
    print(f"Templates per Font     : {len(ALL_TEMPLATES)}")
    print(f"Samples per Template   : {args.samples_per_template}")
    print(f"Dataset Manifest       : {manifest_path}")
    print(f"Font Map               : {font_map_path}")
    print(f"Style Map              : {style_map_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
