#!/usr/bin/env python3
"""
download_google_fonts.py

A standalone Python script to download font files (.ttf) from Google Fonts into a local directory.
Fetches catalog metadata directly without a full git clone, validates downloaded fonts with fontTools,
and supports multi-threaded downloads with exponential backoff and resume capability.
"""

import argparse
import concurrent.futures
import csv
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from fontTools.ttLib import TTFont

# Constants
METADATA_API_URL = "https://fonts.google.com/metadata/fonts"
TAGS_FAMILIES_CSV_URL = "https://raw.githubusercontent.com/google/fonts/main/tags/all/families.csv"
RAW_GITHUB_BASE = "https://raw.githubusercontent.com/google/fonts/main"
LICENSES = ["ofl", "apache", "ufl"]

CATEGORIES = ["all", "sans-serif", "serif", "display", "monospace", "handwriting"]


def create_session(workers: int) -> requests.Session:
    """Create a requests session with connection pooling and exponential backoff retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=workers * 2,
        pool_maxsize=workers * 4,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; GoogleFontsDownloader/1.0; +https://github.com/google/fonts)"
    })
    return session


def normalize_category(cat: Optional[str]) -> str:
    """Normalize category strings for uniform comparison."""
    if not cat:
        return ""
    return re.sub(r"[^a-z]", "", cat.lower())


def fetch_metadata(session: requests.Session) -> List[Dict[str, Any]]:
    """
    Fetch the Google Fonts metadata list.
    Tries the official fonts.google.com metadata endpoint, falling back to families.csv.
    """
    try:
        resp = session.get(METADATA_API_URL, timeout=15)
        if resp.status_code == 200:
            text = resp.text
            # Google JSON responses sometimes have a leading )]}'\n prefix
            if text.startswith(")]}'\n"):
                text = text[5:]
            data = json.loads(text)
            if "familyMetadataList" in data:
                return data["familyMetadataList"]
    except Exception as e:
        tqdm.write(f"Warning: Failed to fetch metadata from {METADATA_API_URL} ({e}). Falling back to families.csv...")

    # Fallback to families.csv
    try:
        resp = session.get(TAGS_FAMILIES_CSV_URL, timeout=15)
        if resp.status_code == 200:
            families_set = set()
            reader = csv.reader(resp.text.splitlines())
            for row in reader:
                if row and row[0].strip():
                    families_set.add(row[0].strip())
            
            return [
                {
                    "family": name,
                    "category": "Sans Serif",
                    "popularity": idx + 1,
                }
                for idx, name in enumerate(sorted(families_set))
            ]
    except Exception as e:
        tqdm.write(f"Error: Failed to fetch fallback metadata from {TAGS_FAMILIES_CSV_URL}: {e}")
        return []

    return []


def generate_family_slugs(family_name: str) -> List[str]:
    """Generate candidate repository directory slugs for a given font family name."""
    slugs = []
    
    # 1. Lowercase alphanumeric only (e.g., "Open Sans" -> "opensans", "Playfair Display SC" -> "playfairdisplaysc")
    s1 = re.sub(r"[^a-zA-Z0-9]", "", family_name.lower())
    if s1 and s1 not in slugs:
        slugs.append(s1)
        
    # 2. Lowercase with spaces removed
    s2 = family_name.lower().replace(" ", "")
    if s2 and s2 not in slugs:
        slugs.append(s2)

    # 3. Lowercase with underscores
    s3 = family_name.lower().replace(" ", "_")
    if s3 and s3 not in slugs:
        slugs.append(s3)

    # 4. Lowercase with hyphens
    s4 = family_name.lower().replace(" ", "-")
    if s4 and s4 not in slugs:
        slugs.append(s4)

    return slugs


def parse_metadata_pb(pb_text: str) -> List[Dict[str, Any]]:
    """
    Parse font entries from METADATA.pb protobuf text.
    Extracts font name, filename, style, and weight.
    """
    fonts = []
    font_blocks = re.findall(r"fonts\s*\{([^}]+)\}", pb_text)
    
    for block in font_blocks:
        fn_match = re.search(r'filename:\s*\"([^\"]+)\"', block)
        name_match = re.search(r'name:\s*\"([^\"]+)\"', block)
        style_match = re.search(r'style:\s*\"([^\"]+)\"', block)
        weight_match = re.search(r'weight:\s*(\d+)', block)
        ps_name_match = re.search(r'post_script_name:\s*\"([^\"]+)\"', block)

        if fn_match:
            filename = fn_match.group(1).strip()
            style = style_match.group(1).strip() if style_match else "normal"
            weight = int(weight_match.group(1).strip()) if weight_match else 400
            name = name_match.group(1).strip() if name_match else ""
            ps_name = ps_name_match.group(1).strip() if ps_name_match else ""

            fonts.append({
                "filename": filename,
                "style": style,
                "weight": weight,
                "name": name,
                "post_script_name": ps_name,
            })

    return fonts


def get_font_weight_priority(font_info: Dict[str, Any]) -> int:
    """
    Assign priority score for sorting font weights.
    Prioritize static weights: Regular (400 normal), Bold (700 normal), Italic (400 italic), Bold-Italic (700 italic).
    """
    style = font_info.get("style", "normal").lower()
    weight = font_info.get("weight", 400)
    filename = font_info.get("filename", "")

    # Static priority
    if weight == 400 and style == "normal":
        return 0  # Regular
    elif weight == 700 and style == "normal":
        return 1  # Bold
    elif weight == 400 and style in ("italic", "italic"):
        return 2  # Italic
    elif weight == 700 and style in ("italic", "italic"):
        return 3  # Bold Italic
    elif "[" not in filename:
        # Other static weights (e.g. 300, 500, 600, 800, 900)
        return 10 + abs(400 - weight)
    else:
        # Variable font
        return 100


def resolve_family_fonts(
    family_name: str, session: requests.Session
) -> Optional[Tuple[str, str, List[Dict[str, Any]]]]:
    """
    Locate the family in google/fonts repo under ofl/, apache/, or ufl/,
    fetches METADATA.pb, and parses available TTF files.
    Returns (license, slug, font_list) or None if not found.
    """
    slugs = generate_family_slugs(family_name)
    for lic in LICENSES:
        for slug in slugs:
            pb_url = f"{RAW_GITHUB_BASE}/{lic}/{slug}/METADATA.pb"
            try:
                resp = session.get(pb_url, timeout=10)
                if resp.status_code == 200:
                    font_list = parse_metadata_pb(resp.text)
                    if font_list:
                        # Sort fonts prioritizing Regular, Bold, Italic, Bold-Italic
                        font_list.sort(key=get_font_weight_priority)
                        return lic, slug, font_list
            except Exception:
                continue
    return None


def is_valid_ttf(file_path: Path) -> bool:
    """Validate that the file is an uncorrupted OpenType/TrueType binary using fontTools."""
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return False
    try:
        with TTFont(file_path, lazy=True) as font:
            # Check for standard TrueType/OpenType tables
            required_tables = {"head", "maxp", "cmap", "name"}
            return bool(required_tables.intersection(set(font.keys())))
    except Exception:
        return False


def download_single_font_file(
    download_url: str,
    target_path: Path,
    session: requests.Session,
) -> Tuple[bool, str]:
    """
    Download a font file to target_path with validation and resume skipping.
    Returns (success: bool, status_message: str).
    """
    # 1. Resume Check: Skip if already downloaded and valid
    if target_path.exists() and is_valid_ttf(target_path):
        return True, "skipped"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")

    try:
        resp = session.get(download_url, timeout=20, stream=True)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"

        with open(temp_target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=16384):
                if chunk:
                    f.write(chunk)

        # 2. Validation check
        if not is_valid_ttf(temp_target):
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
            return False, "Corrupted font binary"

        # 3. Atomic replacement
        temp_target.replace(target_path)
        return True, "downloaded"

    except Exception as e:
        if temp_target.exists():
            temp_target.unlink(missing_ok=True)
        return False, str(e)


def process_family(
    family_info: Dict[str, Any],
    output_dir: Path,
    session: requests.Session,
    file_progress_bar: Optional[tqdm] = None,
) -> Dict[str, Any]:
    """
    Process a single font family: resolve repo location, download TTF files, validate.
    Returns a dictionary of statistics for this family.
    """
    family_name = family_info["family"]
    res = resolve_family_fonts(family_name, session)

    if not res:
        return {
            "family": family_name,
            "status": "failed",
            "reason": "Not found in repository (may be proprietary or non-standard directory)",
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "files": [],
        }

    lic, slug, font_list = res
    family_dir = output_dir / family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    file_results = []

    for font in font_list:
        raw_filename = font["filename"]
        base_filename = Path(raw_filename).name

        # Ensure naming conforms to fonts/<family_name>/<family_name>-<style>.ttf or standard filename
        target_path = family_dir / base_filename

        # URL encode filename components like brackets in variable fonts
        quoted_filename = urllib.parse.quote(raw_filename, safe="/")
        download_url = f"{RAW_GITHUB_BASE}/{lic}/{slug}/{quoted_filename}"

        success, status = download_single_font_file(download_url, target_path, session)

        if success:
            if status == "downloaded":
                downloaded_count += 1
            else:
                skipped_count += 1
        else:
            failed_count += 1

        file_results.append((base_filename, status))
        if file_progress_bar is not None:
            file_progress_bar.update(1)

    family_status = "success" if (failed_count == 0 and (downloaded_count + skipped_count) > 0) else (
        "partial" if (downloaded_count + skipped_count) > 0 else "failed"
    )

    return {
        "family": family_name,
        "status": family_status,
        "reason": "" if family_status != "failed" else "All font files failed to download",
        "downloaded": downloaded_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "files": file_results,
    }


def print_summary_table(
    families_processed: int,
    total_downloaded_files: int,
    total_skipped_files: int,
    total_failed_files: int,
    failed_families: List[Dict[str, Any]],
    output_dir: Path,
    elapsed_time: float,
):
    """Print an ASCII summary table displaying overall download statistics."""
    separator = "=" * 65
    sub_sep = "-" * 65

    print("\n" + separator)
    print("                    GOOGLE FONTS DOWNLOAD SUMMARY")
    print(separator)
    print(f"  Target Directory:          {output_dir.resolve()}")
    print(f"  Time Elapsed:              {elapsed_time:.2f} seconds")
    print(sub_sep)
    print(f"  Total Families Processed:  {families_processed}")
    print(f"  Total Font Files Saved:    {total_downloaded_files}")
    print(f"  Total Font Files Skipped:  {total_skipped_files} (already existing & valid)")
    print(f"  Total Font Files Failed:   {total_failed_files}")
    print(separator)

    if failed_families:
        print("\nSkipped / Failed Families:")
        print(sub_sep)
        for ff in failed_families[:25]:
            print(f"  • {ff['family']:<28} Reason: {ff.get('reason', 'Unknown error')}")
        if len(failed_families) > 25:
            print(f"  ... and {len(failed_families) - 25} more.")
        print(separator)


def parse_arguments() -> argparse.Namespace:
    """Configure and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download TrueType font files (.ttf) from Google Fonts into a local directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fonts",
        help="Target directory to store font files (default: 'fonts')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of font families to download (default: None, all available)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=CATEGORIES,
        help="Filter by font category (choices: 'all', 'sans-serif', 'serif', 'display', 'monospace', 'handwriting'; default: 'all')",
    )
    parser.add_argument(
        "--popular_only",
        nargs="?",
        const=100,
        type=int,
        default=None,
        help="Restrict downloads to the top N most popular font families (e.g. --popular_only 50; default N=100 if passed as flag)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    session = create_session(args.workers)

    print("Fetching Google Fonts catalog metadata...")
    all_metadata = fetch_metadata(session)

    if not all_metadata:
        print("Error: Could not retrieve font metadata. Please check your internet connection.", file=sys.stderr)
        sys.exit(1)

    print(f"Retrieved metadata for {len(all_metadata)} font families.")

    # 1. Filter by category
    selected_category = args.category.lower()
    if selected_category != "all":
        norm_target_cat = normalize_category(selected_category)
        filtered_metadata = [
            f for f in all_metadata
            if normalize_category(f.get("category", "")) == norm_target_cat
        ]
    else:
        filtered_metadata = list(all_metadata)

    # 2. Filter / Sort by Popularity
    if args.popular_only is not None:
        # Sort by popularity rank ascending (1 = most popular)
        filtered_metadata.sort(key=lambda x: x.get("popularity", 999999))
        pop_limit = max(1, args.popular_only)
        filtered_metadata = filtered_metadata[:pop_limit]
    elif args.limit is not None:
        # Keep popularity sort if available
        filtered_metadata.sort(key=lambda x: x.get("popularity", 999999))

    # 3. Apply limit
    if args.limit is not None and args.limit > 0:
        filtered_metadata = filtered_metadata[: args.limit]

    total_families = len(filtered_metadata)
    if total_families == 0:
        print("No font families match the specified filters.")
        sys.exit(0)

    print(f"Starting download for {total_families} font families using {args.workers} workers...")

    total_downloaded_files = 0
    total_skipped_files = 0
    total_failed_files = 0
    failed_families = []

    # Real-time progress bar for families
    with tqdm(total=total_families, desc="Families", unit="family", dynamic_ncols=True) as fam_pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_family = {
                executor.submit(process_family, fam, output_path, session): fam
                for fam in filtered_metadata
            }

            for future in concurrent.futures.as_completed(future_to_family):
                fam_data = future_to_family[future]
                try:
                    res = future.result()
                    total_downloaded_files += res["downloaded"]
                    total_skipped_files += res["skipped"]
                    total_failed_files += res["failed"]

                    if res["status"] == "failed":
                        failed_families.append(res)

                    fam_pbar.set_postfix({
                        "saved": total_downloaded_files,
                        "skipped": total_skipped_files,
                        "failed": total_failed_files,
                    })
                except Exception as exc:
                    failed_families.append({
                        "family": fam_data.get("family", "Unknown"),
                        "reason": str(exc),
                    })
                finally:
                    fam_pbar.update(1)

    elapsed = time.time() - start_time
    print_summary_table(
        families_processed=total_families,
        total_downloaded_files=total_downloaded_files,
        total_skipped_files=total_skipped_files,
        total_failed_files=total_failed_files,
        failed_families=failed_families,
        output_dir=output_path,
        elapsed_time=elapsed,
    )


if __name__ == "__main__":
    main()
