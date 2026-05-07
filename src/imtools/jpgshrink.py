#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

from .jpgquality import get_jpg_quality

# ============================================================
# CORE OPERATION: SHRINK ONE JPG
# ============================================================


def shrink_jpg(
    image_path: str,
    target_quality: int,
    max_size: Optional[int],
) -> Tuple[str, bool]:
    """
    Try to shrink a single JPG image.

    Shrinking means:
      - optionally downscaling image dimensions
      - recompressing with lower JPEG quality
      - replacing original ONLY if file size decreases

    Returns:
        (image_path, was_replaced)
    """

    path = Path(image_path)

    try:
        original_size = path.stat().st_size
    except OSError:
        return image_path, False

    try:
        with Image.open(path) as img:
            # best-effort quality detection
            try:
                current_quality = get_jpg_quality(img)
            except Exception:
                current_quality = None

            img = img.convert("RGB")

            # ----------------------------------------------------
            # optional resize
            # ----------------------------------------------------
            if max_size is not None:
                w, h = img.size
                if max(w, h) > max_size:
                    scale = max_size / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            # ----------------------------------------------------
            # skip if unlikely to improve
            # ----------------------------------------------------
            if current_quality is not None and current_quality <= target_quality:
                return image_path, False

            # ----------------------------------------------------
            # write temporary file
            # ----------------------------------------------------
            temp_path = path.with_suffix(".tmp.jpg")

            img.save(
                temp_path,
                format="JPEG",
                quality=target_quality,
                optimize=True,
                progressive=True,
            )

            new_size = temp_path.stat().st_size

            # ----------------------------------------------------
            # replace only if beneficial
            # ----------------------------------------------------
            if new_size < original_size:
                temp_path.replace(path)
                return image_path, True

            temp_path.unlink(missing_ok=True)
            return image_path, False

    except UnidentifiedImageError:
        return image_path, False
    except Exception:
        return image_path, False


# ============================================================
# FILE DISCOVERY
# ============================================================


def find_jpg_files(root: Path) -> list[Path]:
    """Recursively find all JPG/JPEG files."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"})


# ============================================================
# BATCH PIPELINE
# ============================================================


def shrink_jpg_folder(
    root: Path,
    quality: int,
    max_size: Optional[int],
    workers: Optional[int],
) -> int:
    """
    Shrink all JPG images in a folder in parallel.

    Returns:
        number of images replaced
    """

    files = find_jpg_files(root)

    if not files:
        logging.warning("No JPG images found in %s", root)
        return 0

    worker_count = workers or os.cpu_count() or 4

    logging.info(
        "Shrinking %d images | quality=%d | max_size=%s | workers=%d",
        len(files),
        quality,
        max_size,
        worker_count,
    )

    replaced = 0

    with ProcessPoolExecutor(max_workers=worker_count) as ex:
        futures = [ex.submit(shrink_jpg, str(f), quality, max_size) for f in files]

        for fut in as_completed(futures):
            _, ok = fut.result()
            if ok:
                replaced += 1

    logging.info("Done: %d / %d images replaced", replaced, len(files))
    return replaced


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shrink JPG images in a folder (parallel).")

    parser.add_argument("-p", "--path", type=Path, default=Path("."))
    parser.add_argument("-q", "--quality", type=int, default=80)
    parser.add_argument("-s", "--max-size", type=int, default=None)
    parser.add_argument("-w", "--workers", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = parse_args()

    if not args.path.is_dir():
        raise ValueError(f"Invalid path: {args.path}")

    if not (1 <= args.quality <= 95):
        raise ValueError("quality must be 1..95")

    shrink_jpg_folder(
        root=args.path,
        quality=args.quality,
        max_size=args.max_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
