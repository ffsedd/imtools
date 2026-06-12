from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from skimage import exposure, io

from .models import RGBImageUInt8

# -------------------------
# Core image processing
# -------------------------


def autocontrast(
    img: RGBImageUInt8,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    cutoff: float = 0.5,
    gain: float = 8.0,
) -> RGBImageUInt8:
    """
    Percentile-based auto-contrast + sigmoid tone shaping.
    """

    if img.dtype != np.uint8:
        raise TypeError("Expected uint8 image")

    img_f = img.astype(np.float32) / 255.0
    out = np.empty_like(img_f)

    for c in range(3):
        ch = img_f[..., c]

        lo, hi = np.percentile(ch, (low_percentile, high_percentile))

        if hi <= lo:
            out[..., c] = ch
            continue

        ch = exposure.rescale_intensity(
            ch,
            in_range=(lo, hi),  # type: ignore
            out_range=(0.0, 1.0),  # type: ignore
        )

        ch = exposure.adjust_sigmoid(
            ch,
            cutoff=cutoff,
            gain=gain,  # type: ignore
        )

        out[..., c] = ch

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# -------------------------
# File handling
# -------------------------

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}


def collect_images(src: Path, recursive: bool = False) -> list[Path]:
    """
    Fast deterministic collection (no decoding).
    """
    it = src.rglob("*") if recursive else src.iterdir()

    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# -------------------------
# Backup handling
# -------------------------


def _make_backup(path: Path) -> Path:
    """
    Create a safe backup path without overwriting existing backups.
    """
    base = path.with_suffix(path.suffix + ".bak")

    if not base.exists():
        return base

    i = 1
    while True:
        candidate = path.with_suffix(f"{path.suffix}.bak{i}")
        if not candidate.exists():
            return candidate
        i += 1


# -------------------------
# Pipeline
# -------------------------


def process_folder(
    src: Path,
    dst: Path,
    recursive: bool = False,
) -> int:
    """
    Batch auto-contrast images.

    If src == dst:
        - originals are backed up as .bak / .bak1 / .bak2 ...
        - images are overwritten in place

    Returns
    -------
    int
        Number of successfully processed images
    """

    same_folder = src.resolve() == dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    fpaths = collect_images(src, recursive=recursive)
    logging.info("Found %d images", len(fpaths))

    processed = 0

    for fpath in fpaths:
        try:
            img = io.imread(fpath)

            if img.ndim != 3 or img.shape[2] != 3:
                logging.warning("Skip non-RGB: %s", fpath.name)
                continue

            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 255)).astype(np.uint8)

            out = autocontrast(img)

            if same_folder:
                backup = _make_backup(fpath)
                shutil.copy2(fpath, backup)
                logging.debug("Backup created: %s", backup.name)
                out_path = fpath
            else:
                out_path = dst / fpath.name

            io.imsave(out_path, out)
            processed += 1

            logging.info("Processed %s", fpath.name)

        except Exception:
            logging.exception("Failed: %s", fpath)

    return processed
