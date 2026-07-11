from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage import color, io

from .models import RGBImageUInt8  # shape (H, W, 3), uint8


def autowb(
    img: RGBImageUInt8,
    strength: float = 0.5,
    max_shift: float = 15.0,
) -> Tuple[RGBImageUInt8, float, float]:
    """
    Auto white-balance an RGB image using Lab midtone neutralization.

    Parameters
    ----------
    img : uint8 RGB image, shape (H, W, 3)
    strength : scaling of correction [0..1]
    max_shift : clamp for a/b correction in Lab units

    Returns
    -------
    corrected_img : uint8 RGB image
    a_shift : applied shift in Lab a channel
    b_shift : applied shift in Lab b channel
    """
    if img.dtype != np.uint8:
        raise TypeError("Expected uint8 RGB image")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3)")

    # Convert RGB [0..255] → float [0..1]
    rgb = img.astype(np.float32) / 255.0

    # RGB → Lab
    lab = color.rgb2lab(rgb)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # Midtone mask (same semantics as your OpenCV version)
    mask = (L > 20.0) & (L < 80.0)
    if np.count_nonzero(mask) < 1000:
        raise ValueError("Not enough valid pixels for WB estimation")

    a_mean = float(a[mask].mean())
    b_mean = float(b[mask].mean())

    a_shift = float(np.clip(a_mean * strength, -max_shift, max_shift))
    b_shift = float(np.clip(b_mean * strength, -max_shift, max_shift))

    lab[..., 1] -= a_shift
    lab[..., 2] -= b_shift

    # Lab → RGB
    rgb_corr_lab = color.lab2rgb(lab)
    rgb_corr_float = np.clip(rgb_corr_lab, 0.0, 1.0)
    rgb_corr: RGBImageUInt8 = (rgb_corr_float * 255).astype(np.uint8)  # explicit cast
    return rgb_corr, a_shift, b_shift


def process_folder(
    src: Path,
    dst: Path,
    strength: float = 0.5,
    max_shift: float = 15.0,
) -> int:
    """
    Batch process all JPG images in a folder.

    Returns
    -------
    int
        Number of successfully processed images
    """
    from .autocontrast import collect_images

    dst.mkdir(parents=True, exist_ok=True)
    fpaths = sorted(src.glob("*.jpg"))
    logging.info("Processing %d files", len(fpaths))
    
    processed = 0
    for fpath in fpaths:
        try:
            img = io.imread(fpath)
        except Exception:
            logging.warning("Skipping %s: cannot read", fpath.name)
            continue

        if img.ndim != 3 or img.shape[2] != 3:
            logging.warning("Skipping %s: not RGB", fpath.name)
            continue

        try:
            corrected, a_shift, b_shift = autowb(
                img,
                strength=strength,
                max_shift=max_shift,
            )
        except ValueError:
            logging.warning("Skipping %s: not enough midtones", fpath.name)
            continue

        io.imsave(dst / fpath.name, corrected)
        logging.info(
            "%s: applied a=%.2f, b=%.2f",
            fpath.name,
            a_shift,
            b_shift,
        )
        processed += 1
    
    return processed
