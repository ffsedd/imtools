from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2

def auto_wb(img: np.ndarray, strength: float = 0.5, max_shift: float = 15.0) -> Tuple[np.ndarray, float, float]:
    """
    Pure function: auto white-balance an image (RGB).
    Returns corrected image and applied a,b shifts.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    mask = (L > 20) & (L < 80)
    if np.count_nonzero(mask) < 1000:
        raise ValueError("Not enough valid pixels for WB estimation")

    a_mean = float(a[mask].mean() - 128.0)
    b_mean = float(b[mask].mean() - 128.0)
    a_shift = np.clip(a_mean * strength, -max_shift, max_shift)
    b_shift = np.clip(b_mean * strength, -max_shift, max_shift)

    lab[..., 1] -= a_shift
    lab[..., 2] -= b_shift
    lab = np.clip(lab, 0, 255).astype(np.uint8)

    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    return corrected, a_shift, b_shift

