from __future__ import annotations

import numpy as np


def crop_to_ratio(img: np.ndarray, ratio: float = 1.25) -> np.ndarray:
    """
    Crop the center of an image to a target width/height ratio.

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W, C) or (H, W)
    ratio : float
        Target width/height ratio (w/h)

    Returns
    -------
    np.ndarray
        Cropped image
    """
    h, w = img.shape[:2]
    current_ratio = w / h
    if np.isclose(current_ratio, ratio, atol=1e-3):
        return img  # already correct ratio

    if current_ratio < ratio:
        # image too tall → crop height
        new_h = int(w / ratio)
        y0 = (h - new_h) // 2
        y1 = y0 + new_h
        return img[y0:y1, ...]
    else:
        # image too wide → crop width
        new_w = int(h * ratio)
        x0 = (w - new_w) // 2
        x1 = x0 + new_w
        return img[:, x0:x1, ...]
