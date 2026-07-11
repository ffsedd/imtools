from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : Path
        Path to the image file

    Returns
    -------
    np.ndarray
        Image in RGB format

    Raises
    ------
    FileNotFoundError
        If the image cannot be read
    ValueError
        If the image has invalid format
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))

    except Exception as exc:
        raise ValueError(f"Cannot read image {path}") from exc


def save_image(path: Path, img: np.ndarray) -> None:
    """
    Save an image to disk.

    Parameters
    ----------
    path : Path
        Path to save the image to
    img : np.ndarray
        Image in RGB format

    Raises
    ------
    ValueError
        If the image has invalid format
    IOError
        If the image cannot be saved
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"Invalid image shape: {img.shape}")

    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img.dtype}")

    try:
        Image.fromarray(img).save(path)

    except Exception as exc:
        raise OSError(f"Failed to save image to {path}") from exc
