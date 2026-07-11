from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Load an image from disk.
    
    Parameters
    ----------
    path : Path
        Path to the image file
    flags : int
        OpenCV flags for imread (default: cv2.IMREAD_COLOR)
        
    Returns
    -------
    np.ndarray
        Image in RGB format
        
    Raises
    ------
    FileNotFoundError
        If the image cannot be read
    ValueError
        If the image is corrupted or has invalid format
    """
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), img_bgr)
    if not success:
        raise IOError(f"Failed to save image to {path}")
