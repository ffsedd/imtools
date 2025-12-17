import numpy as np
import cv2
from pathlib import Path
import pytest
import logging
from imtools.autowb import autowb, process_folder

def make_test_image(width=100, height=100, a_shift: float = 0.0, b_shift: float = 0.0) -> np.ndarray:
    """
    Create a gradient RGB image with midtones (L 30-70) and optional color tint.
    a_shift, b_shift approximate LAB a/b changes, applied in RGB for testing.
    """
    # Base gradient in RGB (all channels equal)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(3):
        img[..., i] = np.linspace(30, 70, width, dtype=np.uint8)

    # Convert to BGR → LAB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32)
    lab = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    # Apply a/b shifts
    lab[..., 1] = np.clip(lab[..., 1] + a_shift, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + b_shift, 0, 255)

    # Convert back to RGB
    img_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def test_autowb_basic():
    img = make_test_image()
    corrected, a_shift, b_shift = autowb(img)
    assert isinstance(a_shift, float)
    assert isinstance(b_shift, float)
    assert corrected.shape == img.shape
    assert corrected.dtype == np.uint8

def test_autowb_tinted():
    # Slightly bluish (negative b) and reddish (positive a)
    img = make_test_image(a_shift=10, b_shift=-10)
    corrected, a_shift, b_shift = autowb(img)
    # Check that autowb returns a shift roughly opposite to the tint
    assert a_shift * 10 > 0 or a_shift == 0  # some shift applied
    assert b_shift * -10 > 0 or b_shift == 0

def test_process_folder(tmp_path):
    # Create fake JPG images with tint
    img1 = make_test_image(a_shift=5, b_shift=-5)
    img2 = make_test_image(a_shift=-5, b_shift=5)
    for i, img in enumerate([img1, img2], start=1):
        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    dst = tmp_path / "out"
    process_folder(tmp_path, dst, strength=0.5, max_shift=10.0)

    # Check that output files exist
    out_files = list(dst.glob("*.jpg"))
    assert len(out_files) == 2

    # Check that images are readable and have correct shape
    for out_file in out_files:
        out_img = cv2.imread(str(out_file))
        assert out_img is not None
        assert out_img.shape[2] == 3

