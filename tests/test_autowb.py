from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import color, io
from skimage.util import img_as_ubyte

from imtools.autowb import autowb, process_folder


def make_test_image(
    width: int = 100,
    height: int = 100,
    a_shift: float = 0.0,
    b_shift: float = 0.0,
) -> np.ndarray:
    """
    Create a midtone RGB image with an optional Lab a/b tint.
    Output is uint8 RGB.
    """
    # Neutral midtone gradient
    x = np.linspace(0.3, 0.7, width, dtype=np.float32)
    rgb = np.stack([x, x, x], axis=1)
    rgb = np.tile(rgb[None, :, :], (height, 1, 1))

    # RGB → Lab
    lab = color.rgb2lab(rgb)

    # Apply tint
    lab[..., 1] += a_shift
    lab[..., 2] += b_shift

    # Lab → RGB
    rgb_tinted = color.lab2rgb(lab)
    rgb_tinted = np.clip(rgb_tinted, 0.0, 1.0)

    return img_as_ubyte(rgb_tinted)


def test_autowb_basic():
    img = make_test_image()

    corrected, a_shift, b_shift = autowb(img)

    assert corrected.shape == img.shape
    assert corrected.dtype == np.uint8
    assert isinstance(a_shift, float)
    assert isinstance(b_shift, float)

    # Neutral image should need almost no correction
    assert abs(a_shift) < 1.0
    assert abs(b_shift) < 1.0


def test_autowb_tinted():
    # Red + blue tint
    img = make_test_image(a_shift=10.0, b_shift=-10.0)

    _, a_shift, b_shift = autowb(img)

    # Correction should oppose the tint
    assert a_shift > 0.0
    assert b_shift < 0.0


def test_process_folder(tmp_path: Path):
    img1 = make_test_image(a_shift=5.0, b_shift=-5.0)
    img2 = make_test_image(a_shift=-5.0, b_shift=5.0)

    io.imsave(tmp_path / "img1.jpg", img1)
    io.imsave(tmp_path / "img2.jpg", img2)

    dst = tmp_path / "out"
    process_folder(tmp_path, dst, strength=0.5, max_shift=10.0)

    out_files = sorted(dst.glob("*.jpg"))
    assert len(out_files) == 2

    for fpath in out_files:
        out_img = io.imread(fpath)
        assert out_img.ndim == 3
        assert out_img.shape[2] == 3
        assert out_img.dtype == np.uint8
