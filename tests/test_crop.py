# tests/test_crop.py
import numpy as np
import pytest

from imtools.crop import crop_to_ratio


def make_image(h, w, c=3):
    """Create a dummy image with given shape."""
    if c == 1:
        return np.zeros((h, w), dtype=np.uint8)  # 2D for grayscale
    return np.zeros((h, w, c), dtype=np.uint8)


@pytest.mark.parametrize(
    "h, w, ratio_expected",
    [
        (100, 100, 1.0),  # square
        (100, 200, 2.0),  # wide
        (200, 100, 0.5),  # tall
    ],
)
def test_crop_ratio_square(h, w, ratio_expected):
    img = make_image(h, w)
    cropped = crop_to_ratio(img, ratio_expected)
    h_c, w_c = cropped.shape[:2]
    np.testing.assert_allclose(w_c / h_c, ratio_expected, rtol=1e-2)


def test_crop_rgb_and_gray():
    # RGB

    img_rgb = make_image(120, 100, 3)
    cropped_rgb = crop_to_ratio(img_rgb, 1.0)
    assert cropped_rgb.shape[2] == 3
    # Grayscale
    img_gray = make_image(120, 100, 1)  # 2D
    cropped_gray = crop_to_ratio(img_gray, 1.0)
    assert cropped_gray.ndim == 2

    assert np.isclose(cropped_gray.shape[1] / cropped_gray.shape[0], 1.0, rtol=1e-2)


def test_no_crop_needed():
    img = make_image(100, 125)
    cropped = crop_to_ratio(img, 1.25)
    # Should be unchanged
    assert cropped.shape == img.shape
