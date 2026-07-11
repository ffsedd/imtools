from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from imtools.imgio import load_image, save_image


def test_save_and_load_image(tmp_path: Path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Red in RGB
    path = tmp_path / "red.png"
    save_image(path, img)
    assert path.exists()
    loaded = load_image(path)
    assert loaded.shape == (10, 10, 3)
    # Check that red channel is highest (since cv2 converts BGR<->RGB)
    assert loaded[..., 0].mean() > loaded[..., 2].mean()


def test_load_nonexistent(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.jpg")


def test_save_invalid_shape(tmp_path: Path):
    img = np.zeros((10, 10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        save_image(tmp_path / "bad.png", img)
