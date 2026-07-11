from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from imtools.jpgshrink import find_jpg_files, shrink_jpg, shrink_jpg_folder


def _make_jpg(path: Path, size=(100, 100), quality=95):
    img = Image.new("RGB", size, color=(123, 45, 67))
    img.save(path, format="JPEG", quality=quality)


def test_find_jpg_files(tmp_path: Path):
    _make_jpg(tmp_path / "a.jpg")
    _make_jpg(tmp_path / "b.JPEG")
    (tmp_path / "c.png").write_bytes(b"x")
    files = find_jpg_files(tmp_path)
    assert len(files) == 2


def test_shrink_jpg_reduces_size(tmp_path: Path):
    path = tmp_path / "big.jpg"
    # Create a large random image to ensure high quality file is big
    arr = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=95)
    original_size = path.stat().st_size
    _, replaced = shrink_jpg(str(path), target_quality=20, max_size=None)
    assert replaced is True
    assert path.stat().st_size < original_size


def test_shrink_jpg_skip_if_smaller_quality(tmp_path: Path):
    path = tmp_path / "small.jpg"
    _make_jpg(path, quality=10)
    original_size = path.stat().st_size
    _, replaced = shrink_jpg(str(path), target_quality=80, max_size=None)
    # Should not replace because current quality is already lower
    assert replaced is False
    assert path.stat().st_size == original_size


def test_shrink_jpg_folder(tmp_path: Path):
    _make_jpg(tmp_path / "a.jpg", quality=95)
    _make_jpg(tmp_path / "b.jpg", quality=95)
    count = shrink_jpg_folder(tmp_path, quality=10, max_size=None, workers=1)
    assert count == 2
