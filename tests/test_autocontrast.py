from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imtools.autocontrast import _make_backup, autocontrast, collect_images, process_folder


def test_autocontrast_basic():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    out = autocontrast(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_autocontrast_invalid_dtype():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        autocontrast(img)


def test_autocontrast_invalid_percentiles():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        autocontrast(img, low_percentile=50, high_percentile=10)


def test_collect_images(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"dummy")
    (tmp_path / "b.png").write_bytes(b"dummy")
    (tmp_path / "c.txt").write_bytes(b"dummy")
    imgs = collect_images(tmp_path)
    assert len(imgs) == 2
    assert all(p.suffix.lower() in {".jpg", ".png"} for p in imgs)


def test_make_backup(tmp_path: Path):
    f = tmp_path / "img.jpg"
    f.write_bytes(b"data")
    b1 = _make_backup(f)
    assert str(b1).endswith(".bak")
    b1.write_bytes(b"backup")
    b2 = _make_backup(f)
    assert str(b2).endswith(".bak1")
    b2.write_bytes(b"backup")
    b3 = _make_backup(f)
    assert str(b3).endswith(".bak2")


def test_process_folder(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    from skimage import io

    io.imsave(src / "test.jpg", img)
    count = process_folder(src, dst)
    assert count == 1
    assert (dst / "test.jpg").exists()
