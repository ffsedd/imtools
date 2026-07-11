from __future__ import annotations

import io

from PIL import Image

from imtools.jpgquality import extract_quantization_features, get_jpg_quality


def test_extract_quantization_features():
    qtables = {
        0: list(range(64)),
        1: list(range(64)),
    }
    qsum, qvalue = extract_quantization_features(qtables)
    assert isinstance(qsum, int)
    assert isinstance(qvalue, int)
    assert qsum == sum(range(64)) * 2
    # first[2] + first[53] + second[0] + second[-1]
    assert qvalue == (2 + 53 + 0 + 63)


def test_get_jpg_quality_from_pil():
    # Create a small image and save as JPEG in memory
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    loaded = Image.open(buf)
    quality = get_jpg_quality(loaded)
    assert 1 <= quality <= 100  # Should estimate something


def test_get_jpg_quality_no_qtables():
    img = Image.new("RGB", (10, 10))
    # PNG has no quantization tables
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    loaded = Image.open(buf)
    assert get_jpg_quality(loaded) == -1
