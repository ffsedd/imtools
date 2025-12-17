# imtools

`imtools` is a small Python library for image processing utilities, including **automatic white balance (WB)** for RGB images. Designed for batch processing of architecture and surface photography while preserving color fidelity.

---

## Features

- Per-image automatic white balance using LAB midtones.
- Pure Python + OpenCV (`opencv-python-headless`) implementation.
- Batch processing of image folders.
- CLI and library-friendly API.
- Compatible with Python 3.10+.

---

## Installation

Install from PyPI (future release) or directly from source:

```bash
# Editable install for development
git clone https://github.com/ffsedd/imtools.git
cd imtools
python -m pip install -e .
```

Dependencies are automatically installed:

- `numpy`
- `opencv-python-headless`

---

## Usage

### As a library

```python
import cv2
from imtools.autowb import autowb

# Read an image
img = cv2.imread("photo.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply automatic white balance
corrected, a_shift, b_shift = autowb(img_rgb, strength=0.5, max_shift=15.0)
print(f"Applied shifts: a={a_shift:.2f}, b={b_shift:.2f}")

# Save result
corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
cv2.imwrite("photo_wb.jpg", corrected_bgr)
```

### Command-line interface (CLI)

```bash
# Batch process folder
autowb --src ./images --out ./images_wb --strength 0.8 --max-shift 20
```

Options:

- `--src`: source folder (default: current folder)
- `--out`: output folder (default: `wb_out`)
- `--strength`: WB correction strength [0-1]
- `--max-shift`: maximum LAB a/b shift
- `--debug`: enable debug logging

---

## Development

Run tests with `pytest`:

```bash
python -m pip install -e .[dev]  # install dev dependencies
pytest
```

Project structure:

```
imtools/
├── src/imtools/
│   ├── autowb.py      # auto WB functions
│   ├── autowb_cli.py         # command-line interface
│   ├── color.py       # color utilities
│   ├── io.py          # image I/O utilities
│   └── __init__.py
├── tests/
│   └── test_autowb.py
├── pyproject.toml
└── README.md
```

---

## License

MIT License — see `LICENSE` for details.

