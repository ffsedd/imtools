from pathlib import Path
import cv2
import numpy as np

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(path: Path, img: np.ndarray) -> None:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)

