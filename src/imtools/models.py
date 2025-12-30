import numpy as np
from numpy.typing import NDArray

RGBImageUInt8 = NDArray[np.uint8]  # shape (H, W, 3), 0–255
RGBImageFloat32 = NDArray[np.float32]  # shape (H, W, 3), 0–1 or 0–255

GrayImageUInt8 = NDArray[np.uint8]  # shape (H, W)
GrayImageFloat32 = NDArray[np.float32]  # shape (H, W)
