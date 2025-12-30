def crop_to_ratio(self, ratio=1.25):
    """Crop center of image to target ratio."""

    h, w = self.height, self.width
    r = self.ratio
    if r == ratio:
        return 1
    print(f"crop ratio {r} to {ratio}")
    if self.ratio < ratio:
        new_h = int(h * r / ratio)
        y0 = (h - new_h) // 2
        y1 = h - y0
        self.arr = self.arr[y0:y1, ...]
    else:
        new_w = int(w * ratio / r)
        x0 = (w - new_w) // 2
        x1 = w - x0
        self.arr = self.arr[:, x0:x1, ...]
