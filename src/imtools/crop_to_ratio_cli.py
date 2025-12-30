#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from imtools.crop import crop_to_ratio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch center-crop images to a target ratio")
    p.add_argument(
        "src",
        nargs="?",
        default=".",
        help="Source folder containing images (default: current folder)",
    )
    p.add_argument(
        "--out",
        default="crop_out",
        help="Output folder (default: ./crop_out)",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=1.25,
        help="Target width/height ratio (w/h)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args()


def process_folder(src: Path, dst: Path, ratio: float) -> int:
    """Batch crop all JPG images in src to target ratio and save to dst."""
    from skimage import io

    dst.mkdir(parents=True, exist_ok=True)
    processed = 0

    for fpath in sorted(src.glob("*.jpg")):
        try:
            img = io.imread(fpath)
        except Exception:
            logging.warning(f"Skipping {fpath.name}: cannot read")
            continue

        if img.ndim not in (2, 3):
            logging.warning(f"Skipping {fpath.name}: invalid image shape {img.shape}")
            continue

        try:
            cropped = crop_to_ratio(img, ratio)
        except Exception as e:
            logging.warning(f"Skipping {fpath.name}: crop failed ({e})")
            continue

        io.imsave(dst / fpath.name, cropped)
        processed += 1
        logging.info(f"Saved {fpath.name}, shape {cropped.shape}")

    return processed


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    src = Path(args.src).resolve(strict=True)
    dst = Path(args.out).resolve()

    logging.info(f"Cropping images from {src} → {dst} with ratio {args.ratio}")
    count = process_folder(src, dst, args.ratio)
    logging.info(f"Batch complete: {count} images processed.")


if __name__ == "__main__":
    main()
