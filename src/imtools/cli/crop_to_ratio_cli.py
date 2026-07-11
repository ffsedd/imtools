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
        "-o",
        default="crop_out",
        help="Output folder (default: ./crop_out)",
    )
    p.add_argument(
        "--ratio",
        "-r",
        type=float,
        default=1.25,
        help="Target width/height ratio (w/h)",
    )
    p.add_argument(
        "--verbose",
        "-v",
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
            logging.warning("Skipping %s: cannot read", fpath.name)
            continue

        if img.ndim not in (2, 3):
            logging.warning("Skipping %s: invalid image shape %s", fpath.name, img.shape)
            continue

        try:
            cropped = crop_to_ratio(img, ratio)
        except Exception as e:
            logging.warning("Skipping %s: crop failed (%s)", fpath.name, e)
            continue

        io.imsave(dst / fpath.name, cropped)
        processed += 1
        logging.info("Saved %s, shape %s", fpath.name, cropped.shape)

    return processed


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    src = Path(args.src).resolve(strict=True)
    dst = Path(args.out).resolve()

    logging.info("Cropping images from %s → %s with ratio %s", src, dst, args.ratio)
    count = process_folder(src, dst, args.ratio)
    logging.info("Batch complete: %d images processed.", count)


if __name__ == "__main__":
    main()
