import argparse
import logging
from pathlib import Path

from imtools.autowb import process_folder


def parse_args():
    p = argparse.ArgumentParser(description="Batch auto WB using imtools")
    p.add_argument("src", nargs="?", default=".", help="Source folder (default: current)")
    p.add_argument("-o", "--out", default="wb_out", help="Output folder (default ./wb_out)")
    p.add_argument("-s", "--strength", type=float, default=1, help="WB correction strength [0-1]")
    p.add_argument("-m", "--max_shift", type=float, default=15.0, help="Maximum LAB shift")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    src = Path(args.src).resolve(strict=True)
    dst = Path(args.out).resolve()

    if not src.is_dir():
        raise ValueError(f"Not a dir: {src}")

    if not 0 <= args.strength <= 1:
        logging.error(f"strength must be in [0,1], got {args.strength}")
        raise ValueError("Invalid strength")

    logging.info(
        f"Processing images from {src} → {dst} strength: {args.strength} max_shift: {args.max_shift}"
    )
    processed = process_folder(src, dst, args.strength, args.max_shift)
    logging.info(f"Batch processing complete. {processed} images processed.")


if __name__ == "__main__":
    main()
