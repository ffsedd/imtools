import argparse
import logging
from pathlib import Path

from .autowb import process_folder


def parse_args():
    p = argparse.ArgumentParser(description="Batch auto WB using imtools")
    p.add_argument("src", nargs="?", default=".", help="Source folder (default: current)")
    p.add_argument("--out", default="wb_out", help="Output folder (default ./wb_out)")
    p.add_argument("--strength", type=float, default=1, help="WB correction strength [0-1]")
    p.add_argument("--max_shift", type=float, default=15.0, help="Maximum LAB shift")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    src = Path(args.src).resolve(strict=True)
    dst = Path(args.out).resolve()

    if not 0 <= args.strength <= 1:
        logging.error(f"strength must be in [0,1], got {args.strength}")
        raise ValueError("Invalid strength")

    logging.info(f"Processing images from {src} → {dst}")
    processed = process_folder(src, dst, args.strength, args.max_shift)
    logging.info(f"Batch processing complete. {processed} images processed.")


if __name__ == "__main__":
    main()
