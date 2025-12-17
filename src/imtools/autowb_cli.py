from pathlib import Path
import argparse
import logging
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

    src = Path(args.src).resolve()
    dst = Path(args.out).resolve()

    if not src.is_dir():
        logging.error(f"Source folder does not exist: {src}")
        raise NotADirectoryError(src)

    logging.info(f"Processing images from {src} → {dst}")
    process_folder(src, dst, args.strength, args.max_shift)
    logging.info("Batch processing complete.")


if __name__ == "__main__":
    main()

