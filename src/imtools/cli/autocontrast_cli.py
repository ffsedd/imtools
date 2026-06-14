from __future__ import annotations

import argparse
import logging
from pathlib import Path

from imtools.autocontrast import process_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch auto-contrast JPG images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "src",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Input directory",
    )
    parser.add_argument(
        "-s", "--strength", type=float, default=1, help="Auto-contrast strength [0-1]"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("."),
        help="Output directory. If identical to the source, originals are backed up as *.bak before being overwritten.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v, -vv).",
    )

    return parser.parse_args()


def configure_logging(verbosity: int) -> None:
    level = (
        logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    )

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    src = args.src.resolve(strict=True)
    dst = args.out.resolve()

    if not src.is_dir():
        logging.error("%s is not a directory.", src)
        return 1

    logging.info("Input : %s", src)
    logging.info("Output: %s", dst)

    processed = process_folder(src, dst, strength=args.strength)

    logging.info("Processed %d image(s).", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
