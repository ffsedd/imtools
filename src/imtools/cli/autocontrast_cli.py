from __future__ import annotations

import argparse
import logging
from pathlib import Path

from imtools.autocontrast import process_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch auto-contrast images.",
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
        "-o",
        "--out",
        type=Path,
        default=Path("."),
        help="Output directory (same as src = in-place with backups)",
    )

    parser.add_argument(
        "-s",
        "--strength",
        type=float,
        default=1.0,
        help="Contrast strength multiplier (affects sigmoid gain)",
    )

    parser.add_argument(
        "--low",
        type=float,
        default=1.0,
        help="Lower percentile cutoff (0–100)",
    )

    parser.add_argument(
        "--high",
        type=float,
        default=99.0,
        help="Upper percentile cutoff (0–100)",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively",
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
        logging.WARNING
        if verbosity == 0
        else logging.INFO
        if verbosity == 1
        else logging.DEBUG
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

    if not (0 <= args.low < args.high <= 100):
        logging.error("Invalid percentiles: require 0 <= low < high <= 100")
        return 2

    logging.info("Input : %s", src)
    logging.info("Output: %s", dst)
    logging.info("Percentiles: %.2f / %.2f", args.low, args.high)

    processed = process_folder(
        src,
        dst,
        strength=args.strength,
        low_percentile=args.low,
        high_percentile=args.high,
        recursive=args.recursive,
    )

    logging.info("Processed %d image(s).", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
