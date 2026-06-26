"""Prepare metadata placeholders for NSD-based BrainFuse-LLM experiments.

This script documents the expected command-line interface. Full preprocessing
depends on local NSD access credentials and third-party annotation files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "NSD preprocessing is environment-specific. See docs/dataset.md for the "
        f"expected metadata schema. Requested NSD root: {args.nsd_root}"
    )


if __name__ == "__main__":
    main()
