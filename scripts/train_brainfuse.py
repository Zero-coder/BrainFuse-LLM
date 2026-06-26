"""Training entrypoint placeholder for BrainFuse-LLM."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        "Full LLM training requires the processed NSD/VQA resources and model "
        "checkpoints. This public entrypoint records the intended interface. "
        f"Config: {args.config}; data root: {args.data_root}; output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
