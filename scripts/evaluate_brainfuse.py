"""Evaluation entrypoint placeholder for BrainFuse-LLM."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        "Full benchmark evaluation requires released checkpoints and processed "
        f"annotations. Requested checkpoint: {args.checkpoint}; task: {args.task}"
    )


if __name__ == "__main__":
    main()
