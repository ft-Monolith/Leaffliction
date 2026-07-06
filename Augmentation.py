#!/usr/bin/env python3
"""CLI entry point for Part 2 (logic in src/augmentation.py)."""

import sys

from src.augmentation import run


def main():
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py <image_path | directory>",
              file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
