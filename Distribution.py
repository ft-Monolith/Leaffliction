#!/usr/bin/env python3
"""CLI entry point for Part 1 (logic in src/distribution.py)."""

import sys

from src.distribution import run


def main():
    if len(sys.argv) != 2:
        print("Usage: ./Distribution.py <directory>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
