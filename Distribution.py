#!/usr/bin/env python3
"""Distribution.py

Analyze the class distribution of an image dataset and display it as
a pie chart and a bar chart.

Each immediate subdirectory of the given directory is treated as one
class. The chart title is taken from the directory name.

Usage:
    ./Distribution.py <directory>
"""

import os
import sys

import matplotlib.pyplot as plt

from src.utils import error_exit, list_images


def count_images_per_class(directory):
    """Return a dict {class_name: image_count} for each subdirectory."""
    counts = {}
    try:
        names = sorted(os.listdir(directory))
    except OSError as error:
        error_exit(f"cannot read '{directory}': {error}")
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            counts[name] = len(list_images(path))
    return counts


def plot_distribution(title, counts):
    """Display a pie chart and a bar chart of the class distribution."""
    labels = list(counts.keys())
    values = list(counts.values())

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{title} class distribution")

    ax_pie.pie(values, labels=labels, autopct="%1.1f%%")

    ax_bar.bar(labels, values)
    ax_bar.set_ylabel("Number of images")
    ax_bar.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: ./Distribution.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        error_exit(f"'{directory}' is not a directory")

    counts = count_images_per_class(directory)
    if not counts:
        error_exit(f"no class subdirectories in '{directory}'")

    if sum(counts.values()) == 0:
        error_exit(f"no images found in '{directory}'")

    title = os.path.basename(os.path.normpath(directory))
    plot_distribution(title, counts)


if __name__ == "__main__":
    main()
