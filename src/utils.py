"""Shared helpers for the Leaffliction project."""

import os
import sys

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def error_exit(message):
    """Print an error message to stderr and exit with status 1."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def is_image(filename):
    """Return True if the filename has a known image extension."""
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def list_images(directory):
    """Return the list of image file paths directly inside a directory.

    Returns an empty list if the directory cannot be read.
    """
    images = []
    try:
        names = os.listdir(directory)
    except OSError:
        return images
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path) and is_image(name):
            images.append(path)
    return images
