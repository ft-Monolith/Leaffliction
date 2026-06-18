"""Shared helpers for the Leaffliction project."""

import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_image(filename):
    """Return True if the filename has a known image extension."""
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def list_images(directory):
    """Return the list of image file paths directly inside a directory."""
    images = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and is_image(name):
            images.append(path)
    return images
