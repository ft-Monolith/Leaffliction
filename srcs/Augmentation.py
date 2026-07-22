#!/usr/bin/env python3
"""Data augmentation for the Leaffliction data set (Part 2).

Two modes:
  - image path : display the 6 augmentations and save them next to the
    original file (as shown in the subject).
  - directory  : balance every class (subdirectory) by generating new
    augmented images until each class matches the largest one.
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from srcs.utils import error_exit, is_image, list_classes, list_images
except ModuleNotFoundError:
    from utils import error_exit, is_image, list_classes, list_images

BORDER = cv2.BORDER_REFLECT


def flip(img):
    """Mirror the image horizontally."""
    return cv2.flip(img, 1)


def rotate(img, angle=25):
    """Rotate the image around its center."""
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (width, height), borderMode=BORDER)


def skew(img):
    """Apply a perspective transform (skew)."""
    height, width = img.shape[:2]
    src = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst = np.float32([[width * 0.15, 0], [width * 0.85, 0],
                      [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height),
                               borderMode=BORDER)


def shear(img, factor=0.2):
    """Slant the image along the horizontal axis."""
    height, width = img.shape[:2]
    matrix = np.float32([[1, factor, 0], [0, 1, 0]])
    return cv2.warpAffine(img, matrix, (width, height), borderMode=BORDER)


def crop(img, scale=0.8):
    """Crop a centered region and resize it back to the original size."""
    height, width = img.shape[:2]
    new_h, new_w = max(1, int(height * scale)), max(1, int(width * scale))
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (width, height))


def distortion(img, amplitude=8, period=30.0):
    """Apply a wavy distortion by remapping pixel coordinates."""
    height, width = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (xs + amplitude * np.sin(ys / period)).astype(np.float32)
    map_y = (ys + amplitude * np.sin(xs / period)).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=BORDER)


AUGMENTATIONS = {
    "Flip": flip,
    "Rotate": rotate,
    "Skew": skew,
    "Shear": shear,
    "Crop": crop,
    "Distortion": distortion,
}

AUG_SUFFIXES = tuple(f"_{name}" for name in AUGMENTATIONS)


def is_augmented(path):
    """Return True if the file was produced by a previous augmentation."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.endswith(AUG_SUFFIXES)


def augment_image(path):
    """Augment one image, saving each result next to it.

    Returns a dict {name: image} including the original, or None if the
    image cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        return None

    base, ext = os.path.splitext(path)
    results = {"Original": img}
    for name, func in AUGMENTATIONS.items():
        augmented = func(img)
        results[name] = augmented
        cv2.imwrite(f"{base}_{name}{ext}", augmented)
    return results


def display(results):
    """Show the original image and its augmentations in a single row."""
    fig, axes = plt.subplots(1, len(results), figsize=(3 * len(results), 3))
    for ax, (name, img) in zip(axes, results.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout(pad=2)
    plt.show()


def balance_class(class_dir, target):
    """Create augmented images until the class holds `target` files.

    Only original (non-augmented) images are used as sources. Each pass
    gives every source one augmentation; the offset shifts the chosen
    type each pass so a source never gets the same type twice (hence no
    duplicate file names). Stops as soon as `target` is reached, skips
    files already on disk, and returns the number of images created.
    """
    images = list_images(class_dir)
    need = target - len(images)
    if need <= 0:
        return 0

    sources = [p for p in images if not is_augmented(p)]
    types = list(AUGMENTATIONS.items())
    created = 0
    for offset in range(len(types)):
        for index, path in enumerate(sources):
            if created == need:
                return created
            name, func = types[(index + offset) % len(types)]
            base, ext = os.path.splitext(path)
            out = f"{base}_{name}{ext}"
            if os.path.exists(out):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            cv2.imwrite(out, func(img))
            created += 1

    if created < need:
        print(f"  warning: only {created}/{need} images could be created "
              f"for '{os.path.basename(class_dir)}'", file=sys.stderr)
    return created


def balance_directory(directory):
    """Balance every class in `directory` to the size of the largest."""
    classes = list_classes(directory)
    if not classes:
        error_exit(f"'{directory}' contains no class subdirectory")

    counts = {}
    for name in classes:
        counts[name] = len(list_images(os.path.join(directory, name)))
    target = max(counts.values())

    print(f"Balancing '{directory}' to {target} images per class")
    for name in classes:
        created = balance_class(os.path.join(directory, name), target)
        total = counts[name] + created
        print(f"  {name}: {counts[name]} -> {total} (+{created})")


def run(path):
    """Augment a single image (display) or balance a whole directory."""
    if os.path.isdir(path):
        balance_directory(path)
        return

    if not os.path.isfile(path) or not is_image(path):
        error_exit(f"'{path}' is not a valid image file")

    results = augment_image(path)
    if results is None:
        error_exit(f"cannot read image '{path}'")
    display(results)


def main():
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py <image_path | directory>",
              file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
