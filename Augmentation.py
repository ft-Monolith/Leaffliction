import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import error_exit, is_image

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


def display(results):
    """Show the original image and its augmentations in a single row."""
    fig, axes = plt.subplots(1, len(results), figsize=(3 * len(results), 3))
    for ax, (name, img) in zip(axes, results.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py <image_path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path) or not is_image(path):
        error_exit(f"'{path}' is not a valid image file")

    img = cv2.imread(path)
    if img is None:
        error_exit(f"cannot read image '{path}'")

    base, ext = os.path.splitext(path)
    results = {"Original": img}
    for name, func in AUGMENTATIONS.items():
        augmented = func(img)
        results[name] = augmented
        cv2.imwrite(f"{base}_{name}{ext}", augmented)

    display(results)


if __name__ == "__main__":
    main()
