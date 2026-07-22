#!/usr/bin/env python3
"""Image transformations for the Leaffliction data set (Part 3).

Two modes:
  - image path : display the 6 plantCV transformations and the color
    histogram with matplotlib.
  - -src <dir> -dst <dir> [-mask] : save every transformation of every
    image of the source directory into the destination directory.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

try:
    from srcs.utils import error_exit, is_image, list_images
except ModuleNotFoundError:
    from utils import error_exit, is_image, list_images

# Silence plantCV and stop analyze.size from drawing the object id
pcv.params.verbose = False

# 9 channels of figure IV.7: (label, plot color, colorspace, channel index)
# RGB: Rouge(0), Vert(1), Bleu(2)
# HSV: Teinte(0), Saturation(1), Valeur(2)
# LAB: Lightness(0), a=vert-magenta(1), b=bleu-jaune(2)
HIST_CHANNELS = [
    ("blue", "blue", "rgb", 2),
    ("blue-yellow", "gold", "lab", 2),
    ("green", "green", "rgb", 1),
    ("green-magenta", "magenta", "lab", 1),
    ("hue", "purple", "hsv", 0),
    ("lightness", "gray", "lab", 0),
    ("red", "red", "rgb", 0),
    ("saturation", "cyan", "hsv", 1),
    ("value", "orange", "hsv", 2),
]


def read_rgb(path):
    """Read an image as an RGB numpy array (None if unreadable)."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def raw_leaf_mask(rgb):
    """Build a rough black/white leaf mask (before cleaning).

    Uses the LAB 'a' channel, which tells the green/brown leaf apart
    from the neutral background better than any other channel.
    """
    chan_a = pcv.rgb2gray_lab(rgb_img=rgb, channel="a")
    return pcv.threshold.otsu(gray_img=chan_a, object_type="dark")


def clean_mask(raw):
    """Drop small noise blobs but keep disease spots as holes."""
    return pcv.fill(bin_img=raw, size=200)


def leaf_mask(rgb):
    """Return the cleaned binary mask isolating the leaf."""
    return clean_mask(raw_leaf_mask(rgb))


def t_blur(rgb):
    """Figure 2 - gaussian blur of the raw leaf mask.

    Blurring the raw mask smooths its edges while
    keeping the dark disease spots visible as holes.
    """
    raw = raw_leaf_mask(rgb)
    return pcv.gaussian_blur(img=raw, ksize=(5, 5))


def t_mask(rgb, mask):
    """Figure 3 - leaf isolated on a white background."""
    return pcv.apply_mask(img=rgb, mask=mask, mask_color="white")


def t_roi(rgb, mask):
    """Figure 4 - leaf pixels kept inside the ROI, painted green."""
    height, width = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, width - 1, height - 1
    else:
        pad = 5
        x0 = max(int(xs.min()) - pad, 0)
        y0 = max(int(ys.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad, width - 1)
        y1 = min(int(ys.max()) + pad, height - 1)
    roi = pcv.roi.rectangle(img=rgb, x=x0, y=y0, h=y1 - y0, w=x1 - x0)
    kept = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
    out = rgb.copy()
    out[kept != 0] = (0, 255, 0)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 5)
    return out


def t_analyze(rgb, mask):
    """Figure 5 - shape analysis of the leaf object."""
    labeled, n = pcv.create_labels(mask=mask)
    return pcv.analyze.size(
        img=rgb, labeled_mask=labeled, n_labels=n, label="")


def _draw_points(image, points, color):
    """Draw pseudolandmark points as filled circles."""
    if points is None:
        return
    for point in points:
        coords = np.array(point).flatten()
        if coords.size >= 2:
            x, y = int(coords[0]), int(coords[1])
            cv2.circle(image, (x, y), 3, color, -1)


def t_pseudolandmarks(rgb, mask):
    """Figure 6 - top/bottom/left/right pseudolandmark points."""
    out = rgb.copy()
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=rgb, mask=mask)
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
        img=rgb, mask=mask)
    _draw_points(out, top, (0, 0, 255))
    _draw_points(out, bottom, (255, 0, 255))
    _draw_points(out, center_v, (255, 128, 0))
    _draw_points(out, left, (255, 0, 0))
    _draw_points(out, right, (0, 255, 0))
    _draw_points(out, center_h, (0, 255, 255))
    return out


def transformations(rgb):
    """Return the ordered dict {name: image} for figures IV.1 to IV.6."""
    mask = leaf_mask(rgb)
    return {
        "Original": rgb,
        "Gaussian blur": t_blur(rgb),
        "Mask": t_mask(rgb, mask),
        "Roi objects": t_roi(rgb, mask),
        "Analyze object": t_analyze(rgb, mask),
        "Pseudolandmarks": t_pseudolandmarks(rgb, mask),
    }, mask


def _channel(rgb, colorspace, index):
    """Return one channel of the requested colorspace."""
    if colorspace == "rgb":
        return rgb[:, :, index]
    if colorspace == "hsv":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, index]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, index]


def plot_histogram(rgb, mask, ax):
    """Figure 7 - proportion of pixels per intensity, 9 channels."""
    total = max(int(np.count_nonzero(mask)), 1)
    for label, color, colorspace, index in HIST_CHANNELS:
        channel = _channel(rgb, colorspace, index)
        hist = cv2.calcHist([channel], [0], mask, [256], [0, 256]).flatten()
        ax.plot(hist / total * 100.0, color=color, label=label)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_title("Color histogram")
    ax.legend(title="color Channel", fontsize="small")


def _imshow(ax, name, img):
    """Show an image (grayscale if 2D) on the given axis."""
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(name)
    ax.axis("off")


def display_single(path):
    """Display the 6 transformations and the color histogram."""
    rgb = read_rgb(path)
    if rgb is None:
        error_exit(f"cannot read image '{path}'")

    images, mask = transformations(rgb)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, img) in zip(axes.ravel(), images.items()):
        _imshow(ax, name, img)
    fig.tight_layout(pad=2)

    fig_hist, ax_hist = plt.subplots(figsize=(9, 5))
    plot_histogram(rgb, mask, ax_hist)
    fig_hist.tight_layout()

    plt.show()


def save_image(dst, stem, name, img):
    """Save one transformation, converting RGB back to BGR for cv2."""
    tag = name.replace(" ", "_")
    out = os.path.join(dst, f"{stem}_{tag}.JPG")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out, img)


def process_batch(src, dst, save_mask):
    """Save every transformation of every image of src into dst."""
    if not os.path.isdir(src):
        error_exit(f"'{src}' is not a directory")
    images = list_images(src)
    if not images:
        error_exit(f"no images found in '{src}'")
    os.makedirs(dst, exist_ok=True)

    done = 0
    for path in images:
        try:
            rgb = read_rgb(path)
            if rgb is None:
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            results, mask = transformations(rgb)
            for name, img in results.items():
                save_image(dst, stem, name, img)
            if save_mask:
                save_image(dst, stem, "Leaf_mask", mask)

            fig, ax = plt.subplots(figsize=(9, 5))
            plot_histogram(rgb, mask, ax)
            fig.tight_layout()
            fig.savefig(os.path.join(dst, f"{stem}_Color_histogram.JPG"))
            plt.close(fig)
            done += 1
        except Exception as error:
            print(f"  skipped '{os.path.basename(path)}': {error}",
                  file=sys.stderr)
    print(f"Saved transformations of {done}/{len(images)} images into '{dst}'")


def run(image, src, dst, mask):
    """Dispatch between display (single image) and batch (save) modes."""
    if image is not None:
        if os.path.isdir(image):
            error_exit(f"'{image}' is a directory; for a directory use "
                       f"-src {image} -dst <output_dir>")
        if not os.path.isfile(image) or not is_image(image):
            error_exit(f"'{image}' is not a valid image file")
        display_single(image)
    elif src is not None and dst is not None:
        process_batch(src, dst, mask)
    else:
        error_exit("provide an image path, or -src <dir> -dst <dir>")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="Transformation.py",
        description="Display or save 6 leaf image transformations.")
    parser.add_argument("image", nargs="?",
                        help="single image to display")
    parser.add_argument("-src", help="source directory (batch mode)")
    parser.add_argument("-dst", help="destination directory (batch mode)")
    parser.add_argument("-mask", action="store_true",
                        help="also save the isolated-leaf mask")
    return parser


def main():
    args = build_parser().parse_args()
    run(args.image, args.src, args.dst, args.mask)


if __name__ == "__main__":
    main()
