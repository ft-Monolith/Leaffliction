import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from srcs.classification.dataset import TRANSFORM
from srcs.classification.cnn import load_model
from srcs.utils import error_exit, list_classes, IMAGE_EXTENSIONS
from srcs.Transformation import read_rgb, transformations

DATA_DIR = "images"


def predict(model, classes, image):
    tensor = TRANSFORM(image)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        index = int(logits.argmax(dim=1))
    return classes[index], tensor


def show(original, transformed, predicted):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("=== DL classification ===", fontsize=14)
    ax_left.imshow(original)
    ax_left.set_title("Original")
    ax_left.axis("off")
    if transformed.ndim == 2:
        ax_right.imshow(transformed, cmap="gray")
    else:
        ax_right.imshow(transformed)
    ax_right.set_title("Transformed")
    ax_right.axis("off")
    fig.text(0.5, 0.04, f"Class predicted : {predicted}",
             ha="center", fontsize=14, color="green")
    plt.show()


def run(image_path, model_path):
    if not os.path.isfile(image_path) or \
            not image_path.lower().endswith(IMAGE_EXTENSIONS):
        error_exit(f"'{image_path}' is not a valid image file")

    classes = list_classes(DATA_DIR)
    if not classes:
        error_exit(f"no class subdirectory in '{DATA_DIR}'")

    model = load_model(model_path, len(classes))

    rgb = read_rgb(image_path)
    if rgb is None:
        error_exit(f"cannot open image '{image_path}'")

    predicted, _ = predict(model, classes, Image.fromarray(rgb))
    images, _ = transformations(rgb)

    print(f"Class predicted : {predicted}")
    show(rgb, images["Mask"], predicted)
