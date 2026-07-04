"""Prédiction de la maladie d'une feuille + affichage (Partie 4)."""

import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.classification.dataset import TRANSFORM
from src.classification.cnn import load_model
from src.utils import error_exit, list_classes, IMAGE_EXTENSIONS

DATA_DIR = "images"


def predict(model, classes, image):
    tensor = TRANSFORM(image)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        index = int(logits.argmax(dim=1))
    return classes[index], tensor


def show(original, transformed, predicted):
    transformed_img = transformed.permute(1, 2, 0).numpy()

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("=== DL classification ===", fontsize=14)
    ax_left.imshow(original)
    ax_left.set_title("Original")
    ax_left.axis("off")
    ax_right.imshow(transformed_img)
    ax_right.set_title("Transformee")
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

    try:
        original = Image.open(image_path).convert("RGB")
    except Exception:
        error_exit(f"cannot open image '{image_path}'")

    predicted, transformed = predict(model, classes, original)

    print(f"Class predicted : {predicted}")
    show(original, transformed, predicted)
