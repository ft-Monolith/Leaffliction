"""Prédiction de la maladie d'une feuille + affichage (Partie 4)."""

import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.classification.dataset import TRANSFORM
from src.classification.cnn import make_CNN
from src.utils import error_exit

DATA_DIR = "images"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_classes(directory):
    return sorted(
        entry.name for entry in os.scandir(directory) if entry.is_dir()
    )


def load_model(model_path, num_classes):
    if not os.path.isfile(model_path):
        error_exit(f"model '{model_path}' missing")
    model = make_CNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


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

    classes = get_classes(DATA_DIR)
    model = load_model(model_path, len(classes))

    original = Image.open(image_path).convert("RGB")
    predicted, transformed = predict(model, classes, original)

    print(f"Class predicted : {predicted}")
    show(original, transformed, predicted)
