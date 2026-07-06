"""Évaluation du modèle sur le set de validation (Partie 4)."""

import os
import zipfile

import torch

from srcs.classification.dataset import get_n_split
from srcs.classification.cnn import get_device, load_model
from srcs.utils import error_exit

EXTRACT_DIR = "unzipped"
MODEL_NAME = "model.pth"
IMAGES_NAME = "images"


def extract_zip(zip_path, extract_dir):
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
    except zipfile.BadZipFile:
        error_exit(f"'{zip_path}' is not a valid zip file")
    print(f"Zip extracted to {extract_dir}/")


def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def run(zip_path):
    extract_zip(zip_path, EXTRACT_DIR)
    model_path = os.path.join(EXTRACT_DIR, MODEL_NAME)
    images_dir = os.path.join(EXTRACT_DIR, IMAGES_NAME)

    if not os.path.isfile(model_path):
        error_exit(f"'{MODEL_NAME}' not found in the zip")

    _, val_loader, classes = get_n_split(images_dir)

    device = get_device()
    model = load_model(model_path, len(classes), device)

    acc = compute_accuracy(model, val_loader, device)
    n = len(val_loader.dataset)
    print(f"Validation : {n} images | accuracy : {acc * 100:.2f}%")
