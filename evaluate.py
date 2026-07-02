import os
import sys
import zipfile

import torch

from src.model.dataset import get_n_split
from src.model.cnn import make_CNN

DEFAULT_ZIP = "leaffliction.zip"
EXTRACT_DIR = "unzipped"
MODEL_NAME = "model.pth"
IMAGES_NAME = "images"


def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    print(f"Zip extrait dans {extract_dir}/")


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


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP
    if not os.path.isfile(zip_path):
        print(f"Error: '{zip_path}' introuvable", file=sys.stderr)
        sys.exit(1)

    extract_zip(zip_path, EXTRACT_DIR)
    model_path = os.path.join(EXTRACT_DIR, MODEL_NAME)
    images_dir = os.path.join(EXTRACT_DIR, IMAGES_NAME)

    _, val_loader, classes = get_n_split(images_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_CNN(len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    acc = compute_accuracy(model, val_loader, device)
    n = len(val_loader.dataset)
    print(f"Validation : {n} images | accuracy : {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
