import os
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.model.dataset import TRANSFORM
from src.model.cnn import make_CNN

DATA_DIR = "images"
DEFAULT_MODEL = os.path.join("models", "model.pth")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_classes(directory):
    return sorted(
        entry.name for entry in os.scandir(directory) if entry.is_dir()
    )


def load_model(model_path, num_classes):
    if not os.path.isfile(model_path):
        print(f"Error: model '{model_path}' introuvable", file=sys.stderr)
        sys.exit(1)
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


def main():
    if len(sys.argv) < 2:
        print("Usage: ./predict.py <image> [model.pth]", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    if not os.path.isfile(image_path) or \
            not image_path.lower().endswith(IMAGE_EXTENSIONS):
        print(f"Error: '{image_path}' n'est pas une image",
              file=sys.stderr)
        sys.exit(1)

    classes = get_classes(DATA_DIR)
    model = load_model(model_path, len(classes))

    original = Image.open(image_path).convert("RGB")
    predicted, transformed = predict(model, classes, original)

    print(f"Class predicted : {predicted}")
    show(original, transformed, predicted)


if __name__ == "__main__":
    main()
