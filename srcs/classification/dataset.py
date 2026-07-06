import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from srcs.utils import error_exit, list_classes, IMAGE_EXTENSIONS

IMG_SIZE = 128
BATCH_SIZE = 32

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class LeafDataset(Dataset):

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def list_image_files(class_dir):
    paths = []
    for name in sorted(os.listdir(class_dir)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(os.path.join(class_dir, name))
    return paths


def list_samples(directory):
    classes = list_classes(directory)
    samples = []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        for image_path in list_image_files(class_dir):
            samples.append((image_path, label))
    return classes, samples


def get_n_split(directory, val_ratio=0.2, min_val=100, seed=42):
    classes, samples = list_samples(directory)
    if not samples:
        error_exit(f"no image found in '{directory}'")

    random.Random(seed).shuffle(samples)
    n_val = max(min_val, int(len(samples) * val_ratio))
    n_val = min(n_val, len(samples) - 1)

    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_loader = DataLoader(
        LeafDataset(train_samples, TRANSFORM),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        LeafDataset(val_samples, TRANSFORM),
        batch_size=BATCH_SIZE,
    )
    return train_loader, val_loader, classes
