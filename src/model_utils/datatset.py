
import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMG_SIZE = 128
BATCH_SIZE = 32

# aalignement + tenseur 
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class LeafDataset(Dataset):
    """Dataset (chemin, label) qui ouvre l'image et applique un transform."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def _list_samples(directory):
    """Retourne (classes, samples) avec samples = liste (chemin, index)."""
    classes = sorted(
        entry.name for entry in os.scandir(directory) if entry.is_dir()
    )
    samples = []
    for index, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                samples.append((os.path.join(class_dir, name), index))
    return classes, samples


def get_n_split(directory, val_ratio=0.2, min_val=100, seed=42):
    """Charge les images et renvoie (train_loader, val_loader, classes).

    La validation contient au moins ``min_val`` images et n'est jamais
    vue pendant l'entraînement.
    """
    classes, samples = _list_samples(directory)
    if not samples:
        raise ValueError(f"no image found in '{directory}'")

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
