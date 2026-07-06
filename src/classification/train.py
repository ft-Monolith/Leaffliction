"""Entraînement du CNN + création du zip et de la signature (Partie 4)."""

import hashlib
import os
import zipfile

import torch
import torch.nn as nn

from src.classification.dataset import get_n_split
from src.classification.cnn import make_CNN, get_device

LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 3  # nb d'epochs sans amelioration avant d'arreter
ZIP_PATH = "leaffliction.zip"
SIGNATURE_PATH = "signature.txt"


def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    return val_acc


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_acc = correct / total
    return epoch_acc


def make_zip(model_path, data_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, os.path.basename(model_path))
        for root, _, files in os.walk(data_dir):
            for name in files:
                full = os.path.join(root, name)
                archive.write(full, full)
    print(f"Zip created : {zip_path}")


def write_signature(zip_path, signature_path):
    sha1 = hashlib.sha1()
    with open(zip_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha1.update(chunk)
    digest = sha1.hexdigest()
    with open(signature_path, "w") as handle:
        handle.write(digest + "\n")
    print(f"signature sha1 : {digest}")
    return digest


def run(directory):
    # charge les images, les trie, les split, les met en tenseurs
    train_loader, val_loader, classes = get_n_split(directory)
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)}")

    device = get_device()
    model = make_CNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()  # calcul de la loss
    # Adam : met à jour les poids du modèle
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_path = os.path.join("models", "model.pth")
    os.makedirs("models", exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_acc = evaluate(
            model, val_loader, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"best model saved ({val_acc:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved in {model_path}")

    make_zip(model_path, directory, ZIP_PATH)
    write_signature(ZIP_PATH, SIGNATURE_PATH)
