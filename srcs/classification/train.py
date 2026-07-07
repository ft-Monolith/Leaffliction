import os
import shutil

import torch
import torch.nn as nn

from srcs import Augmentation
from srcs.classification import mask_dataset, zip
from srcs.classification.dataset import get_n_split
from srcs.classification.cnn import make_CNN, get_device

LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 3  # how many epochs without improvement to stop training
AUGMENTED_DIR = "augmented_directory"


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
        # tensor to device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # reset gradients
        outputs = model(inputs)  # forward : model prediction
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()  # backward : compute gradients
        optimizer.step()  # update weights

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_acc = correct / total
    return epoch_acc


def prepare_dataset(base_dir):
    if os.path.isdir(AUGMENTED_DIR):
        print(f"'{AUGMENTED_DIR}' already exists, reusing it")
        return AUGMENTED_DIR
    print(f"Building '{AUGMENTED_DIR}' from '{base_dir}'...")
    shutil.copytree(base_dir, AUGMENTED_DIR)
    Augmentation.balance_directory(AUGMENTED_DIR)
    # mask_dataset.run(AUGMENTED_DIR, AUGMENTED_DIR)
    return AUGMENTED_DIR


def run(directory):
    directory = prepare_dataset(directory)
    train_loader, val_loader, classes = get_n_split(directory)
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)}")

    device = get_device()
    print(f"Using device: {device}")
    model = make_CNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()  # loss function
    # Adam : optimizer that adapts the learning rate for each parameter
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

    zip.run(model_path, directory)