

import os
import sys

import torch
import torch.nn as nn

from src.model_utils.datatset import get_n_split
from src.model_utils.CNN import make_CNN

LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 3  # nb d'epochs sans amelioration avant d'arreter


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    if len(sys.argv) != 2:
        print("Usage: ./train.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"'{directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # charge les images, les trie, les split, les met en tenseurs
    train_loader, val_loader, classes = get_n_split(directory)
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_CNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()  # calcul de la loss
    # Adam : met à jour les poids du modèle
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_path = os.path.join("models", "model.pth")
    os.makedirs("models", exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
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


if __name__ == "__main__":
    main()
