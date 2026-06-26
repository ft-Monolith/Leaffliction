

import os
import sys

import torch
import torch.nn as nn

from src.model_utils.datatset import get_n_split
from src.model_utils.CNN import make_CNN

LEARNING_RATE = 0.001
EPOCHS = 10

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

    train_loader, val_loader, classes = get_n_split(directory) # prend les images les tries les split les format
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu" # gpu ou cpu
    model = make_CNN(len(classes)).to(device) # modele de convolution TEMP Faut le faire nous meme la c est celui de torch
    criterion = nn.CrossEntropyLoss() # calcul de la loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # mise a jour des poids


    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )
    model_path = os.path.join(directory, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()
