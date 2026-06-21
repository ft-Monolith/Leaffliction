

import os
import sys

import torch
import torch.nn as nn

from src.model_utils.datatset import get_n_split
from src.model_utils.CNN import make_CNN

LEARNING_RATE = 0.001
EPOCHS = 10


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
    
    
    # on a tt ce qui faut maintenant faut faire la boucle d entrainement

if __name__ == "__main__":
    main()
