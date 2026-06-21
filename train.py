

import os
import sys
from src.model_utils.datatset import get_n_split
from src.model_utils.CNN import make_CNN

def main():
    if len(sys.argv) != 2:
        print("Usage: ./train.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"'{directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    train_loader, val_loader, classes = get_n_split(directory)
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    
    CNN = make_CNN()

if __name__ == "__main__":
    main()
