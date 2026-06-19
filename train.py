

import os
import sys
from PIL import Image
from dataset import get_n_split
from CNN import Make_CNN

def main():
    if len(sys.argv) != 2:
        print("Usage: ./train.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"'{directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    dataset = get_n_split(directory)
    
    CNN = Make_CNN()

if __name__ == "__main__":
    main()
