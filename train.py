import sys

from setup import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train", *sys.argv[1:]]
    main()
