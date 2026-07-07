import sys

from setup import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "predict", *sys.argv[1:]]
    main()
