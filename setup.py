import argparse
import os

from src.classification import train, predict, evaluate
from src.utils import error_exit

DEFAULT_MODEL = os.path.join("models", "model.pth")
DEFAULT_ZIP = "leaffliction.zip"


def build_parser():
    parser = argparse.ArgumentParser(prog="Leaffliction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the model")
    p_train.add_argument("directory", help="dataset path")

    p_pred = sub.add_parser("predict", help="predict an image")
    p_pred.add_argument("image", help="image path")
    p_pred.add_argument("model", nargs="?", default=DEFAULT_MODEL)

    p_eval = sub.add_parser("evaluate", help="evaluate the model")
    p_eval.add_argument("zip", nargs="?", default=DEFAULT_ZIP)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "train":
        if not os.path.isdir(args.directory):
            error_exit(f"'{args.directory}' is not a directory")
        train.run(args.directory)
    elif args.command == "predict":
        predict.run(args.image, args.model)
    elif args.command == "evaluate":
        if not os.path.isfile(args.zip):
            error_exit(f"'{args.zip}' is not a valid file")
        evaluate.run(args.zip)


if __name__ == "__main__":
    main()
