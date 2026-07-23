import argparse
import os

from srcs import Distribution, Augmentation, Transformation
from srcs.classification import (
    train, predict, evaluate, mask_dataset, zip
)
from srcs.utils import error_exit

DEFAULT_MODEL = os.path.join("models", "model.pth")
DEFAULT_ZIP = "directory.zip"


def build_parser():
    parser = argparse.ArgumentParser(prog="Leaffliction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dist = sub.add_parser("distribution", help="plot class distribution")
    p_dist.add_argument("directory", help="dataset path")

    p_aug = sub.add_parser("augment", help="augment an image or balance a set")
    p_aug.add_argument("path", help="image file or dataset directory")

    p_tr = sub.add_parser("transform", help="transform image(s)")
    p_tr.add_argument("image", nargs="?", help="single image to display")
    p_tr.add_argument("-src", help="source directory (batch mode)")
    p_tr.add_argument("-dst", help="destination directory (batch mode)")
    p_tr.add_argument("-mask", action="store_true", help="also save the mask")

    p_train = sub.add_parser("train", help="train the model")
    p_train.add_argument("directory", help="dataset path")

    p_pred = sub.add_parser("predict", help="predict an image")
    p_pred.add_argument("image", help="image path")
    p_pred.add_argument("model", nargs="?", default=DEFAULT_MODEL)

    p_eval = sub.add_parser("evaluate", help="evaluate the model")
    p_eval.add_argument("zip", nargs="?", default=DEFAULT_ZIP)

    p_mask = sub.add_parser("mask", help="generate a masked dataset")
    p_mask.add_argument("src", help="source dataset directory")
    p_mask.add_argument("dst", help="output masked dataset directory")

    p_pkg = sub.add_parser("zip", help="zip model + dataset + signature")
    p_pkg.add_argument("directory", help="dataset to include in the zip")
    p_pkg.add_argument("model", nargs="?", default=DEFAULT_MODEL)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "distribution":
        Distribution.run(args.directory)
    elif args.command == "augment":
        Augmentation.run(args.path)
    elif args.command == "transform":
        Transformation.run(args.image, args.src, args.dst, args.mask)
    elif args.command == "train":
        if not os.path.isdir(args.directory):
            error_exit(f"'{args.directory}' is not a directory")
        train.run(args.directory)
    elif args.command == "predict":
        predict.run(args.image, args.model)
    elif args.command == "evaluate":
        if not os.path.isfile(args.zip):
            error_exit(f"'{args.zip}' is not a valid file")
        evaluate.run(args.zip)
    elif args.command == "mask":
        mask_dataset.run(args.src, args.dst)
    elif args.command == "zip":
        zip.run(args.model, args.directory)


if __name__ == "__main__":
    main()
