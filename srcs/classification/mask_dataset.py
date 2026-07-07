import os

import cv2

from srcs.utils import error_exit, list_classes, list_images
from srcs.Transformation import read_rgb, leaf_mask, t_mask


def mask_one(path):
    rgb = read_rgb(path)
    if rgb is None:
        return None
    return t_mask(rgb, leaf_mask(rgb))


def run(src, dst):
    classes = list_classes(src)
    if not classes:
        error_exit(f"no class subdirectory in '{src}'")

    total = 0
    for class_name in classes:
        dst_dir = os.path.join(dst, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        for path in list_images(os.path.join(src, class_name)):
            masked = mask_one(path)
            if masked is None:
                continue
            out = os.path.join(dst_dir, os.path.basename(path))
            cv2.imwrite(out, cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
            total += 1
        print(f"  {class_name}: done")
    print(f"Saved {total} masked images into '{dst}'")
