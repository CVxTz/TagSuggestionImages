import os
import shutil
from glob import glob

import numpy as np
import yaml
from imageio import imwrite
from tqdm import tqdm

from preprocessing_utilities import (
    read_img_from_path,
    resize_img,
)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", help="fold", default="train_c",
    )
    args = parser.parse_args()

    with open("../example/training_config.yaml", "r") as f:
        training_config = yaml.load(f, yaml.SafeLoader)

    fold = args.fold
    folder = training_config["data_path"] + fold + "_small"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

    images = glob(training_config["data_path"] + fold + "/*.jpg")

    for img_path in tqdm(images):
        img = read_img_from_path(img_path)
        img = resize_img(img, h=224, w=224)
        img = img.astype(np.uint8)
        imwrite(img_path.replace(fold, fold + "_small"), img)
