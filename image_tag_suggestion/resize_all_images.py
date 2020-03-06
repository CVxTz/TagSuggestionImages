from preprocessing_utilities import (
    read_img_from_path,
    resize_img,
)
from tqdm import tqdm
from glob import glob
import os
import yaml
from imageio import imwrite
import numpy as np
import shutil


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", help="fold", default="validation",  #  oidv6-train
    )
    args = parser.parse_args()

    with open("../example/training_config.yaml", "r") as f:
        training_config = yaml.load(f, yaml.SafeLoader)

    fold = args.fold
    folder = training_config["data_path"] + fold + "_small"
    shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    images = glob(training_config["data_path"] + fold + "/*.jpg")

    for img_path in tqdm(images):
        img = read_img_from_path(img_path)
        img = resize_img(img, h=224, w=224)
        img = img.astype(np.uint8)
        imwrite(img_path.replace(fold, fold + "_small"), img)
