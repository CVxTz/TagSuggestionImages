import argparse
import json

import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from preprocessing_utilities import (
    read_img_from_path,
    resize_img,
    read_from_file,
)
from utils import download_model


class ImagePredictor:
    def __init__(
        self, model_path, resize_size, pre_processing_function=preprocess_input
    ):
        self.model_path = model_path
        self.pre_processing_function = pre_processing_function
        self.model = load_model(self.model_path, compile=False)
        self.resize_size = resize_size

    @classmethod
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        predictor = cls(
            model_path=config["image_model_path"], resize_size=config["resize_shape"],
        )
        return predictor

    @classmethod
    def init_from_config_url(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

        download_model(
            config["image_model_model_url"],
            config["image_model_path"],
            config["image_model_sha256"],
        )

        return cls.init_from_config_path(config_path)

    def predict_from_array(self, arr):
        arr = resize_img(arr, h=self.resize_size[0], w=self.resize_size[1])
        arr = self.pre_processing_function(arr)
        pred = self.model.predict(arr[np.newaxis, ...]).ravel()
        return pred

    def predict_from_path(self, path):
        arr = read_img_from_path(path)
        return self.predict_from_array(arr)

    def predict_from_file(self, file_object):
        arr = read_from_file(file_object)
        return self.predict_from_array(arr), arr


class LabelPredictor:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(self.json_path, "r") as f:
            self.labels = json.load(f)
        self.labels_arrays = {k: np.array(v) for k, v in self.labels.items()}

    @classmethod
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        predictor = cls(json_path=config["label_embedding_path"])
        return predictor

    @classmethod
    def init_from_config_url(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

        download_model(
            config["label_embedding_url"],
            config["label_embedding_path"],
            config["label_embedding_sha256"],
        )

        return cls.init_from_config_path(config_path)

    def predict_from_array(self, img_arr):
        preds = [
            (k, 1 - np.sum((img_arr - v) ** 2) / 4)
            for k, v in self.labels_arrays.items()
        ]
        preds.sort(key=lambda x: x[1], reverse=True)

        return preds[:20]

    def predict_dataframe_from_array(self, img_arr):
        preds = [
            (k, 1 - np.sum((img_arr - v) ** 2) / 4)
            for k, v in self.labels_arrays.items()
        ]
        preds.sort(key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(
            {"label": [x[0] for x in preds[:20]], "scores": [x[1] for x in preds[:20]]}
        )

        return df


if __name__ == "__main__":
    """
    python predictor.py --predictor_config "config.yaml"

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor_config_path", help="predictor_config_path", default="config.yaml",
    )

    args = parser.parse_args()

    predictor_config_path = args.predictor_config_path

    predictor = ImagePredictor.init_from_config_url(predictor_config_path)

    pred = predictor.predict_from_path("../example/data/0b44f28fa177010c.jpg")

    label_predictor = LabelPredictor.init_from_config_url(predictor_config_path)

    preds = label_predictor.predict_dataframe_from_array(pred)

    print(preds)
