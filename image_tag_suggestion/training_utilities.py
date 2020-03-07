from collections import namedtuple
from pathlib import Path
from random import shuffle, choice
import spacy
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from preprocessing_utilities import (
    read_img_from_path,
    resize_img,
)

SampleFromPath = namedtuple("Sample", ["path", "labels"])
import imgaug.augmenters as iaa


def chunks(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(scale={"x": (0.8, 1.2)})),
            sometimes(iaa.Fliplr(p=0.5)),
            sometimes(iaa.Affine(scale={"y": (0.8, 1.2)})),
            sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2)})),
            sometimes(iaa.Affine(translate_percent={"y": (-0.2, 0.2)})),
            sometimes(iaa.Affine(rotate=(-20, 20))),
            sometimes(iaa.Affine(shear=(-20, 20))),
            sometimes(iaa.AdditiveGaussianNoise(scale=0.07 * 255)),
            sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
        ],
        random_order=True,
    )
    return seq


def batch_generator(
    list_samples,
    batch_size=32,
    pre_processing_function=None,
    resize_size=(128, 128),
    augment=False,
    max_value_labels=19982,
    embedding_size=100,
    base_path="",
):
    seq = get_seq()
    pre_processing_function = (
        pre_processing_function
        if pre_processing_function is not None
        else preprocess_input
    )
    while True:
        shuffle(list_samples)
        for batch_samples in chunks(list_samples, size=batch_size):
            images = [
                read_img_from_path(base_path + sample.path) for sample in batch_samples
            ]

            if augment:
                images = seq.augment_images(images=images)

            images = [resize_img(x, h=resize_size[0], w=resize_size[1]) for x in images]

            images = [pre_processing_function(a) for a in images]
            targets_positive = [choice(sample.labels) for sample in batch_samples]
            targets_negative = targets_positive[::-1]
            targets_negative = [
                x
                if x not in sample.labels
                else np.random.randint(0, max_value_labels + 1)
                for x, sample in zip(targets_negative, batch_samples)
            ]

            X = np.array(images)
            Y1 = np.array(targets_positive)[..., np.newaxis]
            Y2 = np.array(targets_negative)[..., np.newaxis]

            yield [X, Y1, Y2], np.zeros((len(batch_samples), 3 * embedding_size))


def df_to_list_samples(df, label_df, fold):
    image_name_col = "ImageID"
    label_col = "LabelName"

    label_to_int_mapping, display_labels_to_int_mapping, W = label_mapping_from_df(
        label_df
    )

    df[label_col] = df[label_col].map(label_to_int_mapping)

    df = df.groupby(image_name_col)[label_col].apply(list).reset_index(name=label_col)

    paths = df[image_name_col].apply(lambda x: str(Path(fold) / (x + ".jpg"))).tolist()
    list_labels = df[label_col].values.tolist()

    samples = [
        SampleFromPath(path=path, labels=labels)
        for path, labels in zip(paths, list_labels)
    ]

    return samples


def label_mapping_from_df(label_df):
    nlp = spacy.load("en_core_web_md")

    labels = label_df["LabelName"].tolist()
    display_labels = label_df["DisplayName"].tolist()
    label_to_int_mapping = {x: i for x, i in zip(labels, range(len(labels)))}
    display_labels_to_int_mapping = {
        x: i for x, i in zip(display_labels, range(len(display_labels)))
    }

    W = np.random.uniform(-0.1, 0.1, size=(len(display_labels), 300))

    for i in range(len(display_labels)):
        d_name = display_labels[i]
        W[i, :] = nlp(d_name).vector

    return label_to_int_mapping, display_labels_to_int_mapping, W


if __name__ == "__main__":
    import yaml
    import pandas as pd
    import os

    config_path = Path("../example/training_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    label_df = pd.read_csv(Path(config["data_path"]) / "oidv6-class-descriptions.csv")

    label_to_int, display_to_int_labels = label_mapping_from_df(label_df)

    print(len(display_to_int_labels))

    print(label_to_int)
    print(display_to_int_labels)

    fold = "oidv6-train"

    df = pd.read_csv(
        Path(config["data_path"]) / ("%s-annotations-human-imagelabels.csv" % fold),
        usecols=["ImageID", "LabelName"],
    )

    samples = df_to_list_samples(df, label_df, fold)

    print(len(samples))
    print(len([x for x in samples if os.path.isfile(x.path)]))
