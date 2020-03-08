from pathlib import Path

import pandas as pd
import yaml

with open("../example/training_config.yaml", "r") as f:
    training_config = yaml.load(f, yaml.SafeLoader)

train = pd.read_csv(
    Path(training_config["data_path"])
    / "oidv6-train-annotations-human-imagelabels.csv",
    usecols=["ImageID", "LabelName"],
)

train = train[
    train["ImageID"].str.startswith("c") + train["ImageID"].str.startswith("3")
]

train.to_csv(
    Path(training_config["data_path"]) / "c3-train-annotations-human-imagelabels.csv",
    index=False,
)
