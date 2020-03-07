import argparse
import json

import numpy as np
import yaml

from models import get_model


def save_sub_models(training_config_path):
    with open(training_config_path, "r") as f:
        training_config = yaml.load(f, yaml.SafeLoader)

    with open(training_config["label_display_to_int"], "r") as f:
        display_labels_to_int_mapping = json.load(f)

    model, model_image, model_label = get_model(
        vocab_size=max(display_labels_to_int_mapping.values()) + 1,
        trainable=training_config["train_embeddings"],
    )

    try:
        model.load_weights(training_config["model_path"])
        model_image.load_weights(training_config["model_path"], by_name=True)
        model_label.load_weights(training_config["model_path"], by_name=True)
    except:
        print("No model to load")

    model_image.save(training_config["image_model_path"], include_optimizer=False)

    n_labels = max(display_labels_to_int_mapping.values())

    print(n_labels)

    pred_vectors = model_label.predict(np.array(range(n_labels + 1))[..., np.newaxis])

    print(pred_vectors.shape)

    labels = {
        k: pred_vectors[v, :].tolist() for k, v in display_labels_to_int_mapping.items()
    }

    with open(training_config["label_embedding_path"], "w") as f:
        json.dump(labels, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config_path",
        help="training_config_path",
        default="../example/training_config.yaml",
    )
    args = parser.parse_args()

    training_config_path = args.training_config_path

    save_sub_models(training_config_path)
