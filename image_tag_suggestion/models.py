import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Concatenate,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Embedding,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    https://github.com/KinWaiCheuk/Triplet-net-keras/blob/master/Triplet%20NN%20Test%20on%20MNIST.ipynb
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    total_lenght = y_pred.shape.as_list()[-1]
    anchor = y_pred[:, 0 : int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3) : int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3) : int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def get_model(
    vocab_size=20000,
    input_shape=(None, None, 3),
    model="mobilenet",
    weights="imagenet",
    embedding_size=100,
    lr=0.0001,
):
    input_1 = Input(input_shape)
    input_2 = Input(shape=(1,))
    input_3 = Input(shape=(1,))

    _norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))

    if model == "mobilenet":
        base_model = MobileNetV2(
            include_top=False, input_shape=input_shape, weights=weights
        )
    else:
        base_model = ResNet50(
            include_top=False, input_shape=input_shape, weights=weights
        )

    x1 = base_model(input_1)
    out1 = GlobalMaxPooling2D()(Dropout(0.5)(x1))
    out2 = GlobalAveragePooling2D()(x1)
    image_representation = Concatenate(axis=-1)([out1, out2])

    image_representation = Dense(embedding_size, name="img_repr")(image_representation)

    image_representation = _norm(image_representation)

    embed = Embedding(vocab_size, embedding_size, name="embed")

    x2 = embed(input_2)

    x2 = Flatten()(x2)

    x3 = embed(input_3)

    x3 = Flatten()(x3)

    label1 = _norm(x2)
    label2 = _norm(x3)

    x = Concatenate(axis=-1)([image_representation, label1, label2])

    model = Model([input_1, input_2, input_3], x)

    model_image = Model(input_1, image_representation)
    model_label = Model([input_2], label1)

    model.compile(loss=triplet_loss, optimizer=Adam(lr))
    model_image.compile(loss="mae", optimizer=Adam(lr))
    model_label.compile(loss="mae", optimizer=Adam(lr))

    model.summary()

    return model, model_image, model_label
