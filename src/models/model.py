"""
Model architectures for cats vs dogs classification.

Two models are defined here on purpose:

1. build_baseline_cnn — a CNN trained from scratch. This is your
   comparison baseline. It's the "before" in a before/after story.

2. build_transfer_model — MobileNetV2 pretrained on ImageNet, with
   a new classification head. This is almost always the better
   choice for a small, standard image classification problem like
   this one: ImageNet already learned rich, general visual features
   (edges, textures, shapes) from 1.4 million images, and transfer
   learning reuses those instead of relearning them from ~20k images.

Both take the same `data_augmentation` block (from preprocess.py) as
their first layer, so augmentation is applied identically regardless
of which architecture is used.
"""

from tensorflow import keras


def build_baseline_cnn(input_shape: tuple, data_augmentation: keras.Sequential) -> keras.Model:
    """A from-scratch CNN — three conv blocks, similar in spirit to
    the original notebook's architecture, but with augmentation and
    dropout added to reduce overfitting.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            data_augmentation,
            keras.layers.Rescaling(1.0 / 255),

            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),

            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),

            keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),

            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),  # randomly drop 50% of activations to fight overfitting
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),  # binary output: 0=cat, 1=dog
        ],
        name="baseline_cnn",
    )
    return model


def build_transfer_model(
    input_shape: tuple,
    data_augmentation: keras.Sequential,
    dropout: float = 0.2,
) -> tuple[keras.Model, keras.Model]:
    """MobileNetV2 backbone (frozen) + a small trainable head.

    Returns (model, base_model) — you need the base_model reference
    separately so train.py can unfreeze its top layers for the
    fine-tuning stage later.

    Note: we do NOT add our own Rescaling(1/255) layer here, because
    MobileNetV2 expects inputs in [-1, 1], and keras.applications
    provides a matching preprocess_input function. To keep things
    simple and consistent with the rest of the pipeline (which
    normalizes to [0, 1] in preprocess.py), we instead rescale from
    [0, 1] to [-1, 1] with a small Rescaling layer here.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,       # drop MobileNetV2's original 1000-class ImageNet head
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for the initial training stage

    inputs = keras.layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = keras.layers.Rescaling(2.0, offset=-1.0)(x)  # maps [0,1] -> [-1,1] for MobileNetV2
    x = base_model(x, training=False)  # training=False keeps BatchNorm stats frozen too
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="transfer_mobilenetv2")
    return model, base_model
