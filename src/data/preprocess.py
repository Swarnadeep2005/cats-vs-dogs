"""
Builds train/validation/test tf.data pipelines for the cats vs dogs
dataset, with augmentation and performance optimizations.

Why this file looks the way it does:

1. Augmentation as MODEL layers, not a separate preprocessing step.
   We return `data_augmentation` as its own Sequential block that
   gets attached to the front of the model (in Phase 2), rather than
   baking it into the dataset pipeline. This matters because Keras
   automatically disables these layers during model.evaluate() and
   model.predict() (inference), but keeps them active during
   model.fit() (training) — meaning you get correct augmented
   training and correct un-augmented evaluation for free, with no
   risk of accidentally evaluating on augmented images.

2. A real three-way split. The original notebook only had a
   train/validation split, which means every decision made while
   tuning the model (architecture, learning rate, epochs to train)
   was implicitly "fit" to the validation set too. A held-out test
   set that is never used for any decision-making gives an honest
   final accuracy number — the one that actually matters for a
   resume/portfolio writeup.

3. .cache().prefetch(AUTOTUNE). Without these, the CPU has to
   re-decode and re-resize every image from disk on every single
   epoch, and the GPU sits idle waiting for data. This is a standard
   tf.data performance pattern that's worth knowing for ML engineer
   interviews specifically — "how would you speed up your data
   pipeline" is a common question.
"""

import yaml
import tensorflow as tf
from tensorflow import keras


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_augmentation_layers() -> keras.Sequential:
    """Returns a Sequential block of augmentation layers.

    Attach this as the first layer(s) of your model (Phase 2) rather
    than applying it to the dataset directly — Keras handles the
    train-vs-inference switching automatically that way.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_datasets(config: dict):
    """Builds train, validation, and test tf.data.Dataset objects.

    The Kaggle dataset ships with a `train/` folder (with cat/dog
    subfolders) and a `test/` folder that is typically unlabeled or
    used differently depending on the dataset version. To get a
    trustworthy held-out test set, we split `train/` three ways
    ourselves: train / validation / test, rather than trusting
    whatever `test/` folder shipped with the download.
    """
    data_cfg = config["data"]
    image_size = tuple(data_cfg["image_size"])
    batch_size = data_cfg["batch_size"]
    seed = data_cfg["seed"]

    # First split: carve off a validation+test chunk from train/.
    # validation_split here represents val+test combined; we'll split
    # that chunk in half below to get separate val and test sets.
    val_test_split = data_cfg["validation_split"]

    full_val_test_ds = keras.utils.image_dataset_from_directory(
        directory=data_cfg["train_dir"],
        labels="inferred",
        label_mode="int",
        validation_split=val_test_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = keras.utils.image_dataset_from_directory(
        directory=data_cfg["train_dir"],
        labels="inferred",
        label_mode="int",
        validation_split=val_test_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Split the held-out chunk in half: one half becomes validation
    # (used during training to monitor overfitting), the other half
    # becomes test (touched exactly once, at the very end).
    val_batches = tf.data.experimental.cardinality(full_val_test_ds)
    test_ds = full_val_test_ds.take(val_batches // 2)
    val_ds = full_val_test_ds.skip(val_batches // 2)

    class_names = train_ds.class_names  # e.g. ['cats', 'dogs']

    # Normalize pixel values to [0, 1]. Note: we do NOT apply
    # augmentation here — that happens inside the model itself (see
    # build_augmentation_layers above).
    normalization_layer = keras.layers.Rescaling(1.0 / 255)

    def normalize(image, label):
        return normalization_layer(image), label

    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Performance: cache decoded images in memory after the first
    # epoch, and prefetch the next batch while the current one is
    # still being used for training.
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    config = load_config()
    train_ds, val_ds, test_ds, class_names = build_datasets(config)

    print(f"Class names: {class_names}")
    print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")

    # Sanity check: pull one batch and confirm shapes/ranges look right.
    for images, labels in train_ds.take(1):
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Pixel value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
