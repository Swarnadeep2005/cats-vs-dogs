"""
Single entry point for running inference with a trained model.

Why this file exists as its own module:

Both the FastAPI serving app (Phase 5) and any future batch/CLI
scripts need to do the exact same three things: load the model, turn
a raw image into the tensor shape/scale the model expects, and
interpret the sigmoid output as a label + confidence. If that logic
were duplicated in app/main.py, a future change to preprocessing
(say, a different image size) would need to be updated in two places
and could silently drift out of sync — a classic source of "works in
training, broken in production" bugs. Centralizing it here means
there is exactly one place that defines "how this model expects to
see an image."

Class order note: `CLASS_NAMES` below must match the order Keras
assigns during training. `keras.utils.image_dataset_from_directory`
(used in src/data/preprocess.py) assigns labels by sorting the
subfolder names alphabetically, so for a `train/cats/`, `train/dogs/`
layout, "cats" is always class 0 and "dogs" is always class 1. This
was also confirmed by the `class_names` printed at the top of every
training run (src/models/train.py) and every evaluation run
(src/evaluate.py) — if you ever reorganize the dataset folders,
re-check that printed order before trusting this constant.
"""

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from tensorflow import keras

from src.data.preprocess import load_config

CLASS_NAMES = ["cats", "dogs"]  # index 0 = cats, index 1 = dogs — see note above

# Module-level cache so repeated predict() calls (e.g. from an API
# handling many requests) don't reload the model from disk every time.
_loaded_models = {}


def _default_model_path(config: dict) -> str:
    paths_cfg = config["paths"]
    model_cfg = config["model"]
    return str(
        Path(paths_cfg["model_output_dir"])
        / f"{model_cfg['architecture']}-{paths_cfg['model_version']}.keras"
    )


def load_trained_model(model_path: str = None) -> keras.Model:
    """Loads (and caches) a trained model from disk.

    Pass an explicit model_path to override the default one built
    from configs/config.yaml (useful for comparing baseline_cnn vs
    mobilenetv2 without editing the config).
    """
    config = load_config()
    if model_path is None:
        model_path = _default_model_path(config)

    if model_path not in _loaded_models:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run "
                f"`python -m src.models.train` first, or pass the "
                f"correct --model path."
            )
        _loaded_models[model_path] = keras.models.load_model(model_path)

    return _loaded_models[model_path]


def preprocess_image(
    image_input: Union[bytes, str, Path, Image.Image, np.ndarray],
    image_size: tuple = None,
) -> np.ndarray:
    """Converts a raw image (bytes, file path, PIL Image, or numpy
    array) into the (1, H, W, 3) float32 [0, 1] batch the model
    expects — matching exactly what src/data/preprocess.py produces
    during training, minus augmentation (which only applies at
    training time; see model.py).
    """
    if image_size is None:
        config = load_config()
        image_size = tuple(config["data"]["image_size"])

    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype("uint8"))
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    image = image.convert("RGB")  # drop alpha channel / handle grayscale uploads
    image = image.resize(image_size)

    array = np.asarray(image, dtype=np.float32) / 255.0  # match preprocess.py's [0,1] normalization
    return np.expand_dims(array, axis=0)  # add batch dimension -> (1, H, W, 3)


def predict(
    image_input: Union[bytes, str, Path, Image.Image, np.ndarray],
    model_path: str = None,
) -> tuple:
    """Runs inference on a single image.

    Returns (label: str, confidence: float) where label is "cats" or
    "dogs" and confidence is the model's probability for that
    predicted class (always >= 0.5, since it's whichever side of the
    threshold the sigmoid output landed on).
    """
    model = load_trained_model(model_path)
    batch = preprocess_image(image_input)

    raw_output = model.predict(batch, verbose=0)[0][0]  # sigmoid output, P(class == "dogs")
    predicted_index = int(raw_output >= 0.5)
    confidence = float(raw_output if predicted_index == 1 else 1 - raw_output)

    return CLASS_NAMES[predicted_index], confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image_path", type=str, help="Path to an image file")
    parser.add_argument("--model", type=str, default=None, help="Path to a trained .keras model")
    args = parser.parse_args()

    label, confidence = predict(args.image_path, model_path=args.model)
    print(f"Prediction: {label} (confidence: {confidence:.4f})")
