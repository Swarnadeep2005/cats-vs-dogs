"""
Unit tests for src/predict.py.

Two kinds of test here on purpose:

1. Tests for preprocess_image() — these need no trained model and no
   dataset, just a synthetic in-memory image, so they always run
   (including in CI, see .github/workflows/ci.yml in Phase 7).

2. Tests for predict() end-to-end — these need an actual trained
   .keras file on disk. They're skipped automatically if no model is
   found, rather than failing, since CI won't have a multi-hundred-MB
   trained model committed to the repo (models are gitignored on
   purpose — see .gitignore). Run these locally after training to
   get real coverage; they'll no-op in CI until Phase 7 wires up a
   small test-fixture model or downloads one from storage.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.predict import preprocess_image, predict, load_trained_model, CLASS_NAMES, _default_model_path
from src.data.preprocess import load_config


def make_synthetic_image(width=400, height=300) -> Image.Image:
    """A random RGB image — stands in for a real photo. We only care
    about shape/type handling here, not what the image depicts.
    """
    array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


class TestPreprocessImage:
    def test_output_shape_matches_config_image_size(self):
        config = load_config()
        expected_size = tuple(config["data"]["image_size"])

        image = make_synthetic_image()
        batch = preprocess_image(image)

        assert batch.shape == (1, expected_size[0], expected_size[1], 3)

    def test_output_dtype_is_float32(self):
        batch = preprocess_image(make_synthetic_image())
        assert batch.dtype == np.float32

    def test_output_range_is_zero_to_one(self):
        batch = preprocess_image(make_synthetic_image())
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0

    def test_accepts_file_path(self, tmp_path):
        image_path = tmp_path / "test_image.jpg"
        make_synthetic_image().save(image_path)

        batch = preprocess_image(image_path)
        assert batch.shape[0] == 1

    def test_accepts_raw_bytes(self):
        import io

        buffer = io.BytesIO()
        make_synthetic_image().save(buffer, format="PNG")

        batch = preprocess_image(buffer.getvalue())
        assert batch.shape[0] == 1

    def test_handles_grayscale_input(self):
        # Simulates a user uploading a grayscale image — should not crash,
        # should still produce a 3-channel batch.
        grayscale = Image.fromarray(
            np.random.randint(0, 256, size=(300, 400), dtype=np.uint8), mode="L"
        )
        batch = preprocess_image(grayscale)
        assert batch.shape[-1] == 3

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            preprocess_image(12345)


def _model_available() -> bool:
    try:
        config = load_config()
        return Path(_default_model_path(config)).exists()
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="No trained model found — run training first")
class TestPredictEndToEnd:
    def test_predict_returns_valid_label(self):
        label, confidence = predict(make_synthetic_image())
        assert label in CLASS_NAMES

    def test_predict_confidence_in_valid_range(self):
        _, confidence = predict(make_synthetic_image())
        assert 0.5 <= confidence <= 1.0  # always >=0.5 since it's the winning class's probability

    def test_model_is_cached_between_calls(self):
        model_a = load_trained_model()
        model_b = load_trained_model()
        assert model_a is model_b  # same object, not reloaded from disk
