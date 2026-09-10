"""
Tests for the FastAPI app in app/main.py, using FastAPI's TestClient
(which runs requests in-process — no real server or network needed).

Like tests/test_predict.py, the /predict tests that need a real
prediction are skipped if no trained model exists, since CI doesn't
have one committed (models are gitignored by design).
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from src.predict import _default_model_path
from src.data.preprocess import load_config
from pathlib import Path


@pytest.fixture
def client():
    # TestClient must be used as a context manager for FastAPI's
    # `lifespan` startup/shutdown events to actually run. Without
    # this, app.state.model_loaded is never set (lifespan never
    # fires), and every request raises AttributeError — this was a
    # real bug caught by running the suite the first time.
    with TestClient(app) as test_client:
        yield test_client


def _model_available() -> bool:
    try:
        config = load_config()
        return Path(_default_model_path(config)).exists()
    except Exception:
        return False


def make_test_image_bytes(fmt="JPEG") -> bytes:
    import numpy as np

    array = np.random.randint(0, 256, size=(300, 400, 3), dtype="uint8")
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_health_endpoint_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_predict_rejects_unsupported_file_type(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_rejects_corrupt_image_data(client):
    # Correct content-type header, but garbage bytes underneath —
    # should fail gracefully at the PIL decode step, not crash.
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", b"definitely not real image data", "image/jpeg")},
    )
    assert response.status_code == 400


@pytest.mark.skipif(not _model_available(), reason="No trained model found — run training first")
def test_predict_returns_valid_response_for_real_image(client):
    image_bytes = make_test_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("cats", "dogs")
    assert 0.5 <= body["confidence"] <= 1.0
