"""
FastAPI serving app for the cats vs dogs classifier.

Design choices worth understanding:

- /health exists separately from /predict because production
  orchestration tools (Docker healthchecks, Kubernetes liveness
  probes, load balancers) need a cheap, fast way to ask "is this
  service alive?" without paying the cost of running a real
  inference. A missing health endpoint is one of the first things
  that trips people up when containerizing a service for the first
  time (Phase 6).

- The model is loaded once at startup (via a FastAPI startup event),
  not on every request. Loading a multi-hundred-MB Keras model from
  disk takes real time — doing it per-request would make every
  prediction slow and would hammer the disk under load.

- Validation happens before calling predict(): file type and size are
  checked up front, and PIL decode errors are caught, so a bad upload
  returns a clean 400 with a useful message instead of a raw 500
  stack trace. An API that crashes on bad input isn't production
  quality.
"""

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.predict import load_trained_model, predict

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class PredictionResponse(BaseModel):
    label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, at startup, and keep it in memory for the
    # life of the process. predict.py's internal cache means this is
    # also safe to call again later without reloading from disk.
    try:
        load_trained_model()
        app.state.model_loaded = True
    except FileNotFoundError:
        # Let the app start anyway so /health can report the problem
        # clearly, rather than the container crash-looping on boot.
        app.state.model_loaded = False
    yield


app = FastAPI(title="Cats vs Dogs Classifier API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if app.state.model_loaded else "model not loaded",
        model_loaded=app.state.model_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # Validate the request itself first — file type and size are
    # client-side problems (400) regardless of whether a model is
    # loaded. Checking model_loaded first was a real bug: it made
    # every request return 503 in any environment without a trained
    # model (like a fresh CI checkout), even requests that were
    # invalid for reasons that have nothing to do with the model.
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
            f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}.",
        )

    image_bytes = await file.read()

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max size is {MAX_UPLOAD_SIZE_MB}MB.",
        )

    # Validate the bytes are actually a decodable image before
    # checking model state — this way "is this a valid image?" is
    # answered consistently whether or not a model happens to be
    # loaded (important for CI, which has no model file at all; see
    # .gitignore and tests/test_api.py).
    try:
        Image.open(BytesIO(image_bytes)).verify()
    except Exception:
        # PIL can raise several different exception types depending
        # on exactly what's wrong with the bytes (UnidentifiedImageError,
        # OSError, struct errors on truncated files, etc.) — catching
        # broadly here means any of these become a clean 400 instead
        # of a raw 500.
        raise HTTPException(status_code=400, detail="Could not process image: not a valid image file.")

    # Only now do we need the model to actually exist.
    if not app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train a model and restart the service.",
        )

    try:
        label, confidence = predict(image_bytes)
    except Exception as exc:
        # Belt-and-suspenders: covers any other unreadable-image edge
        # case predict() might hit that .verify() above didn't catch.
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")

    return PredictionResponse(label=label, confidence=confidence)
