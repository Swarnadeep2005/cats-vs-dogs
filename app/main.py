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

from fastapi import FastAPI, File, HTTPException, UploadFile
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
    if not app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train a model and restart the service.",
        )

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

    try:
        label, confidence = predict(image_bytes)
    except Exception as exc:
        # Covers corrupt/unreadable image data — PIL raises various
        # exception types depending on what's wrong with the file,
        # so we catch broadly here and surface it as a clean 400
        # rather than leaking a raw traceback to the client.
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")

    return PredictionResponse(label=label, confidence=confidence)
