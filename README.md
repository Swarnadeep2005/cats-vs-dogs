# Cats vs Dogs Classifier

![CI](https://github.com/Swarnadeep2005/cats-vs-dogs/actions/workflows/ci.yml/badge.svg)

An image classifier upgraded from a single Colab notebook into a
reproducible, containerized ML pipeline: data augmentation, a
from-scratch CNN benchmarked against MobileNetV2 transfer learning,
rigorous evaluation, a FastAPI serving layer, a Streamlit demo UI,
Docker images for training and serving, and CI on every push.

**Quick links:** [Results](#results) · [Architecture](#architecture)
· [Run it locally](#run-it-locally) · [Project structure](#project-structure)
· [Build log](#build-log-phase-by-phase)

## Results

| Architecture | Test Accuracy | Test AUC | Notes |
|---|---|---|---|
| CNN trained from scratch | 77% | 0.8475 | Baseline — see [note below](#why-the-baseline-lags) |
| MobileNetV2 (transfer learning) | 99% | 0.9998 | Fine-tuned, 2-stage training |

<p align="center">
  <img src="docs/images/mobilenetv2_confusion_matrix.png" width="32%" />
  <img src="docs/images/mobilenetv2_roc_curve.png" width="32%" />
  <img src="docs/images/mobilenetv2_misclassified.png" width="32%" />
</p>
<p align="center"><em>MobileNetV2 evaluation on the held-out test set (never used in training or tuning).</em></p>

#### Why the baseline lags

This gap is the point of the comparison, not a flaw in it.
MobileNetV2 starts from ImageNet-pretrained features (edges,
textures, shapes learned from 1.4M images) and only has to adapt
them to cats vs dogs. The from-scratch CNN has to learn all of that
using ~16k training images — a much harder task that plateaus lower
given the same training budget. Full baseline images and the
earlier bug that stalled it entirely at 48% (chance level) are in
the [build log](#phase-2--phase-3).

## Architecture

```
                    ┌─────────────────┐
                    │  Kaggle dataset  │
                    │  (dogsvscats)    │
                    └────────┬─────────┘
                             │  src/data/download.py
                             ▼
                    ┌─────────────────┐
                    │   tf.data        │  src/data/preprocess.py
                    │   pipeline +     │  train/val/test split,
                    │   augmentation   │  augmentation layers
                    └────────┬─────────┘
                             │
                 ┌───────────┴────────────┐
                 ▼                        ▼
        ┌─────────────────┐     ┌──────────────────┐
        │  baseline_cnn    │     │  MobileNetV2      │  src/models/
        │  (from scratch)  │     │  (transfer learn) │  model.py, train.py
        └────────┬─────────┘     └────────┬──────────┘
                 └───────────┬────────────┘
                             ▼
                    ┌─────────────────┐
                    │   evaluate.py    │  confusion matrix, ROC-AUC,
                    │                  │  classification report
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │   predict.py     │  single inference function,
                    │                  │  shared by API + CLI
                    └────────┬─────────┘
                             ▼
        ┌───────────────────┴────────────────────┐
        ▼                                         ▼
┌─────────────────┐                     ┌──────────────────┐
│  FastAPI (app/)  │◄────HTTP───────────│  Streamlit demo    │
│  /predict         │                    │  (frontend/)       │
│  /health          │                    └──────────────────┘
└─────────────────┘
   Both containerized (Docker) and checked on every push (GitHub Actions CI)
```

## Run it locally

Fastest path — Docker Compose, assuming you already have a trained
model in `model/` (see [Full setup](#full-setup) if not):

```bash
docker compose up --build
```

Open `http://localhost:8501` for the demo UI, or
`http://localhost:8000/docs` for the API directly.

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/image.jpg"
```

```python
import requests
response = requests.post(
    "http://127.0.0.1:8000/predict",
    files={"file": open("path/to/image.jpg", "rb")},
)
print(response.json())  # {"label": "dogs", "confidence": 0.97}
```

## Full setup

1. Clone the repo and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a Kaggle API token: go to kaggle.com/settings -> API ->
   "Create New Token" (format: `KGAT_...`), then set it as an
   environment variable (never commit it):
   ```bash
   export KAGGLE_API_TOKEN=KGAT_your_token_here     # Windows: set KAGGLE_API_TOKEN=...
   ```
   An older "Legacy API Key" also works — the download script
   supports `KAGGLE_USERNAME` + `KAGGLE_KEY` as a fallback.

3. Download the dataset and train a model:
   ```bash
   python src/data/download.py
   python -m src.models.train        # edit configs/config.yaml to pick baseline_cnn vs mobilenetv2
   ```

4. Evaluate it:
   ```bash
   python -m src.evaluate --model model/mobilenetv2-v1.keras
   ```

5. Serve it:
   ```bash
   uvicorn app.main:app --reload
   ```

All hyperparameters and paths live in `configs/config.yaml` — nothing
is hardcoded in source. Run the test suite anytime with `pytest -v`.

## Project structure

```
cats-vs-dogs/
├── data/                  # gitignored — populated by download.py
├── notebooks/             # exploratory work only
├── src/
│   ├── data/
│   │   ├── download.py     # Kaggle dataset download
│   │   └── preprocess.py   # tf.data pipeline + augmentation
│   ├── models/
│   │   ├── model.py         # baseline CNN + transfer learning architectures
│   │   └── train.py         # training loop, callbacks, run logging
│   ├── evaluate.py         # confusion matrix, ROC-AUC, misclassified examples
│   └── predict.py          # single reusable inference function
├── docs/images/           # evaluation images embedded in this README
├── app/
│   └── main.py            # FastAPI serving app: /health, /predict
├── frontend/
│   └── app.py             # Streamlit demo UI
├── model/                 # saved model artifacts (gitignored)
├── logs/                  # run logs (gitignored)
├── tests/                 # unit tests
├── configs/config.yaml    # all hyperparameters and paths
├── requirements.txt       # full dependencies (training + serving)
├── requirements-serve.txt # trimmed dependencies for the Docker serving image
├── requirements-frontend.txt # trimmed dependencies for the Streamlit container
├── Dockerfile.serve       # multi-stage build for the API image
├── Dockerfile.train       # training container image
├── Dockerfile.frontend    # Streamlit container image
├── docker-compose.yml     # runs api + frontend together
├── .dockerignore
├── .github/workflows/ci.yml # tests + Docker build on every push
└── .gitignore
```

## Skills demonstrated

- **Data engineering:** `tf.data` pipelines with caching/prefetch,
  a real train/val/test split (not just train/val), augmentation
  applied correctly (as model layers, so it's active only during
  training).
- **Modeling:** transfer learning with staged fine-tuning (frozen
  base → selective unfreezing at a reduced learning rate), benchmarked
  against a from-scratch baseline rather than reported in isolation.
- **Evaluation:** precision/recall/F1 per class, ROC-AUC, confusion
  matrices, and misclassified-example inspection — not just accuracy.
- **Software engineering:** config-driven hyperparameters, a single
  inference function shared between API and CLI, a real test suite
  (with CI-appropriate skip conditions for tests needing large
  artifacts), and debugging real bugs (see build log) rather than
  reporting only clean results.
- **MLOps / deployment:** a FastAPI service with health checks and
  input validation, multi-stage Docker builds, separate
  training/serving images and dependency sets, Docker Compose for a
  multi-service demo, and CI that runs tests and validates the Docker
  build on every push.

## Build log (phase by phase)

This section documents what was built and debugged in each phase,
including bugs that were found and fixed rather than only the final
clean state — a green CI badge or a high accuracy number means more
with the debugging history attached.

<details>
<summary><strong>Phase 0 — Repo structure & data download</strong></summary>

Config-driven setup (`configs/config.yaml`) instead of hardcoded
hyperparameters. `src/data/download.py` reads Kaggle credentials from
environment variables rather than a committed `kaggle.json`, and
supports both Kaggle's newer single-token auth (`KAGGLE_API_TOKEN`)
and the legacy `KAGGLE_USERNAME`/`KAGGLE_KEY` pair.
</details>

<details>
<summary><strong>Phase 1 — Data pipeline & augmentation</strong></summary>

`src/data/preprocess.py` builds train/validation/**test** `tf.data`
pipelines — the original notebook only had train/val, meaning every
tuning decision was implicitly fit to the validation set. Augmentation
(flip, rotation, zoom, contrast) is defined as Keras layers attached
to the model itself (Phase 2), not applied in the data pipeline
directly, so Keras automatically disables it during evaluation/
inference and keeps it active during training.
</details>

<details>
<summary><strong>Phase 2 — Baseline CNN + MobileNetV2 transfer learning</strong></summary>

Two architectures, controlled by `model.architecture` in the config.
`baseline_cnn` is a 3-block CNN trained from scratch. `mobilenetv2`
uses a frozen ImageNet-pretrained backbone, trained in two stages:
first the new classification head alone, then fine-tuned by
unfreezing the top ~30 layers at a 100x lower learning rate — training
the whole network at a normal learning rate from the start would
wreck the pretrained weights before the new head learns anything.

**Bug found:** an early version of `baseline_cnn` included a
`Rescaling(1/255)` layer inside the model *in addition to* the
normalization already applied in the data pipeline — double-
normalizing pixel values down to a near-zero range and stalling
training completely (48% accuracy, chance level, loss ≈ ln(2)).
Fixed by removing the duplicate layer.
</details>

<details>
<summary><strong>Phase 3 — Rigorous evaluation</strong></summary>

`src/evaluate.py` produces a confusion matrix, ROC curve with AUC,
full classification report, and a grid of misclassified examples —
all computed on the held-out test set, never touched during training
or tuning. After fixing the Phase 2 bug and training for a full 20
epochs, `baseline_cnn` reached 77% test accuracy (AUC 0.8475);
`mobilenetv2` reached 99% (AUC 0.9998).

<p align="center">
  <img src="docs/images/baseline_cnn_confusion_matrix.png" width="32%" />
  <img src="docs/images/baseline_cnn_roc_curve.png" width="32%" />
  <img src="docs/images/baseline_cnn_misclassified.png" width="32%" />
</p>

**Bug found:** the first version namespaced evaluation outputs by a
fixed filename, so evaluating a second model silently overwrote the
first model's confusion matrix/ROC/misclassified images. Fixed by
namespacing output folders per model (`logs/evaluation/<model-name>/`).
</details>

<details>
<summary><strong>Phase 4 — Inference packaging</strong></summary>

`src/predict.py` exposes one `predict(image_input)` function used by
both the API and any CLI/batch usage, so preprocessing logic exists
in exactly one place. It accepts a file path, raw bytes, a PIL Image,
or a numpy array. Unit tests split into preprocessing tests (run
always, using synthetic images) and end-to-end prediction tests
(auto-skipped when no trained model is present, since models are
gitignored and CI has none committed).
</details>

<details>
<summary><strong>Phase 5 — FastAPI serving</strong></summary>

`app/main.py` — `GET /health` for orchestration health checks,
`POST /predict` for inference. The model loads once at startup, not
per-request. Input validation (file type, size, decodability) happens
before invoking the model.

**Bug found (caught by CI, see Phase 7):** the model-loaded check
originally ran *before* input validation, so any request in an
environment without a trained model returned `503` regardless of
whether the input itself was valid — including requests that should
have failed validation with `400` for reasons unrelated to the model.
Fixed by validating the request (file type, size, and now an explicit
PIL decode check) before checking model availability.
</details>

<details>
<summary><strong>Phase 6 — Docker</strong></summary>

Separate images for separate lifecycles: `Dockerfile.serve` (multi-
stage build, trimmed `requirements-serve.txt`, no training
dependencies), `Dockerfile.train` (full dependencies, data/model
mounted as volumes rather than baked in), `Dockerfile.frontend`
(Streamlit demo). `docker-compose.yml` runs the API and frontend as
separate services on Docker's internal network.

**Bugs found:** (1) `requirements-frontend.txt` initially pinned a
Streamlit version older than when `st.image`'s `use_container_width`
argument was introduced, causing a `TypeError` at runtime — fixed by
bumping the pin. (2) `Dockerfile.serve` originally tried to `COPY`
the `model/` folder into the image, which doesn't exist in a clean
checkout since model files are gitignored — fixed by relying on the
volume mount already defined in `docker-compose.yml` instead.
</details>

<details>
<summary><strong>Phase 7 — CI/CD</strong></summary>

`.github/workflows/ci.yml` runs on every push/PR: a `test` job
(`pytest`) and a `docker-build` job (confirms `Dockerfile.serve`
actually compiles). Both jobs failed on the very first run — see the
bugs documented in Phase 5 and Phase 6 above, both surfaced directly
by CI rather than found manually. Both are fixed and CI is green.
</details>
