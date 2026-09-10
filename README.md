# Cats vs Dogs Classifier

An image classification project upgraded from a single notebook into a
reproducible ML pipeline with data augmentation, transfer learning,
Docker-based serving, and CI.

> This README is updated as each phase is completed. Currently covers
> through Phase 5.

## Project status

- [x] Phase 0 — Repo structure, config, data download script
- [x] Phase 1 — Data pipeline & augmentation
- [x] Phase 2 — Model: baseline CNN + MobileNetV2 transfer learning
- [x] Phase 3 — Evaluation
- [x] Phase 4 — Inference packaging
- [x] Phase 5 — Serving API
- [ ] Phase 6 — Docker
- [ ] Phase 7 — CI/CD
- [ ] Phase 8 — Documentation & polish

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a Kaggle API token: go to kaggle.com/settings -> API ->
   "Create New Token". This gives you a single token string
   (format: `KGAT_...`).

3. Set it as an environment variable (never commit this token):
   ```bash
   export KAGGLE_API_TOKEN=KGAT_your_token_here     # Windows: set KAGGLE_API_TOKEN=...
   ```
   If you generated an older "Legacy API Key" instead, set
   `KAGGLE_USERNAME` and `KAGGLE_KEY` instead — the download script
   supports both.

4. Download and extract the dataset:
   ```bash
   python src/data/download.py
   ```
   This will populate `data/raw/train` and `data/raw/test`.

5. All hyperparameters and paths live in `configs/config.yaml` — edit
   that file to change image size, batch size, epochs, or which model
   architecture to use, rather than editing source code.

## Data pipeline (Phase 1)

`src/data/preprocess.py` builds train/validation/test `tf.data`
pipelines from `data/raw/train`, splitting off a held-out test set
that is never used for training or hyperparameter tuning — only for
final evaluation. Augmentation (random flip, rotation, zoom, contrast)
is defined here but applied inside the model itself (see Phase 2),
so it's automatically active during training and inactive during
evaluation/inference.

Run it standalone to sanity-check the pipeline:
```bash
python src/data/preprocess.py
```

## Models (Phase 2)

Two architectures are available, controlled by `model.architecture`
in `configs/config.yaml`:

- `baseline_cnn` — a 3-block CNN trained from scratch. Serves as the
  comparison baseline.
- `mobilenetv2` — a MobileNetV2 backbone pretrained on ImageNet, with
  a new classification head. Trained in two stages: first the head
  alone (base frozen), then fine-tuned by unfreezing the top ~30
  layers at a much lower learning rate.

Train either one with:
```bash
python -m src.models.train
```

Each run saves the best checkpoint to `model/<architecture>-<version>.keras`
and appends its config + final test/validation accuracy to
`logs/runs.json`, so results from both architectures can be compared
side by side.

| Architecture | Val Accuracy | Test Accuracy | Test AUC |
|---|---|---|---|
| baseline_cnn | — | 77% | 0.8475 |
| mobilenetv2 | — | 99% | 0.9998 |

**Baseline CNN — Phase 3 result (after bug fix + full training):**
An earlier training run had a bug (see git history / commit notes) where
images were being normalized twice — once in the data pipeline and
again inside the model — which pushed pixel values so close to zero
that the model couldn't learn at all (48% accuracy, chance level).
After removing the duplicate rescaling layer and training for a full
20 epochs instead of 5, `baseline_cnn` reaches **77% test accuracy**
with an AUC of 0.8475 (precision/recall/F1 ≈ 0.75–0.79 across both
classes — see `logs/evaluation/baseline_cnn-v1/classification_report.txt`).

<p align="center">
  <img src="docs/images/baseline_cnn_confusion_matrix.png" width="32%" />
  <img src="docs/images/baseline_cnn_roc_curve.png" width="32%" />
  <img src="docs/images/baseline_cnn_misclassified.png" width="32%" />
</p>

**MobileNetV2 — Phase 3 result:** reaches **99% test accuracy** with
an AUC of 0.9998 and 0.99 precision/recall/F1 on both classes (see
`logs/evaluation/mobilenetv2-v1/classification_report.txt`). This
gap over the baseline is expected and is the core point of the
comparison: MobileNetV2 starts from ImageNet-pretrained features
(edges, textures, shapes learned from 1.4M images) and only needs to
adapt them to cats vs dogs, while `baseline_cnn` has to learn all of
that from scratch using ~16k training images — a much harder task
that plateaus at a meaningfully lower ceiling given the same training
budget.

<p align="center">
  <img src="docs/images/mobilenetv2_confusion_matrix.png" width="32%" />
  <img src="docs/images/mobilenetv2_roc_curve.png" width="32%" />
  <img src="docs/images/mobilenetv2_misclassified.png" width="32%" />
</p>

### Evaluation artifacts

Running `python -m src.evaluate --model <path>` produces, per model,
a confusion matrix, ROC curve, classification report, and a grid of
misclassified examples, saved under `logs/evaluation/<model-name>/`:

```bash
python -m src.evaluate --model model/baseline_cnn-v1.keras
python -m src.evaluate --model model/mobilenetv2-v1.keras
```

## Inference (Phase 4)

`src/predict.py` exposes a single `predict(image_input)` function
that both the serving API and any CLI/batch usage call — this keeps
preprocessing logic in exactly one place. It accepts a file path,
raw bytes, a PIL Image, or a numpy array, and returns `(label,
confidence)`.

```bash
python -m src.predict path/to/some_image.jpg
```

Run the test suite (preprocessing tests always run; end-to-end
prediction tests are skipped automatically if no trained model is
present):

```bash
pytest -v
```

## Serving API (Phase 5)

`app/main.py` is a FastAPI app with two endpoints:

- `GET /health` — reports whether a trained model is loaded. Used by
  container/orchestration health checks in Phase 6.
- `POST /predict` — accepts an image file upload (`multipart/form-data`,
  field name `file`) and returns `{"label": "cats"|"dogs",
  "confidence": float}`. Rejects unsupported file types, oversized
  uploads (>10MB), and corrupt image data with a clean 400 response
  instead of a raw server error.

Run it locally:

```bash
uvicorn app.main:app --reload
```

Then test it:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/some_image.jpg"
```

Interactive API docs (Swagger UI) are auto-generated by FastAPI at
`http://127.0.0.1:8000/docs`.

## Project structure

```
cats-vs-dogs/
├── data/                  # gitignored — populated by download.py
├── notebooks/             # exploratory work only
├── src/
│   ├── data/
│   │   ├── download.py     # Kaggle dataset download (Phase 0)
│   │   └── preprocess.py   # tf.data pipeline + augmentation (Phase 1)
│   ├── models/
│   │   ├── model.py         # baseline CNN + transfer learning architectures (Phase 2)
│   │   └── train.py         # training loop, callbacks, run logging (Phase 2)
│   ├── evaluate.py         # confusion matrix, ROC-AUC, misclassified examples (Phase 3)
│   └── predict.py          # single reusable inference function (Phase 4)
├── docs/images/           # evaluation images embedded in this README (committed, not gitignored)
├── app/
│   └── main.py            # FastAPI serving app: /health, /predict (Phase 5)
├── frontend/              # Streamlit/Gradio demo (Phase 6)
├── model/                 # saved model artifacts (gitignored)
├── logs/                  # run logs (gitignored)
├── tests/                 # unit tests (Phases 4-5)
├── configs/config.yaml    # all hyperparameters and paths
├── requirements.txt       # full dependencies (training + serving)
├── requirements-serve.txt # trimmed dependencies for the Docker serving image (Phase 6)
└── .gitignore
```
