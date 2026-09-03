# Cats vs Dogs Classifier

An image classification project upgraded from a single notebook into a
reproducible ML pipeline with data augmentation, transfer learning,
Docker-based serving, and CI.

> This README will be filled out fully in Phase 8. For now it documents
> Phase 0 setup only.

## Project status

- [x] Phase 0 — Repo structure, config, data download script
- [ ] Phase 1 — Data pipeline & augmentation
- [ ] Phase 2 — Model: baseline + transfer learning
- [ ] Phase 3 — Evaluation
- [ ] Phase 4 — Inference packaging
- [ ] Phase 5 — Serving API
- [ ] Phase 6 — Docker
- [ ] Phase 7 — CI/CD
- [ ] Phase 8 — Documentation & polish

## Setup (Phase 0)

1. Clone the repo and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get Kaggle API credentials: go to kaggle.com/settings -> API ->
   "Create New Token". This downloads a `kaggle.json` file containing
   your username and key.

3. Set them as environment variables (do NOT commit kaggle.json):
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_key_from_kaggle_json
   ```

4. Download and extract the dataset:
   ```bash
   python src/data/download.py
   ```
   This will populate `data/raw/train` and `data/raw/test`.

5. All hyperparameters and paths live in `configs/config.yaml` — edit
   that file to change image size, batch size, epochs, or which model
   architecture to use, rather than editing source code.

## Project structure

```
cats-vs-dogs/
├── data/                  # gitignored — populated by download.py
├── notebooks/             # exploratory work only
├── src/
│   ├── data/               # download + preprocessing (Phase 1)
│   └── models/              # model definitions + training (Phase 2)
├── app/                   # FastAPI serving app (Phase 5)
├── frontend/              # Streamlit/Gradio demo (Phase 6)
├── model/                 # saved model artifacts (gitignored)
├── tests/                 # unit tests (Phase 4)
├── configs/config.yaml    # all hyperparameters and paths
├── requirements.txt
└── .gitignore
```
