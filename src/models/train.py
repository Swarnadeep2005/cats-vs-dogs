"""
Trains a model on the cats vs dogs dataset and saves it, along with a
JSON log of the run's config and final metrics.

Why it's built this way:

- Reads everything from configs/config.yaml — no hardcoded
  hyperparameters. Change config["model"]["architecture"] to
  "baseline_cnn" or "mobilenetv2" and re-run to compare them.

- Two-stage training for transfer learning: first train only the
  new classification head with the MobileNetV2 base frozen (fast,
  stable), then unfreeze the top layers of the base and continue
  training at a much lower learning rate (fine-tuning). Training
  the whole network at a normal learning rate from the start would
  wreck the pretrained weights before the new head has learned
  anything useful.

- Every run's config + final metrics get appended to logs/runs.json,
  so after running both architectures you have a simple file to pull
  numbers from for your README's comparison table.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tensorflow import keras

sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow `src.` imports when run directly

from src.data.preprocess import build_datasets, build_augmentation_layers
from src.models.model import build_baseline_cnn, build_transfer_model


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_callbacks(model_path: str, patience: int):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-7,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def log_run(log_path: str, run_info: dict):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    runs = []
    if Path(log_path).exists():
        with open(log_path, "r") as f:
            runs = json.load(f)
    runs.append(run_info)
    with open(log_path, "w") as f:
        json.dump(runs, f, indent=2)


def train():
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    input_shape = tuple(data_cfg["image_size"]) + (3,)

    print("Building datasets...")
    train_ds, val_ds, test_ds, class_names = build_datasets(config)
    print(f"Classes: {class_names}")

    augmentation = build_augmentation_layers()

    architecture = model_cfg["architecture"]
    model_dir = Path(paths_cfg["model_output_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / f"{architecture}-{paths_cfg['model_version']}.keras")

    if architecture == "baseline_cnn":
        model = build_baseline_cnn(input_shape, augmentation)
        model.compile(
            optimizer=keras.optimizers.Adam(train_cfg["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg["epochs"],
            callbacks=get_callbacks(model_path, train_cfg["early_stopping_patience"]),
        )
        final_val_accuracy = max(history.history["val_accuracy"])

    elif architecture == "mobilenetv2":
        model, base_model = build_transfer_model(
            input_shape, augmentation, dropout=model_cfg["dropout"]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(train_cfg["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())

        # Stage 1: train only the new head, base model frozen.
        print("\n--- Stage 1: training classification head (base frozen) ---")
        history_stage1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg["epochs"],
            callbacks=get_callbacks(model_path, train_cfg["early_stopping_patience"]),
        )

        # Stage 2: unfreeze the top of the base model and fine-tune
        # at a much lower learning rate. Unfreezing everything (or
        # using too high a learning rate here) would destroy the
        # pretrained ImageNet weights.
        print("\n--- Stage 2: fine-tuning top layers of MobileNetV2 ---")
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 30  # unfreeze roughly the last 30 layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(train_cfg["fine_tune_learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg["fine_tune_epochs"],
            callbacks=get_callbacks(model_path, train_cfg["early_stopping_patience"]),
        )
        final_val_accuracy = max(history_stage2.history["val_accuracy"])

    else:
        raise ValueError(f"Unknown architecture in config: {architecture}")

    # Evaluate on the held-out test set — the number that actually
    # matters, since it was never touched during training or tuning.
    print("\n--- Evaluating on held-out test set ---")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {test_accuracy:.4f}")

    log_run(
        paths_cfg["logs_dir"] + "/runs.json",
        {
            "architecture": architecture,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "final_val_accuracy": float(final_val_accuracy),
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "model_path": model_path,
        },
    )
    print(f"\nModel saved to {model_path}")
    print(f"Run logged to {paths_cfg['logs_dir']}/runs.json")


if __name__ == "__main__":
    train()
