"""
Downloads and extracts the dogs-vs-cats dataset from Kaggle.

Why this exists:
The original notebook copied a committed kaggle.json into ~/.kaggle/.
That means Kaggle credentials could end up in git history — a real
security mistake in production code. Instead, this script reads
credentials from an environment variable and writes them to disk at
runtime, so nothing sensitive ever touches the repo.

Kaggle now issues a single API token (format: KGAT_xxxx) instead of
the old username+key pair. This script supports that new token via
KAGGLE_API_TOKEN, and falls back to the legacy KAGGLE_USERNAME /
KAGGLE_KEY pair if you're using an older "Legacy API Key" instead.

Usage (new token — recommended):
    Windows (cmd.exe):  set KAGGLE_API_TOKEN=KGAT_your_token_here
    macOS/Linux:         export KAGGLE_API_TOKEN=KGAT_your_token_here
    python src/data/download.py

Usage (legacy username/key, if that's what you generated):
    set KAGGLE_USERNAME=your_username
    set KAGGLE_KEY=your_api_key
    python src/data/download.py
"""

import os
import json
import zipfile
import subprocess
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def write_kaggle_credentials():
    """Writes Kaggle credentials to disk from environment variables.

    Prefers the new single-token auth (KAGGLE_API_TOKEN, written to
    ~/.kaggle/access_token) and falls back to the legacy
    username/key pair (KAGGLE_USERNAME + KAGGLE_KEY, written to
    ~/.kaggle/kaggle.json) if the new token isn't set.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    api_token = os.environ.get("KAGGLE_API_TOKEN")
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if api_token:
        token_path = kaggle_dir / "access_token"
        with open(token_path, "w") as f:
            f.write(api_token.strip())
        os.chmod(token_path, 0o600)
        return

    if username and key:
        creds_path = kaggle_dir / "kaggle.json"
        with open(creds_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        # Kaggle CLI requires this file to be readable only by the owner.
        os.chmod(creds_path, 0o600)
        return

    raise EnvironmentError(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN to your token "
        "from kaggle.com/settings -> API -> Create New Token, or set the "
        "legacy KAGGLE_USERNAME + KAGGLE_KEY pair if you generated a "
        "Legacy API Key instead."
    )


def download_dataset(raw_dir: str):
    """Downloads the dataset zip via the Kaggle CLI and extracts it."""
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    zip_path = Path(raw_dir) / "dogsvscats.zip"

    if zip_path.exists():
        print(f"Dataset zip already present at {zip_path}, skipping download.")
    else:
        print("Downloading dataset from Kaggle...")
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "salader/dogsvscats",
                "-p", raw_dir,
            ],
            check=True,
        )

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    print(f"Dataset ready at {raw_dir}")


if __name__ == "__main__":
    config = load_config()
    write_kaggle_credentials()
    download_dataset(config["data"]["raw_dir"])
