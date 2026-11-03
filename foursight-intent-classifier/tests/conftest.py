import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import pytest  # noqa: E402

MODEL_DIR = Path(os.environ.get("FOURSIGHT_MODEL_DIR", ROOT / "models" / "final"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def model_dir() -> Path:
    if not (MODEL_DIR / "config.json").exists():
        pytest.skip(f"trained model not found at {MODEL_DIR}; run scripts/train_final.py first")
    return MODEL_DIR


@pytest.fixture(scope="session")
def classifier(model_dir):
    from foursight_intent.inference import load_classifier

    return load_classifier(model_dir, device="cpu", num_threads=2)
