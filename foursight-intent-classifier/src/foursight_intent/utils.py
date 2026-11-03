"""Seeding, device selection, logging and small IO helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "foursight") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("FOURSIGHT_LOGLEVEL", "INFO"), format=LOG_FORMAT)
        for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "filelock", "sentence_transformers", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and (if installed) torch. Also sets the env vars that make
    cuBLAS / tokenizers deterministic. Safe to call before torch is imported."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - older torch
                pass
    except ImportError:  # torch is optional for the classical baselines
        pass


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string. ``auto`` prefers cuda > mps > cpu.

    Inference defaults to CPU elsewhere (see inference.py) because CPU is the only
    backend on which we test bit-exact determinism across runs.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


@contextmanager
def timer(name: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Context manager that records elapsed wall time into the yielded dict."""
    out: dict = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - t0
        if logger:
            logger.info("%s took %.2fs", name, out["seconds"])


def project_root() -> Path:
    """Repository root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
