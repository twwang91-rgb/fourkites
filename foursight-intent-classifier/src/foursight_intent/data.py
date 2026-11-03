"""Data loading and leakage-aware splitting.

Two validation protocols are supported (both seeded):

* ``stratified``        plain StratifiedKFold on the label.
* ``grouped``           StratifiedGroupKFold where the group is the normalised question
                        text, so exact duplicates (there are 159 duplicate groups, 36 of
                        them with conflicting labels) can never sit on both sides of a
                        split and inflate validation scores.

The grouped protocol is the default used for every model-selection decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from foursight_intent.labels import LABEL2ID, LABELS, LABEL_SET
from foursight_intent.text import clean_text, dedup_key
from foursight_intent.utils import project_root


def default_data_dir() -> Path:
    return project_root() / "data"


def load_train(path: str | Path | None = None) -> pd.DataFrame:
    """Load train.csv with cleaned text. Columns: id, question, intent, text, label, group."""
    path = Path(path) if path else default_data_dir() / "train.csv"
    df = pd.read_csv(path, dtype={"id": str, "question": str, "intent": str}, keep_default_na=False)
    missing = {"id", "question", "intent"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    bad = set(df["intent"].str.strip().str.lower()) - LABEL_SET
    if bad:
        raise ValueError(f"{path}: labels outside taxonomy: {sorted(bad)}")
    df["intent"] = df["intent"].str.strip().str.lower()
    df["text"] = df["question"].map(clean_text)
    df["label"] = df["intent"].map(LABEL2ID).astype(int)
    df["group"] = df["question"].map(dedup_key)
    return df


def load_test(path: str | Path | None = None) -> pd.DataFrame:
    """Load test.csv (no labels). Preserves row order and ids as strings."""
    path = Path(path) if path else default_data_dir() / "test.csv"
    df = pd.read_csv(path, dtype={"id": str, "question": str}, keep_default_na=False)
    missing = {"id", "question"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["text"] = df["question"].map(clean_text)
    return df


@dataclass(frozen=True)
class Fold:
    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray


def _group_codes(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df["group"])[0]


def cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    protocol: str = "grouped",
) -> list[Fold]:
    """Seeded k-fold split indices. ``protocol`` in {"grouped", "stratified"}."""
    y = df["label"].to_numpy()
    if protocol == "grouped":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(df)), y, groups=_group_codes(df))
    elif protocol == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(df)), y)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    folds = [Fold(i, tr, va) for i, (tr, va) in enumerate(iterator)]
    _assert_disjoint(folds, len(df))
    return folds


def holdout_split(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42, protocol: str = "grouped"
) -> Fold:
    """A single seeded stratified hold-out split (used for the W&B-logged reference run)."""
    y = df["label"].to_numpy()
    if protocol == "grouped":
        # Take fold 0 of a k-fold where k = round(1 / val_fraction) so that groups stay intact.
        k = max(2, int(round(1.0 / val_fraction)))
        return cv_folds(df, n_splits=k, seed=seed, protocol="grouped")[0]
    idx = np.arange(len(df))
    tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=y)
    return Fold(0, np.sort(tr), np.sort(va))


def _assert_disjoint(folds: list[Fold], n: int) -> None:
    seen = np.zeros(n, dtype=int)
    for f in folds:
        assert len(np.intersect1d(f.train_idx, f.val_idx)) == 0, "train/val overlap"
        seen[f.val_idx] += 1
    assert (seen == 1).all(), "every row must be validated exactly once"


def leakage_report(df: pd.DataFrame, folds: list[Fold]) -> dict:
    """How many validation rows share their exact text with a training row in the same fold."""
    out = {"folds": []}
    for f in folds:
        train_groups = set(df.iloc[f.train_idx]["group"])
        val_groups = df.iloc[f.val_idx]["group"]
        leaked = int(val_groups.isin(train_groups).sum())
        out["folds"].append({"fold": f.index, "val_rows": int(len(f.val_idx)), "leaked_rows": leaked})
    out["total_leaked_rows"] = int(sum(x["leaked_rows"] for x in out["folds"]))
    return out


def class_weights(labels: np.ndarray, scheme: str = "sqrt_inv") -> np.ndarray:
    """Per-class weights for weighted cross-entropy.

    ``sqrt_inv``  w_c = sqrt(N / (K * n_c)), a tempered inverse frequency.
    ``inv``       w_c = N / (K * n_c)
    ``none``      all ones
    """
    counts = np.bincount(labels, minlength=len(LABELS)).astype(float)
    counts = np.maximum(counts, 1.0)
    n, k = counts.sum(), len(LABELS)
    if scheme == "none":
        return np.ones(k)
    if scheme == "inv":
        w = n / (k * counts)
    elif scheme == "sqrt_inv":
        w = np.sqrt(n / (k * counts))
    else:
        raise ValueError(scheme)
    return w / w.mean()


def iter_label_names(labels: np.ndarray) -> Iterator[str]:
    for i in labels:
        yield LABELS[int(i)]
