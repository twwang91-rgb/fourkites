"""Classical baselines: TF-IDF + linear models, and sentence-embedding + linear models.

All baselines expose ``fit(texts, labels)`` / ``predict_proba(texts)`` so the same
cross-validation driver and the same calibration / review-flag code can be used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from foursight_intent.labels import LABELS


@dataclass
class TfidfLinear:
    """Word + char TF-IDF with logistic regression or linear SVM.

    ``features``: "word", "char" or "word+char".
    ``clf``: "logreg" or "svm". SVM probabilities come from a Platt-scaled
    CalibratedClassifierCV (3-fold, inside the training fold only).
    """

    features: str = "word+char"
    clf: str = "logreg"
    C: float = 4.0
    class_weight: str | None = "balanced"
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (2, 5)
    min_df: int = 1
    seed: int = 42
    _vecs: list = field(default_factory=list, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"tfidf[{self.features}]+{self.clf}(C={self.C},cw={self.class_weight})"

    def _build_vecs(self) -> None:
        self._vecs = []
        if "word" in self.features:
            self._vecs.append(
                TfidfVectorizer(
                    analyzer="word", ngram_range=self.word_ngrams, min_df=self.min_df,
                    sublinear_tf=True, lowercase=True, token_pattern=r"(?u)\b\w+\b",
                )
            )
        if "char" in self.features:
            self._vecs.append(
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=self.char_ngrams, min_df=self.min_df,
                    sublinear_tf=True, lowercase=True,
                )
            )

    def _transform(self, texts: Sequence[str], fit: bool = False):
        mats = [v.fit_transform(texts) if fit else v.transform(texts) for v in self._vecs]
        return hstack(mats).tocsr() if len(mats) > 1 else mats[0]

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfLinear":
        self._build_vecs()
        X = self._transform(texts, fit=True)
        y = np.asarray(labels)
        if self.clf == "logreg":
            self._model = LogisticRegression(
                C=self.C, class_weight=self.class_weight, max_iter=5000, random_state=self.seed,
            )
            self._model.fit(X, y)
        elif self.clf == "svm":
            base = LinearSVC(C=self.C, class_weight=self.class_weight, random_state=self.seed, max_iter=20000)
            self._model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            self._model.fit(X, y)
        else:
            raise ValueError(self.clf)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self._transform(texts)
        proba = self._model.predict_proba(X)
        return _expand_to_all_classes(proba, self._model.classes_)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)


def _expand_to_all_classes(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Guarantee a (n, 7) matrix even if a training fold misses a class."""
    out = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out


class EmbeddingCache:
    """Encode texts once per (model, text) and reuse across folds.

    Encoding is deterministic for a fixed model on CPU; we normalise embeddings so a
    linear classifier on top is cosine-like.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64, cache_dir=None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = min(self.model.max_seq_length or 256, 256)
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}
        self.prefix = "query: " if "e5" in model_name.lower() else ""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            embs = self.model.encode(
                [self.prefix + t for t in todo], batch_size=self.batch_size,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
            )
            for t, e in zip(todo, embs):
                self._cache[t] = e.astype(np.float32)
        return np.stack([self._cache[t] for t in texts])


@dataclass
class EmbeddingLinear:
    """Frozen sentence embeddings + logistic regression."""

    cache: EmbeddingCache
    C: float = 4.0
    class_weight: str | None = "balanced"
    seed: int = 42
    _model: LogisticRegression | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        short = self.cache.model_name.split("/")[-1]
        return f"emb[{short}]+logreg(C={self.C},cw={self.class_weight})"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "EmbeddingLinear":
        X = self.cache.encode(texts)
        self._model = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=5000, random_state=self.seed)
        self._model.fit(X, np.asarray(labels))
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self.cache.encode(texts)
        return _expand_to_all_classes(self._model.predict_proba(X), self._model.classes_)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)
