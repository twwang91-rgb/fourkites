"""Deterministic CPU inference for a saved FourSight intent model.

Determinism contract (see README "Determinism"): for a fixed model directory and a fixed
software environment, ``IntentClassifier.predict`` returns bit-identical probabilities and
hence identical labels for identical input strings. Inference is run in fp32 on CPU with a
fixed thread count, no dropout (eval mode), no sampling, and the argmax tie-break is the
lowest label index. The model is pure: it has no state between calls.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.rules import apply_rules
from foursight_intent.tokenization import encode_texts, pad_batch, special_token_template
from foursight_intent.utils import get_logger

log = get_logger("inference")


@dataclass
class Prediction:
    intent: str
    confidence: float          # calibrated max probability
    margin: float              # calibrated p(top1) - p(top2)
    needs_review: bool         # confidence below the validation-selected threshold
    probs: dict[str, float]    # calibrated probability per label
    second_intent: str

    def to_dict(self) -> dict:
        return asdict(self)


class IntentClassifier:
    """Load a model directory produced by ``train.save_model`` and classify questions."""

    def __init__(self, model_dir: str | Path, device: str = "cpu", num_threads: int = 4, batch_size: int = 32):
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        transformers.logging.set_verbosity_error()
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"no model found at {self.model_dir}")
        self.device = device
        self.batch_size = batch_size
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if device == "cpu":
            torch.set_num_threads(num_threads)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, dtype=torch.float32)
        self.model.to(device).eval()
        cfg_path = self.model_dir / "inference_config.json"
        self.cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.max_length = int(self.cfg.get("max_length", 128))
        self.tail_tokens = int(self.cfg.get("tail_tokens", 16))
        self.temperature = float(self.cfg.get("temperature", 1.0))
        self.review_threshold = float(self.cfg.get("review_threshold", 0.0))
        self.review_metric = self.cfg.get("review_metric", "confidence")
        self.logit_bias = np.asarray(self.cfg.get("logit_bias", [0.0] * len(LABELS)), dtype=np.float64)
        if self.logit_bias.shape != (len(LABELS),):
            raise ValueError("logit_bias must have one entry per label")
        self.use_rules = bool(self.cfg.get("use_rules", False))
        self.rules_max_conf = float(self.cfg.get("rules_max_conf", 0.6))
        self.last_rules_fired: list[list[str]] = []
        label_map = self.model_dir / "label_map.json"
        if label_map.exists():
            saved = json.loads(label_map.read_text())["labels"]
            if saved != LABELS:
                raise ValueError(f"label order in {label_map} ({saved}) differs from taxonomy {LABELS}")
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self._template = special_token_template(self.tokenizer)

    # ------------------------------------------------------------------ encoding
    def _encode(self, texts: Sequence[str]) -> list[list[int]]:
        return encode_texts(self.tokenizer, texts, self.max_length, self.tail_tokens, self._template)

    def _batch(self, seqs: list[list[int]]) -> dict[str, torch.Tensor]:
        ids, mask = pad_batch(seqs, self.pad_id)
        return {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Raw (uncalibrated) logits, shape (n, 7). Rows are processed in input order
        in fixed-size batches so results do not depend on the surrounding inputs' order."""
        seqs = self._encode(texts)
        out = []
        for i in range(0, len(seqs), self.batch_size):
            batch = self._batch(seqs[i : i + self.batch_size])
            out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Calibrated probabilities: softmax(logits / T + logit_bias), then the auditable
        rule layer when the artefact was saved with ``use_rules``. ``logit_bias`` is the
        per-class decision adjustment selected on out-of-fold validation (all zeros when
        not selected). Fired rule names for the last call are kept in ``last_rules_fired``."""
        texts = list(texts)
        probs = softmax(self.logits(texts).astype(np.float64) / self.temperature + self.logit_bias)
        if self.use_rules:
            probs, self.last_rules_fired = apply_rules(texts, probs, max_conf=self.rules_max_conf)
        else:
            self.last_rules_fired = [[] for _ in texts]
        return probs

    def predict(self, texts: Sequence[str] | str) -> list[Prediction]:
        if isinstance(texts, str):
            texts = [texts]
        probs = self.probabilities(list(texts))
        return [self._to_prediction(p) for p in probs]

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]

    def _to_prediction(self, p: np.ndarray) -> Prediction:
        return to_prediction(p, self.review_threshold, self.review_metric)

    def label_order(self) -> list[str]:
        id2label = self.model.config.id2label
        return [id2label.get(i, id2label.get(str(i))) for i in range(len(LABELS))]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_prediction(p: np.ndarray, review_threshold: float, review_metric: str = "confidence") -> Prediction:
    """Turn one calibrated probability vector into a Prediction (stable argmax: lowest
    label index wins exact ties, so the output never depends on dict/hash order)."""
    order = np.argsort(-p, kind="stable")
    top, second = int(order[0]), int(order[1])
    conf, margin = float(p[top]), float(p[top] - p[second])
    score = conf if review_metric == "confidence" else margin
    return Prediction(
        intent=ID2LABEL[top], confidence=conf, margin=margin,
        needs_review=bool(score < review_threshold),
        probs={ID2LABEL[i]: float(p[i]) for i in range(len(LABELS))},
        second_intent=ID2LABEL[second],
    )


class EnsembleClassifier:
    """Probability-averaging ensemble of single-model artefacts saved under one directory.

    Layout::

        <dir>/ensemble.json          members, weights, temperatures, review policy, rules
        <dir>/<member_name>/         a complete single-model artefact per member

    Each member contributes its own temperature-scaled softmax; the weighted mean is
    optionally sharpened by an ensemble temperature (power-renormalisation) and then
    passed once through the rule layer. Determinism holds for the same reasons as for a
    single member (fixed member order, fp32, CPU, pure functions)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu", num_threads: int = 4, batch_size: int = 32):
        self.model_dir = Path(model_dir)
        spec_path = self.model_dir / "ensemble.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"no ensemble.json at {self.model_dir}")
        self.cfg = json.loads(spec_path.read_text())
        self.device = device
        self.members: list[IntentClassifier] = []
        weights = []
        for m in self.cfg["members"]:
            clf = IntentClassifier(self.model_dir / m["dir"], device=device, num_threads=num_threads, batch_size=batch_size)
            clf.use_rules = False  # rules are applied once, at ensemble level
            if "temperature" in m:
                clf.temperature = float(m["temperature"])
            self.members.append(clf)
            weights.append(float(m.get("weight", 1.0)))
        self.weights = np.asarray(weights, dtype=np.float64) / float(np.sum(weights))
        self.ensemble_temperature = float(self.cfg.get("ensemble_temperature", 1.0))
        self.temperature = self.ensemble_temperature
        self.review_threshold = float(self.cfg.get("review_threshold", 0.0))
        self.review_metric = self.cfg.get("review_metric", "confidence")
        self.use_rules = bool(self.cfg.get("use_rules", False))
        self.rules_max_conf = float(self.cfg.get("rules_max_conf", 0.6))
        self.last_rules_fired: list[list[str]] = []
        orders = {tuple(m.label_order()) for m in self.members}
        if len(orders) != 1 or list(next(iter(orders))) != LABELS:
            raise ValueError(f"ensemble members disagree on label order or differ from taxonomy: {orders}")

    def label_order(self) -> list[str]:
        return self.members[0].label_order()

    def member_probabilities(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [m.probabilities(texts) for m in self.members]

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        member_probs = self.member_probabilities(texts)
        probs = np.zeros_like(member_probs[0], dtype=np.float64)
        for w, p in zip(self.weights, member_probs):
            probs += w * p
        if self.ensemble_temperature != 1.0:
            probs = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / self.ensemble_temperature)
            probs = probs / probs.sum(axis=1, keepdims=True)
        if self.use_rules:
            probs, self.last_rules_fired = apply_rules(texts, probs, max_conf=self.rules_max_conf)
        else:
            self.last_rules_fired = [[] for _ in texts]
        return probs

    def predict(self, texts: Sequence[str] | str) -> list[Prediction]:
        if isinstance(texts, str):
            texts = [texts]
        return [self._to_prediction(p) for p in self.probabilities(list(texts))]

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]

    def _to_prediction(self, p: np.ndarray) -> Prediction:
        return to_prediction(p, self.review_threshold, self.review_metric)


def load_classifier(model_dir: str | Path, **kw):
    """Return an EnsembleClassifier if ``ensemble.json`` is present, else an IntentClassifier."""
    model_dir = Path(model_dir)
    if (model_dir / "ensemble.json").exists():
        return EnsembleClassifier(model_dir, **kw)
    return IntentClassifier(model_dir, **kw)


def predict_file(model_dir: str | Path, texts: Iterable[str], **kw) -> list[Prediction]:
    clf = load_classifier(model_dir, **kw)
    return clf.predict(list(texts))
