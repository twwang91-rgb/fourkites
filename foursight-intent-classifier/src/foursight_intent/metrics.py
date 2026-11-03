"""Exact re-implementation of the official scorer (score.py) plus richer diagnostics.

``official_scores`` reproduces score.py arithmetic: per-class P/R/F1 from the confusion
matrix, macro-F1 as the unweighted mean over the seven labels (classes with zero support
still count, with F1 = 0), accuracy, and

    OVERALL = 0.6 * macro_F1 + 0.4 * accuracy

Predictions outside the label set count as wrong (never as a true positive for any class),
exactly like the scorer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from foursight_intent.labels import LABELS, LABEL_SET


def _norm(v: object) -> str:
    return (str(v) if v is not None else "").strip().lower()


def confusion_matrix(gold: Sequence[str], pred: Sequence[str]) -> np.ndarray:
    """Rows = gold, cols = predicted, in LABELS order. Invalid predictions are dropped
    from the columns (they are still counted as errors by ``official_scores``)."""
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for g, p in zip(gold, pred):
        g, p = _norm(g), _norm(p)
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1
    return cm


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def official_scores(gold: Sequence[str], pred: Sequence[str]) -> dict:
    """Return the same numbers score.py prints, as a dict.

    Keys: n, accuracy, macro_f1, weighted_f1, macro_precision, macro_recall, OVERALL,
    per_class{label: {precision, recall, f1, support}}, confusion (nested dict),
    confused_pairs [(count, gold, pred), ...] sorted descending, invalid_predictions.
    """
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} rows but pred has {len(pred)}")
    gold_n = [_norm(g) for g in gold]
    pred_n = [_norm(p) for p in pred]
    n = len(gold_n)
    if n == 0:
        raise ValueError("empty evaluation set")

    conf: dict[str, Counter] = {g: Counter() for g in LABELS}
    correct = 0
    invalid = 0
    for g, p in zip(gold_n, pred_n):
        if g not in LABEL_SET:
            raise ValueError(f"gold label {g!r} outside taxonomy")
        if p not in LABEL_SET:
            p = "__invalid__"
            invalid += 1
        conf[g][p] += 1
        if p == g:
            correct += 1

    accuracy = correct / n
    per = {}
    for c in LABELS:
        tp = conf[c].get(c, 0)
        fn = sum(v for k, v in conf[c].items() if k != c)
        fp = sum(conf[g].get(c, 0) for g in LABELS if g != c)
        p, r, f = _prf(tp, fp, fn)
        per[c] = {"precision": p, "recall": r, "f1": f, "support": sum(conf[c].values())}

    macro_f1 = sum(per[c]["f1"] for c in LABELS) / len(LABELS)
    macro_p = sum(per[c]["precision"] for c in LABELS) / len(LABELS)
    macro_r = sum(per[c]["recall"] for c in LABELS) / len(LABELS)
    weighted_f1 = sum(per[c]["f1"] * per[c]["support"] for c in LABELS) / n
    overall = 0.6 * macro_f1 + 0.4 * accuracy

    pairs = sorted(
        ((cnt, g, p) for g in LABELS for p, cnt in conf[g].items() if p != g and cnt > 0),
        reverse=True,
    )
    return {
        "n": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "weighted_f1": weighted_f1,
        "OVERALL": overall,
        "per_class": per,
        "confusion": {g: dict(conf[g]) for g in LABELS},
        "confused_pairs": [list(x) for x in pairs],
        "invalid_predictions": invalid,
    }


def overall_from_arrays(gold: Sequence[str], pred: Sequence[str]) -> float:
    return official_scores(gold, pred)["OVERALL"]


def format_report(scores: dict, title: str | None = None, top_pairs: int = 8) -> str:
    """Human readable summary in the spirit of score.py's stdout."""
    lines = []
    if title:
        lines.append(f"== {title} ==")
    lines.append(f"rows scored          {scores['n']}")
    lines.append(f"accuracy             {100 * scores['accuracy']:6.2f}%")
    lines.append(f"macro-F1             {100 * scores['macro_f1']:6.2f}%")
    lines.append(f"weighted-F1          {100 * scores['weighted_f1']:6.2f}%")
    lines.append(f"OVERALL              {100 * scores['OVERALL']:6.2f}")
    lines.append("")
    lines.append(f"{'class':<22} {'prec':>7} {'recall':>7} {'f1':>7} {'support':>8}")
    for c in LABELS:
        m = scores["per_class"][c]
        lines.append(
            f"{c:<22} {100 * m['precision']:6.1f}% {100 * m['recall']:6.1f}% "
            f"{100 * m['f1']:6.1f}% {m['support']:8d}"
        )
    lines.append("")
    lines.append("most confused (gold -> predicted):")
    for cnt, g, p in scores["confused_pairs"][:top_pairs]:
        lines.append(f"   {g:<22} -> {p:<22} {cnt}")
    lines.append("")
    lines.append("confusion matrix (rows = gold, cols = predicted):")
    lines.append(" " * 22 + "".join(f"{c[:8]:>9}" for c in LABELS))
    for g in LABELS:
        row = "".join(f"{scores['confusion'][g].get(c, 0):>9}" for c in LABELS)
        lines.append(f"{g:<22}{row}")
    return "\n".join(lines)


@dataclass
class FoldScores:
    """Container for cross-validated results: per-fold and pooled out-of-fold."""

    per_fold: list[dict] = field(default_factory=list)
    pooled: dict | None = None

    def summary(self) -> dict:
        keys = ["accuracy", "macro_f1", "OVERALL"]
        out: dict = {}
        for k in keys:
            vals = np.array([f[k] for f in self.per_fold], dtype=float)
            out[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            out[f"{k}_std"] = float(vals.std(ddof=0)) if len(vals) else float("nan")
        if self.pooled is not None:
            for k in keys:
                out[f"{k}_oof"] = float(self.pooled[k])
            out["per_class_f1_oof"] = {c: self.pooled["per_class"][c]["f1"] for c in LABELS}
        return out


def triad_confusion(scores: dict) -> dict:
    """Errors confined to the shipment_information / realtime_query / analytics triad."""
    from foursight_intent.labels import OPERATIONAL_TRIAD

    triad = set(OPERATIONAL_TRIAD)
    out = {}
    for cnt, g, p in scores["confused_pairs"]:
        if g in triad and p in triad:
            out[f"{g}->{p}"] = cnt
    out["total_triad_errors"] = sum(out.values())
    return out
