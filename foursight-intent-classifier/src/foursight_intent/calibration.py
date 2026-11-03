"""Confidence calibration, selective prediction and the ``needs_review`` flag.

Everything here operates on *out-of-fold* logits so that temperature and thresholds are
never fitted on rows the model was trained on.

* ``fit_temperature``      single-parameter temperature scaling (minimises NLL).
* ``expected_calibration_error`` / ``reliability_bins``  standard 15-bin ECE.
* ``select_review_threshold``  pick the confidence threshold that flags at most
                           ``max_flag_rate`` of traffic while maximising captured errors.
* ``review_metrics``       precision / recall / coverage of the flag at a threshold.
* ``risk_coverage_curve``  selective-prediction curve + AURC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(logits / T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.25, 10.0)) -> float:
    """Return the temperature T > 0 minimising validation NLL."""
    res = minimize_scalar(lambda T: nll(logits, y, T), bounds=bounds, method="bounded", options={"xatol": 1e-4})
    return float(res.x)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            out.append({"lo": float(lo), "hi": float(hi), "count": 0, "confidence": None, "accuracy": None})
            continue
        out.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()), "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return out


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_bins(conf, correct, n_bins)
    n = len(conf)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]))


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


@dataclass
class ReviewPolicy:
    metric: str          # "confidence" (max prob) or "margin" (top1 - top2)
    threshold: float
    flag_rate: float     # share of validation rows flagged
    error_recall: float  # share of errors flagged
    precision: float     # share of flagged rows that are errors
    accuracy_retained: float  # accuracy on unflagged rows
    n_flagged: int
    n_errors_flagged: int


def _scores(probs: np.ndarray, metric: str) -> np.ndarray:
    top2 = np.sort(probs, axis=1)[:, ::-1][:, :2]
    return top2[:, 0] if metric == "confidence" else top2[:, 0] - top2[:, 1]


def review_metrics(probs: np.ndarray, y: np.ndarray, threshold: float, metric: str = "confidence") -> ReviewPolicy:
    s = _scores(probs, metric)
    pred = probs.argmax(1)
    wrong = pred != y
    flagged = s < threshold
    n_fl = int(flagged.sum())
    n_err_fl = int((flagged & wrong).sum())
    retained = ~flagged
    return ReviewPolicy(
        metric=metric, threshold=float(threshold), flag_rate=float(flagged.mean()),
        error_recall=float(n_err_fl / max(wrong.sum(), 1)), precision=float(n_err_fl / max(n_fl, 1)),
        accuracy_retained=float((pred[retained] == y[retained]).mean()) if retained.any() else float("nan"),
        n_flagged=n_fl, n_errors_flagged=n_err_fl,
    )


def select_review_threshold(probs: np.ndarray, y: np.ndarray, max_flag_rate: float = 0.2, metric: str = "confidence") -> ReviewPolicy:
    """Largest threshold whose flag rate stays within budget (monotone => max error recall)."""
    s = _scores(probs, metric)
    candidates = np.unique(np.round(s, 4))
    best = None
    for t in candidates:
        pol = review_metrics(probs, y, t, metric)
        if pol.flag_rate <= max_flag_rate and (best is None or pol.error_recall >= best.error_recall):
            best = pol
    return best if best is not None else review_metrics(probs, y, 0.0, metric)


def risk_coverage_curve(probs: np.ndarray, y: np.ndarray, metric: str = "confidence") -> dict:
    """Selective risk at each coverage level (sorted by decreasing score) and the AURC."""
    s = _scores(probs, metric)
    wrong = (probs.argmax(1) != y).astype(float)
    order = np.argsort(-s, kind="stable")
    cum_err = np.cumsum(wrong[order])
    n = len(y)
    coverage = np.arange(1, n + 1) / n
    risk = cum_err / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, coverage)) if hasattr(np, "trapezoid") else float(np.trapz(risk, coverage))
    pts = [{"coverage": float(c), "risk": float(r)} for c, r in zip(coverage[::max(1, n // 50)], risk[::max(1, n // 50)])]
    return {"aurc": aurc, "points": pts, "full_risk": float(wrong.mean())}


def calibration_summary(logits: np.ndarray, y: np.ndarray, T: float | None = None, n_bins: int = 15) -> dict:
    """ECE / NLL / Brier before and after temperature scaling plus the fitted T."""
    T = fit_temperature(logits, y) if T is None else T
    raw, cal = softmax(logits), softmax(logits / T)
    pred = raw.argmax(1)
    correct = (pred == y)
    out = {
        "temperature": float(T),
        "raw": {"ece": expected_calibration_error(raw.max(1), correct, n_bins), "nll": nll(logits, y, 1.0), "brier": brier_score(raw, y), "mean_confidence": float(raw.max(1).mean())},
        "calibrated": {"ece": expected_calibration_error(cal.max(1), correct, n_bins), "nll": nll(logits, y, T), "brier": brier_score(cal, y), "mean_confidence": float(cal.max(1).mean())},
        "accuracy": float(correct.mean()),
        "reliability_calibrated": reliability_bins(cal.max(1), correct, n_bins),
        "reliability_raw": reliability_bins(raw.max(1), correct, n_bins),
    }
    return out


# --------------------------------------------------------------------------- decision bias
def _fast_overall(pred: np.ndarray, y: np.ndarray, k: int = 7) -> float:
    """OVERALL from integer predictions without the dict machinery (used in tuning loops)."""
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return 0.6 * f1.mean() + 0.4 * tp.sum() / len(y)


def tune_logit_bias(logits: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None, n_rounds: int = 3) -> np.ndarray:
    """Additive per-class log-biases maximising OVERALL on the given (out-of-fold) logits.

    Coordinate ascent over a fixed grid, class by class, a few rounds. This is the usual
    decision-threshold adjustment for imbalanced classification: it trades majority-class
    recall for minority-class recall where the macro-F1 term rewards it. Must be fitted on
    out-of-fold logits and evaluated cross-fitted (see ``cross_fitted_bias``)."""
    grid = np.linspace(-2.0, 2.0, 41) if grid is None else grid
    k = logits.shape[1]
    b = np.zeros(k)
    best = _fast_overall((logits + b).argmax(1), y, k)
    for _ in range(n_rounds):
        improved = False
        for c in range(k):
            for v in grid:
                cand = b.copy()
                cand[c] = v
                s = _fast_overall((logits + cand).argmax(1), y, k)
                if s > best + 1e-12:
                    best, b, improved = s, cand, True
        if not improved:
            break
    return b


def cross_fitted_bias(logits: np.ndarray, y: np.ndarray, folds, **kw) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fit the bias on the other folds' OOF logits, apply to each held-out fold.
    Returns (biased_logits, per-fold bias vectors)."""
    out = logits.copy()
    biases = []
    for f in folds:
        b = tune_logit_bias(logits[f.train_idx], y[f.train_idx], **kw)
        out[f.val_idx] = logits[f.val_idx] + b
        biases.append(b)
    return out, biases
