"""Shared cross-validation driver and error-analysis helpers.

``run_cv`` takes any factory producing an object with fit/predict_proba and returns
pooled out-of-fold probabilities plus per-fold official scores. Out-of-fold (OOF)
predictions are what every downstream analysis (calibration, review threshold,
confusions, rules) is computed on, so no decision ever sees a row it was trained on.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from foursight_intent.data import Fold
from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.metrics import FoldScores, official_scores
from foursight_intent.utils import get_logger

log = get_logger("evaluate")


def run_cv(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    factory: Callable[[], object],
    text_col: str = "text",
    verbose: bool = True,
) -> tuple[np.ndarray, FoldScores, dict]:
    """Fit ``factory()`` on each fold; return (oof_proba, FoldScores, timing)."""
    n = len(df)
    oof = np.zeros((n, len(LABELS)), dtype=float)
    fs = FoldScores()
    fit_s, pred_s = 0.0, 0.0
    for f in folds:
        tr, va = df.iloc[f.train_idx], df.iloc[f.val_idx]
        model = factory()
        t0 = time.perf_counter()
        model.fit(tr[text_col].tolist(), tr["label"].to_numpy())
        t1 = time.perf_counter()
        proba = model.predict_proba(va[text_col].tolist())
        t2 = time.perf_counter()
        fit_s += t1 - t0
        pred_s += t2 - t1
        oof[f.val_idx] = proba
        pred_labels = [ID2LABEL[int(i)] for i in proba.argmax(1)]
        sc = official_scores(va["intent"].tolist(), pred_labels)
        fs.per_fold.append(sc)
        if verbose:
            log.info("fold %d  acc=%.4f  macroF1=%.4f  OVERALL=%.4f", f.index, sc["accuracy"], sc["macro_f1"], sc["OVERALL"])
    fs.pooled = official_scores(df["intent"].tolist(), [ID2LABEL[int(i)] for i in oof.argmax(1)])
    timing = {"fit_seconds_total": fit_s, "predict_seconds_total": pred_s, "predict_ms_per_row": 1000 * pred_s / n}
    return oof, fs, timing


def error_table(df: pd.DataFrame, proba: np.ndarray, text_col: str = "text") -> pd.DataFrame:
    """Row-level table with prediction, confidence, margin and correctness."""
    top2 = np.sort(proba, axis=1)[:, ::-1][:, :2]
    pred = proba.argmax(1)
    out = pd.DataFrame(
        {
            "id": df["id"].values,
            "text": df[text_col].values,
            "gold": df["intent"].values,
            "pred": [ID2LABEL[int(i)] for i in pred],
            "confidence": top2[:, 0],
            "margin": top2[:, 0] - top2[:, 1],
            "n_words": df[text_col].str.split().str.len().values,
        }
    )
    out["correct"] = out["gold"] == out["pred"]
    return out


def categorise_errors(err: pd.DataFrame) -> pd.DataFrame:
    """Attach a coarse failure-category label to each misclassified row."""
    import re

    def cat(row) -> str:
        g, p, t = row["gold"], row["pred"], row["text"].lower()
        triad = {"shipment_information", "realtime_query", "analytics"}
        if row["n_words"] <= 2:
            return "short/underspecified"
        if g in triad and p in triad:
            if re.search(r"\b(last|past|yesterday|previous|ago|since|how many|count|percent|average|trend)\b", t):
                return "triad: temporal/aggregate cue"
            return "triad: semantic"
        if g == "knowledge_base" or p == "knowledge_base":
            return "knowledge_base vs operational"
        if g == "disruptions" or p == "disruptions":
            return "disruptions boundary"
        if g == "other" or p == "other":
            return "other / out-of-scope boundary"
        if g == "chitchat" or p == "chitchat":
            return "chitchat boundary"
        return "misc"

    e = err[~err["correct"]].copy()
    e["category"] = e.apply(cat, axis=1)
    return e


def summarize_errors(err: pd.DataFrame, k: int = 6) -> dict:
    e = categorise_errors(err)
    summary = {"n_errors": int(len(e)), "categories": {}}
    for c, grp in e.groupby("category"):
        grp = grp.sort_values("confidence", ascending=False)
        summary["categories"][c] = {
            "count": int(len(grp)),
            "mean_confidence": float(grp["confidence"].mean()),
            "examples": grp[["text", "gold", "pred", "confidence"]].head(k).assign(
                text=lambda d: d["text"].str.slice(0, 110)
            ).to_dict(orient="records"),
        }
    summary["mean_conf_correct"] = float(err.loc[err["correct"], "confidence"].mean())
    summary["mean_conf_wrong"] = float(err.loc[~err["correct"], "confidence"].mean())
    summary["high_conf_errors_ge_0.9"] = int(((~err["correct"]) & (err["confidence"] >= 0.9)).sum())
    return summary
