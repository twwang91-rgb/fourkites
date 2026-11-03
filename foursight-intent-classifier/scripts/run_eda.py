#!/usr/bin/env python
"""Exploratory data analysis for the FourSight intent dataset.

Writes reports/eda.json, reports/eda.md and figures under reports/figures/.
Only training data is analysed in depth; the test set is touched only for
schema / length / overlap statistics (never for labels).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from foursight_intent.data import cv_folds, leakage_report, load_test, load_train
from foursight_intent.labels import LABELS
from foursight_intent.text import REFERENCE_ID, dedup_key
from foursight_intent.utils import get_logger, save_json

log = get_logger("eda")

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # validated reference palette

CUES = {
    "temporal_past": r"\b(last|past|previous|yesterday|ago|q[1-4]|quarter|ytd|mtd|since)\b",
    "temporal_now": r"\b(now|currently|current|today|right now|live|at the moment|tonight)\b",
    "temporal_future": r"\b(tomorrow|next week|next month|upcoming|scheduled)\b",
    "aggregate": r"\b(how many|count|number of|total|average|avg|percent|percentage|%|trend|breakdown|top \d+|most|least|rate|median|volume|summary|report)\b",
    "eta_status": r"\b(eta|status|arriv\w*|deliver\w*|pick ?up|appointment|appt)\b",
    "where": r"\bwhere\b",
    "track": r"\btrack\w*\b",
    "how_to": r"\b(how (do|can|to|does|should)|what is (a|an|the)|what does|explain|difference between|meaning of|mean)\b",
    "disruption": r"\b(weather|storm|hurricane|typhoon|wildfire|flood|congestion|strike|closure|disruption|impact\w*|risk|delay\w* due|port condition|border|canal)\b",
    "greeting": r"\b(hi|hello|hey|thanks|thank you|good morning|good afternoon|what can you do)\b",
    "reference_id": None,  # handled by REFERENCE_ID regex
    "multi_ids": None,
}


def cue_flags(text: str) -> dict[str, bool]:
    t = text.lower()
    flags = {k: bool(re.search(p, t)) for k, p in CUES.items() if p}
    ids = REFERENCE_ID.findall(text)
    flags["reference_id"] = len(ids) > 0
    flags["multi_ids"] = len(ids) >= 2
    return flags


def bar(ax, labels, values, title, color=None, horizontal=False):
    if horizontal:
        ax.barh(labels, values, color=color or PALETTE[0])
        ax.invert_yaxis()
    else:
        ax.bar(labels, values, color=color or PALETTE[0])
        ax.tick_params(axis="x", rotation=35)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/train.csv"))
    ap.add_argument("--test", default=str(ROOT / "data/test.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_train(a.train), load_test(a.test)
    report: dict = {"n_train": len(train), "n_test": len(test)}

    # ---- class distribution -------------------------------------------------------
    counts = train["intent"].value_counts().reindex(LABELS).fillna(0).astype(int)
    report["class_counts"] = counts.to_dict()
    report["class_fraction"] = (counts / len(train)).round(4).to_dict()
    report["imbalance_ratio_max_min"] = float(counts.max() / counts.min())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bar(ax, LABELS, counts.values, "Training class distribution (n=1,904)", PALETTE[0])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 8, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    # ---- lengths ------------------------------------------------------------------
    for name, df in [("train", train), ("test", test)]:
        words = df["text"].str.split().str.len()
        chars = df["text"].str.len()
        report[f"{name}_length"] = {
            "words_mean": float(words.mean()),
            "words_median": float(words.median()),
            "words_p95": float(words.quantile(0.95)),
            "words_max": int(words.max()),
            "chars_median": float(chars.median()),
            "chars_max": int(chars.max()),
            "pct_rows_le_3_words": float((words <= 3).mean()),
            "pct_rows_gt_128_words": float((words > 128).mean()),
            "multiline_rows": int(df["question"].str.contains("\n").sum()),
        }
    train["n_words"] = train["text"].str.split().str.len()
    per_class_len = train.groupby("intent")["n_words"].agg(["mean", "median", "max"]).reindex(LABELS)
    report["words_per_class"] = per_class_len.round(2).to_dict(orient="index")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = [np.clip(train.loc[train.intent == c, "n_words"], 0, 60) for c in LABELS]
    ax.boxplot(data, tick_labels=LABELS, showfliers=False, medianprops={"color": "#2a78d6", "linewidth": 1.6}, boxprops={"color": "#52514e"}, whiskerprops={"color": "#52514e"}, capprops={"color": "#52514e"})
    ax.set_title("Words per question by class (clipped at 60)")
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "length_by_class.png", dpi=160)
    plt.close(fig)

    # ---- duplicates ---------------------------------------------------------------
    groups = train.groupby("group")["intent"].agg(list)
    dup_groups = groups[groups.str.len() > 1]
    conflict = dup_groups[dup_groups.map(lambda v: len(set(v)) > 1)]
    report["duplicates"] = {
        "exact_duplicate_groups": int(len(dup_groups)),
        "rows_in_duplicate_groups": int(dup_groups.str.len().sum()),
        "conflicting_label_groups": int(len(conflict)),
        "rows_in_conflicting_groups": int(conflict.str.len().sum()),
        "conflict_examples": [
            {"question": q[:100], "labels": dict(Counter(v))} for q, v in conflict.items()
        ][:40],
    }
    # Irreducible error from conflicts: best constant per duplicate group
    irreducible = sum(len(v) - Counter(v).most_common(1)[0][1] for v in groups)
    report["duplicates"]["min_errors_from_label_conflicts"] = int(irreducible)
    report["duplicates"]["accuracy_ceiling_from_conflicts"] = float(1 - irreducible / len(train))

    # ---- near duplicates (templated questions) --------------------------------------
    prefix3 = Counter(" ".join(g.split()[:3]) for g in train["group"])
    report["common_3word_prefixes"] = prefix3.most_common(20)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = tfidf.fit_transform(train["group"])
    sims = (X @ X.T).toarray()
    np.fill_diagonal(sims, 0)
    near = np.argwhere(np.triu(sims, 1) >= 0.9)
    near_pairs = [(int(i), int(j)) for i, j in near if train.iloc[i]["group"] != train.iloc[j]["group"]]
    near_conf = [(i, j) for i, j in near_pairs if train.iloc[i]["intent"] != train.iloc[j]["intent"]]
    report["near_duplicates"] = {
        "pairs_cosine_ge_0.9_not_exact": len(near_pairs),
        "of_which_conflicting_label": len(near_conf),
        "examples": [
            {
                "a": train.iloc[i]["text"][:90],
                "a_label": train.iloc[i]["intent"],
                "b": train.iloc[j]["text"][:90],
                "b_label": train.iloc[j]["intent"],
            }
            for i, j in near_conf[:15]
        ],
    }

    # ---- train/test overlap -------------------------------------------------------
    test_keys = test["question"].map(dedup_key)
    overlap_mask = test_keys.isin(set(train["group"]))
    test_dups = test_keys.value_counts()
    report["train_test_overlap"] = {
        "test_rows_with_exact_train_match": int(overlap_mask.sum()),
        "test_unique_questions": int(test_keys.nunique()),
        "test_internal_duplicate_rows": int((test_dups[test_dups > 1]).sum()),
        "test_most_repeated": [[q[:80], int(c)] for q, c in test_dups.head(8).items()],
        "test_pct_rows_in_duplicate_groups": float((test_dups[test_dups > 1]).sum() / len(test)),
    }

    # ---- lexical cues by class ----------------------------------------------------
    flags = pd.DataFrame([cue_flags(t) for t in train["text"]])
    flags["intent"] = train["intent"].values
    prevalence = flags.groupby("intent").mean().reindex(LABELS)
    report["cue_prevalence_by_class"] = prevalence.round(3).to_dict(orient="index")
    tflags = pd.DataFrame([cue_flags(t) for t in test["text"]])
    report["cue_prevalence_test"] = tflags.mean().round(3).to_dict()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    im = ax.imshow(prevalence.values.T, cmap="Blues", aspect="auto", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right")
    ax.set_yticks(range(prevalence.shape[1]))
    ax.set_yticklabels(prevalence.columns)
    for i in range(prevalence.shape[0]):
        for j in range(prevalence.shape[1]):
            ax.text(i, j, f"{100 * prevalence.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Share of class rows containing each lexical cue (%)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "cue_prevalence.png", dpi=160)
    plt.close(fig)

    # ---- bare ids / short questions -----------------------------------------------
    bare = train["text"].str.fullmatch(r"[#\s]*[A-Za-z]{0,5}[\d\-]{5,}[A-Za-z]{0,2}[;\s]*")
    report["bare_id_questions"] = {
        "train_by_class": train.loc[bare, "intent"].value_counts().to_dict(),
        "test_count": int(test["text"].str.fullmatch(r"[#\s]*[A-Za-z]{0,5}[\d\-]{5,}[A-Za-z]{0,2}[;\s]*").sum()),
    }
    short = train[train["n_words"] <= 2]
    report["short_questions_le_2_words"] = {
        "count": int(len(short)),
        "by_class": short["intent"].value_counts().to_dict(),
        "examples": short[["text", "intent"]].head(40).values.tolist(),
    }

    # ---- most discriminative tokens per class (logistic regression weights) --------
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xw = vec.fit_transform(train["text"].str.lower())
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xw, train["intent"])
    feats = np.array(vec.get_feature_names_out())
    top_tokens = {}
    for i, c in enumerate(clf.classes_):
        top = np.argsort(clf.coef_[i])[::-1][:15]
        top_tokens[c] = feats[top].tolist()
    report["top_tokens_per_class"] = top_tokens

    # ---- temporal expression x triad ----------------------------------------------
    triad = train[train["intent"].isin(["shipment_information", "realtime_query", "analytics"])]
    tri_flags = pd.DataFrame([cue_flags(t) for t in triad["text"]])
    tri_flags["intent"] = triad["intent"].values
    report["triad_cue_prevalence"] = tri_flags.groupby("intent").mean().round(3).to_dict(orient="index")

    # ---- languages ----------------------------------------------------------------
    def lang(q: str) -> str:
        if re.search(r"[぀-ヿ一-鿿]", q):
            return "ja/zh"
        ql = q.lower()
        es = len(re.findall(r"\b(que|qué|cómo|dónde|cuándo|los|las|una|por|para|está|tiene|puedes|mis|envío|carga|embarque|pedido)\b", ql))
        pt = len(re.findall(r"\b(você|não|onde|quais|estão|entrega|transportadora|isso|pra|uma|com)\b", ql))
        fr = len(re.findall(r"\b(est|une|les|des|pour|quel|quelle|transporteur|livraison|comment|pas|vous|nous|dans|avec)\b", ql))
        if max(es, pt, fr) >= 2 and len(ql.split()) >= 3:
            return ["es", "pt", "fr"][[es, pt, fr].index(max(es, pt, fr))]
        return "en"

    report["language_heuristic"] = {
        "train": train["text"].map(lang).value_counts().to_dict(),
        "test": test["text"].map(lang).value_counts().to_dict(),
    }

    # ---- validation protocol leakage check ---------------------------------------
    strat = cv_folds(train, 5, a.seed, "stratified")
    grouped = cv_folds(train, 5, a.seed, "grouped")
    report["leakage"] = {
        "stratified_5fold": leakage_report(train, strat),
        "grouped_5fold": leakage_report(train, grouped),
    }
    report["grouped_fold_class_counts"] = [
        train.iloc[f.val_idx]["intent"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()
        for f in grouped
    ]

    save_json(report, out / "eda.json")
    write_markdown(report, out / "eda.md")
    log.info("EDA written to %s", out)


def write_markdown(r: dict, path: Path) -> None:
    L = []
    L.append("# EDA summary\n")
    L.append(f"Train rows: {r['n_train']}  |  Test rows: {r['n_test']}\n")
    L.append("## Class distribution\n")
    L.append("| class | n | % |\n|---|---:|---:|")
    for c in LABELS:
        L.append(f"| {c} | {r['class_counts'][c]} | {100 * r['class_fraction'][c]:.1f} |")
    L.append(f"\nImbalance ratio (max/min): {r['imbalance_ratio_max_min']:.0f}x\n")
    L.append("## Lengths\n")
    for name in ("train", "test"):
        d = r[f"{name}_length"]
        L.append(
            f"- **{name}**: median {d['words_median']:.0f} words, p95 {d['words_p95']:.0f}, max {d['words_max']}; "
            f"{100 * d['pct_rows_le_3_words']:.1f}% rows ≤3 words; {100 * d['pct_rows_gt_128_words']:.1f}% rows >128 words; "
            f"{d['multiline_rows']} multi-line (pasted lists/tables)."
        )
    L.append("\n## Duplicates and label noise\n")
    d = r["duplicates"]
    L.append(
        f"- {d['exact_duplicate_groups']} exact-duplicate groups covering {d['rows_in_duplicate_groups']} rows; "
        f"**{d['conflicting_label_groups']} groups carry conflicting labels** ({d['rows_in_conflicting_groups']} rows)."
    )
    L.append(
        f"- Best possible training-set accuracy given those conflicts: {100 * d['accuracy_ceiling_from_conflicts']:.2f}% "
        f"({d['min_errors_from_label_conflicts']} unavoidable errors)."
    )
    L.append("- Examples: " + "; ".join(f"`{e['question']}` → {e['labels']}" for e in d["conflict_examples"][:8]))
    nd = r["near_duplicates"]
    L.append(
        f"- Near duplicates (char-ngram cosine ≥ 0.9, not exact): {nd['pairs_cosine_ge_0.9_not_exact']} pairs, "
        f"{nd['of_which_conflicting_label']} with conflicting labels."
    )
    L.append("\n## Train/test overlap\n")
    o = r["train_test_overlap"]
    L.append(
        f"- {o['test_rows_with_exact_train_match']} test rows match a training question exactly; "
        f"test has {o['test_unique_questions']} unique questions and {o['test_internal_duplicate_rows']} rows inside internal duplicate groups "
        f"({100 * o['test_pct_rows_in_duplicate_groups']:.0f}%)."
    )
    L.append("\n## Validation leakage check (5-fold, seed 42)\n")
    L.append(
        f"- Plain stratified k-fold: {r['leakage']['stratified_5fold']['total_leaked_rows']} validation rows have an exact-text twin in their training fold."
    )
    L.append(f"- Grouped stratified k-fold: {r['leakage']['grouped_5fold']['total_leaked_rows']} leaked rows (by construction).")
    L.append("\n## Lexical cues (share of class rows containing the cue)\n")
    cues = list(next(iter(r["cue_prevalence_by_class"].values())).keys())
    L.append("| cue | " + " | ".join(c[:10] for c in LABELS) + " |")
    L.append("|---|" + "---:|" * len(LABELS))
    for cue in cues:
        L.append(f"| {cue} | " + " | ".join(f"{100 * r['cue_prevalence_by_class'][c][cue]:.0f}" for c in LABELS) + " |")
    L.append("\n## Bare reference-number questions\n")
    L.append(f"- Train: {r['bare_id_questions']['train_by_class']}; test: {r['bare_id_questions']['test_count']} rows.")
    L.append("\n## Languages (heuristic)\n")
    L.append(f"- Train: {r['language_heuristic']['train']}; test: {r['language_heuristic']['test']}")
    L.append("\n## Top tokens per class (TF-IDF logistic regression weights)\n")
    for c in LABELS:
        L.append(f"- **{c}**: " + ", ".join(r["top_tokens_per_class"][c][:12]))
    L.append("\n## Design implications\n")
    L.append("1. Macro-F1 dominates the score, and three classes have < 60 examples: use class weighting and judge models on grouped CV, not a single split.")
    L.append(
        f"2. {d['conflicting_label_groups']} conflicting-label duplicate groups and many one-word context fragments (`yes`, `both`, `eta`) are irreducible single-turn noise; "
        "they motivate the multi-turn design and the needs_review flag."
    )
    leak = r["leakage"]["stratified_5fold"]["total_leaked_rows"]
    L.append(
        f"3. Duplicates must be grouped during validation, otherwise {leak} of {r['n_train']} validation rows ({100 * leak / r['n_train']:.0f}%) "
        "have a verbatim twin in training and scores are inflated."
    )
    L.append("4. Reference numbers are strong realtime_query evidence (50% of that class) but 18% of shipment_information rows also contain them: the triad boundary is semantic, not lexical.")
    L.append("5. Temporal-past and aggregate cues are the main lexical signal for analytics; `today` leans realtime_query.")
    L.append("6. A long tail of pasted tables (up to 1,533 words) means truncation policy matters: keep the head and a short tail.")
    L.append("7. Non-English rows exist in both splits (French only in test): the encoder must not fall over on them, but there are too few to tune for.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
