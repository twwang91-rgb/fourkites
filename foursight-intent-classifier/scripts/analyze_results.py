#!/usr/bin/env python
"""Post-hoc analysis of out-of-fold predictions for one candidate.

For a transformer experiment (``--name X`` -> reports/results/oof_logits_X.npy) or a
baseline (``--baseline tfidf_logreg`` -> reports/results/oof_<key>_grouped.npy, already
probabilities) this script computes, strictly on out-of-fold rows:

  * official scores, per-class metrics, confusion matrix, confused pairs, triad errors
  * calibration: cross-fitted temperature scaling (T fitted on 4 folds, applied to the
    5th), ECE / NLL / Brier before and after, reliability diagram
  * needs_review policy: threshold chosen on 4 folds, evaluated on the 5th (pooled),
    plus the deployment threshold fitted on all OOF rows
  * risk-coverage curve and AURC
  * rule-layer ablation (apply_rules on calibrated probabilities, several max_conf)
  * categorised error analysis with representative examples

Outputs reports/results/analysis_<name>.json and figures under reports/figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from foursight_intent.calibration import (
    calibration_summary,
    expected_calibration_error,
    fit_temperature,
    review_metrics,
    risk_coverage_curve,
    select_review_threshold,
    softmax,
)
from foursight_intent.data import cv_folds, load_train
from foursight_intent.evaluate import error_table, summarize_errors
from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.metrics import confusion_matrix, format_report, official_scores, triad_confusion
from foursight_intent.rules import apply_rules
from foursight_intent.utils import get_logger, save_json

log = get_logger("analyze")


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels([l.replace("_", "\n") for l in LABELS], fontsize=7)
    ax.set_yticklabels(LABELS, fontsize=7)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_reliability(cal: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, key, sub in zip(axes, ("reliability_raw", "reliability_calibrated"), ("raw softmax", f"temperature-scaled (T={cal['temperature']:.2f})")):
        bins = [b for b in cal[key] if b["count"]]
        conf = [b["confidence"] for b in bins]; acc = [b["accuracy"] for b in bins]; cnt = np.array([b["count"] for b in bins])
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.scatter(conf, acc, s=8 + 200 * cnt / cnt.max(), color="#2a78d6", alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.plot(conf, acc, color="#2a78d6", lw=1.2)
        ece = cal["raw" if key == "reliability_raw" else "calibrated"]["ece"]
        ax.set_title(f"{sub}\nECE={ece:.3f}", fontsize=9)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_risk_coverage(rc: dict, path: Path, title: str, policy=None) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot([p["coverage"] for p in rc["points"]], [p["risk"] for p in rc["points"]], color="#2a78d6", lw=1.8)
    ax.axhline(rc["full_risk"], ls="--", color="#52514e", lw=1, label=f"full-coverage error {rc['full_risk']:.3f}")
    if policy is not None:
        ax.axvline(1 - policy.flag_rate, ls=":", color="#eb6834", lw=1.4, label=f"review policy: cover {1 - policy.flag_rate:.0%}, err {1 - policy.accuracy_retained:.3f}")
    ax.set_xlabel("coverage (share of questions answered automatically)"); ax.set_ylabel("error rate on covered"); ax.set_title(f"{title}\nAURC={rc['aurc']:.3f}", fontsize=9)
    ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def cross_fitted(logits: np.ndarray, y: np.ndarray, folds, max_flag_rate: float, metric: str):
    """Temperature and review threshold chosen on 4 folds, applied to the 5th."""
    cal_probs = np.zeros_like(logits)
    flagged = np.zeros(len(y), dtype=bool)
    per_fold = []
    for f in folds:
        tr, va = f.train_idx, f.val_idx
        T = fit_temperature(logits[tr], y[tr])
        p_tr, p_va = softmax(logits[tr] / T), softmax(logits[va] / T)
        pol = select_review_threshold(p_tr, y[tr], max_flag_rate, metric)
        cal_probs[va] = p_va
        s = p_va.max(1) if metric == "confidence" else (np.sort(p_va, 1)[:, -1] - np.sort(p_va, 1)[:, -2])
        flagged[va] = s < pol.threshold
        per_fold.append({"fold": f.index, "temperature": T, "threshold": pol.threshold})
    return cal_probs, flagged, per_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--name", help="transformer experiment name (reports/results/oof_logits_<name>.npy)")
    ap.add_argument("--baseline", help="baseline key, e.g. tfidf_logreg / embedding (probabilities)")
    ap.add_argument("--max-flag-rate", type=float, default=None)
    ap.add_argument("--review-metric", default="confidence", choices=["confidence", "margin"])
    ap.add_argument("--protocol", default=None, help="fold protocol the OOF predictions were produced with (default: config)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seed = int(cfg["seed"]); protocol = a.protocol or cfg["validation"]["protocol"]
    max_flag_rate = a.max_flag_rate or float(cfg["calibration"]["review"]["max_flag_rate"])
    df = load_train(ROOT / cfg["data"]["train_csv"])
    folds = cv_folds(df, int(cfg["validation"]["n_splits"]), seed, protocol)
    y = df["label"].to_numpy()
    res_dir = ROOT / "reports" / "results"; fig_dir = ROOT / "reports" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    if a.name:
        name = a.name
        logits = np.load(res_dir / f"oof_logits_{name}.npy")
        probs_raw = softmax(logits)
    else:
        name = f"baseline_{a.baseline}"
        probs_raw = np.load(res_dir / f"oof_{a.baseline}_{protocol}.npy")
        logits = np.log(np.clip(probs_raw, 1e-9, 1.0))  # treat log-probs as logits for temperature scaling
    assert not np.isnan(logits).any(), "OOF logits contain NaN: were all folds run?"

    gold = df["intent"].tolist()
    pred_raw = [ID2LABEL[int(i)] for i in probs_raw.argmax(1)]
    scores = official_scores(gold, pred_raw)
    log.info("\n%s", format_report(scores, f"{name}: pooled out-of-fold ({protocol} {len(folds)}-fold)"))
    out: dict = {"name": name, "protocol": protocol, "scores": scores, "triad_errors": triad_confusion(scores)}
    plot_confusion(confusion_matrix(gold, pred_raw), fig_dir / f"confusion_{name}.png", f"{name}: out-of-fold confusion (rows = gold)")

    # ---- calibration -----------------------------------------------------------
    cal = calibration_summary(logits, y)
    cal_probs_cf, flagged_cf, per_fold_cal = cross_fitted(logits, y, folds, max_flag_rate, a.review_metric)
    correct = (probs_raw.argmax(1) == y)
    out["calibration"] = {
        "fitted_on_all_oof": {k: cal[k] for k in ("temperature", "raw", "calibrated", "accuracy")},
        "cross_fitted_ece": expected_calibration_error(cal_probs_cf.max(1), correct),
        "per_fold": per_fold_cal,
    }
    plot_reliability(cal, fig_dir / f"reliability_{name}.png", f"{name}: reliability (out-of-fold)")
    log.info("calibration: T=%.3f  ECE raw=%.4f -> calibrated=%.4f (cross-fitted %.4f)  NLL %.4f -> %.4f", cal["temperature"], cal["raw"]["ece"], cal["calibrated"]["ece"], out["calibration"]["cross_fitted_ece"], cal["raw"]["nll"], cal["calibrated"]["nll"])

    # ---- review flag -----------------------------------------------------------
    cal_probs = softmax(logits / cal["temperature"])
    deploy_pol = select_review_threshold(cal_probs, y, max_flag_rate, a.review_metric)
    wrong = probs_raw.argmax(1) != y
    cf = {
        "flag_rate": float(flagged_cf.mean()), "error_recall": float((flagged_cf & wrong).sum() / max(wrong.sum(), 1)),
        "precision": float((flagged_cf & wrong).sum() / max(flagged_cf.sum(), 1)),
        "accuracy_retained": float(correct[~flagged_cf].mean()), "n_flagged": int(flagged_cf.sum()), "n_errors_flagged": int((flagged_cf & wrong).sum()),
    }
    # official score on the auto-answered subset (what the user sees without review)
    keep = ~flagged_cf
    retained_scores = official_scores([g for g, k in zip(gold, keep) if k], [p for p, k in zip(pred_raw, keep) if k])
    sweep = [review_metrics(cal_probs, y, t, a.review_metric).__dict__ for t in np.linspace(0.3, 0.95, 14)]
    rc = risk_coverage_curve(cal_probs, y, a.review_metric)
    adj_for_review, _ = apply_rules(df["text"].tolist(), cal_probs, max_conf=0.6)
    deploy_pol_rules = select_review_threshold(adj_for_review, y, max_flag_rate, a.review_metric)
    out["review"] = {"metric": a.review_metric, "max_flag_rate": max_flag_rate, "deployment_policy": deploy_pol.__dict__, "deployment_policy_with_rules": deploy_pol_rules.__dict__, "cross_fitted": cf,
                     "OVERALL_on_unflagged_cross_fitted": retained_scores["OVERALL"], "accuracy_on_unflagged_cross_fitted": retained_scores["accuracy"],
                     "threshold_sweep": sweep, "risk_coverage": {"aurc": rc["aurc"], "full_risk": rc["full_risk"]}}
    plot_risk_coverage(rc, fig_dir / f"risk_coverage_{name}.png", f"{name}: selective prediction", deploy_pol)
    log.info("review flag (%s < %.3f): flags %.1f%% of traffic, catches %.1f%% of errors, precision %.2f, accuracy on the rest %.3f (cross-fitted: flag %.1f%%, recall %.1f%%, precision %.2f, retained acc %.3f)",
             a.review_metric, deploy_pol.threshold, 100 * deploy_pol.flag_rate, 100 * deploy_pol.error_recall, deploy_pol.precision, deploy_pol.accuracy_retained,
             100 * cf["flag_rate"], 100 * cf["error_recall"], cf["precision"], cf["accuracy_retained"])
    # confidence stats for correct vs wrong
    out["confidence"] = {"mean_conf_correct": float(cal_probs.max(1)[correct].mean()), "mean_conf_wrong": float(cal_probs.max(1)[wrong].mean()),
                         "errors_with_conf_ge_0.9": int((wrong & (cal_probs.max(1) >= 0.9)).sum()), "n_errors": int(wrong.sum()),
                         "margin_mean_correct": float((np.sort(cal_probs, 1)[:, -1] - np.sort(cal_probs, 1)[:, -2])[correct].mean()),
                         "margin_mean_wrong": float((np.sort(cal_probs, 1)[:, -1] - np.sort(cal_probs, 1)[:, -2])[wrong].mean())}

    # ---- rule ablation ---------------------------------------------------------
    abl = []
    for max_conf in (0.0, 0.5, 0.6, 0.7, 0.8, 1.01):
        adj, fired = apply_rules(df["text"].tolist(), cal_probs, max_conf=max_conf)
        pr = [ID2LABEL[int(i)] for i in adj.argmax(1)]
        sc = official_scores(gold, pr)
        changed = int((adj.argmax(1) != cal_probs.argmax(1)).sum())
        fixed = int(((adj.argmax(1) == y) & (cal_probs.argmax(1) != y)).sum()); broke = int(((adj.argmax(1) != y) & (cal_probs.argmax(1) == y)).sum())
        from collections import Counter

        abl.append({"max_conf": max_conf, "rows_eligible": int((cal_probs.max(1) < max_conf).sum()), "rows_changed": changed, "fixed": fixed, "broke": broke,
                    "accuracy": sc["accuracy"], "macro_f1": sc["macro_f1"], "OVERALL": sc["OVERALL"], "per_class_f1": {c: sc["per_class"][c]["f1"] for c in LABELS},
                    "triad_errors": triad_confusion(sc)["total_triad_errors"], "rules_fired": dict(Counter(r for fs in fired for r in fs))})
    out["rule_ablation"] = {"baseline_OVERALL": scores["OVERALL"], "sweep": abl}
    best_abl = max(abl, key=lambda r: r["OVERALL"])
    log.info("rules: no rules OVERALL=%.4f; best max_conf=%.2f -> OVERALL=%.4f (fixed %d, broke %d)", scores["OVERALL"], best_abl["max_conf"], best_abl["OVERALL"], best_abl["fixed"], best_abl["broke"])
    for r in abl:
        log.info("   max_conf=%.2f eligible=%4d changed=%3d fixed=%3d broke=%3d OVERALL=%.4f macroF1=%.4f acc=%.4f triad_err=%d", r["max_conf"], r["rows_eligible"], r["rows_changed"], r["fixed"], r["broke"], r["OVERALL"], r["macro_f1"], r["accuracy"], r["triad_errors"])

    # ---- error analysis --------------------------------------------------------
    err = error_table(df, cal_probs)
    es = summarize_errors(err, k=8)
    # rare classes
    es["rare_class_recall"] = {c: scores["per_class"][c]["recall"] for c in ("disruptions", "chitchat", "other")}
    es["short_questions"] = {"n": int((err["n_words"] <= 2).sum()), "accuracy": float(err.loc[err["n_words"] <= 2, "correct"].mean())}
    es["long_questions_gt_64_words"] = {"n": int((err["n_words"] > 64).sum()), "accuracy": float(err.loc[err["n_words"] > 64, "correct"].mean()) if (err["n_words"] > 64).any() else None}
    # label-conflict rows: duplicates with conflicting gold labels are irreducible
    grp = df.groupby("group")["intent"].agg(lambda v: len(set(v)) > 1)
    conflict_mask = df["group"].map(grp).to_numpy()
    es["conflicting_duplicate_rows"] = {"n": int(conflict_mask.sum()), "accuracy": float(err.loc[conflict_mask, "correct"].mean()), "share_of_all_errors": float((~err["correct"] & conflict_mask).sum() / max(wrong.sum(), 1))}
    out["errors"] = es
    err.sort_values("confidence", ascending=False).to_csv(res_dir / f"oof_predictions_{name}.csv", index=False)
    log.info("errors: %d total; %d with calibrated confidence >= 0.9; short(<=2 words) acc=%.3f; conflicting-duplicate rows acc=%.3f (%.0f%% of errors)",
             es["n_errors"], out["confidence"]["errors_with_conf_ge_0.9"], es["short_questions"]["accuracy"], es["conflicting_duplicate_rows"]["accuracy"], 100 * es["conflicting_duplicate_rows"]["share_of_all_errors"])
    for c, d in sorted(es["categories"].items(), key=lambda kv: -kv[1]["count"]):
        log.info("   %-34s n=%3d  mean_conf=%.2f", c, d["count"], d["mean_confidence"])

    save_json(out, res_dir / f"analysis_{name}.json")
    write_error_markdown(out, ROOT / "reports" / f"error_analysis_{name}.md")
    log.info("written %s", res_dir / f"analysis_{name}.json")


def write_error_markdown(out: dict, path: Path) -> None:
    """Human-readable error analysis: categories, examples, calibration and review numbers."""
    s = out["scores"]; es = out["errors"]; cal = out["calibration"]["fitted_on_all_oof"]; pol = out["review"]["deployment_policy"]; cf = out["review"]["cross_fitted"]
    L = [f"# Error analysis — {out['name']} (pooled out-of-fold, {out['protocol']} CV)\n",
         f"accuracy {s['accuracy']:.4f} · macro-F1 {s['macro_f1']:.4f} · **OVERALL {s['OVERALL']:.4f}** · errors {es['n_errors']} / {s['n']}\n",
         "## Per-class\n", "| class | precision | recall | F1 | support |", "|---|---:|---:|---:|---:|"]
    for c in LABELS:
        m = s["per_class"][c]; L.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    L += ["\n## Most confused pairs (gold → predicted)\n"] + [f"- {g} → {p}: {n}" for n, g, p in s["confused_pairs"][:10]]
    L += [f"\nOperational-triad errors: {out['triad_errors']}\n", "## Calibration and review flag\n",
          f"- temperature T = {cal['temperature']:.3f}; ECE {cal['raw']['ece']:.4f} → {cal['calibrated']['ece']:.4f}; NLL {cal['raw']['nll']:.4f} → {cal['calibrated']['nll']:.4f}; Brier {cal['raw']['brier']:.4f} → {cal['calibrated']['brier']:.4f}",
          f"- mean calibrated confidence: correct {out['confidence']['mean_conf_correct']:.3f} vs wrong {out['confidence']['mean_conf_wrong']:.3f}; margin {out['confidence']['margin_mean_correct']:.3f} vs {out['confidence']['margin_mean_wrong']:.3f}; {out['confidence']['errors_with_conf_ge_0.9']} errors with confidence ≥ 0.9",
          f"- needs_review = {pol['metric']} < {pol['threshold']:.3f}: flags {100 * pol['flag_rate']:.1f}% of rows, error recall {100 * pol['error_recall']:.1f}%, precision {pol['precision']:.2f}, accuracy on unflagged {pol['accuracy_retained']:.3f}",
          f"- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag {100 * cf['flag_rate']:.1f}%, error recall {100 * cf['error_recall']:.1f}%, precision {cf['precision']:.2f}, accuracy on unflagged {cf['accuracy_retained']:.3f}, OVERALL on unflagged {out['review']['OVERALL_on_unflagged_cross_fitted']:.4f}",
          f"- AURC {out['review']['risk_coverage']['aurc']:.4f} (full-coverage risk {out['review']['risk_coverage']['full_risk']:.4f})\n",
          "## Rule-layer ablation (calibrated probabilities)\n", "| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out["rule_ablation"]["sweep"]:
        L.append(f"| {r['max_conf']:.2f} | {r['rows_eligible']} | {r['rows_changed']} | {r['fixed']} | {r['broke']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['OVERALL']:.4f} | {r['triad_errors']} |")
    L += ["\n## Failure categories\n",
          f"Short (≤2 words) inputs: n={es['short_questions']['n']}, accuracy {es['short_questions']['accuracy']:.3f}. "
          f"Rows in conflicting-label duplicate groups: n={es['conflicting_duplicate_rows']['n']}, accuracy {es['conflicting_duplicate_rows']['accuracy']:.3f} ({100 * es['conflicting_duplicate_rows']['share_of_all_errors']:.0f}% of all errors). "
          f"Rare-class recall: {', '.join(f'{k} {v:.2f}' for k, v in es['rare_class_recall'].items())}.\n"]
    for c, d in sorted(es["categories"].items(), key=lambda kv: -kv[1]["count"]):
        L.append(f"### {c} — {d['count']} errors, mean confidence {d['mean_confidence']:.2f}\n")
        for e in d["examples"]:
            L.append(f"- `{e['text']}` — gold **{e['gold']}**, predicted {e['pred']} ({e['confidence']:.2f})")
        L.append("")
    path.write_text("\n".join(L).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
