#!/usr/bin/env python
"""Build the model-comparison table and evaluate OOF ensembles / rule variants.

Reads every reports/results/transformer_*.json (single-model CV runs) and
baselines_<protocol>.json, optionally measures CPU inference latency and artefact size,
evaluates per-rule ablations and probability-averaging ensembles on out-of-fold
predictions (members' temperatures cross-fitted per fold), runs a paired bootstrap over
rows, and writes reports/results/comparison.json + reports/comparison.md.

Ensembles are built greedily from the *primary* candidates (one configuration per base
model, ranked by their single-model OOF OVERALL): top-2, top-3, ... top-k, plus
"best single + frozen-embedding logistic regression". Robustness / ablation runs (other
seeds, other class weights) are reported as candidates but never used as members, which
keeps the number of ensembles evaluated small and fixed in advance.
For every ensemble the pooled OOF probabilities are saved as
reports/results/oof_logits_<ensemble>.npy (log-probabilities) together with a summary
reports/results/transformer_<ensemble>.json (kind = "ensemble") so analyze_results.py,
score_oof.py and train_final.py can treat it like any other candidate.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml

from foursight_intent.calibration import fit_temperature, softmax
from foursight_intent.data import cv_folds, load_train
from foursight_intent.labels import ID2LABEL, LABELS
from foursight_intent.metrics import official_scores, triad_confusion
from foursight_intent.rules import RULES, apply_rules
from foursight_intent.utils import get_logger, save_json

log = get_logger("compare")

PARAM_COUNTS_M = {  # fp32 parameter counts in millions (measured when loading the models)
    "distilbert/distilbert-base-uncased": 67, "microsoft/deberta-v3-small": 142, "FacebookAI/roberta-base": 125,
    "intfloat/multilingual-e5-small": 118, "BAAI/bge-small-en-v1.5": 33, "sentence-transformers/all-MiniLM-L6-v2": 22,
}
# One configuration per base model; fixed before any ensemble was evaluated.
PRIMARY_CANDIDATES = ["deberta_v3_small", "e5_small_ft", "roberta_base", "distilbert", "bge_small_ft"]


def measure_latency(model_name: str, texts: list[str], max_length: int = 128, tail: int = 16, threads: int = 4, batch_size: int = 32) -> dict:
    """CPU forward-pass latency of a base encoder with a 7-way head (weights do not affect speed)."""
    import torch
    import transformers

    from foursight_intent.tokenization import encode_texts, pad_batch, special_token_template

    transformers.logging.set_verbosity_error()
    torch.set_num_threads(threads)
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7, dtype=torch.float32).eval()
    tpl = special_token_template(tok)
    seqs = encode_texts(tok, texts, max_length, tail, tpl)
    pad = tok.pad_token_id or 0
    with torch.no_grad():
        for _ in range(2):
            ids, mask = pad_batch(seqs[:batch_size], pad); model(input_ids=ids, attention_mask=mask)
        t0 = time.perf_counter()
        for i in range(0, len(seqs), batch_size):
            ids, mask = pad_batch(seqs[i : i + batch_size], pad); model(input_ids=ids, attention_mask=mask)
        batched_ms = 1000 * (time.perf_counter() - t0) / len(seqs)
        single = seqs[:50]
        t0 = time.perf_counter()
        for s in single:
            ids, mask = pad_batch([s], pad); model(input_ids=ids, attention_mask=mask)
        single_ms = 1000 * (time.perf_counter() - t0) / len(single)
    size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    return {"cpu_ms_per_row_batched": batched_ms, "cpu_ms_per_question_single": single_ms, "fp32_size_mb": size_mb, "threads": threads}


def bootstrap_ci(gold: list[str], preds: dict[str, list[str]], n_boot: int = 1000, seed: int = 0):
    """Paired bootstrap over rows: OVERALL CI per candidate (same resamples for everyone)."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    gold_a = np.array(gold)
    arrs = {k: np.array(v) for k, v in preds.items()}
    samples = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        g = gold_a[idx].tolist()
        for k, a in arrs.items():
            samples[k].append(official_scores(g, a[idx].tolist())["OVERALL"])
    out = {k: {"mean": float(np.mean(s)), "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]} for k, s in samples.items()}
    return out, samples


def per_fold_overall(pred_idx: np.ndarray, gold: list[str], folds) -> list[float]:
    return [official_scores([gold[i] for i in f.val_idx], [ID2LABEL[int(pred_idx[i])] for i in f.val_idx])["OVERALL"] for f in folds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--latency", action="store_true", help="measure CPU inference latency / size of every fine-tuned candidate's base model on the test questions")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    protocol = cfg["validation"]["protocol"]; seed = int(cfg["seed"])
    df = load_train(ROOT / cfg["data"]["train_csv"])
    folds = cv_folds(df, int(cfg["validation"]["n_splits"]), seed, protocol)
    gold = df["intent"].tolist(); y = df["label"].to_numpy(); texts = df["text"].tolist()
    res_dir = ROOT / "reports" / "results"

    rows: list[dict] = []
    probs_by_name: dict[str, np.ndarray] = {}
    exp_by_name: dict[str, dict] = {}
    # ---------------------------------------------------------------- single fine-tuned models
    for path in sorted(glob.glob(str(res_dir / "transformer_*.json"))):
        r = json.load(open(path))
        if r.get("kind") == "ensemble":
            continue  # recomputed below
        if r.get("protocol", protocol) != protocol:
            log.info("skipping %s: protocol %s != %s", path, r.get("protocol"), protocol); continue
        if r["rows_evaluated"] != len(df):
            log.warning("skipping %s: only %d rows evaluated", path, r["rows_evaluated"]); continue
        logits = np.load(res_dir / f"oof_logits_{r['name']}.npy")
        probs_by_name[r["name"]] = softmax(logits)
        exp_by_name[r["name"]] = r
        rows.append({
            "model": r["name"], "family": "fine-tuned encoder", "kind": "single", "base": r["config"]["model_name"], "params_M": PARAM_COUNTS_M.get(r["config"]["model_name"]),
            "seed": r["config"]["seed"], "class_weight": r["config"]["class_weight"], "primary": r["name"] in PRIMARY_CANDIDATES,
            "accuracy": r["accuracy_oof"], "macro_f1": r["macro_f1_oof"], "OVERALL": r["OVERALL_oof"], "fold_std": r["OVERALL_fold_std"],
            "per_class_f1": {c: r["per_class"][c]["f1"] for c in LABELS}, "triad_errors": r["triad_errors"]["total_triad_errors"],
            "train_seconds_5fold": r["train_seconds_total"], "best_epochs": r["best_epochs"], "lr": r["config"]["learning_rate"],
        })
    # CPU latency per base model: measured once and cached in reports/results/latency.json so
    # re-running the comparison (e.g. after adding a robustness run) reuses identical numbers.
    lat_path = res_dir / "latency.json"
    cache: dict[str, dict] = json.load(open(lat_path)) if lat_path.exists() else {}
    if a.latency:
        from foursight_intent.data import load_test

        test_texts = load_test(ROOT / cfg["data"]["test_csv"])["text"].tolist()
        for r in rows:
            if r["base"] in cache:
                continue
            try:
                cache[r["base"]] = measure_latency(r["base"], test_texts, cfg["text"]["max_length"], cfg["text"]["tail_tokens"])
                log.info("latency %-40s batched %.1f ms/row, single %.1f ms/question, %.0f MB fp32", r["base"], cache[r["base"]]["cpu_ms_per_row_batched"], cache[r["base"]]["cpu_ms_per_question_single"], cache[r["base"]]["fp32_size_mb"])
            except Exception as exc:
                log.warning("latency measurement failed for %s: %s", r["base"], exc)
        save_json(cache, lat_path)
    for r in rows:
        if r["base"] in cache:
            r["latency"] = cache[r["base"]]

    # ---------------------------------------------------------------- baselines
    bpath = res_dir / f"baselines_{protocol}.json"
    if bpath.exists():
        seen = set()
        for r in json.load(open(bpath))["results"]:
            if "error" in r:
                continue
            key = (r["family"], r.get("clf"), r.get("encoder"))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "model": r["model"], "family": "baseline: " + r["family"], "kind": "baseline", "base": r.get("encoder") or "tf-idf", "params_M": PARAM_COUNTS_M.get(r.get("encoder"), 0), "primary": False,
                "accuracy": r["accuracy_oof"], "macro_f1": r["macro_f1_oof"], "OVERALL": r["OVERALL_oof"], "fold_std": r["OVERALL_fold_std"],
                "per_class_f1": r["per_class_f1"], "triad_errors": r["triad_errors"]["total_triad_errors"], "predict_ms_per_row": r["predict_ms_per_row"],
            })
        for key in ("tfidf_logreg", "tfidf_svm", "embedding"):
            p = res_dir / f"oof_{key}_{protocol}.npy"
            if p.exists():
                probs_by_name[f"baseline_{key}"] = np.load(p)

    # ---------------------------------------------------------------- cross-fitted calibration per candidate
    def calibrated(name: str) -> np.ndarray:
        p = probs_by_name[name]
        logits = np.log(np.clip(p, 1e-9, 1))
        out = np.zeros_like(p)
        for f in folds:
            T = fit_temperature(logits[f.train_idx], y[f.train_idx])
            out[f.val_idx] = softmax(logits[f.val_idx] / T)
        return out

    cal = {n: calibrated(n) for n in probs_by_name}
    ft_names = [r["model"] for r in rows if r["kind"] == "single"]
    overall = {r["model"]: r["OVERALL"] for r in rows}
    for r in rows:
        if r["model"] in cal:
            r["per_fold_OVERALL"] = per_fold_overall(cal[r["model"]].argmax(1), gold, folds)

    # ---------------------------------------------------------------- ensembles (greedy top-k over primary candidates)
    ranked = sorted([n for n in ft_names if n in PRIMARY_CANDIDATES], key=lambda n: -overall[n])
    combos: list[tuple[str, list[str]]] = [(f"ensemble_top{k}", ranked[:k]) for k in range(2, len(ranked) + 1)]
    if ranked and "baseline_embedding" in probs_by_name:
        combos.append(("ensemble_top1_emb", [ranked[0], "baseline_embedding"]))
    ens_rows = []
    for name, members in combos:
        p = np.mean([cal[m] for m in members], axis=0)
        probs_by_name[name] = p
        cal[name] = p
        pred = p.argmax(1)
        sc = official_scores(gold, [ID2LABEL[int(i)] for i in pred])
        pf = per_fold_overall(pred, gold, folds)
        member_exps = [exp_by_name[m] for m in members if m in exp_by_name]
        fold_recs = []
        for f in folds:
            sub = official_scores([gold[i] for i in f.val_idx], [ID2LABEL[int(pred[i])] for i in f.val_idx])
            fold_recs.append({"fold": f.index, "best_epoch": None, "accuracy": sub["accuracy"], "macro_f1": sub["macro_f1"], "OVERALL": sub["OVERALL"],
                              "per_class_f1": {c: sub["per_class"][c]["f1"] for c in LABELS}, "history": [],
                              "train_seconds": float(sum(next((fr["train_seconds"] for fr in e["folds"] if fr["fold"] == f.index), 0.0) for e in member_exps))})
        row = {"model": name, "family": "ensemble (OOF, prob-avg)", "kind": "ensemble", "members": members, "params_M": int(sum(PARAM_COUNTS_M.get(exp_by_name[m]["config"]["model_name"], 0) for m in members if m in exp_by_name)),
               "accuracy": sc["accuracy"], "macro_f1": sc["macro_f1"], "OVERALL": sc["OVERALL"], "fold_std": float(np.std(pf)), "per_fold_OVERALL": pf,
               "per_class_f1": {c: sc["per_class"][c]["f1"] for c in LABELS}, "triad_errors": triad_confusion(sc)["total_triad_errors"]}
        ens_rows.append(row)
        np.save(res_dir / f"oof_logits_{name}.npy", np.log(np.clip(p, 1e-12, 1.0)))
        save_json({
            "name": name, "kind": "ensemble", "members": members, "protocol": protocol, "folds_run": [f.index for f in folds], "rows_evaluated": len(df),
            "config": {**(member_exps[0]["config"] if member_exps else {}), "model_name": "ensemble: " + " + ".join(members)},
            "member_configs": {m: exp_by_name[m]["config"] for m in members if m in exp_by_name},
            "accuracy_oof": sc["accuracy"], "macro_f1_oof": sc["macro_f1"], "OVERALL_oof": sc["OVERALL"], "OVERALL_fold_mean": float(np.mean(pf)), "OVERALL_fold_std": float(np.std(pf)),
            "per_class": sc["per_class"], "confusion": sc["confusion"], "confused_pairs": sc["confused_pairs"][:12], "triad_errors": triad_confusion(sc),
            "best_epochs": {m: exp_by_name[m]["best_epochs"] for m in members if m in exp_by_name},
            "train_seconds_total": float(sum(e["train_seconds_total"] for e in member_exps)), "folds": fold_recs,
        }, res_dir / f"transformer_{name}.json")
        log.info("%-20s members=%s OVERALL=%.4f macroF1=%.4f acc=%.4f per-fold=%s", name, members, sc["OVERALL"], sc["macro_f1"], sc["accuracy"], [round(x, 3) for x in pf])

    # ---------------------------------------------------------------- per-rule ablation on fine-tuned candidates and ensembles
    rule_abl = {}
    for n in ft_names + [r["model"] for r in ens_rows]:
        p = cal[n]
        base_pred = p.argmax(1)
        base_sc = official_scores(gold, [ID2LABEL[int(i)] for i in base_pred])
        entry = {"no_rules_OVERALL": base_sc["OVERALL"], "rules": {}}
        for rule in RULES:
            adj, fired = apply_rules(texts, p, max_conf=0.6, rules=[rule], use_bare_reference=False)
            pr = adj.argmax(1); sc = official_scores(gold, [ID2LABEL[int(i)] for i in pr])
            entry["rules"][rule.name] = {"fired": int(sum(bool(f) for f in fired)), "fixed": int(((pr == y) & (base_pred != y)).sum()), "broke": int(((pr != y) & (base_pred == y)).sum()), "delta_OVERALL": sc["OVERALL"] - base_sc["OVERALL"], "delta_macro_f1": sc["macro_f1"] - base_sc["macro_f1"]}
        adj, fired = apply_rules(texts, p, max_conf=0.6, rules=[], use_bare_reference=True)
        pr = adj.argmax(1); sc = official_scores(gold, [ID2LABEL[int(i)] for i in pr])
        entry["rules"]["bare_reference"] = {"fired": int(sum(bool(f) for f in fired)), "fixed": int(((pr == y) & (base_pred != y)).sum()), "broke": int(((pr != y) & (base_pred == y)).sum()), "delta_OVERALL": sc["OVERALL"] - base_sc["OVERALL"], "delta_macro_f1": sc["macro_f1"] - base_sc["macro_f1"]}
        adj, _ = apply_rules(texts, p, max_conf=0.6)
        sc = official_scores(gold, [ID2LABEL[int(i)] for i in adj.argmax(1)])
        entry["all_rules_OVERALL"] = sc["OVERALL"]
        rule_abl[n] = entry

    # ---------------------------------------------------------------- paired bootstrap
    preds = {n: [ID2LABEL[int(i)] for i in p.argmax(1)] for n, p in probs_by_name.items()}
    for n in ft_names + [r["model"] for r in ens_rows]:
        adj, _ = apply_rules(texts, cal[n], max_conf=0.6)
        preds[n + "+rules"] = [ID2LABEL[int(i)] for i in adj.argmax(1)]
    ci, samples = bootstrap_ci(gold, preds, n_boot=a.n_boot)
    primary_ft = [n for n in ft_names if n in PRIMARY_CANDIDATES] or ft_names
    ref = max(primary_ft, key=lambda n: ci[n]["mean"]) if primary_ft else None  # reference = best *primary* single model, never an ablation run
    if ref:
        for n in samples:
            ci[n]["p_better_than_" + ref] = float((np.array(samples[n]) > np.array(samples[ref])).mean())

    rows.sort(key=lambda r: -r["OVERALL"])
    out = {"protocol": protocol, "n_rows": len(df), "primary_candidates": PRIMARY_CANDIDATES, "candidates": rows, "ensembles": ens_rows, "rule_ablation": rule_abl, "bootstrap": ci, "reference_for_bootstrap": ref}
    save_json(out, res_dir / "comparison.json")

    # ---------------------------------------------------------------- markdown
    L = ["# Model comparison (pooled out-of-fold, grouped 5-fold CV, seed 42)\n",
         "| model | family | params (M) | CPU ms/question (batched / single) | accuracy | macro-F1 | OVERALL | fold std | 95% CI (bootstrap) | triad errors | rare-class F1 (disr/chit/other) |", "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in rows:
        c = ci.get(r["model"], {})
        ci_s = f"{c['ci95'][0]:.3f}–{c['ci95'][1]:.3f}" if c else ""
        rare = "/".join(f"{r['per_class_f1'][k]:.2f}" for k in ("disruptions", "chitchat", "other"))
        lat = r.get("latency")
        lat_s = f"{lat['cpu_ms_per_row_batched']:.1f} / {lat['cpu_ms_per_question_single']:.1f}" if lat else (f"{r['predict_ms_per_row']:.2f} (incl. encoding)" if r.get("predict_ms_per_row") is not None else "")
        tag = "" if r.get("primary", True) or r["kind"] != "single" else " (robustness/ablation run)"
        L.append(f"| {r['model']}{tag} | {r['family']} | {r.get('params_M') or ''} | {lat_s} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | **{r['OVERALL']:.3f}** | {r.get('fold_std', 0):.3f} | {ci_s} | {r['triad_errors']} | {rare} |")
    if ens_rows:
        L.append("\n## Ensembles (probability averaging of cross-fitted temperature-scaled members, OOF)\n")
        L.append("| ensemble | members | params (M) | accuracy | macro-F1 | OVERALL | fold std | 95% CI | per-fold OVERALL | triad errors |"); L.append("|---|---|---:|---:|---:|---:|---:|---|---|---:|")
        for r in ens_rows:
            c = ci.get(r["model"], {})
            ci_s = f"{c['ci95'][0]:.3f}–{c['ci95'][1]:.3f}" if c else ""
            L.append(f"| {r['model']} | {' + '.join(r['members'])} | {r['params_M']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | **{r['OVERALL']:.3f}** | {r['fold_std']:.3f} | {ci_s} | {', '.join(f'{x:.3f}' for x in r['per_fold_OVERALL'])} | {r['triad_errors']} |")
    L.append("\n## Rule layer ablation (rules applied only when calibrated confidence < 0.6)\n")
    for n, e in rule_abl.items():
        L.append(f"\n**{n}** — no rules {e['no_rules_OVERALL']:.4f}, all rules {e['all_rules_OVERALL']:.4f}\n")
        L.append("| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |"); L.append("|---|---:|---:|---:|---:|---:|")
        for rn, d in e["rules"].items():
            L.append(f"| {rn} | {d['fired']} | {d['fixed']} | {d['broke']} | {d['delta_OVERALL']:+.4f} | {d['delta_macro_f1']:+.4f} |")
    if ref:
        L.append(f"\n## Paired bootstrap vs `{ref}` ({a.n_boot:,} resamples over rows)\n")
        L.append("| candidate | mean OVERALL | 95% CI | P(better than reference) |"); L.append("|---|---:|---|---:|")
        for n, c in sorted(ci.items(), key=lambda kv: -kv[1]["mean"]):
            L.append(f"| {n} | {c['mean']:.4f} | {c['ci95'][0]:.3f}–{c['ci95'][1]:.3f} | {c.get('p_better_than_' + ref, float('nan')):.2f} |")
    (ROOT / "reports" / "comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(L[: len(rows) + 3]))


if __name__ == "__main__":
    main()
