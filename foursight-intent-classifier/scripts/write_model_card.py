#!/usr/bin/env python
"""Generate models/final/README.md (Hugging Face model card) from the recorded results.
Handles both a single-model artefact and an ensemble (ensemble.json present)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foursight_intent.labels import LABELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models/final"))
    a = ap.parse_args()
    md = Path(a.model_dir)
    inf = json.load(open(md / "inference_config.json"))
    summ = json.load(open(md / "training_summary.json"))
    name = inf["selected_experiment"]
    exp = json.load(open(ROOT / "reports/results" / f"transformer_{name}.json"))
    ana = json.load(open(ROOT / "reports/results" / f"analysis_{name}.json"))
    kind = inf.get("kind", "single")
    per = exp["per_class"]
    rows = "\n".join(f"| {c} | {per[c]['precision']:.3f} | {per[c]['recall']:.3f} | {per[c]['f1']:.3f} | {per[c]['support']} |" for c in LABELS)
    pol = ana["review"]["deployment_policy_with_rules"] if (inf.get("use_rules") and "deployment_policy_with_rules" in ana["review"]) else ana["review"]["deployment_policy"]
    cfg = summ["config"]
    if kind == "single":
        title = f"FourSight intent classifier ({inf['base_model'].split('/')[-1]})"
        base_model_meta = inf["base_model"]
        arch = f"fine-tuning of `{inf['base_model']}` with a linear classification head, fp32, trained on **all 1,904 rows** for {inf['epochs']} epochs"
        members_section = ""
        epochs_text = f"{inf['epochs']} epochs; epoch count = median of the CV best epochs ({exp['best_epochs']})"
        temp_text = f"temperature scaling T = {inf['temperature']:.3f} fitted on out-of-fold logits"
        load_snippet = 'clf = IntentClassifier("models/final", device="cpu", num_threads=4)'
    else:
        mem = inf["members"]
        title = f"FourSight intent classifier (ensemble of {len(mem)} fine-tuned encoders)"
        base_model_meta = "\n".join(f"  - {m['base_model']}" for m in mem)
        arch = "probability-averaging ensemble of independently fine-tuned encoders (each with a linear head, fp32), every member trained on **all 1,904 rows**"
        members_section = "\n## Members\n\n| member | base model | epochs (full fit) | temperature | OOF OVERALL alone |\n|---|---|---:|---:|---:|\n" + "\n".join(
            f"| {m['name']} | `{m['base_model']}` | {m['epochs']} | {m['temperature']:.3f} | {summ['members'][m['name']]['cv_reference']['OVERALL_oof']:.3f} |" for m in mem) + "\n"
        epochs_text = "per-member epoch counts = median of each member's CV best epochs: " + ", ".join(f"{m['name']} {m['epochs']}" for m in mem)
        temp_text = f"per-member temperature scaling fitted on each member's out-of-fold logits; ensemble temperature {inf['temperature']:.3f}"
        load_snippet = 'clf = load_classifier("models/final", device="cpu", num_threads=4)  # returns an EnsembleClassifier'
    card = f"""---
language: en
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model:
{base_model_meta if kind == 'ensemble' else '  - ' + base_model_meta}
tags: [intent-classification, logistics, supply-chain, fourkites, deterministic]
---

# {title}

Seven-way **first-hop intent router** for the FourKites FourSight assistant. Given a single
user question it returns exactly one of:
`{"`, `".join(LABELS)}`.

## Training data

1,904 anonymised, single-turn user questions supplied with the FourKites intent-classification
take-home challenge (confidential; **not** redistributed with this model). Class counts:
realtime_query 782, knowledge_base 509, analytics 257, shipment_information 256, other 55,
disruptions 28, chitchat 17. All metrics below come from grouped 5-fold cross-validation run
*before* the final fit on all rows (pooled out-of-fold predictions; duplicate questions never
straddle folds).

## Architecture and training procedure

* {arch}
* AdamW lr {cfg['learning_rate']}, weight decay {cfg['weight_decay']}, linear warm-up {cfg['warmup_ratio']}, grad-clip {cfg['grad_clip']}; batch size {cfg['batch_size']}
* max length {inf['max_length']} tokens with head+tail truncation ({inf['tail_tokens']} tail tokens) so pasted tables keep the trailing question
* class-weighted cross-entropy (`{cfg['class_weight']}`), early stopping on the official OVERALL metric during CV
* seed {inf['seed']}; {epochs_text}
* {temp_text}
* {'rule layer: five auditable regex rules act only when calibrated confidence < ' + str(inf['rules_max_conf']) if inf.get('use_rules') else 'no rule layer'}
{members_section}
## Validation metrics (out-of-fold, grouped 5-fold CV, n = 1,904)

| metric | value |
|---|---|
| accuracy | {exp['accuracy_oof']:.3f} |
| macro-F1 | {exp['macro_f1_oof']:.3f} |
| OVERALL = 0.6·macro-F1 + 0.4·accuracy | **{exp['OVERALL_oof']:.3f}** |
| fold-to-fold std of OVERALL | {exp['OVERALL_fold_std']:.3f} |
| ECE after temperature scaling | {ana['calibration']['fitted_on_all_oof']['calibrated']['ece']:.3f} |

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
{rows}

`needs_review` flag: calibrated confidence < {pol['threshold']:.3f} flags {100 * pol['flag_rate']:.0f}% of validation
questions, catching {100 * pol['error_recall']:.0f}% of the errors (precision {pol['precision']:.2f}); accuracy on the
remaining {100 * (1 - pol['flag_rate']):.0f}% is {pol['accuracy_retained']:.3f}.

## Intended use

First-hop routing of *single* FourSight questions to a product backend, with the
`needs_review` flag used to route low-confidence questions to a clarification prompt or a
human. For multi-turn chat it must be wrapped in the contextual resolver shipped in the
source repository (`foursight_intent.conversation`), which decides when to inherit the
previous turn's intent, shipment ids and time range.

## Limitations and known failure modes

* **shipment_information vs realtime_query** is the dominant confusion (the training
  labels themselves disagree on 37 duplicated questions); bare reference numbers and
  one-word fragments ("yes", "scheduled") are inherently ambiguous without context.
* Rare classes rest on very few examples (chitchat 17, disruptions 28, other 55); their
  F1 has wide confidence intervals.
* English-centric; the handful of Spanish / Portuguese / Japanese questions in training
  are not enough to claim multilingual support.
* Questions that paste large tables are truncated to {inf['max_length']} tokens (head + tail).
* Confidence is calibrated on validation data, not a guarantee of correctness; drift in
  production (new products, new phrasing) must be monitored.

## Deterministic inference

```python
from foursight_intent.inference import IntentClassifier, load_classifier
{load_snippet}
pred = clf.predict_one("where is my load 8006907462")
print(pred.intent, pred.confidence, pred.needs_review)
```

fp32, CPU, eval mode, fixed thread count, `torch.use_deterministic_algorithms(True)`,
argmax with lowest-index tie-break. With the same artefact and the same package versions
the probabilities are bit-identical across calls and processes (verified by
`tests/test_determinism.py`). Across different hardware/BLAS builds, logits may differ at
the 1e-6 level; labels change only for inputs whose top-2 probabilities are within that
tolerance.

## Taxonomy version

`{inf['taxonomy_version']}` — the seven labels above, in this order. A new intent requires a new
taxonomy version, retraining and a router update (see `docs/new_intents.md`).
"""
    (md / "README.md").write_text(card, encoding="utf-8")
    print("model card written to", md / "README.md")


if __name__ == "__main__":
    main()
