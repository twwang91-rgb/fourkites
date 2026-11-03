---
language: en
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model:
  - microsoft/deberta-v3-small
tags: [intent-classification, logistics, supply-chain, fourkites, deterministic]
---

# FourSight intent classifier (deberta-v3-small)

Seven-way **first-hop intent router** for the FourKites FourSight assistant. Given a single
user question it returns exactly one of:
`shipment_information`, `realtime_query`, `analytics`, `disruptions`, `chitchat`, `knowledge_base`, `other`.

## Training data

1,904 anonymised, single-turn user questions supplied with the FourKites intent-classification
take-home challenge (confidential; **not** redistributed with this model). Class counts:
realtime_query 782, knowledge_base 509, analytics 257, shipment_information 256, other 55,
disruptions 28, chitchat 17. All metrics below come from grouped 5-fold cross-validation run
*before* the final fit on all rows (pooled out-of-fold predictions; duplicate questions never
straddle folds).

## Architecture and training procedure

* fine-tuning of `microsoft/deberta-v3-small` with a linear classification head, fp32, trained on **all 1,904 rows** for 3 epochs
* AdamW lr 5e-05, weight decay 0.01, linear warm-up 0.1, grad-clip 1.0; batch size 16
* max length 128 tokens with head+tail truncation (16 tail tokens) so pasted tables keep the trailing question
* class-weighted cross-entropy (`sqrt_inv`), early stopping on the official OVERALL metric during CV
* seed 42; 3 epochs; epoch count = median of the CV best epochs ([6, 3, 4, 3, 3])
* temperature scaling T = 1.406 fitted on out-of-fold logits
* rule layer: five auditable regex rules act only when calibrated confidence < 0.6

## Validation metrics (out-of-fold, grouped 5-fold CV, n = 1,904)

| metric | value |
|---|---|
| accuracy | 0.707 |
| macro-F1 | 0.611 |
| OVERALL = 0.6·macro-F1 + 0.4·accuracy | **0.649** |
| fold-to-fold std of OVERALL | 0.031 |
| ECE after temperature scaling | 0.034 |

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.322 | 0.250 | 0.281 | 256 |
| realtime_query | 0.717 | 0.827 | 0.768 | 782 |
| analytics | 0.689 | 0.525 | 0.596 | 257 |
| disruptions | 0.800 | 0.714 | 0.755 | 28 |
| chitchat | 0.778 | 0.824 | 0.800 | 17 |
| knowledge_base | 0.875 | 0.898 | 0.887 | 509 |
| other | 0.220 | 0.164 | 0.187 | 55 |

`needs_review` flag: calibrated confidence < 0.540 flags 20% of validation
questions, catching 40% of the errors (precision 0.58); accuracy on the
remaining 80% is 0.780.

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
* Questions that paste large tables are truncated to 128 tokens (head + tail).
* Confidence is calibrated on validation data, not a guarantee of correctness; drift in
  production (new products, new phrasing) must be monitored.

## Deterministic inference

```python
from foursight_intent.inference import IntentClassifier, load_classifier
clf = IntentClassifier("models/final", device="cpu", num_threads=4)
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

`v1` — the seven labels above, in this order. A new intent requires a new
taxonomy version, retraining and a router update (see `docs/new_intents.md`).
