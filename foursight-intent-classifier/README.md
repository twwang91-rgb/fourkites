# FourSight intent classifier

Deterministic seven-way intent router for the FourKites **FourSight** assistant, built for
the FourSight Intent Classification take-home challenge.

* **Single-turn model**: fine-tuned compact encoder (selection recorded in
  `reports/results/final_selection.json`) with temperature-scaled probabilities, an
  auditable low-confidence rule layer (kept only if validation evidence supports it) and
  a calibrated `needs_review` flag.
* **Multi-turn layer**: a deterministic contextual resolver (`foursight_intent.conversation`)
  with compact session state, follow-up detectors and entity / time-range carry-over —
  see [docs/multi_turn_architecture.md](docs/multi_turn_architecture.md).
* **New intents**: taxonomy versioning, OOD monitoring and a discovery loop —
  see [docs/new_intents.md](docs/new_intents.md).
* **Deliverables**: `test_predictions.csv` (566 rows, `id,intent`), `report.pdf`,
  `models/final/` (weights + tokenizer + label map + model card), `reports/` (EDA,
  all experiment results, figures).

Everything below was run on the data in the candidate package; **no hidden test labels
exist locally and none were used**. All model-selection numbers are pooled out-of-fold
scores from a seeded, grouped 5-fold cross-validation on `data/train.csv`.

---

## 1. Results at a glance

All numbers are **pooled out-of-fold** scores from the grouped, seeded 5-fold CV on the
1,904 training rows (each row predicted by a fold model that never saw it or its
duplicates). The hidden test set was never used for any decision. Full tables:
`reports/comparison.md`; selection trace: `reports/results/final_selection.json`.

| candidate | accuracy | macro-F1 | **OVERALL** | notes |
|---|---:|---:|---:|---|
| **fine-tuned DeBERTa-v3-small + rule layer (shipped)** | 0.708 | 0.615 | **0.652** | 95% paired-bootstrap CI 0.626–0.676; 142M params, 10 ms/question batched on CPU |
| fine-tuned DeBERTa-v3-small (no rules) | 0.707 | 0.611 | 0.649 | best learned model on its own |
| fine-tuned RoBERTa-base + rule layer | 0.720 | 0.609 | 0.653 | ties within 0.005, but only 0.628 without rules (the greeting rule rescues its chitchat class) |
| fine-tuned bge-small / e5-small / DistilBERT | 0.727 / 0.717 / 0.694 | 0.581 / 0.569 / 0.570 | 0.640 / 0.628 / 0.619 | smaller encoders |
| frozen multilingual-e5-small embeddings + logistic regression | 0.671 | 0.601 | 0.629 | strongest baseline, no fine-tuning |
| TF-IDF word+char + logistic regression | 0.681 | 0.513 | 0.580 | 0.1 ms/question |
| 3-model probability-averaging ensemble (reported, not shipped) | 0.734 | 0.620 | 0.666 | fails the pre-stated ensemble gate: +0.013 OVERALL is below the +0.02 bar and it loses ≥0.03 F1 on a rare class |

Per-class (shipped configuration, out-of-fold):

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.323 | 0.246 | 0.279 | 256 |
| realtime_query | 0.717 | 0.829 | 0.769 | 782 |
| analytics | 0.684 | 0.521 | 0.592 | 257 |
| disruptions | 0.786 | 0.786 | 0.786 | 28 |
| chitchat | 0.778 | 0.824 | 0.800 | 17 |
| knowledge_base | 0.874 | 0.900 | 0.887 | 509 |
| other | 0.231 | 0.164 | 0.191 | 55 |

* **Calibration**: temperature T = 1.41 brings ECE from 0.103 to 0.034.
* **needs_review** (calibrated confidence < 0.540, chosen to flag ≤ 20% of traffic): flags 20% of
  validation questions and catches 40% of the errors (precision 0.58); accuracy on the
  auto-answered rest is 0.780. Cross-fitted (threshold chosen on 4 folds, applied to the 5th):
  flag 21%, error recall 42%, precision 0.59.
* **Where it fails**: shipment_information ↔ realtime_query is the dominant confusion and is partly
  irreducible (37 duplicate groups carry conflicting gold labels; bare reference numbers are labelled
  both ways). See `reports/error_analysis_deberta_v3_small.md`.
* **Robustness checks** (reported, never eligible for selection): DeBERTa-v3-small re-run with seed 7
  gives 0.652 ± 0.031 (seed 42: 0.649 ± 0.031), so the leader is stable across seeds.
  Full inverse-frequency class weights (`inv`) instead of `sqrt_inv` lift DistilBERT from
  0.619 to 0.646 and the shipped encoder from 0.649 to 0.656
  (macro-F1 0.611 → 0.628, accuracy 0.707 → 0.699, fold std 0.031 → 0.015).
  **The shipped configuration keeps `sqrt_inv`**: the selection policy was frozen before these
  ablations ran and declares them ineligible, and promoting the best post-hoc variant would be
  the validation-set over-fitting the policy exists to prevent. `inv` weighting is the first
  experiment of the next iteration (confirm with a second seed, then re-run the full pipeline).
* **Negative results kept on record**: per-class logit-bias tuning does not transfer across folds
  (`reports/results/logit_bias_study.json`); plain stratified CV inflates OVERALL by ~0.05 for TF-IDF
  (`reports/results/protocol_comparison.json`).

| protocol | what it measures |
|---|---|
| grouped 5-fold CV (default) | duplicates never straddle folds; honest estimate |
| plain stratified 5-fold | 17% of validation rows have a verbatim training twin; optimistic |

Official metric: `OVERALL = 0.6 * macro_F1 + 0.4 * accuracy` (`foursight_intent.metrics`
is cross-checked against `scripts/score.py` in `tests/test_labels.py`; `scripts/score_oof.py`
runs the official scorer itself on the out-of-fold validation files).

---

## 2. Setup

Hardware used: Apple M1 Max (10 cores, 32 GB, MPS). Training also works on CPU (slower,
~1 min/epoch for DistilBERT) or any CUDA GPU; inference needs only a CPU.

```bash
# Python 3.10+ (3.11 used). Create an isolated environment:
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"            # or: pip install -r requirements.txt && pip install -e .

# Put the confidential challenge files in place (they are git-ignored):
cp /path/to/candidate_package/data/{train,test}.csv data/
cp /path/to/candidate_package/{labels.json,sample_submission.csv,schema.json} data/
cp /path/to/candidate_package/score.py scripts/score.py    # official scorer (already included)
```

Pinned versions used for the reported results are in `requirements-lock.txt`
(generated with `pip freeze`): torch 2.15, transformers 5.22, scikit-learn 1.9,
sentence-transformers 6.2, numpy 2.4, pandas 3.1.

Model downloads: the first run fetches the base encoders from the Hugging Face Hub
(`distilbert/distilbert-base-uncased`, `microsoft/deberta-v3-small`,
`FacebookAI/roberta-base`, `intfloat/multilingual-e5-small`, `BAAI/bge-small-en-v1.5`,
`sentence-transformers/all-MiniLM-L6-v2`). No token is needed; set `HF_HOME` to relocate
the cache.

---

## 3. Reproduce everything

```bash
bash scripts/run_all.sh            # auto-selects cuda > mps > cpu; ~75 min on M1 Max (5 encoders x 5 folds + baselines)
ROBUSTNESS=1 bash scripts/run_all.sh   # additionally runs the seed-7 and inverse-weight ablations (+~40 min)
```

`run_all.sh` re-creates every file under `reports/` and `models/final/` from scratch; the
pre-stated selection policy in `scripts/select_final.py` is applied mechanically, so the
chosen configuration is whatever the regenerated numbers say (it was DeBERTa-v3-small +
rules in the recorded run).

or step by step:

```bash
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline     # online if you have credentials
python scripts/run_eda.py                                   # reports/eda.{json,md}, figures
python scripts/run_baselines.py                             # TF-IDF + embeddings grids (grouped CV)
python scripts/run_experiments.py --name distilbert --model distilbert/distilbert-base-uncased --lr 3e-5
python scripts/run_experiments.py --name deberta_v3_small --model microsoft/deberta-v3-small --lr 5e-5
#   ... (see run_all.sh for the full candidate list)
python scripts/analyze_results.py --name distilbert         # calibration, review flag, rules, errors
python scripts/compare_candidates.py                        # comparison table, ensembles, bootstrap
python scripts/study_logit_bias.py                          # negative-result study (kept for the record)
python scripts/select_final.py                              # applies the pre-stated selection policy
python scripts/train_final.py --device cpu --threads 8      # retrain the selection on ALL rows -> models/final/ (bit-reproducible on CPU)
python scripts/score_oof.py --name deberta_v3_small --use-rules   # official score.py on out-of-fold validation files
python scripts/write_model_card.py                          # models/final/README.md
python scripts/generate_submission.py                       # test_predictions.csv (+ validation)
python -m pytest -q                                         # 79 tests incl. determinism (model tests skip until models/final exists)
python scripts/build_report.py                              # report.pdf
```

Self-check with the official scorer on a held-out slice (the scorer needs gold labels, so
this is for validation files you create, e.g. from the OOF predictions):

```bash
python3 scripts/score.py --gold my_val.csv --pred my_val_predictions.csv
```

### Predict

```bash
python scripts/predict.py "where is my load 8006907462" "how many loads were late last month?"
python scripts/predict.py --file questions.txt --json
python scripts/predict.py --conversation          # interactive multi-turn demo
```

```python
from foursight_intent.inference import IntentClassifier
clf = IntentClassifier("models/final", device="cpu", num_threads=4)
p = clf.predict_one("what about last month?")
p.intent, p.confidence, p.margin, p.needs_review, p.probs
```

Multi-turn:

```python
from foursight_intent.conversation import ContextualIntentResolver, ConversationState
resolver = ContextualIntentResolver(lambda t: clf.probabilities([t])[0], review_threshold=clf.review_threshold)
state = ConversationState(session_id="abc")
resolver.resolve("How many shipments were delayed last week?", state).intent   # analytics
r = resolver.resolve("What about last month?", state)
r.intent, r.used_context, r.resolved_time_expression, r.reason
```

---

## 4. Repository layout

```
foursight-intent-classifier/
├── README.md                     this file
├── report.pdf                    two-page technical report (generated by scripts/build_report.py)
├── test_predictions.csv          official submission (566 rows, id,intent)
├── pyproject.toml / requirements.txt / requirements-lock.txt
├── configs/default.yaml          seeds, split protocol, model + training hyper-parameters
├── data/                         confidential inputs (git-ignored) + README
├── src/foursight_intent/
│   ├── labels.py                 closed label set, ids, taxonomy version
│   ├── text.py / tokenization.py cleaning, reference-id regex, head+tail truncation (tokenizer-agnostic)
│   ├── data.py                   loading, grouped / stratified seeded folds, leakage report, class weights
│   ├── metrics.py                exact re-implementation of score.py + diagnostics
│   ├── baselines.py              TF-IDF + linear models, frozen embeddings + logistic regression
│   ├── train.py                  seeded fp32 fine-tuning loop, early stopping on OVERALL, artefact saving
│   ├── inference.py              deterministic CPU inference (single model or ensemble), calibrated probs, needs_review
│   ├── calibration.py            temperature scaling, ECE, reliability, review-threshold selection, AURC, logit-bias study
│   ├── rules.py                  minimal auditable rule layer (acts only below a confidence threshold)
│   ├── evaluate.py               CV driver, error tables, failure categorisation
│   └── conversation.py           multi-turn state, follow-up detectors, contextual resolver, synthetic scenarios
├── scripts/
│   ├── run_eda.py, run_baselines.py, run_experiments.py      data analysis, baselines, encoder CV
│   ├── analyze_results.py, study_logit_bias.py              calibration / review flag / rules / errors; negative-result study
│   ├── compare_candidates.py, select_final.py               comparison table + ensembles + bootstrap; pre-stated selection policy
│   ├── train_final.py, write_model_card.py                  retrain on all rows (single or ensemble artefact); HF model card
│   ├── generate_submission.py, score_oof.py, predict.py     submission + validation; official scorer on OOF files; CLI / multi-turn demo
│   ├── make_figures.py, build_report.py                     training curves + comparison figure; report.pdf
│   ├── inspect_wandb_offline.py, publish_artifacts.py       read offline W&B runs; dry-run / authorised upload
│   ├── run_all.sh                                           end-to-end reproduction
│   └── score.py                                             the official scorer, unchanged
├── tests/                        labels/metric, data/splits, submission schema, rules, conversation, determinism
├── models/final/                 trained artefact + model card (weights git-ignored; see publishing)
├── reports/                      eda.md, comparison.md, results/*.json, figures/*.png, logs
└── docs/                         multi_turn_architecture.md, new_intents.md, experiment_log.md
```

---

## 5. Methodology notes

**Validation.** `StratifiedGroupKFold(5, seed=42)` with the group = normalised question
text. EDA found 165 exact-duplicate groups (37 with conflicting labels) and that plain
stratified folds leak 319 validation rows; grouped folds leak 0. Every reported number is
*pooled out-of-fold*, i.e. each row predicted by a model that never saw it. Fold-to-fold
standard deviations are reported because 17- and 28-example classes make single splits
noisy.

**Calibration and review flag.** Temperature scaling is fitted on OOF logits; the review
threshold is the largest calibrated-confidence cut that flags ≤ 20% of traffic. Both are
also *cross-fitted* (fitted on four folds, evaluated on the fifth) and the cross-fitted
numbers are the ones quoted as expected production behaviour. `needs_review` is metadata:
every input still receives exactly one of the seven labels.

**Rule layer.** Five regex rules (greeting/thanks → chitchat, disruption vocabulary,
aggregate-over-past-period → analytics, procedural "how do I" → knowledge_base, bare
reference id → realtime_query) that only fire when calibrated confidence < 0.6 and only
*add* a bounded boost. `scripts/compare_candidates.py` reports fired / fixed / broke per
rule; `scripts/select_final.py` keeps the layer only if OVERALL improves and no class loses
more than 0.03 F1.

**Selection policy** (`scripts/select_final.py`, written before the ensembles were scored):
rank single models by OOF OVERALL; keep the rule layer only if it improves OVERALL without
dropping any class F1 by more than 0.03; an ensemble ships only if it beats the best single
model by more than 0.02, wins at least 4 of 5 folds and causes no rare-class regression;
ties within 0.005 go to the stronger model *without* rules, then lower CPU latency. The
tie-break wording was tightened once (parameter count made DeBERTa-v3-small look larger
than RoBERTa-base although 98M of its 142M parameters are an embedding table and it is
faster on CPU); both outcomes are recorded in `final_selection.json` under `tie_break_audit`.

**Final artefact.** After selection is frozen, the chosen configuration is retrained on all
1,904 rows for the median CV best-epoch count (3 epochs; note the linear schedule then
decays over 3 epochs rather than the 8 planned during CV). Its validation metrics are, by
construction, the OOF numbers of the CV run it was selected from — there is no held-out
data left, and the report says so. Training on CPU with a fixed thread count makes the
final fit itself bit-reproducible on the same machine.

---

## 6. Determinism guarantee

*Contract*: for a fixed `models/final/` directory and a fixed software environment, the
same input string yields bit-identical calibrated probabilities (and therefore the same
label and flag) on every call, in every process, regardless of batch composition.

How: fp32, CPU, `model.eval()`, `torch.set_num_threads(n)`,
`torch.use_deterministic_algorithms(True)`, `TOKENIZERS_PARALLELISM=false`, inputs
tokenised and truncated by pure functions, fixed-size batches in input order, softmax with
a stored temperature, `argsort(kind="stable")` for ties, regex rules. Verified by
`tests/test_determinism.py`.

Limits: different hardware or BLAS builds may change logits at the ~1e-6 level, which can
flip a label only when the top two probabilities are within that tolerance. GPU inference
is supported but is **not** covered by the bit-exact guarantee (MPS/CUDA kernels such as
`embedding_dense_backward` are nondeterministic in training; inference kernels are
generally stable but we do not claim it). Training reproducibility: seeded end to end;
CPU training is bit-reproducible, MPS/CUDA training is reproducible up to kernel
nondeterminism (documented in the CV logs by per-fold scores).

---

## 7. Experiment tracking and publishing

* **W&B**: every CV fold and the final fit log config, per-epoch train/val loss,
  accuracy, macro-F1, OVERALL and per-class F1 (`WANDB_MODE=offline` by default; runs are
  in `wandb/`). With credentials: `wandb login && wandb sync wandb/offline-run-*`.
  No run URL is claimed until a sync has actually happened.
* **Hugging Face**: `models/final/` is a complete `transformers` model directory with a
  model card. `python scripts/publish_artifacts.py` prints the exact upload commands and
  only uploads with `--push` and an `HF_TOKEN`. The challenge data is confidential and is
  never part of the artefact.
* **GitHub**: the repository is self-contained; `.gitignore` excludes the raw data, the
  challenge PDF and model weights.

---

## 8. Confidentiality

The challenge package is marked confidential. What this repository does about it:

* `data/*.csv`, the challenge PDF, model weights, `reports/validation/`, the per-row
  `reports/results/oof_predictions_*.csv`, the raw `*.log` files and `models/_smoke/` are
  git-ignored. (The W&B offline runs under `wandb/` are also ignored; sync them rather than
  committing them.)
* The artefact in `models/final/` contains weights, tokenizer, label map and a model card;
  **no training rows**.
* A few report files quote anonymised training questions verbatim because an error
  analysis without examples is useless to a reviewer: `reports/eda.{json,md}` (duplicate /
  conflict examples), `reports/error_analysis_*.md` and `reports/results/analysis_*.json`
  (≤ 8 examples per failure category). `scripts/publish_artifacts.py` lists them. Share the
  repository privately with FourKites, or delete those files before making it public.
* No row of `test.csv` was labelled by hand, inspected for its gold label, or used for any
  modelling decision; the only test-set statistics computed are length, duplicate and
  train-overlap counts in the EDA.

## 9. Troubleshooting

| symptom | fix |
|---|---|
| `expected scalar type Half but found Float` on MPS | fixed in `train.py` (models are forced to fp32); update if you changed `load_tokenizer_and_model` |
| HF download 404 for `distilbert-base-uncased` | use the namespaced id `distilbert/distilbert-base-uncased` (done in configs/scripts) |
| `UserWarning: embedding_dense_backward_mps does not have a deterministic implementation` | expected on MPS during *training*; inference runs on CPU |
| very slow CPU training | pass `--device mps` / `--device cuda`, or reduce `max_epochs` in `configs/default.yaml` |
| `trained model not found at models/final` (tests skipped) | run `scripts/train_final.py` (or the full `run_all.sh`) first |
| `SubmissionError` from `generate_submission.py` | the test file changed; the script refuses to write a submission whose ids/order differ from `data/test.csv` |
| W&B asks to log in | `export WANDB_MODE=offline` (default) or `disabled` |

---

## 10. Limitations (honest summary)

* Absolute scores are bounded by label noise: 37 duplicate groups carry conflicting gold
  labels and ~15% of rows are context fragments that are unanswerable single-turn.
* "Count over a recent window" questions ("how many shipments were delayed last week?") are
  labelled four different ways in the training data (analytics ×4, shipment_information ×2,
  realtime_query ×1, knowledge_base ×1 among the eight such rows). The shipped model answers
  the challenge's own example with `realtime_query` at 0.73, above the review threshold, and
  the resolver then inherits that for "what about last month?". A labelling guideline for this
  pattern would help more than any modelling change (see `docs/experiment_log.md`).
* Rare-class F1 (chitchat n=17, disruptions n=28) has wide confidence intervals.
* The multi-turn resolver is validated only on synthetic scenarios; no multi-turn accuracy
  is claimed (see docs for the benchmark plan).
* Non-English inputs are handled but not tuned for.
* Links to GitHub / Hugging Face / W&B are marked **NOT PUBLISHED** until an authorised
  upload happens.
