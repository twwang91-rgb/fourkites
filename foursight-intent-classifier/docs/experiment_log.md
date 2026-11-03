# Experiment log

Chronological record of what was tried, what was measured and what was decided. Every
number is pooled out-of-fold OVERALL under the grouped, seeded 5-fold CV unless stated
otherwise; the JSON source is named so each line can be re-derived.

| # | step | result | decision | source |
|---|---|---|---|---|
| 1 | EDA | 165 duplicate groups, 37 with conflicting labels (accuracy ceiling 98.0% on train); plain stratified folds leak 319 rows; 15% of rows ≤3 words; reference ids in 54% realtime vs 21% shipment_information | grouped CV for every decision; class weights; head+tail truncation; review flag + multi-turn design motivated | `reports/eda.json` |
| 2 | TF-IDF grids (word / char / word+char × logreg / SVM × C × class weight) | best 0.580 (word+char, logreg, C=8, balanced); SVM without weights gets 0.700 accuracy but 0.452 macro-F1 | class weighting is mandatory for the metric; TF-IDF kept as the latency reference (0.1 ms) | `reports/results/baselines_grouped.json` |
| 3 | frozen embeddings + logreg (MiniLM, bge-small, multilingual-e5-small) | best 0.629 (e5-small, C=4, balanced); MiniLM 0.572 | strong, cheap baseline; e5-small also nominated for fine-tuning | same |
| 4 | stratified vs grouped protocol, same grids | stratified inflates OVERALL by +0.047 on average (+0.055 for TF-IDF) | confirms protocol choice; reported in README | `reports/results/protocol_comparison.json` |
| 5 | DistilBERT fine-tune (lr 3e-5, sqrt-inv weights, 5 folds) | 0.619 ± 0.046; chitchat F1 0.40 | first encoder; high fold variance on 17-example class | `reports/results/transformer_distilbert.json` |
| 6 | DeBERTa-v3-small (lr 5e-5) | first attempt crashed on MPS (checkpoint loads in fp16 → "expected Half but found Float"); fixed by forcing fp32 at load. Result 0.649 ± 0.031, chitchat 0.80, knowledge_base 0.89 | leader | `reports/results/transformer_deberta_v3_small.json` |
| 7 | multilingual-e5-small fine-tune (lr 5e-5) | 0.628 ± 0.029 but `other` F1 = 0.00 and shipment_information 0.22 | collapses minority classes; not selected | `reports/results/transformer_e5_small_ft.json` |
| 8 | RoBERTa-base (lr 2e-5) | 0.628 ± 0.034; chitchat 0.33 → 0.653 with the greeting rule | strong only with rules; loses the tie-break to DeBERTa (0.649 without rules vs 0.628) | `reports/results/transformer_roberta_base.json` |
| 9 | bge-small fine-tune (lr 5e-5) | 0.640 ± 0.026, best accuracy (0.727) but `other` 0.09 | fastest encoder (4 ms); not selected | `reports/results/transformer_bge_small_ft.json` |
| 10 | temperature scaling + review threshold (cross-fitted) | DeBERTa: ECE 0.103 → 0.034; threshold 0.54 flags 20%, catches 40% of errors, precision 0.59, accuracy on the rest 0.78 | shipped in `inference_config.json` | `reports/results/analysis_deberta_v3_small.json` |
| 11 | rule layer (5 regex rules below confidence 0.6), per-rule ablation | DeBERTa +0.003 (disruption vocabulary +0.003, how-to +0.0004, bare-reference −0.0002); RoBERTa +0.025 (almost all from the greeting rule); DistilBERT +0.024 | kept under the pre-stated criterion (improves OVERALL, no class −0.03); acknowledged as marginal for the shipped model | `reports/results/comparison.json` → `rule_ablation` |
| 12 | per-class logit bias (cross-fitted coordinate ascent) | DeBERTa −0.008, DistilBERT −0.005, e5 +0.001, bge ≈0, RoBERTa +0.015 | negative result for the selected model; bias vector shipped as zeros | `reports/results/logit_bias_study.json` |
| 13 | ensembles (greedy top-k probability averaging, members temperature-scaled per fold) | top-3 0.666, top-4 0.665, top-5 0.665, DeBERTa+frozen-e5 0.664; +0.013 over the shipped model, 4/5 fold wins, but a rare class loses ≥0.03 F1 | fail the pre-stated gate (+0.02, 4/5 folds, no rare-class regression) → single model ships; ensemble artefact path implemented and tested anyway | `reports/results/final_selection.json` |
| 14 | paired bootstrap (1,000 resamples) | DeBERTa+rules 95% CI 0.626–0.676; top-3 ensemble 0.641–0.688; P(ensemble > DeBERTa) ≈ 0.92 | differences between the top single models are within noise; ensemble gain is real but small | `reports/comparison.md` |
| 15 | seed robustness: DeBERTa-v3-small with seed 7 | 0.652 ± 0.031 (seed 42: 0.649 ± 0.031); per-class F1 within ±0.03 | the leader is reproducible across seeds; the DeBERTa-vs-rest ranking does not depend on the seed | `reports/results/transformer_deberta_v3_small_seed7.json` |
| 15b | class-weight ablation: DistilBERT with full inverse-frequency weights (`inv`) instead of `sqrt_inv` | 0.646 vs 0.619 (macro-F1 0.615 vs 0.570, accuracy 0.692 vs 0.694) | a real effect on the smallest encoder: stronger weighting buys rare-class recall at no accuracy cost. Per the pre-stated policy ablation runs are not eligible for selection, so the shipped model keeps `sqrt_inv`; the same ablation was then run for the shipped encoder (row 15c) | `reports/results/transformer_distilbert_invw.json` |
| 15c | class-weight ablation on the shipped encoder: DeBERTa-v3-small with `inv` weights | 0.656 ± 0.015 vs 0.649 ± 0.031 (macro-F1 0.628 vs 0.611, accuracy 0.699 vs 0.707; 0.657 with rules; 95% CI 0.626–0.682; P(better than shipped, paired bootstrap) 0.68) | **better than the shipped configuration, but not shipped**: the selection policy was frozen before the ablations ran and says ablation runs are ineligible; swapping the deliverable for the best post-hoc variant is exactly the validation-set over-fitting the policy exists to prevent. It is the first experiment of the next iteration, to be confirmed with a second seed | `reports/results/transformer_deberta_v3_small_invw.json` |
| 16 | final fit on all 1,904 rows (3 epochs, CPU, 8 threads) + model card + submission + tests + report | `models/final/`, `test_predictions.csv` (566 rows, validated), `report.pdf` | done | `models/final/training_summary.json`, `reports/results/submission_summary.json` |

## Observation worth flagging to the panel

The challenge's own multi-turn example, *"How many shipments were delayed last week?"*, is
classified `realtime_query` (0.73) by the shipped single-turn model, so the resolver then
inherits `realtime_query` for *"What about last month?"*. The training data is split on this
exact pattern: among the eight "how many … last week / past month / yesterday" rows,
four are labelled `analytics`, two `shipment_information`, one `realtime_query` and one
`knowledge_base` (`reports/results/oof_predictions_deberta_v3_small.csv`). The rule layer
does not intervene because it only acts below 0.6 confidence, by design. This is the
clearest case where a labelling guideline for "count over a recent window" would be worth
more than any modelling change, and it is where a clarification prompt (the review flag at a
higher threshold) would pay off in production.

## What did not make it in (and why)

* **Data augmentation / back-translation** — the dominant errors are label-conflict and
  context-fragment rows; more paraphrases of clean questions would not address them.
* **Larger encoders (DeBERTa-v3-base, 184M)** — fold variance (±0.03) already exceeds the
  gaps between candidates; a bigger model on 1,904 rows was judged unlikely to change the
  ranking and would double CPU latency.
* **Threshold tuning per class beyond the logit-bias study** — the study showed in-sample
  gains that do not transfer across folds; anything finer would overfit the validation set.
* **Multi-turn fine-tuning** — no conversational labels exist; see
  `docs/multi_turn_architecture.md` for the planned data collection and the deterministic
  resolver that stands in until then.
