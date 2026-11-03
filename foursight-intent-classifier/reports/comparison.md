# Model comparison (pooled out-of-fold, grouped 5-fold CV, seed 42)

| model | family | params (M) | CPU ms/question (batched / single) | accuracy | macro-F1 | OVERALL | fold std | 95% CI (bootstrap) | triad errors | rare-class F1 (disr/chit/other) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| deberta_v3_small_invw (robustness/ablation run) | fine-tuned encoder | 142 | 10.4 / 50.5 | 0.699 | 0.628 | **0.656** | 0.015 | 0.626–0.682 | 373 | 0.78/0.71/0.22 |
| deberta_v3_small_seed7 (robustness/ablation run) | fine-tuned encoder | 142 | 10.4 / 50.5 | 0.709 | 0.615 | **0.652** | 0.031 | 0.621–0.680 | 367 | 0.75/0.71/0.21 |
| deberta_v3_small | fine-tuned encoder | 142 | 10.4 / 50.5 | 0.707 | 0.611 | **0.649** | 0.031 | 0.622–0.674 | 380 | 0.75/0.80/0.19 |
| distilbert_invw (robustness/ablation run) | fine-tuned encoder | 67 | 5.3 / 33.5 | 0.692 | 0.615 | **0.646** | 0.028 | 0.617–0.672 | 377 | 0.71/0.73/0.24 |
| bge_small_ft | fine-tuned encoder | 33 | 4.1 / 16.9 | 0.727 | 0.581 | **0.640** | 0.026 | 0.605–0.664 | 353 | 0.61/0.67/0.09 |
| emb[multilingual-e5-small]+logreg(C=4.0,cw=balanced) | baseline: embedding | 118 | 36.46 (incl. encoding) | 0.671 | 0.601 | **0.629** | 0.034 |  | 361 | 0.71/0.68/0.20 |
| e5_small_ft | fine-tuned encoder | 118 | 5.0 / 19.2 | 0.717 | 0.569 | **0.628** | 0.029 | 0.603–0.652 | 367 | 0.76/0.75/0.00 |
| roberta_base | fine-tuned encoder | 125 | 11.1 / 56.6 | 0.717 | 0.568 | **0.628** | 0.034 | 0.592–0.657 | 361 | 0.78/0.33/0.16 |
| distilbert | fine-tuned encoder | 67 | 5.3 / 33.5 | 0.694 | 0.570 | **0.619** | 0.046 | 0.585–0.647 | 379 | 0.75/0.40/0.24 |
| emb[bge-small-en-v1.5]+logreg(C=8.0,cw=balanced) | baseline: embedding | 33 | 21.60 (incl. encoding) | 0.644 | 0.571 | **0.600** | 0.036 |  | 396 | 0.55/0.76/0.21 |
| tfidf[word+char]+logreg(C=8.0,cw=balanced) | baseline: tfidf |  | 0.09 (incl. encoding) | 0.681 | 0.513 | **0.580** | 0.045 |  | 392 | 0.65/0.29/0.22 |
| emb[all-MiniLM-L6-v2]+logreg(C=4.0,cw=balanced) | baseline: embedding | 22 | 15.51 (incl. encoding) | 0.629 | 0.534 | **0.572** | 0.016 |  | 395 | 0.56/0.62/0.16 |
| tfidf[word+char]+svm(C=0.5,cw=None) | baseline: tfidf |  | 0.11 (incl. encoding) | 0.700 | 0.452 | **0.551** | 0.028 |  | 347 | 0.67/0.11/0.04 |

## Ensembles (probability averaging of cross-fitted temperature-scaled members, OOF)

| ensemble | members | params (M) | accuracy | macro-F1 | OVERALL | fold std | 95% CI | per-fold OVERALL | triad errors |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| ensemble_top2 | deberta_v3_small + bge_small_ft | 175 | 0.726 | 0.606 | **0.654** | 0.023 | 0.628–0.676 | 0.682, 0.641, 0.650, 0.614, 0.663 | 363 |
| ensemble_top3 | deberta_v3_small + bge_small_ft + e5_small_ft | 293 | 0.734 | 0.620 | **0.666** | 0.012 | 0.641–0.688 | 0.682, 0.655, 0.648, 0.669, 0.658 | 352 |
| ensemble_top4 | deberta_v3_small + bge_small_ft + e5_small_ft + roberta_base | 418 | 0.739 | 0.616 | **0.665** | 0.011 | 0.638–0.689 | 0.671, 0.673, 0.657, 0.643, 0.657 | 349 |
| ensemble_top5 | deberta_v3_small + bge_small_ft + e5_small_ft + roberta_base + distilbert | 485 | 0.735 | 0.619 | **0.665** | 0.011 | 0.639–0.689 | 0.668, 0.675, 0.655, 0.642, 0.662 | 354 |
| ensemble_top1_emb | deberta_v3_small + baseline_embedding | 142 | 0.713 | 0.631 | **0.664** | 0.032 | 0.636–0.688 | 0.696, 0.673, 0.619, 0.697, 0.638 | 363 |

## Rule layer ablation (rules applied only when calibrated confidence < 0.6)


**bge_small_ft** — no rules 0.6395, all rules 0.6442

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 8 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 5 | 3 | 0 | +0.0068 | +0.0102 |
| past_aggregate | 2 | 1 | 1 | +0.0001 | +0.0002 |
| how_to_product | 6 | 1 | 0 | +0.0004 | +0.0003 |
| bare_reference | 18 | 0 | 5 | -0.0026 | -0.0026 |

**deberta_v3_small** — no rules 0.6491, all rules 0.6521

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 1 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 3 | 2 | 1 | +0.0028 | +0.0044 |
| past_aggregate | 1 | 0 | 0 | -0.0001 | -0.0001 |
| how_to_product | 7 | 1 | 0 | +0.0004 | +0.0004 |
| bare_reference | 12 | 1 | 1 | -0.0002 | -0.0004 |

**deberta_v3_small_invw** — no rules 0.6561, all rules 0.6562

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 0 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 1 | 0 | 0 | +0.0000 | +0.0000 |
| past_aggregate | 1 | 1 | 0 | +0.0005 | +0.0005 |
| how_to_product | 8 | 3 | 0 | +0.0012 | +0.0010 |
| bare_reference | 48 | 4 | 6 | -0.0015 | -0.0019 |

**deberta_v3_small_seed7** — no rules 0.6523, all rules 0.6538

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 0 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 4 | 1 | 0 | +0.0012 | +0.0016 |
| past_aggregate | 3 | 0 | 0 | -0.0001 | -0.0001 |
| how_to_product | 6 | 2 | 0 | +0.0008 | +0.0006 |
| bare_reference | 10 | 0 | 1 | -0.0005 | -0.0005 |

**distilbert** — no rules 0.6195, all rules 0.6439

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 7 | 6 | 0 | +0.0282 | +0.0448 |
| disruption_vocabulary | 3 | 0 | 1 | -0.0028 | -0.0043 |
| past_aggregate | 1 | 0 | 0 | +0.0000 | +0.0000 |
| how_to_product | 7 | 2 | 1 | +0.0004 | +0.0002 |
| bare_reference | 23 | 2 | 4 | -0.0013 | -0.0015 |

**distilbert_invw** — no rules 0.6458, all rules 0.6439

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 3 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 1 | 0 | 1 | -0.0014 | -0.0019 |
| past_aggregate | 1 | 0 | 0 | +0.0000 | +0.0000 |
| how_to_product | 6 | 2 | 0 | +0.0008 | +0.0006 |
| bare_reference | 24 | 2 | 4 | -0.0013 | -0.0015 |

**e5_small_ft** — no rules 0.6283, all rules 0.6288

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 8 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 3 | 1 | 0 | +0.0011 | +0.0015 |
| past_aggregate | 0 | 0 | 0 | +0.0000 | +0.0000 |
| how_to_product | 8 | 1 | 2 | -0.0006 | -0.0007 |
| bare_reference | 10 | 0 | 0 | +0.0000 | +0.0000 |

**roberta_base** — no rules 0.6277, all rules 0.6532

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 7 | 5 | 0 | +0.0261 | +0.0417 |
| disruption_vocabulary | 1 | 0 | 0 | +0.0000 | +0.0000 |
| past_aggregate | 0 | 0 | 0 | +0.0000 | +0.0000 |
| how_to_product | 3 | 1 | 0 | +0.0004 | +0.0003 |
| bare_reference | 18 | 3 | 4 | -0.0010 | -0.0013 |

**ensemble_top2** — no rules 0.6539, all rules 0.6527

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 2 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 2 | 1 | 1 | +0.0008 | +0.0013 |
| past_aggregate | 2 | 1 | 0 | +0.0005 | +0.0005 |
| how_to_product | 9 | 1 | 3 | -0.0013 | -0.0015 |
| bare_reference | 17 | 0 | 2 | -0.0011 | -0.0012 |

**ensemble_top3** — no rules 0.6655, all rules 0.6629

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 7 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 3 | 1 | 1 | +0.0007 | +0.0012 |
| past_aggregate | 1 | 0 | 0 | +0.0000 | +0.0000 |
| how_to_product | 10 | 2 | 3 | -0.0010 | -0.0014 |
| bare_reference | 15 | 0 | 4 | -0.0023 | -0.0024 |

**ensemble_top4** — no rules 0.6655, all rules 0.6651

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 8 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 4 | 2 | 2 | +0.0014 | +0.0023 |
| past_aggregate | 1 | 1 | 0 | +0.0005 | +0.0005 |
| how_to_product | 8 | 1 | 0 | +0.0005 | +0.0005 |
| bare_reference | 11 | 0 | 5 | -0.0027 | -0.0028 |

**ensemble_top5** — no rules 0.6654, all rules 0.6639

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 8 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 4 | 2 | 2 | +0.0013 | +0.0021 |
| past_aggregate | 1 | 1 | 0 | +0.0005 | +0.0005 |
| how_to_product | 7 | 0 | 2 | -0.0011 | -0.0011 |
| bare_reference | 14 | 0 | 4 | -0.0021 | -0.0022 |

**ensemble_top1_emb** — no rules 0.6638, all rules 0.6650

| rule | fired | fixed | broke | ΔOVERALL | Δmacro-F1 |
|---|---:|---:|---:|---:|---:|
| greeting_or_thanks | 0 | 0 | 0 | +0.0000 | +0.0000 |
| disruption_vocabulary | 2 | 1 | 0 | +0.0021 | +0.0032 |
| past_aggregate | 1 | 1 | 0 | +0.0005 | +0.0005 |
| how_to_product | 4 | 1 | 3 | -0.0010 | -0.0010 |
| bare_reference | 35 | 2 | 2 | -0.0004 | -0.0006 |

## Paired bootstrap vs `deberta_v3_small` (1,000 resamples over rows)

| candidate | mean OVERALL | 95% CI | P(better than reference) |
|---|---:|---|---:|
| ensemble_top3 | 0.6646 | 0.641–0.688 | 0.92 |
| ensemble_top5 | 0.6645 | 0.639–0.689 | 0.94 |
| ensemble_top4 | 0.6644 | 0.638–0.689 | 0.94 |
| ensemble_top4+rules | 0.6640 | 0.637–0.689 | 0.91 |
| ensemble_top1_emb+rules | 0.6634 | 0.639–0.689 | 0.91 |
| ensemble_top5+rules | 0.6631 | 0.638–0.688 | 0.91 |
| ensemble_top1_emb | 0.6623 | 0.636–0.688 | 0.91 |
| ensemble_top3+rules | 0.6620 | 0.638–0.685 | 0.88 |
| deberta_v3_small_invw+rules | 0.6545 | 0.626–0.683 | 0.69 |
| deberta_v3_small_invw | 0.6543 | 0.626–0.682 | 0.68 |
| ensemble_top2 | 0.6528 | 0.628–0.676 | 0.69 |
| deberta_v3_small_seed7+rules | 0.6527 | 0.623–0.682 | 0.63 |
| ensemble_top2+rules | 0.6516 | 0.625–0.675 | 0.63 |
| roberta_base+rules | 0.6513 | 0.620–0.681 | 0.58 |
| deberta_v3_small_seed7 | 0.6513 | 0.621–0.680 | 0.59 |
| deberta_v3_small+rules | 0.6510 | 0.626–0.676 | 0.78 |
| deberta_v3_small | 0.6482 | 0.622–0.674 | 0.00 |
| distilbert_invw | 0.6443 | 0.617–0.672 | 0.40 |
| distilbert_invw+rules | 0.6425 | 0.615–0.670 | 0.35 |
| bge_small_ft+rules | 0.6423 | 0.612–0.669 | 0.37 |
| distilbert+rules | 0.6421 | 0.612–0.670 | 0.34 |
| bge_small_ft | 0.6374 | 0.605–0.664 | 0.27 |
| e5_small_ft+rules | 0.6280 | 0.605–0.652 | 0.09 |
| baseline_embedding | 0.6279 | 0.600–0.654 | 0.10 |
| e5_small_ft | 0.6274 | 0.603–0.652 | 0.08 |
| roberta_base | 0.6256 | 0.592–0.657 | 0.09 |
| distilbert | 0.6177 | 0.585–0.647 | 0.03 |
| baseline_tfidf_logreg | 0.5781 | 0.545–0.613 | 0.00 |
| baseline_tfidf_svm | 0.5503 | 0.522–0.578 | 0.00 |
