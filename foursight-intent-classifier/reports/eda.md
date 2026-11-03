# EDA summary

Train rows: 1904  |  Test rows: 566

## Class distribution

| class | n | % |
|---|---:|---:|
| shipment_information | 256 | 13.5 |
| realtime_query | 782 | 41.1 |
| analytics | 257 | 13.5 |
| disruptions | 28 | 1.5 |
| chitchat | 17 | 0.9 |
| knowledge_base | 509 | 26.7 |
| other | 55 | 2.9 |

Imbalance ratio (max/min): 46x

## Lengths

- **train**: median 8 words, p95 29, max 1533; 15.2% rows ≤3 words; 0.8% rows >128 words; 176 multi-line (pasted lists/tables).
- **test**: median 8 words, p95 28, max 1247; 15.5% rows ≤3 words; 0.2% rows >128 words; 13 multi-line (pasted lists/tables).

## Duplicates and label noise

- 165 exact-duplicate groups covering 379 rows; **37 groups carry conflicting labels** (81 rows).
- Best possible training-set accuracy given those conflicts: 98.00% (38 unavoidable errors).
- Examples: `4385338688` → {'shipment_information': 1, 'realtime_query': 1}; `56910268` → {'shipment_information': 1, 'realtime_query': 1}; `are you able to get me a list of currently active inbound international shipments that are going thr` → {'realtime_query': 3, 'shipment_information': 1}; `both` → {'other': 1, 'realtime_query': 3}; `can you please show only the loads with inbound tags` → {'shipment_information': 1, 'realtime_query': 1}; `can you show me all of the shipments i have listed for jan 30 2026` → {'analytics': 1, 'shipment_information': 1}; `cma bex is there any benchmark on transit time` → {'other': 1, 'analytics': 1}; `como passar isso pra transportadora` → {'other': 1, 'knowledge_base': 1}
- Near duplicates (char-ngram cosine ≥ 0.9, not exact): 170 pairs, 16 with conflicting labels.

## Train/test overlap

- 77 test rows match a training question exactly; test has 453 unique questions and 174 rows inside internal duplicate groups (31%).

## Validation leakage check (5-fold, seed 42)

- Plain stratified k-fold: 319 validation rows have an exact-text twin in their training fold.
- Grouped stratified k-fold: 0 leaked rows (by construction).

## Lexical cues (share of class rows containing the cue)

| cue | shipment_i | realtime_q | analytics | disruption | chitchat | knowledge_ | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal_past | 3 | 3 | 21 | 4 | 0 | 1 | 2 |
| temporal_now | 10 | 16 | 7 | 18 | 0 | 3 | 0 |
| temporal_future | 5 | 4 | 0 | 4 | 0 | 3 | 0 |
| aggregate | 4 | 4 | 34 | 4 | 0 | 8 | 2 |
| eta_status | 27 | 40 | 28 | 4 | 0 | 16 | 16 |
| where | 2 | 5 | 5 | 0 | 0 | 9 | 4 |
| track | 6 | 6 | 7 | 0 | 0 | 13 | 0 |
| how_to | 4 | 7 | 9 | 7 | 0 | 46 | 7 |
| disruption | 1 | 1 | 2 | 64 | 0 | 0 | 0 |
| greeting | 0 | 1 | 1 | 4 | 65 | 0 | 0 |
| reference_id | 21 | 54 | 11 | 7 | 0 | 1 | 11 |
| multi_ids | 7 | 12 | 8 | 0 | 0 | 0 | 2 |

## Bare reference-number questions

- Train: {'realtime_query': 65, 'shipment_information': 16}; test: 25 rows.

## Languages (heuristic)

- Train: {'en': 1882, 'ja/zh': 10, 'es': 8, 'pt': 4}; test: {'en': 558, 'fr': 8}

## Top tokens per class (TF-IDF logistic regression weights)

- **shipment_information**: frozen, shipments, truck, need to, yes, any loads, be delivered, or, see, burgas, shipment, andalusia
- **realtime_query**: status, where is, shipment, pod, loads, delivered, did, about, today, deliver, status on, these loads
- **analytics**: order, time, orders, otd, month, which, performance, hlcubsc2606dgtv0, on time, report, and, of
- **disruptions**: disruptions, impacted, sea, atlantic, port, wildfire, impacted by, ontario, due, by, tropical, could be
- **chitchat**: hi, hello, you, whats, awesome, this is, that is, thanks, what can, you do, que, up
- **knowledge_base**: how, fourkites, does, can, how do, do, to, how to, filter, how can, the, where can
- **other**: sorting, cest, country, make, read, both, tracy, email, csv, map, activity, new activity

## Design implications

1. Macro-F1 dominates the score, and three classes have < 60 examples: use class weighting and judge models on grouped CV, not a single split.
2. 37 conflicting-label duplicate groups and many one-word context fragments (`yes`, `both`, `eta`) are irreducible single-turn noise; they motivate the multi-turn design and the needs_review flag.
3. Duplicates must be grouped during validation, otherwise 319 of 1904 validation rows (17%) have a verbatim twin in training and scores are inflated.
4. Reference numbers are strong realtime_query evidence (50% of that class) but 18% of shipment_information rows also contain them: the triad boundary is semantic, not lexical.
5. Temporal-past and aggregate cues are the main lexical signal for analytics; `today` leans realtime_query.
6. A long tail of pasted tables (up to 1,533 words) means truncation policy matters: keep the head and a short tail.
7. Non-English rows exist in both splits (French only in test): the encoder must not fall over on them, but there are too few to tune for.
