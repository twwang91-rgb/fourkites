# Error analysis — deberta_v3_small_invw (pooled out-of-fold, grouped CV)

accuracy 0.6985 · macro-F1 0.6278 · **OVERALL 0.6561** · errors 574 / 1904

## Per-class

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.357 | 0.480 | 0.409 | 256 |
| realtime_query | 0.815 | 0.708 | 0.758 | 782 |
| analytics | 0.609 | 0.661 | 0.634 | 257 |
| disruptions | 0.742 | 0.821 | 0.780 | 28 |
| chitchat | 0.786 | 0.647 | 0.710 | 17 |
| knowledge_base | 0.916 | 0.853 | 0.883 | 509 |
| other | 0.185 | 0.273 | 0.221 | 55 |

## Most confused pairs (gold → predicted)

- realtime_query → shipment_information: 146
- shipment_information → realtime_query: 82
- realtime_query → analytics: 52
- analytics → realtime_query: 32
- shipment_information → analytics: 31
- analytics → shipment_information: 30
- knowledge_base → other: 23
- knowledge_base → shipment_information: 22
- realtime_query → other: 21
- knowledge_base → analytics: 21

Operational-triad errors: {'realtime_query->shipment_information': 146, 'shipment_information->realtime_query': 82, 'realtime_query->analytics': 52, 'analytics->realtime_query': 32, 'shipment_information->analytics': 31, 'analytics->shipment_information': 30, 'total_triad_errors': 373}

## Calibration and review flag

- temperature T = 1.499; ECE 0.1030 → 0.0489; NLL 0.9862 → 0.8986; Brier 0.4540 → 0.4383
- mean calibrated confidence: correct 0.730 vs wrong 0.590; margin 0.565 vs 0.352; 24 errors with confidence ≥ 0.9
- needs_review = confidence < 0.515: flags 20.0% of rows, error recall 37.8%, precision 0.57, accuracy on unflagged 0.766
- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag 20.3%, error recall 38.0%, precision 0.56, accuracy on unflagged 0.765, OVERALL on unflagged 0.7088
- AURC 0.1641 (full-coverage risk 0.3015)

## Rule-layer ablation (calibrated probabilities)

| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0.6985 | 0.6278 | 0.6561 | 373 |
| 0.50 | 336 | 7 | 4 | 3 | 0.6991 | 0.6279 | 0.6563 | 375 |
| 0.60 | 643 | 12 | 8 | 4 | 0.7006 | 0.6283 | 0.6572 | 372 |
| 0.70 | 949 | 12 | 8 | 4 | 0.7006 | 0.6283 | 0.6572 | 372 |
| 0.80 | 1303 | 12 | 8 | 4 | 0.7006 | 0.6283 | 0.6572 | 372 |
| 1.01 | 1904 | 12 | 8 | 4 | 0.7006 | 0.6283 | 0.6572 | 372 |

## Failure categories

Short (≤2 words) inputs: n=204, accuracy 0.613. Rows in conflicting-label duplicate groups: n=81, accuracy 0.370 (9% of all errors). Rare-class recall: disruptions 0.82, chitchat 0.65, other 0.27.

### triad: semantic — 280 errors, mean confidence 0.57

- `where is sn 0034819281` — gold **shipment_information**, predicted realtime_query (0.93)
- `I need a data pull of all LTL shipments YTD` — gold **realtime_query**, predicted analytics (0.93)
- `are there any loads out of 1028 tracking late` — gold **shipment_information**, predicted realtime_query (0.92)
- `Trailer #23044 Delivery for Dauphin Manitoba` — gold **shipment_information**, predicted realtime_query (0.92)
- `388990085 what is the current locaiton of this truck?` — gold **shipment_information**, predicted realtime_query (0.91)
- `me mostre dos últimos 10 dias` — gold **realtime_query**, predicted analytics (0.90)
- `Loads IDs. i need on time pick up and on time delivery % for the shipments listed` — gold **realtime_query**, predicted analytics (0.90)
- `me mostre todos com duração de 10 dias` — gold **realtime_query**, predicted analytics (0.90)

### knowledge_base vs operational — 111 errors, mean confidence 0.61

- `how do i make it to be only 1 contract` — gold **other**, predicted knowledge_base (0.96)
- `Who do I contact about a driver issue?` — gold **other**, predicted knowledge_base (0.95)
- `how do I know if a truck is a team driver` — gold **other**, predicted knowledge_base (0.95)
- `where can I review these metrics?` — gold **analytics**, predicted knowledge_base (0.95)
- `How is Hill Brothers sending Four Kites location updates? They asked us and I don't have an answer.` — gold **analytics**, predicted knowledge_base (0.92)
- `where can I see carriers set up and their status` — gold **analytics**, predicted knowledge_base (0.91)
- `is there a report that can show me destination dwell time` — gold **knowledge_base**, predicted analytics (0.91)
- `on time performance report - I need to add new field` — gold **knowledge_base**, predicted analytics (0.91)

### short/underspecified — 79 errors, mean confidence 0.58

- `1ZX600550392692343` — gold **shipment_information**, predicted realtime_query (0.93)
- `CMV3913820` — gold **shipment_information**, predicted realtime_query (0.92)
- `4308462335` — gold **shipment_information**, predicted realtime_query (0.90)
- `SOU1001399023` — gold **shipment_information**, predicted realtime_query (0.89)
- `524104441364` — gold **shipment_information**, predicted realtime_query (0.86)
- `通関状況は?` — gold **shipment_information**, predicted analytics (0.84)
- `通関状況は?` — gold **realtime_query**, predicted analytics (0.84)
- `082826-413` — gold **realtime_query**, predicted shipment_information (0.83)

### other / out-of-scope boundary — 50 errors, mean confidence 0.61

- `purchasing document status - 4507125913` — gold **other**, predicted realtime_query (0.92)
- `purchasing document status-PO 4507153487` — gold **other**, predicted realtime_query (0.92)
- `Please put the results in presentation format` — gold **analytics**, predicted other (0.91)
- `CMA BEX. Is there any benchmark on transit time?` — gold **other**, predicted analytics (0.90)
- `can you give me an address?` — gold **realtime_query**, predicted other (0.88)
- `male a excle file` — gold **realtime_query**, predicted other (0.87)
- `Can you make reports?` — gold **analytics**, predicted other (0.82)
- `download what i just searched` — gold **shipment_information**, predicted other (0.81)

### triad: temporal/aggregate cue — 42 errors, mean confidence 0.62

- `How many shipmets total last week` — gold **realtime_query**, predicted analytics (0.90)
- `I need to know how many shipments were tracked in Fourkites for the past 12 months.` — gold **realtime_query**, predicted analytics (0.89)
- `can you tell me how many loads for the scac SLSV that has reason codes for lates inputted into the loads` — gold **realtime_query**, predicted analytics (0.89)
- `Give me the total shipment count for last week.` — gold **realtime_query**, predicted analytics (0.88)
- `Query loads since 9/1/2026 marked as delivered with actual time of arrival = auto arrival` — gold **analytics**, predicted realtime_query (0.88)
- `How many missed deliveries have we had for the locations Butner, Maulden, Elloree and Dunn in the past 30 days` — gold **realtime_query**, predicted analytics (0.87)
- `The average transit time from ontario, canada to Puerto Rico US` — gold **realtime_query**, predicted analytics (0.85)
- `how many loads do i have that have not been collected from Bayer Raceway this week` — gold **realtime_query**, predicted analytics (0.83)

### disruptions boundary — 9 errors, mean confidence 0.69

- `What is the wait time for carriers crossing the World Trade Bridge in Laredo today 7/29/2026?` — gold **realtime_query**, predicted disruptions (0.93)
- `tiempos y riesgo` — gold **analytics**, predicted disruptions (0.90)
- `tiempos y riesgo` — gold **realtime_query**, predicted disruptions (0.90)
- `hoeveel levenciers komen er morgen op PF 22` — gold **shipment_information**, predicted disruptions (0.79)
- `me refiero solo al BS20260723-1235. Tiene o pasara poe algun punto de riesgo?` — gold **disruptions**, predicted realtime_query (0.70)
- `please focus analysis only on ocean lanes.` — gold **shipment_information**, predicted disruptions (0.66)
- `Can i get the data regarding detention and demurrage for Puerto Rico?` — gold **analytics**, predicted disruptions (0.52)
- `the route of BS20260723-1235 have montain regions with curves?` — gold **disruptions**, predicted realtime_query (0.41)

### chitchat boundary — 3 errors, mean confidence 0.54

- `Hello You are my supply chain analyst who is dealing with very complex supply chain from north america to phil` — gold **chitchat**, predicted analytics (0.88)
- `okay what u find` — gold **realtime_query**, predicted chitchat (0.41)
- `this is awesome !` — gold **chitchat**, predicted shipment_information (0.31)
