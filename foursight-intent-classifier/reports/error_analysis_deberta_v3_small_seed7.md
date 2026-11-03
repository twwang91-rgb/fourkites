# Error analysis — deberta_v3_small_seed7 (pooled out-of-fold, grouped CV)

accuracy 0.7085 · macro-F1 0.6149 · **OVERALL 0.6523** · errors 555 / 1904

## Per-class

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.358 | 0.391 | 0.374 | 256 |
| realtime_query | 0.795 | 0.767 | 0.781 | 782 |
| analytics | 0.590 | 0.623 | 0.606 | 257 |
| disruptions | 0.826 | 0.679 | 0.745 | 28 |
| chitchat | 0.909 | 0.588 | 0.714 | 17 |
| knowledge_base | 0.875 | 0.882 | 0.879 | 509 |
| other | 0.212 | 0.200 | 0.206 | 55 |

## Most confused pairs (gold → predicted)

- realtime_query → shipment_information: 115
- shipment_information → realtime_query: 104
- realtime_query → analytics: 49
- analytics → realtime_query: 37
- analytics → shipment_information: 34
- shipment_information → analytics: 28
- knowledge_base → analytics: 24
- analytics → knowledge_base: 22
- other → knowledge_base: 18
- knowledge_base → other: 18

Operational-triad errors: {'realtime_query->shipment_information': 115, 'shipment_information->realtime_query': 104, 'realtime_query->analytics': 49, 'analytics->realtime_query': 37, 'analytics->shipment_information': 34, 'shipment_information->analytics': 28, 'total_triad_errors': 367}

## Calibration and review flag

- temperature T = 1.804; ECE 0.1642 → 0.0158; NLL 1.1019 → 0.8891; Brier 0.4685 → 0.4260
- mean calibrated confidence: correct 0.762 vs wrong 0.612; margin 0.637 vs 0.410; 15 errors with confidence ≥ 0.9
- needs_review = confidence < 0.545: flags 19.9% of rows, error recall 39.1%, precision 0.57, accuracy on unflagged 0.778
- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag 19.7%, error recall 38.9%, precision 0.57, accuracy on unflagged 0.778, OVERALL on unflagged 0.6898
- AURC 0.1531 (full-coverage risk 0.2915)

## Rule-layer ablation (calibrated probabilities)

| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0.7085 | 0.6149 | 0.6523 | 367 |
| 0.50 | 291 | 5 | 3 | 1 | 0.7096 | 0.6166 | 0.6538 | 367 |
| 0.60 | 485 | 6 | 3 | 1 | 0.7096 | 0.6166 | 0.6538 | 367 |
| 0.70 | 787 | 6 | 3 | 1 | 0.7096 | 0.6166 | 0.6538 | 367 |
| 0.80 | 1046 | 6 | 3 | 1 | 0.7096 | 0.6166 | 0.6538 | 367 |
| 1.01 | 1904 | 6 | 3 | 1 | 0.7096 | 0.6166 | 0.6538 | 367 |

## Failure categories

Short (≤2 words) inputs: n=204, accuracy 0.598. Rows in conflicting-label duplicate groups: n=81, accuracy 0.407 (9% of all errors). Rare-class recall: disruptions 0.68, chitchat 0.59, other 0.20.

### triad: semantic — 276 errors, mean confidence 0.60

- `0034897218 0034897237 0034898655 0034899895 0034896967 0034897226 0034897229 0034895453 0034896972 are these l` — gold **realtime_query**, predicted analytics (0.89)
- `0034906452 0034904122 0034903816 0034904154 0034906515 0034906442 0034905870 0034896967 0034904152 0034903821 ` — gold **analytics**, predicted realtime_query (0.89)
- `what is the current transit time on ocean for Ship from Thailand to Ship to United states` — gold **shipment_information**, predicted analytics (0.88)
- `Origin is Store #3115, 7100 Tecumseh Road East Windsor` — gold **shipment_information**, predicted realtime_query (0.88)
- `please provide estimated delivery date to TurkeyJPCEVTYO260150886 JPCEVTYO260155965 JPCEVTYO260178134 JPCEVTYO` — gold **shipment_information**, predicted realtime_query (0.88)
- `SIDNEY OH US DSO 1001 – 2400 Industrial Dr, Sidney, OH, US. show me the load schedule of this origin` — gold **shipment_information**, predicted realtime_query (0.88)
- `get me an update of these shipments removing the NA NA78512143 NA78512164 NA78554143 NA78540935 NA78540930 NA7` — gold **shipment_information**, predicted realtime_query (0.88)
- `schneider trailer TA214262` — gold **shipment_information**, predicted realtime_query (0.88)

### knowledge_base vs operational — 121 errors, mean confidence 0.65

- `how can I connect you to a powerBI report from where you can read shipment data` — gold **other**, predicted knowledge_base (0.93)
- `how to send the link` — gold **other**, predicted knowledge_base (0.93)
- `Can you create a filtered report that I can export?` — gold **analytics**, predicted knowledge_base (0.93)
- `what is dimerco SCAC code` — gold **other**, predicted knowledge_base (0.92)
- `Honestly, I'm not sure how we flag them. We do them as exceptions within SAP` — gold **realtime_query**, predicted knowledge_base (0.92)
- `what is the precise GPS ping` — gold **realtime_query**, predicted knowledge_base (0.92)
- `is there a way to see which set up we have with each carrier in fourkites, if eld or phone number` — gold **analytics**, predicted knowledge_base (0.91)
- `how do i make it to be only 1 contract` — gold **other**, predicted knowledge_base (0.91)

### short/underspecified — 82 errors, mean confidence 0.59

- `SOU1001410653` — gold **shipment_information**, predicted realtime_query (0.88)
- `PATX008410` — gold **shipment_information**, predicted realtime_query (0.88)
- `TRAILER #` — gold **shipment_information**, predicted realtime_query (0.87)
- `PO 823180` — gold **shipment_information**, predicted realtime_query (0.87)
- `CMV3913820` — gold **shipment_information**, predicted realtime_query (0.84)
- `TRHU5135923` — gold **shipment_information**, predicted realtime_query (0.84)
- `524104441364` — gold **shipment_information**, predicted realtime_query (0.82)
- `73694283` — gold **shipment_information**, predicted realtime_query (0.81)

### triad: temporal/aggregate cue — 36 errors, mean confidence 0.66

- `The average transit time from ontario, canada to Puerto Rico US` — gold **realtime_query**, predicted analytics (0.90)
- `show me all departures for carrier DLS in the past 7 days` — gold **shipment_information**, predicted analytics (0.89)
- `Give me the total shipment count for last week.` — gold **realtime_query**, predicted analytics (0.88)
- `can you tell me how many loads for the scac SLSV that has reason codes for lates inputted into the loads` — gold **realtime_query**, predicted analytics (0.88)
- `how many loads do we have since 9/1/2026 marked as delivered with actual time of arrival = auto arrival` — gold **realtime_query**, predicted analytics (0.86)
- `How many PO's were missed pickup yesterday inbounding Canada` — gold **analytics**, predicted realtime_query (0.86)
- `How many shipmets total last week` — gold **realtime_query**, predicted analytics (0.84)
- `Can you show all shipment in the last week` — gold **realtime_query**, predicted analytics (0.84)

### other / out-of-scope boundary — 30 errors, mean confidence 0.57

- `purchasing document status - 4507125913` — gold **other**, predicted realtime_query (0.87)
- `purchasing document status-PO 4507153487` — gold **other**, predicted realtime_query (0.86)
- `make a dashboard 07A` — gold **other**, predicted shipment_information (0.81)
- `male a excle file` — gold **realtime_query**, predicted other (0.77)
- `CMA BEX. Is there any benchmark on transit time?` — gold **other**, predicted analytics (0.75)
- `schedule an appt at the below location with this reference number` — gold **other**, predicted shipment_information (0.75)
- `purchasing document status- 4507153487` — gold **realtime_query**, predicted other (0.73)
- `What are questions that I can ask you?` — gold **chitchat**, predicted other (0.70)

### disruptions boundary — 9 errors, mean confidence 0.68

- `any issues happened in the past one week for prince rupert port? Thanks` — gold **disruptions**, predicted analytics (0.89)
- `tiempos y riesgo` — gold **analytics**, predicted disruptions (0.85)
- `tiempos y riesgo` — gold **realtime_query**, predicted disruptions (0.85)
- `What is the wait time for carriers crossing the World Trade Bridge in Laredo today 7/29/2026?` — gold **realtime_query**, predicted disruptions (0.85)
- `me refiero solo al BS20260723-1235. Tiene o pasara poe algun punto de riesgo?` — gold **disruptions**, predicted realtime_query (0.85)
- `provide summary of plant wise, shipments impacted by dolpin` — gold **disruptions**, predicted analytics (0.60)
- `the route of BS20260723-1235 have montain regions with curves?` — gold **disruptions**, predicted realtime_query (0.55)
- `Can you show me all of our containers that will experience delays due to having to go around the Panama Canal` — gold **disruptions**, predicted realtime_query (0.42)

### chitchat boundary — 1 errors, mean confidence 0.65

- `Hello You are my supply chain analyst who is dealing with very complex supply chain from north america to phil` — gold **chitchat**, predicted analytics (0.65)
