# Error analysis — deberta_v3_small (pooled out-of-fold, grouped CV)

accuracy 0.7069 · macro-F1 0.6106 · **OVERALL 0.6491** · errors 558 / 1904

## Per-class

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.322 | 0.250 | 0.281 | 256 |
| realtime_query | 0.717 | 0.827 | 0.768 | 782 |
| analytics | 0.689 | 0.525 | 0.596 | 257 |
| disruptions | 0.800 | 0.714 | 0.755 | 28 |
| chitchat | 0.778 | 0.824 | 0.800 | 17 |
| knowledge_base | 0.875 | 0.898 | 0.887 | 509 |
| other | 0.220 | 0.164 | 0.187 | 55 |

## Most confused pairs (gold → predicted)

- shipment_information → realtime_query: 157
- realtime_query → shipment_information: 86
- analytics → realtime_query: 69
- realtime_query → analytics: 29
- analytics → shipment_information: 24
- analytics → knowledge_base: 23
- other → knowledge_base: 19
- shipment_information → analytics: 15
- knowledge_base → realtime_query: 15
- other → realtime_query: 13

Operational-triad errors: {'shipment_information->realtime_query': 157, 'realtime_query->shipment_information': 86, 'analytics->realtime_query': 69, 'realtime_query->analytics': 29, 'analytics->shipment_information': 24, 'shipment_information->analytics': 15, 'total_triad_errors': 380}

## Calibration and review flag

- temperature T = 1.406; ECE 0.1030 → 0.0343; NLL 0.9036 → 0.8426; Brier 0.4280 → 0.4121
- mean calibrated confidence: correct 0.766 vs wrong 0.595; margin 0.627 vs 0.365; 20 errors with confidence ≥ 0.9
- needs_review = confidence < 0.538: flags 20.0% of rows, error recall 40.0%, precision 0.59, accuracy on unflagged 0.780
- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag 21.3%, error recall 42.5%, precision 0.59, accuracy on unflagged 0.786, OVERALL on unflagged 0.6981
- AURC 0.1457 (full-coverage risk 0.2931)

## Rule-layer ablation (calibrated probabilities)

| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0.7069 | 0.6106 | 0.6491 | 380 |
| 0.50 | 294 | 5 | 3 | 0 | 0.7085 | 0.6175 | 0.6539 | 380 |
| 0.60 | 522 | 9 | 4 | 2 | 0.7080 | 0.6148 | 0.6521 | 380 |
| 0.70 | 743 | 9 | 4 | 2 | 0.7080 | 0.6148 | 0.6521 | 380 |
| 0.80 | 1132 | 9 | 4 | 2 | 0.7080 | 0.6148 | 0.6521 | 380 |
| 1.01 | 1904 | 9 | 4 | 2 | 0.7080 | 0.6148 | 0.6521 | 380 |

## Failure categories

Short (≤2 words) inputs: n=204, accuracy 0.588. Rows in conflicting-label duplicate groups: n=81, accuracy 0.309 (10% of all errors). Rare-class recall: disruptions 0.71, chitchat 0.82, other 0.16.

### triad: semantic — 283 errors, mean confidence 0.60

- `Origin Site Code Destination Site Code Material Material Description Material Quantity Sales Unit SU QTY 4PL N` — gold **realtime_query**, predicted analytics (0.94)
- `Are these shipments behind or on time for delivery today?` — gold **shipment_information**, predicted realtime_query (0.90)
- `Please provide a list of loads departed late from Fresno, CA` — gold **shipment_information**, predicted realtime_query (0.90)
- `8006895826 is there an appt yet` — gold **shipment_information**, predicted realtime_query (0.90)
- `Where is my truck tracking for the week on 9/28/2026` — gold **shipment_information**, predicted realtime_query (0.89)
- `what shipments is port hawkesbury nova scotia getting aug 8 2026` — gold **analytics**, predicted realtime_query (0.89)
- `what shipments is port hawkesbury nova scotia getting aug 8 2026` — gold **shipment_information**, predicted realtime_query (0.89)
- `Bealls #318 Swift` — gold **realtime_query**, predicted shipment_information (0.88)

### knowledge_base vs operational — 113 errors, mean confidence 0.60

- `is there a way to see which set up we have with each carrier in fourkites, if eld or phone number` — gold **analytics**, predicted knowledge_base (0.96)
- `como passar isso pra transportadora?` — gold **other**, predicted knowledge_base (0.96)
- `is Fourkites using Grapphopper? to get geolocations?` — gold **other**, predicted knowledge_base (0.96)
- `what about the late reason codes` — gold **realtime_query**, predicted knowledge_base (0.96)
- `how can I connect you to a powerBI report from where you can read shipment data` — gold **other**, predicted knowledge_base (0.96)
- `on time performance report - I need to add new field` — gold **knowledge_base**, predicted analytics (0.95)
- `what is the precise GPS ping` — gold **realtime_query**, predicted knowledge_base (0.95)
- `creating qr code for loading delivery` — gold **other**, predicted knowledge_base (0.94)

### short/underspecified — 84 errors, mean confidence 0.58

- `eta` — gold **shipment_information**, predicted realtime_query (0.89)
- `120118269` — gold **shipment_information**, predicted realtime_query (0.86)
- `8006899320` — gold **shipment_information**, predicted realtime_query (0.86)
- `0034835176` — gold **shipment_information**, predicted realtime_query (0.84)
- `56910268` — gold **shipment_information**, predicted realtime_query (0.84)
- `20260901` — gold **shipment_information**, predicted realtime_query (0.83)
- `YES TODAY` — gold **shipment_information**, predicted realtime_query (0.81)
- `YES TODAY` — gold **shipment_information**, predicted realtime_query (0.81)

### triad: temporal/aggregate cue — 38 errors, mean confidence 0.57

- `How many shipmets total last week` — gold **realtime_query**, predicted analytics (0.87)
- `how many items in shipment` — gold **shipment_information**, predicted realtime_query (0.84)
- `extract all loads for the past month destination city memphis` — gold **analytics**, predicted realtime_query (0.83)
- `What trucks to i have in transit that are between Laredo, TX and Pharr, TX within the last 2 hours?` — gold **analytics**, predicted realtime_query (0.83)
- `how many tl moves took place yesterday?` — gold **shipment_information**, predicted analytics (0.82)
- `How many missed deliveries have we had for the locations Butner, Maulden, Elloree and Dunn in the past 30 days` — gold **realtime_query**, predicted analytics (0.76)
- `I need to know how many shipments were tracked in Fourkites for the past 12 months.` — gold **realtime_query**, predicted analytics (0.72)
- `can you show me all deliveries to piggly wiggly since 01/01/2026?` — gold **shipment_information**, predicted realtime_query (0.71)

### other / out-of-scope boundary — 28 errors, mean confidence 0.57

- `CMA BEX. Is there any benchmark on transit time?` — gold **other**, predicted analytics (0.89)
- `Please put the results in presentation format` — gold **analytics**, predicted other (0.89)
- `Can you make reports?` — gold **analytics**, predicted other (0.86)
- `purchasing document status - 4507125913` — gold **other**, predicted realtime_query (0.80)
- `purchasing document status-PO 4507153487` — gold **other**, predicted realtime_query (0.79)
- `Delivery: America Chung Nam Inc – ND Paper 621 N Biron Dr, Wisconsin Rapids, WI 54494` — gold **other**, predicted realtime_query (0.77)
- `Pickup and delivery number to use for this will be N1091681` — gold **other**, predicted realtime_query (0.75)
- `list coorspoding sn for these pos 0108596326 0107628575 0108834446 0108842669 0108596790 0108841590 0107460504` — gold **other**, predicted shipment_information (0.75)

### disruptions boundary — 10 errors, mean confidence 0.68

- `provide summary of plant wise, shipments impacted by dolpin` — gold **disruptions**, predicted analytics (0.95)
- `tiempos y riesgo` — gold **analytics**, predicted disruptions (0.91)
- `tiempos y riesgo` — gold **realtime_query**, predicted disruptions (0.91)
- `me refiero solo al BS20260723-1235. Tiene o pasara poe algun punto de riesgo?` — gold **disruptions**, predicted realtime_query (0.87)
- `the route of BS20260723-1235 have montain regions with curves?` — gold **disruptions**, predicted realtime_query (0.75)
- `any issues happened in the past one week for prince rupert port? Thanks` — gold **disruptions**, predicted analytics (0.57)
- `show last one week ocean failures` — gold **realtime_query**, predicted disruptions (0.54)
- `What shipments are at risk` — gold **realtime_query**, predicted disruptions (0.52)

### chitchat boundary — 2 errors, mean confidence 0.58

- `ya do that` — gold **realtime_query**, predicted chitchat (0.74)
- `Hello You are my supply chain analyst who is dealing with very complex supply chain from north america to phil` — gold **chitchat**, predicted analytics (0.42)
