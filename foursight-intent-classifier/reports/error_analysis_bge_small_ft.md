# Error analysis — bge_small_ft (pooled out-of-fold, grouped CV)

accuracy 0.7274 · macro-F1 0.5809 · **OVERALL 0.6395** · errors 519 / 1904

## Per-class

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.377 | 0.363 | 0.370 | 256 |
| realtime_query | 0.760 | 0.817 | 0.787 | 782 |
| analytics | 0.651 | 0.630 | 0.640 | 257 |
| disruptions | 0.714 | 0.536 | 0.612 | 28 |
| chitchat | 0.632 | 0.706 | 0.667 | 17 |
| knowledge_base | 0.895 | 0.906 | 0.900 | 509 |
| other | 0.250 | 0.055 | 0.090 | 55 |

## Most confused pairs (gold → predicted)

- shipment_information → realtime_query: 135
- realtime_query → shipment_information: 89
- analytics → realtime_query: 45
- realtime_query → analytics: 40
- analytics → shipment_information: 28
- other → knowledge_base: 21
- analytics → knowledge_base: 21
- knowledge_base → shipment_information: 18
- knowledge_base → analytics: 17
- shipment_information → analytics: 16

Operational-triad errors: {'shipment_information->realtime_query': 135, 'realtime_query->shipment_information': 89, 'analytics->realtime_query': 45, 'realtime_query->analytics': 40, 'analytics->shipment_information': 28, 'shipment_information->analytics': 16, 'total_triad_errors': 353}

## Calibration and review flag

- temperature T = 1.039; ECE 0.0406 → 0.0276; NLL 0.8460 → 0.8451; Brier 0.4044 → 0.4033
- mean calibrated confidence: correct 0.779 vs wrong 0.641; margin 0.662 vs 0.448; 7 errors with confidence ≥ 0.9
- needs_review = confidence < 0.591: flags 20.0% of rows, error recall 38.7%, precision 0.53, accuracy on unflagged 0.791
- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag 20.1%, error recall 38.2%, precision 0.52, accuracy on unflagged 0.789, OVERALL on unflagged 0.6059
- AURC 0.1393 (full-coverage risk 0.2726)

## Rule-layer ablation (calibrated probabilities)

| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0.7274 | 0.5809 | 0.6395 | 353 |
| 0.50 | 238 | 12 | 5 | 6 | 0.7269 | 0.5891 | 0.6442 | 357 |
| 0.60 | 392 | 12 | 5 | 6 | 0.7269 | 0.5891 | 0.6442 | 357 |
| 0.70 | 583 | 12 | 5 | 6 | 0.7269 | 0.5891 | 0.6442 | 357 |
| 0.80 | 1000 | 12 | 5 | 6 | 0.7269 | 0.5891 | 0.6442 | 357 |
| 1.01 | 1904 | 12 | 5 | 6 | 0.7269 | 0.5891 | 0.6442 | 357 |

## Failure categories

Short (≤2 words) inputs: n=204, accuracy 0.642. Rows in conflicting-label duplicate groups: n=81, accuracy 0.383 (10% of all errors). Rare-class recall: disruptions 0.54, chitchat 0.71, other 0.05.

### triad: semantic — 262 errors, mean confidence 0.66

- `Show me only delivered loads from 2026/1/1 to now` — gold **analytics**, predicted realtime_query (0.89)
- `Following 1531002 1531001 1531324` — gold **shipment_information**, predicted realtime_query (0.89)
- `IS THERE A DELIVERY TODAY BY A TRAILER WITH THE DIGITS 1159` — gold **shipment_information**, predicted realtime_query (0.88)
- `truck number 208398576-249` — gold **shipment_information**, predicted realtime_query (0.88)
- `Can you show me the shipments from 1197 on 08/06/2026?` — gold **shipment_information**, predicted realtime_query (0.88)
- `Can you show me all of the shipments I have listed for Jan 30 2026?` — gold **analytics**, predicted realtime_query (0.88)
- `Can you show me all of the shipments I have listed for Jan 30 2026?` — gold **shipment_information**, predicted realtime_query (0.88)
- `which carriers has the most loads late today` — gold **analytics**, predicted realtime_query (0.88)

### knowledge_base vs operational — 99 errors, mean confidence 0.67

- `find where the "expedite" flag/field is located on the load details screen in the UI?` — gold **analytics**, predicted knowledge_base (0.91)
- `how to check all ther trailer to store 3024` — gold **analytics**, predicted knowledge_base (0.91)
- `how to send the link` — gold **other**, predicted knowledge_base (0.90)
- `is Fourkites using Grapphopper? to get geolocations?` — gold **other**, predicted knowledge_base (0.90)
- `is there a way to see which set up we have with each carrier in fourkites, if eld or phone number` — gold **analytics**, predicted knowledge_base (0.90)
- `how I can get all containers in transit ETA MVD?` — gold **analytics**, predicted knowledge_base (0.90)
- `how can I connect you to a powerBI report from where you can read shipment data` — gold **other**, predicted knowledge_base (0.90)
- `why was this marked supertracked` — gold **realtime_query**, predicted knowledge_base (0.90)

### short/underspecified — 73 errors, mean confidence 0.59

- `0034835176` — gold **shipment_information**, predicted realtime_query (0.88)
- `20260901` — gold **shipment_information**, predicted realtime_query (0.88)
- `56910268` — gold **shipment_information**, predicted realtime_query (0.87)
- `Carrier management` — gold **analytics**, predicted knowledge_base (0.82)
- `TRHU5135923` — gold **shipment_information**, predicted realtime_query (0.82)
- `120118269` — gold **shipment_information**, predicted realtime_query (0.81)
- `SOU1001410653` — gold **shipment_information**, predicted realtime_query (0.80)
- `scheduled` — gold **realtime_query**, predicted shipment_information (0.77)

### triad: temporal/aggregate cue — 37 errors, mean confidence 0.68

- `I want to see last month, TL, source from hazelton, geodis nashville and sparks. Just give me FourKites status` — gold **realtime_query**, predicted analytics (0.88)
- `what deliveries are behind schedule for delivery tomorrow in Newnan GA, Hazleton PA and Marshall MO? also summ` — gold **shipment_information**, predicted realtime_query (0.87)
- `last 30 days` — gold **shipment_information**, predicted realtime_query (0.87)
- `How many missed deliveries have we had for the locations Butner, Maulden, Elloree and Dunn in the past 30 days` — gold **realtime_query**, predicted analytics (0.87)
- `show me all departures for carrier DLS in the past 7 days` — gold **shipment_information**, predicted analytics (0.86)
- `show me all shipments in spain which were late in the last three months` — gold **analytics**, predicted realtime_query (0.86)
- `how many tl moves took place yesterday?` — gold **shipment_information**, predicted analytics (0.86)
- `can you show me all deliveries to piggly wiggly since 01/01/2026?` — gold **shipment_information**, predicted realtime_query (0.86)

### other / out-of-scope boundary — 26 errors, mean confidence 0.57

- `CMA BEX. Is there any benchmark on transit time?` — gold **other**, predicted analytics (0.87)
- `there is a existing powerBI data set I want you to read that powerBI` — gold **other**, predicted analytics (0.86)
- `put results in word doc` — gold **other**, predicted analytics (0.78)
- `purchasing document status-PO 4507153487` — gold **other**, predicted realtime_query (0.78)
- `purchasing document status - 4507125913` — gold **other**, predicted realtime_query (0.78)
- `Pickup and delivery number to use for this will be N1091681` — gold **other**, predicted realtime_query (0.77)
- `list coorspoding sn for these pos 0108596326 0107628575 0108834446 0108842669 0108596790 0108841590 0107460504` — gold **other**, predicted realtime_query (0.77)
- `Make the pass by pass in english` — gold **other**, predicted shipment_information (0.77)

### disruptions boundary — 15 errors, mean confidence 0.49

- `the route of BS20260723-1235 have montain regions with curves?` — gold **disruptions**, predicted realtime_query (0.88)
- `me refiero solo al BS20260723-1235. Tiene o pasara poe algun punto de riesgo?` — gold **disruptions**, predicted realtime_query (0.84)
- `What is the current port condition at Shanghai port` — gold **disruptions**, predicted shipment_information (0.58)
- `Can you show me all of our containers that will experience delays due to having to go around the Panama Canal` — gold **disruptions**, predicted shipment_information (0.52)
- `What is the wait time for carriers crossing the World Trade Bridge in Laredo?` — gold **disruptions**, predicted realtime_query (0.49)
- `existe algun peligro reportado en toda la ruta?` — gold **disruptions**, predicted analytics (0.49)
- `existe algun peligro reportado en toda la ruta?` — gold **disruptions**, predicted analytics (0.49)
- `provide summary of plant wise, shipments impacted by dolpin` — gold **disruptions**, predicted analytics (0.44)

### chitchat boundary — 7 errors, mean confidence 0.38

- `que es lo que puedes hacer?` — gold **chitchat**, predicted shipment_information (0.49)
- `que es lo que puedes hacer?` — gold **chitchat**, predicted shipment_information (0.49)
- `Hello You are my supply chain analyst who is dealing with very complex supply chain from north america to phil` — gold **chitchat**, predicted analytics (0.40)
- `¿Me puedes decir que envíos de los que vienen dirigidos a México vienen tarde?` — gold **realtime_query**, predicted chitchat (0.37)
- `ya do that` — gold **realtime_query**, predicted chitchat (0.33)
- `Put your response in a format that can be copied and shared` — gold **realtime_query**, predicted chitchat (0.30)
- `okay what u find` — gold **realtime_query**, predicted chitchat (0.27)
