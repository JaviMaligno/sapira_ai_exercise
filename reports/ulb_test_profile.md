### Source: ulb_test

- Total rows scanned: 200000
- Label distribution: 0=199138, 1=862 (fraud rate ~ 0.4310%)
- Time coverage (approx): 2013-06-21T12:14:25+00:00 → 2013-08-30T20:43:01+00:00 UTC

- Top columns by missingness (10):
  - zip: missing ~ 0.00%
  - transaction_type: missing ~ 0.00%
  - transaction_id: missing ~ 0.00%
  - street: missing ~ 0.00%
  - state: missing ~ 0.00%
  - merchant_name: missing ~ 0.00%
  - merchant_long: missing ~ 0.00%
  - merchant_lat: missing ~ 0.00%
  - long: missing ~ 0.00%
  - lat: missing ~ 0.00%

- Selected column summaries:
  - amount: fill=100.00%, dtypes=['float64']
    - stats: min=1.0, p50≈N/A, max=13149.15, mean=69.204, std=145.990
    - top: 1.19 (30), 1.97 (30), 1.26 (27), 2.54 (27), 1.92 (26)
  - transaction_type: fill=100.00%, dtypes=['object']
    - top: gas_transport (5080), grocery_pos (4647), home (4611), kids_pets (4495), shopping_pos (4473)
  - merchant_name: fill=100.00%, dtypes=['object']
    - top: fraud_Schumm PLC (167), fraud_Kilback LLC (161), fraud_Cormier LLC (138), fraud_Kuhn LLC (133), fraud_Boyer PLC (133)
  - merchant_lat: fill=100.00%, dtypes=['float64']
    - stats: min=19.039532, p50≈N/A, max=66.66935600000001, mean=38.541, std=5.109
    - top: 39.559507 (2), 39.8424 (2), 40.447515 (2), 40.705473 (2), 38.754405 (2)
  - merchant_long: fill=100.00%, dtypes=['float64']
    - stats: min=-166.671575, p50≈N/A, max=-66.952352, mean=-90.214, std=13.746
    - top: -76.430902 (2), -73.343677 (2), -86.101749 (2), -98.389 (2), -87.00382900000001 (2)
  - event_time_ts: fill=100.00%, dtypes=['int64']
    - stats: min=1371816865.0, p50≈N/A, max=1377895381.0, mean=1374816644.088, std=1758789.391
    - top: 1373828405 (2), 1377841444 (2), 1377436130 (2), 1376778665 (2), 1374409599 (2)