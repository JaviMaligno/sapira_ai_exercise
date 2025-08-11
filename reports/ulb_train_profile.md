### Source: ulb_train

- Total rows scanned: 200000
- Label distribution: 0=198355, 1=1645 (fraud rate ~ 0.8225%)
- Time coverage (approx): 2012-01-01T00:00:18+00:00 → 2012-04-13T08:31:25+00:00 UTC

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
    - stats: min=1.0, p50≈N/A, max=17897.24, mean=71.170, std=161.929
    - top: 1.14 (33), 2.24 (29), 1.3 (27), 3.08 (26), 1.01 (26)
  - transaction_type: fill=100.00%, dtypes=['object']
    - top: gas_transport (5072), grocery_pos (4779), home (4757), shopping_pos (4525), kids_pets (4295)
  - merchant_name: fill=100.00%, dtypes=['object']
    - top: fraud_Kilback LLC (168), fraud_Schumm PLC (165), fraud_Dickinson Ltd (153), fraud_Kuhn LLC (144), fraud_Cormier LLC (142)
  - merchant_lat: fill=100.00%, dtypes=['float64']
    - stats: min=19.029798, p50≈N/A, max=67.510267, mean=38.539, std=5.105
    - top: 42.340158 (2), 42.192448 (2), 38.024152 (2), 41.483709000000005 (2), 43.317890000000006 (2)
  - merchant_long: fill=100.00%, dtypes=['float64']
    - stats: min=-166.671242, p50≈N/A, max=-66.967742, mean=-90.202, std=13.759
    - top: -83.849305 (2), -94.762634 (2), -81.732225 (2), -77.92343199999999 (2), -79.56087 (2)
  - event_time_ts: fill=100.00%, dtypes=['int64']
    - stats: min=1325376018.0, p50≈N/A, max=1334305885.0, mean=1330141691.962, std=2573885.909
    - top: 1332521314 (2), 1331380523 (2), 1330119430 (2), 1330872551 (2), 1326921432 (2)