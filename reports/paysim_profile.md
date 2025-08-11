### Source: paysim

- Total rows scanned: 400000
- Label distribution: 0=399794, 1=206 (fraud rate ~ 0.0515%)
- Time coverage (approx): 1970-01-01T00:00:01+00:00 → 1970-01-01T00:00:18+00:00 UTC

- Top columns by missingness (10):
  - transaction_type: missing ~ 0.00%
  - orig_balance_before: missing ~ 0.00%
  - orig_balance_after: missing ~ 0.00%
  - nameOrig: missing ~ 0.00%
  - is_fraud: missing ~ 0.00%
  - isFlaggedFraud: missing ~ 0.00%
  - event_time_ts: missing ~ 0.00%
  - dest_balance_before: missing ~ 0.00%
  - dest_balance_after: missing ~ 0.00%
  - counterparty_id: missing ~ 0.00%

- Selected column summaries:
  - amount: fill=100.00%, dtypes=['float64']
    - stats: min=0.1, p50≈N/A, max=10000000.0, mean=172750.636, std=286132.770
    - top: 4315.9 (3), 7257.7 (3), 46243.45 (3), 5176.47 (2), 7014.89 (2)
  - transaction_type: fill=100.00%, dtypes=['object']
    - top: CASH_OUT (36042), PAYMENT (33393), CASH_IN (21541), TRANSFER (8302), DEBIT (722)
  - event_time_ts: fill=100.00%, dtypes=['int64']
    - stats: min=1.0, p50≈N/A, max=18.0, mean=12.717, std=3.201
    - top: 15 (11090), 17 (11000), 16 (10587), 14 (10347), 9 (9351)