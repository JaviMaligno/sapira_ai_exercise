### Source: dguard_tx

- Total rows scanned: 355
- Time coverage (approx): 2023-05-01T00:00:00+00:00 → 2025-08-07T14:40:42.817000+00:00 UTC

- Top columns by missingness (10):
  - risk_score: missing ~ 96.62%
  - risk_flag: missing ~ 92.68%
  - fraud_status: missing ~ 92.68%
  - transaction_type: missing ~ 82.82%
  - currency: missing ~ 82.54%
  - balance: missing ~ 20.85%
  - categories: missing ~ 16.06%
  - merchant_name: missing ~ 15.77%
  - account_id: missing ~ 6.48%
  - transaction_id: missing ~ 1.41%

- Selected column summaries:
  - amount: fill=99.72%, dtypes=['float64']
    - stats: min=-301.33, p50≈N/A, max=2500.0, mean=-12.614, std=180.626
    - top: -25.0 (21), 10.0 (14), -26.5 (13), -100.0 (13), 28.25 (12)
  - transaction_type: fill=17.18%, dtypes=['object']
    - top: DEBIT (30), card_payment (11), online_payment (9), subscription (5), transfer (2)
  - merchant_name: fill=84.23%, dtypes=['object']
    - top: not_found (79), Google Pay (25), Amazon (23), La Banque Postale (21), AliExpress (17)
  - risk_score: fill=3.38%, dtypes=['float64']
    - stats: min=0.95, p50≈N/A, max=0.9534, mean=0.953, std=0.001
    - top: 0.9534 (11), 0.95 (1)
  - risk_flag: fill=7.32%, dtypes=['object']
    - top: False (14), True (12)
  - event_time_str: fill=99.72%, dtypes=["datetime64[ns, FixedOffset(datetime.timedelta(0), 'UTC')]"]
    - top: 2023-05-01 00:00:00+00:00 (36), 2023-05-29 00:00:00+00:00 (35), 2023-05-07 00:00:00+00:00 (28), 2023-05-28 00:00:00+00:00 (22), 2023-05-21 00:00:00+00:00 (21)