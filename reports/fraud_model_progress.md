### Fraud Model Progress Log

This document tracks the status, decisions, and key metrics of the fraud detection MVP and subsequent iterations.

#### Project context
- Objective: MVP fraud detection for DGuard `bank_transactions` using Isolation Forest + basic rules (Phase 1), with a Phase 2 supervised upgrade (GBDT) trained on unified labeled sources (ULB, IEEE) and augmented with DGuard labels when available.
- Latency target: <150 ms for online scoring.
- Outputs: Alerts with reason codes and ranked risk scores.

---

### Current status (latest)

- Phase: 1 (Unsupervised + Rules)
- Data source: Mongo `dguard.bank_transactions`
- Pipeline: feature engineering (amount, time-of-day, velocity), Isolation Forest scoring, rules aggregation.
- Latest run: see metrics below.

#### Latest run metrics
```
total_rows: 355
alerts: 3
alert_rate: 0.00845 (~0.85%)
anomaly_tau: 0.10469
top_rules: { extreme_amount: 1, high_amount: 2, new_merchant_high: 1, rapid_repeats: 47 }
```

#### Notable findings
- Merchant coverage is high; `transaction_type` is sparse; risk signals are present but not trustworthy (random).
- Velocity rule (`rapid_repeats`) frequently triggers, likely due to low volume and clustered times; needs calibration.

---

### Decision log

- 2025-08-12: Shipped Phase 1 IF + rules MVP with percentiled anomaly threshold and minimal ruleset.
- 2025-08-12: Added robust rolling window counts (1h/24h) to handle NaT and unsorted timestamps.

---

### Backlog / next steps

- Tune thresholds: lower rule weight for `rapid_repeats`, cap alerts per merchant/account/day.
- Exclude extreme rule hits from IF training to reduce contamination; retrain.
- Add additional features: per-merchant frequency per account (7/30d), balance depletion features.
- Build analyst feedback loop and weekly rule review.
- Begin Phase 2 prep: unify ULB/IEEE, train baseline GBDT; add IF and rule scores as features.

---

### Metrics to track over time

- Alert volume: daily alert rate, unique accounts/merchants alerted, top rules by frequency.
- Score distributions: anomaly score histogram, rule score distribution.
- Drift: feature quantiles vs. training, missingness spikes, new merchants share.
- Ops: average scoring latency, error rate.

---

### Artifact pointers

- Phase 1 code: `src/fraud_mvp/phase1.py`
- Latest alerts CSV: `reports/phase1/alerts_phase1.csv`
- Latest summary JSON: `reports/phase1/alerts_phase1_summary.json`
- Data profiling: `reports/*_profile.md`



#### Labeled sanity-check (ULB, Isolation Forest)
```
rows: 200000
fraud_rate: 0.008225
average_precision: 0.00826
precision_at:
  0.1%: 0.0050
  0.5%: 0.0100
  1.0%: 0.0100
```
