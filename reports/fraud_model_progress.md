### Fraud Model Progress Log

This document tracks the status, decisions, and key metrics of the fraud detection MVP and subsequent iterations.

#### Project context
- Objective: MVP fraud detection for DGuard `bank_transactions` using Isolation Forest + basic rules (Phase 1), with a Phase 2 supervised upgrade (GBDT) trained on unified labeled sources (ULB, IEEE) and augmented with DGuard labels when available.
- Latency target: <150 ms for online scoring.
- Outputs: Alerts with reason codes and ranked risk scores.

---

### Current status (latest)

- Phase: 1 (Unsupervised + Rules)
- Data source: Mongo `dguard_transactions.bank_transactions` (updated)
- Pipeline: feature engineering (amount, time-of-day, velocity), Isolation Forest scoring, rules aggregation.
- Latest run: see metrics below.

#### Latest run metrics
```
total_rows: 355
alerts: 3
alert_rate: 0.00845 (~0.85%)
anomaly_tau: 0.01683
top_rules: { extreme_amount: 1, high_amount: 2, new_merchant_high: 1, rapid_repeats: 47 }
alerts_per_category: { salary: 1, transfer: 1, UNK: 1 }
```

#### Notable findings
- Merchant coverage is high; `transaction_type` is sparse; risk signals are present but not trustworthy (random).
- Velocity rule (`rapid_repeats`) frequently triggers, likely due to low volume and clustered times; needs calibration.

---

### Decision log

- 2025-08-13: Switched Mongo database to `dguard_transactions`; re-ran exploration and source profiling; updated ETL and docs.
- 2025-08-12: Shipped Phase 1 IF + rules MVP with percentiled anomaly threshold and minimal ruleset.
- 2025-08-12: Added robust rolling window counts (1h/24h) to handle NaT and unsorted timestamps.
- 2025-08-12: Enhanced features (time-since-last, new-merchant flags, per-account amount stats/z-score) and added contamination guard for IF training; re-ran DGuard and ULB evaluation.
- 2025-08-12: Added merchant frequency features (7/30d) and per-(account,day) alert cap; reduced rapid-repeat rule weight. Pipeline ready to re-run when Mongo fetch succeeds.
 - 2025-08-13: Added cyclic hour features and category-aware robust amount z-score (IQR) to ULB eval and DGuard; tuned IF on ULB (best n_estimators=200); re-ran DGuard.
 - 2025-08-13: Added per-category novelty/time-since-last features and per-category thresholds in Phase 1; refreshed DGuard alerts.
 - 2025-08-13: Logged latest DGuard metrics (alerts=4, alert_rate≈1.13%, tau≈0.14505) after per-category thresholding.
 - 2025-08-13: Applied ULB-derived per-category thresholds and category transition rarity; latest DGuard alerts=3 (≈0.85%), tau≈0.14398.
 - 2025-08-13: Added fixed-vocab encoders, merchant frequency features, and drift logging; regenerated DGuard alerts and daily drift report.
 - 2025-08-13: Applied ULB ablation findings to Phase 1. Feature stack set to amount_core + velocity + cat_robust + novelty_cat based on best P@0.5% (0.013) and overall AP on ULB. Time-of-day features de-prioritized for IF; merchant frequency retained for lower-threshold review tiers.
 - 2025-08-13: Kicked off Phase 2 (GBDT) on ULB with time-aware split. Baseline scores: AP≈0.872, P@0.5%≈0.80, P@1%≈0.458, P@5%≈0.098. Next: add IEEE, calibrate thresholds, SHAP for explanations.
 - 2025-08-13: Added stacked IF features (anomaly_score + rule_score) to GBDT. New ULB scores: AP≈0.878, P@0.5%≈0.81, P@1%≈0.46, P@5%≈0.0985.
  - 2025-08-14: Ran segmentation ablation (`src/fraud_mvp/gbdt_ulb_segmented.py`). Results saved to `reports/phase2/ulb_gbdt/ulb_gbdt_ablation.{json,csv}`. Decision: keep single unified model (slightly higher AP and simpler ops); revisit segmentation post-IEEE/domain features.

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
- ULB eval summary: `reports/phase1/ulb_eval/ulb_if_eval_summary.json`
- GBDT (ULB) summary: `reports/phase2/ulb_gbdt/ulb_gbdt_summary.json`
- GBDT (ULB) code: `src/fraud_mvp/gbdt_ulb.py`
- GBDT segmented code: `src/fraud_mvp/gbdt_ulb_segmented.py`
- GBDT segmentation ablation: `reports/phase2/ulb_gbdt/ulb_gbdt_ablation.csv`, `reports/phase2/ulb_gbdt/ulb_gbdt_ablation.json`
### Phase 2 (GBDT) – initial ULB results

```
rows: 200000
train_rows: 160000, test_rows: 40000
fraud_rate_train: 0.00904, fraud_rate_test: 0.00495
average_precision: 0.8709
precision_at:
  0.5%: 0.7850
  1.0%: 0.4575
  5.0%: 0.0980
```

Planned next steps:
- Add IEEE data and re-train with domain feature; time-aware validation.
- Calibrate serving thresholds for DGuard alert budgets; log P@k on shadow traffic.
- Add SHAP explanations for global and per-transaction insights; feed rule refinements.
- Data profiling: `reports/*_profile.md`

#### Phase 2 (GBDT) – segmentation ablation
```
gbdt_single+history:
  AP: 0.8863
  P@0.5%: 0.805
  P@1.0%: 0.470
  P@5.0%: 0.0985

gbdt_segmented_by_category:
  AP: 0.8836
  P@0.5%: 0.815
  P@1.0%: 0.465
  P@5.0%: 0.0970
```

Decision:
- Keep single unified model for now (slightly higher AP and simpler ops). Revisit segmentation after adding IEEE/domain indicator.

#### Phase 2 (GBDT) – enhanced features + calibration
```
overall:
  AP: 0.9231
  P@0.5%: 0.835
  P@1.0%: 0.475
  P@5.0%: 0.0985
time-aware CV:
  train<=60% → val 60–80%: AP=0.9386, P@0.5%=0.985, P@1%=0.7225, P@5%=0.1545
  train<=80% → val 80–100%: AP=0.9231, P@0.5%=0.835, P@1%=0.475, P@5%=0.0985
per-category thresholds: stored in `reports/phase2/ulb_gbdt/ulb_gbdt_enhanced_summary.json`
```

Decision:
- Adopt enhanced point-in-time features, stacked IF/rule scores, and per-category threshold calibration for Phase 2 baseline. Proceed to add IEEE and domain indicator next.

Artifacts added:
- `reports/phase2/ulb_gbdt/ulb_gbdt_enhanced_summary.json`
- `reports/phase2/ulb_gbdt/ulb_gbdt_cv.json`
- `reports/phase2/ulb_gbdt/ulb_gbdt_ablation_overall.json`



#### Labeled sanity-check (ULB, Isolation Forest)
```
rows: 200000
fraud_rate: 0.008225
average_precision: 0.00841
precision_at:
  0.1%: 0.0000
  0.5%: 0.0130
  1.0%: 0.0080
  5.0%: 0.0082
```

ULB ablation (P@0.5%):
- baseline_amount: 0.0070
- +time: 0.0060
- +velocity: 0.0100
- +cat_robust (winsor z_iqr): 0.0130 (best)
- +novelty_cat: 0.0120
- +merchant_freq: 0.0100 (but best P@5% ≈ 0.0091)

Rationale: prioritize features that lift top-of-queue precision; reserve merchant frequency for broader review budgets.

Targets for precision@5%:
- MVP: >= 0.05
- Product: >= 0.10
- Ideal: >= 0.20
