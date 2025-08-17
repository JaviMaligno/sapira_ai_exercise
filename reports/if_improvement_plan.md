### Isolation Forest Improvement Plan (Phase 1)

Scope: Strengthen the unsupervised MVP leveraging Isolation Forest while staying aligned with fraud_detection.md, validated primarily on ULB labels and applied to DGuard.

Prioritized actions
- Feature upgrades
  - Amount context:
    - Per-entity percentiles and z-scores: p50/p90/p99 and standard z per account/card.
    - Category-aware scaling: median/IQR and z_iqr per `operation_type`; winsorize amounts per category at p99.5.
  - Temporal:
    - Cyclic hour features (`sin(2πh/24)`, `cos(2πh/24)`); night-hour indicator.
    - Time since last transaction per entity (account/card) and per entity-category pair.
  - Velocity and novelty:
    - Rolling counts/sums per account for 1h/24h (existing) and per (entity, merchant/category) for 7/30d.
    - New-merchant and new-category flags for entity.
  - Merchant/category profiling:
    - Merchant frequency rank for card/account; category transition novelty flag.
  - Balance-derived (if available):
    - Balance depletion ratio over last 24h; debit share of recent activity.

- Robust preprocessing
  - Winsorize amount at p99.5 per category before computing z_iqr.
  - Fixed vocab encoders for high-cardinality strings (merchant) to avoid train/serve skew.

- Training hygiene
  - Contamination guard: exclude extreme/high category-aware amounts for IF fitting.
  - Balance per-entity sampling when training IF to avoid dominance by heavy users.
  - Minor grid: `n_estimators` (done), `max_features` in {0.5, 0.7, 1.0}.

Work completed (to date)
- Implemented: time-since-last, per-account amount stats (mean/std/p50/p90/p99), amount z-score, new-merchant flag, merchant rolling counts (7/30d), cyclic hour features, category-aware robust z-score (IQR), per-category novelty/time-since-last, contamination guard, per-category thresholds, daily alert caps.
- ULB tuning: best n_estimators=200; AP≈0.00841; P@0.5%≈0.0130; P@5%≈0.0082.
- DGuard pipeline re-runs with new features and updated DB; latest alerts=4 (≈1.13%).

Next work items
- Evaluate merchant/category transition features and frequency ranks; consider adding top-k transition rarity as feature. (category transition rarity implemented per-account)
- Add fixed-vocabulary merchant encoders to avoid train/serve skew.
- Add category-aware amount winsorization at p99.5 before z_iqr computation in Phase 1.
- Conduct per-category threshold tuning on ULB slices and port thresholds to Phase 1. (ULB per-category thresholds generated and consumed by Phase 1)
- Add drift monitors (new-merchant share, category-wise amount quantiles, missingness) and logging hooks.

### Phase 2 GBDT – next steps
- Segment models/thresholds by `operation_type`; train per-category GBDTs for top-volume categories and a fallback model.
- Add Isolation Forest outputs (anomaly_score, rule_score proxy) as features with time-aware fitting (fit IF on train-legit only; transform train/test).
- Add stronger per-card history features (point-in-time): expanding/rolling per-card amount stats (mean/std/median, p90/p99), z-scores; counts/sums in 1h/6h/24h; unique merchants 7/30d; days since last seen merchant/category.
- Replace one-hot for high-cardinality `merchant_name` with frequency/target encoding (fit on train folds only; use frequency ranks at serve-time).
- Time-aware hyperparameter tuning for GBDT: depth/leaves, learning_rate vs iterations with early stopping, regularization.
- Per-step ablation: baseline → +IF score → +rolling/history → +target/frequency encoding → +segmented models; log AP and P@k after each step.

- Thresholding & alerting
  - Separate anomaly thresholds per `operation_type` reflecting different base rates.
  - Daily caps per (account, merchant) to avoid alert floods.
  - Rule refinements: category-aware amount thresholds; night-hour boost; first-time-category + high amount.

- Evaluation and monitoring
  - ULB as proxy labels: track AP and P@k, especially P@5% (targets: MVP ≥0.05, Product ≥0.10, Ideal ≥0.20).
  - Slice metrics by category, hour bands, merchant buckets; ablation to quantify feature gains.
  - Logging for drift: new merchants share, amount quantiles per category, missingness.

Implementation notes
- Add features to both ULB evaluation script and DGuard Phase 1 pipeline.
- Ensure point-in-time correctness when computing rolling/novelty features.
- Cache online counters in Redis for latency (<150ms) during serving.


