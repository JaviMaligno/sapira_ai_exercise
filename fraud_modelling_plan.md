### MVP Plan: Isolation Forest + Basic Rules for DGuard, with Supervised Phase 2 Upgrade

#### Goals

- Ship a functional fraud MVP on live DGuard `bank_transactions` without relying on labels.
- Produce actionable alerts with stable volume, low latency, and simple ops.
- Establish a path to a stronger supervised model once labels become available.

---

### Phase 1: Isolation Forest + Basic Rules (Unsupervised)

#### 1) Data and features

- Sources

  - DGuard Mongo collection `bank_transactions`: use `amount`, `operation_type`, `merchant_clean_name`, `operation_date` (as time), `balance`, `currency`, `description`, `categories`.
  - Ignore `fraud_score` and `is_suspicious` as they are random and not valid labels.
- Point‑in‑time correctness

  - Ensure features for a transaction only use data up to its `operation_date`.
  - For historical backtests, process data in timestamp order and maintain per-account/customer state.
- Feature set v1 (low-latency, robust)

  - Amount and scaling
    - `amount` (signed), `abs_amount`, `log1p_abs_amount`.
    - Robust scaling: QuantileTransformer or IQR-based scaling for heavy tails.
  - Operation type and merchant
    - `operation_type_onehot` (DEBIT/CREDIT/TRANSFER/…).
    - `merchant_token` via hashing trick (e.g., 10–50 bins frequency encoding).
    - Merchant frequency per account (short window): count of transactions to this merchant in last 7/30 days.
  - Velocity features
    - `txn_count_1h`, `txn_count_24h`, `amount_sum_24h` per account_id/user_id.
    - `distinct_merchants_7d` per account.
  - Balance dynamics (if reliable)
    - `balance_delta_1d`, `balance_depletion_rate_24h` (sum of debits / available balance).
  - Time features
    - `hour_of_day`, `day_of_week`, `is_night` flag.
- Categorical handling

  - One-hot for small cardinality: `operation_type`.
  - Hashing or frequency encoding for `merchant_clean_name`.
  - Avoid target encoding (no labels).
- Missing values

  - Impute 0 for counts/sums; median or 0 for amounts; set explicit “UNK” for categoricals.

#### 2) Rules v1 (simple, interpretable)

- Amount rules

  - High absolute amount: above global P99 or account-specific historical P99.5 → flag.
  - Sudden large debit after low activity: `abs_amount > k * median_amount_30d` and `txn_count_7d < m`.
- Velocity rules

  - Rapid repeats: `txn_count_1h >= r` or ≥N debits in 10 minutes.
  - New merchant + high amount: first seen merchant for account and `abs_amount > threshold`.
- Balance rules

  - Balance depletion: `amount_sum_debits_24h / starting_balance_24h > α` (e.g., >60%).
- Merchant/descriptor rules

  - Known risky patterns in `description` or `merchant_clean_name` (“cash out”, known high-risk MCC tags if derivable).
  - “not_found” merchant_name + medium/large amount.
- Rule severity

  - Hard block rules (rare): e.g., extreme amount thresholds.
  - Soft alert rules: all others, to be triaged.

#### 3) Isolation Forest (unsupervised) setup

- Training set definition

  - Use recent N months of transactions.
  - Exclude events flagged by the most extreme rules (top 0.1% outliers) to avoid contaminating “normal” training.
  - Balance sample by account to avoid over-representing high-activity users.
- Preprocessing

  - Fit scalers/encoders on training data only; serialize transformers and IF model.
- Model

  - scikit-learn IsolationForest
    - `n_estimators=300–500`, `max_samples='auto'`, `contamination` unset (score-only); set `random_state`.
  - Output anomaly score via `decision_function` (higher ≈ more normal; invert to “anomaly_score = -decision_function”).
- Thresholding (no labels)

  - Percentile-based: choose τ to cap alert rate (e.g., top 0.3–0.7% most anomalous).
  - Calibrate τ to alert budget and ops capacity; track daily/weekly drift and adjust gradually.

#### 4) Decision policy (hybrid)

- Final flag if ANY:
  - Hard rule triggers, OR
  - Anomaly score > τ2 (very anomalous), OR
  - Soft rule triggers AND anomaly score > τ1 (lower threshold).
- Output risk score
  - Combine rule score (weighted sum of triggered rules) and normalized anomaly score for ranking.
  - Produce reason codes (which rules fired; top 5 contributing features by z-score).

#### 5) Testing and scoring (without labels)

- Backtesting on DGuard

  - Replay last M months in order; compute features point-in-time; estimate alert volume per day.
  - Check alert clusters: by merchant, account, amount bands, hours; ensure no single entity overwhelms.
  - Analyst spot-check: sample top K alerts per day for manual review to estimate precision proxy.
  - Stability tests: sensitivity of alert rate to small τ changes; CPU/latency budget.
- Cross-dataset sanity checks via consolidation

  - Apply the same feature pipeline to ULB and IEEE (map overlapping features: `amount`, category-as-type, time-of-day, basic velocities). Use labels there to:
    - Validate IF hyperparameters (tree count, feature subset, scaling choice) by checking whether top anomalies correlate with fraud in labeled datasets.
    - Validate and tune rules (precision/recall estimates on labeled data) for future supervised phase.
  - Caveat: domain shift; use as directional validation, not final.
- Monitoring metrics

  - Alert rate (% of transactions), unique accounts alerted, per-merchant alert spikes.
  - Mean/median anomaly score; rule hit distribution.
  - Data drift: feature means/quantiles vs. training; sudden spikes in missingness.

#### 6) Serving and ops

- Batch/stream scoring

  - Batch: cron job to score previous hour/day; write alerts to a table/collection.
  - Stream: microservice computes features on the fly (limited to fast counters from Redis/DB), encodes, scores with IF + rules (<150ms).
- Observability

  - Dashboards: daily alert rate, top rules, top merchants/accounts, distribution of anomaly scores.
  - Audit logs: inputs, features, scores, rules fired, versioned model/transformers.
- Governance and iteration

  - Maintain a rule catalog with metadata (owner, rationale, creation date).
  - Monthly rule review: retire low-value rules, add constraints, document changes.

---

### Phase 2: Supervised Upgrade (GBDT) + Hybridization

#### 1) Label strategy for DGuard

- Near-term labels

  - Build a review workflow: analysts label a subset of alerts (fraud/legit). Use as seed labels.
  - Weak supervision: treat strong rules (e.g., confirmed chargeback proxies if any) as noisy positives; random sample of non-alerts as likely negatives (with caution).
  - Time-delayed truth: if chargeback/confirmation processes exist later, ingest as gold labels.
- Data retention

  - Store features and predictions at event time to create an offline training set (point-in-time correct).

#### 2) Training datasets

- Supervised sources (now)

  - ULB as primary supervised training data (dense merchant + geo + transaction types). Time-split CV.
  - IEEE as secondary: add with domain feature `dataset` and selective engineered blocks (avoid very sparse `id_*` unless helpful).
- Domain adaptation for DGuard

  - Start with a model trained on ULB (+IEEE) and evaluate offline on DGuard (no labels) for score distribution sanity.
  - Collect DGuard labels over time; then fine-tune or retrain with mixed data:
    - Strategy A: Train separate models per domain and blend scores.
    - Strategy B: Unified model with `dataset`/domain indicators and sample weighting to favor DGuard.
- Feature set (supervised)

  - All Phase 1 features plus:
    - Merchant history features (e.g., proportion of first-time merchants that were fraud).
    - Account tenure, inter-arrival times, recent mean/std of amounts.
    - Email/device (if available in DGuard later).
  - Add Phase 1 anomaly score and rule score as features (stacking).

#### 3) Modeling framework

- GBDT (scikit-learn HistGradientBoosting)

  - Handle imbalance with `class_weight={1: pos_weight}` (inverse prevalence) on train split.
  - Hyperparameters (initial): `learning_rate=0.1`, `max_depth=8`, `max_iter=300`, early stopping.
  - Calibration: Platt/Isotonic if needed; threshold tuned to business cost curve.
- Evaluation

  - Offline: time-aware CV, primary AUPRC; report precision@k (alert budget), cost-weighted F1.
  - Online: shadow mode on DGuard traffic; progressive rollout with gates (alert rate, latency, FP drift).
- Explainability

  - SHAP global importance; per-alert reason codes.
  - Compare with rule reasons; use insights to refine rules and features.
  - Enhanced ULB baseline (200k, chronological split) now achieves AP≈0.923, P@0.5%≈0.835, P@1%≈0.475, P@5%≈0.0985 using expanded point-in-time features, stacked IF/rule scores, and per-category calibration.

Notes:
- Latest ULB baseline (200k rows, chronological split) yields AP≈0.8709, P@0.5%≈0.785, P@1%≈0.4575, P@5%≈0.098. A segmented-by-category variant lifts P@0.5% to ≈0.815 with slightly lower AP≈0.8836; we keep a single unified model for now due to simplicity and similar overall performance.

#### 4) Testing and scoring (supervised)

- Unified offline bench

  - Train on ULB (+IEEE), validate on held-out time windows.
  - Measure lift from adding IF anomaly score + rule score as features.
  - Robustness: check per-category, per-merchant segments; simulate drift.
- Online gates

  - Roll out to 1–5% of DGuard traffic in shadow → then actioned.
  - Safety thresholds: +10% FP rate or >20ms latency regression triggers rollback.

#### 5) Iteration and continuous improvement

- Feature store migration (optional but recommended)

  - Define features once; use offline (training) and online (serving) layers to prevent train-serve skew.
  - Redis/Cosmos/Feast for low-latency online features.
- Rule learning and refinement

  - Mine decision paths and SHAP patterns to propose new rules.
  - Maintain precision/recall estimates for each rule on labeled datasets; retire underperformers.
- Data quality and drift

  - Automated checks: schema changes, missingness spikes, distribution shifts.
  - Retrain cadence: monthly or on drift trigger.

---

### Step-by-step breakdown

- Week 0–1: Phase 1 MVP

  - Implement DGuard extractor and point-in-time feature builder.
  - Fit scalers/encoders; train IF on “normal” data; pick τ for alert budget.
  - Implement rules v1; combine with IF; produce alerts with reason codes.
  - Backtest, sanity-check volumes; create dashboards; ship shadow mode or batch alerts.
- Week 2–3: Phase 1 hardening

  - Analyst feedback loop; tune τ and rule thresholds.
  - Add velocity/balance features; improve merchant encoding.
  - Build unit tests for feature transforms; add drift monitors.
- Week 4–6: Phase 2 preparation

  - Unify ULB (+IEEE) via current ETL; build supervised training pipeline.
  - Train baseline GBDT; integrate anomaly and rule scores as features.
  - Evaluate and document; prep for shadow on DGuard.
- Week 7+: Phase 2 rollout

  - Shadow on DGuard; compare vs Phase 1 alerts.
  - Progressive enablement; maintain hybrid (GBDT primary, IF+rules safety net).

---

### What to analyze and watch for (both phases)

- Alert volume stability and concentration (merchant/account bursts).
- FP patterns from analyst review; add targeted rules/features.
- Feature drift: `amount` distribution shifts, new merchants, changing operation types.
- Latency budget (<150ms) and throughput; memory of encoders/hashing.
- Model health: score distributions, SHAP drift, leakage checks, stale features.

---

### Tooling and frameworks

- Batch: pandas/pyarrow; scikit-learn IsolationForest/XGBoost/LightGBM.
- Stream: FastAPI (or similar), joblib/pkl model artifacts, Redis for counters.
- Storage: Mongo read; Parquet for offline snapshots; simple registry for model/versioning; later move to Feast/MLflow as needed.
- Observability: Grafana/Prometheus or lightweight logging + notebooks; alerting on drift/volume.

This plan gets a working unsupervised MVP into production quickly with clear rule-based safety nets, while laying the groundwork to incorporate supervised learning using ULB/IEEE and DGuard labels for a substantially stronger Phase 2 hybrid system.

---

### Current Implementation Status vs Plan

#### Completed Phase 1 Components
- ✅ DGuard data extraction and point-in-time feature engineering
- ✅ Isolation Forest training with contamination handling
- ✅ Basic rules implementation (high amount, velocity, novelty)
- ✅ Anomaly score thresholding with per-category fallback
- ✅ Alert volume controls and drift monitoring (basic)
- ✅ FastAPI serving infrastructure with MongoDB integration

#### Completed Phase 2 Components  
- ✅ Supervised GBDT training on ULB/IEEE datasets
- ✅ Feature stacking (IF anomaly score + rule score)
- ✅ Calibrated probability outputs with isotonic regression
- ✅ Per-category threshold optimization
- ✅ SHAP explainability and global feature importance
- ✅ Shadow scoring pipeline for validation

#### Implementation Gaps Identified

**Phase 1 Gaps:**
- ❌ **Transaction deduplication**: No explicit dedupe in ETL pipeline
- ❌ **Operation type normalization**: Inconsistent lowercase/variant mapping
- ❌ **Currency handling**: No FX conversion or normalization beyond categorical
- ❌ **Global merchant frequency artifact**: merchant_vocab.json built but no frequency map for DGuard scoring
- ❌ **Enhanced calendar features**: Missing weekend/holiday flags beyond basic time features

**Phase 2 Gaps:**
- ❌ **Advanced drift monitoring**: Need PSI/rank-stability metrics beyond basic quantile tracking  
- ❌ **Analyst feedback loop**: No workflow for label collection or weekly review integration
- ❌ **Domain adaptation refinement**: Limited fine-tuning strategy for DGuard vs ULB/IEEE differences

**Optional Future Enhancements:**
- BIN-to-brand/country mapping for enhanced merchant intelligence
- MCC/category lookup tables for transaction classification
- IP geolocation and ASN mapping for fraud patterns
- Feature store migration for train-serve consistency
