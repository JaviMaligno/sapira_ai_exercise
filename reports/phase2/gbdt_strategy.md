### Phase 2 – GBDT Modelling Strategy (ULB first)

#### Objective
Build a strong supervised baseline for card-fraud using Gradient-Boosted Decision Trees, with low-latency serving and clear paths for explainability and thresholding. Start on ULB, then extend to IEEE and adapt to DGuard.

---

### Data and split
- Dataset: ULB Credit Card Fraud (European Cardholders)
  - Files: `fraudTrain.csv`, `fraudTest.csv` (we currently use train with 200k rows for fast iteration)
  - Label: `is_fraud`
- Time-aware split
  - Sort chronologically by `event_time` (from `unix_time`) and split 80/20 into train/test to mimic production recency
  - Rationale: avoid temporal leakage; reflect deployment conditions

---

### Features and engineering
- Amount core
  - `amount`, `abs_amount`, `log1p_abs_amount`
  - Rationale: heavy-tailed spending; log stabilizes variance
- Category-robust amount
  - Per-`operation_type` winsorization at p99.5 → `amount_winsor_cat`
  - Robust z-score per category: `amount_z_iqr_cat = (winsor − median) / IQR`
  - Rationale: aligns distributions across categories; our IF ablation showed this strongly lifts top-k precision
- Time features (lightweight)
  - `hour`, `hour_sin/cos`, `dow`, `is_night`
  - Rationale: captures diurnal patterns; kept minimal to avoid overemphasis on time-only signals
- Short-term velocities (per card)
  - `txn_count_24h`, `time_since_last_sec`
  - Rationale: recent bursts and recency are predictive for fraud
- Categorical
  - `operation_type`, `merchant_name` (sanitized: lowercase, strip `fraud_` prefix, remove non-alphanumerics, collapse whitespace)
  - One-hot encoding with `min_frequency=10` to reduce high-cardinality noise
- Imputation
  - Numeric: median; Categorical: most-frequent

Notes
- Merchant text sanitization removes label-like leakage (`fraud_*` prefixes in ULB vendor names) and aligns closer to DGuard’s `merchant_clean_name`.
- The feature set mirrors our Isolation Forest “winning blocks”: amount_core + velocity + category-robust amount; novelty signals can be added later if helpful.

---

### Model and packages
- Classifier: `sklearn.ensemble.HistGradientBoostingClassifier`
  - Reasons: fast, robust tabular baseline in scikit-learn; native handling of dense inputs; easy to integrate in a pipeline; good latency
- Preprocessing: `sklearn` pipeline with `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`
- Metrics: `average_precision_score` (AP), precision@k (%): 0.5%, 1.0%, 5.0%
  - Rationale: imbalanced problem; AP and precision@budget align with fraud ops

---

### Hyperparameters (initial)
- `learning_rate=0.1`, `max_depth=8`, `max_iter=300`
  - Reasonable depth/trees for non-linearities without overfitting; to be tuned with time-aware CV
- `class_weight={0:1.0, 1:pos_weight}` where `pos_weight = N_neg / N_pos` on the training split
  - Rationale: cost-sensitive weighting to handle imbalance; alternative is threshold tuning only
- Categorical one-hot: `min_frequency=10`, `handle_unknown='ignore'`, `sparse_output=False`
  - Dense matrix required by HGBDT; frequency threshold reduces noise

Planned tuning
- Time-aware CV grid/Bayesian search over: `max_depth`, `max_iter`, `learning_rate`, `l2_regularization`, `min_samples_leaf`
- Feature ablations and target-encoding variants for high-cardinality categories (careful with leakage)

---

### Procedure
1) Load ULB and compute `event_time` from `unix_time` (UTC)
2) Engineer features (above)
3) Chronological train/test split (80/20)
4) Build `Pipeline(preprocess → HGBDT)`; fit on train
5) Score on test; compute AP and precision@k
6) Persist summary JSON: `reports/phase2/ulb_gbdt/ulb_gbdt_summary.json`

Initial results (from current run)
```
rows: 200000
train_rows: 160000, test_rows: 40000
fraud_rate_train: 0.00904, fraud_rate_test: 0.00495
AP: 0.8709
precision@0.5%: 0.785
precision@1.0%: 0.4575
precision@5.0%: 0.0980
```

Segmentation ablation (per `operation_type`)
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

Enhanced ULB with expanded history + stacking + per-category calibration
```
AP: 0.9231
precision@0.5%: 0.835
precision@1.0%: 0.475
precision@5.0%: 0.0985
CV folds:
  train<=60% → val 60–80%: AP=0.9386, P@0.5%=0.985, P@1%=0.7225, P@5%=0.1545
  train<=80% → val 80–100%: AP=0.9231, P@0.5%=0.835, P@1%=0.475, P@5%=0.0985
Calibrated per-category thresholds saved in `reports/phase2/ulb_gbdt/ulb_gbdt_enhanced_summary.json`.
```

ULB later-period holdout (fraudTest.csv)
```
AP: 0.8616
precision@0.5%: 0.6566
precision@1.0%: 0.3655
precision@5.0%: 0.0764
Notes: lower base rate (~0.386%) and temporal drift cause expected drop; per-category calibration remains effective.
```

Combined ULB + IEEE (with `dataset` indicator)
```
overall (test mostly IEEE):
  AP: 0.1458
  precision@0.5%: 0.338
  precision@1.0%: 0.305
  precision@5.0%: 0.2082
per-dataset (IEEE slice): same as overall above
Notes: different label prevalence and feature space; despite lower AP, top-k precision at 5% is strong. Next: enrich IEEE feature mapping and calibrate per-dataset thresholds.
```

---

### Why this stack
- GBDT is state-of-the-art for tabular fraud data and achieves strong top-k precision; it improves markedly over IF (unsupervised) given labels
- The combination of robust category-aware amount normalization and short-term velocities captures the most predictive fraud patterns, as our IF ablation showed
- The scikit-learn pipeline approach keeps train/serve skew low and makes explainability (e.g., SHAP) straightforward
- Segmentation by `operation_type` yields a small lift in top-of-queue precision (P@0.5% +0.01) at the cost of slightly lower AP and higher operational complexity. We can keep a single model as default and revisit segmentation after adding IEEE/domain signals.

---

### Feature Enhancement Results (September 2025)

**Candidate Feature Evaluation Completed**: Systematic evaluation of 7 candidate features from `modeling_feature_candidates.md` using time-aware CV methodology.

**Recommended Feature Additions (Retraining Required):**
1. **Calendar seasonality**: month_sin, month_cos encoding
   - Impact: +1.0pp P@0.5%, +0.5pp P@1.0%, +0.008 AP
   - Rationale: Captures longer-term temporal patterns beyond hour/dow
2. **Balance velocity extensions**: amount_rolling_std for variance detection  
   - Impact: +1.0pp P@0.5%, +1.5pp P@1.0%, +0.003 AP
   - Rationale: Transaction burstiness and variance provide fraud signal

**Combined Expected Improvement:**
- Baseline: P@0.5%=97.0%, P@1.0%=81.0%, AP=0.869
- Enhanced: P@0.5%=98.0%, P@1.0%=82.5%, AP=0.872+

**Features Evaluated but Not Recommended:**
- is_weekend, is_holiday: No additional signal beyond existing time features
- currency_normalized_amount: Requires real FX data for meaningful evaluation  
- merchant_share: Small positive signal but below threshold
- merchant_freq_global: Already implemented in current enhanced model

### Next steps
- **Implement recommended features**: Add seasonality and balance velocity features to enhanced pipeline
- Add IEEE dataset, include a `dataset`/domain indicator, and evaluate domain shift
- Calibrate thresholds to alert budgets (per-category and global); export configs
- Add SHAP global and local explanations; log top features per alert for analysts
- Perform time-aware hyperparameter tuning; add target-encoding experiments for high-cardinality categories if needed
- Shadow on DGuard, collect labels/feedback, and iterate


Artifacts
- Summary JSON: `reports/phase2/ulb_gbdt/ulb_gbdt_summary.json`
- Segmentation ablation: `reports/phase2/ulb_gbdt/ulb_gbdt_ablation.csv` and `reports/phase2/ulb_gbdt/ulb_gbdt_ablation.json`
- Enhanced summary: `reports/phase2/ulb_gbdt/ulb_gbdt_enhanced_summary.json`
- Time-aware CV: `reports/phase2/ulb_gbdt/ulb_gbdt_cv.json`
- Overall ablation: `reports/phase2/ulb_gbdt/ulb_gbdt_ablation_overall.json`
- ULB later holdout: `reports/phase2/ulb_gbdt/ulb_gbdt_enhanced_ulbtest_summary.json`
- Per-category thresholds (serving):
  - `reports/phase2/ulb_gbdt/gbdt_per_category_thresholds_0p005.json`
  - `reports/phase2/ulb_gbdt/gbdt_per_category_thresholds_0p01.json`
- Hyperparameter tuning (time-aware): `reports/phase2/ulb_gbdt/ulb_gbdt_tuning.json`
 - Combined ULB+IEEE: `reports/phase2/ulb_gbdt/ulb_ieee_gbdt_enhanced_summary.json`

Serving thresholds guidance
- Start with 0.5% per-category thresholds; cap per-category/day alerts; adjust gradually based on shadow results.
- Apply isotonic-calibrated scores for thresholding stability.

Best CV params (small grid)
```
max_depth=8, min_samples_leaf=20, learning_rate=0.1, l2_regularization=0.003
AP_mean (two folds): 0.9313
```

Serving config and shadow
- Config: `reports/phase2/ulb_gbdt/serving_config_v1.json`
- Shadow scorer: `src/fraud_mvp/shadow_score.py`
- Current outputs (ULB fraudTest @0.5%):
  - CSV: `reports/phase2/ulb_gbdt/shadow_alerts_ulb.csv`
  - JSON: `reports/phase2/ulb_gbdt/shadow_alerts_ulb.json`


