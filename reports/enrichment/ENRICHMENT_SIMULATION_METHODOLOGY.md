# Enrichment Feature Simulation Methodology

**Version**: 1.0
**Date**: 2025-01-17
**Status**: Phase A - Testing without DGuard Ground Truth

## Executive Summary

This document describes the methodology for simulating enrichment features on test datasets (ULB, IEEE, BAF, PaySim) to evaluate their impact on fraud detection models **before** DGuard ground truth labels are available.

### Key Points

- **Problem**: Enriched data exists in production DGuard, but we have NO ground truth labels
- **Solution**: Simulate enrichment features on labeled test datasets (ULB, IEEE) to measure impact
- **Goal**: Train improved models with enrichment features, validate on test sets, deploy to production
- **Limitation**: Cannot validate performance on real DGuard data without labels (Phase B requirement)

---

## 1. Context and Motivation

### Current State

The fraud-scoring-service successfully integrated enrichment capabilities:
- 40+ enriched fields available in production (merchant info, location, categories, contact, refunds)
- 14 derived ML features created in [features.py:194-467](../fraud-scoring-service/src/app/services/features.py#L194-L467)
- Enrichment deployed to AWS App Runner and processing production traffic

**However**: The current fraud detection model (v2_enhanced) was trained on ULB dataset without any enrichment features. The enrichment data flows through the API but is not used for scoring.

### The Gap

We need to answer: **Do enrichment features improve fraud detection?**

Without DGuard ground truth, we cannot:
- Measure fraud detection performance on real production data
- Validate that enrichment signals correlate with actual fraud
- Run A/B tests comparing baseline vs enriched models

### The Approach

**Phase A (Current)**: Simulate enrichment on labeled datasets
- Augment ULB/IEEE with mock enrichment features
- Train models with enriched features
- Measure performance improvements on test sets
- Deploy enriched model to production (unvalidated)

**Phase B (Future - when labels exist)**: Validate on DGuard
- Collect ground truth labels (chargebacks, user disputes, analyst reviews)
- Retrain model with real DGuard enrichment data
- Validate performance on production traffic
- Implement active learning loop

---

## 2. Enrichment Features to Simulate

### 2.1 Feature Categories

Based on [fraud-scoring-service enrichment schema](../fraud-scoring-service/docs/ENRICHMENT_PLAN.md):

#### A. Merchant Category Features (3 features)
- `category` - Primary merchant category
- `subcategory` - Merchant subcategory
- `category_group` - Category grouping
- `has_category` - Binary flag for category availability

**Simulation Strategy**:
- ULB: Direct mapping from native `category` field (14 categories)
- IEEE: Mapping from `ProductCD` (W/H/C/S/R → categories)
- Others: Rule-based inference or null

#### B. Merchant Flags (6 features)
- `is_chain_merchant` - Chain store indicator
- `is_subscription_merchant` - Subscription service flag
- `merchant_has_website` - Website presence indicator
- `has_contact_info` - Phone or email availability
- `has_business_hours` - Business hours presence

**Simulation Strategy**:
- `is_chain`: Top 10% most frequent merchants
- `is_subscription`: Category-based (streaming, software, online services)
- `merchant_has_website`: 70% for chains, 40% for non-chains (random)
- `has_contact_info`: 50% for chains, 30% for non-chains (random)
- `has_business_hours`: 40% overall (random)

#### C. Location Features (3 features)
- `has_location` - Location data availability
- `has_coordinates` - Valid coordinates present
- `location_country` - Country code

**Simulation Strategy**:
- ULB/Sparkov: Have `merch_lat`, `merch_long` → set flags to True
- IEEE: No location data → False
- `location_country`: "US" for ULB (domestic), "unknown" for IEEE

#### D. Enrichment Quality Indicators (4 features)
- `enrichment_confidence_score` - Confidence level (0.0-1.0)
- `enrichment_confidence_level` - Categorical confidence (high/medium/low/very_low)
- `is_fully_enriched` - Composite quality flag
- `enrichment_source_count` - Number of enrichment sources
- `is_claude_enriched` - Claude AI enrichment indicator

**Simulation Strategy**:
- `confidence_score`: 0.50 + (0.10 × enrichment_field_count) + noise [-0.05, +0.05]
- `confidence_level`: Binned from score (≥0.75=high, ≥0.60=medium, ≥0.45=low, else=very_low)
- `is_fully_enriched`: 1 if 4+ enrichment fields populated
- `source_count`: Random 1-3 with distribution [30%, 50%, 20%]
- `is_claude_enriched`: 30% probability

#### E. Refund/Risk Features (2 features - NOT SIMULATED)
- `has_refund_history` - Historical refund flag
- `refund_risk_flag` - Combined refund risk

**Simulation Strategy**: Set to 0 (cannot simulate without historical refund data)

### 2.2 Feature Mapping by Dataset

| Feature | ULB | IEEE | BAF | PaySim | Notes |
|---------|-----|------|-----|--------|-------|
| category | ✓ Native | ~ Mapped | ~ Inferred | ~ Inferred | ULB has 14 native categories |
| has_category | ✓ | ✓ | ✓ | ✓ | Derived from category |
| is_chain_merchant | ✓ Freq | ✓ Freq | ✗ | ✗ | Based on merchant frequency |
| is_subscription | ✓ Cat | ✓ Cat | ~ | ~ | Category-based inference |
| merchant_has_website | ✓ Mock | ✓ Mock | ✗ | ✗ | Random with probability |
| has_contact_info | ✓ Mock | ✓ Mock | ~ | ✗ | Random with probability |
| has_location | ✓ Native | ✗ | ✗ | ✗ | ULB has merch lat/lon |
| has_coordinates | ✓ Native | ✗ | ✗ | ✗ | ULB has merch lat/lon |
| location_country | ✓ US | ~ Unknown | ~ | ✗ | Default values |
| enrichment_confidence_score | ✓ | ✓ | ✓ | ✓ | Completeness-based |
| enrichment_confidence_level | ✓ | ✓ | ✓ | ✓ | Binned from score |
| is_fully_enriched | ✓ | ✓ | ✓ | ✓ | Derived from field count |
| enrichment_source_count | ✓ | ✓ | ✓ | ✓ | Random 1-3 |
| is_claude_enriched | ✓ | ✓ | ✓ | ✓ | 30% probability |

✓ = Good simulation quality
~ = Partial/approximate simulation
✗ = Not applicable

---

## 3. Category Mapping Rules

### 3.1 ULB Categories (14 categories)

| ULB Category | Enriched Category | Subcategory | Category Group |
|--------------|-------------------|-------------|----------------|
| gas_transport | Transportation | Gas Station | Travel & Transport |
| grocery_pos | Food & Dining | Grocery Store | Food & Dining |
| grocery_net | Food & Dining | Online Grocery | Food & Dining |
| food_dining | Food & Dining | Restaurant | Food & Dining |
| shopping_pos | Shopping | Retail Store | Shopping |
| shopping_net | Shopping | Online Shopping | Shopping |
| home | Shopping | Home & Garden | Shopping |
| kids_pets | Shopping | Kids & Pets | Shopping |
| entertainment | Entertainment | Entertainment | Entertainment |
| personal_care | Health & Wellness | Personal Care | Health & Wellness |
| health_fitness | Health & Wellness | Health & Fitness | Health & Wellness |
| misc_pos | Services | Miscellaneous | Services |
| misc_net | Services | Online Services | Services |
| travel | Travel | Travel Services | Travel & Transport |

### 3.2 IEEE ProductCD Mapping (5 codes)

| IEEE ProductCD | Enriched Category | Subcategory | Category Group |
|----------------|-------------------|-------------|----------------|
| W | Services | Digital Services | Services |
| H | Home Services | Home Services | Services |
| C | Technology | Consumer Electronics | Shopping |
| S | Services | Professional Services | Services |
| R | Recreation | Recreation | Entertainment |

### 3.3 Subscription Categories

Categories likely to be subscription services:
- Services
- Digital Services
- Online Services
- Entertainment
- Online Shopping
- Software (if added)
- Streaming (if added)

---

## 4. Simulation Assumptions and Limitations

### 4.1 Assumptions

1. **Frequency as proxy for chain status**: Merchants appearing in top 10% frequency are likely chains
2. **Category inference**: Transaction categories can proxy for merchant categories
3. **Geographic defaults**: ULB is US-based (default country="US")
4. **Random flags follow realistic distributions**: Website/contact probabilities based on domain knowledge
5. **Confidence correlates with completeness**: More enrichment fields → higher confidence
6. **No temporal effects**: Enrichment quality is static (real enrichment may have time-dependent quality)

### 4.2 Limitations

#### Simulation Limitations

1. **Not real enrichment**: Mock values don't reflect actual enrichment API behavior
2. **No domain expertise**: Real enrichment uses business knowledge, we use rules
3. **Deterministic patterns**: Models may learn artificial simulation patterns instead of fraud signals
4. **No edge cases**: Real enrichment has API failures, timeouts, partial data - not simulated
5. **Refund history unavailable**: Cannot simulate `has_refund_history` or `refund_risk_flag`

#### Validation Limitations

1. **Domain shift**: ULB/IEEE fraud patterns may differ from DGuard
2. **No production feedback**: Cannot validate if enrichment features work on real DGuard fraud
3. **Selection bias**: Test datasets may not represent production transaction distributions
4. **Label quality**: Test labels are clean; real labels (chargebacks) are noisy and delayed

### 4.3 Risk Mitigation

**To minimize overfitting to simulation:**

1. ✓ Conservative simulation (avoid strong artificial signals)
2. ✓ Add random noise to prevent deterministic patterns
3. ✓ Use realistic probability distributions (not 100% or 0%)
4. ✓ Focus on simple, interpretable features
5. ✓ Validate on multiple datasets (ULB + IEEE cross-validation)
6. ✓ Monitor production score distributions for unexpected changes
7. ✓ Document all assumptions clearly
8. ✓ Plan for rollback if production behavior is anomalous

---

## 5. Implementation

### 5.1 Script: `scripts/simulate_enrichment.py`

**Purpose**: Augment test datasets with simulated enrichment features

**Usage**:
```bash
# ULB train
poetry run python scripts/simulate_enrichment.py \
  --dataset ulb_train \
  --data-path /path/to/fraudTrain.csv \
  --output-path data/unified_enriched/ulb_enriched.parquet \
  --limit 200000

# IEEE train
poetry run python scripts/simulate_enrichment.py \
  --dataset ieee_train \
  --data-path /path/to/train_transaction.csv \
  --output-path data/unified_enriched/ieee_enriched.parquet \
  --limit 200000
```

**Output**: Parquet files with original columns + 20 enrichment columns

### 5.2 Test Script: `scripts/test_simulate_enrichment.py`

**Purpose**: Validate enrichment simulation with synthetic data

**Usage**:
```bash
poetry run python scripts/test_simulate_enrichment.py
```

**Output**:
- Validation statistics
- Sample enriched dataset: `scripts/test_enriched_sample.parquet`

---

## 6. Expected Feature Statistics

Based on simulation logic, expected feature distributions on ULB:

| Feature | Expected Value | Rationale |
|---------|---------------|-----------|
| has_category | 100% | All ULB transactions have category |
| is_chain_merchant | 10% | By definition (top 10% frequency) |
| is_subscription_merchant | ~10-15% | Based on category distribution |
| merchant_has_website | ~50-55% | Weighted avg: 0.70×0.10 + 0.40×0.90 |
| has_contact_info | ~35-40% | Weighted avg: 0.50×0.10 + 0.30×0.90 |
| has_business_hours | 40% | Fixed probability |
| has_location | 100% | ULB has merch lat/lon |
| has_coordinates | 100% | ULB has merch lat/lon |
| location_country | 100% "US" | ULB default |
| enrichment_confidence_score | 0.75-0.85 | High completeness in ULB |
| is_fully_enriched | ~70-80% | Most transactions have 4+ fields |
| enrichment_source_count | ~2.0 | Mean of [1,2,3] with weights [0.3,0.5,0.2] = 1.9 |
| is_claude_enriched | 30% | Fixed probability |
| has_refund_history | 0% | Not simulated |
| refund_risk_flag | 0% | Not simulated |

---

## 7. Validation Strategy

### 7.1 Pre-Training Validation

**Before training enriched models:**

1. ✓ Run `test_simulate_enrichment.py` to validate simulation logic
2. ✓ Verify feature distributions match expected values (Section 6)
3. ✓ Check for data leakage (enrichment should not correlate perfectly with fraud label)
4. ✓ Validate no null values in critical features (e.g., `enrichment_confidence_score`)
5. ✓ Ensure backward compatibility (non-enriched columns unchanged)

### 7.2 Model Training Validation

**During model training:**

1. ✓ Time-based cross-validation (maintain temporal order)
2. ✓ Compare baseline (no enrichment) vs enriched models
3. ✓ Measure lift in Precision@0.5%, Precision@1.0%
4. ✓ SHAP analysis: enrichment feature importance
5. ✓ Ablation study: remove enrichment features individually
6. ✓ Per-category performance (ensure no category degradation)

**Success Criteria**:
- P@0.5% lift ≥1 percentage point OR
- P@1.0% lift ≥2 percentage points AND
- No category shows >5pp precision drop

### 7.3 Post-Deployment Monitoring

**After deploying enriched model to production:**

1. ✓ Shadow scoring: run baseline and enriched models in parallel
2. ✓ Compare score distributions (mean, std, quantiles)
3. ✓ Monitor alert rate stability (should remain ~0.5-1%)
4. ✓ Track enrichment feature availability (% transactions with enrichment data)
5. ✓ Latency monitoring (must stay <150ms p99)
6. ✓ Drift detection: feature distributions over time

**Alert Conditions**:
- Score distribution shift >20%
- Alert rate change >10% relative
- Enrichment availability drop >15%
- Latency p99 >150ms

---

## 8. Next Steps

### Phase A Roadmap

**Week 1:**
- [x] Create simulation script
- [x] Create test script
- [ ] Validate simulation on synthetic data
- [ ] Generate ULB enriched dataset

**Week 2:**
- [ ] Update training pipeline to load enriched parquet
- [ ] Implement enrichment feature extraction (match features.py logic)
- [ ] Train baseline vs enriched models (3 variants)
- [ ] SHAP analysis and ablation study

**Week 3:**
- [ ] Package model artifacts (v3_enriched)
- [ ] Deploy to fraud-scoring-service
- [ ] Shadow deployment (1-5% traffic)
- [ ] Monitor production metrics

### Phase B Triggers

**When ground truth becomes available:**

1. Collect 500-1000 DGuard labeled transactions (fraud/legitimate)
2. Re-evaluate enriched model on real DGuard data
3. Compare simulated vs real enrichment feature performance
4. Retrain model with DGuard labels + real enrichment
5. Implement active learning loop

---

## 9. References

### Documentation
- [fraud-scoring-service ENRICHMENT_PLAN.md](../fraud-scoring-service/docs/ENRICHMENT_PLAN.md) - Enrichment integration details
- [data_consolidation_etl.md](./data_consolidation_etl.md) - Dataset schemas and mappings
- [reports/feature_evaluation_executive_summary.md](./reports/feature_evaluation_executive_summary.md) - Feature evaluation methodology

### Code
- [fraud-scoring-service features.py:194-467](../fraud-scoring-service/src/app/services/features.py#L194-L467) - Production enrichment feature extraction
- [scripts/simulate_enrichment.py](./scripts/simulate_enrichment.py) - Enrichment simulation implementation
- [src/fraud_mvp/gbdt_ulb_enhanced.py](./src/fraud_mvp/gbdt_ulb_enhanced.py) - Current production model (baseline)

### Datasets
- [reports/ulb_train_profile.md](./reports/ulb_train_profile.md) - ULB schema (200k rows, 0.82% fraud)
- [reports/ieee_transaction_profile.md](./reports/ieee_transaction_profile.md) - IEEE schema (200k rows, 3.01% fraud)

---

## 10. Changelog

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-17 | Claude Code | Initial methodology document |

---

## Appendix A: Full Feature List

### Original Transaction Features (from ULB/IEEE)
- transaction_id
- event_time_ts, event_time
- operation_type
- amount
- merchant_name
- merch_lat, merch_long (ULB only)
- is_fraud (label)
- cc_num
- dataset

### Simulated Enrichment Features (20 total)

**Merchant Category (4)**:
1. category
2. subcategory
3. category_group
4. has_category

**Merchant Signals (6)**:
5. merchant_clean_name
6. is_chain_merchant
7. is_subscription_merchant
8. merchant_has_website
9. has_contact_info
10. has_business_hours

**Location (3)**:
11. has_location
12. has_coordinates
13. location_country

**Enrichment Quality (5)**:
14. enrichment_confidence_score
15. enrichment_confidence_level
16. is_fully_enriched
17. enrichment_source_count
18. is_claude_enriched

**Refund/Risk (2 - not simulated)**:
19. has_refund_history
20. refund_risk_flag

**Total enriched columns**: 9 (original) + 20 (enrichment) = 29 columns
