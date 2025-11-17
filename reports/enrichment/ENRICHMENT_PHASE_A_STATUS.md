# Enrichment Feature Integration - Phase A Status

**Date**: 2025-01-17
**Phase**: A - Testing without DGuard Ground Truth
**Status**: **SIMULATION INFRASTRUCTURE COMPLETE** ✓

---

## Executive Summary

Phase A infrastructure is **complete and validated**. All tools are ready to generate enriched datasets and train improved fraud detection models.

**Next Action Required**: Obtain ULB/IEEE CSV files to generate enriched training datasets.

---

## Completed Deliverables ✓

### 1. Enrichment Simulation Script ✓
**File**: [`scripts/simulate_enrichment.py`](scripts/simulate_enrichment.py)

**Capabilities**:
- Augments ULB/IEEE datasets with 20 enrichment features
- Matches production enrichment schema from fraud-scoring-service
- Handles multiple dataset formats (ULB, IEEE)
- Outputs enriched parquet files

**Features Simulated**:
- ✓ Merchant categories (category, subcategory, category_group)
- ✓ Merchant flags (is_chain, is_subscription, has_website, has_contact, has_business_hours)
- ✓ Location indicators (has_location, has_coordinates, location_country)
- ✓ Enrichment quality metrics (confidence_score, confidence_level, is_fully_enriched)
- ✓ Metadata (source_count, is_claude_enriched)
- Note: Refund flags (has_refund_history, refund_risk_flag) set to 0 (cannot simulate)

**Validation Status**: ✓ Tested with 1000 synthetic transactions
**Test Results**: All 20 features generated correctly with realistic distributions

### 2. Test Script ✓
**File**: [`scripts/test_simulate_enrichment.py`](scripts/test_simulate_enrichment.py)

**Capabilities**:
- Creates synthetic ULB-like data (1000 rows)
- Validates enrichment feature generation
- Outputs sample enriched parquet: `scripts/test_enriched_sample.parquet`

**Test Results**:
```
[OK] Created 1000 rows (0.60% fraud rate)
[OK] 100 unique merchants, 14 categories
[OK] All 20 enrichment features present
[OK] Feature distributions match expectations
[OK] Sample parquet: 86.7 KB
```

### 3. Methodology Documentation ✓
**File**: [`ENRICHMENT_SIMULATION_METHODOLOGY.md`](ENRICHMENT_SIMULATION_METHODOLOGY.md)

**Contents**:
- Complete simulation methodology (40+ pages)
- Category mapping rules (ULB: 14 categories, IEEE: 5 ProductCD)
- Feature-by-feature simulation logic
- Expected feature statistics
- Validation strategy
- Assumptions and limitations
- Phase B roadmap (when ground truth available)

### 4. Usage Documentation ✓
**File**: [`scripts/README.md`](scripts/README.md)

**Contents**:
- Step-by-step usage instructions
- Command examples for ULB/IEEE
- Output schema (29 columns total)
- Troubleshooting guide
- Next steps

---

## Infrastructure Ready ✓

### Environment
- ✓ Poetry dependencies installed (167 packages)
- ✓ Python environment configured
- ✓ Required libraries: pandas, numpy, pyarrow, scikit-learn, shap

### Code Artifacts
- ✓ Simulation script: `scripts/simulate_enrichment.py`
- ✓ Test script: `scripts/test_simulate_enrichment.py`
- ✓ Documentation: `ENRICHMENT_SIMULATION_METHODOLOGY.md`, `scripts/README.md`

### Validation
- ✓ Test execution successful (1000 synthetic rows)
- ✓ Feature generation validated
- ✓ Output format confirmed (parquet)

---

## Pending Tasks (Requires Data Access)

### ⏳ Task 1: Generate Enriched ULB Dataset

**Blocker**: Need access to ULB CSV files
**Expected Paths** (from existing code):
- Train: `/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv`
- Test: `/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTest.csv`

**Command (once available)**:
```bash
cd sapira_ai_exercise

# Train set
poetry run python scripts/simulate_enrichment.py \
  --dataset ulb_train \
  --data-path /path/to/fraudTrain.csv \
  --output-path data/unified_enriched/ulb_train_enriched.parquet \
  --limit 200000

# Test set
poetry run python scripts/simulate_enrichment.py \
  --dataset ulb_test \
  --data-path /path/to/fraudTest.csv \
  --output-path data/unified_enriched/ulb_test_enriched.parquet
```

**Expected Output**:
- `data/unified_enriched/ulb_train_enriched.parquet` (~200k rows, 29 columns)
- `data/unified_enriched/ulb_test_enriched.parquet` (test set size, 29 columns)

**Estimated Time**: 5-10 minutes per dataset

### ⏳ Task 2: Update Training Pipeline

**File to Modify**: [`src/fraud_mvp/gbdt_ulb_enhanced.py`](src/fraud_mvp/gbdt_ulb_enhanced.py)

**Required Changes**:

1. **Update data loading** (lines 37-70):
   ```python
   def load_ulb_enriched(limit: int) -> pd.DataFrame:
       """Load enriched ULB parquet with simulated enrichment features."""
       parquet_path = Path("data/unified_enriched/ulb_train_enriched.parquet")
       df = pd.read_parquet(parquet_path)

       if limit:
           df = df.head(limit)

       # Already has event_time and dataset columns from enrichment script
       return df
   ```

2. **Add enrichment features to feature engineering** (lines 112+):
   ```python
   def engineer_enhanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
       df = df.copy()

       # ... existing feature engineering ...

       # Add enrichment features (already in df from simulation)
       enrichment_numeric = [
           "has_category",
           "is_chain_merchant",
           "is_subscription_merchant",
           "merchant_has_website",
           "has_contact_info",
           "has_business_hours",
           "has_location",
           "has_coordinates",
           "enrichment_confidence_score",
           "is_fully_enriched",
           "enrichment_source_count",
           "is_claude_enriched",
           "has_refund_history",
           "refund_risk_flag",
       ]

       # Add categorical enrichment features
       enrichment_categorical = [
           "enrichment_confidence_level",  # high/medium/low/very_low
           "location_country",  # US/unknown/...
       ]

       # Update feature lists
       numeric_features += enrichment_numeric
       categorical_features += enrichment_categorical

       return df, numeric_features, categorical_features
   ```

**Estimated Time**: 2-3 hours

### ⏳ Task 3: Train Enriched Models

**Variants to Train**:

1. **Baseline** (control):
   - Current v2_enhanced features (31 features)
   - No enrichment
   - Purpose: Establish baseline performance

2. **Enriched-All**:
   - Baseline + all 14 numeric enrichment features
   - Purpose: Measure maximum enrichment impact

3. **Enriched-Selective**:
   - Baseline + top enrichment features only:
     - has_category
     - enrichment_confidence_score
     - is_chain_merchant
     - location_country
     - is_subscription_merchant
   - Purpose: Identify minimal effective feature set

**Training Command** (after pipeline update):
```bash
cd sapira_ai_exercise
poetry run python src/fraud_mvp/gbdt_ulb_enhanced.py
```

**Expected Outputs**:
- Model artifacts: `pipeline.pkl`, `isotonic.pkl`, `if_pipe.pkl`
- Threshold configs: `gbdt_per_category_thresholds_0.005.json`
- Performance metrics: P@0.5%, P@1.0%, AP

**Success Criteria**:
- P@0.5% lift ≥1 percentage point OR
- P@1.0% lift ≥2 percentage points

**Estimated Time**: 4-6 hours (training + evaluation)

### ⏳ Task 4: SHAP Analysis

**Purpose**: Measure enrichment feature importance

**Command**:
```bash
poetry run python src/fraud_mvp/gbdt_ulb_enhanced.py --shap-analysis
```

**Expected Outputs**:
- SHAP values for enrichment features
- Feature importance rankings
- Interaction effects analysis

**Key Questions**:
- Which enrichment features contribute most to fraud detection?
- Are enrichment features in top 15 by importance?
- Do enrichment features interact with existing features?

**Estimated Time**: 2-3 hours

### ⏳ Task 5: Deployment (if metrics improve)

**Actions**:
1. Package v3 model artifacts
2. Update fraud-scoring-service with new model
3. Deploy to AWS App Runner
4. Shadow scoring (run v2 and v3 in parallel)
5. Monitor production metrics

**Estimated Time**: 1-2 days

---

## Timeline Estimate

**Assuming ULB/IEEE data becomes available:**

| Task | Duration | Dependencies |
|------|----------|--------------|
| Generate enriched datasets | 0.5 day | Data access |
| Update training pipeline | 0.5 day | Enriched datasets |
| Train 3 model variants | 1 day | Updated pipeline |
| SHAP analysis | 0.5 day | Trained models |
| Deployment (if successful) | 1-2 days | Positive results |
| **Total** | **3-4 days** | |

---

## Alternative: Continue without Real Data

If ULB/IEEE data is not immediately available, consider:

### Option A: Use Existing Test Sample
- Work with `scripts/test_enriched_sample.parquet` (1000 rows)
- Train toy model to validate pipeline changes
- Limited evaluation, but validates code paths

### Option B: Generate Larger Synthetic Dataset
- Modify `test_simulate_enrichment.py` to generate 10k-50k synthetic rows
- Add more realistic fraud patterns
- Use for pipeline development and testing

### Option C: Wait for Phase B
- Skip Phase A entirely
- Wait for DGuard ground truth labels (Phase B)
- Train directly on real enrichment data

**Recommendation**: Proceed with Phase A if ULB/IEEE data available within 1-2 weeks. Otherwise, move to Phase B planning.

---

## Phase B Prerequisites

**Phase B** (training with DGuard ground truth) requires:

1. **Label Collection System**:
   - User dispute workflow
   - Chargeback data integration
   - Analyst review interface
   - Minimum 500-1000 labeled transactions

2. **DGuard Training Dataset**:
   - `bank_transactions` + `fraud_cases` joined
   - Real enrichment fields (not simulated)
   - Point-in-time correctness

3. **Active Learning Infrastructure**:
   - Continuous label collection
   - Monthly retraining pipeline
   - Intelligent sampling (high uncertainty, novel patterns)

See [ENRICHMENT_SIMULATION_METHODOLOGY.md](ENRICHMENT_SIMULATION_METHODOLOGY.md) Section "Phase B: Future (When Ground Truth Becomes Available)" for details.

---

## Key Files Reference

### Created in This Phase
- ✓ [`scripts/simulate_enrichment.py`](scripts/simulate_enrichment.py) - Main simulation script (450 lines)
- ✓ [`scripts/test_simulate_enrichment.py`](scripts/test_simulate_enrichment.py) - Test script (150 lines)
- ✓ [`ENRICHMENT_SIMULATION_METHODOLOGY.md`](ENRICHMENT_SIMULATION_METHODOLOGY.md) - Complete methodology (1000+ lines)
- ✓ [`scripts/README.md`](scripts/README.md) - Usage guide (400+ lines)
- ✓ [`scripts/test_enriched_sample.parquet`](scripts/test_enriched_sample.parquet) - Sample output (87 KB)
- ✓ [`ENRICHMENT_PHASE_A_STATUS.md`](ENRICHMENT_PHASE_A_STATUS.md) - This status document

### To Be Modified
- ⏳ [`src/fraud_mvp/gbdt_ulb_enhanced.py`](src/fraud_mvp/gbdt_ulb_enhanced.py) - Training pipeline (needs enrichment features)

### Reference Documentation
- [`fraud-scoring-service/docs/ENRICHMENT_PLAN.md`](../fraud-scoring-service/docs/ENRICHMENT_PLAN.md) - Production enrichment (40+ fields)
- [`fraud-scoring-service/src/app/services/features.py:194-467`](../fraud-scoring-service/src/app/services/features.py#L194-L467) - Production feature extraction
- [`data_consolidation_etl.md`](data_consolidation_etl.md) - Dataset schemas
- [`reports/model_status.md`](reports/model_status.md) - Current model (v2_enhanced)
- [`reports/feature_evaluation_executive_summary.md`](reports/feature_evaluation_executive_summary.md) - Feature evaluation methodology

---

## Success Metrics

### Infrastructure Completion (Current)
- [x] Simulation script created and validated
- [x] Test script passing
- [x] Documentation complete
- [x] Dependencies installed

### Model Training (Pending Data)
- [ ] ULB enriched datasets generated
- [ ] Training pipeline updated
- [ ] 3 model variants trained
- [ ] SHAP analysis completed

### Performance Targets (Phase A)
- [ ] P@0.5% improvement: ≥1pp (target: 78.5% → 79.5%+)
- [ ] P@1.0% improvement: ≥2pp
- [ ] Enrichment features in top 15 SHAP importance

### Deployment (Phase A)
- [ ] v3 model packaged
- [ ] Deployed to production
- [ ] Shadow scoring active
- [ ] Production monitoring stable

---

## Risks and Mitigation

### Risk 1: Data Access Delay
**Impact**: Cannot complete Phase A
**Mitigation**: Proceed with synthetic data or move to Phase B planning

### Risk 2: Simulation Doesn't Transfer
**Impact**: Enrichment features don't improve test metrics
**Mitigation**: Conservative simulation, document assumptions, Phase B will validate

### Risk 3: Overfitting to Simulation
**Impact**: Model learns artificial patterns
**Mitigation**: Random noise, realistic distributions, cross-validation, production monitoring

### Risk 4: Domain Shift (ULB → DGuard)
**Impact**: Improvements on ULB don't transfer to DGuard
**Mitigation**: Multi-dataset validation (IEEE), Phase B validation with real data

---

## Questions for Discussion

1. **Data Access**: When can ULB/IEEE CSV files be obtained? Alternative sources?

2. **Scope Decision**: Proceed with Phase A (simulated enrichment) or wait for Phase B (real labels)?

3. **Timeline**: Is 3-4 day completion realistic given other priorities?

4. **Phase B Planning**: Should we start label collection infrastructure now (parallel workstream)?

5. **Deployment Strategy**: If Phase A succeeds, deploy to production without DGuard validation?

---

## Contact and Support

**Repository**: `sapira_ai_exercise`
**Related Service**: `fraud-scoring-service` (production enrichment)
**Documentation**: All files listed in "Key Files Reference" above

For methodology questions, see [ENRICHMENT_SIMULATION_METHODOLOGY.md](ENRICHMENT_SIMULATION_METHODOLOGY.md)
For usage instructions, see [scripts/README.md](scripts/README.md)

---

**Status**: ✓ **SIMULATION INFRASTRUCTURE COMPLETE - READY FOR DATA**
