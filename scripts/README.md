# Enrichment Simulation Scripts

This directory contains scripts for simulating enrichment features on test datasets to evaluate their impact on fraud detection models.

## Scripts

### `simulate_enrichment.py`
Main script that augments test datasets (ULB, IEEE) with simulated enrichment features matching the production schema.

**Features Added**: 20 enrichment features including:
- Merchant categories (category, subcategory, category_group, has_category)
- Merchant flags (is_chain, is_subscription, has_website, has_contact, has_business_hours)
- Location indicators (has_location, has_coordinates, location_country)
- Enrichment quality metrics (confidence_score, confidence_level, is_fully_enriched, source_count, is_claude_enriched)
- Refund flags (has_refund_history, refund_risk_flag - set to 0, cannot simulate)

### `test_simulate_enrichment.py`
Validation script that tests the enrichment simulation with synthetic data.

## Usage

### Run Test (Validate Simulation Logic)

```bash
cd sapira_ai_exercise
poetry run python scripts/test_simulate_enrichment.py
```

**Expected Output**:
- Creates 1000 synthetic ULB-like transactions
- Applies enrichment simulation
- Validates all 20 features are present
- Outputs feature statistics
- Saves sample: `scripts/test_enriched_sample.parquet`

### Generate Enriched ULB Dataset

**Requirements**: Access to ULB CSV files

```bash
poetry run python scripts/simulate_enrichment.py \
  --dataset ulb_train \
  --data-path /path/to/fraudTrain.csv \
  --output-path data/unified_enriched/ulb_train_enriched.parquet \
  --limit 200000
```

**Parameters**:
- `--dataset`: Dataset type (ulb_train, ulb_test, ieee_train)
- `--data-path`: Path to input CSV file
- `--output-path`: Path for output enriched parquet
- `--limit`: (Optional) Limit rows for testing

### Generate Enriched IEEE Dataset

```bash
poetry run python scripts/simulate_enrichment.py \
  --dataset ieee_train \
  --data-path /path/to/train_transaction.csv \
  --output-path data/unified_enriched/ieee_train_enriched.parquet \
  --limit 200000
```

## Output Schema

**Original columns** (from ULB/IEEE):
- transaction_id
- event_time_ts, event_time
- operation_type (category)
- amount
- merchant_name
- merch_lat, merch_long (ULB only)
- is_fraud (label)
- cc_num
- dataset

**Enrichment columns added** (20 total):

| Column | Type | Description |
|--------|------|-------------|
| category | str | Primary merchant category |
| subcategory | str | Merchant subcategory |
| category_group | str | Category grouping |
| has_category | int | Category availability flag (0/1) |
| merchant_clean_name | str | Normalized merchant name |
| is_chain_merchant | int | Chain store indicator (0/1) |
| is_subscription_merchant | int | Subscription service flag (0/1) |
| merchant_has_website | int | Website presence indicator (0/1) |
| has_contact_info | int | Phone or email availability (0/1) |
| has_business_hours | int | Business hours presence (0/1) |
| has_location | int | Location data availability (0/1) |
| has_coordinates | int | Valid coordinates present (0/1) |
| location_country | str | Country code ("US" for ULB, "unknown" for IEEE) |
| enrichment_confidence_score | float | Confidence level (0.0-1.0) |
| enrichment_confidence_level | str | Categorical confidence (high/medium/low/very_low) |
| is_fully_enriched | int | Composite quality flag (0/1) |
| enrichment_source_count | int | Number of enrichment sources (1-3) |
| is_claude_enriched | int | Claude AI enrichment indicator (0/1) |
| has_refund_history | int | Historical refund flag (always 0 - not simulated) |
| refund_risk_flag | int | Combined refund risk (always 0 - not simulated) |

**Total columns**: 29 (9 original + 20 enrichment)

## Simulation Logic

### Category Mapping

**ULB (14 native categories)** → Enriched categories:
- `gas_transport` → Transportation / Gas Station / Travel & Transport
- `grocery_pos`, `grocery_net`, `food_dining` → Food & Dining / ...
- `shopping_pos`, `shopping_net`, `home`, `kids_pets` → Shopping / ...
- `entertainment` → Entertainment / ...
- `personal_care`, `health_fitness` → Health & Wellness / ...
- `misc_pos`, `misc_net` → Services / ...
- `travel` → Travel / Travel Services / ...

**IEEE (5 ProductCD)** → Enriched categories:
- `W` → Services / Digital Services
- `H` → Home Services / Home Services
- `C` → Technology / Consumer Electronics
- `S` → Services / Professional Services
- `R` → Recreation / Recreation

### Merchant Flags

- **is_chain_merchant**: Top 10% most frequent merchants (by count)
- **is_subscription_merchant**: Category-based (Services, Digital Services, Entertainment, Online Services)
- **merchant_has_website**: Random with probability:
  - 70% for chain merchants
  - 40% for non-chain merchants
- **has_contact_info**: Random with probability:
  - 50% for chain merchants
  - 30% for non-chain merchants
- **has_business_hours**: Random 40% probability

### Location Flags

- **ULB**: Has merch_lat/merch_long → `has_location=1`, `has_coordinates=1`, `location_country="US"`
- **IEEE**: No location data → `has_location=0`, `has_coordinates=0`, `location_country="unknown"`

### Enrichment Quality

- **confidence_score**: `0.50 + (0.10 × enrichment_field_count) + noise[-0.05, +0.05]`, clipped to [0.0, 0.95]
- **confidence_level**: Binned from score:
  - ≥0.75 → "high"
  - ≥0.60 → "medium"
  - ≥0.45 → "low"
  - <0.45 → "very_low"
- **is_fully_enriched**: 1 if ≥4 enrichment fields populated, else 0
- **source_count**: Random choice [1, 2, 3] with probabilities [30%, 50%, 20%]
- **is_claude_enriched**: Random 30% probability

### Refund Flags

- **has_refund_history**: Always 0 (cannot simulate without historical data)
- **refund_risk_flag**: Always 0 (cannot simulate without historical data)

## Expected Feature Statistics (ULB)

Based on ULB characteristics:

| Feature | Expected Value | Rationale |
|---------|---------------|-----------|
| has_category | 100% | All ULB transactions have category |
| is_chain_merchant | ~10-13% | Top 10% by frequency |
| is_subscription_merchant | ~10-20% | Based on category distribution |
| merchant_has_website | ~50-55% | Weighted avg: 0.70×0.10 + 0.40×0.90 |
| has_contact_info | ~35-40% | Weighted avg: 0.50×0.10 + 0.30×0.90 |
| has_business_hours | 40% | Fixed probability |
| has_location | 100% | ULB has merch lat/lon |
| has_coordinates | 100% | ULB has merch lat/lon |
| location_country | 100% "US" | ULB default |
| enrichment_confidence_score | 0.75-0.85 | High completeness in ULB |
| is_fully_enriched | ~10-20% | Depends on random flags |
| enrichment_source_count | ~1.9 | Mean of distribution |
| is_claude_enriched | 30% | Fixed probability |

## Validation

After generating enriched datasets, validate:

1. **Row count**: Same as input
2. **Column count**: Original + 20 enrichment columns
3. **No nulls in critical fields**: enrichment_confidence_score, has_category
4. **Feature distributions match expected values** (see table above)
5. **Fraud label unchanged**: Check `is_fraud` mean before/after

## Next Steps

After generating enriched datasets:

1. **Update Training Pipeline**: Modify `src/fraud_mvp/gbdt_ulb_enhanced.py` to load enriched parquet
2. **Feature Engineering**: Extract enrichment features in `engineer_enhanced()` function
3. **Model Training**: Train baseline vs enriched models (3 variants)
4. **Evaluation**: Compare P@0.5%, P@1.0%, AP metrics
5. **SHAP Analysis**: Measure enrichment feature importance
6. **Deployment**: Package v3 model artifacts

## Documentation

- [ENRICHMENT_SIMULATION_METHODOLOGY.md](../ENRICHMENT_SIMULATION_METHODOLOGY.md) - Complete methodology and assumptions
- [fraud-scoring-service ENRICHMENT_PLAN.md](../../fraud-scoring-service/docs/ENRICHMENT_PLAN.md) - Production enrichment details
- [data_consolidation_etl.md](../data_consolidation_etl.md) - Dataset schemas

## Limitations

### Simulation Limitations

1. **Not real enrichment**: Mock values don't reflect actual enrichment API behavior
2. **No domain expertise**: Real enrichment uses business knowledge, we use rules
3. **Deterministic patterns**: Models may learn artificial simulation patterns
4. **No edge cases**: Real enrichment has API failures, timeouts - not simulated
5. **Refund history unavailable**: Cannot simulate historical refund signals

### Validation Limitations

1. **Domain shift**: ULB/IEEE fraud patterns may differ from DGuard
2. **No production feedback**: Cannot validate on real DGuard without labels
3. **Selection bias**: Test datasets may not represent production distributions

See [ENRICHMENT_SIMULATION_METHODOLOGY.md](../ENRICHMENT_SIMULATION_METHODOLOGY.md) Section 4 for detailed assumptions and mitigation strategies.

## Troubleshooting

### "No module named 'pandas'"
```bash
poetry install
```

### "FileNotFoundError: fraudTrain.csv"
Update `--data-path` to correct ULB CSV location. If you don't have ULB data, use the test script instead:
```bash
poetry run python scripts/test_simulate_enrichment.py
```

### "UnicodeEncodeError" in test script
Fixed in latest version. If you still see this, ensure you're using the updated `test_simulate_enrichment.py` with `[OK]` instead of Unicode characters.

## Contact

For questions or issues, refer to:
- [ENRICHMENT_SIMULATION_METHODOLOGY.md](../ENRICHMENT_SIMULATION_METHODOLOGY.md)
- Model training repository: `sapira_ai_exercise`
- Production service: `fraud-scoring-service`
