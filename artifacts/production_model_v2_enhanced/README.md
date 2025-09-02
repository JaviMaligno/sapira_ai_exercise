# Production Model v2 Enhanced - Deployment Artifacts

**Version**: v2_enhanced  
**Implementation Date**: September 2, 2025  
**Performance**: AP=0.9068, P@0.5%=100%, P@1.0%=88%

## Essential Artifacts for Production Deployment

### Core Model Files (Required)
1. **`pipeline.pkl`** - Main GBDT model pipeline with enhanced features
   - HistGradientBoostingClassifier with 31 features (29 numeric + 2 categorical)
   - Includes new seasonality and balance velocity features
   
2. **`isotonic.pkl`** - Isotonic calibrator for probability calibration
   - Required for converting model scores to calibrated probabilities

3. **`if_pipe.pkl`** - Isolation Forest pipeline for anomaly scoring
   - Used as stacked feature in enhanced model

### Threshold Configuration (Required)
4. **`enhanced_thresholds_0p005.json`** - Per-category thresholds for 0.5% alert rate
5. **`enhanced_thresholds_0p01.json`** - Per-category thresholds for 1.0% alert rate

### Feature Engineering Support (Required)
6. **`merchant_freq_map.json`** - Global merchant frequency mapping
   - Required for merchant_freq_global feature computation
   
7. **`rule_params.json`** - Rule-based scoring parameters
   - Defines thresholds for high-amount and anomaly rules

### Model Implementation (Required)
8. **`gbdt_ulb_enhanced.py`** - Production model implementation
   - Contains feature engineering logic with enhanced features
   - Implements seasonality (month_sin, month_cos) and balance velocity features

### Configuration (Required)
9. **`serving_config_v2_enhanced.json`** - Serving configuration specification
   - Documents model hyperparameters, features, and deployment requirements

## New Features Implemented

### Seasonality Features
- **month_sin**: `sin(2π × month / 12)` - Calendar seasonality patterns
- **month_cos**: `cos(2π × month / 12)` - Calendar seasonality patterns

### Balance Velocity Features  
- **amount_rolling_std_24h**: Standard deviation of amounts in 24h window per card
- **amount_rolling_mean_24h**: Mean of amounts in 24h window per card

## Deployment Requirements

### Infrastructure
- **Python Environment**: Requires scikit-learn, numpy, pandas
- **Memory**: Model artifacts ~50MB total
- **Compute**: Minimal latency impact - features derived from existing fields

### Data Dependencies
- **No additional data sources required**
- **No ETL modifications needed**
- Features computed from existing `event_time` and `amount` fields

### Backward Compatibility
- ✅ Compatible with existing feature engineering pipeline
- ✅ No breaking changes to serving interface
- ✅ Maintains existing fraud detection workflow

## Performance Validation

- **Cross-validation**: AP=0.9345 (average across time-aware folds)
- **Final test**: AP=0.9068, P@0.5%=100%, P@1.0%=88%
- **Improvement**: +3.5pp AP, +3.0pp P@0.5%, +7.0pp P@1.0% vs baseline
- **Status**: ✅ Exceeds evaluation expectations

## Usage Instructions

1. **Load artifacts** in serving environment
2. **Deploy enhanced model code** (`gbdt_ulb_enhanced.py`)
3. **Configure thresholds** based on desired alert rate (0.5% or 1.0%)
4. **Apply merchant frequency mapping** for feature engineering
5. **Validate** with test transactions before production rollout

## Validation Checklist

- [x] All artifacts present and loadable
- [x] Feature computation verified
- [x] Performance metrics validated  
- [x] Threshold calibration confirmed
- [x] Backward compatibility tested
- [x] No additional data dependencies