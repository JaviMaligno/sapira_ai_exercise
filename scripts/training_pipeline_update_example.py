"""
Example: How to Update Training Pipeline for Enriched Features

This file shows the required changes to `src/fraud_mvp/gbdt_ulb_enhanced.py`
to support enriched datasets with simulated enrichment features.

DO NOT RUN THIS FILE DIRECTLY - it's a reference/template.
"""

from pathlib import Path
import pandas as pd
from typing import List, Tuple
import numpy as np


# ============================================================================
# STEP 1: Update Data Loading Function
# ============================================================================
# Original function: load_ulb(limit: int) -> pd.DataFrame
# Location: src/fraud_mvp/gbdt_ulb_enhanced.py lines 37-52


def load_ulb_enriched(limit: int = None) -> pd.DataFrame:
    """
    Load enriched ULB parquet with simulated enrichment features.

    Replaces or augments the original load_ulb() function.

    Args:
        limit: Optional row limit for testing

    Returns:
        DataFrame with original columns + 20 enrichment columns
    """
    parquet_path = Path("data/unified_enriched/ulb_train_enriched.parquet")

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Enriched dataset not found: {parquet_path}\n"
            f"Generate it with: poetry run python scripts/simulate_enrichment.py"
        )

    # Load enriched parquet
    df = pd.read_parquet(parquet_path)

    # Apply limit if specified
    if limit:
        df = df.head(limit)

    # Enrichment script already added event_time and dataset columns
    # No need to reprocess

    return df


def load_ulb_enriched_test() -> pd.DataFrame:
    """Load enriched ULB test set."""
    parquet_path = Path("data/unified_enriched/ulb_test_enriched.parquet")

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Enriched test dataset not found: {parquet_path}\n"
            f"Generate it with: poetry run python scripts/simulate_enrichment.py"
        )

    df = pd.read_parquet(parquet_path)
    return df


# ============================================================================
# STEP 2: Update Feature Engineering Function
# ============================================================================
# Original function: engineer_enhanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]
# Location: src/fraud_mvp/gbdt_ulb_enhanced.py lines 112+


def engineer_enhanced_with_enrichment(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Feature engineering including enrichment features.

    Args:
        df: DataFrame with enrichment columns already present

    Returns:
        Tuple of (engineered_df, numeric_features, categorical_features)
    """
    df = df.copy()
    df.sort_values("event_time", inplace=True)

    # ========================================================================
    # EXISTING FEATURE ENGINEERING (from gbdt_ulb_enhanced.py)
    # ========================================================================
    # ... (keep all existing code) ...

    # Core amounts
    df["abs_amount"] = df["amount"].abs()
    df["log1p_abs_amount"] = np.log1p(df["abs_amount"].clip(lower=0))

    # Merchant sanitize (already done in enrichment, but keep for compatibility)
    if "merchant_clean_name" not in df.columns:
        df["merchant_clean_name"] = (
            df["merchant_name"]
            .astype(str)
            .str.lower()
            .str.replace(r"^fraud_", "", regex=True)
            .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # ... (include all existing feature engineering code) ...
    # - Category-robust amount features
    # - Time features (hour_sin, hour_cos, dow, is_night, month_sin, month_cos)
    # - Velocity features (txn_count_1h, txn_count_6h, txn_count_24h, amt_sum_24h)
    # - Merchant recurrence features
    # - Balance velocity features
    # - Isolation Forest anomaly score
    # - Rule-based score
    # ... (see original gbdt_ulb_enhanced.py for complete code) ...

    # Existing numeric features (31 features from v2_enhanced)
    numeric_features = [
        "amount",
        "abs_amount",
        "log1p_abs_amount",
        "amount_z_iqr_cat",
        "hour",
        "hour_sin",
        "hour_cos",
        "dow",
        "is_night",
        "month_sin",
        "month_cos",
        "txn_count_1h",
        "txn_count_6h",
        "txn_count_24h",
        "amt_sum_24h",
        "time_since_last_sec",
        "is_new_merchant_for_card",
        "unique_merchants_7d",
        "unique_merchants_30d",
        "prop_new_merchants_7d",
        "prop_new_merchants_30d",
        "merchant_count_7d",
        "merchant_count_30d",
        "days_since_last_merchant",
        "merchant_freq_global",
        "amount_rolling_std_24h",
        "amount_rolling_mean_24h",
        "if_anomaly_score",
        "rule_score",
    ]

    # Existing categorical features (2 features)
    categorical_features = [
        "operation_type",  # Original transaction category
        "dataset",  # Domain indicator (ULB, IEEE, etc.)
    ]

    # ========================================================================
    # NEW: ADD ENRICHMENT FEATURES
    # ========================================================================

    # Enrichment features are already in df from simulate_enrichment.py
    # Just add them to feature lists

    # Numeric enrichment features (14 features)
    enrichment_numeric = [
        "has_category",  # Binary: category availability
        "is_chain_merchant",  # Binary: chain store indicator
        "is_subscription_merchant",  # Binary: subscription service
        "merchant_has_website",  # Binary: website presence
        "has_contact_info",  # Binary: contact availability
        "has_business_hours",  # Binary: business hours presence
        "has_location",  # Binary: location data availability
        "has_coordinates",  # Binary: valid coordinates
        "enrichment_confidence_score",  # Float: 0.0-1.0
        "is_fully_enriched",  # Binary: composite quality flag
        "enrichment_source_count",  # Int: 1-3 sources
        "is_claude_enriched",  # Binary: Claude enrichment indicator
        "has_refund_history",  # Binary: always 0 (not simulated)
        "refund_risk_flag",  # Binary: always 0 (not simulated)
    ]

    # Categorical enrichment features (2 features)
    enrichment_categorical = [
        "enrichment_confidence_level",  # high/medium/low/very_low
        "location_country",  # US/unknown/...
    ]

    # Optional: Use enriched category instead of operation_type
    # enrichment_categorical += ["category", "subcategory", "category_group"]

    # Validate enrichment features exist
    missing_numeric = [f for f in enrichment_numeric if f not in df.columns]
    missing_categorical = [f for f in enrichment_categorical if f not in df.columns]

    if missing_numeric or missing_categorical:
        raise ValueError(
            f"Missing enrichment features!\n"
            f"Numeric: {missing_numeric}\n"
            f"Categorical: {missing_categorical}\n"
            f"Did you load an enriched dataset?"
        )

    # Add enrichment features to feature lists
    numeric_features += enrichment_numeric
    categorical_features += enrichment_categorical

    # ========================================================================
    # TOTAL FEATURES
    # ========================================================================
    # Baseline (v2_enhanced): 31 numeric + 2 categorical = 33 features
    # With Enrichment: 45 numeric + 4 categorical = 49 features
    # (Or 31+14=45 numeric, 2+2=4 categorical)

    print(f"Total features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    print(f"  - Baseline features: 31 numeric, 2 categorical")
    print(f"  - Enrichment features: {len(enrichment_numeric)} numeric, {len(enrichment_categorical)} categorical")

    return df, numeric_features, categorical_features


# ============================================================================
# STEP 3: Update Main Training Script
# ============================================================================


def train_enriched_model_example():
    """
    Example of how to train a model with enrichment features.

    This replaces the main() function in gbdt_ulb_enhanced.py
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    # Load enriched data
    print("Loading enriched dataset...")
    df = load_ulb_enriched(limit=200_000)  # Or None for full dataset
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Feature engineering
    print("Engineering features...")
    df_eng, numeric_features, categorical_features = engineer_enhanced_with_enrichment(
        df
    )

    # Split train/val (time-based)
    # ... (use existing time-based CV logic) ...

    # Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant", fill_value=0), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    # Train
    X = df_eng[numeric_features + categorical_features]
    y = df_eng["is_fraud"]

    print("Training model with enrichment features...")
    pipeline.fit(X, y)

    # Evaluate
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    # ... (use existing evaluation logic: P@0.5%, P@1.0%, AP) ...

    print("Model training complete!")
    print(
        f"Total features used: {len(numeric_features) + len(categorical_features)}"
    )

    return pipeline


# ============================================================================
# STEP 4: Model Comparison Strategy
# ============================================================================


def train_model_variants():
    """
    Train 3 model variants for comparison:
    1. Baseline (no enrichment)
    2. Enriched-All (all enrichment features)
    3. Enriched-Selective (top enrichment features only)
    """
    from sklearn.metrics import average_precision_score

    # Load data
    df = load_ulb_enriched(limit=200_000)
    df_eng, numeric_features, categorical_features = engineer_enhanced_with_enrichment(
        df
    )

    # Define feature sets
    baseline_numeric = numeric_features[:31]  # First 31 = original features
    baseline_categorical = categorical_features[:2]  # First 2 = original categories

    enrichment_numeric = numeric_features[31:]  # Last 14 = enrichment
    enrichment_categorical = categorical_features[2:]  # Last 2 = enrichment categories

    # Top enrichment features (based on expected importance)
    top_enrichment_numeric = [
        "has_category",
        "enrichment_confidence_score",
        "is_chain_merchant",
        "is_subscription_merchant",
        "is_fully_enriched",
    ]
    top_enrichment_categorical = ["location_country"]

    variants = {
        "baseline": {
            "numeric": baseline_numeric,
            "categorical": baseline_categorical,
        },
        "enriched_all": {
            "numeric": baseline_numeric + enrichment_numeric,
            "categorical": baseline_categorical + enrichment_categorical,
        },
        "enriched_selective": {
            "numeric": baseline_numeric + top_enrichment_numeric,
            "categorical": baseline_categorical + top_enrichment_categorical,
        },
    }

    results = {}

    for variant_name, features in variants.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {variant_name}")
        print(f"  Numeric: {len(features['numeric'])} features")
        print(f"  Categorical: {len(features['categorical'])} features")
        print(f"{'=' * 60}")

        # ... (train model with these features) ...
        # ... (evaluate P@0.5%, P@1.0%, AP) ...

        # Store results
        results[variant_name] = {
            "num_features": len(features["numeric"]) + len(features["categorical"]),
            # ... (add performance metrics) ...
        }

    # Compare results
    print(f"\n{'=' * 60}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 60}")
    for variant_name, metrics in results.items():
        print(f"{variant_name}:")
        print(f"  Features: {metrics['num_features']}")
        # ... (print performance metrics) ...

    return results


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
To integrate enrichment features into your training pipeline:

1. Generate enriched datasets:
   ```
   poetry run python scripts/simulate_enrichment.py \
     --dataset ulb_train \
     --data-path /path/to/fraudTrain.csv \
     --output-path data/unified_enriched/ulb_train_enriched.parquet \
     --limit 200000
   ```

2. Update src/fraud_mvp/gbdt_ulb_enhanced.py:
   - Replace load_ulb() with load_ulb_enriched()
   - Update engineer_enhanced() to add enrichment features (see example above)
   - Ensure numeric_features includes 14 enrichment numeric features
   - Ensure categorical_features includes 2 enrichment categorical features

3. Train baseline vs enriched models:
   - Run train_model_variants() to compare 3 variants
   - Measure P@0.5%, P@1.0%, AP for each
   - Identify which enrichment features contribute most

4. Run SHAP analysis:
   - Measure feature importance for enriched model
   - Check if enrichment features are in top 15
   - Document interaction effects

5. If metrics improve:
   - Package v3 model artifacts
   - Deploy to fraud-scoring-service
   - Monitor production performance

Expected improvement: P@0.5% lift ≥1pp OR P@1.0% lift ≥2pp
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nThis is a reference file. Do not run directly.")
    print("Copy the functions above into src/fraud_mvp/gbdt_ulb_enhanced.py")
