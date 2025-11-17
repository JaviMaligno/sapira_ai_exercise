"""
Simulation of Enrichment Features for Test Datasets

This script augments test datasets (ULB, IEEE) with simulated enrichment features
that match the production enrichment schema from fraud-scoring-service.

Purpose:
- Test enrichment feature impact on labeled test datasets
- Validate feature engineering before DGuard ground truth is available
- Create training data with enrichment-like signals

Simulated Features:
1. Merchant category mapping (category, subcategory, category_group)
2. Merchant flags (is_chain, merchant_has_website, has_contact_info)
3. Location flags (has_location, has_coordinates, location_country)
4. Enrichment quality indicators (confidence scores, completeness flags)

Methodology:
- Conservative simulation (avoid strong artificial signals)
- Based on data completeness and merchant frequency
- Mock values follow realistic distributions

Output:
- {dataset}_enriched.parquet files with 14 additional enrichment features
- Compatible with training pipeline feature extraction
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ULB Category → Enrichment Mapping
# Based on ULB's 14 transaction categories
CATEGORY_MAPPINGS = {
    # Travel & Transport
    "gas_transport": {
        "category": "Transportation",
        "subcategory": "Gas Station",
        "category_group": "Travel & Transport",
    },
    # Food & Dining
    "grocery_pos": {
        "category": "Food & Dining",
        "subcategory": "Grocery Store",
        "category_group": "Food & Dining",
    },
    "grocery_net": {
        "category": "Food & Dining",
        "subcategory": "Online Grocery",
        "category_group": "Food & Dining",
    },
    "food_dining": {
        "category": "Food & Dining",
        "subcategory": "Restaurant",
        "category_group": "Food & Dining",
    },
    # Shopping
    "shopping_pos": {
        "category": "Shopping",
        "subcategory": "Retail Store",
        "category_group": "Shopping",
    },
    "shopping_net": {
        "category": "Shopping",
        "subcategory": "Online Shopping",
        "category_group": "Shopping",
    },
    "home": {
        "category": "Shopping",
        "subcategory": "Home & Garden",
        "category_group": "Shopping",
    },
    "kids_pets": {
        "category": "Shopping",
        "subcategory": "Kids & Pets",
        "category_group": "Shopping",
    },
    # Entertainment
    "entertainment": {
        "category": "Entertainment",
        "subcategory": "Entertainment",
        "category_group": "Entertainment",
    },
    # Health & Wellness
    "personal_care": {
        "category": "Health & Wellness",
        "subcategory": "Personal Care",
        "category_group": "Health & Wellness",
    },
    "health_fitness": {
        "category": "Health & Wellness",
        "subcategory": "Health & Fitness",
        "category_group": "Health & Wellness",
    },
    # Services
    "misc_pos": {
        "category": "Services",
        "subcategory": "Miscellaneous",
        "category_group": "Services",
    },
    "misc_net": {
        "category": "Services",
        "subcategory": "Online Services",
        "category_group": "Services",
    },
    # Travel
    "travel": {
        "category": "Travel",
        "subcategory": "Travel Services",
        "category_group": "Travel & Transport",
    },
}

# IEEE ProductCD → Category Mapping (simplified)
IEEE_CATEGORY_MAPPINGS = {
    "W": {
        "category": "Services",
        "subcategory": "Digital Services",
        "category_group": "Services",
    },
    "H": {
        "category": "Home Services",
        "subcategory": "Home Services",
        "category_group": "Services",
    },
    "C": {
        "category": "Technology",
        "subcategory": "Consumer Electronics",
        "category_group": "Shopping",
    },
    "S": {
        "category": "Services",
        "subcategory": "Professional Services",
        "category_group": "Services",
    },
    "R": {
        "category": "Recreation",
        "subcategory": "Recreation",
        "category_group": "Entertainment",
    },
}

# Subscription-likely categories
SUBSCRIPTION_CATEGORIES = {
    "Services",
    "Digital Services",
    "Online Services",
    "Entertainment",
    "Online Shopping",
}


def load_ulb(data_path: Path, limit: int = None) -> pd.DataFrame:
    """Load ULB dataset with minimal columns needed for enrichment simulation."""
    usecols = [
        "trans_num",
        "unix_time",
        "category",
        "amt",
        "merchant",
        "merch_lat",
        "merch_long",
        "is_fraud",
        "cc_num",
    ]
    df = pd.read_csv(data_path, usecols=usecols, nrows=limit)

    # Standardize column names
    df.rename(
        columns={
            "trans_num": "transaction_id",
            "unix_time": "event_time_ts",
            "category": "operation_type",
            "amt": "amount",
            "merchant": "merchant_name",
        },
        inplace=True,
    )
    df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)
    df["dataset"] = "ULB"

    return df


def load_ieee(data_path: Path, limit: int = None) -> pd.DataFrame:
    """Load IEEE dataset with minimal columns needed for enrichment simulation."""
    usecols = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card4",
        "P_emaildomain",
        "isFraud",
    ]
    df = pd.read_csv(data_path, usecols=usecols, nrows=limit)

    # Standardize
    df.rename(
        columns={
            "TransactionID": "transaction_id",
            "TransactionDT": "event_time_ts",
            "TransactionAmt": "amount",
            "ProductCD": "operation_type",
            "isFraud": "is_fraud",
        },
        inplace=True,
    )

    # Build event_time from relative seconds
    base = pd.Timestamp("2017-12-01", tz="UTC")
    df["event_time"] = base + pd.to_timedelta(df["event_time_ts"], unit="s")

    # Merchant proxy from email domain or card type
    merch = df["P_emaildomain"].fillna(df["card4"].astype(str)).fillna("UNK").astype(str)
    df["merchant_name"] = merch
    df["cc_num"] = df["card1"].fillna(-1).astype(int)
    df["dataset"] = "IEEE"

    # No lat/lon in IEEE
    df["merch_lat"] = np.nan
    df["merch_long"] = np.nan

    return df[
        [
            "transaction_id",
            "event_time_ts",
            "event_time",
            "operation_type",
            "amount",
            "merchant_name",
            "merch_lat",
            "merch_long",
            "is_fraud",
            "cc_num",
            "dataset",
        ]
    ]


def simulate_enrichment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate enrichment features based on existing data.

    Adds 14 enrichment-derived features that match fraud-scoring-service schema:
    - Merchant category fields (3)
    - Merchant flags (4)
    - Location flags (3)
    - Enrichment quality indicators (4)

    Args:
        df: DataFrame with columns: transaction_id, event_time, operation_type,
            amount, merchant_name, merch_lat, merch_long, is_fraud, cc_num, dataset

    Returns:
        DataFrame with original columns plus 14 simulated enrichment features
    """
    df = df.copy()

    # === 1. MERCHANT CATEGORY MAPPING ===

    # Determine category mappings based on dataset
    def map_category(row):
        dataset = row["dataset"]
        op_type = row["operation_type"]

        if dataset == "ULB":
            mapping = CATEGORY_MAPPINGS.get(str(op_type), {})
        elif dataset == "IEEE":
            mapping = IEEE_CATEGORY_MAPPINGS.get(str(op_type), {})
        else:
            mapping = {}

        return pd.Series({
            "category": mapping.get("category", None),
            "subcategory": mapping.get("subcategory", None),
            "category_group": mapping.get("category_group", None),
        })

    category_df = df.apply(map_category, axis=1)
    df["category"] = category_df["category"]
    df["subcategory"] = category_df["subcategory"]
    df["category_group"] = category_df["category_group"]

    # Derived: has_category
    df["has_category"] = df["category"].notna().astype(int)

    # === 2. MERCHANT CLEAN NAME ===

    df["merchant_clean_name"] = (
        df["merchant_name"]
        .astype(str)
        .str.lower()
        .str.replace(r"^fraud_", "", regex=True)
        .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # === 3. MERCHANT FLAGS (Based on Frequency) ===

    # Calculate merchant frequency (global)
    merchant_counts = df.groupby("merchant_clean_name").size()
    df["merchant_frequency"] = df["merchant_clean_name"].map(merchant_counts)

    # is_chain: Top 10% most frequent merchants
    chain_threshold = df["merchant_frequency"].quantile(0.90)
    df["is_chain_merchant"] = (df["merchant_frequency"] >= chain_threshold).astype(int)

    # is_subscription: Based on category
    df["is_subscription_merchant"] = (
        df["subcategory"].isin(SUBSCRIPTION_CATEGORIES).astype(int)
    )

    # merchant_has_website: 70% for frequent merchants, 40% for others
    np.random.seed(42)  # For reproducibility
    website_prob = np.where(
        df["is_chain_merchant"] == 1,
        0.70,  # Chain merchants more likely to have website
        0.40,  # Others less likely
    )
    df["merchant_has_website"] = (np.random.random(len(df)) < website_prob).astype(int)

    # has_contact_info: 50% for chains, 30% for non-chains
    contact_prob = np.where(df["is_chain_merchant"] == 1, 0.50, 0.30)
    df["has_contact_info"] = (np.random.random(len(df)) < contact_prob).astype(int)

    # has_business_hours: 40% overall
    df["has_business_hours"] = (np.random.random(len(df)) < 0.40).astype(int)

    # === 4. LOCATION FLAGS ===

    df["has_location"] = df["merch_lat"].notna().astype(int)
    df["has_coordinates"] = (
        df["merch_lat"].notna() & df["merch_long"].notna()
    ).astype(int)

    # location_country: Default to "US" for ULB (domestic), "unknown" for IEEE
    df["location_country"] = np.where(df["dataset"] == "ULB", "US", "unknown")

    # === 5. ENRICHMENT QUALITY INDICATORS ===

    # Count how many enrichment fields are populated
    enrichment_fields = [
        "has_category",
        "has_location",
        "merchant_has_website",
        "has_contact_info",
    ]
    df["enrichment_field_count"] = df[enrichment_fields].sum(axis=1)

    # enrichment_confidence_score: Based on completeness
    # Range: 0.5-0.95 (conservative, no perfect confidence)
    base_confidence = 0.50
    confidence_per_field = 0.10  # +0.10 for each populated field
    df["enrichment_confidence_score"] = (
        base_confidence + (df["enrichment_field_count"] * confidence_per_field)
    ).clip(upper=0.95)

    # Add small random noise to avoid deterministic patterns
    noise = np.random.uniform(-0.05, 0.05, len(df))
    df["enrichment_confidence_score"] = (
        df["enrichment_confidence_score"] + noise
    ).clip(0.0, 0.95)

    # enrichment_confidence_level: Categorical binning
    def confidence_level(score):
        if pd.isna(score):
            return "very_low"
        elif score >= 0.75:
            return "high"
        elif score >= 0.60:
            return "medium"
        elif score >= 0.45:
            return "low"
        else:
            return "very_low"

    df["enrichment_confidence_level"] = df["enrichment_confidence_score"].apply(
        confidence_level
    )

    # is_fully_enriched: Has 4+ enrichment fields
    df["is_fully_enriched"] = (df["enrichment_field_count"] >= 4).astype(int)

    # enrichment_source_count: Random 1-3 sources
    df["enrichment_source_count"] = np.random.choice([1, 2, 3], size=len(df), p=[0.3, 0.5, 0.2])

    # is_claude_enriched: 30% probability
    df["is_claude_enriched"] = (np.random.random(len(df)) < 0.30).astype(int)

    # === 6. REFUND/RISK FLAGS (Cannot simulate - leave as 0) ===
    # These require historical refund data not available in test datasets
    df["has_refund_history"] = 0
    df["refund_risk_flag"] = 0

    # Drop temporary columns
    df.drop(columns=["merchant_frequency", "enrichment_field_count"], inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Simulate enrichment features for test datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["ulb_train", "ulb_test", "ieee_train"],
        required=True,
        help="Dataset to enrich",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to output enriched parquet file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to process (for testing)",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} from {args.data_path}")

    # Load dataset
    if args.dataset.startswith("ulb"):
        df = load_ulb(args.data_path, limit=args.limit)
    elif args.dataset.startswith("ieee"):
        df = load_ieee(args.data_path, limit=args.limit)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Loaded {len(df):,} rows")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

    # Simulate enrichment features
    print("Simulating enrichment features...")
    df_enriched = simulate_enrichment_features(df)

    # Validate enrichment features
    enrichment_cols = [
        "category",
        "subcategory",
        "category_group",
        "has_category",
        "merchant_clean_name",
        "is_chain_merchant",
        "is_subscription_merchant",
        "merchant_has_website",
        "has_contact_info",
        "has_business_hours",
        "has_location",
        "has_coordinates",
        "location_country",
        "enrichment_confidence_score",
        "enrichment_confidence_level",
        "is_fully_enriched",
        "enrichment_source_count",
        "is_claude_enriched",
        "has_refund_history",
        "refund_risk_flag",
    ]

    print(f"\nEnrichment feature statistics:")
    for col in enrichment_cols[:10]:  # Print first 10
        if col in df_enriched.columns:
            if df_enriched[col].dtype in ["int64", "float64"]:
                print(f"  {col}: mean={df_enriched[col].mean():.3f}")
            else:
                print(f"  {col}: {df_enriched[col].nunique()} unique values")

    # Save enriched dataset
    print(f"\nSaving enriched dataset to {args.output_path}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_parquet(args.output_path, index=False, engine="pyarrow")

    print(f"✓ Enriched dataset saved: {len(df_enriched):,} rows, {len(df_enriched.columns)} columns")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  Enrichment columns added: {len(df_enriched.columns) - len(df.columns)}")


if __name__ == "__main__":
    main()
