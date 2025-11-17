"""
Test script for simulate_enrichment.py with synthetic data

Creates a small synthetic ULB-like dataset to validate the enrichment simulation logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simulate_enrichment import simulate_enrichment_features, CATEGORY_MAPPINGS


def create_synthetic_ulb_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create synthetic ULB-like data for testing."""
    np.random.seed(42)

    # ULB categories
    categories = list(CATEGORY_MAPPINGS.keys())

    # Generate synthetic data
    data = {
        "transaction_id": [f"tx_{i:06d}" for i in range(n_rows)],
        "event_time_ts": np.random.randint(1325376000, 1334332800, n_rows),  # 2012 Q1
        "operation_type": np.random.choice(categories, n_rows),
        "amount": np.random.lognormal(3.5, 1.5, n_rows).clip(1, 10000),
        "merchant_name": np.random.choice(
            [f"merchant_{i}" for i in range(100)], n_rows
        ),  # 100 merchants
        "merch_lat": np.random.uniform(25, 49, n_rows),  # US latitudes
        "merch_long": np.random.uniform(-125, -65, n_rows),  # US longitudes
        "is_fraud": np.random.choice([0, 1], n_rows, p=[0.99, 0.01]),  # 1% fraud
        "cc_num": np.random.randint(1000000, 9999999, n_rows),
        "dataset": "ULB",
    }

    df = pd.DataFrame(data)
    df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)

    return df


def test_enrichment_simulation():
    """Test enrichment feature simulation."""
    print("=" * 60)
    print("Testing Enrichment Simulation")
    print("=" * 60)

    # Create synthetic data
    print("\n1. Creating synthetic ULB data (1000 rows)...")
    df = create_synthetic_ulb_data(1000)
    print(f"   [OK] Created {len(df)} rows")
    print(f"   [OK] Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"   [OK] Unique merchants: {df['merchant_name'].nunique()}")
    print(f"   [OK] Categories: {df['operation_type'].nunique()}")

    # Apply enrichment simulation
    print("\n2. Simulating enrichment features...")
    df_enriched = simulate_enrichment_features(df)
    print(f"   [OK] Enriched {len(df_enriched)} rows")
    print(f"   [OK] Added {len(df_enriched.columns) - len(df.columns)} columns")

    # Validate enrichment features
    print("\n3. Validating enrichment features:")

    enrichment_features = [
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

    missing = [col for col in enrichment_features if col not in df_enriched.columns]
    if missing:
        print(f"   [FAIL] Missing columns: {missing}")
        return False

    print(f"   [OK] All {len(enrichment_features)} enrichment features present")

    # Statistics
    print("\n4. Enrichment feature statistics:")
    print(f"   • has_category: {df_enriched['has_category'].mean():.1%}")
    print(f"   • is_chain_merchant: {df_enriched['is_chain_merchant'].mean():.1%}")
    print(
        f"   • is_subscription_merchant: {df_enriched['is_subscription_merchant'].mean():.1%}"
    )
    print(
        f"   • merchant_has_website: {df_enriched['merchant_has_website'].mean():.1%}"
    )
    print(f"   • has_contact_info: {df_enriched['has_contact_info'].mean():.1%}")
    print(f"   • has_location: {df_enriched['has_location'].mean():.1%}")
    print(f"   • has_coordinates: {df_enriched['has_coordinates'].mean():.1%}")
    print(
        f"   • enrichment_confidence_score: {df_enriched['enrichment_confidence_score'].mean():.3f} ± {df_enriched['enrichment_confidence_score'].std():.3f}"
    )
    print(f"   • is_fully_enriched: {df_enriched['is_fully_enriched'].mean():.1%}")
    print(
        f"   • enrichment_source_count: {df_enriched['enrichment_source_count'].mean():.2f}"
    )

    # Category distribution
    print("\n5. Category distribution:")
    category_counts = df_enriched["category"].value_counts()
    for cat, count in category_counts.head(5).items():
        print(f"   • {cat}: {count} ({count/len(df_enriched)*100:.1f}%)")

    # Confidence level distribution
    print("\n6. Confidence level distribution:")
    conf_counts = df_enriched["enrichment_confidence_level"].value_counts()
    for level, count in conf_counts.items():
        print(f"   • {level}: {count} ({count/len(df_enriched)*100:.1f}%)")

    # Save sample output
    output_path = Path(__file__).parent / "test_enriched_sample.parquet"
    df_enriched.to_parquet(output_path, index=False)
    print(f"\n7. Sample output saved to: {output_path}")
    print(f"   [OK] Size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_enrichment_simulation()
    sys.exit(0 if success else 1)
