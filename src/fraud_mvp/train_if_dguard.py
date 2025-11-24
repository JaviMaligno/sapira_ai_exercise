"""
Train Isolation Forest on DGuard MongoDB data with enrichment features.

This script:
1. Extracts transactions from DGuard MongoDB
2. Engineers features (time, amount, velocity, enrichment)
3. Trains Isolation Forest to learn normal patterns
4. Evaluates if enrichment features help detect anomalies
5. Saves model for deployment
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from datetime import datetime
import joblib


@dataclass
class Config:
    mongo_uri: str = "mongodb://mongo:uGiyrTQJDXyusAZNqlzBOHRdaWxGrSGJ@junction.proxy.rlwy.net:15000/dguard_transactions?authSource=admin"
    mongo_db: str = "dguard_transactions"
    collection: str = "bank_transactions"
    out_dir: Path = Path(__file__).parent.parent.parent / "reports" / "phase2" / "if_dguard"
    random_state: int = 42
    contamination: float = 0.01  # Expected anomaly rate


def extract_dguard_data(cfg: Config) -> pd.DataFrame:
    """Extract transactions from DGuard MongoDB."""
    print("Connecting to MongoDB...")
    client = MongoClient(cfg.mongo_uri)
    db = client[cfg.mongo_db]
    collection = db[cfg.collection]

    # Query all enriched transactions
    cursor = collection.find(
        {"merchant_clean_name": {"$ne": None}},  # Has enrichment
        {
            "_id": 1,
            "user_id": 1,
            "account_id": 1,
            "amount": 1,
            "description": 1,
            "transaction_date": 1,
            "merchant_clean_name": 1,
            "category": 1,
            "enrichment_confidence": 1,
            "enrichment_sources": 1,
            "enrichment_strategy": 1,
            "merchant_website": 1,
            "merchant_chain": 1,
            "merchant_location": 1,
        }
    )

    records = list(cursor)
    client.close()

    if not records:
        raise ValueError("No enriched transactions found in MongoDB")

    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} transactions from MongoDB")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for IF training."""
    print("Engineering features...")

    # Parse transaction date
    df["trans_date_trans_time"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values(["user_id", "trans_date_trans_time"]).reset_index(drop=True)

    # Amount features
    df["abs_amount"] = df["amount"].abs()
    df["log1p_abs_amount"] = np.log1p(df["abs_amount"])

    # Time features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["dow"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month"] = df["trans_date_trans_time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Velocity features per user
    df["time_since_last_sec"] = df.groupby("user_id")["trans_date_trans_time"].diff().dt.total_seconds().fillna(86400)

    # Transaction counts (simplified - would need window functions for full implementation)
    df["txn_count_user"] = df.groupby("user_id").cumcount() + 1

    # Amount statistics per category
    cat_stats = df.groupby("category")["abs_amount"].agg(["mean", "std"]).fillna(1)
    df = df.merge(cat_stats, on="category", how="left", suffixes=("", "_cat"))
    df["amount_z_cat"] = (df["abs_amount"] - df["mean"]) / df["std"].replace(0, 1)

    # Enrichment features
    df["enrichment_confidence_score"] = df["enrichment_confidence"].fillna(0.0)
    df["has_category"] = df["category"].notna().astype(int)
    df["has_website"] = df["merchant_website"].notna().astype(int)
    df["is_chain"] = df["merchant_chain"].fillna(False).astype(int)
    df["has_location"] = df["merchant_location"].notna().astype(int)
    df["enrichment_source_count"] = df["enrichment_sources"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Enrichment quality score
    df["enrichment_quality"] = (
        df["has_category"] * 0.3 +
        df["has_website"] * 0.2 +
        df["has_location"] * 0.2 +
        df["enrichment_confidence_score"] * 0.3
    )

    print(f"Engineered {len(df.columns)} features")
    return df


def train_if_model(df: pd.DataFrame, cfg: Config) -> tuple:
    """Train Isolation Forest on DGuard data."""
    print("Training Isolation Forest...")

    # Features for IF
    numeric_features = [
        # Amount features
        "abs_amount",
        "log1p_abs_amount",
        "amount_z_cat",
        # Time features
        "hour",
        "dow",
        "is_night",
        "hour_sin",
        "hour_cos",
        # Velocity
        "time_since_last_sec",
        "txn_count_user",
        # Enrichment features
        "enrichment_confidence_score",
        "has_category",
        "has_website",
        "is_chain",
        "has_location",
        "enrichment_source_count",
        "enrichment_quality",
    ]

    # Filter to available features
    available = [f for f in numeric_features if f in df.columns]
    print(f"Using {len(available)} features: {available}")

    X = df[available].copy()

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), available)
        ]
    )

    # IF model
    if_model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_jobs=-1
    )

    # Full pipeline
    if_pipe = Pipeline([
        ("prep", preprocessor),
        ("if", if_model)
    ])

    # Train on all data (unsupervised)
    if_pipe.fit(X)

    # Get anomaly scores
    scores = -if_pipe.named_steps["if"].decision_function(
        if_pipe.named_steps["prep"].transform(X)
    )
    df["if_score"] = scores

    print(f"IF scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")

    return if_pipe, df, available


def evaluate_enrichment_impact(df: pd.DataFrame, cfg: Config) -> dict:
    """Evaluate if enrichment features correlate with anomaly scores."""
    print("\nEvaluating enrichment impact...")

    results = {}

    # Correlation between IF score and enrichment features
    enrichment_cols = [
        "enrichment_confidence_score",
        "enrichment_quality",
        "has_category",
        "has_website",
        "is_chain",
        "has_location",
        "enrichment_source_count",
    ]

    correlations = {}
    for col in enrichment_cols:
        if col in df.columns:
            corr = df["if_score"].corr(df[col])
            correlations[col] = float(corr)
            print(f"  {col}: correlation = {corr:.3f}")

    results["correlations"] = correlations

    # Compare IF scores by enrichment quality
    df["enrichment_bin"] = pd.cut(
        df["enrichment_quality"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["low", "medium", "high"]
    )

    quality_analysis = df.groupby("enrichment_bin")["if_score"].agg(["mean", "std", "count"]).to_dict()
    results["by_enrichment_quality"] = quality_analysis

    print("\nIF scores by enrichment quality:")
    for quality in ["low", "medium", "high"]:
        if quality in df["enrichment_bin"].values:
            subset = df[df["enrichment_bin"] == quality]
            print(f"  {quality}: mean={subset['if_score'].mean():.3f}, n={len(subset)}")

    # Top anomalies analysis
    top_n = min(20, len(df))
    top_anomalies = df.nlargest(top_n, "if_score")

    results["top_anomalies"] = {
        "avg_enrichment_confidence": float(top_anomalies["enrichment_confidence_score"].mean()),
        "avg_enrichment_quality": float(top_anomalies["enrichment_quality"].mean()),
        "pct_low_confidence": float((top_anomalies["enrichment_confidence_score"] < 0.5).mean()),
        "categories": top_anomalies["category"].value_counts().to_dict(),
    }

    print(f"\nTop {top_n} anomalies:")
    print(f"  Avg enrichment confidence: {results['top_anomalies']['avg_enrichment_confidence']:.2f}")
    print(f"  % with low confidence: {results['top_anomalies']['pct_low_confidence']*100:.1f}%")

    return results


def main():
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Isolation Forest on DGuard Data")
    print("=" * 60)

    # Extract data
    df = extract_dguard_data(cfg)

    # Engineer features
    df = engineer_features(df)

    # Train IF
    if_pipe, df, features = train_if_model(df, cfg)

    # Evaluate enrichment impact
    eval_results = evaluate_enrichment_impact(df, cfg)

    # Save artifacts
    print("\nSaving artifacts...")

    # Save IF pipeline
    if_path = cfg.out_dir / "if_pipe_dguard.pkl"
    joblib.dump(if_pipe, if_path)
    print(f"Saved: {if_path}")

    # Save evaluation results
    eval_path = cfg.out_dir / "if_dguard_evaluation.json"
    eval_path.write_text(json.dumps(eval_results, indent=2, default=str))
    print(f"Saved: {eval_path}")

    # Save feature list
    features_path = cfg.out_dir / "if_dguard_features.json"
    features_path.write_text(json.dumps(features, indent=2))
    print(f"Saved: {features_path}")

    # Save scored transactions for review
    top_anomalies = df.nlargest(50, "if_score")[
        ["_id", "description", "amount", "category", "if_score",
         "enrichment_confidence_score", "enrichment_quality"]
    ]
    anomalies_path = cfg.out_dir / "top_anomalies.csv"
    top_anomalies.to_csv(anomalies_path, index=False)
    print(f"Saved: {anomalies_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Summary
    print(f"\nSummary:")
    print(f"  Transactions: {len(df)}")
    print(f"  Features: {len(features)}")
    print(f"  Output: {cfg.out_dir}")

    return eval_results


if __name__ == "__main__":
    main()
