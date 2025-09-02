"""
Test script to validate the enhanced features work correctly
"""
import sys
sys.path.append('src')

from fraud_mvp.gbdt_ulb_enhanced import engineer_enhanced, load_ulb, build_pipeline, compute_scores
from fraud_mvp.gbdt_ulb_enhanced import precision_at_k, fit_if_and_rule_scores
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
from pathlib import Path

def test_enhanced_features():
    print("Testing enhanced features with seasonality and balance velocity...")
    
    # Load smaller sample for testing
    df = load_ulb(50_000)  # Smaller sample for faster testing
    print(f"Loaded {len(df)} rows")
    
    # Engineer features
    df, num_cols, cat_cols = engineer_enhanced(df)
    print(f"Feature engineering complete. Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    
    # Check new features are present
    required_features = ["month_sin", "month_cos", "amount_rolling_std_24h", "amount_rolling_mean_24h"]
    missing_features = [f for f in required_features if f not in num_cols]
    if missing_features:
        print(f"ERROR: Missing features: {missing_features}")
        return False
    else:
        print("✅ All required new features are present in the feature list")
    
    # Check data has the features computed
    for feature in required_features:
        if feature not in df.columns:
            print(f"ERROR: Feature {feature} not computed in dataframe")
            return False
        else:
            # Check for reasonable values
            vals = df[feature].dropna()
            if len(vals) == 0:
                print(f"WARNING: Feature {feature} has no non-null values")
            else:
                print(f"✅ Feature {feature}: mean={vals.mean():.4f}, std={vals.std():.4f}, range=[{vals.min():.4f}, {vals.max():.4f}]")
    
    # Quick train/test split and model test
    df_sorted = df.sort_values("event_time")
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Split data: {len(train_df)} train, {len(test_df)} test")
    
    # Merchant frequency mapping
    freq_map = train_df["merchant_name"].fillna("UNK").value_counts(normalize=True).to_dict()
    train_df["merchant_freq_global"] = train_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
    test_df["merchant_freq_global"] = test_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
    
    # Quick model training test
    try:
        # Simple stacking
        train_if, test_if, train_rule, test_rule = fit_if_and_rule_scores(
            train_df, test_df, num_cols, cat_cols_for_if=["operation_type"], rs=42
        )
        train_df["if_score"] = train_if
        train_df["rule_score"] = train_rule
        test_df["if_score"] = test_if  
        test_df["rule_score"] = test_rule
        
        num_cols_ext = num_cols + ["if_score", "rule_score"]
        pos_weight = (train_df["is_fraud"] == 0).sum() / max(1, (train_df["is_fraud"] == 1).sum())
        
        print(f"Building pipeline with {len(num_cols_ext)} numeric + {len(cat_cols)} categorical features")
        pipe = build_pipeline(num_cols_ext, cat_cols, pos_weight, 42)
        pipe.fit(train_df, train_df["is_fraud"])
        
        # Test scoring
        scores = compute_scores(pipe, test_df)
        y_test = test_df["is_fraud"].values
        
        ap = average_precision_score(y_test, scores)
        p05 = precision_at_k(y_test, scores, 0.005)
        p1 = precision_at_k(y_test, scores, 0.01)
        
        print(f"✅ Model training successful!")
        print(f"   AP: {ap:.4f}")
        print(f"   P@0.5%: {p05:.4f}")
        print(f"   P@1.0%: {p1:.4f}")
        print(f"   Fraud rate: {y_test.mean():.4f}")
        
        # Save quick results
        results = {
            "test_date": "2025-09-02",
            "features_tested": required_features,
            "sample_size": len(df),
            "performance": {
                "AP": float(ap),
                "P@0.5%": float(p05),
                "P@1.0%": float(p1),
                "fraud_rate": float(y_test.mean())
            },
            "feature_count": {
                "numeric": len(num_cols_ext),
                "categorical": len(cat_cols),
                "total": len(num_cols_ext) + len(cat_cols)
            }
        }
        
        out_dir = Path("reports/phase2/ulb_gbdt")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        (out_dir / "enhanced_features_test_results.json").write_text(json.dumps(results, indent=2))
        print(f"✅ Results saved to {out_dir / 'enhanced_features_test_results.json'}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in model training: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_features()
    if success:
        print("\n🎉 All tests passed! Enhanced features are working correctly.")
    else:
        print("\n❌ Tests failed. Check implementation.")
        sys.exit(1)