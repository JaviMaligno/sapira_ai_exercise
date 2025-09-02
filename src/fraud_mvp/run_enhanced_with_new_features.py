"""
Run enhanced GBDT with new recommended features
"""
import sys
sys.path.append('src')

from fraud_mvp.gbdt_ulb_enhanced import (
    Config, load_ulb, engineer_enhanced, time_aware_cv,
    train_eval_and_calibrate, evaluate_on_ulb_test,
    train_eval_on_combined_ulb_ieee, load_ieee
)
import json
from pathlib import Path

def main():
    # Use reduced config for faster execution
    cfg = Config()
    cfg.row_limit = 100_000  # Manageable size for testing
    cfg.out_dir = Path("reports/phase2/ulb_gbdt_enhanced_features")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running enhanced GBDT with new seasonality and balance velocity features...")
    print(f"Sample size: {cfg.row_limit:,}")
    
    # Load and engineer features
    df = load_ulb(cfg.row_limit)
    df, num_cols, cat_cols = engineer_enhanced(df)
    print(f"Feature engineering complete: {len(num_cols)} numeric + {len(cat_cols)} categorical features")
    
    # Time-aware CV
    print("\nRunning time-aware cross-validation...")
    cv_summary = time_aware_cv(df, num_cols, cat_cols, cfg)
    cv_path = cfg.out_dir / "enhanced_features_cv.json"
    cv_path.write_text(json.dumps(cv_summary, indent=2))
    print(f"CV results saved to {cv_path}")
    
    # Print CV results
    for fold in cv_summary["folds"]:
        print(f"  Fold {fold['train_frac']}-{fold['val_frac']}: AP={fold['AP']:.4f}, P@0.5%={fold['P@0.5%']:.4f}, P@1.0%={fold['P@1.0%']:.4f}")
    
    # Main train/eval with calibration
    print("\nRunning main training and evaluation...")
    final_summary = train_eval_and_calibrate(df, num_cols, cat_cols, cfg)
    final_path = cfg.out_dir / "enhanced_features_summary.json"
    final_path.write_text(json.dumps(final_summary, indent=2))
    print(f"Final results saved to {final_path}")
    
    # Print key results
    overall = final_summary["overall"]
    print(f"\nFinal Performance:")
    print(f"  AP: {overall['average_precision']:.4f}")
    print(f"  P@0.5%: {overall['precision_at']['0.5%']:.4f}")
    print(f"  P@1.0%: {overall['precision_at']['1.0%']:.4f}")
    print(f"  P@5.0%: {overall['precision_at']['5.0%']:.4f}")
    print(f"  Train fraud rate: {overall['fraud_rate_train']:.4f}")
    print(f"  Test fraud rate: {overall['fraud_rate_test']:.4f}")
    
    # Export thresholds for serving
    for frac in cfg.export_alert_fracs:
        key = str(frac)
        thr_map = final_summary.get("thresholds", {}).get(key, {})
        if thr_map:
            out_path = cfg.out_dir / f"enhanced_thresholds_{str(frac).replace('.', 'p')}.json"
            out_path.write_text(json.dumps(thr_map, indent=2))
            print(f"Thresholds saved to {out_path}")
    
    # Quick comparison with evaluation expectations
    expected_ap = 0.872  # From evaluation
    expected_p05 = 0.98   # From evaluation  
    expected_p1 = 0.825   # From evaluation
    
    actual_ap = overall['average_precision']
    actual_p05 = overall['precision_at']['0.5%']
    actual_p1 = overall['precision_at']['1.0%']
    
    print(f"\nComparison with evaluation expectations:")
    print(f"  AP: {actual_ap:.4f} vs expected ~{expected_ap:.4f} ({actual_ap-expected_ap:+.4f})")
    print(f"  P@0.5%: {actual_p05:.4f} vs expected ~{expected_p05:.4f} ({actual_p05-expected_p05:+.4f})")
    print(f"  P@1.0%: {actual_p1:.4f} vs expected ~{expected_p1:.4f} ({actual_p1-expected_p1:+.4f})")
    
    # Create implementation summary
    implementation_summary = {
        "implementation_date": "2025-09-02",
        "features_added": ["month_sin", "month_cos", "amount_rolling_std_24h", "amount_rolling_mean_24h"],
        "sample_size": cfg.row_limit,
        "performance": {
            "AP": actual_ap,
            "P@0.5%": actual_p05,
            "P@1.0%": actual_p1,
            "P@5.0%": overall['precision_at']['5.0%'],
        },
        "comparison_vs_evaluation": {
            "AP_delta": actual_ap - expected_ap,
            "P@0.5%_delta": actual_p05 - expected_p05,
            "P@1.0%_delta": actual_p1 - expected_p1,
        },
        "feature_count": {
            "numeric": len(num_cols),
            "categorical": len(cat_cols),
            "total": len(num_cols) + len(cat_cols),
        },
        "artifacts": {
            "cv_results": str(cv_path),
            "main_results": str(final_path),
            "thresholds": [str(cfg.out_dir / f"enhanced_thresholds_{str(frac).replace('.', 'p')}.json") for frac in cfg.export_alert_fracs]
        }
    }
    
    impl_path = cfg.out_dir / "implementation_summary.json"
    impl_path.write_text(json.dumps(implementation_summary, indent=2))
    print(f"\nImplementation summary saved to {impl_path}")
    
    print(f"\n✅ Enhanced feature implementation complete!")
    print(f"   All artifacts saved to: {cfg.out_dir}")

if __name__ == "__main__":
    main()