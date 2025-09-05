#!/usr/bin/env python3
"""
Test Production MLflow Integration
==================================

This script tests our production MLflow deployment by:
1. Loading a sample of fraud detection data
2. Training a simple model
3. Logging everything to production MLflow
4. Verifying the experiment shows up in the UI

This is based on the existing gbdt_ulb_enhanced.py but simplified for testing.
"""

import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Production MLflow Configuration
MLFLOW_TRACKING_URI = "http://fraud-scoring-mlflow-alb-1648380913.eu-west-2.elb.amazonaws.com"
ULB_PATH = Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv")

def load_sample_data(n_rows: int = 10000) -> pd.DataFrame:
    """Load a small sample of ULB fraud data for testing."""
    print(f"📊 Loading {n_rows:,} rows from ULB dataset...")
    
    # Load subset of columns for simple test
    usecols = ["trans_num", "unix_time", "category", "amt", "merchant", "is_fraud", "cc_num"]
    df = pd.read_csv(ULB_PATH, usecols=usecols, nrows=n_rows)
    
    # Basic preprocessing
    df.rename(columns={
        "trans_num": "transaction_id",
        "unix_time": "event_time_ts",
        "category": "operation_type", 
        "amt": "amount",
        "merchant": "merchant_name",
    }, inplace=True)
    
    # Create event_time
    df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)
    
    # Basic features
    df["abs_amount"] = df["amount"].abs()
    df["log_amount"] = np.log1p(df["abs_amount"])
    df["hour"] = df["event_time"].dt.hour
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_weekend"] = (df["event_time"].dt.dayofweek >= 5).astype(int)
    
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   Fraud rate: {df['is_fraud'].mean():.3f}")
    print(f"   Date range: {df['event_time'].min()} to {df['event_time'].max()}")
    
    return df

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare simple features for modeling."""
    print("🔧 Preparing features...")
    
    # Select numeric features only for simplicity
    feature_cols = ["abs_amount", "log_amount", "hour", "is_night", "is_weekend"]
    X = df[feature_cols].copy()
    y = df["is_fraud"].astype(int)
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    print(f"✅ Prepared {X.shape[1]} features for {X.shape[0]:,} samples")
    return X, y, feature_cols

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train a simple fraud detection model."""
    print("🤖 Training fraud detection model...")
    
    # Split data chronologically (simpler than the original script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=6,
        max_iter=100,  # Reduced for faster testing
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=42,
        class_weight="balanced"  # Handle class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    # Precision at different alert rates
    def precision_at_k(y_true, scores, k_frac):
        n = len(scores)
        k = max(1, int(n * k_frac))
        top_k_idx = np.argsort(scores)[-k:]
        return (y_true.iloc[top_k_idx] == 1).mean()
    
    p_at_1pct = precision_at_k(y_test, y_pred_proba, 0.01)
    p_at_5pct = precision_at_k(y_test, y_pred_proba, 0.05)
    
    metrics = {
        "auc_roc": auc,
        "average_precision": ap,
        "precision_at_1pct": p_at_1pct,
        "precision_at_5pct": p_at_5pct,
        "fraud_rate_train": y_train.mean(),
        "fraud_rate_test": y_test.mean(),
        "train_samples": len(y_train),
        "test_samples": len(y_test)
    }
    
    print(f"✅ Model trained successfully!")
    print(f"   AUC-ROC: {auc:.3f}")
    print(f"   Average Precision: {ap:.3f}")
    print(f"   Precision @ 1%: {p_at_1pct:.3f}")
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba
    }

def test_production_mlflow():
    """Test our production MLflow deployment with a fraud detection experiment."""
    print("🚀 Testing Production MLflow Integration")
    print("=" * 50)
    
    # Configure MLflow
    print(f"🔗 Connecting to production MLflow: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Test connection
    try:
        experiments = mlflow.search_experiments()
        print(f"✅ Connected successfully! Found {len(experiments)} existing experiments")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Create/get experiment
    experiment_name = "fraud-detection-production-test"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Created new experiment: {experiment_name}")
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {experiment_name}")
    
    # Load data
    df = load_sample_data(n_rows=10000)
    X, y, feature_cols = prepare_features(df)
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"gbdt-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        print(f"📊 Started MLflow run: {run.info.run_id}")
        
        # Log parameters
        params = {
            "algorithm": "HistGradientBoostingClassifier",
            "learning_rate": 0.1,
            "max_depth": 6,
            "max_iter": 100,
            "random_state": 42,
            "test_size": 0.3,
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "data_source": "ULB Credit Card Fraud Dataset",
            "mlflow_server": "production-aws"
        }
        
        print("📝 Logging parameters...")
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Train model
        results = train_model(X, y)
        
        # Log metrics
        print("📈 Logging metrics...")
        for key, value in results["metrics"].items():
            mlflow.log_metric(key, value)
        
        # Log tags
        mlflow.set_tags({
            "model_family": "fraud-detection",
            "dataset": "ulb",
            "environment": "production-test",
            "version": "test-v1",
            "created_by": "production-mlflow-test"
        })
        
        # Log model
        print("🤖 Logging model...")
        mlflow.sklearn.log_model(
            results["model"], 
            "fraud_detection_model",
            registered_model_name="fraud-detector-test"
        )
        
        # Save and log artifacts
        print("📁 Logging artifacts...")
        
        # Feature list
        feature_info = {
            "feature_names": feature_cols,
            "feature_count": len(feature_cols),
            "preprocessing": "StandardScaler"
        }
        with open("feature_info.json", "w") as f:
            json.dump(feature_info, f, indent=2)
        mlflow.log_artifact("feature_info.json", "model_metadata")
        
        # Model summary
        model_summary = {
            "model_type": "HistGradientBoostingClassifier",
            "training_date": datetime.now().isoformat(),
            "performance": results["metrics"],
            "hyperparameters": params
        }
        with open("model_summary.json", "w") as f:
            json.dump(model_summary, f, indent=2)
        mlflow.log_artifact("model_summary.json", "model_metadata")
        
        # Clean up temporary files
        os.remove("feature_info.json")
        os.remove("model_summary.json")
        
        # Success!
        run_url = f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run.info.run_id}"
        print("\n🎉 SUCCESS! Production MLflow test completed!")
        print("=" * 50)
        print(f"📊 Run ID: {run.info.run_id}")
        print(f"🔗 View in UI: {run_url}")
        print(f"📈 Key Metrics:")
        print(f"   • AUC-ROC: {results['metrics']['auc_roc']:.3f}")
        print(f"   • Average Precision: {results['metrics']['average_precision']:.3f}")
        print(f"   • Precision @ 1%: {results['metrics']['precision_at_1pct']:.3f}")
        print(f"   • Training Samples: {results['metrics']['train_samples']:,}")
        
        return {
            "run_id": run.info.run_id,
            "experiment_id": experiment_id,
            "run_url": run_url,
            "metrics": results["metrics"]
        }

if __name__ == "__main__":
    result = test_production_mlflow()
    
    if result:
        print("\n✅ Production MLflow integration test PASSED!")
        print("🌐 Your fraud detection experiment is now live in production MLflow!")
    else:
        print("\n❌ Production MLflow integration test FAILED!")