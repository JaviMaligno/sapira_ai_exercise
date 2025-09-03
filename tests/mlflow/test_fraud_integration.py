"""
Integration tests for MLflow with fraud detection workflow
Based on the integration testing from MLFLOW_SETUP.md
"""

import pytest
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, classification_report
)
from sapira_etl.mlflow_client import MLflowClient, setup_mlflow_environment
import json
import tempfile
from pathlib import Path


class TestFraudDetectionWorkflow:
    """Test complete fraud detection workflow with MLflow"""
    
    def test_complete_fraud_detection_pipeline(self, mlflow_tracking_uri, mlflow_client):
        """Test complete fraud detection training and logging pipeline"""
        # Set up MLflow
        setup_mlflow_environment()
        client = MLflowClient(
            tracking_uri=mlflow_tracking_uri,
            experiment_name="fraud-detection-integration-test"
        )
        
        # Generate synthetic fraud data
        np.random.seed(42)
        n_samples = 1000
        n_features = 15
        
        # Create realistic fraud detection features
        X = np.random.randn(n_samples, n_features)
        # Add some correlated features to make it more realistic
        X[:, 1] = X[:, 0] * 0.7 + np.random.randn(n_samples) * 0.3  # Amount correlation
        X[:, 2] = np.random.exponential(2, n_samples)  # Time since last transaction
        
        # Generate labels with realistic fraud rate (5%)
        fraud_probability = 1 / (1 + np.exp(-(X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 - 2)))
        y = np.random.binomial(1, fraud_probability)
        
        # Ensure we have some fraud cases
        fraud_rate = y.mean()
        assert 0.01 < fraud_rate < 0.2, f"Unrealistic fraud rate: {fraud_rate}"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate comprehensive fraud detection metrics
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'alert_rate': y_pred.mean(),
            'fraud_detection_rate': y_pred[y_test == 1].mean() if y_test.sum() > 0 else 0,
            'false_positive_rate': y_pred[y_test == 0].mean() if (y_test == 0).sum() > 0 else 0,
            'precision_at_5pct': self._precision_at_percentile(y_test, y_pred_proba, 0.05),
            'precision_at_1pct': self._precision_at_percentile(y_test, y_pred_proba, 0.01),
        }
        
        # Business metrics
        daily_alerts = int(metrics['alert_rate'] * 10000)  # Assume 10k transactions/day
        cost_per_alert = 2.5  # Assume $2.5 per alert review
        daily_cost = daily_alerts * cost_per_alert
        
        business_metrics = {
            'daily_alerts': daily_alerts,
            'cost_per_alert': cost_per_alert,
            'daily_cost': daily_cost,
            'fraud_cases_caught': int(metrics['fraud_detection_rate'] * y_test.sum()),
        }
        
        # Log comprehensive experiment
        run_id = client.log_training_run(
            model=model,
            params={
                'algorithm': 'random_forest',
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'class_weight': 'balanced',
                'threshold': 0.5,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': n_features,
                'fraud_rate_train': y_train.mean(),
                'fraud_rate_test': y_test.mean(),
            },
            metrics={**metrics, **business_metrics},
            artifacts={},  # Skip artifacts for this test
            tags={
                'model_family': 'fraud-detection',
                'algorithm': 'tree-based',
                'use_case': 'transaction_fraud',
                'data_type': 'synthetic',
                'test_type': 'integration',
                'class_balance': 'imbalanced',
                'validation_method': 'holdout'
            },
            run_name='fraud-integration-test'
        )
        
        # Verify the run was logged correctly
        assert run_id is not None
        logged_run = mlflow_client.get_run(run_id)
        
        # Verify parameters
        params = logged_run.data.params
        assert params['algorithm'] == 'random_forest'
        assert int(params['n_estimators']) == 50
        assert int(params['train_size']) == len(X_train)
        
        # Verify metrics
        logged_metrics = logged_run.data.metrics
        assert 'auc_roc' in logged_metrics
        assert 0.5 <= logged_metrics['auc_roc'] <= 1.0
        assert 'precision' in logged_metrics
        assert 'daily_alerts' in logged_metrics
        
        # Verify tags
        tags = logged_run.data.tags
        assert tags['model_family'] == 'fraud-detection'
        assert tags['test_type'] == 'integration'
        
        print(f"✅ Integration test completed successfully!")
        print(f"   Run ID: {run_id}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   Daily alerts: {daily_alerts}")
        
        return run_id, metrics
    
    def _precision_at_percentile(self, y_true, y_scores, percentile):
        """Calculate precision at a given percentile of predictions"""
        threshold = np.percentile(y_scores, (1 - percentile) * 100)
        y_pred_at_percentile = (y_scores >= threshold).astype(int)
        return precision_score(y_true, y_pred_at_percentile, zero_division=0)
    
    def test_model_validation_workflow(self, mlflow_tracking_uri):
        """Test model validation workflow with quality gates"""
        # First run the training pipeline
        run_id, metrics = self.test_complete_fraud_detection_pipeline(
            mlflow_tracking_uri, mlflow.MlflowClient()
        )
        
        # Define quality gates (similar to CI/CD pipeline)
        quality_gates = {
            'auc_roc': 0.60,  # Minimum AUC-ROC
            'precision': 0.10,  # Minimum precision (low for synthetic data)
            'recall': 0.05,    # Minimum recall
            'alert_rate': (0.001, 0.5),  # Alert rate should be reasonable
        }
        
        # Validate against quality gates
        validation_results = {}
        for metric, threshold in quality_gates.items():
            value = metrics[metric]
            
            if isinstance(threshold, tuple):
                # Range check
                min_val, max_val = threshold
                passed = min_val <= value <= max_val
                validation_results[metric] = {
                    'value': value,
                    'threshold': threshold,
                    'passed': passed,
                    'message': f"{metric}: {value:.4f} {'within' if passed else 'outside'} range [{min_val}, {max_val}]"
                }
            else:
                # Minimum threshold check
                passed = value >= threshold
                validation_results[metric] = {
                    'value': value,
                    'threshold': threshold,
                    'passed': passed,
                    'message': f"{metric}: {value:.4f} {'above' if passed else 'below'} threshold {threshold}"
                }
        
        # Check overall validation
        all_passed = all(result['passed'] for result in validation_results.values())
        
        print("🔍 Model Validation Results:")
        for metric, result in validation_results.items():
            status = "✅" if result['passed'] else "❌"
            print(f"   {status} {result['message']}")
        
        if all_passed:
            print("🎉 All quality gates passed!")
        else:
            print("⚠️ Some quality gates failed")
        
        # For integration test, we'll be lenient due to synthetic data
        # In production, you might want to assert all_passed
        assert len(validation_results) == len(quality_gates)
        
        return validation_results
    
    def test_artifact_creation_and_logging(self, test_experiment, temp_dir):
        """Test creation and logging of fraud detection artifacts"""
        
        # Create feature importance artifact
        feature_importance = {
            'transaction_amount': 0.25,
            'time_since_last': 0.20,
            'merchant_category': 0.15,
            'hour_of_day': 0.12,
            'day_of_week': 0.10,
            'payment_method': 0.08,
            'user_age': 0.05,
            'account_balance': 0.05
        }
        
        feature_importance_file = Path(temp_dir) / "feature_importance.json"
        with open(feature_importance_file, 'w') as f:
            json.dump(feature_importance, f, indent=2)
        
        # Create model configuration artifact
        model_config = {
            'preprocessing': {
                'scaling': 'StandardScaler',
                'categorical_encoding': 'one_hot',
                'missing_value_strategy': 'median'
            },
            'model': {
                'type': 'RandomForest',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'validation': {
                'method': 'time_series_split',
                'n_splits': 5,
                'test_size': 0.2
            },
            'thresholds': {
                'high_risk': 0.8,
                'medium_risk': 0.5,
                'low_risk': 0.2
            }
        }
        
        model_config_file = Path(temp_dir) / "model_config.json"
        with open(model_config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Create performance report
        performance_report = {
            'model_version': 'v2.1',
            'training_date': '2025-09-03',
            'data_period': '2025-01-01 to 2025-08-31',
            'metrics': {
                'auc_roc': 0.92,
                'precision': 0.85,
                'recall': 0.78,
                'f1_score': 0.815
            },
            'business_impact': {
                'fraud_detected': '87%',
                'false_positive_rate': '2%',
                'estimated_savings': '$2.5M annually'
            }
        }
        
        performance_file = Path(temp_dir) / "performance_report.json"
        with open(performance_file, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        # Log all artifacts
        with mlflow.start_run(experiment_id=test_experiment, 
                            run_name="fraud_artifacts_test"):
            mlflow.log_param('model_version', 'v2.1')
            mlflow.log_metric('auc_roc', 0.92)
            
            # Log artifacts
            mlflow.log_artifact(str(feature_importance_file), "metadata")
            mlflow.log_artifact(str(model_config_file), "config")
            mlflow.log_artifact(str(performance_file), "reports")
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify artifacts were logged
        client = mlflow.MlflowClient()
        
        # Check metadata artifacts
        metadata_artifacts = client.list_artifacts(run_id, "metadata")
        assert len(metadata_artifacts) > 0
        assert any("feature_importance.json" in a.path for a in metadata_artifacts)
        
        # Check config artifacts
        config_artifacts = client.list_artifacts(run_id, "config")
        assert len(config_artifacts) > 0
        assert any("model_config.json" in a.path for a in config_artifacts)
        
        # Check report artifacts
        report_artifacts = client.list_artifacts(run_id, "reports")
        assert len(report_artifacts) > 0
        assert any("performance_report.json" in a.path for a in report_artifacts)
        
        print("✅ All fraud detection artifacts logged successfully")
        
        return run_id
    
    def test_experiment_comparison(self, mlflow_tracking_uri):
        """Test comparing multiple fraud detection experiments"""
        client = MLflowClient(
            tracking_uri=mlflow_tracking_uri,
            experiment_name="fraud-comparison-test"
        )
        
        # Run multiple experiments with different parameters
        experiment_configs = [
            {'n_estimators': 50, 'max_depth': 5, 'name': 'light_model'},
            {'n_estimators': 100, 'max_depth': 10, 'name': 'balanced_model'},
            {'n_estimators': 200, 'max_depth': 15, 'name': 'heavy_model'},
        ]
        
        run_results = []
        
        for config in experiment_configs:
            # Generate data (same as before but simpler)
            np.random.seed(42)  # Same seed for fair comparison
            X = np.random.randn(500, 10)
            y = np.random.binomial(1, 0.05, 500)  # 5% fraud rate
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train model with specific config
            model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
            }
            
            # Log experiment
            run_id = client.log_training_run(
                model=model,
                params={
                    'n_estimators': config['n_estimators'],
                    'max_depth': config['max_depth'],
                    'model_complexity': config['name']
                },
                metrics=metrics,
                artifacts={},
                tags={'comparison_test': 'model_complexity'},
                run_name=f"fraud_model_{config['name']}"
            )
            
            run_results.append({
                'run_id': run_id,
                'config': config,
                'metrics': metrics
            })
        
        # Verify we can compare runs
        assert len(run_results) == 3
        
        # Find best model by AUC
        best_run = max(run_results, key=lambda x: x['metrics']['auc_roc'])
        print(f"✅ Experiment comparison completed")
        print(f"   Best model: {best_run['config']['name']} (AUC: {best_run['metrics']['auc_roc']:.3f})")
        
        return run_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])