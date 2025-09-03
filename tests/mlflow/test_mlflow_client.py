"""
Test MLflow Python client functionality
Based on the Python client testing from MLFLOW_SETUP.md
"""

import pytest
import mlflow
import mlflow.sklearn
import os
import json
import tempfile
from pathlib import Path
from sapira_etl.mlflow_client import MLflowClient, setup_mlflow_environment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score


class TestMLflowPythonClient:
    """Test basic MLflow Python client functionality"""
    
    def test_mlflow_client_initialization(self, mlflow_tracking_uri):
        """Test MLflow client can be initialized"""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        
        # Test basic client functionality
        experiments = client.search_experiments()
        assert isinstance(experiments, list)
    
    def test_experiment_creation_and_retrieval(self, mlflow_client):
        """Test creating and retrieving experiments"""
        experiment_name = f"test_experiment_creation_{int(__import__('time').time())}"
        
        # Create experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        assert experiment_id is not None
        
        # Retrieve experiment
        experiment = mlflow_client.get_experiment(experiment_id)
        assert experiment.name == experiment_name
        assert experiment.experiment_id == experiment_id
        
        # Test get by name
        experiment_by_name = mlflow_client.get_experiment_by_name(experiment_name)
        assert experiment_by_name.experiment_id == experiment_id
        
        # Cleanup
        mlflow_client.delete_experiment(experiment_id)
    
    def test_run_creation_and_logging(self, test_experiment):
        """Test creating runs and logging data"""
        with mlflow.start_run(experiment_id=test_experiment, run_name="test_run") as run:
            # Log parameters
            mlflow.log_param("algorithm", "test")
            mlflow.log_param("threshold", 0.5)
            
            # Log metrics
            mlflow.log_metric("accuracy", 0.95)
            mlflow.log_metric("auc_roc", 0.87)
            
            # Log tags
            mlflow.set_tag("model_type", "test")
            
            run_id = run.info.run_id
        
        # Verify logged data
        client = mlflow.MlflowClient()
        run_data = client.get_run(run_id)
        
        assert run_data.data.params["algorithm"] == "test"
        assert float(run_data.data.params["threshold"]) == 0.5
        assert run_data.data.metrics["accuracy"] == 0.95
        assert run_data.data.metrics["auc_roc"] == 0.87
        assert run_data.data.tags["model_type"] == "test"
    
    def test_artifact_logging(self, test_experiment, temp_dir):
        """Test logging artifacts"""
        # Create test artifact
        artifact_file = Path(temp_dir) / "test_artifact.json"
        test_data = {"test_key": "test_value", "metrics": [1, 2, 3]}
        
        with open(artifact_file, 'w') as f:
            json.dump(test_data, f)
        
        with mlflow.start_run(experiment_id=test_experiment):
            # Log artifact
            mlflow.log_artifact(str(artifact_file), "test_artifacts")
            run_id = mlflow.active_run().info.run_id
        
        # Verify artifact was logged
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, "test_artifacts")
        
        assert len(artifacts) > 0
        artifact_names = [a.path for a in artifacts]
        assert "test_artifacts/test_artifact.json" in artifact_names
    
    def test_model_logging(self, test_experiment, trained_model, sample_fraud_data):
        """Test logging sklearn models"""
        X, y = sample_fraud_data
        
        with mlflow.start_run(experiment_id=test_experiment):
            # Log model
            mlflow.sklearn.log_model(
                trained_model,
                "model",
                input_example=X[:5],  # Provide input example to avoid warnings
            )
            run_id = mlflow.active_run().info.run_id
        
        # Test model loading
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Test predictions
        predictions = loaded_model.predict(X[:10])
        original_predictions = trained_model.predict(X[:10])
        
        assert len(predictions) == len(original_predictions)
        assert all(predictions == original_predictions)


class TestCustomMLflowClient:
    """Test our custom MLflowClient class"""
    
    def test_custom_client_initialization(self, mlflow_tracking_uri):
        """Test custom MLflow client initialization"""
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
        setup_mlflow_environment()
        
        client = MLflowClient(experiment_name="test-custom-client")
        assert client.tracking_uri == mlflow_tracking_uri
        assert client.experiment is not None
    
    def test_list_experiments(self, mlflow_tracking_uri):
        """Test listing experiments with custom client"""
        client = MLflowClient(tracking_uri=mlflow_tracking_uri)
        experiments = client.list_experiments()
        
        assert isinstance(experiments, list)
        if experiments:
            # Check structure of returned experiments
            exp = experiments[0]
            assert "experiment_id" in exp
            assert "name" in exp
            assert "lifecycle_stage" in exp
    
    def test_list_runs(self, mlflow_tracking_uri, test_experiment):
        """Test listing runs with custom client"""
        # Create some test runs
        for i in range(3):
            with mlflow.start_run(experiment_id=test_experiment):
                mlflow.log_param("test_param", i)
                mlflow.log_metric("test_metric", i * 0.1)
        
        client = MLflowClient(tracking_uri=mlflow_tracking_uri)
        runs = client.list_runs(max_results=5)
        
        assert isinstance(runs, list)
        assert len(runs) >= 3
        
        # Check structure of returned runs
        if runs:
            run = runs[0]
            assert "run_id" in run
            assert "experiment_id" in run
            assert "status" in run
            assert "metrics" in run
            assert "params" in run


class TestFraudDetectionWorkflow:
    """Test fraud detection specific workflow"""
    
    @pytest.mark.skip(reason="Requires artifact storage configuration")
    def test_fraud_detection_experiment_logging(self, test_experiment, 
                                               sample_fraud_data, trained_model):
        """Test complete fraud detection experiment logging"""
        X, y = sample_fraud_data
        
        # Make predictions
        y_pred = trained_model.predict(X)
        y_pred_proba = trained_model.predict_proba(X)[:, 1]
        
        # Calculate fraud detection metrics
        metrics = {
            'auc_roc': roc_auc_score(y, y_pred_proba),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'alert_rate': y_pred.mean(),
            'fraud_detection_rate': y_pred[y == 1].mean() if y.sum() > 0 else 0
        }
        
        # Log complete experiment
        with mlflow.start_run(experiment_id=test_experiment, 
                            run_name="fraud_detection_test"):
            # Log fraud detection parameters
            mlflow.log_param('algorithm', 'random_forest')
            mlflow.log_param('n_estimators', trained_model.n_estimators)
            mlflow.log_param('threshold', 0.5)
            mlflow.log_param('train_size', len(X))
            
            # Log fraud detection metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log business metrics
            mlflow.log_metric('daily_alerts', 50)  # Simulated
            mlflow.log_metric('cost_per_alert', 2.5)
            
            # Log tags
            mlflow.set_tags({
                'model_family': 'fraud-detection',
                'environment': 'test',
                'data_type': 'synthetic'
            })
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify all data was logged correctly
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        
        # Check parameters
        assert run.data.params['algorithm'] == 'random_forest'
        assert float(run.data.params['threshold']) == 0.5
        
        # Check metrics exist
        for metric in ['auc_roc', 'precision', 'recall', 'alert_rate']:
            assert metric in run.data.metrics
            assert 0 <= run.data.metrics[metric] <= 1
        
        # Check tags
        assert run.data.tags['model_family'] == 'fraud-detection'
        assert run.data.tags['environment'] == 'test'
    
    @pytest.mark.skip(reason="Requires model registry and artifact storage")
    def test_model_registry_simulation(self, test_experiment, trained_model, 
                                     sample_fraud_data, mlflow_tracking_uri):
        """Test model registry workflow (without actual artifact storage)"""
        X, y = sample_fraud_data
        
        with mlflow.start_run(experiment_id=test_experiment):
            # Log model and metrics
            mlflow.log_param('algorithm', 'test_model')
            mlflow.log_metric('auc_roc', 0.85)
            
            # For testing purposes, we'll skip actual model logging
            # since it requires proper artifact storage configuration
            run_id = mlflow.active_run().info.run_id
        
        # Test custom client model registration methods
        client = MLflowClient(tracking_uri=mlflow_tracking_uri)
        
        # Test export functionality (should handle gracefully)
        try:
            artifacts = client.export_production_artifacts(
                model_name="test-model",
                stage="Production",
                output_dir="/tmp/test_export"
            )
            # If this succeeds, great! If not, it should fail gracefully
        except Exception as e:
            # Expected for test environment without full registry setup
            assert "No model found" in str(e) or "Model not found" in str(e)


class TestMLflowPerformance:
    """Test MLflow performance characteristics"""
    
    def test_logging_performance(self, test_experiment):
        """Test logging performance for multiple metrics"""
        import time
        
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=test_experiment):
            # Log many parameters and metrics
            for i in range(50):
                mlflow.log_param(f"param_{i}", f"value_{i}")
                mlflow.log_metric(f"metric_{i}", i * 0.1)
        
        duration = time.time() - start_time
        
        # Should complete within reasonable time (30 seconds is generous)
        assert duration < 30, f"Logging took too long: {duration:.2f}s"
    
    def test_concurrent_runs(self, test_experiment):
        """Test concurrent run creation"""
        import concurrent.futures
        import time
        
        def create_run(run_num):
            with mlflow.start_run(experiment_id=test_experiment,
                                run_name=f"concurrent_run_{run_num}"):
                mlflow.log_param("run_number", run_num)
                mlflow.log_metric("test_metric", run_num * 0.1)
                return mlflow.active_run().info.run_id
        
        start_time = time.time()
        
        # Create 5 concurrent runs
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_run, i) for i in range(5)]
            run_ids = [future.result(timeout=30) for future in futures]
        
        duration = time.time() - start_time
        
        # All runs should complete successfully
        assert len(run_ids) == 5
        assert len(set(run_ids)) == 5  # All unique
        
        # Should complete within reasonable time
        assert duration < 60, f"Concurrent logging took too long: {duration:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])