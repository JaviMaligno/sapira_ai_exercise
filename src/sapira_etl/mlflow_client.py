"""
MLflow client configuration and utilities for the fraud detection project.
"""

import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging

logger = logging.getLogger(__name__)

class MLflowClient:
    """Client for interacting with MLflow tracking server and artifact storage."""
    
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "fraud-detection"):
        """
        Initialize MLflow client.
        
        Args:
            tracking_uri: MLflow tracking server URI. Defaults to env var or localhost.
            experiment_name: Name of the MLflow experiment.
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.experiment_name = experiment_name
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set up experiment
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
            logger.info(f"Using MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set experiment {experiment_name}: {e}")
            self.experiment = None
    
    def log_training_run(
        self,
        model,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Dict[str, str],
        tags: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None
    ) -> str:
        """
        Log a complete training run to MLflow.
        
        Args:
            model: The trained model object
            params: Training parameters
            metrics: Model metrics
            artifacts: Dictionary of artifact_name -> file_path
            tags: Optional tags for the run
            run_name: Optional name for the run
            
        Returns:
            MLflow run ID
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Log model
            if hasattr(model, 'predict'):
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    serialization_format="pickle"
                )
            
            # Log artifacts
            for artifact_name, file_path in artifacts.items():
                if os.path.exists(file_path):
                    mlflow.log_artifact(file_path, artifact_name)
                else:
                    logger.warning(f"Artifact file not found: {file_path}")
            
            # Log additional metadata
            metadata = {
                "git_sha": self._get_git_sha(),
                "python_version": self._get_python_version(),
                "timestamp": self._get_timestamp()
            }
            mlflow.log_dict(metadata, "metadata.json")
            
            logger.info(f"Logged training run: {run.info.run_id}")
            return run.info.run_id
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        stage: str = "None"
    ) -> str:
        """
        Register a model from an MLflow run.
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            stage: Model stage (None, Staging, Production)
            
        Returns:
            Model version
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Set stage if specified
            if stage and stage != "None":
                client = mlflow.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage=stage
                )
            
            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def export_production_artifacts(
        self,
        model_name: str,
        stage: str = "Production",
        output_dir: str = "artifacts/mlflow_export"
    ) -> Dict[str, str]:
        """
        Export artifacts from a production model to local directory.
        
        Args:
            model_name: Name of the registered model
            stage: Model stage to export from
            output_dir: Local directory to export to
            
        Returns:
            Dictionary of exported artifact paths
        """
        try:
            client = mlflow.MlflowClient()
            
            # Get latest model version in the specified stage
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"No model found in {stage} stage")
            
            latest_version = versions[0]
            run_id = latest_version.run_id
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Download all artifacts from the run
            artifacts = client.download_artifacts(run_id, "", str(output_path))
            
            # Create manifest
            manifest = {
                "model_name": model_name,
                "model_version": latest_version.version,
                "run_id": run_id,
                "stage": stage,
                "exported_at": self._get_timestamp(),
                "artifacts_path": str(output_path)
            }
            
            manifest_path = output_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Exported model artifacts to {output_path}")
            return {"manifest": str(manifest_path), "artifacts": str(output_path)}
            
        except Exception as e:
            logger.error(f"Failed to export artifacts: {e}")
            raise
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all MLflow experiments."""
        try:
            client = mlflow.MlflowClient()
            experiments = client.search_experiments()
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def list_runs(self, experiment_name: Optional[str] = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """List recent runs from an experiment."""
        try:
            client = mlflow.MlflowClient()
            
            if experiment_name:
                experiment = client.get_experiment_by_name(experiment_name)
                experiment_ids = [experiment.experiment_id] if experiment else []
            else:
                # Get all experiments if no specific one is requested
                experiments = client.search_experiments()
                experiment_ids = [exp.experiment_id for exp in experiments]
            
            runs = client.search_runs(
                experiment_ids=experiment_ids,
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                for run in runs
            ]
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []
    
    def check_s3_connectivity(self) -> bool:
        """Check if S3 backend is accessible."""
        try:
            # Try to create a boto3 client with current configuration
            s3_client = boto3.client('s3')
            
            # Try to list buckets (minimal operation)
            s3_client.list_buckets()
            
            logger.info("S3 connectivity check passed")
            return True
            
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"S3 connectivity check failed: {e}")
            return False
    
    def _get_git_sha(self) -> str:
        """Get current git SHA."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


def setup_mlflow_environment():
    """Set up MLflow environment variables from .env.mlflow if it exists."""
    from dotenv import load_dotenv
    
    # Load MLflow-specific environment variables
    mlflow_env_path = Path(".env.mlflow")
    if mlflow_env_path.exists():
        load_dotenv(mlflow_env_path)
        logger.info("Loaded MLflow environment from .env.mlflow")
    else:
        logger.warning(".env.mlflow not found. Using default configuration.")


if __name__ == "__main__":
    # Example usage
    setup_mlflow_environment()
    
    client = MLflowClient()
    
    # Check connectivity
    print("MLflow Server:", client.tracking_uri)
    print("S3 Connectivity:", "✓" if client.check_s3_connectivity() else "✗")
    
    # List experiments
    experiments = client.list_experiments()
    print(f"Experiments: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp['name']} ({exp['experiment_id']})")