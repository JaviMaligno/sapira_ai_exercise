"""
Pytest configuration and fixtures for MLflow tests
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import requests
from urllib.parse import urlparse
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def mlflow_tracking_uri():
    """Get MLflow tracking URI for tests"""
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@pytest.fixture(scope="session")
def check_mlflow_server(mlflow_tracking_uri):
    """Check if MLflow server is running, skip tests if not available"""
    try:
        response = requests.get(f"{mlflow_tracking_uri}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ MLflow server available at {mlflow_tracking_uri}")
            return True
        else:
            pytest.skip(f"MLflow server not healthy at {mlflow_tracking_uri}")
    except Exception as e:
        logger.warning(f"MLflow server not available: {e}")
        pytest.skip(f"MLflow server not available at {mlflow_tracking_uri}")


@pytest.fixture(scope="function")
def mlflow_client(mlflow_tracking_uri, check_mlflow_server):
    """Create MLflow client for tests"""
    # Ensure MLflow uses the server for both tracking and artifacts
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    
    # Set AWS credentials for client-side S3 access (from environment or .env file)
    # Load from .env file if available
    from pathlib import Path
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    except AttributeError:
        # Fallback for different MLflow versions
        pass
    
    return MlflowClient(tracking_uri=mlflow_tracking_uri)


@pytest.fixture(scope="function")
def test_experiment(mlflow_client):
    """Create a test experiment"""
    import uuid
    # Use UUID to avoid name collisions completely
    experiment_name = f"test_experiment_{uuid.uuid4().hex[:12]}"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except AttributeError:
        # Fallback for different MLflow versions
        experiment_id = mlflow_client.create_experiment(experiment_name)
    except Exception as e:
        if "already exists" in str(e):
            # If it exists, get the existing experiment
            try:
                experiment = mlflow_client.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id
            except:
                # Generate a new unique name
                experiment_name = f"test_experiment_{uuid.uuid4().hex[:12]}"
                experiment_id = mlflow_client.create_experiment(experiment_name)
        else:
            raise e
    
    yield experiment_id
    
    # Cleanup - delete experiment after test (optional - let it accumulate for debugging)
    # try:
    #     mlflow_client.delete_experiment(experiment_id)
    # except Exception as e:
    #     logger.warning(f"Could not cleanup experiment {experiment_id}: {e}")


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for test artifacts"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_fraud_data():
    """Generate sample fraud detection data for testing"""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    # Generate labels (10% fraud rate)
    y = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    return X, y


@pytest.fixture(scope="function")
def trained_model(sample_fraud_data):
    """Train a simple model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = sample_fraud_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model


@pytest.fixture(scope="function")
def sample_metrics():
    """Sample fraud detection metrics"""
    return {
        'auc_roc': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'f1_score': 0.77,
        'alert_rate': 0.05,
        'false_positive_rate': 0.02,
        'precision_at_1pct': 0.90
    }


@pytest.fixture(scope="function")
def sample_params():
    """Sample model parameters"""
    return {
        'algorithm': 'random_forest',
        'n_estimators': 10,
        'max_depth': 5,
        'threshold': 0.5,
        'model_type': 'fraud_detection'
    }


@pytest.fixture(scope="function")
def sample_tags():
    """Sample tags for experiments"""
    return {
        'model_family': 'fraud-detection',
        'environment': 'test',
        'version': 'test',
        'data_source': 'synthetic'
    }


# Test configuration constants
PERFORMANCE_THRESHOLDS = {
    'auc_roc': 0.70,  # Lower threshold for tests
    'precision': 0.60,
    'recall': 0.50,
    'f1_score': 0.55,
    'alert_rate': (0.001, 0.15)  # Wider range for tests
}

TEST_TIMEOUT = 30  # seconds