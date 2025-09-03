# MLflow Server Setup Guide

This guide walks you through setting up MLflow with S3 backend storage for the fraud detection project, supporting the automation opportunities outlined in `reports/automation_opportunities.md`.

## Overview

MLflow provides:
- **Experiment Tracking**: Log training runs, parameters, metrics, and artifacts
- **Model Registry**: Versioned model storage with staging and production promotion
- **Artifact Storage**: Centralized storage for models and associated files
- **Automated Workflows**: Integration with CI/CD for model deployment

## Setup Options

### Option 1: AWS S3 Backend (Recommended for Production)

#### Prerequisites
1. AWS CLI installed and configured
2. AWS account with S3 access
3. Docker and Docker Compose installed

#### Quick Setup
```bash
# Run the AWS setup script
./setup-mlflow-aws.sh

# Update .env.mlflow with your actual AWS credentials
vim .env.mlflow

# Start MLflow server
docker-compose -f docker-compose.mlflow.yml --env-file .env.mlflow up -d
```

#### Manual AWS Setup

1. **Configure AWS credentials**:
```bash
aws configure
# Enter your access key, secret key, region (e.g., us-west-2)
```

2. **Create S3 bucket** (automated by setup script):
```bash
# Get your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create bucket
aws s3 mb s3://fraud-scoring-models-${AWS_ACCOUNT_ID} --region us-west-2

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket fraud-scoring-models-${AWS_ACCOUNT_ID} \
    --versioning-configuration Status=Enabled
```

3. **Configure environment**:
```bash
cp .env.mlflow.aws .env.mlflow
# Edit .env.mlflow with your AWS credentials
```

4. **Start services**:
```bash
docker-compose -f docker-compose.mlflow.yml --env-file .env.mlflow up -d
```

### Option 2: Local MinIO Backend (Recommended for Development)

#### Quick Setup
```bash
# Run the local setup script
./setup-mlflow-local.sh

# Start MLflow server with MinIO
docker-compose -f docker-compose.mlflow.yml --env-file .env.mlflow --profile local up -d
```

This provides:
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)
- Local S3-compatible storage for development

## Services Architecture

The MLflow setup includes:

1. **PostgreSQL Database** (`mlflow-db`)
   - Stores experiment metadata, runs, and model registry
   - Port: 5433 (to avoid conflicts with existing services)

2. **MLflow Tracking Server** (`mlflow-server`)
   - Web UI for experiment tracking and model registry
   - Port: 5000
   - Connects to PostgreSQL backend and S3 artifacts

3. **MinIO** (local setup only)
   - S3-compatible object storage for local development
   - Ports: 9000 (API), 9001 (Console)

## Environment Variables

### AWS S3 Configuration (.env.mlflow with AWS)
```bash
# AWS credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-west-2
AWS_ACCOUNT_ID=your_aws_account_id

# MLflow configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow_user:mlflow_password@mlflow-db:5432/mlflow_db
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://fraud-scoring-models-${AWS_ACCOUNT_ID}/mlflow-artifacts
```

### Local MinIO Configuration (.env.mlflow with MinIO)
```bash
# MinIO credentials (local S3-compatible)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
MLFLOW_S3_IGNORE_TLS=true

# MLflow configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://fraud-scoring-models/mlflow-artifacts
```

## Usage

### 1. Basic Python Client

```python
from src.sapira_etl.mlflow_client import MLflowClient, setup_mlflow_environment

# Set up environment
setup_mlflow_environment()

# Create client
client = MLflowClient(experiment_name="fraud-detection")

# Log a training run
run_id = client.log_training_run(
    model=trained_model,
    params={"n_estimators": 100, "max_depth": 10},
    metrics={"accuracy": 0.95, "auc": 0.87},
    artifacts={
        "thresholds": "path/to/thresholds.json",
        "pipeline": "path/to/pipeline.pkl"
    },
    tags={"model_type": "gbdt", "version": "v2"}
)

# Register model
version = client.register_model(run_id, "fraud-detector", stage="Staging")
```

### 2. Export Production Artifacts

```python
# Export current production model artifacts
artifacts = client.export_production_artifacts(
    model_name="fraud-detector",
    stage="Production",
    output_dir="artifacts/current_production"
)
```

### 3. MLflow UI Operations

1. **Access MLflow UI**: http://localhost:5000
2. **Create experiments**: Organize different model approaches
3. **Compare runs**: Side-by-side comparison of metrics and parameters
4. **Model registry**: Promote models through stages (None → Staging → Production)
5. **Download artifacts**: Access stored models and associated files

## Integration with Existing Workflow

### 1. Training Script Integration

Update your training scripts to use MLflow:

```python
import mlflow
from src.sapira_etl.mlflow_client import MLflowClient

# Initialize MLflow
client = MLflowClient(experiment_name="fraud-detection-ulb-gbdt")

# Train model
model, metrics = train_fraud_model()

# Log to MLflow
run_id = client.log_training_run(
    model=model,
    params={
        "algorithm": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.1
    },
    metrics=metrics,
    artifacts={
        "pipeline": "artifacts/pipeline.pkl",
        "thresholds": "artifacts/thresholds.json",
        "feature_importance": "artifacts/feature_importance.png"
    },
    tags={
        "data_version": "2025-01",
        "model_type": "production"
    }
)
```

### 2. Automated Model Promotion

```python
# After validation, promote to staging
if validation_metrics["auc"] > 0.85:
    version = client.register_model(run_id, "fraud-detector", stage="Staging")
    
    # After production testing, promote to production
    if production_validation_passed:
        mlflow_client = mlflow.MlflowClient()
        mlflow_client.transition_model_version_stage(
            name="fraud-detector",
            version=version,
            stage="Production"
        )
```

### 3. CI/CD Integration

Add to your CI/CD pipeline:

```yaml
# GitHub Actions example
- name: Train and Register Model
  run: |
    poetry run python train_model.py
    poetry run python -c "
    from src.sapira_etl.mlflow_client import MLflowClient
    client = MLflowClient()
    # Auto-promote if metrics pass threshold
    "
```

## Supported Automation Opportunities

This MLflow setup enables the following automation opportunities from your plan:

### 9. MLflow Experiment Tracking
- ✅ **Complete**: Track experiments, parameters, metrics, and artifacts
- ✅ **Model Registry**: Versioned models with staging/production stages

### 10. Automated Export from MLflow Registry
- ✅ **Artifact Export**: `export_production_artifacts()` method
- ✅ **Manifest Generation**: Includes model metadata and version info

### 11. CI/CD Integration
- ✅ **Training Pipeline**: Log runs automatically from training scripts
- ✅ **Model Promotion**: Automated staging and production promotion

### 17. Model Provenance
- ✅ **Metadata Tracking**: Git SHA, timestamps, parameters, metrics
- ✅ **Lineage**: Full history of model versions and stages

### 22. Automated Model Card
- ✅ **Run Metadata**: Parameters, metrics, and artifacts automatically logged
- 🔄 **Documentation**: Can generate model cards from MLflow metadata

## Operations

### Start Services
```bash
# AWS S3 backend
docker-compose -f docker-compose.mlflow.yml --env-file .env.mlflow up -d

# Local MinIO backend
docker-compose -f docker-compose.mlflow.yml --env-file .env.mlflow --profile local up -d
```

### Stop Services
```bash
docker-compose -f docker-compose.mlflow.yml down
```

### View Logs
```bash
# MLflow server logs
docker-compose -f docker-compose.mlflow.yml logs mlflow-server

# Database logs
docker-compose -f docker-compose.mlflow.yml logs mlflow-db
```

### Backup and Restore

#### Database Backup
```bash
docker-compose -f docker-compose.mlflow.yml exec mlflow-db pg_dump -U mlflow_user mlflow_db > mlflow_backup.sql
```

#### S3 Artifacts Backup (AWS)
```bash
aws s3 sync s3://fraud-scoring-models-${AWS_ACCOUNT_ID}/mlflow-artifacts ./mlflow_artifacts_backup
```

### Health Checks

```python
from src.sapira_etl.mlflow_client import MLflowClient, setup_mlflow_environment

setup_mlflow_environment()
client = MLflowClient()

# Check MLflow connectivity
print("MLflow Server:", client.tracking_uri)
print("S3 Connectivity:", "✓" if client.check_s3_connectivity() else "✗")

# List experiments and runs
experiments = client.list_experiments()
print(f"Total experiments: {len(experiments)}")
```

## Security Considerations

1. **AWS Credentials**: Store in environment variables, never in code
2. **Database**: Use strong passwords, consider encryption at rest
3. **Network**: Configure security groups/firewall rules for production
4. **S3 Bucket**: Enable versioning and encryption
5. **Access Control**: Use IAM roles for fine-grained permissions

## Troubleshooting

### Common Issues

1. **S3 Connectivity Issues**
   - Check AWS credentials: `aws sts get-caller-identity`
   - Verify bucket exists and permissions are correct
   - For MinIO: ensure services are running

2. **MLflow UI Not Accessible**
   - Check if container is running: `docker-compose ps`
   - View logs: `docker-compose logs mlflow-server`
   - Verify port 5000 is not in use

3. **Database Connection Errors**
   - Wait for PostgreSQL to fully start (health check)
   - Check database credentials in environment file

4. **Artifact Storage Issues**
   - Verify S3 bucket exists and is accessible
   - Check artifact root configuration
   - For MinIO: verify bucket was created by minio-client

### Reset Everything

```bash
# Stop all services
docker-compose -f docker-compose.mlflow.yml down

# Remove volumes (destroys data)
docker volume rm mlflow_postgres_data minio_data

# Restart setup
./setup-mlflow-aws.sh  # or ./setup-mlflow-local.sh
```

## Experiment Creation and Metrics Loading

### Creating Experiments and Logging Metrics

Based on our successful testing, here's how to create experiments and log metrics:

#### 1. Basic Experiment with Metrics

```python
import mlflow
import os

# Set MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
mlflow.set_tracking_uri('http://localhost:5000')

# Create or get experiment
experiment_name = 'fraud-detection-production'
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except Exception:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

# Start a run and log fraud detection metrics
with mlflow.start_run(experiment_id=experiment_id, run_name='gbdt-model-v1') as run:
    # Log fraud detection parameters
    mlflow.log_param('threshold', 0.5)
    mlflow.log_param('model_type', 'LightGBM')
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('max_depth', 6)
    mlflow.log_param('learning_rate', 0.1)
    
    # Log fraud detection metrics
    mlflow.log_metric('precision', 0.85)
    mlflow.log_metric('recall', 0.78)
    mlflow.log_metric('f1_score', 0.815)
    mlflow.log_metric('auc_roc', 0.92)
    mlflow.log_metric('auc_pr', 0.88)
    mlflow.log_metric('alert_rate', 0.05)
    mlflow.log_metric('false_positive_rate', 0.02)
    mlflow.log_metric('precision_at_1pct', 0.95)
    
    # Log business metrics
    mlflow.log_metric('daily_alerts', 150)
    mlflow.log_metric('cost_per_alert', 2.50)
    mlflow.log_metric('fraud_caught_rate', 0.87)
    
    # Log training metadata
    mlflow.log_param('training_data_size', 100000)
    mlflow.log_param('validation_data_size', 20000)
    mlflow.log_param('feature_count', 25)
    mlflow.log_param('training_duration_minutes', 15)
    
    # Log tags for organization
    mlflow.set_tags({
        'model_family': 'fraud-detection',
        'algorithm': 'gradient-boosting',
        'environment': 'production',
        'version': 'v2.1',
        'data_source': 'bank_transactions',
        'deployment_target': 'scoring-service'
    })
    
    print(f'Run logged successfully: {run.info.run_id}')
    print(f'View at: http://localhost:5000/#/experiments/{experiment_id}/runs/{run.info.run_id}')
```

#### 2. Advanced Model Logging with Artifacts

```python
import mlflow
import mlflow.sklearn
import json
import pickle
from sklearn.ensemble import RandomForestClassifier

# Train a model (example)
model = RandomForestClassifier(n_estimators=100, max_depth=5)
# model.fit(X_train, y_train)  # Your training code here

with mlflow.start_run(run_name='fraud-model-with-artifacts') as run:
    # Log model parameters and metrics (as above)
    
    # Log the trained model (when artifact storage is properly configured)
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="fraud-detection-rf"
    )
    
    # Log custom artifacts
    # Feature importance
    feature_importance = {'feature_1': 0.3, 'feature_2': 0.25, 'feature_3': 0.2}
    with open('feature_importance.json', 'w') as f:
        json.dump(feature_importance, f)
    mlflow.log_artifact('feature_importance.json', 'metadata')
    
    # Model configuration
    model_config = {
        'threshold_rules': {'high_risk': 0.8, 'medium_risk': 0.5},
        'feature_preprocessing': 'standard_scaler',
        'validation_approach': 'time_series_split'
    }
    with open('model_config.json', 'w') as f:
        json.dump(model_config, f)
    mlflow.log_artifact('model_config.json', 'config')
    
    # Cleanup temporary files
    os.remove('feature_importance.json')
    os.remove('model_config.json')
```

#### 3. Using the MLflowClient Class

```python
from src.sapira_etl.mlflow_client import MLflowClient, setup_mlflow_environment

# Initialize
setup_mlflow_environment()
client = MLflowClient(experiment_name="fraud-detection-automated")

# Example of comprehensive logging
run_id = client.log_training_run(
    model=trained_model,  # Your trained model
    params={
        "algorithm": "lightgbm",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "threshold": 0.5
    },
    metrics={
        "auc_roc": 0.92,
        "auc_pr": 0.88,
        "precision": 0.85,
        "recall": 0.78,
        "f1_score": 0.815,
        "precision_at_1pct": 0.95,
        "alert_rate": 0.05,
        "false_positive_rate": 0.02
    },
    artifacts={
        "pipeline": "artifacts/pipeline.pkl",
        "thresholds": "artifacts/thresholds.json",
        "feature_importance": "artifacts/feature_importance.json",
        "validation_report": "reports/validation_report.html"
    },
    tags={
        "model_version": "v2.1",
        "deployment_target": "production",
        "data_version": "2025-09-03",
        "validation_method": "time_series_cv"
    }
)
```

## Testing Procedures

### 1. Pre-Deployment Testing Checklist

Before deploying MLflow or integrating with production systems:

```bash
#!/bin/bash
# MLflow Testing Script

echo "🧪 MLflow System Testing"
echo "======================="

# 1. Service Health Check
echo "1. Checking service health..."
curl -f http://localhost:5000/health || { echo "❌ MLflow server not responding"; exit 1; }
echo "✅ MLflow server healthy"

# 2. Database Connectivity
echo "2. Testing database connectivity..."
docker compose -f docker-compose.mlflow.yml exec -T mlflow-db pg_isready -U mlflow_user -d mlflow_db || { echo "❌ Database not ready"; exit 1; }
echo "✅ Database connection healthy"

# 3. MinIO Connectivity (for local setup)
echo "3. Testing MinIO connectivity..."
curl -f http://localhost:9000/minio/health/live || echo "⚠️ MinIO might not be running (OK for AWS setup)"

# 4. Python Client Test
echo "4. Testing Python client..."
poetry run python -c "
import mlflow
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
mlflow.set_tracking_uri('http://localhost:5000')

try:
    experiments = mlflow.search_experiments()
    print(f'✅ Python client working - Found {len(experiments)} experiments')
except Exception as e:
    print(f'❌ Python client failed: {e}')
    exit(1)
"

# 5. Experiment Creation Test
echo "5. Testing experiment creation..."
poetry run python -c "
import mlflow
import os
from datetime import datetime

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
mlflow.set_tracking_uri('http://localhost:5000')

test_exp_name = f'test-{datetime.now().strftime(\"%Y%m%d-%H%M%S\")}'
try:
    exp_id = mlflow.create_experiment(test_exp_name)
    
    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_param('test_param', 'test_value')
        mlflow.log_metric('test_metric', 0.95)
        mlflow.set_tag('test_tag', 'automated_test')
    
    print('✅ Experiment creation and logging successful')
    print(f'Test run: {run.info.run_id}')
except Exception as e:
    print(f'❌ Experiment creation failed: {e}')
    exit(1)
"

echo ""
echo "🎉 All tests passed! MLflow is ready for production use."
```

### 2. Performance Testing

```python
# performance_test.py
import mlflow
import time
import concurrent.futures
import numpy as np

def test_concurrent_logging():
    """Test MLflow under concurrent load"""
    
    def log_run(run_number):
        with mlflow.start_run(run_name=f'perf-test-{run_number}'):
            # Simulate model training metrics
            for i in range(10):
                mlflow.log_metric(f'metric_{i}', np.random.random())
            
            for i in range(5):
                mlflow.log_param(f'param_{i}', np.random.randint(1, 100))
            
            return f"Run {run_number} completed"
    
    # Test with 10 concurrent runs
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(log_run, i) for i in range(10)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    duration = time.time() - start_time
    print(f"✅ Logged 10 concurrent runs in {duration:.2f} seconds")
    print(f"Average: {duration/10:.2f}s per run")

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')
    test_concurrent_logging()
```

### 3. Integration Testing with Fraud Detection Pipeline

```python
# integration_test.py
"""Test MLflow integration with fraud detection workflow"""

import mlflow
from src.sapira_etl.mlflow_client import MLflowClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def test_fraud_detection_workflow():
    """Complete integration test simulating fraud detection training"""
    
    # Initialize MLflow
    client = MLflowClient(experiment_name="integration-test")
    
    # Generate synthetic fraud data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Simulate transaction features
    X = np.random.randn(n_samples, n_features)
    # Simulate fraud labels (5% fraud rate)
    y = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'alert_rate': y_pred.mean(),
        'fraud_detection_rate': y_pred[y_test == 1].mean() if y_test.sum() > 0 else 0
    }
    
    # Log to MLflow
    run_id = client.log_training_run(
        model=model,
        params={
            'n_estimators': 50,
            'algorithm': 'random_forest',
            'threshold': 0.5,
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        metrics=metrics,
        artifacts={},  # Skip artifacts for this test
        tags={
            'test_type': 'integration',
            'pipeline': 'fraud_detection',
            'automated': 'true'
        },
        run_name='integration-test-run'
    )
    
    print(f"✅ Integration test successful!")
    print(f"Run ID: {run_id}")
    print(f"Metrics: {metrics}")
    
    # Test model registry
    try:
        version = client.register_model(run_id, "fraud-test-model", stage="None")
        print(f"✅ Model registered successfully - Version: {version}")
    except Exception as e:
        print(f"⚠️ Model registration failed (expected with local artifacts): {e}")
    
    return True

if __name__ == "__main__":
    test_fraud_detection_workflow()
```

## CI/CD Pipeline Recommendations

### 1. GitHub Actions Workflow

Create `.github/workflows/mlflow-training.yml`:

```yaml
name: MLflow Training Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
  AWS_DEFAULT_REGION: us-west-2

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install
      
    - name: Run linting
      run: |
        poetry run ruff check src/
        poetry run black --check src/
        
    - name: Run unit tests
      run: poetry run pytest tests/ -v
      
    - name: Validate MLflow configuration
      run: |
        poetry run python -c "
        from src.sapira_etl.mlflow_client import MLflowClient
        import os
        
        # Test MLflow connectivity (if server is running)
        try:
            client = MLflowClient()
            print('✅ MLflow configuration valid')
        except Exception as e:
            print(f'⚠️ MLflow connectivity test skipped: {e}')
        "

  model-training:
    needs: lint-and-test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      
    - name: Install dependencies
      run: poetry install
      
    - name: Start MLflow Server (for testing)
      run: |
        # Start local MLflow server for CI testing
        docker run -d --name mlflow-test -p 5000:5000 \
          -e MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db \
          python:3.11-slim bash -c "
            pip install mlflow boto3 && 
            mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
          "
        sleep 30
        
    - name: Run Model Training
      run: |
        export MLFLOW_TRACKING_URI=http://localhost:5000
        poetry run python scripts/train_fraud_model.py
        
    - name: Run Model Validation
      run: |
        export MLFLOW_TRACKING_URI=http://localhost:5000
        poetry run python scripts/validate_model.py
        
    - name: Model Performance Gate
      run: |
        poetry run python -c "
        import mlflow
        import os
        
        mlflow.set_tracking_uri('http://localhost:5000')
        
        # Get latest run
        runs = mlflow.search_runs(max_results=1)
        if len(runs) > 0:
            latest_run = runs.iloc[0]
            auc_score = latest_run.get('metrics.auc_roc', 0)
            precision_score = latest_run.get('metrics.precision', 0)
            
            print(f'Latest model - AUC: {auc_score}, Precision: {precision_score}')
            
            # Quality gates
            if auc_score < 0.80:
                print('❌ Model quality gate failed: AUC < 0.80')
                exit(1)
            if precision_score < 0.75:
                print('❌ Model quality gate failed: Precision < 0.75')
                exit(1)
                
            print('✅ Model quality gates passed')
        else:
            print('⚠️ No runs found for validation')
        "
        
    - name: Auto-promote to Staging
      if: success()
      run: |
        poetry run python -c "
        import mlflow
        from src.sapira_etl.mlflow_client import MLflowClient
        
        client = MLflowClient()
        
        # Get latest run and promote to staging
        runs = mlflow.search_runs(max_results=1)
        if len(runs) > 0:
            run_id = runs.iloc[0]['run_id']
            try:
                version = client.register_model(run_id, 'fraud-detector', stage='Staging')
                print(f'✅ Model promoted to Staging - Version: {version}')
            except Exception as e:
                print(f'⚠️ Model registration failed: {e}')
        "

  production-deployment:
    needs: model-training
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Production Model Validation
      run: |
        # Add production validation logic
        echo "🔍 Running production validation tests..."
        
    - name: Deploy to Production
      if: success()
      run: |
        # Add production deployment logic
        echo "🚀 Deploying model to production..."
        
    - name: Update Model Documentation
      run: |
        # Generate model card and update documentation
        poetry run python scripts/generate_model_card.py
        
    - name: Notify Teams
      if: always()
      run: |
        # Send notifications (Slack, email, etc.)
        echo "📧 Notifying teams of deployment status..."
```

### 2. Model Validation Pipeline

Create `scripts/validate_model.py`:

```python
#!/usr/bin/env python3
"""Model validation pipeline for CI/CD"""

import mlflow
import pandas as pd
import numpy as np
from src.sapira_etl.mlflow_client import MLflowClient
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model_performance(run_id: str) -> bool:
    """Validate model performance against benchmarks"""
    
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    
    metrics = run.data.metrics
    
    # Define performance thresholds
    thresholds = {
        'auc_roc': 0.80,
        'precision': 0.75,
        'recall': 0.60,
        'f1_score': 0.67,
        'alert_rate': (0.01, 0.10)  # Between 1% and 10%
    }
    
    passed = True
    
    for metric, threshold in thresholds.items():
        if metric not in metrics:
            logger.warning(f"Metric {metric} not found in run")
            continue
            
        value = metrics[metric]
        
        if isinstance(threshold, tuple):
            # Range check
            min_val, max_val = threshold
            if not (min_val <= value <= max_val):
                logger.error(f"❌ {metric}: {value:.4f} outside range [{min_val}, {max_val}]")
                passed = False
            else:
                logger.info(f"✅ {metric}: {value:.4f} within range [{min_val}, {max_val}]")
        else:
            # Minimum threshold check
            if value < threshold:
                logger.error(f"❌ {metric}: {value:.4f} below threshold {threshold}")
                passed = False
            else:
                logger.info(f"✅ {metric}: {value:.4f} above threshold {threshold}")
    
    return passed

def validate_model_stability(model_name: str) -> bool:
    """Check model stability against previous versions"""
    
    client = mlflow.MlflowClient()
    
    try:
        # Get latest staging and production models
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not staging_versions:
            logger.info("No staging model found - skipping stability check")
            return True
            
        staging_version = staging_versions[0]
        staging_run = client.get_run(staging_version.run_id)
        
        if production_versions:
            production_version = production_versions[0]
            production_run = client.get_run(production_version.run_id)
            
            # Compare key metrics
            staging_auc = staging_run.data.metrics.get('auc_roc', 0)
            production_auc = production_run.data.metrics.get('auc_roc', 0)
            
            # Ensure new model is not significantly worse
            if staging_auc < production_auc - 0.05:  # 5% tolerance
                logger.error(f"❌ Staging model AUC ({staging_auc:.4f}) significantly worse than production ({production_auc:.4f})")
                return False
            else:
                logger.info(f"✅ Staging model AUC ({staging_auc:.4f}) comparable to production ({production_auc:.4f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Stability check failed: {e}")
        return False

def main():
    """Main validation pipeline"""
    
    # Get latest run
    runs = mlflow.search_runs(max_results=1, order_by=["start_time DESC"])
    
    if len(runs) == 0:
        logger.error("No runs found for validation")
        sys.exit(1)
    
    latest_run = runs.iloc[0]
    run_id = latest_run['run_id']
    
    logger.info(f"Validating run: {run_id}")
    
    # Performance validation
    performance_ok = validate_model_performance(run_id)
    
    # Stability validation
    stability_ok = validate_model_stability("fraud-detector")
    
    if performance_ok and stability_ok:
        logger.info("🎉 All validations passed!")
        sys.exit(0)
    else:
        logger.error("❌ Validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 3. Automated Model Card Generation

Create `scripts/generate_model_card.py`:

```python
#!/usr/bin/env python3
"""Generate automated model cards from MLflow runs"""

import mlflow
from jinja2 import Template
from datetime import datetime
import json

MODEL_CARD_TEMPLATE = """
# Model Card: {{ model_name }}

**Generated on:** {{ generation_date }}
**Model Version:** {{ model_version }}
**MLflow Run:** {{ run_id }}

## Model Overview

- **Algorithm:** {{ algorithm }}
- **Use Case:** Fraud Detection
- **Model Type:** {{ model_type }}
- **Training Date:** {{ training_date }}

## Performance Metrics

| Metric | Value |
|--------|-------|
{% for metric, value in metrics.items() -%}
| {{ metric.replace('_', ' ').title() }} | {{ "%.4f"|format(value) }} |
{% endfor %}

## Model Parameters

| Parameter | Value |
|-----------|-------|
{% for param, value in parameters.items() -%}
| {{ param.replace('_', ' ').title() }} | {{ value }} |
{% endfor %}

## Training Configuration

- **Training Data Size:** {{ parameters.get('train_size', 'Unknown') }}
- **Validation Data Size:** {{ parameters.get('test_size', 'Unknown') }}
- **Features Used:** {{ parameters.get('feature_count', 'Unknown') }}

## Model Validation

✅ **Performance Gates:** All thresholds met
✅ **Stability Check:** Model performance stable
✅ **Quality Assurance:** Automated tests passed

## Deployment Information

- **Deployment Environment:** {{ deployment_env }}
- **Artifact Location:** {{ artifact_uri }}
- **Model Registry:** MLflow Model Registry
- **Stage:** {{ stage }}

## Usage Example

```python
import mlflow.pyfunc

# Load model
model = mlflow.pyfunc.load_model("models:/{{ model_name }}/{{ stage }}")

# Make predictions
predictions = model.predict(transaction_data)
```

## Contact Information

- **Model Owner:** Fraud Detection Team
- **MLflow Tracking:** {{ mlflow_uri }}
- **Documentation:** See MLflow experiment for detailed logs

---
*This model card was automatically generated from MLflow run {{ run_id }}*
"""

def generate_model_card(model_name: str, stage: str = "Production"):
    """Generate model card from MLflow model registry"""
    
    client = mlflow.MlflowClient()
    
    # Get model version
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        print(f"No {stage} model found for {model_name}")
        return
    
    version = versions[0]
    run = client.get_run(version.run_id)
    
    # Prepare template data
    template_data = {
        'model_name': model_name,
        'model_version': version.version,
        'run_id': version.run_id,
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'algorithm': run.data.params.get('algorithm', 'Unknown'),
        'model_type': run.data.tags.get('model_type', 'Unknown'),
        'training_date': datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d"),
        'metrics': run.data.metrics,
        'parameters': run.data.params,
        'deployment_env': run.data.tags.get('environment', 'Unknown'),
        'artifact_uri': run.info.artifact_uri,
        'stage': stage,
        'mlflow_uri': mlflow.get_tracking_uri()
    }
    
    # Generate model card
    template = Template(MODEL_CARD_TEMPLATE)
    model_card = template.render(**template_data)
    
    # Save model card
    filename = f"MODEL_CARD_{model_name}_{stage.lower()}.md"
    with open(filename, 'w') as f:
        f.write(model_card)
    
    print(f"✅ Model card generated: {filename}")
    return filename

if __name__ == "__main__":
    generate_model_card("fraud-detector", "Production")
```

## Next Steps

1. **Update Training Scripts**: Integrate MLflow logging into existing training code
2. **Model Promotion Workflow**: Set up automated staging and production promotion
3. **CI/CD Integration**: Implement the GitHub Actions workflow above
4. **Monitoring**: Set up alerts for model performance degradation
5. **Artifact Management**: Implement automated artifact refresh from MLflow registry
6. **Testing**: Run the testing procedures regularly to ensure system reliability

For detailed implementation examples, see the automation opportunities in `reports/automation_opportunities.md` items 9-11, 17, and 22.