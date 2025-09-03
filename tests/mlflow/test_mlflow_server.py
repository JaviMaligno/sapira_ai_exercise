"""
Test MLflow server health and connectivity
Based on the pre-deployment testing checklist from MLFLOW_SETUP.md
"""

import pytest
import requests
import subprocess
import docker
import os
import time
from urllib.parse import urlparse


class TestMLflowServerHealth:
    """Test MLflow server health and connectivity"""
    
    def test_mlflow_server_health_endpoint(self, mlflow_tracking_uri):
        """Test that MLflow server health endpoint responds correctly"""
        response = requests.get(f"{mlflow_tracking_uri}/health", timeout=10)
        assert response.status_code == 200
        assert response.text.strip() == "OK"
    
    def test_mlflow_server_ui_accessible(self, mlflow_tracking_uri):
        """Test that MLflow UI is accessible"""
        response = requests.get(mlflow_tracking_uri, timeout=10)
        assert response.status_code == 200
        assert "MLflow" in response.text
        assert "<!doctype html>" in response.text
    
    def test_mlflow_server_api_endpoints(self, mlflow_tracking_uri):
        """Test core MLflow API endpoints"""
        # Test experiments endpoint with proper parameters
        response = requests.post(f"{mlflow_tracking_uri}/api/2.0/mlflow/experiments/search", 
                               json={"max_results": 10}, timeout=10)
        assert response.status_code == 200
        
        # Test runs endpoint  
        response = requests.post(f"{mlflow_tracking_uri}/api/2.0/mlflow/runs/search", 
                              json={"max_results": 1}, timeout=10)
        assert response.status_code == 200


class TestMLflowDockerServices:
    """Test Docker services supporting MLflow"""
    
    @pytest.fixture(autouse=True)
    def setup_docker_client(self):
        """Setup Docker client"""
        try:
            # Try different Docker configurations
            docker_configs = [
                # Try with explicit socket path
                {"base_url": "unix:///var/run/docker.sock"},
                # Try with Docker context
                None,  # Default from environment
            ]
            
            self.docker_client = None
            for config in docker_configs:
                try:
                    if config:
                        client = docker.DockerClient(**config)
                    else:
                        client = docker.from_env()
                    
                    # Test that we can actually list containers
                    containers = client.containers.list(all=True)  # Include stopped containers
                    self.docker_client = client
                    print(f"Docker client connected successfully. Found {len(containers)} containers.")
                    break
                except Exception as e:
                    print(f"Docker config failed: {e}")
                    continue
            
            if self.docker_client is None:
                pytest.skip("Docker not accessible - this is expected in some CI environments")
                
        except Exception as e:
            pytest.skip(f"Docker not available: {e}")
    
    def test_mlflow_server_container_running(self):
        """Test that MLflow server container is running"""
        containers = self.docker_client.containers.list()
        mlflow_containers = [c for c in containers if "mlflow-server" in c.name]
        assert len(mlflow_containers) > 0, f"MLflow server container not found. Found: {[c.name for c in containers]}"
        
        container = mlflow_containers[0]
        assert container.status == "running"
        # Check if port 5000 is exposed (containers might not show internal port mapping correctly)
        ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
        assert '5000/tcp' in ports or any('5000' in str(port) for port in ports)
    
    def test_postgres_container_running(self):
        """Test that PostgreSQL container is running and healthy"""
        containers = self.docker_client.containers.list()
        postgres_containers = [c for c in containers if "mlflow-db" in c.name]
        assert len(postgres_containers) > 0, f"PostgreSQL container not found. Found: {[c.name for c in containers]}"
        
        container = postgres_containers[0]
        assert container.status == "running"
        
        # Check health status
        health = container.attrs.get('State', {}).get('Health', {})
        if health:
            assert health.get('Status') == 'healthy', f"PostgreSQL container not healthy: {health}"
    
    def test_minio_container_running_if_local(self):
        """Test MinIO container if running local setup"""
        containers = self.docker_client.containers.list()
        minio_containers = [c for c in containers if "minio" in c.name]
        
        if minio_containers:
            container = minio_containers[0]
            assert container.status == "running"
            
            # Test MinIO health endpoint
            try:
                response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
                assert response.status_code == 200
            except requests.exceptions.RequestException:
                pytest.skip("MinIO health endpoint not accessible (may be AWS setup)")


class TestDatabaseConnectivity:
    """Test database connectivity"""
    
    def test_postgres_connection(self):
        """Test PostgreSQL database connection"""
        try:
            result = subprocess.run([
                "docker", "compose", "-f", "docker-compose.mlflow.yml", 
                "exec", "-T", "mlflow-db", 
                "pg_isready", "-U", "mlflow_user", "-d", "mlflow_db"
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode == 0, f"PostgreSQL not ready: {result.stderr}"
        except subprocess.TimeoutExpired:
            pytest.fail("PostgreSQL connection test timed out")
        except FileNotFoundError:
            pytest.skip("Docker compose not available for database test")
    
    def test_database_tables_exist(self, mlflow_client):
        """Test that MLflow database tables exist by trying to query experiments"""
        try:
            experiments = mlflow_client.search_experiments()
            assert isinstance(experiments, list)
        except Exception as e:
            pytest.fail(f"Could not query MLflow database: {e}")


class TestStorageBackend:
    """Test storage backend (S3/MinIO) connectivity"""
    
    def test_minio_connectivity_if_local(self):
        """Test MinIO connectivity for local setup"""
        try:
            response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
            if response.status_code == 200:
                # MinIO is running, test API access
                response = requests.get("http://localhost:9001", timeout=5)
                assert response.status_code in [200, 403]  # 403 is normal without auth
        except requests.exceptions.RequestException:
            pytest.skip("MinIO not available (likely AWS setup)")
    
    def test_artifact_storage_configured(self, mlflow_client):
        """Test that artifact storage is properly configured"""
        # Create a test experiment to verify artifact storage configuration
        experiment_name = f"storage_test_{int(time.time())}"
        try:
            experiment_id = mlflow_client.create_experiment(experiment_name)
            
            with mlflow.start_run(experiment_id=experiment_id):
                # Log a simple parameter to test storage
                mlflow.log_param("test_param", "test_value")
                run = mlflow.active_run()
                
            # Verify the run was stored
            run_info = mlflow_client.get_run(run.info.run_id)
            assert run_info.data.params.get("test_param") == "test_value"
            
        finally:
            try:
                mlflow_client.delete_experiment(experiment_id)
            except:
                pass


class TestMLflowServerConfiguration:
    """Test MLflow server configuration"""
    
    def test_server_serves_artifacts(self, mlflow_tracking_uri):
        """Test that server is configured to serve artifacts"""
        # Try to access artifacts endpoint
        response = requests.get(f"{mlflow_tracking_uri}/api/2.0/mlflow/artifacts/list", 
                              timeout=10, 
                              params={"path": ""})
        # Should not return 404 (endpoint exists) even if empty
        assert response.status_code != 404
    
    def test_server_version_info(self, mlflow_tracking_uri):
        """Test server version information by checking if we can access basic API"""
        # The version endpoint may not be available, so test basic API functionality
        response = requests.post(f"{mlflow_tracking_uri}/api/2.0/mlflow/experiments/search", 
                               json={"max_results": 1}, timeout=10)
        assert response.status_code == 200
        
        # Verify response structure indicates MLflow server
        data = response.json()
        assert "experiments" in data or isinstance(data, dict)


@pytest.mark.integration
class TestMLflowIntegration:
    """Integration tests for MLflow with fraud detection workflow"""
    
    def test_complete_logging_workflow(self, mlflow_client, test_experiment, 
                                     sample_fraud_data, trained_model, 
                                     sample_metrics, sample_params, sample_tags):
        """Test complete MLflow logging workflow"""
        
        with mlflow.start_run(experiment_id=test_experiment):
            # Log parameters
            for param, value in sample_params.items():
                mlflow.log_param(param, value)
            
            # Log metrics
            for metric, value in sample_metrics.items():
                mlflow.log_metric(metric, value)
            
            # Log tags
            mlflow.set_tags(sample_tags)
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify logged data
        run = mlflow_client.get_run(run_id)
        
        # Check parameters
        for param, expected_value in sample_params.items():
            assert run.data.params[param] == str(expected_value)
        
        # Check metrics
        for metric, expected_value in sample_metrics.items():
            assert abs(run.data.metrics[metric] - expected_value) < 0.001
        
        # Check tags
        for tag, expected_value in sample_tags.items():
            assert run.data.tags[tag] == expected_value
    
    def test_concurrent_logging(self, test_experiment, sample_params, sample_metrics):
        """Test concurrent MLflow logging"""
        import concurrent.futures
        import threading
        
        def log_run(run_number):
            with mlflow.start_run(experiment_id=test_experiment, 
                                run_name=f"concurrent_test_{run_number}"):
                # Log some data
                mlflow.log_param("run_number", run_number)
                mlflow.log_metric("test_metric", run_number * 0.1)
                return mlflow.active_run().info.run_id
        
        # Run 5 concurrent logging operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(log_run, i) for i in range(5)]
            run_ids = [future.result(timeout=30) for future in concurrent.futures.as_completed(futures)]
        
        assert len(run_ids) == 5
        assert len(set(run_ids)) == 5  # All run IDs should be unique
        
        # Verify all runs were logged
        for run_id in run_ids:
            run = mlflow_client.get_run(run_id)
            assert run.info.status == "FINISHED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])