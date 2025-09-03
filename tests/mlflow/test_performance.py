"""
MLflow performance tests
Based on the performance testing from MLFLOW_SETUP.md
"""

import pytest
import mlflow
import time
import concurrent.futures
import numpy as np
import requests
import json
from datetime import datetime, timedelta


class TestMLflowPerformance:
    """Test MLflow performance under various loads"""
    
    def test_single_run_logging_performance(self, test_experiment):
        """Test performance of logging single run with many metrics"""
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=test_experiment, 
                            run_name="performance_test_single"):
            # Log many parameters
            for i in range(100):
                mlflow.log_param(f'param_{i}', f'value_{i}')
            
            # Log many metrics
            for i in range(100):
                mlflow.log_metric(f'metric_{i}', np.random.random())
            
            # Log tags
            mlflow.set_tags({
                'performance_test': 'single_run',
                'metrics_count': '100',
                'params_count': '100'
            })
        
        duration = time.time() - start_time
        
        # Should complete within 30 seconds (generous threshold)
        assert duration < 30, f"Single run logging took {duration:.2f}s (too slow)"
        print(f"✅ Single run with 200 items logged in {duration:.2f}s")
    
    def test_concurrent_run_logging_performance(self, test_experiment):
        """Test performance of concurrent run logging"""
        def log_run(run_number):
            """Log a single run with fraud detection metrics"""
            start_time = time.time()
            
            with mlflow.start_run(experiment_id=test_experiment,
                                run_name=f'perf_test_{run_number}'):
                # Log fraud detection parameters
                mlflow.log_param('algorithm', 'lightgbm')
                mlflow.log_param('threshold', np.random.uniform(0.3, 0.7))
                mlflow.log_param('n_estimators', np.random.randint(50, 200))
                mlflow.log_param('run_number', run_number)
                
                # Log fraud detection metrics
                mlflow.log_metric('auc_roc', np.random.uniform(0.7, 0.95))
                mlflow.log_metric('precision', np.random.uniform(0.6, 0.9))
                mlflow.log_metric('recall', np.random.uniform(0.5, 0.85))
                mlflow.log_metric('f1_score', np.random.uniform(0.55, 0.87))
                mlflow.log_metric('alert_rate', np.random.uniform(0.01, 0.1))
                
                # Log business metrics
                mlflow.log_metric('daily_alerts', np.random.randint(100, 500))
                mlflow.log_metric('cost_per_alert', np.random.uniform(1.5, 4.0))
                
                # Log tags
                mlflow.set_tags({
                    'model_family': 'fraud-detection',
                    'environment': 'performance_test',
                    'test_run': 'concurrent'
                })
            
            duration = time.time() - start_time
            return run_number, duration
        
        # Test with 10 concurrent runs
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(log_run, i) for i in range(10)]
            results = [future.result(timeout=60) for future in concurrent.futures.as_completed(futures)]
        
        total_duration = time.time() - start_time
        
        # Verify all runs completed
        assert len(results) == 10
        
        # Calculate statistics
        run_durations = [duration for _, duration in results]
        avg_duration = sum(run_durations) / len(run_durations)
        max_duration = max(run_durations)
        
        print(f"✅ Concurrent logging: {len(results)} runs in {total_duration:.2f}s")
        print(f"   Average per run: {avg_duration:.2f}s, Max: {max_duration:.2f}s")
        
        # Performance assertions
        assert total_duration < 120, f"Concurrent logging took too long: {total_duration:.2f}s"
        assert avg_duration < 10, f"Average run duration too slow: {avg_duration:.2f}s"
    
    def test_large_metric_logging_performance(self, test_experiment):
        """Test logging performance with many metrics over time"""
        with mlflow.start_run(experiment_id=test_experiment,
                            run_name="large_metrics_test"):
            start_time = time.time()
            
            # Simulate logging metrics over time (like training epochs)
            for epoch in range(50):
                # Log multiple metrics per epoch
                mlflow.log_metric('train_loss', np.random.exponential(0.5), step=epoch)
                mlflow.log_metric('val_loss', np.random.exponential(0.6), step=epoch)
                mlflow.log_metric('train_auc', np.random.uniform(0.7, 0.95), step=epoch)
                mlflow.log_metric('val_auc', np.random.uniform(0.65, 0.92), step=epoch)
                mlflow.log_metric('precision', np.random.uniform(0.6, 0.9), step=epoch)
                mlflow.log_metric('recall', np.random.uniform(0.5, 0.85), step=epoch)
            
            duration = time.time() - start_time
        
        print(f"✅ Large metrics logging: 300 metric points in {duration:.2f}s")
        assert duration < 45, f"Large metrics logging took too long: {duration:.2f}s"
    
    @pytest.mark.slow
    def test_server_response_time_under_load(self, mlflow_tracking_uri):
        """Test server response times under load"""
        def make_api_request():
            """Make a simple API request and measure response time"""
            start_time = time.time()
            response = requests.post(f"{mlflow_tracking_uri}/api/2.0/mlflow/experiments/search",
                                  json={"max_results": 10})
            duration = time.time() - start_time
            return response.status_code, duration
        
        # Test sequential requests
        sequential_times = []
        for _ in range(10):
            status_code, duration = make_api_request()
            assert status_code == 200
            sequential_times.append(duration)
        
        # Test concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_api_request) for _ in range(20)]
            concurrent_results = [future.result(timeout=30) for future in concurrent.futures.as_completed(futures)]
        
        concurrent_times = [duration for status_code, duration in concurrent_results if status_code == 200]
        
        # Calculate statistics
        avg_sequential = sum(sequential_times) / len(sequential_times)
        avg_concurrent = sum(concurrent_times) / len(concurrent_times)
        max_concurrent = max(concurrent_times)
        
        print(f"✅ API Response times:")
        print(f"   Sequential avg: {avg_sequential:.3f}s")
        print(f"   Concurrent avg: {avg_concurrent:.3f}s, max: {max_concurrent:.3f}s")
        
        # Performance assertions
        assert avg_sequential < 2.0, f"Sequential API too slow: {avg_sequential:.3f}s"
        assert avg_concurrent < 5.0, f"Concurrent API too slow: {avg_concurrent:.3f}s"
        assert max_concurrent < 10.0, f"Max API response too slow: {max_concurrent:.3f}s"


class TestMLflowScalability:
    """Test MLflow scalability with larger datasets"""
    
    def test_experiment_with_many_runs(self, mlflow_client):
        """Test experiment with many runs"""
        experiment_name = f"scalability_test_{int(time.time())}"
        experiment_id = mlflow.create_experiment(experiment_name)
        
        try:
            start_time = time.time()
            
            # Create many runs quickly
            run_ids = []
            for i in range(25):  # Reduced from 100 for CI performance
                with mlflow.start_run(experiment_id=experiment_id, 
                                    run_name=f"scale_run_{i}"):
                    mlflow.log_param('run_index', i)
                    mlflow.log_metric('test_metric', i * 0.1)
                    run_ids.append(mlflow.active_run().info.run_id)
            
            creation_time = time.time() - start_time
            
            # Test querying all runs
            query_start = time.time()
            runs = mlflow_client.search_runs([experiment_id], max_results=50)
            query_time = time.time() - query_start
            
            print(f"✅ Scalability test: {len(run_ids)} runs created in {creation_time:.2f}s")
            print(f"   Query time for {len(runs)} runs: {query_time:.3f}s")
            
            # Assertions
            assert len(run_ids) == 25
            assert len(runs) >= 25
            assert creation_time < 60, f"Run creation too slow: {creation_time:.2f}s"
            assert query_time < 10, f"Run query too slow: {query_time:.3f}s"
            
        finally:
            # Cleanup
            mlflow_client.delete_experiment(experiment_id)
    
    def test_large_parameter_values(self, test_experiment):
        """Test logging large parameter values"""
        large_param_value = "x" * 1000  # 1KB string
        large_json_config = json.dumps({
            "model_config": {f"param_{i}": f"value_{i}" for i in range(100)},
            "training_data": [i for i in range(100)],
            "description": "This is a large configuration parameter for testing"
        })
        
        start_time = time.time()
        
        with mlflow.start_run(experiment_id=test_experiment, 
                            run_name="large_params_test"):
            mlflow.log_param('large_string', large_param_value)
            mlflow.log_param('large_json', large_json_config)
            mlflow.log_param('normal_param', 'normal_value')
            
            # Log some metrics too
            mlflow.log_metric('test_metric', 0.85)
        
        duration = time.time() - start_time
        
        print(f"✅ Large parameters logged in {duration:.2f}s")
        assert duration < 15, f"Large parameter logging too slow: {duration:.2f}s"


class TestMLflowMemoryUsage:
    """Test memory usage patterns"""
    
    @pytest.mark.slow
    def test_memory_usage_with_many_metrics(self, test_experiment):
        """Test memory usage doesn't grow excessively with many metrics"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with mlflow.start_run(experiment_id=test_experiment, 
                            run_name="memory_test"):
            for i in range(500):
                mlflow.log_metric(f'metric_{i}', np.random.random())
                
                # Check memory every 100 metrics
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Memory shouldn't grow excessively (allow up to 100MB increase)
                    assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f}MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"✅ Memory test: {total_increase:.1f}MB increase for 500 metrics")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])