import torch
import torch.nn as nn
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import threading
from dataclasses import dataclass

# Data Classes for Monitoring
@dataclass
class PredictionMetrics:
    """Metrics for a single prediction"""
    timestamp: float
    inference_time_ms: float
    input_shape: tuple
    prediction_confidence: float
    predicted_class: int
    memory_usage_mb: float
    
@dataclass
class ModelHealthMetrics:
    """Overall model health metrics"""
    timestamp: float
    avg_inference_time_ms: float
    throughput_per_second: float
    error_rate: float
    avg_confidence: float
    memory_usage_mb: float
    cpu_usage_percent: float
    predictions_count: int

# Production Model Wrapper
class ProductionModel:
    """Wrapper for production model with monitoring"""
    
    def __init__(self, model: nn.Module, model_name: str = "model"):
        self.model = model
        self.model_name = model_name
        self.model.eval()
        
        # Monitoring data
        self.prediction_history = deque(maxlen=1000)
        self.error_log = deque(maxlen=100)
        self.performance_metrics = deque(maxlen=100)
        
        # Statistics
        self.total_predictions = 0
        self.total_errors = 0
        self.start_time = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
    
    def predict(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, PredictionMetrics]:
        """Make prediction with monitoring"""
        
        start_time = time.time()
        
        try:
            # Memory before prediction
            memory_before = self._get_memory_usage()
            
            # Run prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = torch.max(probabilities, dim=1)[0].item()
                predicted_class = torch.argmax(output, dim=1).item()
            
            # Calculate metrics
            inference_time = (time.time() - start_time) * 1000
            memory_after = self._get_memory_usage()
            
            metrics = PredictionMetrics(
                timestamp=time.time(),
                inference_time_ms=inference_time,
                input_shape=tuple(input_tensor.shape),
                prediction_confidence=confidence,
                predicted_class=predicted_class,
                memory_usage_mb=memory_after
            )
            
            # Update monitoring data
            with self.lock:
                self.prediction_history.append(metrics)
                self.total_predictions += 1
            
            return output, metrics
            
        except Exception as e:
            # Log error
            error_info = {
                'timestamp': time.time(),
                'error': str(e),
                'input_shape': tuple(input_tensor.shape) if input_tensor is not None else None
            }
            
            with self.lock:
                self.error_log.append(error_info)
                self.total_errors += 1
            
            raise e
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        else:
            # For CPU, this is a simplified estimate
            return 0.0
    
    def get_health_metrics(self) -> ModelHealthMetrics:
        """Get current model health metrics"""
        
        with self.lock:
            if not self.prediction_history:
                return ModelHealthMetrics(
                    timestamp=time.time(),
                    avg_inference_time_ms=0.0,
                    throughput_per_second=0.0,
                    error_rate=0.0,
                    avg_confidence=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    predictions_count=0
                )
            
            # Calculate metrics from recent history
            recent_metrics = list(self.prediction_history)
            
            avg_inference_time = np.mean([m.inference_time_ms for m in recent_metrics])
            avg_confidence = np.mean([m.prediction_confidence for m in recent_metrics])
            current_memory = recent_metrics[-1].memory_usage_mb if recent_metrics else 0.0
            
            # Calculate throughput
            time_window = 60  # Last 60 seconds
            recent_time = time.time() - time_window
            recent_predictions = [m for m in recent_metrics if m.timestamp > recent_time]
            throughput = len(recent_predictions) / time_window if recent_predictions else 0.0
            
            # Error rate
            total_operations = self.total_predictions + self.total_errors
            error_rate = self.total_errors / total_operations if total_operations > 0 else 0.0
            
            return ModelHealthMetrics(
                timestamp=time.time(),
                avg_inference_time_ms=avg_inference_time,
                throughput_per_second=throughput,
                error_rate=error_rate,
                avg_confidence=avg_confidence,
                memory_usage_mb=current_memory,
                cpu_usage_percent=0.0,  # Would need psutil for actual CPU usage
                predictions_count=len(recent_metrics)
            )

# Performance Monitor
class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, model: ProductionModel):
        self.model = model
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=1000)
        
        # Alerting thresholds
        self.thresholds = {
            'max_inference_time_ms': 1000,
            'min_throughput_per_second': 1.0,
            'max_error_rate': 0.05,
            'min_confidence': 0.5,
            'max_memory_usage_mb': 2048
        }
        
        # Alerts
        self.alerts = deque(maxlen=50)
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring"""
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,)
        )
        self.monitoring_thread.start()
        print(f"✓ Monitoring started for {self.model.model_name}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print(f"✓ Monitoring stopped for {self.model.model_name}")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect metrics
                health_metrics = self.model.get_health_metrics()
                self.metrics_history.append(health_metrics)
                
                # Check for alerts
                self._check_alerts(health_metrics)
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def _check_alerts(self, metrics: ModelHealthMetrics):
        """Check metrics against thresholds and generate alerts"""
        
        alerts_triggered = []
        
        # Check inference time
        if metrics.avg_inference_time_ms > self.thresholds['max_inference_time_ms']:
            alerts_triggered.append({
                'type': 'HIGH_LATENCY',
                'message': f"High inference time: {metrics.avg_inference_time_ms:.1f}ms",
                'severity': 'WARNING',
                'timestamp': metrics.timestamp
            })
        
        # Check throughput
        if metrics.throughput_per_second < self.thresholds['min_throughput_per_second']:
            alerts_triggered.append({
                'type': 'LOW_THROUGHPUT',
                'message': f"Low throughput: {metrics.throughput_per_second:.2f} req/sec",
                'severity': 'WARNING',
                'timestamp': metrics.timestamp
            })
        
        # Check error rate
        if metrics.error_rate > self.thresholds['max_error_rate']:
            alerts_triggered.append({
                'type': 'HIGH_ERROR_RATE',
                'message': f"High error rate: {metrics.error_rate:.2%}",
                'severity': 'CRITICAL',
                'timestamp': metrics.timestamp
            })
        
        # Check confidence
        if metrics.avg_confidence < self.thresholds['min_confidence']:
            alerts_triggered.append({
                'type': 'LOW_CONFIDENCE',
                'message': f"Low prediction confidence: {metrics.avg_confidence:.2f}",
                'severity': 'WARNING',
                'timestamp': metrics.timestamp
            })
        
        # Check memory usage
        if metrics.memory_usage_mb > self.thresholds['max_memory_usage_mb']:
            alerts_triggered.append({
                'type': 'HIGH_MEMORY_USAGE',
                'message': f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                'severity': 'WARNING',
                'timestamp': metrics.timestamp
            })
        
        # Store alerts
        for alert in alerts_triggered:
            self.alerts.append(alert)
            print(f"🚨 ALERT [{alert['severity']}]: {alert['message']}")
    
    def get_performance_report(self, window_hours: int = 24) -> Dict[str, Any]:
        """Generate performance report"""
        
        cutoff_time = time.time() - (window_hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time window"}
        
        # Calculate statistics
        inference_times = [m.avg_inference_time_ms for m in recent_metrics]
        throughputs = [m.throughput_per_second for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        confidences = [m.avg_confidence for m in recent_metrics]
        
        report = {
            'time_window_hours': window_hours,
            'total_data_points': len(recent_metrics),
            'inference_time': {
                'mean_ms': np.mean(inference_times),
                'p50_ms': np.percentile(inference_times, 50),
                'p95_ms': np.percentile(inference_times, 95),
                'p99_ms': np.percentile(inference_times, 99),
                'max_ms': np.max(inference_times)
            },
            'throughput': {
                'mean_req_per_sec': np.mean(throughputs),
                'min_req_per_sec': np.min(throughputs),
                'max_req_per_sec': np.max(throughputs)
            },
            'error_rate': {
                'mean': np.mean(error_rates),
                'max': np.max(error_rates)
            },
            'prediction_confidence': {
                'mean': np.mean(confidences),
                'min': np.min(confidences)
            },
            'alerts_count': len([a for a in self.alerts if a['timestamp'] > cutoff_time])
        }
        
        return report

# Data Drift Monitor
class DataDriftMonitor:
    """Monitor for data drift in production"""
    
    def __init__(self, reference_stats: Dict[str, Any] = None):
        self.reference_stats = reference_stats or {}
        self.current_batch_stats = []
        self.drift_scores = deque(maxlen=100)
        
    def update_reference_stats(self, reference_data: torch.Tensor):
        """Update reference statistics from training data"""
        
        with torch.no_grad():
            self.reference_stats = {
                'mean': torch.mean(reference_data, dim=(0, 2, 3)).cpu().numpy(),
                'std': torch.std(reference_data, dim=(0, 2, 3)).cpu().numpy(),
                'min': torch.min(reference_data).item(),
                'max': torch.max(reference_data).item()
            }
        
        print("✓ Reference statistics updated")
    
    def check_drift(self, batch_data: torch.Tensor, threshold: float = 0.1) -> Dict[str, Any]:
        """Check for data drift in current batch"""
        
        if not self.reference_stats:
            return {"error": "No reference statistics available"}
        
        with torch.no_grad():
            # Calculate current batch statistics
            current_stats = {
                'mean': torch.mean(batch_data, dim=(0, 2, 3)).cpu().numpy(),
                'std': torch.std(batch_data, dim=(0, 2, 3)).cpu().numpy(),
                'min': torch.min(batch_data).item(),
                'max': torch.max(batch_data).item()
            }
            
            # Calculate drift scores
            mean_drift = np.mean(np.abs(current_stats['mean'] - self.reference_stats['mean']))
            std_drift = np.mean(np.abs(current_stats['std'] - self.reference_stats['std']))
            range_drift = abs((current_stats['max'] - current_stats['min']) - 
                            (self.reference_stats['max'] - self.reference_stats['min']))
            
            overall_drift = (mean_drift + std_drift + range_drift) / 3
            
            drift_result = {
                'timestamp': time.time(),
                'overall_drift_score': overall_drift,
                'mean_drift': mean_drift,
                'std_drift': std_drift,
                'range_drift': range_drift,
                'drift_detected': overall_drift > threshold,
                'threshold': threshold
            }
            
            self.drift_scores.append(drift_result)
            
            return drift_result

# Model A/B Testing Monitor
class ABTestMonitor:
    """Monitor A/B testing between model versions"""
    
    def __init__(self):
        self.model_metrics = defaultdict(lambda: {
            'predictions': [],
            'accuracies': [],
            'inference_times': [],
            'confidences': []
        })
    
    def record_prediction(self, model_name: str, 
                         inference_time_ms: float,
                         confidence: float,
                         is_correct: bool = None):
        """Record prediction for A/B testing"""
        
        metrics = self.model_metrics[model_name]
        
        metrics['inference_times'].append(inference_time_ms)
        metrics['confidences'].append(confidence)
        
        if is_correct is not None:
            metrics['accuracies'].append(1.0 if is_correct else 0.0)
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """Generate A/B test comparison report"""
        
        report = {}
        
        for model_name, metrics in self.model_metrics.items():
            if not metrics['inference_times']:
                continue
                
            model_report = {
                'total_predictions': len(metrics['inference_times']),
                'avg_inference_time_ms': np.mean(metrics['inference_times']),
                'avg_confidence': np.mean(metrics['confidences']),
                'p95_inference_time_ms': np.percentile(metrics['inference_times'], 95)
            }
            
            if metrics['accuracies']:
                model_report['accuracy'] = np.mean(metrics['accuracies'])
            
            report[model_name] = model_report
        
        return report

# Logging and Alerting
class ProductionLogger:
    """Production logging system"""
    
    def __init__(self, log_file: str = "model_production.log"):
        self.logger = logging.getLogger("ModelProduction")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_prediction(self, model_name: str, metrics: PredictionMetrics):
        """Log prediction details"""
        
        self.logger.info(f"Prediction - Model: {model_name}, "
                        f"Time: {metrics.inference_time_ms:.2f}ms, "
                        f"Confidence: {metrics.prediction_confidence:.3f}")
    
    def log_error(self, model_name: str, error: str, context: Dict[str, Any] = None):
        """Log error details"""
        
        context_str = json.dumps(context) if context else ""
        self.logger.error(f"Error - Model: {model_name}, "
                         f"Error: {error}, Context: {context_str}")
    
    def log_alert(self, alert: Dict[str, Any]):
        """Log alert"""
        
        self.logger.warning(f"Alert - Type: {alert['type']}, "
                           f"Message: {alert['message']}, "
                           f"Severity: {alert['severity']}")

if __name__ == "__main__":
    print("Production Model Monitoring")
    print("=" * 32)
    
    # Create sample model
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    model = SampleModel()
    
    print("\n1. Production Model Setup")
    print("-" * 29)
    
    # Wrap model for production
    production_model = ProductionModel(model, "sample_classifier")
    
    # Setup monitoring
    monitor = ModelPerformanceMonitor(production_model)
    monitor.start_monitoring(interval_seconds=5)  # Monitor every 5 seconds
    
    # Setup logging
    logger = ProductionLogger("demo_production.log")
    
    print("✓ Production model and monitoring setup complete")
    
    print("\n2. Simulating Production Traffic")
    print("-" * 38)
    
    # Simulate production predictions
    for i in range(20):
        try:
            # Create random input
            input_tensor = torch.randn(1, 3, 32, 32)
            
            # Make prediction
            output, metrics = production_model.predict(input_tensor)
            
            # Log prediction
            logger.log_prediction("sample_classifier", metrics)
            
            print(f"Prediction {i+1}: {metrics.inference_time_ms:.2f}ms, "
                  f"confidence: {metrics.prediction_confidence:.3f}")
            
            # Simulate some delay
            time.sleep(0.1)
            
        except Exception as e:
            logger.log_error("sample_classifier", str(e))
    
    # Wait a bit for monitoring to collect data
    time.sleep(6)
    
    print("\n3. Health Metrics")
    print("-" * 19)
    
    # Get current health metrics
    health = production_model.get_health_metrics()
    
    print(f"Model Health Report:")
    print(f"  Average inference time: {health.avg_inference_time_ms:.2f} ms")
    print(f"  Throughput: {health.throughput_per_second:.2f} predictions/sec")
    print(f"  Error rate: {health.error_rate:.2%}")
    print(f"  Average confidence: {health.avg_confidence:.3f}")
    print(f"  Memory usage: {health.memory_usage_mb:.1f} MB")
    print(f"  Total predictions: {health.predictions_count}")
    
    print("\n4. Performance Report")
    print("-" * 24)
    
    # Generate performance report
    performance_report = monitor.get_performance_report(window_hours=1)  # Last hour
    
    if "error" not in performance_report:
        print("Performance Report (Last Hour):")
        print(f"  Data points: {performance_report['total_data_points']}")
        print(f"  Mean inference time: {performance_report['inference_time']['mean_ms']:.2f} ms")
        print(f"  P95 inference time: {performance_report['inference_time']['p95_ms']:.2f} ms")
        print(f"  Mean throughput: {performance_report['throughput']['mean_req_per_sec']:.2f} req/sec")
        print(f"  Error rate: {performance_report['error_rate']['mean']:.2%}")
        print(f"  Alerts triggered: {performance_report['alerts_count']}")
    else:
        print(f"Report error: {performance_report['error']}")
    
    print("\n5. Data Drift Monitoring")
    print("-" * 30)
    
    # Setup data drift monitoring
    drift_monitor = DataDriftMonitor()
    
    # Create reference data (simulating training data)
    reference_data = torch.randn(100, 3, 32, 32)
    drift_monitor.update_reference_stats(reference_data)
    
    # Test with similar data (no drift)
    similar_data = torch.randn(10, 3, 32, 32)
    drift_result_1 = drift_monitor.check_drift(similar_data)
    
    print(f"Similar data drift check:")
    print(f"  Drift score: {drift_result_1['overall_drift_score']:.4f}")
    print(f"  Drift detected: {drift_result_1['drift_detected']}")
    
    # Test with different data (potential drift)
    different_data = torch.randn(10, 3, 32, 32) * 2 + 1  # Different distribution
    drift_result_2 = drift_monitor.check_drift(different_data)
    
    print(f"Different data drift check:")
    print(f"  Drift score: {drift_result_2['overall_drift_score']:.4f}")
    print(f"  Drift detected: {drift_result_2['drift_detected']}")
    
    print("\n6. A/B Testing Monitor")
    print("-" * 26)
    
    # Setup A/B testing
    ab_monitor = ABTestMonitor()
    
    # Simulate A/B test data
    for i in range(50):
        # Model A - faster but less accurate
        ab_monitor.record_prediction("model_a", 
                                    inference_time_ms=10 + np.random.normal(0, 2),
                                    confidence=0.7 + np.random.normal(0, 0.1),
                                    is_correct=np.random.random() > 0.15)  # 85% accuracy
        
        # Model B - slower but more accurate
        ab_monitor.record_prediction("model_b",
                                    inference_time_ms=15 + np.random.normal(0, 3),
                                    confidence=0.8 + np.random.normal(0, 0.1),
                                    is_correct=np.random.random() > 0.10)  # 90% accuracy
    
    # Get A/B test report
    ab_report = ab_monitor.get_comparison_report()
    
    print("A/B Test Comparison:")
    print("-" * 20)
    for model_name, metrics in ab_report.items():
        print(f"{model_name}:")
        print(f"  Predictions: {metrics['total_predictions']}")
        print(f"  Avg inference time: {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  Avg confidence: {metrics['avg_confidence']:.3f}")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
    
    print("\n7. Monitoring Best Practices")
    print("-" * 33)
    
    best_practices = [
        "Monitor key metrics: latency, throughput, error rate, accuracy",
        "Set up alerting thresholds based on business requirements",
        "Track data drift to detect when models need retraining",
        "Use A/B testing to compare model versions safely",
        "Log all predictions and errors for debugging",
        "Monitor resource usage (CPU, memory, GPU)",
        "Implement health checks for automated deployments",
        "Track business metrics alongside technical metrics",
        "Set up dashboards for real-time monitoring",
        "Regularly review and update monitoring thresholds",
        "Monitor model confidence and prediction distribution",
        "Implement automated rollback on performance degradation"
    ]
    
    print("Production Monitoring Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("\n8. Alert Types and Responses")
    print("-" * 33)
    
    alert_responses = {
        "HIGH_LATENCY": "Scale up infrastructure, optimize model, check resource contention",
        "LOW_THROUGHPUT": "Increase worker instances, optimize batch processing",
        "HIGH_ERROR_RATE": "Check input validation, rollback to previous version",
        "LOW_CONFIDENCE": "Investigate data quality, consider model retraining",
        "HIGH_MEMORY_USAGE": "Optimize model memory usage, scale infrastructure",
        "DATA_DRIFT": "Retrain model with recent data, update preprocessing"
    }
    
    print("Alert Types and Recommended Responses:")
    for alert_type, response in alert_responses.items():
        print(f"  {alert_type}: {response}")
    
    print("\nProduction monitoring demonstration completed!")
    print("Generated files:")
    print("  - demo_production.log (prediction and error logs)")
    
    print("\nKey monitoring components demonstrated:")
    print("  - Real-time performance monitoring")
    print("  - Health metrics collection")
    print("  - Data drift detection")
    print("  - A/B testing framework")
    print("  - Comprehensive logging")
    print("  - Automated alerting")