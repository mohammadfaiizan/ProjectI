# Model Performance Monitoring

## Table of Contents

1. [Introduction to Model Monitoring](#introduction-to-model-monitoring)
2. [Performance Metrics Tracking](#performance-metrics-tracking)
3. [Alerting Systems](#alerting-systems)
4. [Dashboards and Visualization](#dashboards-and-visualization)
5. [SLOs and SLIs](#slos-and-slis)
6. [Logging Best Practices](#logging-best-practices)
7. [Prometheus and Grafana](#prometheus-and-grafana)
8. [Real-Time Monitoring](#real-time-monitoring)
9. [Anomaly Detection](#anomaly-detection)
10. [Key Takeaways](#key-takeaways)

## Introduction to Model Monitoring

Model performance monitoring tracks how ML models behave in production to ensure they continue to meet business objectives. Unlike traditional application monitoring, ML monitoring must track:

- **Model Performance**: Accuracy, precision, recall, F1-score
- **Data Quality**: Input distribution, feature drift
- **Prediction Quality**: Confidence scores, prediction distributions
- **Business Metrics**: Revenue impact, user engagement
- **Infrastructure**: Latency, throughput, error rates

### Monitoring Challenges

- **Ground Truth Delay**: Labels may arrive hours or days later
- **Concept Drift**: Model performance degrades over time
- **Data Drift**: Input distribution changes
- **Multiple Metrics**: Balancing accuracy, latency, cost
- **Scale**: Monitoring millions of predictions

### Monitoring Architecture

```
┌─────────────┐
│   Model     │
│  Inference  │
└──────┬──────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────┐    ┌──────────┐
│ Metrics  │    │  Logs    │
│Collector│    │Collector │
└────┬─────┘    └────┬─────┘
     │              │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Monitoring  │
     │   Backend    │
     └──────┬───────┘
            │
     ┌──────┴───────┐
     │              │
     ▼              ▼
┌──────────┐  ┌──────────┐
│Dashboard │  │ Alerting│
└──────────┘  └──────────┘
```

## Performance Metrics Tracking

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class ClassificationMonitor:
    def __init__(self):
        self.predictions = []
        self.true_labels = []
        self.timestamps = []
    
    def record_prediction(self, prediction, true_label, timestamp):
        """Record prediction and true label"""
        self.predictions.append(prediction)
        self.true_labels.append(true_label)
        self.timestamps.append(timestamp)
    
    def calculate_metrics(self, window_size=1000):
        """Calculate metrics over recent window"""
        if len(self.predictions) < window_size:
            preds = self.predictions
            labels = self.true_labels
        else:
            preds = self.predictions[-window_size:]
            labels = self.true_labels[-window_size:]
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted'),
            'recall': recall_score(labels, preds, average='weighted'),
            'f1_score': f1_score(labels, preds, average='weighted'),
            'sample_size': len(preds)
        }
    
    def track_over_time(self, interval=3600):
        """Track metrics over time intervals"""
        metrics_history = []
        current_interval_start = self.timestamps[0] if self.timestamps else None
        
        for i, timestamp in enumerate(self.timestamps):
            if timestamp - current_interval_start >= interval:
                interval_preds = self.predictions[
                    self.timestamps.index(current_interval_start):i
                ]
                interval_labels = self.true_labels[
                    self.timestamps.index(current_interval_start):i
                ]
                
                metrics = {
                    'timestamp': current_interval_start,
                    'accuracy': accuracy_score(interval_labels, interval_preds),
                    'precision': precision_score(interval_labels, interval_preds, average='weighted'),
                    'recall': recall_score(interval_labels, interval_preds, average='weighted')
                }
                metrics_history.append(metrics)
                
                current_interval_start = timestamp
        
        return metrics_history
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionMonitor:
    def __init__(self):
        self.predictions = []
        self.true_values = []
        self.timestamps = []
    
    def record_prediction(self, prediction, true_value, timestamp):
        """Record prediction and true value"""
        self.predictions.append(prediction)
        self.true_values.append(true_value)
        self.timestamps.append(timestamp)
    
    def calculate_metrics(self, window_size=1000):
        """Calculate regression metrics"""
        if len(self.predictions) < window_size:
            preds = np.array(self.predictions)
            values = np.array(self.true_values)
        else:
            preds = np.array(self.predictions[-window_size:])
            values = np.array(self.true_values[-window_size:])
        
        return {
            'mse': mean_squared_error(values, preds),
            'rmse': np.sqrt(mean_squared_error(values, preds)),
            'mae': mean_absolute_error(values, preds),
            'r2_score': r2_score(values, preds),
            'mean_error': np.mean(preds - values),
            'sample_size': len(preds)
        }
```

### Confidence Score Tracking

```python
class ConfidenceMonitor:
    def __init__(self):
        self.confidence_scores = []
        self.correct_predictions = []
    
    def record_prediction(self, prediction, true_label, confidence):
        """Record prediction with confidence"""
        self.confidence_scores.append(confidence)
        self.correct_predictions.append(prediction == true_label)
    
    def analyze_confidence_calibration(self):
        """Analyze if confidence scores are well-calibrated"""
        # Bin confidence scores
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(self.confidence_scores, bins)
        
        calibration_data = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_confidence = np.mean([self.confidence_scores[j] 
                                        for j in range(len(mask)) if mask[j]])
                accuracy = np.mean([self.correct_predictions[j] 
                                   for j in range(len(mask)) if mask[j]])
                calibration_data.append({
                    'confidence_bin': bins[i-1],
                    'avg_confidence': avg_confidence,
                    'accuracy': accuracy,
                    'count': np.sum(mask)
                })
        
        return calibration_data
```

## Alerting Systems

### Threshold-Based Alerts

```python
class ThresholdAlert:
    def __init__(self, metric_name, threshold, comparison='below'):
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison
        self.alert_history = []
    
    def check(self, current_value):
        """Check if alert condition is met"""
        if self.comparison == 'below':
            triggered = current_value < self.threshold
        elif self.comparison == 'above':
            triggered = current_value > self.threshold
        else:
            triggered = abs(current_value - self.threshold) > self.threshold * 0.1
        
        if triggered:
            alert = {
                'metric': self.metric_name,
                'value': current_value,
                'threshold': self.threshold,
                'timestamp': time.time()
            }
            self.alert_history.append(alert)
            self.send_alert(alert)
        
        return triggered
    
    def send_alert(self, alert):
        """Send alert notification"""
        # Implement notification (email, Slack, PagerDuty, etc.)
        print(f"ALERT: {alert['metric']} = {alert['value']} "
              f"(threshold: {alert['threshold']})")
```

### Statistical Alerting

```python
class StatisticalAlert:
    def __init__(self, window_size=100, z_threshold=3):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history = []
    
    def check(self, current_value):
        """Check if value is statistically anomalous"""
        self.history.append(current_value)
        
        if len(self.history) < self.window_size:
            return False
        
        # Keep only recent history
        self.history = self.history[-self.window_size:]
        
        # Calculate z-score
        mean = np.mean(self.history[:-1])
        std = np.std(self.history[:-1])
        
        if std == 0:
            return False
        
        z_score = abs((current_value - mean) / std)
        
        if z_score > self.z_threshold:
            return {
                'anomalous': True,
                'z_score': z_score,
                'current_value': current_value,
                'mean': mean,
                'std': std
            }
        
        return {'anomalous': False}
```

### Multi-Metric Alerting

```python
class MultiMetricAlert:
    def __init__(self):
        self.metrics = {}
        self.alert_rules = []
    
    def add_metric(self, name, value):
        """Add metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            'value': value,
            'timestamp': time.time()
        })
    
    def add_rule(self, condition_func, alert_message):
        """Add alert rule"""
        self.alert_rules.append({
            'condition': condition_func,
            'message': alert_message
        })
    
    def check_all_rules(self):
        """Check all alert rules"""
        alerts = []
        for rule in self.alert_rules:
            if rule['condition'](self.metrics):
                alerts.append({
                    'message': rule['message'],
                    'timestamp': time.time()
                })
        return alerts
```

## Dashboards and Visualization

### Real-Time Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MonitoringDashboard:
    def __init__(self):
        self.metrics_history = {
            'accuracy': [],
            'latency': [],
            'throughput': [],
            'error_rate': []
        }
    
    def update_metrics(self, metrics):
        """Update metrics history"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append({
                    'value': value,
                    'timestamp': time.time()
                })
    
    def create_dashboard(self):
        """Create dashboard visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Latency', 'Throughput', 'Error Rate')
        )
        
        # Accuracy plot
        if self.metrics_history['accuracy']:
            timestamps = [m['timestamp'] for m in self.metrics_history['accuracy']]
            values = [m['value'] for m in self.metrics_history['accuracy']]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Accuracy'),
                row=1, col=1
            )
        
        # Latency plot
        if self.metrics_history['latency']:
            timestamps = [m['timestamp'] for m in self.metrics_history['latency']]
            values = [m['value'] for m in self.metrics_history['latency']]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Latency'),
                row=1, col=2
            )
        
        # Throughput plot
        if self.metrics_history['throughput']:
            timestamps = [m['timestamp'] for m in self.metrics_history['throughput']]
            values = [m['value'] for m in self.metrics_history['throughput']]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Throughput'),
                row=2, col=1
            )
        
        # Error rate plot
        if self.metrics_history['error_rate']:
            timestamps = [m['timestamp'] for m in self.metrics_history['error_rate']]
            values = [m['value'] for m in self.metrics_history['error_rate']]
            fig.add_trace(
                go.Scatter(x=timestamps, y=values, name='Error Rate'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Model Performance Dashboard")
        return fig
```

## SLOs and SLIs

### Service Level Indicators (SLIs)

```python
class SLI:
    def __init__(self, name, measurement_func):
        self.name = name
        self.measurement_func = measurement_func
        self.values = []
    
    def measure(self):
        """Measure SLI value"""
        value = self.measurement_func()
        self.values.append({
            'value': value,
            'timestamp': time.time()
        })
        return value
    
    def get_current_value(self):
        """Get current SLI value"""
        if self.values:
            return self.values[-1]['value']
        return None
    
    def get_average(self, window_minutes=60):
        """Get average SLI over time window"""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [v['value'] for v in self.values 
                        if v['timestamp'] > cutoff_time]
        if recent_values:
            return np.mean(recent_values)
        return None
```

### Service Level Objectives (SLOs)

```python
class SLO:
    def __init__(self, name, sli, target, window_minutes=60):
        self.name = name
        self.sli = sli
        self.target = target
        self.window_minutes = window_minutes
    
    def check_compliance(self):
        """Check if SLO is being met"""
        current_value = self.sli.get_average(self.window_minutes)
        if current_value is None:
            return None
        
        return {
            'slo_name': self.name,
            'current_value': current_value,
            'target': self.target,
            'compliant': current_value >= self.target,
            'error_budget': self.calculate_error_budget()
        }
    
    def calculate_error_budget(self):
        """Calculate remaining error budget"""
        current_value = self.sli.get_average(self.window_minutes)
        if current_value is None:
            return None
        
        return max(0, (current_value - self.target) / self.target * 100)
```

### SLO Implementation

```python
# Define SLIs
accuracy_sli = SLI('accuracy', lambda: monitor.calculate_metrics()['accuracy'])
latency_sli = SLI('latency_p95', lambda: monitor.get_latency_p95())
availability_sli = SLI('availability', lambda: monitor.get_availability())

# Define SLOs
accuracy_slo = SLO('accuracy', accuracy_sli, target=0.95)
latency_slo = SLO('latency', latency_sli, target=100, comparison='below')
availability_slo = SLO('availability', availability_sli, target=0.999)

# Check SLOs
slos = [accuracy_slo, latency_slo, availability_slo]
for slo in slos:
    compliance = slo.check_compliance()
    if compliance and not compliance['compliant']:
        send_alert(f"SLO violation: {slo.name}")
```

## Logging Best Practices

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_prediction(self, request_id, features, prediction, 
                      confidence=None, latency=None):
        """Log prediction with structured data"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'prediction',
            'request_id': request_id,
            'features': features,
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, request_id, error_type, error_message, stack_trace=None):
        """Log error with structured data"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'error',
            'request_id': request_id,
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack_trace
        }
        self.logger.error(json.dumps(log_entry))
```

### Log Aggregation

```python
class LogAggregator:
    def __init__(self):
        self.logs = []
    
    def add_log(self, log_entry):
        """Add log entry"""
        self.logs.append(log_entry)
    
    def aggregate_by_time(self, interval_seconds=3600):
        """Aggregate logs by time interval"""
        intervals = {}
        for log in self.logs:
            interval_key = int(log['timestamp'] / interval_seconds) * interval_seconds
            if interval_key not in intervals:
                intervals[interval_key] = []
            intervals[interval_key].append(log)
        return intervals
    
    def get_error_rate(self, time_window_minutes=60):
        """Calculate error rate over time window"""
        cutoff = time.time() - (time_window_minutes * 60)
        recent_logs = [l for l in self.logs if l['timestamp'] > cutoff]
        
        if not recent_logs:
            return 0
        
        error_count = sum(1 for l in recent_logs if l.get('event_type') == 'error')
        return error_count / len(recent_logs)
```

## Prometheus and Grafana

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
error_counter = Counter('prediction_errors_total', 'Total prediction errors')

class PrometheusMonitor:
    def __init__(self, port=8000):
        start_http_server(port)
    
    def record_prediction(self, latency, success=True):
        """Record prediction metrics"""
        prediction_counter.inc()
        prediction_latency.observe(latency)
        if not success:
            error_counter.inc()
    
    def update_accuracy(self, accuracy):
        """Update accuracy metric"""
        model_accuracy.set(accuracy)
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Model Performance",
    "panels": [
      {
        "title": "Prediction Rate",
        "targets": [
          {
            "expr": "rate(model_predictions_total[5m])"
          }
        ]
      },
      {
        "title": "Latency P95",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, prediction_latency_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Accuracy",
        "targets": [
          {
            "expr": "model_accuracy"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(prediction_errors_total[5m]) / rate(model_predictions_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Real-Time Monitoring

### Streaming Metrics

```python
from kafka import KafkaConsumer
import json

class StreamingMonitor:
    def __init__(self, kafka_topic):
        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.metrics_window = []
        self.window_size = 1000
    
    def process_stream(self):
        """Process streaming metrics"""
        for message in self.consumer:
            metric = message.value
            self.metrics_window.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics_window) > self.window_size:
                self.metrics_window = self.metrics_window[-self.window_size:]
            
            # Calculate real-time metrics
            self.update_real_time_metrics()
    
    def update_real_time_metrics(self):
        """Update real-time metrics"""
        if len(self.metrics_window) < 100:
            return
        
        recent_metrics = self.metrics_window[-100:]
        
        # Calculate metrics
        accuracy = np.mean([m['correct'] for m in recent_metrics])
        latency_p95 = np.percentile([m['latency'] for m in recent_metrics], 95)
        error_rate = np.mean([m['error'] for m in recent_metrics])
        
        # Update Prometheus metrics
        model_accuracy.set(accuracy)
        prediction_latency.observe(latency_p95)
```

## Anomaly Detection

### Performance Anomaly Detection

```python
from sklearn.ensemble import IsolationForest

class PerformanceAnomalyDetector:
    def __init__(self):
        self.metrics_history = []
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.is_fitted = False
    
    def add_metrics(self, metrics):
        """Add metrics to history"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def detect_anomalies(self):
        """Detect anomalies in recent metrics"""
        if len(self.metrics_history) < 100:
            return []
        
        # Prepare features
        features = []
        for m in self.metrics_history[-100:]:
            features.append([
                m.get('accuracy', 0),
                m.get('latency', 0),
                m.get('throughput', 0),
                m.get('error_rate', 0)
            ])
        
        # Fit detector if needed
        if not self.is_fitted and len(self.metrics_history) >= 500:
            self.anomaly_detector.fit(features)
            self.is_fitted = True
        
        if self.is_fitted:
            # Detect anomalies
            predictions = self.anomaly_detector.predict(features[-10:])
            anomalies = [i for i, p in enumerate(predictions) 
                        if p == -1]
            return anomalies
        
        return []
```

## Key Takeaways

- Model performance monitoring tracks accuracy, latency, and business metrics in production
- Performance metrics tracking requires careful selection of relevant metrics for the use case
- Alerting systems notify teams when metrics exceed thresholds or show anomalies
- Dashboards provide visual representation of model performance over time
- SLOs and SLIs define and measure service quality objectives
- Structured logging enables efficient log analysis and debugging
- Prometheus and Grafana provide powerful monitoring and visualization capabilities
- Real-time monitoring enables immediate response to performance issues
- Anomaly detection identifies unusual patterns in model behavior
- Comprehensive monitoring requires tracking both technical and business metrics
