# Debugging and Troubleshooting ML Systems

## Table of Contents

1. [Introduction to ML Debugging](#introduction-to-ml-debugging)
2. [Error Analysis Methodology](#error-analysis-methodology)
3. [Model Debugging Tools](#model-debugging-tools)
4. [Performance Profiling](#performance-profiling)
5. [Data Debugging](#data-debugging)
6. [Failure Mode Analysis](#failure-mode-analysis)
7. [Incident Response for ML](#incident-response-for-ml)
8. [Root Cause Analysis](#root-cause-analysis)
9. [Debugging Checklist](#debugging-checklist)
10. [Key Takeaways](#key-takeaways)

## Introduction to ML Debugging

Debugging ML systems presents unique challenges compared to traditional software debugging. ML systems involve complex data pipelines, non-deterministic behavior, and dependencies on data distributions that can change over time.

### ML Debugging Challenges

**Non-Determinism**:
- Random initialization and sampling introduce variability
- Same code may produce different results
- Difficult to reproduce exact failures

**Data Dependencies**:
- Models depend on data distributions
- Data quality issues may not manifest immediately
- Changes in data pipeline can cause silent failures

**Complex Pipelines**:
- Multiple stages from data ingestion to prediction
- Failures can occur at any stage
- Difficult to trace issues through the pipeline

**Performance vs Correctness**:
- Models may be correct but too slow
- Optimization may introduce subtle bugs
- Trade-offs between accuracy and latency

### Debugging Mindset

Effective ML debugging requires:
- **Systematic Approach**: Follow structured methodologies
- **Data-Centric Thinking**: Consider data as first-class citizen
- **Hypothesis-Driven**: Form hypotheses and test systematically
- **Observability**: Instrument systems for visibility
- **Documentation**: Record debugging process and findings

### Debugging Workflow

```
┌──────────────┐
│   Identify   │
│    Issue     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Reproduce   │
│   Problem    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Formulate   │
│  Hypothesis  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Test       │
│  Hypothesis  │
└──────┬───────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│   Fix        │ │  Refine     │
│   Issue      │ │ Hypothesis  │
└──────┬───────┘ └──────┬──────┘
       │                │
       └────────┬───────┘
                │
                ▼
         ┌──────────────┐
         │   Verify    │
         │    Fix      │
         └──────────────┘
```

## Error Analysis Methodology

Systematic error analysis helps identify patterns in model failures and guides improvement efforts.

### Error Categorization

Categorize errors by type and severity:

```python
class ErrorAnalyzer:
    def __init__(self, model, test_data, true_labels):
        self.model = model
        self.test_data = test_data
        self.true_labels = true_labels
        self.predictions = model.predict(test_data)
        self.errors = self._identify_errors()
    
    def _identify_errors(self):
        """Identify all prediction errors"""
        errors = []
        for i, (pred, true) in enumerate(zip(self.predictions, self.true_labels)):
            if pred != true:
                errors.append({
                    'index': i,
                    'prediction': pred,
                    'true_label': true,
                    'features': self.test_data.iloc[i].to_dict(),
                    'confidence': self._get_confidence(i)
                })
        return errors
    
    def _get_confidence(self, index):
        """Get prediction confidence"""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(self.test_data.iloc[[index]])[0]
            return max(proba)
        return None
    
    def categorize_errors(self):
        """Categorize errors by type"""
        categories = {
            'false_positives': [],
            'false_negatives': [],
            'low_confidence': [],
            'high_confidence': []
        }
        
        for error in self.errors:
            # False positive/negative
            if error['prediction'] == 1 and error['true_label'] == 0:
                categories['false_positives'].append(error)
            elif error['prediction'] == 0 and error['true_label'] == 1:
                categories['false_negatives'].append(error)
            
            # Confidence-based
            if error['confidence'] is not None:
                if error['confidence'] < 0.5:
                    categories['low_confidence'].append(error)
                elif error['confidence'] > 0.9:
                    categories['high_confidence'].append(error)
        
        return categories
    
    def analyze_error_patterns(self):
        """Analyze patterns in errors"""
        error_df = pd.DataFrame(self.errors)
        
        patterns = {
            'class_distribution': error_df['true_label'].value_counts().to_dict(),
            'prediction_distribution': error_df['prediction'].value_counts().to_dict(),
            'feature_statistics': error_df['features'].apply(pd.Series).describe()
        }
        
        return patterns
```

### Confusion Matrix Analysis

Analyze confusion matrix to understand error types:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class ConfusionMatrixAnalyzer:
    def __init__(self, y_true, y_pred, class_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names or list(set(y_true))
        self.cm = confusion_matrix(y_true, y_pred)
    
    def visualize(self):
        """Visualize confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
    
    def analyze(self):
        """Analyze confusion matrix patterns"""
        analysis = {
            'total_errors': self.cm.sum() - np.trace(self.cm),
            'error_rate': (self.cm.sum() - np.trace(self.cm)) / self.cm.sum(),
            'class_wise_errors': {},
            'most_confused_pairs': []
        }
        
        # Class-wise error analysis
        for i, class_name in enumerate(self.class_names):
            total_true = self.cm[i, :].sum()
            correct = self.cm[i, i]
            errors = total_true - correct
            analysis['class_wise_errors'][class_name] = {
                'total': total_true,
                'correct': correct,
                'errors': errors,
                'error_rate': errors / total_true if total_true > 0 else 0
            }
        
        # Most confused pairs
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and self.cm[i, j] > 0:
                    analysis['most_confused_pairs'].append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': self.cm[i, j]
                    })
        
        analysis['most_confused_pairs'].sort(
            key=lambda x: x['count'],
            reverse=True
        )
        
        return analysis
```

### Error Sampling Strategies

Sample errors for manual inspection:

```python
class ErrorSampler:
    def __init__(self, errors, test_data):
        self.errors = errors
        self.test_data = test_data
    
    def random_sample(self, n=10):
        """Random sample of errors"""
        import random
        return random.sample(self.errors, min(n, len(self.errors)))
    
    def stratified_sample(self, n_per_category=5):
        """Stratified sample across error categories"""
        categories = {}
        for error in self.errors:
            category = f"{error['true_label']}_to_{error['prediction']}"
            if category not in categories:
                categories[category] = []
            categories[category].append(error)
        
        samples = []
        for category, category_errors in categories.items():
            samples.extend(
                self.random_sample_from_list(category_errors, n_per_category)
            )
        
        return samples
    
    def high_confidence_errors(self, threshold=0.9):
        """Sample high-confidence errors (most concerning)"""
        high_conf_errors = [
            e for e in self.errors
            if e.get('confidence', 0) > threshold
        ]
        return high_conf_errors[:10]
    
    def low_confidence_errors(self, threshold=0.5):
        """Sample low-confidence errors"""
        low_conf_errors = [
            e for e in self.errors
            if e.get('confidence', 1) < threshold
        ]
        return low_conf_errors[:10]
    
    def random_sample_from_list(self, lst, n):
        """Helper method for random sampling"""
        import random
        return random.sample(lst, min(n, len(lst)))
```

## Model Debugging Tools

Specialized tools help debug ML models by providing visibility into model behavior and decision-making processes.

### Model Interpretability Tools

```python
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

class ModelInterpreter:
    def __init__(self, model, training_data):
        self.model = model
        self.training_data = training_data
        self.shap_explainer = None
        self.lime_explainer = None
    
    def initialize_shap(self):
        """Initialize SHAP explainer"""
        if self.shap_explainer is None:
            # Use appropriate explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict,
                    self.training_data.sample(100)
                )
    
    def explain_prediction_shap(self, instance):
        """Explain prediction using SHAP"""
        self.initialize_shap()
        shap_values = self.shap_explainer.shap_values(instance)
        return shap_values
    
    def initialize_lime(self):
        """Initialize LIME explainer"""
        if self.lime_explainer is None:
            self.lime_explainer = LimeTabularExplainer(
                self.training_data.values,
                feature_names=self.training_data.columns,
                mode='classification'
            )
    
    def explain_prediction_lime(self, instance, num_features=10):
        """Explain prediction using LIME"""
        self.initialize_lime()
        explanation = self.lime_explainer.explain_instance(
            instance.values[0],
            self.model.predict_proba,
            num_features=num_features
        )
        return explanation
```

### Model Debugging with TensorBoard

```python
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

class TensorBoardDebugger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def log_training_metrics(self, epoch, metrics):
        """Log training metrics"""
        with self.writer.as_default():
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(f'train/{metric_name}', metric_value, step=epoch)
    
    def log_validation_metrics(self, epoch, metrics):
        """Log validation metrics"""
        with self.writer.as_default():
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(f'val/{metric_name}', metric_value, step=epoch)
    
    def log_gradients(self, model, epoch):
        """Log gradient information"""
        with self.writer.as_default():
            for layer in model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.kernel
                    grads = tf.gradients(model.loss, weights)
                    tf.summary.histogram(f'{layer.name}/gradients', grads, step=epoch)
                    tf.summary.histogram(f'{layer.name}/weights', weights, step=epoch)
    
    def log_hyperparameters(self, hparams, metrics):
        """Log hyperparameter experiments"""
        with self.writer.as_default():
            hp.hparams(hparams)
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(metric_name, metric_value)
```

### Debugging with Weights and Biases

```python
import wandb

class WandBDebugger:
    def __init__(self, project_name, config=None):
        wandb.init(project=project_name, config=config)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to W&B"""
        wandb.log(metrics, step=step)
    
    def log_predictions(self, predictions, true_labels, class_names=None):
        """Log predictions for analysis"""
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_labels,
                preds=predictions,
                class_names=class_names
            )
        })
    
    def log_data_table(self, data, columns):
        """Log data as table for inspection"""
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"data_table": table})
    
    def log_model_artifact(self, model_path, model_name):
        """Log model artifact"""
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
```

## Performance Profiling

Performance profiling identifies bottlenecks in ML systems, helping optimize inference latency and throughput.

### Inference Profiling

```python
import cProfile
import pstats
import io
from contextlib import contextmanager

class InferenceProfiler:
    def __init__(self, model):
        self.model = model
        self.profiler = cProfile.Profile()
    
    @contextmanager
    def profile(self):
        """Context manager for profiling"""
        self.profiler.enable()
        try:
            yield
        finally:
            self.profiler.disable()
    
    def profile_prediction(self, input_data, n_iterations=100):
        """Profile prediction performance"""
        with self.profile():
            for _ in range(n_iterations):
                _ = self.model.predict(input_data)
        
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        return s.getvalue()
    
    def analyze_latency(self, input_data, n_iterations=1000):
        """Analyze prediction latency"""
        import time
        
        latencies = []
        for _ in range(n_iterations):
            start = time.time()
            _ = self.model.predict(input_data)
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'std': np.std(latencies)
        }
```

### Memory Profiling

```python
import tracemalloc
import line_profiler

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []
    
    def start_tracing(self):
        """Start memory tracing"""
        tracemalloc.start()
    
    def take_snapshot(self, label=""):
        """Take memory snapshot"""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'label': label,
            'snapshot': snapshot,
            'timestamp': datetime.now()
        })
        return snapshot
    
    def compare_snapshots(self, snapshot1_label, snapshot2_label):
        """Compare two snapshots"""
        snap1 = next(s['snapshot'] for s in self.snapshots if s['label'] == snapshot1_label)
        snap2 = next(s['snapshot'] for s in self.snapshots if s['label'] == snapshot2_label)
        
        top_stats = snap2.compare_to(snap1, 'lineno')
        
        print(f"Top 10 memory differences:")
        for stat in top_stats[:10]:
            print(stat)
        
        return top_stats
    
    def get_memory_usage(self):
        """Get current memory usage"""
        current, peak = tracemalloc.get_traced_memory()
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        }
```

### Pipeline Profiling

```python
class PipelineProfiler:
    def __init__(self):
        self.stage_timings = {}
        self.stage_memory = {}
    
    def profile_stage(self, stage_name, func, *args, **kwargs):
        """Profile a pipeline stage"""
        import time
        import tracemalloc
        
        # Time profiling
        start_time = time.time()
        
        # Memory profiling
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Collect metrics
        end_time = time.time()
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Store metrics
        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = []
            self.stage_memory[stage_name] = []
        
        self.stage_timings[stage_name].append(end_time - start_time)
        self.stage_memory[stage_name].append(
            (end_memory - start_memory) / 1024 / 1024
        )
        
        return result
    
    def get_stage_statistics(self):
        """Get statistics for all stages"""
        statistics = {}
        for stage_name in self.stage_timings.keys():
            statistics[stage_name] = {
                'timing': {
                    'mean': np.mean(self.stage_timings[stage_name]),
                    'median': np.median(self.stage_timings[stage_name]),
                    'p95': np.percentile(self.stage_timings[stage_name], 95),
                    'total': np.sum(self.stage_timings[stage_name])
                },
                'memory': {
                    'mean_mb': np.mean(self.stage_memory[stage_name]),
                    'peak_mb': np.max(self.stage_memory[stage_name])
                }
            }
        return statistics
    
    def identify_bottlenecks(self):
        """Identify pipeline bottlenecks"""
        stats = self.get_stage_statistics()
        
        # Sort by total time
        sorted_stages = sorted(
            stats.items(),
            key=lambda x: x[1]['timing']['total'],
            reverse=True
        )
        
        bottlenecks = []
        total_time = sum(s['timing']['total'] for s in stats.values())
        
        cumulative_time = 0
        for stage_name, stage_stats in sorted_stages:
            cumulative_time += stage_stats['timing']['total']
            percentage = (cumulative_time / total_time) * 100
            bottlenecks.append({
                'stage': stage_name,
                'total_time': stage_stats['timing']['total'],
                'percentage': percentage,
                'mean_time': stage_stats['timing']['mean']
            })
            
            if percentage >= 80:  # Top 80% of time
                break
        
        return bottlenecks
```

## Data Debugging

Data issues are a common source of ML system failures. Systematic data debugging helps identify and resolve data quality problems.

### Data Quality Checks

```python
class DataQualityChecker:
    def __init__(self, reference_data=None):
        self.reference_data = reference_data
        self.checks = []
    
    def add_check(self, name, check_function, severity='warning'):
        """Add a data quality check"""
        self.checks.append({
            'name': name,
            'function': check_function,
            'severity': severity
        })
    
    def run_checks(self, data):
        """Run all quality checks"""
        results = []
        for check in self.checks:
            try:
                result = check['function'](data)
                results.append({
                    'check': check['name'],
                    'passed': result,
                    'severity': check['severity'],
                    'message': f"{check['name']} {'passed' if result else 'failed'}"
                })
            except Exception as e:
                results.append({
                    'check': check['name'],
                    'passed': False,
                    'severity': 'error',
                    'message': f"{check['name']} raised exception: {str(e)}"
                })
        return results
    
    def check_missing_values(self, threshold=0.1):
        """Check for excessive missing values"""
        def check(data):
            missing_pct = data.isnull().sum() / len(data)
            return (missing_pct <= threshold).all()
        return check
    
    def check_data_types(self):
        """Check data types match expected"""
        def check(data):
            if self.reference_data is None:
                return True
            return data.dtypes.equals(self.reference_data.dtypes)
        return check
    
    def check_value_ranges(self):
        """Check values are within expected ranges"""
        def check(data):
            if self.reference_data is None:
                return True
            
            for col in data.select_dtypes(include=[np.number]).columns:
                if col in self.reference_data.columns:
                    ref_min = self.reference_data[col].min()
                    ref_max = self.reference_data[col].max()
                    curr_min = data[col].min()
                    curr_max = data[col].max()
                    
                    # Allow some tolerance
                    if curr_min < ref_min * 0.9 or curr_max > ref_max * 1.1:
                        return False
            return True
        return check
```

### Data Distribution Analysis

```python
class DataDistributionAnalyzer:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_stats = self._compute_statistics(reference_data)
    
    def _compute_statistics(self, data):
        """Compute statistical summary"""
        stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75)
            }
        return stats
    
    def compare_distributions(self, current_data):
        """Compare current data distribution to reference"""
        current_stats = self._compute_statistics(current_data)
        
        comparisons = {}
        for col in self.reference_stats.keys():
            if col in current_stats:
                ref = self.reference_stats[col]
                curr = current_stats[col]
                
                comparisons[col] = {
                    'mean_diff': abs(curr['mean'] - ref['mean']) / ref['mean'],
                    'std_diff': abs(curr['std'] - ref['std']) / ref['std'],
                    'min_diff': abs(curr['min'] - ref['min']) / abs(ref['min']) if ref['min'] != 0 else float('inf'),
                    'max_diff': abs(curr['max'] - ref['max']) / abs(ref['max']) if ref['max'] != 0 else float('inf')
                }
        
        return comparisons
    
    def detect_outliers(self, data, method='iqr', threshold=1.5):
        """Detect outliers in data"""
        outliers = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': outlier_mask.mean() * 100,
                    'indices': data[outlier_mask].index.tolist()
                }
        
        return outliers
```

### Data Lineage Debugging

```python
class DataLineageDebugger:
    def __init__(self):
        self.lineage_graph = {}
    
    def track_transformation(self, input_data_id, output_data_id, transformation):
        """Track data transformation"""
        if output_data_id not in self.lineage_graph:
            self.lineage_graph[output_data_id] = {
                'inputs': [],
                'transformation': None,
                'metadata': {}
            }
        
        self.lineage_graph[output_data_id]['inputs'].append(input_data_id)
        self.lineage_graph[output_data_id]['transformation'] = transformation
    
    def trace_back(self, data_id):
        """Trace data lineage backwards"""
        if data_id not in self.lineage_graph:
            return {'data_id': data_id, 'lineage': []}
        
        node = self.lineage_graph[data_id]
        lineage = [{
            'data_id': data_id,
            'transformation': node['transformation'],
            'inputs': node['inputs']
        }]
        
        for input_id in node['inputs']:
            input_lineage = self.trace_back(input_id)
            lineage.extend(input_lineage['lineage'])
        
        return {'data_id': data_id, 'lineage': lineage}
    
    def find_affected_downstream(self, data_id):
        """Find all downstream data affected by changes"""
        affected = []
        for output_id, node in self.lineage_graph.items():
            if data_id in node['inputs']:
                affected.append(output_id)
                affected.extend(self.find_affected_downstream(output_id))
        return list(set(affected))
```

## Failure Mode Analysis

Systematic analysis of failure modes helps identify common patterns and prevent future issues.

### Failure Mode Classification

```python
class FailureModeAnalyzer:
    def __init__(self):
        self.failure_modes = {
            'data_quality': [],
            'model_degradation': [],
            'infrastructure': [],
            'integration': [],
            'configuration': []
        }
    
    def classify_failure(self, failure_description, error_type, stack_trace=None):
        """Classify failure into categories"""
        classification = {
            'description': failure_description,
            'error_type': error_type,
            'stack_trace': stack_trace,
            'category': None,
            'severity': None,
            'timestamp': datetime.now()
        }
        
        # Classify based on error type and description
        if 'data' in failure_description.lower() or 'null' in failure_description.lower():
            classification['category'] = 'data_quality'
        elif 'model' in failure_description.lower() or 'prediction' in failure_description.lower():
            classification['category'] = 'model_degradation'
        elif 'connection' in failure_description.lower() or 'timeout' in failure_description.lower():
            classification['category'] = 'infrastructure'
        elif 'api' in failure_description.lower() or 'service' in failure_description.lower():
            classification['category'] = 'integration'
        else:
            classification['category'] = 'configuration'
        
        # Determine severity
        if 'critical' in failure_description.lower() or 'fatal' in error_type.lower():
            classification['severity'] = 'critical'
        elif 'error' in error_type.lower():
            classification['severity'] = 'high'
        elif 'warning' in error_type.lower():
            classification['severity'] = 'medium'
        else:
            classification['severity'] = 'low'
        
        self.failure_modes[classification['category']].append(classification)
        return classification
    
    def get_failure_statistics(self):
        """Get statistics on failure modes"""
        stats = {}
        for category, failures in self.failure_modes.items():
            stats[category] = {
                'total': len(failures),
                'by_severity': {},
                'recent_failures': failures[-10:] if failures else []
            }
            
            for failure in failures:
                severity = failure['severity']
                stats[category]['by_severity'][severity] = \
                    stats[category]['by_severity'].get(severity, 0) + 1
        
        return stats
```

### Common Failure Patterns

```python
class FailurePatternDetector:
    def __init__(self):
        self.patterns = {
            'data_drift': {
                'indicators': ['accuracy_decrease', 'prediction_distribution_change'],
                'frequency': 0
            },
            'memory_leak': {
                'indicators': ['memory_increase_over_time', 'gradual_slowdown'],
                'frequency': 0
            },
            'race_condition': {
                'indicators': ['non_deterministic_errors', 'concurrent_access'],
                'frequency': 0
            },
            'configuration_error': {
                'indicators': ['startup_failure', 'missing_config'],
                'frequency': 0
            }
        }
    
    def detect_patterns(self, failure_history):
        """Detect common failure patterns"""
        detected_patterns = []
        
        for pattern_name, pattern_info in self.patterns.items():
            indicators_present = sum(
                1 for indicator in pattern_info['indicators']
                if any(indicator in str(failure).lower() for failure in failure_history)
            )
            
            if indicators_present >= len(pattern_info['indicators']) * 0.5:
                detected_patterns.append({
                    'pattern': pattern_name,
                    'confidence': indicators_present / len(pattern_info['indicators']),
                    'indicators_found': indicators_present
                })
                self.patterns[pattern_name]['frequency'] += 1
        
        return detected_patterns
```

## Incident Response for ML

Structured incident response procedures help minimize impact and restore service quickly.

### Incident Classification

```python
class IncidentClassifier:
    def __init__(self):
        self.incident_levels = {
            'P0': {
                'description': 'Critical - Service completely down',
                'response_time': 15,  # minutes
                'resolution_time': 60  # minutes
            },
            'P1': {
                'description': 'High - Significant degradation',
                'response_time': 60,
                'resolution_time': 240
            },
            'P2': {
                'description': 'Medium - Partial functionality affected',
                'response_time': 240,
                'resolution_time': 480
            },
            'P3': {
                'description': 'Low - Minor issues',
                'response_time': 480,
                'resolution_time': 1440
            }
        }
    
    def classify_incident(self, impact_description, affected_users, error_rate):
        """Classify incident severity"""
        # P0: Complete service outage
        if 'down' in impact_description.lower() or error_rate > 0.5:
            return 'P0'
        
        # P1: High error rate or many users affected
        if error_rate > 0.2 or affected_users > 10000:
            return 'P1'
        
        # P2: Moderate impact
        if error_rate > 0.05 or affected_users > 1000:
            return 'P2'
        
        # P3: Low impact
        return 'P3'
```

### Incident Response Workflow

```python
class IncidentResponseManager:
    def __init__(self):
        self.active_incidents = {}
        self.incident_history = []
    
    def create_incident(self, description, severity, detected_by):
        """Create new incident"""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        incident = {
            'id': incident_id,
            'description': description,
            'severity': severity,
            'status': 'open',
            'detected_by': detected_by,
            'detected_at': datetime.now(),
            'assigned_to': None,
            'updates': [],
            'root_cause': None,
            'resolution': None,
            'resolved_at': None
        }
        
        self.active_incidents[incident_id] = incident
        return incident_id
    
    def update_incident(self, incident_id, update_text, status=None):
        """Update incident with new information"""
        if incident_id not in self.active_incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.active_incidents[incident_id]
        incident['updates'].append({
            'timestamp': datetime.now(),
            'update': update_text
        })
        
        if status:
            incident['status'] = status
            if status == 'resolved':
                incident['resolved_at'] = datetime.now()
                self.incident_history.append(incident)
                del self.active_incidents[incident_id]
    
    def escalate_incident(self, incident_id, reason):
        """Escalate incident to higher severity"""
        incident = self.active_incidents[incident_id]
        current_severity = incident['severity']
        
        severity_order = ['P3', 'P2', 'P1', 'P0']
        current_index = severity_order.index(current_severity)
        
        if current_index < len(severity_order) - 1:
            incident['severity'] = severity_order[current_index + 1]
            incident['updates'].append({
                'timestamp': datetime.now(),
                'update': f"Escalated to {incident['severity']}: {reason}"
            })
```

### Rollback Procedures

```python
class ModelRollbackManager:
    def __init__(self, model_registry):
        self.model_registry = model_registry
        self.rollback_history = []
    
    def rollback_model(self, current_version, target_version=None):
        """Rollback model to previous version"""
        if target_version is None:
            # Rollback to previous version
            versions = self.model_registry.list_versions()
            current_index = versions.index(current_version)
            if current_index > 0:
                target_version = versions[current_index - 1]
            else:
                raise ValueError("No previous version available")
        
        rollback_record = {
            'from_version': current_version,
            'to_version': target_version,
            'timestamp': datetime.now(),
            'reason': 'incident_response'
        }
        
        # Perform rollback
        try:
            self.model_registry.deploy_version(target_version)
            rollback_record['status'] = 'success'
        except Exception as e:
            rollback_record['status'] = 'failed'
            rollback_record['error'] = str(e)
            raise
        
        self.rollback_history.append(rollback_record)
        return rollback_record
```

## Root Cause Analysis

Systematic root cause analysis helps identify underlying issues and prevent recurrence.

### 5 Whys Analysis

```python
class RootCauseAnalyzer:
    def __init__(self):
        self.analyses = []
    
    def five_whys_analysis(self, problem_description):
        """Perform 5 Whys root cause analysis"""
        analysis = {
            'problem': problem_description,
            'whys': [],
            'root_cause': None,
            'timestamp': datetime.now()
        }
        
        current_why = problem_description
        for i in range(5):
            # In practice, this would involve investigation
            # Here we structure the analysis
            analysis['whys'].append({
                'level': i + 1,
                'question': f"Why {current_why.lower()}?",
                'answer': None  # To be filled during investigation
            })
            # In real scenario, answer would lead to next why
            current_why = f"Because of {current_why.lower()}"
        
        self.analyses.append(analysis)
        return analysis
    
    def fishbone_diagram(self, problem):
        """Structure fishbone (Ishikawa) diagram analysis"""
        categories = {
            'machine': [],
            'method': [],
            'material': [],
            'manpower': [],
            'measurement': [],
            'environment': []
        }
        
        return {
            'problem': problem,
            'categories': categories,
            'root_causes': []
        }
```

### Systematic Investigation

```python
class SystematicInvestigator:
    def __init__(self):
        self.investigation_log = []
    
    def investigate_component(self, component_name, checks):
        """Systematically investigate a component"""
        investigation = {
            'component': component_name,
            'timestamp': datetime.now(),
            'checks': [],
            'findings': []
        }
        
        for check_name, check_function in checks.items():
            try:
                result = check_function()
                investigation['checks'].append({
                    'name': check_name,
                    'status': 'passed' if result else 'failed',
                    'result': result
                })
                
                if not result:
                    investigation['findings'].append({
                        'check': check_name,
                        'issue': 'Check failed',
                        'severity': 'high'
                    })
            except Exception as e:
                investigation['checks'].append({
                    'name': check_name,
                    'status': 'error',
                    'error': str(e)
                })
                investigation['findings'].append({
                    'check': check_name,
                    'issue': f"Exception: {str(e)}",
                    'severity': 'critical'
                })
        
        self.investigation_log.append(investigation)
        return investigation
    
    def generate_report(self):
        """Generate investigation report"""
        report = {
            'total_investigations': len(self.investigation_log),
            'components_checked': [inv['component'] for inv in self.investigation_log],
            'total_findings': sum(len(inv['findings']) for inv in self.investigation_log),
            'critical_findings': [
                finding for inv in self.investigation_log
                for finding in inv['findings']
                if finding['severity'] == 'critical'
            ]
        }
        return report
```

## Debugging Checklist

A comprehensive checklist ensures systematic debugging and prevents overlooking common issues.

### Pre-Debugging Checklist

```python
class DebuggingChecklist:
    def __init__(self):
        self.checklist = {
            'reproduction': {
                'can_reproduce_locally': False,
                'reproduction_steps_documented': False,
                'environment_matches_production': False
            },
            'data': {
                'data_version_verified': False,
                'data_quality_checked': False,
                'data_schema_validated': False,
                'missing_values_handled': False
            },
            'model': {
                'model_version_verified': False,
                'model_loaded_correctly': False,
                'predictions_reproducible': False,
                'model_interpretability_checked': False
            },
            'infrastructure': {
                'dependencies_verified': False,
                'resource_limits_checked': False,
                'network_connectivity_verified': False,
                'service_health_checked': False
            },
            'configuration': {
                'config_files_validated': False,
                'environment_variables_set': False,
                'secrets_accessible': False,
                'feature_flags_checked': False
            },
            'monitoring': {
                'logs_reviewed': False,
                'metrics_analyzed': False,
                'alerts_checked': False,
                'error_tracking_reviewed': False
            }
        }
    
    def check_item(self, category, item):
        """Mark checklist item as complete"""
        if category in self.checklist and item in self.checklist[category]:
            self.checklist[category][item] = True
    
    def get_completion_status(self):
        """Get overall completion status"""
        total_items = sum(len(items) for items in self.checklist.values())
        completed_items = sum(
            sum(1 for v in items.values() if v)
            for items in self.checklist.values()
        )
        
        return {
            'total': total_items,
            'completed': completed_items,
            'percentage': (completed_items / total_items) * 100,
            'remaining': [
                f"{category}.{item}"
                for category, items in self.checklist.items()
                for item, completed in items.items()
                if not completed
            ]
        }
    
    def generate_debugging_plan(self):
        """Generate debugging plan based on checklist"""
        plan = []
        
        for category, items in self.checklist.items():
            incomplete = [item for item, completed in items.items() if not completed]
            if incomplete:
                plan.append({
                    'category': category,
                    'next_steps': incomplete,
                    'priority': 'high' if category in ['reproduction', 'data', 'model'] else 'medium'
                })
        
        return plan
```

### Common Issues Checklist

```python
class CommonIssuesChecklist:
    def __init__(self):
        self.common_issues = {
            'data_issues': [
                'Missing or null values not handled',
                'Data type mismatches',
                'Schema changes not reflected',
                'Data drift not detected',
                'Feature scaling inconsistent'
            ],
            'model_issues': [
                'Model version mismatch',
                'Model not loaded correctly',
                'Preprocessing mismatch',
                'Feature engineering inconsistency',
                'Model degradation over time'
            ],
            'infrastructure_issues': [
                'Insufficient memory',
                'CPU/GPU utilization issues',
                'Network latency',
                'Service dependencies down',
                'Configuration errors'
            ],
            'integration_issues': [
                'API version mismatch',
                'Authentication failures',
                'Rate limiting',
                'Timeout issues',
                'Serialization errors'
            ]
        }
    
    def check_common_issues(self, issue_logs):
        """Check if common issues are present"""
        detected_issues = {}
        
        for category, issues in self.common_issues.items():
            detected_issues[category] = []
            for issue in issues:
                # Check if issue keywords appear in logs
                if any(keyword.lower() in str(issue_logs).lower() 
                      for keyword in issue.split()):
                    detected_issues[category].append(issue)
        
        return detected_issues
```

## Key Takeaways

- ML debugging requires systematic approaches due to non-determinism, data dependencies, and complex pipelines
- Error analysis methodology helps identify patterns in failures and guides improvement efforts
- Model debugging tools (SHAP, LIME, TensorBoard, W&B) provide visibility into model behavior
- Performance profiling identifies bottlenecks in inference, memory usage, and pipeline stages
- Data debugging is critical as data issues are common sources of ML system failures
- Failure mode analysis helps identify common patterns and prevent future issues
- Incident response procedures minimize impact and enable quick service restoration
- Root cause analysis (5 Whys, fishbone diagrams) identifies underlying issues systematically
- Debugging checklists ensure comprehensive investigation and prevent overlooking common issues
- Effective debugging requires combining multiple tools and methodologies for comprehensive problem resolution
