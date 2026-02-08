# Data Drift and Concept Drift

## Table of Contents

1. [Introduction to Drift](#introduction-to-drift)
2. [Types of Data Drift](#types-of-data-drift)
3. [Concept Drift Fundamentals](#concept-drift-fundamentals)
4. [Drift Detection Methods](#drift-detection-methods)
5. [Statistical Tests for Drift](#statistical-tests-for-drift)
6. [Monitoring Strategies](#monitoring-strategies)
7. [Retraining Triggers](#retraining-triggers)
8. [Windowing Strategies](#windowing-strategies)
9. [Adaptive Learning Approaches](#adaptive-learning-approaches)
10. [Key Takeaways](#key-takeaways)

## Introduction to Drift

Drift refers to changes in the underlying data distribution or the relationship between inputs and outputs over time. In production ML systems, drift is inevitable and can significantly degrade model performance if not detected and addressed promptly.

### Drift Categories

**Data Drift (Covariate Shift)**:
- Changes in the distribution of input features
- The relationship P(Y|X) remains constant
- Model performance may degrade due to distribution mismatch

**Concept Drift (Label Shift)**:
- Changes in the relationship P(Y|X) between inputs and outputs
- Input distribution may remain constant
- Requires model retraining or adaptation

**Prior Probability Shift**:
- Changes in the distribution of target variable P(Y)
- Input distribution P(X) and relationship P(Y|X) remain constant
- Common in imbalanced classification scenarios

### Impact of Drift

Drift can cause:
- **Performance Degradation**: Model accuracy decreases over time
- **Business Impact**: Reduced revenue, increased costs, poor user experience
- **Compliance Issues**: Models may violate fairness or accuracy requirements
- **Operational Problems**: Increased false positives/negatives, system failures

### Drift Detection Pipeline

```
┌──────────────┐
│  Production   │
│     Data     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Drift      │
│  Detection   │
└──────┬───────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│   Alert      │ │  Retrain     │
│   System     │ │   Trigger    │
└──────────────┘ └──────────────┘
```

## Types of Data Drift

Data drift occurs when the distribution of input features changes over time, potentially causing the model to encounter inputs outside its training distribution.

### Covariate Shift

Covariate shift occurs when P(X) changes but P(Y|X) remains constant:

**Characteristics**:
- Input feature distributions shift
- Conditional distribution P(Y|X) unchanged
- Model may perform poorly on new regions of feature space

**Example**:
- E-commerce model trained on US customers, deployed globally
- Feature distributions (age, income, preferences) differ by region
- Purchase behavior relationship remains similar

**Detection**:
```python
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp

class CovariateShiftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_distributions = self._estimate_distributions(reference_data)
    
    def _estimate_distributions(self, data):
        """Estimate distributions for each feature"""
        distributions = {}
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                distributions[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                }
            else:
                distributions[col] = data[col].value_counts(normalize=True).to_dict()
        return distributions
    
    def detect_shift(self, current_data, threshold=0.05):
        """Detect covariate shift using Kolmogorov-Smirnov test"""
        drift_results = {}
        
        for col in self.reference_distributions.keys():
            if col in current_data.columns:
                ref_values = self.reference_data[col].values
                curr_values = current_data[col].values
                
                # KS test for continuous features
                if self.reference_data[col].dtype in [np.float64, np.int64]:
                    statistic, p_value = ks_2samp(ref_values, curr_values)
                    drift_results[col] = {
                        'drift_detected': p_value < threshold,
                        'ks_statistic': statistic,
                        'p_value': p_value
                    }
                else:
                    # Chi-square test for categorical features
                    ref_counts = self.reference_data[col].value_counts()
                    curr_counts = current_data[col].value_counts()
                    
                    # Align categories
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                    
                    chi2, p_value = stats.chisquare(curr_aligned, ref_aligned)
                    drift_results[col] = {
                        'drift_detected': p_value < threshold,
                        'chi2_statistic': chi2,
                        'p_value': p_value
                    }
        
        return drift_results
```

### Prior Probability Shift

Prior probability shift occurs when P(Y) changes but P(X|Y) and P(Y|X) remain constant:

**Characteristics**:
- Target variable distribution changes
- Feature distributions conditional on class remain constant
- Common in imbalanced learning scenarios

**Example**:
- Fraud detection model trained when fraud rate was 1%
- Fraud rate increases to 5% over time
- Relationship between features and fraud remains the same

**Detection**:
```python
class PriorProbabilityShiftDetector:
    def __init__(self, reference_labels):
        self.reference_distribution = self._estimate_label_distribution(reference_labels)
    
    def _estimate_label_distribution(self, labels):
        """Estimate label distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts / len(labels)))
    
    def detect_shift(self, current_labels, threshold=0.05):
        """Detect prior probability shift"""
        current_distribution = self._estimate_label_distribution(current_labels)
        
        # Calculate KL divergence
        kl_divergence = 0
        all_labels = set(self.reference_distribution.keys()) | set(current_distribution.keys())
        
        for label in all_labels:
            ref_prob = self.reference_distribution.get(label, 1e-10)
            curr_prob = current_distribution.get(label, 1e-10)
            kl_divergence += ref_prob * np.log(ref_prob / curr_prob)
        
        # Chi-square test
        ref_counts = [self.reference_distribution.get(l, 0) * len(current_labels) 
                     for l in sorted(all_labels)]
        curr_counts = [current_distribution.get(l, 0) * len(current_labels) 
                      for l in sorted(all_labels)]
        
        chi2, p_value = stats.chisquare(curr_counts, ref_counts)
        
        return {
            'drift_detected': p_value < threshold,
            'kl_divergence': kl_divergence,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'reference_distribution': self.reference_distribution,
            'current_distribution': current_distribution
        }
```

### Feature Interaction Drift

Changes in relationships between features, even if marginal distributions remain constant:

**Detection**:
```python
class FeatureInteractionDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_correlations = reference_data.corr()
    
    def detect_shift(self, current_data, threshold=0.1):
        """Detect changes in feature correlations"""
        current_correlations = current_data.corr()
        
        # Compare correlation matrices
        correlation_diff = np.abs(
            self.reference_correlations - current_correlations
        )
        
        drift_detected = correlation_diff > threshold
        
        return {
            'drift_detected': drift_detected.any().any(),
            'max_correlation_change': correlation_diff.max().max(),
            'affected_pairs': self._get_affected_pairs(correlation_diff, threshold),
            'correlation_diff_matrix': correlation_diff
        }
    
    def _get_affected_pairs(self, diff_matrix, threshold):
        """Get feature pairs with significant correlation changes"""
        affected = []
        for i in range(len(diff_matrix.columns)):
            for j in range(i+1, len(diff_matrix.columns)):
                if diff_matrix.iloc[i, j] > threshold:
                    affected.append((
                        diff_matrix.columns[i],
                        diff_matrix.columns[j],
                        diff_matrix.iloc[i, j]
                    ))
        return affected
```

## Concept Drift Fundamentals

Concept drift occurs when the relationship P(Y|X) between inputs and outputs changes over time, requiring model adaptation or retraining.

### Types of Concept Drift

**Sudden Drift**:
- Abrupt change in the concept
- Example: Policy change, market crash
- Requires immediate detection and response

**Gradual Drift**:
- Slow, continuous change over time
- Example: Changing user preferences
- May be difficult to detect early

**Incremental Drift**:
- Step-wise changes at specific points
- Example: Seasonal patterns, product updates
- Requires periodic monitoring

**Recurring Drift**:
- Concept returns to previous states
- Example: Seasonal trends, cyclical patterns
- May benefit from ensemble approaches

### Concept Drift Detection

```python
class ConceptDriftDetector:
    def __init__(self, window_size=1000, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = []
        self.current_window = []
    
    def add_prediction(self, prediction, true_label):
        """Add prediction and label to current window"""
        self.current_window.append({
            'prediction': prediction,
            'true_label': true_label,
            'error': abs(prediction - true_label) if isinstance(prediction, (int, float)) 
                    else (prediction != true_label)
        })
        
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
    
    def detect_drift(self):
        """Detect concept drift using error rate comparison"""
        if len(self.reference_window) < self.window_size:
            return {'drift_detected': False, 'reason': 'insufficient_reference_data'}
        
        if len(self.current_window) < self.window_size:
            return {'drift_detected': False, 'reason': 'insufficient_current_data'}
        
        ref_errors = [item['error'] for item in self.reference_window]
        curr_errors = [item['error'] for item in self.current_window]
        
        ref_error_rate = np.mean(ref_errors)
        curr_error_rate = np.mean(curr_errors)
        
        # Statistical test for error rate change
        from scipy.stats import proportions_ztest
        
        ref_successes = len(self.reference_window) - sum(ref_errors)
        curr_successes = len(self.current_window) - sum(curr_errors)
        
        count = np.array([ref_successes, curr_successes])
        nobs = np.array([len(self.reference_window), len(self.current_window)])
        
        z_stat, p_value = proportions_ztest(count, nobs)
        
        drift_detected = p_value < self.threshold
        
        return {
            'drift_detected': drift_detected,
            'reference_error_rate': ref_error_rate,
            'current_error_rate': curr_error_rate,
            'error_rate_change': curr_error_rate - ref_error_rate,
            'z_statistic': z_stat,
            'p_value': p_value
        }
    
    def update_reference(self):
        """Update reference window with current window"""
        self.reference_window = self.current_window.copy()
```

## Drift Detection Methods

Various methods exist for detecting drift, each with different strengths and use cases.

### Population Stability Index (PSI)

PSI measures the difference between two distributions:

```python
def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index"""
    def scale_range(start, end, length):
        return np.linspace(start, end, length + 1)
    
    # Create buckets
    breakpoints = scale_range(min(expected), max(expected), buckets)
    
    # Calculate expected and actual distributions
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 1e-10, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-10, actual_percents)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi

class PSIDriftDetector:
    def __init__(self, reference_data, threshold=0.2):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, current_data):
        """Detect drift using PSI"""
        drift_results = {}
        
        for col in self.reference_data.columns:
            if col in current_data.columns:
                psi = calculate_psi(
                    self.reference_data[col].values,
                    current_data[col].values
                )
                
                # PSI interpretation:
                # < 0.1: No significant change
                # 0.1 - 0.2: Minor change
                # > 0.2: Significant change
                drift_results[col] = {
                    'psi': psi,
                    'drift_detected': psi > self.threshold,
                    'severity': 'high' if psi > 0.25 else 'medium' if psi > 0.1 else 'low'
                }
        
        return drift_results
```

### Maximum Mean Discrepancy (MMD)

MMD measures the distance between distributions in a reproducing kernel Hilbert space:

```python
from sklearn.metrics.pairwise import rbf_kernel

def mmd_rbf(X, Y, gamma=1.0):
    """Calculate Maximum Mean Discrepancy using RBF kernel"""
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

class MMDDriftDetector:
    def __init__(self, reference_data, threshold=None):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, current_data, gamma=1.0):
        """Detect drift using MMD"""
        # Select numeric columns
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        drift_results = {}
        
        for col in numeric_cols:
            if col in current_data.columns:
                ref_values = self.reference_data[[col]].values
                curr_values = current_data[[col]].values
                
                mmd_value = mmd_rbf(ref_values, curr_values, gamma=gamma)
                
                # If threshold not set, use permutation test
                if self.threshold is None:
                    p_value = self._permutation_test(ref_values, curr_values, gamma)
                    drift_detected = p_value < 0.05
                else:
                    drift_detected = mmd_value > self.threshold
                    p_value = None
                
                drift_results[col] = {
                    'mmd': mmd_value,
                    'drift_detected': drift_detected,
                    'p_value': p_value
                }
        
        return drift_results
    
    def _permutation_test(self, X, Y, gamma, n_permutations=1000):
        """Permutation test for MMD significance"""
        mmd_observed = mmd_rbf(X, Y, gamma)
        
        # Combine and permute
        combined = np.vstack([X, Y])
        n_X = len(X)
        
        mmd_permuted = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            X_perm = combined[:n_X]
            Y_perm = combined[n_X:]
            mmd_permuted.append(mmd_rbf(X_perm, Y_perm, gamma))
        
        p_value = np.mean(np.array(mmd_permuted) >= mmd_observed)
        return p_value
```

### KL Divergence

Kullback-Leibler divergence measures the difference between probability distributions:

```python
from scipy.stats import entropy

def calculate_kl_divergence(P, Q):
    """Calculate KL divergence D_KL(P || Q)"""
    # Ensure probabilities sum to 1
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Avoid log(0)
    Q = np.where(Q == 0, 1e-10, Q)
    
    kl_div = entropy(P, Q)
    return kl_div

class KLDivergenceDetector:
    def __init__(self, reference_data, bins=50):
        self.reference_data = reference_data
        self.bins = bins
        self.reference_histograms = self._compute_histograms(reference_data)
    
    def _compute_histograms(self, data):
        """Compute histograms for each feature"""
        histograms = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            hist, bins = np.histogram(data[col].values, bins=self.bins)
            histograms[col] = {'hist': hist, 'bins': bins}
        return histograms
    
    def detect_drift(self, current_data, threshold=0.1):
        """Detect drift using KL divergence"""
        drift_results = {}
        
        for col, ref_hist_info in self.reference_histograms.items():
            if col in current_data.columns:
                # Compute current histogram using same bins
                curr_hist, _ = np.histogram(
                    current_data[col].values,
                    bins=ref_hist_info['bins']
                )
                
                kl_div = calculate_kl_divergence(
                    ref_hist_info['hist'],
                    curr_hist
                )
                
                drift_results[col] = {
                    'kl_divergence': kl_div,
                    'drift_detected': kl_div > threshold
                }
        
        return drift_results
```

## Statistical Tests for Drift

Statistical hypothesis tests provide formal methods for drift detection with known error rates.

### Kolmogorov-Smirnov Test

Two-sample KS test compares empirical distributions:

```python
from scipy.stats import ks_2samp

class KSTestDetector:
    def __init__(self, reference_data, alpha=0.05):
        self.reference_data = reference_data
        self.alpha = alpha
    
    def detect_drift(self, current_data):
        """Detect drift using Kolmogorov-Smirnov test"""
        drift_results = {}
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in current_data.columns:
                statistic, p_value = ks_2samp(
                    self.reference_data[col].values,
                    current_data[col].values
                )
                
                drift_results[col] = {
                    'ks_statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < self.alpha
                }
        
        return drift_results
```

### Chi-Squared Test

Chi-squared test for categorical features:

```python
from scipy.stats import chisquare

class ChiSquareDetector:
    def __init__(self, reference_data, alpha=0.05):
        self.reference_data = reference_data
        self.alpha = alpha
        self.reference_counts = self._compute_counts(reference_data)
    
    def _compute_counts(self, data):
        """Compute value counts for categorical features"""
        counts = {}
        for col in data.select_dtypes(include=['object', 'category']).columns:
            counts[col] = data[col].value_counts().to_dict()
        return counts
    
    def detect_drift(self, current_data):
        """Detect drift using chi-squared test"""
        drift_results = {}
        
        for col, ref_counts in self.reference_counts.items():
            if col in current_data.columns:
                curr_counts = current_data[col].value_counts().to_dict()
                
                # Align categories
                all_categories = set(ref_counts.keys()) | set(curr_counts.keys())
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                # Normalize to expected frequencies
                total_ref = sum(ref_aligned)
                total_curr = sum(curr_aligned)
                expected = [count * total_curr / total_ref for count in ref_aligned]
                
                chi2, p_value = chisquare(curr_aligned, expected)
                
                drift_results[col] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'drift_detected': p_value < self.alpha
                }
        
        return drift_results
```

### Anderson-Darling Test

More sensitive to tail differences than KS test:

```python
from scipy.stats import anderson_ksamp

class AndersonDarlingDetector:
    def __init__(self, reference_data, alpha=0.05):
        self.reference_data = reference_data
        self.alpha = alpha
    
    def detect_drift(self, current_data):
        """Detect drift using Anderson-Darling test"""
        drift_results = {}
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in current_data.columns:
                statistic, critical_values, p_value = anderson_ksamp([
                    self.reference_data[col].values,
                    current_data[col].values
                ])
                
                drift_results[col] = {
                    'ad_statistic': statistic,
                    'critical_value': critical_values[2] if len(critical_values) > 2 else None,
                    'p_value': p_value,
                    'drift_detected': statistic > critical_values[2] if len(critical_values) > 2 else False
                }
        
        return drift_results
```

## Monitoring Strategies

Effective drift monitoring requires strategic approaches to data collection, analysis, and alerting.

### Monitoring Architecture

```
┌──────────────┐
│  Production   │
│   Predictions │
└──────┬───────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│   Feature    │ │   Prediction │
│   Monitor    │ │    Monitor   │
└──────┬───────┘ └──────┬───────┘
       │                │
       └────────┬───────┘
                │
                ▼
         ┌──────────────┐
         │   Drift      │
         │  Detection   │
         │   Engine     │
         └──────┬───────┘
                │
         ┌──────┴───────┐
         │              │
         ▼              ▼
    ┌──────────┐  ┌──────────┐
    │ Alerts   │  │Dashboard │
    └──────────┘  └──────────┘
```

### Continuous Monitoring

Monitor drift continuously as new data arrives:

```python
class ContinuousDriftMonitor:
    def __init__(self, detection_method, window_size=1000, check_interval=100):
        self.detection_method = detection_method
        self.window_size = window_size
        self.check_interval = check_interval
        self.data_buffer = []
        self.drift_history = []
    
    def add_data_point(self, data_point):
        """Add new data point to buffer"""
        self.data_buffer.append(data_point)
        
        # Remove old data if buffer exceeds window size
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
        
        # Check for drift periodically
        if len(self.data_buffer) % self.check_interval == 0:
            return self.check_drift()
        
        return None
    
    def check_drift(self):
        """Check for drift in current window"""
        if len(self.data_buffer) < self.window_size:
            return None
        
        current_window = pd.DataFrame(self.data_buffer[-self.window_size:])
        drift_result = self.detection_method.detect_drift(current_window)
        
        self.drift_history.append({
            'timestamp': datetime.now(),
            'result': drift_result
        })
        
        return drift_result
```

### Scheduled Monitoring

Monitor drift on a fixed schedule:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def check_drift_daily(**context):
    """Daily drift check task"""
    execution_date = context['execution_date']
    
    # Load reference data
    reference_data = load_reference_data()
    
    # Load current day's data
    current_data = load_production_data(
        start_date=execution_date - timedelta(days=1),
        end_date=execution_date
    )
    
    # Detect drift
    detector = PSIDriftDetector(reference_data, threshold=0.2)
    drift_results = detector.detect_drift(current_data)
    
    # Alert if drift detected
    if any(r['drift_detected'] for r in drift_results.values()):
        send_alert(drift_results)
    
    # Log results
    log_drift_results(execution_date, drift_results)

dag = DAG(
    'drift_monitoring',
    default_args={
        'owner': 'ml_ops',
        'start_date': datetime(2024, 1, 1),
        'retries': 1
    },
    schedule_interval='@daily'
)

drift_check_task = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift_daily,
    dag=dag
)
```

### Multi-Level Monitoring

Monitor at different granularities:

```python
class MultiLevelMonitor:
    def __init__(self):
        self.feature_level_monitors = {}
        self.model_level_monitor = None
        self.system_level_monitor = None
    
    def monitor_feature_level(self, feature_name, data):
        """Monitor individual features"""
        if feature_name not in self.feature_level_monitors:
            self.feature_level_monitors[feature_name] = PSIDriftDetector(
                reference_data=data[:1000],
                threshold=0.2
            )
        
        detector = self.feature_level_monitors[feature_name]
        return detector.detect_drift(data[-100:])
    
    def monitor_model_level(self, predictions, labels):
        """Monitor overall model performance"""
        if self.model_level_monitor is None:
            self.model_level_monitor = ConceptDriftDetector()
        
        for pred, label in zip(predictions, labels):
            self.model_level_monitor.add_prediction(pred, label)
        
        return self.model_level_monitor.detect_drift()
    
    def monitor_system_level(self, metrics):
        """Monitor system-level metrics"""
        # Monitor latency, throughput, error rates
        pass
```

## Retraining Triggers

Automated retraining triggers based on drift detection and performance metrics.

### Performance-Based Triggers

```python
class PerformanceBasedTrigger:
    def __init__(self, baseline_performance, degradation_threshold=0.05):
        self.baseline_performance = baseline_performance
        self.degradation_threshold = degradation_threshold
        self.performance_history = []
    
    def evaluate(self, current_performance):
        """Evaluate if retraining is needed based on performance"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'performance': current_performance
        })
        
        performance_degradation = (
            self.baseline_performance - current_performance
        ) / self.baseline_performance
        
        should_retrain = performance_degradation > self.degradation_threshold
        
        return {
            'should_retrain': should_retrain,
            'performance_degradation': performance_degradation,
            'baseline': self.baseline_performance,
            'current': current_performance
        }
```

### Drift-Based Triggers

```python
class DriftBasedTrigger:
    def __init__(self, drift_threshold=0.2, severity_threshold=0.25):
        self.drift_threshold = drift_threshold
        self.severity_threshold = severity_threshold
        self.drift_history = []
    
    def evaluate(self, drift_results):
        """Evaluate if retraining is needed based on drift"""
        # Count features with drift
        drifted_features = [
            name for name, result in drift_results.items()
            if result.get('drift_detected', False)
        ]
        
        # Calculate average drift severity
        drift_scores = [
            result.get('psi', 0) or result.get('ks_statistic', 0) or 0
            for result in drift_results.values()
        ]
        avg_drift_severity = np.mean(drift_scores) if drift_scores else 0
        
        # Determine if retraining needed
        should_retrain = (
            len(drifted_features) > len(drift_results) * 0.3 or
            avg_drift_severity > self.severity_threshold
        )
        
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drifted_features': drifted_features,
            'avg_severity': avg_drift_severity,
            'should_retrain': should_retrain
        })
        
        return {
            'should_retrain': should_retrain,
            'drifted_features': drifted_features,
            'drift_severity': avg_drift_severity,
            'reason': 'feature_drift' if drifted_features else 'no_drift'
        }
```

### Combined Trigger Strategy

```python
class CombinedRetrainingTrigger:
    def __init__(self):
        self.performance_trigger = PerformanceBasedTrigger(
            baseline_performance=0.95,
            degradation_threshold=0.05
        )
        self.drift_trigger = DriftBasedTrigger(
            drift_threshold=0.2,
            severity_threshold=0.25
        )
    
    def evaluate(self, current_performance, drift_results):
        """Evaluate using both performance and drift"""
        perf_result = self.performance_trigger.evaluate(current_performance)
        drift_result = self.drift_trigger.evaluate(drift_results)
        
        # Retrain if either condition met
        should_retrain = (
            perf_result['should_retrain'] or
            drift_result['should_retrain']
        )
        
        return {
            'should_retrain': should_retrain,
            'performance_based': perf_result['should_retrain'],
            'drift_based': drift_result['should_retrain'],
            'performance_degradation': perf_result['performance_degradation'],
            'drift_severity': drift_result['drift_severity']
        }
```

## Windowing Strategies

Windowing strategies determine how historical data is used for drift detection and model updates.

### Fixed Window

Use a fixed-size window of recent data:

```python
class FixedWindowStrategy:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.window = []
    
    def add_data(self, data_point):
        """Add data point to window"""
        self.window.append(data_point)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
    def get_window(self):
        """Get current window"""
        return self.window.copy()
```

### Sliding Window

Continuously update window with new data:

```python
class SlidingWindowStrategy:
    def __init__(self, window_size=1000, slide_size=100):
        self.window_size = window_size
        self.slide_size = slide_size
        self.window = []
    
    def add_data(self, data_point):
        """Add data point and slide window if needed"""
        self.window.append(data_point)
        if len(self.window) > self.window_size:
            # Remove oldest slide_size elements
            self.window = self.window[self.slide_size:]
    
    def get_window(self):
        """Get current window"""
        return self.window.copy()
```

### Adaptive Window

Adjust window size based on drift detection:

```python
class AdaptiveWindowStrategy:
    def __init__(self, min_window_size=500, max_window_size=5000, initial_size=1000):
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.window_size = initial_size
        self.window = []
    
    def add_data(self, data_point):
        """Add data point to window"""
        self.window.append(data_point)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
    def adjust_window_size(self, drift_detected, drift_severity):
        """Adjust window size based on drift"""
        if drift_detected:
            # Increase window size to capture more history
            self.window_size = min(
                self.window_size + 500,
                self.max_window_size
            )
        else:
            # Decrease window size for faster adaptation
            self.window_size = max(
                self.window_size - 100,
                self.min_window_size
            )
    
    def get_window(self):
        """Get current window"""
        return self.window.copy()
```

### Time-Based Window

Use time-based windows instead of sample-based:

```python
class TimeBasedWindowStrategy:
    def __init__(self, window_duration_hours=24):
        self.window_duration = timedelta(hours=window_duration_hours)
        self.window = []
    
    def add_data(self, data_point, timestamp):
        """Add data point with timestamp"""
        self.window.append({
            'data': data_point,
            'timestamp': timestamp
        })
        
        # Remove data outside window
        cutoff_time = timestamp - self.window_duration
        self.window = [
            item for item in self.window
            if item['timestamp'] > cutoff_time
        ]
    
    def get_window(self):
        """Get current window"""
        return [item['data'] for item in self.window]
```

## Adaptive Learning Approaches

Adaptive learning techniques update models incrementally to handle drift without full retraining.

### Online Learning

Update model with each new data point:

```python
from sklearn.linear_model import SGDClassifier

class OnlineLearner:
    def __init__(self, base_model=None):
        if base_model is None:
            self.model = SGDClassifier(loss='log_loss', learning_rate='adaptive')
        else:
            self.model = base_model
    
    def partial_fit(self, X, y):
        """Update model with new data"""
        self.model.partial_fit(X, y, classes=np.unique(y))
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def adapt_to_drift(self, X_new, y_new):
        """Adapt model to new data distribution"""
        # Use higher learning rate for adaptation
        original_eta0 = self.model.eta0
        self.model.eta0 = original_eta0 * 2
        
        self.partial_fit(X_new, y_new)
        
        # Restore learning rate
        self.model.eta0 = original_eta0
```

### Ensemble Methods

Use ensemble of models with different update frequencies:

```python
class AdaptiveEnsemble:
    def __init__(self, base_models, update_frequencies):
        self.models = base_models
        self.update_frequencies = update_frequencies
        self.update_counts = [0] * len(base_models)
    
    def update(self, X, y, model_index=None):
        """Update specific model or all models"""
        if model_index is not None:
            self.models[model_index].partial_fit(X, y)
            self.update_counts[model_index] += 1
        else:
            for i, model in enumerate(self.models):
                if self.update_counts[i] % self.update_frequencies[i] == 0:
                    model.partial_fit(X, y)
                self.update_counts[i] += 1
    
    def predict(self, X):
        """Weighted ensemble prediction"""
        predictions = [model.predict(X) for model in self.models]
        
        # Weight by recency (more recently updated models get higher weight)
        weights = [
            1.0 / (count + 1) for count in self.update_counts
        ]
        weights = np.array(weights) / sum(weights)
        
        # Weighted voting
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
```

### Concept Drift Adaptation

Detect and adapt to concept drift:

```python
class ConceptDriftAdapter:
    def __init__(self, base_model, drift_detector):
        self.model = base_model
        self.drift_detector = drift_detector
        self.adaptation_history = []
    
    def adapt(self, X_new, y_new):
        """Adapt model when drift detected"""
        # Check for drift
        drift_result = self.drift_detector.detect_drift()
        
        if drift_result['drift_detected']:
            # Increase learning rate for adaptation
            if hasattr(self.model, 'learning_rate'):
                original_lr = self.model.learning_rate
                self.model.learning_rate = original_lr * 2
            
            # Update model with new data
            self.model.partial_fit(X_new, y_new)
            
            # Restore learning rate
            if hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = original_lr
            
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'drift_severity': drift_result.get('severity', 0),
                'adaptation_applied': True
            })
        
        return drift_result
```

## Key Takeaways

- Drift is inevitable in production ML systems and requires systematic detection and response strategies
- Data drift (covariate shift) and concept drift represent different types of distribution changes requiring different approaches
- Multiple detection methods (PSI, KS test, MMD, KL divergence) provide complementary views of drift
- Statistical tests provide formal hypothesis testing with known error rates for drift detection
- Monitoring strategies should combine continuous and scheduled approaches with multi-level granularity
- Retraining triggers should consider both performance degradation and drift severity
- Windowing strategies balance between capturing sufficient history and adapting to recent changes
- Adaptive learning approaches enable incremental model updates without full retraining
- Effective drift management requires integration of detection, monitoring, alerting, and adaptation systems
- Drift dashboards should provide visibility into drift trends, affected features, and model performance over time
