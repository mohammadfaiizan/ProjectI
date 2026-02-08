# A/B Testing and Canary Deployment

## Table of Contents

1. [Introduction to A/B Testing for ML](#introduction-to-ab-testing-for-ml)
2. [A/B Testing Fundamentals](#ab-testing-fundamentals)
3. [Canary Releases](#canary-releases)
4. [Blue-Green Deployment](#blue-green-deployment)
5. [Shadow Mode Deployment](#shadow-mode-deployment)
6. [Gradual Rollout](#gradual-rollout)
7. [Statistical Significance](#statistical-significance)
8. [Metrics and Evaluation](#metrics-and-evaluation)
9. [Implementation Patterns](#implementation-patterns)
10. [Key Takeaways](#key-takeaways)

## Introduction to A/B Testing for ML

A/B testing for ML models compares different model versions to determine which performs better in production. Unlike traditional A/B testing, ML A/B testing involves:

- **Model Variants**: Different model architectures or versions
- **Performance Metrics**: Accuracy, latency, business metrics
- **Statistical Rigor**: Proper experimental design and analysis
- **Risk Mitigation**: Gradual rollout to minimize impact of failures

### Why A/B Testing for ML

- **Validate Improvements**: Confirm new models outperform old ones
- **Risk Reduction**: Test changes before full deployment
- **Business Impact**: Measure real-world business metrics
- **User Experience**: Ensure changes don't degrade user experience
- **Data-Driven Decisions**: Make deployment decisions based on data

### Challenges

- **Sample Size**: Need sufficient traffic for statistical significance
- **External Factors**: Seasonality, external events affect results
- **Multiple Metrics**: Balancing accuracy, latency, cost
- **Long-Term Effects**: Some impacts only visible over time

## A/B Testing Fundamentals

### Experimental Design

```
┌─────────────┐
│   Traffic   │
│  Splitter   │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│  A  │ │  B  │
│(Old)│ │(New)│
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       │
       ▼
┌─────────────┐
│   Metrics   │
│ Collection  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Statistical │
│  Analysis   │
└─────────────┘
```

### Traffic Splitting

```python
import random
import hashlib

class TrafficSplitter:
    def __init__(self, split_ratio=0.5):
        self.split_ratio = split_ratio
    
    def get_variant(self, user_id):
        """Deterministic assignment based on user_id"""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return 'A' if (hash_value % 100) < (self.split_ratio * 100) else 'B'
    
    def get_variant_random(self):
        """Random assignment"""
        return 'A' if random.random() < self.split_ratio else 'B'
```

### Request Routing

```python
class ABTestRouter:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.traffic_splitter = TrafficSplitter(split_ratio)
    
    def predict(self, request, user_id=None):
        variant = self.traffic_splitter.get_variant(user_id or request.id)
        
        if variant == 'A':
            prediction = self.model_a.predict(request)
            model_used = 'A'
        else:
            prediction = self.model_b.predict(request)
            model_used = 'B'
        
        # Log for analysis
        self.log_prediction(request.id, model_used, prediction)
        
        return prediction
```

## Canary Releases

Canary releases gradually expose new models to a small percentage of traffic.

### Canary Architecture

```
┌─────────────┐
│   Traffic   │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│ 90% │ │ 10% │
│ Old │ │New  │
│Model│ │Model│
└─────┘ └─────┘
```

### Implementation

```python
class CanaryDeployment:
    def __init__(self, old_model, new_model, initial_traffic=0.1):
        self.old_model = old_model
        self.new_model = new_model
        self.traffic_percentage = initial_traffic
        self.metrics_a = []
        self.metrics_b = []
    
    def predict(self, request):
        import random
        
        # Route based on traffic percentage
        use_new = random.random() < self.traffic_percentage
        
        if use_new:
            prediction = self.new_model.predict(request)
            variant = 'canary'
        else:
            prediction = self.old_model.predict(request)
            variant = 'stable'
        
        # Track metrics
        self.track_metrics(request, prediction, variant)
        
        return prediction
    
    def evaluate_and_scale(self):
        """Evaluate metrics and increase traffic if successful"""
        if self.is_canary_successful():
            # Gradually increase traffic
            self.traffic_percentage = min(
                self.traffic_percentage * 2,
                1.0
            )
            return True
        else:
            # Rollback
            self.traffic_percentage = 0
            return False
    
    def is_canary_successful(self):
        """Check if canary metrics are acceptable"""
        canary_metrics = self.metrics_b
        stable_metrics = self.metrics_a
        
        # Compare error rates, latency, etc.
        canary_error_rate = self.calculate_error_rate(canary_metrics)
        stable_error_rate = self.calculate_error_rate(stable_metrics)
        
        # Canary should not be significantly worse
        return canary_error_rate <= stable_error_rate * 1.1
```

### Gradual Traffic Increase

```python
class GradualCanary:
    def __init__(self, stages=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0]):
        self.stages = stages
        self.current_stage = 0
        self.evaluation_period = 3600  # 1 hour
    
    def get_traffic_percentage(self):
        return self.stages[self.current_stage]
    
    def advance_stage(self):
        """Move to next stage if current stage is successful"""
        if self.evaluate_current_stage():
            self.current_stage = min(
                self.current_stage + 1,
                len(self.stages) - 1
            )
            return True
        return False
```

## Blue-Green Deployment

Blue-Green deployment maintains two identical production environments.

### Architecture

```
┌─────────────┐
│   Router    │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│Blue │ │Green│
│(Old)│ │(New)│
└─────┘ └─────┘
```

### Implementation

```python
class BlueGreenDeployment:
    def __init__(self):
        self.blue_model = load_model('blue_model.pkl')
        self.green_model = None
        self.active_color = 'blue'
    
    def deploy_green(self, new_model_path):
        """Deploy new model to green environment"""
        self.green_model = load_model(new_model_path)
        # Run smoke tests
        if self.validate_green():
            return True
        else:
            self.green_model = None
            return False
    
    def switch_traffic(self, percentage=1.0):
        """Switch traffic from blue to green"""
        if self.green_model is None:
            raise ValueError("Green environment not deployed")
        
        # Gradually switch traffic
        self.traffic_split = {
            'blue': 1.0 - percentage,
            'green': percentage
        }
    
    def rollback(self):
        """Switch back to blue"""
        self.traffic_split = {'blue': 1.0, 'green': 0.0}
        self.green_model = None
    
    def predict(self, request):
        import random
        
        # Route based on traffic split
        rand = random.random()
        if rand < self.traffic_split.get('green', 0):
            return self.green_model.predict(request)
        else:
            return self.blue_model.predict(request)
```

## Shadow Mode Deployment

Shadow mode runs new models alongside production without affecting users.

### Architecture

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────┐    ┌──────────┐
│Production│    │  Shadow  │
│  Model   │    │  Model   │
└────┬─────┘    └────┬─────┘
     │              │
     │              │ (Log only)
     │              ▼
     │         ┌──────────┐
     │         │ Metrics   │
     │         │ Collection│
     │         └──────────┘
     │
     ▼
┌──────────┐
│ Response │
│ (to user)│
└──────────┘
```

### Implementation

```python
class ShadowMode:
    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.comparisons = []
    
    def predict(self, request):
        # Production prediction (returned to user)
        prod_prediction = self.production_model.predict(request)
        
        # Shadow prediction (logged for comparison)
        shadow_prediction = self.shadow_model.predict(request)
        
        # Compare and log
        comparison = {
            'request_id': request.id,
            'production_prediction': prod_prediction,
            'shadow_prediction': shadow_prediction,
            'difference': abs(prod_prediction - shadow_prediction),
            'timestamp': time.time()
        }
        self.comparisons.append(comparison)
        
        # Return production result
        return prod_prediction
    
    def analyze_shadow_results(self):
        """Analyze shadow mode results"""
        if len(self.comparisons) < 1000:
            return None
        
        differences = [c['difference'] for c in self.comparisons]
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        
        # Check if shadow model is ready
        if mean_diff < 0.05 and max_diff < 0.1:
            return {
                'ready_for_deployment': True,
                'mean_difference': mean_diff,
                'max_difference': max_diff
            }
        else:
            return {
                'ready_for_deployment': False,
                'mean_difference': mean_diff,
                'max_difference': max_diff
            }
```

## Gradual Rollout

Gradual rollout increases traffic to new models incrementally.

### Rollout Strategy

```python
class GradualRollout:
    def __init__(self):
        self.rollout_plan = [
            {'percentage': 0.01, 'duration': 3600},   # 1% for 1 hour
            {'percentage': 0.05, 'duration': 3600},   # 5% for 1 hour
            {'percentage': 0.10, 'duration': 7200},    # 10% for 2 hours
            {'percentage': 0.25, 'duration': 14400},  # 25% for 4 hours
            {'percentage': 0.50, 'duration': 28800},  # 50% for 8 hours
            {'percentage': 1.0, 'duration': None}     # 100% indefinitely
        ]
        self.current_stage = 0
        self.stage_start_time = time.time()
    
    def get_traffic_percentage(self):
        stage = self.rollout_plan[self.current_stage]
        
        # Check if stage duration has elapsed
        if stage['duration'] and \
           (time.time() - self.stage_start_time) > stage['duration']:
            # Move to next stage if successful
            if self.evaluate_stage():
                self.current_stage = min(
                    self.current_stage + 1,
                    len(self.rollout_plan) - 1
                )
                self.stage_start_time = time.time()
        
        return self.rollout_plan[self.current_stage]['percentage']
    
    def evaluate_stage(self):
        """Evaluate if current stage is successful"""
        # Check error rates, latency, business metrics
        error_rate = self.get_error_rate()
        latency_p95 = self.get_latency_p95()
        
        # Success criteria
        return error_rate < 0.01 and latency_p95 < 100
```

## Statistical Significance

### Hypothesis Testing

```python
from scipy import stats
import numpy as np

class ABTestAnalyzer:
    def __init__(self, group_a_metrics, group_b_metrics):
        self.group_a = group_a_metrics
        self.group_b = group_b_metrics
    
    def t_test(self):
        """Perform t-test for difference in means"""
        t_stat, p_value = stats.ttest_ind(self.group_a, self.group_b)
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def chi_square_test(self, group_a_success, group_a_total,
                       group_b_success, group_b_total):
        """Chi-square test for proportions"""
        contingency_table = [
            [group_a_success, group_a_total - group_a_success],
            [group_b_success, group_b_total - group_b_success]
        ]
        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def calculate_confidence_interval(self, confidence=0.95):
        """Calculate confidence interval for difference"""
        mean_a = np.mean(self.group_a)
        mean_b = np.mean(self.group_b)
        std_a = np.std(self.group_a, ddof=1)
        std_b = np.std(self.group_b, ddof=1)
        n_a = len(self.group_a)
        n_b = len(self.group_b)
        
        # Standard error
        se = np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
        
        # t-value for confidence level
        t_value = stats.t.ppf((1 + confidence) / 2, n_a + n_b - 2)
        
        # Confidence interval
        diff = mean_b - mean_a
        margin = t_value * se
        
        return {
            'difference': diff,
            'lower_bound': diff - margin,
            'upper_bound': diff + margin,
            'confidence': confidence
        }
```

### Sample Size Calculation

```python
def calculate_sample_size(effect_size, alpha=0.05, power=0.80):
    """Calculate required sample size for A/B test"""
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# Example
required_sample = calculate_sample_size(effect_size=0.1)
print(f"Required sample size per group: {required_sample}")
```

## Metrics and Evaluation

### Business Metrics

```python
class BusinessMetrics:
    def __init__(self):
        self.metrics = {
            'revenue': [],
            'conversion_rate': [],
            'user_engagement': [],
            'retention_rate': []
        }
    
    def track_metric(self, metric_name, value, variant):
        """Track business metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'value': value,
            'variant': variant,
            'timestamp': time.time()
        })
    
    def compare_variants(self, metric_name):
        """Compare variants for a metric"""
        variant_a = [m['value'] for m in self.metrics[metric_name] 
                     if m['variant'] == 'A']
        variant_b = [m['value'] for m in self.metrics[metric_name] 
                     if m['variant'] == 'B']
        
        analyzer = ABTestAnalyzer(variant_a, variant_b)
        return analyzer.t_test()
```

### Technical Metrics

```python
class TechnicalMetrics:
    def __init__(self):
        self.latencies = {'A': [], 'B': []}
        self.error_rates = {'A': 0, 'B': 0}
        self.request_counts = {'A': 0, 'B': 0}
    
    def record_request(self, variant, latency, success):
        """Record request metrics"""
        self.latencies[variant].append(latency)
        self.request_counts[variant] += 1
        if not success:
            self.error_rates[variant] += 1
    
    def get_latency_stats(self, variant):
        """Get latency statistics"""
        latencies = self.latencies[variant]
        return {
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def get_error_rate(self, variant):
        """Get error rate"""
        total = self.request_counts[variant]
        if total == 0:
            return 0
        return self.error_rates[variant] / total
```

## Implementation Patterns

### Feature Flags

```python
class FeatureFlagManager:
    def __init__(self):
        self.flags = {}
    
    def set_flag(self, flag_name, enabled, percentage=1.0):
        """Set feature flag"""
        self.flags[flag_name] = {
            'enabled': enabled,
            'percentage': percentage
        }
    
    def is_enabled(self, flag_name, user_id=None):
        """Check if feature flag is enabled"""
        if flag_name not in self.flags:
            return False
        
        flag = self.flags[flag_name]
        if not flag['enabled']:
            return False
        
        if flag['percentage'] < 1.0:
            # Percentage-based rollout
            hash_value = int(hashlib.md5(
                (user_id or str(random.random())).encode()
            ).hexdigest(), 16)
            return (hash_value % 100) < (flag['percentage'] * 100)
        
        return True
```

### Experiment Tracking

```python
class ExperimentTracker:
    def __init__(self):
        self.experiments = {}
    
    def start_experiment(self, experiment_id, config):
        """Start new experiment"""
        self.experiments[experiment_id] = {
            'config': config,
            'start_time': time.time(),
            'metrics': {'A': [], 'B': []}
        }
    
    def record_result(self, experiment_id, variant, metrics):
        """Record experiment result"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['metrics'][variant].append(metrics)
    
    def get_experiment_status(self, experiment_id):
        """Get experiment status"""
        exp = self.experiments[experiment_id]
        analyzer = ABTestAnalyzer(
            exp['metrics']['A'],
            exp['metrics']['B']
        )
        return {
            'duration': time.time() - exp['start_time'],
            'sample_size': {
                'A': len(exp['metrics']['A']),
                'B': len(exp['metrics']['B'])
            },
            'significance': analyzer.t_test()
        }
```

## Key Takeaways

- A/B testing validates model improvements and measures real-world impact
- Canary releases gradually expose new models to minimize risk
- Blue-Green deployment maintains two environments for instant rollback
- Shadow mode tests new models without affecting users
- Gradual rollout increases traffic incrementally based on success criteria
- Statistical significance ensures reliable conclusions from experiments
- Business and technical metrics both matter for comprehensive evaluation
- Feature flags enable flexible experiment management
- Proper experimental design requires adequate sample sizes and control groups
- A/B testing for ML requires careful consideration of multiple metrics and long-term effects
