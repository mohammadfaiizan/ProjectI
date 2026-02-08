# Cost Optimization and Resource Management

## Table of Contents

1. [Introduction to Cost Optimization](#introduction-to-cost-optimization)
2. [Cost Analysis](#cost-analysis)
3. [Auto-Scaling Strategies](#auto-scaling-strategies)
4. [Resource Allocation](#resource-allocation)
5. [Spot and Preemptible Instances](#spot-and-preemptible-instances)
6. [Model Optimization for Cost](#model-optimization-for-cost)
7. [FinOps for ML](#finops-for-ml)
8. [Cost Monitoring](#cost-monitoring)
9. [Budget Management](#budget-management)
10. [Key Takeaways](#key-takeaways)

## Introduction to Cost Optimization

ML systems can incur significant costs across:

- **Compute**: Training and inference infrastructure
- **Storage**: Data lakes, model artifacts, logs
- **Network**: Data transfer between services
- **Services**: Managed ML platforms, APIs

### Cost Drivers

- **Model Complexity**: Larger models require more resources
- **Training Frequency**: More frequent retraining increases costs
- **Inference Volume**: Higher request volumes increase serving costs
- **Data Volume**: Larger datasets increase storage and processing costs
- **Infrastructure Overhead**: Idle resources waste money

## Cost Analysis

### Cost Breakdown

```python
class CostAnalyzer:
    def __init__(self):
        self.costs = {
            'compute': {},
            'storage': {},
            'network': {},
            'services': {}
        }
    
    def analyze_training_costs(self, training_jobs):
        """Analyze training costs"""
        total_cost = 0
        
        for job in training_jobs:
            instance_cost = self.get_instance_cost(
                job['instance_type'],
                job['duration_hours']
            )
            storage_cost = self.get_storage_cost(
                job['data_size_gb'],
                job['duration_hours']
            )
            
            job_cost = instance_cost + storage_cost
            total_cost += job_cost
        
        return {
            'total_cost': total_cost,
            'per_job_costs': [job_cost for job in training_jobs],
            'average_cost': total_cost / len(training_jobs) if training_jobs else 0
        }
    
    def analyze_inference_costs(self, inference_metrics):
        """Analyze inference costs"""
        instance_cost = self.get_instance_cost(
            inference_metrics['instance_type'],
            inference_metrics['uptime_hours']
        )
        
        request_cost = (
            inference_metrics['request_count'] *
            inference_metrics['cost_per_request']
        )
        
        return {
            'instance_cost': instance_cost,
            'request_cost': request_cost,
            'total_cost': instance_cost + request_cost
        }
    
    def get_instance_cost(self, instance_type, hours):
        """Get instance cost"""
        hourly_rates = {
            'ml.m5.xlarge': 0.23,
            'ml.m5.2xlarge': 0.46,
            'ml.p3.2xlarge': 3.06,
            'ml.p3.8xlarge': 12.24
        }
        return hourly_rates.get(instance_type, 0) * hours
```

### Cost per Prediction

```python
class PredictionCostAnalyzer:
    def __init__(self):
        self.metrics = {
            'total_predictions': 0,
            'total_cost': 0,
            'instance_hours': 0
        }
    
    def calculate_cost_per_prediction(self):
        """Calculate cost per prediction"""
        if self.metrics['total_predictions'] == 0:
            return 0
        
        return self.metrics['total_cost'] / self.metrics['total_predictions']
    
    def track_prediction(self, cost):
        """Track prediction cost"""
        self.metrics['total_predictions'] += 1
        self.metrics['total_cost'] += cost
    
    def get_cost_breakdown(self):
        """Get cost breakdown"""
        return {
            'cost_per_prediction': self.calculate_cost_per_prediction(),
            'total_predictions': self.metrics['total_predictions'],
            'total_cost': self.metrics['total_cost'],
            'instance_cost_per_hour': (
                self.metrics['total_cost'] / self.metrics['instance_hours']
                if self.metrics['instance_hours'] > 0 else 0
            )
        }
```

## Auto-Scaling Strategies

### Horizontal Auto-Scaling

```python
class HorizontalAutoScaler:
    def __init__(self, min_replicas=1, max_replicas=10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        self.metrics_history = []
    
    def scale_decision(self, current_metrics):
        """Make scaling decision"""
        cpu_utilization = current_metrics.get('cpu_utilization', 0)
        request_rate = current_metrics.get('request_rate', 0)
        latency_p95 = current_metrics.get('latency_p95', 0)
        
        # Scale up conditions
        if cpu_utilization > 70 or latency_p95 > 200:
            if self.current_replicas < self.max_replicas:
                return 'scale_up'
        
        # Scale down conditions
        if cpu_utilization < 30 and request_rate < 100:
            if self.current_replicas > self.min_replicas:
                return 'scale_down'
        
        return 'no_change'
    
    def calculate_target_replicas(self, decision, current_replicas):
        """Calculate target number of replicas"""
        if decision == 'scale_up':
            return min(current_replicas * 2, self.max_replicas)
        elif decision == 'scale_down':
            return max(current_replicas // 2, self.min_replicas)
        else:
            return current_replicas
```

### Predictive Auto-Scaling

```python
class PredictiveAutoScaler:
    def __init__(self):
        self.request_history = []
        self.model = None
    
    def predict_demand(self, hours_ahead=1):
        """Predict future demand"""
        if len(self.request_history) < 100:
            return None
        
        # Simple moving average prediction
        recent_avg = np.mean(self.request_history[-24:])
        return recent_avg
    
    def scale_based_on_prediction(self):
        """Scale based on predicted demand"""
        predicted_demand = self.predict_demand()
        
        if predicted_demand:
            required_replicas = self.calculate_required_replicas(predicted_demand)
            return required_replicas
        
        return None
    
    def calculate_required_replicas(self, predicted_request_rate):
        """Calculate required replicas for predicted demand"""
        requests_per_replica = 100  # Capacity per replica
        return int(np.ceil(predicted_request_rate / requests_per_replica))
```

## Resource Allocation

### Resource Allocation Optimizer

```python
class ResourceAllocator:
    def __init__(self):
        self.available_resources = {
            'cpu_cores': 100,
            'memory_gb': 400,
            'gpus': 8
        }
        self.allocated_resources = {}
    
    def allocate_resources(self, job_id, requirements):
        """Allocate resources to job"""
        if self.can_allocate(requirements):
            self.allocated_resources[job_id] = requirements
            self.available_resources['cpu_cores'] -= requirements['cpu_cores']
            self.available_resources['memory_gb'] -= requirements['memory_gb']
            self.available_resources['gpus'] -= requirements.get('gpus', 0)
            return True
        return False
    
    def can_allocate(self, requirements):
        """Check if resources can be allocated"""
        return (
            requirements['cpu_cores'] <= self.available_resources['cpu_cores'] and
            requirements['memory_gb'] <= self.available_resources['memory_gb'] and
            requirements.get('gpus', 0) <= self.available_resources['gpus']
        )
    
    def release_resources(self, job_id):
        """Release allocated resources"""
        if job_id in self.allocated_resources:
            requirements = self.allocated_resources[job_id]
            self.available_resources['cpu_cores'] += requirements['cpu_cores']
            self.available_resources['memory_gb'] += requirements['memory_gb']
            self.available_resources['gpus'] += requirements.get('gpus', 0)
            del self.allocated_resources[job_id]
```

### Right-Sizing

```python
class RightSizingOptimizer:
    def __init__(self):
        self.instance_types = {
            'small': {'cpu': 2, 'memory': 4, 'cost': 0.05},
            'medium': {'cpu': 4, 'memory': 8, 'cost': 0.10},
            'large': {'cpu': 8, 'memory': 16, 'cost': 0.20},
            'xlarge': {'cpu': 16, 'memory': 32, 'cost': 0.40}
        }
    
    def recommend_instance(self, requirements, utilization_history):
        """Recommend instance type based on requirements and usage"""
        required_cpu = requirements['cpu']
        required_memory = requirements['memory']
        
        # Analyze utilization
        avg_cpu_util = np.mean([u['cpu'] for u in utilization_history])
        avg_memory_util = np.mean([u['memory'] for u in utilization_history])
        
        # Find smallest instance that meets requirements
        for size in ['small', 'medium', 'large', 'xlarge']:
            instance = self.instance_types[size]
            if (instance['cpu'] >= required_cpu and 
                instance['memory'] >= required_memory):
                return {
                    'instance_type': size,
                    'specs': instance,
                    'estimated_cost': instance['cost'],
                    'utilization': {
                        'cpu': avg_cpu_util / instance['cpu'] * 100,
                        'memory': avg_memory_util / instance['memory'] * 100
                    }
                }
        
        return None
```

## Spot and Preemptible Instances

### Spot Instance Manager

```python
class SpotInstanceManager:
    def __init__(self):
        self.spot_instances = {}
        self.checkpoint_interval = 300  # 5 minutes
    
    def launch_spot_instance(self, instance_type, max_price):
        """Launch spot instance"""
        instance = {
            'instance_id': f'spot-{uuid.uuid4()}',
            'instance_type': instance_type,
            'max_price': max_price,
            'status': 'running',
            'launch_time': time.time()
        }
        self.spot_instances[instance['instance_id']] = instance
        return instance
    
    def handle_interruption(self, instance_id):
        """Handle spot instance interruption"""
        if instance_id in self.spot_instances:
            instance = self.spot_instances[instance_id]
            instance['status'] = 'interrupted'
            instance['interruption_time'] = time.time()
            
            # Save checkpoint
            self.save_checkpoint(instance_id)
            
            # Request new spot instance
            new_instance = self.launch_spot_instance(
                instance['instance_type'],
                instance['max_price']
            )
            
            return new_instance
    
    def calculate_savings(self, instance_id):
        """Calculate cost savings from spot instance"""
        if instance_id not in self.spot_instances:
            return 0
        
        instance = self.spot_instances[instance_id]
        on_demand_price = self.get_on_demand_price(instance['instance_type'])
        spot_price = instance.get('actual_price', on_demand_price * 0.7)
        
        hours_used = (time.time() - instance['launch_time']) / 3600
        savings = (on_demand_price - spot_price) * hours_used
        
        return savings
```

## Model Optimization for Cost

### Model Compression

```python
class ModelCompressor:
    def __init__(self, model):
        self.model = model
    
    def quantize_model(self, quantization_type='int8'):
        """Quantize model to reduce size and inference cost"""
        if quantization_type == 'int8':
            # Post-training quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            return quantized_model
        elif quantization_type == 'fp16':
            # Half precision
            return self.model.half()
    
    def prune_model(self, sparsity=0.5):
        """Prune model to reduce size"""
        import torch.nn.utils.prune as prune
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
        
        return self.model
    
    def estimate_cost_reduction(self, original_model, optimized_model):
        """Estimate cost reduction from optimization"""
        original_size = self.get_model_size(original_model)
        optimized_size = self.get_model_size(optimized_model)
        
        size_reduction = (original_size - optimized_size) / original_size
        
        # Estimate inference cost reduction
        # Smaller models typically have lower latency and memory requirements
        cost_reduction = size_reduction * 0.7  # Assume 70% correlation
        
        return {
            'size_reduction': size_reduction,
            'estimated_cost_reduction': cost_reduction
        }
```

## FinOps for ML

### FinOps Framework

```python
class FinOpsManager:
    def __init__(self):
        self.cost_centers = {}
        self.budgets = {}
        self.alerts = []
    
    def allocate_cost(self, cost_center, amount, resource_type):
        """Allocate cost to cost center"""
        if cost_center not in self.cost_centers:
            self.cost_centers[cost_center] = {
                'total_cost': 0,
                'by_resource': {}
            }
        
        self.cost_centers[cost_center]['total_cost'] += amount
        if resource_type not in self.cost_centers[cost_center]['by_resource']:
            self.cost_centers[cost_center]['by_resource'][resource_type] = 0
        self.cost_centers[cost_center]['by_resource'][resource_type] += amount
    
    def set_budget(self, cost_center, budget_amount, period='monthly'):
        """Set budget for cost center"""
        self.budgets[cost_center] = {
            'amount': budget_amount,
            'period': period,
            'start_date': time.time()
        }
    
    def check_budget(self, cost_center):
        """Check if cost center is within budget"""
        if cost_center not in self.budgets:
            return None
        
        budget = self.budgets[cost_center]
        current_cost = self.cost_centers.get(cost_center, {}).get('total_cost', 0)
        
        budget_utilization = current_cost / budget['amount']
        
        if budget_utilization > 0.9:
            self.alerts.append({
                'cost_center': cost_center,
                'budget_utilization': budget_utilization,
                'alert_level': 'critical' if budget_utilization > 1.0 else 'warning'
            })
        
        return {
            'budget_utilization': budget_utilization,
            'remaining_budget': budget['amount'] - current_cost
        }
```

## Cost Monitoring

### Cost Dashboard

```python
class CostDashboard:
    def __init__(self):
        self.cost_data = []
    
    def add_cost_entry(self, timestamp, service, cost, resource_type):
        """Add cost entry"""
        self.cost_data.append({
            'timestamp': timestamp,
            'service': service,
            'cost': cost,
            'resource_type': resource_type
        })
    
    def get_cost_summary(self, start_date, end_date):
        """Get cost summary for period"""
        period_data = [
            entry for entry in self.cost_data
            if start_date <= entry['timestamp'] <= end_date
        ]
        
        total_cost = sum(entry['cost'] for entry in period_data)
        
        by_service = {}
        by_resource = {}
        
        for entry in period_data:
            # By service
            if entry['service'] not in by_service:
                by_service[entry['service']] = 0
            by_service[entry['service']] += entry['cost']
            
            # By resource
            if entry['resource_type'] not in by_resource:
                by_resource[entry['resource_type']] = 0
            by_resource[entry['resource_type']] += entry['cost']
        
        return {
            'total_cost': total_cost,
            'by_service': by_service,
            'by_resource': by_resource,
            'daily_average': total_cost / ((end_date - start_date) / 86400)
        }
```

## Budget Management

### Budget Alerts

```python
class BudgetManager:
    def __init__(self):
        self.budgets = {}
        self.spending = {}
    
    def set_budget(self, category, amount, period='monthly'):
        """Set budget"""
        self.budgets[category] = {
            'amount': amount,
            'period': period,
            'start_date': time.time()
        }
        if category not in self.spending:
            self.spending[category] = 0
    
    def track_spending(self, category, amount):
        """Track spending"""
        if category not in self.spending:
            self.spending[category] = 0
        self.spending[category] += amount
    
    def check_budgets(self):
        """Check all budgets and generate alerts"""
        alerts = []
        
        for category, budget in self.budgets.items():
            current_spending = self.spending.get(category, 0)
            utilization = current_spending / budget['amount']
            
            if utilization >= 1.0:
                alerts.append({
                    'category': category,
                    'level': 'critical',
                    'message': f'Budget exceeded for {category}'
                })
            elif utilization >= 0.9:
                alerts.append({
                    'category': category,
                    'level': 'warning',
                    'message': f'Approaching budget limit for {category}'
                })
        
        return alerts
```

## Key Takeaways

- Cost optimization requires understanding all cost drivers in ML systems
- Cost analysis identifies where money is being spent
- Auto-scaling ensures resources match demand
- Resource allocation optimization maximizes utilization
- Spot and preemptible instances offer significant cost savings
- Model optimization reduces inference costs
- FinOps provides framework for managing ML costs
- Cost monitoring enables data-driven optimization decisions
- Budget management prevents cost overruns
- Effective cost optimization balances performance, reliability, and budget constraints
