# Compute Optimization and GPU Clusters

## Table of Contents

1. [Introduction to GPU Computing](#introduction-to-gpu-computing)
2. [GPU Utilization Optimization](#gpu-utilization-optimization)
3. [Mixed Precision Training](#mixed-precision-training)
4. [Cluster Management](#cluster-management)
5. [SLURM Workload Manager](#slurm-workload-manager)
6. [Spot Instances and Cost Optimization](#spot-instances-and-cost-optimization)
7. [Multi-GPU Training](#multi-gpu-training)
8. [Performance Profiling](#performance-profiling)
9. [Resource Scheduling](#resource-scheduling)
10. [Key Takeaways](#key-takeaways)

## Introduction to GPU Computing

GPUs provide parallel processing power essential for deep learning:

- **Parallelism**: Thousands of cores for matrix operations
- **Memory Bandwidth**: High bandwidth for large model weights
- **Specialized Operations**: Optimized for neural network operations
- **Scalability**: Multiple GPUs for distributed training

### GPU Architecture

```
┌─────────────────────────────────┐
│         GPU Device              │
│  ┌───────────────────────────┐  │
│  │    Streaming Multiprocessors│  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐  │  │
│  │  │ CUDA│ │ CUDA│ │ CUDA│  │  │
│  │  │ Core│ │ Core│ │ Core│  │  │
│  │  └─────┘ └─────┘ └─────┘  │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │    Global Memory (VRAM)   │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

## GPU Utilization Optimization

### Monitoring GPU Usage

```python
import torch
import pynvml

class GPUMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
    
    def get_gpu_utilization(self, device_id=0):
        """Get GPU utilization percentage"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    
    def get_memory_usage(self, device_id=0):
        """Get GPU memory usage"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'used': mem_info.used,
            'total': mem_info.total,
            'free': mem_info.free,
            'usage_percent': (mem_info.used / mem_info.total) * 100
        }
    
    def get_temperature(self, device_id=0):
        """Get GPU temperature"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    
    def monitor_all_gpus(self):
        """Monitor all GPUs"""
        stats = []
        for i in range(self.device_count):
            stats.append({
                'device_id': i,
                'utilization': self.get_gpu_utilization(i),
                'memory': self.get_memory_usage(i),
                'temperature': self.get_temperature(i)
            })
        return stats
```

### Optimizing Data Loading

```python
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size, num_workers=4, pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def create_loader(self):
        """Create optimized data loader"""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            persistent_workers=True
        )
```

### Batch Size Optimization

```python
class BatchSizeOptimizer:
    def __init__(self, model, input_shape, device):
        self.model = model
        self.input_shape = input_shape
        self.device = device
    
    def find_optimal_batch_size(self, start_size=1, max_size=1024):
        """Find optimal batch size"""
        optimal_size = start_size
        
        for batch_size in [start_size * 2**i for i in range(10)]:
            if batch_size > max_size:
                break
            
            try:
                # Try to allocate memory
                dummy_input = torch.randn(
                    batch_size, *self.input_shape
                ).to(self.device)
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                optimal_size = batch_size
                del dummy_input
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    break
                raise
        
        return optimal_size
```

## Mixed Precision Training

### Automatic Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, data, target):
        """Training step with mixed precision"""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = self.model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### FP16 Training

```python
class FP16Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.model = self.model.half()  # Convert to FP16
    
    def train_step(self, data, target):
        """FP16 training step"""
        data = data.half()
        target = target.half()
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Cluster Management

### Kubernetes GPU Scheduling

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training-pod
spec:
  containers:
  - name: trainer
    image: training-image:latest
    resources:
      limits:
        nvidia.com/gpu: 2
      requests:
        nvidia.com/gpu: 2
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0,1"
```

### GPU Node Management

```python
class GPUClusterManager:
    def __init__(self):
        self.nodes = {}
    
    def register_node(self, node_id, gpu_count, gpu_type):
        """Register GPU node"""
        self.nodes[node_id] = {
            'gpu_count': gpu_count,
            'gpu_type': gpu_type,
            'available_gpus': gpu_count,
            'allocated_gpus': 0
        }
    
    def allocate_gpus(self, job_id, gpu_count):
        """Allocate GPUs for job"""
        for node_id, node_info in self.nodes.items():
            if node_info['available_gpus'] >= gpu_count:
                node_info['available_gpus'] -= gpu_count
                node_info['allocated_gpus'] += gpu_count
                return {
                    'node_id': node_id,
                    'gpu_ids': list(range(gpu_count)),
                    'job_id': job_id
                }
        return None
    
    def release_gpus(self, node_id, gpu_count):
        """Release GPUs"""
        if node_id in self.nodes:
            self.nodes[node_id]['available_gpus'] += gpu_count
            self.nodes[node_id]['allocated_gpus'] -= gpu_count
```

## SLURM Workload Manager

### SLURM Job Submission

```bash
#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=training_%j.out

module load cuda/11.8
module load python/3.9

python train.py
```

### SLURM Python Integration

```python
import subprocess

class SLURMJob:
    def __init__(self, script_path):
        self.script_path = script_path
    
    def submit(self, **kwargs):
        """Submit SLURM job"""
        cmd = ['sbatch']
        
        for key, value in kwargs.items():
            cmd.extend([f'--{key}', str(value)])
        
        cmd.append(self.script_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    
    def check_status(self, job_id):
        """Check job status"""
        result = subprocess.run(
            ['squeue', '-j', str(job_id)],
            capture_output=True,
            text=True
        )
        return result.stdout
    
    def cancel(self, job_id):
        """Cancel job"""
        subprocess.run(['scancel', str(job_id)])
```

## Spot Instances and Cost Optimization

### Spot Instance Management

```python
class SpotInstanceManager:
    def __init__(self):
        self.spot_instances = []
        self.checkpoint_interval = 300  # 5 minutes
    
    def launch_spot_instance(self, instance_type, max_price):
        """Launch spot instance"""
        # Implementation depends on cloud provider
        instance = {
            'instance_id': 'spot-instance-123',
            'instance_type': instance_type,
            'max_price': max_price,
            'status': 'running'
        }
        self.spot_instances.append(instance)
        return instance
    
    def handle_spot_interruption(self, instance_id):
        """Handle spot instance interruption"""
        # Save checkpoint
        self.save_checkpoint()
        
        # Request new spot instance
        new_instance = self.launch_spot_instance(
            self.get_instance_type(instance_id),
            self.get_max_price(instance_id)
        )
        
        # Resume training from checkpoint
        self.resume_from_checkpoint(new_instance)
```

### Cost Optimization Strategies

```python
class CostOptimizer:
    def __init__(self):
        self.instance_costs = {
            'p3.2xlarge': 3.06,
            'p3.8xlarge': 12.24,
            'p3.16xlarge': 24.48,
            'p3dn.24xlarge': 31.22
        }
    
    def optimize_instance_selection(self, requirements):
        """Select cost-optimal instance"""
        gpu_count = requirements['gpu_count']
        memory_gb = requirements['memory_gb']
        
        candidates = []
        for instance_type, cost in self.instance_costs.items():
            if self.meets_requirements(instance_type, requirements):
                candidates.append({
                    'instance_type': instance_type,
                    'cost': cost,
                    'cost_per_gpu': cost / self.get_gpu_count(instance_type)
                })
        
        # Select lowest cost per GPU
        return min(candidates, key=lambda x: x['cost_per_gpu'])
    
    def calculate_training_cost(self, instance_type, training_hours):
        """Calculate training cost"""
        hourly_cost = self.instance_costs.get(instance_type, 0)
        return hourly_cost * training_hours
```

## Multi-GPU Training

### Data Parallel Training

```python
import torch.nn as nn
from torch.nn.parallel import DataParallel

class MultiGPUTrainer:
    def __init__(self, model, device_ids):
        self.device_ids = device_ids
        self.model = DataParallel(model, device_ids=device_ids)
        self.model = self.model.cuda()
    
    def train(self, dataloader, optimizer, criterion):
        """Multi-GPU training"""
        for batch in dataloader:
            data, target = batch
            data = data.cuda()
            target = target.cuda()
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

### Model Parallel Training

```python
class ModelParallelTrainer:
    def __init__(self, model, device_ids):
        self.device_ids = device_ids
        self.model = self.split_model(model, device_ids)
    
    def split_model(self, model, device_ids):
        """Split model across GPUs"""
        layers = list(model.children())
        split_point = len(layers) // len(device_ids)
        
        model_parts = []
        for i, device_id in enumerate(device_ids):
            start = i * split_point
            end = (i + 1) * split_point if i < len(device_ids) - 1 else len(layers)
            part = nn.Sequential(*layers[start:end]).to(device_id)
            model_parts.append(part)
        
        return model_parts
    
    def forward(self, x):
        """Forward pass through model parts"""
        for part in self.model:
            x = part(x)
        return x
```

## Performance Profiling

### CUDA Profiling

```python
import torch.profiler

class CUDAProfiler:
    def __init__(self, model):
        self.model = model
    
    def profile_training(self, dataloader, num_batches=10):
        """Profile training performance"""
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True
        ) as prof:
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                data, target = batch
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                prof.step()
        
        return prof.key_averages().table()
```

### Memory Profiling

```python
class MemoryProfiler:
    def __init__(self):
        self.memory_snapshots = []
    
    def take_snapshot(self, label):
        """Take memory snapshot"""
        if torch.cuda.is_available():
            snapshot = {
                'label': label,
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
            self.memory_snapshots.append(snapshot)
            return snapshot
    
    def print_summary(self):
        """Print memory usage summary"""
        for snapshot in self.memory_snapshots:
            print(f"{snapshot['label']}:")
            print(f"  Allocated: {snapshot['allocated'] / 1024**2:.2f} MB")
            print(f"  Reserved: {snapshot['reserved'] / 1024**2:.2f} MB")
            print(f"  Max Allocated: {snapshot['max_allocated'] / 1024**2:.2f} MB")
```

## Resource Scheduling

### Job Scheduler

```python
class JobScheduler:
    def __init__(self):
        self.job_queue = []
        self.running_jobs = []
        self.available_resources = {
            'gpu_count': 8,
            'memory_gb': 256
        }
    
    def submit_job(self, job):
        """Submit job to queue"""
        self.job_queue.append(job)
        self.schedule_jobs()
    
    def schedule_jobs(self):
        """Schedule jobs from queue"""
        while self.job_queue:
            job = self.job_queue[0]
            
            if self.can_allocate(job):
                self.allocate_resources(job)
                self.job_queue.pop(0)
                self.running_jobs.append(job)
            else:
                break
    
    def can_allocate(self, job):
        """Check if resources can be allocated"""
        required = job['resource_requirements']
        return (
            required['gpu_count'] <= self.available_resources['gpu_count'] and
            required['memory_gb'] <= self.available_resources['memory_gb']
        )
    
    def allocate_resources(self, job):
        """Allocate resources to job"""
        required = job['resource_requirements']
        self.available_resources['gpu_count'] -= required['gpu_count']
        self.available_resources['memory_gb'] -= required['memory_gb']
    
    def release_resources(self, job):
        """Release resources from completed job"""
        required = job['resource_requirements']
        self.available_resources['gpu_count'] += required['gpu_count']
        self.available_resources['memory_gb'] += required['memory_gb']
        self.running_jobs.remove(job)
        self.schedule_jobs()
```

## Key Takeaways

- GPU utilization optimization maximizes compute efficiency
- Mixed precision training reduces memory usage and speeds up training
- Cluster management enables efficient use of multiple GPUs
- SLURM provides job scheduling and resource management for clusters
- Spot instances offer significant cost savings for fault-tolerant workloads
- Multi-GPU training scales training across multiple devices
- Performance profiling identifies bottlenecks and optimization opportunities
- Resource scheduling ensures fair and efficient allocation of GPU resources
- Cost optimization balances performance requirements with budget constraints
- Effective GPU cluster management requires monitoring, scheduling, and optimization
