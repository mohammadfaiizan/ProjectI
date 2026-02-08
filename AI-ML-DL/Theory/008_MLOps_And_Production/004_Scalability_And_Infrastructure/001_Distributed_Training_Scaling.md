# Distributed Training Scaling

## Table of Contents

1. [Introduction to Distributed Training](#introduction-to-distributed-training)
2. [Data Parallelism](#data-parallelism)
3. [Model Parallelism](#model-parallelism)
4. [Pipeline Parallelism](#pipeline-parallelism)
5. [Distributed Training Frameworks](#distributed-training-frameworks)
6. [Communication Protocols](#communication-protocols)
7. [Synchronization Strategies](#synchronization-strategies)
8. [Fault Tolerance](#fault-tolerance)
9. [Performance Optimization](#performance-optimization)
10. [Key Takeaways](#key-takeaways)

## Introduction to Distributed Training

Distributed training enables training large models on multiple devices by parallelizing computation. Key motivations:

- **Model Size**: Models too large for single GPU memory
- **Training Speed**: Reduce training time through parallelism
- **Data Volume**: Process large datasets efficiently
- **Cost Efficiency**: Utilize multiple cheaper devices

### Parallelism Strategies

**Data Parallelism**: Split data across devices, replicate model
**Model Parallelism**: Split model across devices, replicate data
**Pipeline Parallelism**: Split model into stages, process in pipeline

### Architecture Overview

```
┌─────────────┐
│   Master    │
│   Node      │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│GPU 1│ │GPU 2│
└─────┘ └─────┘
   │       │
   └───┬───┘
       │
   ┌───▼───┐
┌─────┐ ┌─────┐
│GPU 3│ │GPU 4│
└─────┘ └─────┘
```

## Data Parallelism

### Basic Data Parallelism

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Single GPU
model = nn.Linear(10, 1)
model = model.cuda()

# DataParallel (single machine, multiple GPUs)
model = DataParallel(model)

# DistributedDataParallel (multiple machines)
model = DistributedDataParallel(model)
```

### PyTorch Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Initialize distributed process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size):
    """Distributed training function"""
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=sampler
    )
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.target)
            loss.backward()
            optimizer.step()
    
    dist.destroy_process_group()

# Launch distributed training
if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size
    )
```

### Gradient Aggregation

```python
class GradientAggregator:
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
    
    def allreduce_gradients(self):
        """Aggregate gradients across all processes"""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
```

## Model Parallelism

### Model Sharding

```python
import torch.nn as nn

class ModelParallelModel(nn.Module):
    def __init__(self, device0, device1):
        super().__init__()
        self.device0 = device0
        self.device1 = device1
        
        # Split model across devices
        self.layer1 = nn.Linear(1000, 500).to(device0)
        self.layer2 = nn.Linear(500, 250).to(device1)
        self.layer3 = nn.Linear(250, 10).to(device1)
    
    def forward(self, x):
        # Move input to first device
        x = x.to(self.device0)
        x = self.layer1(x)
        
        # Move intermediate result to second device
        x = x.to(self.device1)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x
```

### Pipeline Parallelism

```python
class PipelineParallelModel(nn.Module):
    def __init__(self, num_stages, devices):
        super().__init__()
        self.num_stages = num_stages
        self.devices = devices
        
        # Split model into stages
        self.stages = nn.ModuleList([
            self.create_stage(i) for i in range(num_stages)
        ])
        
        # Move each stage to its device
        for i, stage in enumerate(self.stages):
            stage.to(devices[i])
    
    def create_stage(self, stage_id):
        """Create a stage of the model"""
        layers = []
        # Define layers for this stage
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through pipeline"""
        # Process through stages
        for stage, device in zip(self.stages, self.devices):
            x = x.to(device)
            x = stage(x)
        return x
```

## Pipeline Parallelism

### GPipe Implementation

```python
class GPipeModel(nn.Module):
    def __init__(self, model, num_microbatches=4):
        super().__init__()
        self.model = model
        self.num_microbatches = num_microbatches
    
    def forward(self, inputs):
        """GPipe forward pass"""
        # Split input into microbatches
        microbatches = torch.chunk(inputs, self.num_microbatches)
        
        # Process microbatches in pipeline
        outputs = []
        for i, microbatch in enumerate(microbatches):
            output = self.model(microbatch)
            outputs.append(output)
        
        # Concatenate outputs
        return torch.cat(outputs, dim=0)
```

### Pipeline Scheduling

```python
class PipelineScheduler:
    def __init__(self, stages, num_microbatches=4):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.pipeline = []
    
    def schedule(self, inputs):
        """Schedule microbatches through pipeline"""
        microbatches = torch.chunk(inputs, self.num_microbatches)
        
        # Initialize pipeline
        for i in range(len(self.stages)):
            self.pipeline.append([])
        
        # Fill pipeline
        for mb_idx, microbatch in enumerate(microbatches):
            for stage_idx, stage in enumerate(self.stages):
                if mb_idx == 0 or mb_idx > stage_idx:
                    output = stage(microbatch)
                    if stage_idx < len(self.stages) - 1:
                        microbatch = output
                    else:
                        yield output
```

## Distributed Training Frameworks

### Horovod

```python
import horovod.torch as hvd

def train_horovod():
    """Training with Horovod"""
    # Initialize Horovod
    hvd.init()
    
    # Pin GPU to local rank
    torch.cuda.set_device(hvd.local_rank())
    
    # Create model
    model = MyModel().cuda()
    
    # Wrap optimizer
    optimizer = torch.optim.Adam(model.parameters())
    optimizer = hvd.DistributedOptimizer(optimizer)
    
    # Broadcast initial parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    
    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler
    )
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
```

### DeepSpeed

```python
import deepspeed

def train_deepspeed():
    """Training with DeepSpeed"""
    # Initialize model and optimizer
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        training_data=train_dataset,
        config="deepspeed_config.json"
    )
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()
```

### DeepSpeed Configuration

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001
    }
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000
  }
}
```

### FSDP (Fully Sharded Data Parallel)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

def train_fsdp():
    """Training with FSDP"""
    # Initialize process group
    dist.init_process_group("nccl")
    
    # Create model
    model = MyModel()
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=True
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.target)
            loss.backward()
            optimizer.step()
```

## Communication Protocols

### All-Reduce

```python
def allreduce_gradients(model, world_size):
    """All-reduce gradients"""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
```

### Ring All-Reduce

```python
class RingAllReduce:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
    
    def ring_allreduce(self, tensor):
        """Ring all-reduce algorithm"""
        chunk_size = tensor.numel() // self.world_size
        chunks = tensor.chunk(self.world_size)
        
        # Scatter-reduce phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1) % self.world_size
            
            send_chunk = chunks[send_rank]
            recv_chunk = chunks[recv_rank]
            
            # Send and receive
            dist.send(send_chunk, send_rank)
            dist.recv(recv_chunk, recv_rank)
            
            # Accumulate
            chunks[recv_rank] += recv_chunk
        
        # Allgather phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1) % self.world_size
            
            send_chunk = chunks[send_rank]
            recv_chunk = chunks[recv_rank]
            
            dist.send(send_chunk, send_rank)
            dist.recv(recv_chunk, recv_rank)
        
        return torch.cat(chunks)
```

## Synchronization Strategies

### Synchronous SGD

```python
class SynchronousSGD:
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
    
    def step(self):
        """Synchronous SGD step"""
        # Compute gradients
        loss.backward()
        
        # Synchronize gradients
        self.allreduce_gradients()
        
        # Update parameters
        optimizer.step()
```

### Asynchronous SGD

```python
class AsynchronousSGD:
    def __init__(self, model, parameter_server):
        self.model = model
        self.parameter_server = parameter_server
    
    def step(self):
        """Asynchronous SGD step"""
        # Compute gradients
        loss.backward()
        
        # Send gradients to parameter server
        self.parameter_server.update(self.model.parameters())
        
        # Pull updated parameters
        updated_params = self.parameter_server.get_parameters()
        self.model.load_state_dict(updated_params)
```

## Fault Tolerance

### Checkpointing

```python
class DistributedCheckpoint:
    def __init__(self, model, optimizer, rank):
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
    
    def save_checkpoint(self, epoch, path):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Only rank 0 saves
        if self.rank == 0:
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
```

### Fault Recovery

```python
class FaultTolerantTrainer:
    def __init__(self, model, optimizer, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 1000
    
    def train_with_recovery(self):
        """Train with automatic recovery"""
        start_epoch = 0
        
        # Try to load checkpoint
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            start_epoch = self.load_checkpoint(latest_checkpoint)
        
        try:
            for epoch in range(start_epoch, num_epochs):
                self.train_epoch(epoch)
                
                # Save checkpoint periodically
                if epoch % self.checkpoint_interval == 0:
                    self.save_checkpoint(epoch)
        except Exception as e:
            print(f"Training failed: {e}")
            print("Resuming from latest checkpoint...")
            self.train_with_recovery()
```

## Performance Optimization

### Gradient Compression

```python
class GradientCompression:
    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio
    
    def compress_gradients(self, gradients):
        """Compress gradients before communication"""
        # Top-k sparsification
        k = int(len(gradients) * self.compression_ratio)
        top_k_indices = torch.topk(torch.abs(gradients), k).indices
        
        compressed = torch.zeros_like(gradients)
        compressed[top_k_indices] = gradients[top_k_indices]
        
        return compressed, top_k_indices
    
    def decompress_gradients(self, compressed, indices):
        """Decompress gradients"""
        gradients = torch.zeros_like(compressed)
        gradients[indices] = compressed[indices]
        return gradients
```

### Mixed Precision Training

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
```

## Key Takeaways

- Distributed training enables training large models by parallelizing computation
- Data parallelism splits data across devices while replicating the model
- Model parallelism splits the model across devices for very large models
- Pipeline parallelism processes data through model stages in a pipeline
- Frameworks like Horovod, DeepSpeed, and FSDP simplify distributed training
- Communication protocols (all-reduce, ring) enable efficient gradient synchronization
- Synchronization strategies balance convergence and communication overhead
- Fault tolerance through checkpointing enables recovery from failures
- Performance optimization through compression and mixed precision improves efficiency
- Choosing the right parallelism strategy depends on model size, data volume, and hardware
