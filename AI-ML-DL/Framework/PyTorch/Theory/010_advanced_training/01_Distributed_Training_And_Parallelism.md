# Distributed Training and Parallelism

## Table of Contents
1. [Overview](#overview)
2. [Distributed Data Parallel (DDP)](#distributed-data-parallel-ddp)
3. [Model Parallelism](#model-parallelism)
4. [Gradient Accumulation](#gradient-accumulation)
5. [DataParallel vs DDP Comparison](#dataparallel-vs-ddp-comparison)

---

## Overview

**Distributed training** enables scaling neural network training across multiple GPUs and machines. PyTorch provides several paradigms: **Distributed Data Parallel (DDP)** for data parallelism, **model parallelism** for splitting large models across devices, and **gradient accumulation** for simulating larger batch sizes when memory is limited.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

---

## Distributed Data Parallel (DDP)

### init_process_group and Backends

**DDP** replicates the model on each GPU and synchronizes gradients during backward pass. Each process runs independently; initialization requires `init_process_group` with a backend. **NCCL** is preferred for GPU training; **Gloo** works for CPU or heterogeneous setups.

```python
def setup_distributed(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()
```

| Backend | Use Case | Notes |
|---------|----------|-------|
| nccl | Multi-GPU (CUDA) | Best performance, NVIDIA only |
| gloo | CPU, heterogeneous | Cross-platform, slower than NCCL |
| mpi | HPC clusters | Requires MPI installation |

### DistributedSampler

**DistributedSampler** partitions data across processes so each rank sees a non-overlapping subset. Shuffle must be coordinated via `set_epoch(epoch)` to ensure different order each epoch.

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        pass
```

### DDP Wrapping

Wrap the model with **DDP** after moving to the correct device. Access the underlying module via `ddp_model.module` when saving checkpoints.

```python
model = MyModel().to(device)
ddp_model = DDP(model, device_ids=[rank])

ddp_model = DDP(
    model,
    device_ids=[rank],
    output_device=rank,
    find_unused_parameters=True,
    gradient_as_bucket_view=True,
    broadcast_buffers=False
)
```

| Parameter | Purpose |
|-----------|---------|
| `find_unused_parameters` | Required for models with conditional branches |
| `gradient_as_bucket_view` | Memory optimization for gradient buckets |
| `broadcast_buffers` | Whether to sync BatchNorm buffers |

### torchrun Launch

Launch distributed training with **torchrun** (or `python -m torch.distributed.run`). Each process gets `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment variables.

```python
torchrun --nproc_per_node=4 train_script.py
```

```python
def train_distributed(rank, world_size, epochs=5):
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    model = MyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for data, targets in dataloader:
            outputs = ddp_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    cleanup_distributed()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
```

### Gradient Synchronization Control

Use `no_sync()` to skip gradient all-reduce for gradient accumulation within DDP.

```python
accumulation_steps = 4
for batch_idx, (data, targets) in enumerate(dataloader):
    if (batch_idx + 1) % accumulation_steps != 0:
        with ddp_model.no_sync():
            outputs = ddp_model(data)
            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()
    else:
        outputs = ddp_model(data)
        loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Metric Aggregation

Synchronize metrics across ranks with **all_reduce**.

```python
total_loss_tensor = torch.tensor(total_loss).to(device)
dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
avg_loss = total_loss_tensor.item() / (len(dataloader) * world_size)
```

---

## Model Parallelism

### Pipeline Parallelism

**Pipeline parallelism** splits the model into stages; each stage runs on a different GPU. Micro-batches flow through stages to overlap computation.

```python
class PipelineStage(nn.Module):
    def __init__(self, stage_id, device):
        super().__init__()
        self.stage_id = stage_id
        self.device = device

class ConvStage(PipelineStage):
    def __init__(self, stage_id, device, in_channels, out_channels):
        super().__init__(stage_id, device)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1).to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
```

### Placing Layers on Different GPUs

Manually place layers on devices and move tensors between them during forward.

```python
class ModelParallelCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1).to('cuda:0')
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1).to('cuda:0')
        device1 = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1).to(device1)
        self.fc = nn.Linear(256 * 8 * 8, num_classes).to(device1)
        self.device1 = device1

    def forward(self, x):
        x = x.to('cuda:0')
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.to(self.device1)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

### Tensor Parallelism

**Tensor parallelism** splits weight matrices across devices. Each rank computes a slice; outputs are combined via all-gather or all-reduce.

```python
class ParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank):
        super().__init__()
        self.output_size_per_rank = output_size // world_size
        self.linear = nn.Linear(input_size, self.output_size_per_rank)

    def forward(self, x):
        output = self.linear(x)
        if dist.is_initialized():
            gathered = [torch.zeros_like(output) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, output)
            output = torch.cat(gathered, dim=-1)
        return output
```

### Sharded Linear for Large Models

Shard a linear layer across multiple devices and concatenate outputs.

```python
class ShardedLinear(nn.Module):
    def __init__(self, input_size, output_size, devices):
        super().__init__()
        self.devices = devices
        self.output_size_per_shard = output_size // len(devices)
        self.shards = nn.ModuleList([
            nn.Linear(input_size, self.output_size_per_shard).to(d)
            for d in devices
        ])

    def forward(self, x):
        shard_outputs = []
        for i, shard in enumerate(self.shards):
            out = shard(x.to(self.devices[i]))
            shard_outputs.append(out.to(self.devices[0]))
        return torch.cat(shard_outputs, dim=-1)
```

---

## Gradient Accumulation

### Simulating Large Batches

**Gradient accumulation** runs multiple forward-backward passes before updating parameters. Effective batch size = `batch_size * accumulation_steps`.

```python
accumulation_steps = 4
for batch_idx, (data, targets) in enumerate(dataloader):
    outputs = model(data)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Loss Scaling

Scale the loss by `1 / accumulation_steps` so the effective gradient magnitude matches a single large batch.

```python
loss_scale = 1.0 / accumulation_steps
outputs = model(data)
loss = criterion(outputs, targets) * loss_scale
loss.backward()
```

### Accumulation with Mixed Precision

Combine gradient accumulation with **autocast** and **GradScaler** for memory-efficient training.

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
for batch_idx, (data, targets) in enumerate(dataloader):
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets) / accumulation_steps

    scaler.scale(loss).backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Dynamic Accumulation Steps

Compute accumulation steps from target and base batch sizes.

```python
target_batch_size = 512
base_batch_size = 32
accumulation_steps = target_batch_size // base_batch_size
```

### Adaptive Accumulation

Adjust accumulation steps based on GPU memory usage.

```python
def adapt_accumulation_steps(self):
    current_memory = torch.cuda.memory_allocated() / 1024**3
    if current_memory / self.max_memory_gb > 0.8:
        self.accumulation_steps = min(self.accumulation_steps * 2, self.max_accumulation_steps)
    elif current_memory / self.max_memory_gb < 0.4:
        self.accumulation_steps = max(self.accumulation_steps // 2, self.min_accumulation_steps)
```

---

## DataParallel vs DDP Comparison

| Aspect | DataParallel | DDP |
|--------|--------------|-----|
| Process model | Single process, multi-threaded | Multi-process, one per GPU |
| Gradient sync | After backward on main GPU | All-reduce during backward |
| Speed | Slower, GIL bottleneck | Faster, no GIL |
| Model replication | Replicated on main GPU first | Each process has full copy |
| Recommended | Deprecated for new code | Preferred for multi-GPU |

```python
model = MyModel()
model_dp = nn.DataParallel(model, device_ids=[0, 1])
outputs = model_dp(data)
```

DDP is the recommended approach for distributed training in PyTorch. Use **torchrun** or **torch.multiprocessing.spawn** to launch multi-process training.
