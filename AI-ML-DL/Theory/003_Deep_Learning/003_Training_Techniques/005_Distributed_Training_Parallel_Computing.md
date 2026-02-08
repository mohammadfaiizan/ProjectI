# Distributed Training and Parallel Computing

## Table of Contents

1. [Introduction](#introduction)
2. [Data Parallelism](#data-parallelism)
3. [Model Parallelism](#model-parallelism)
4. [Pipeline Parallelism](#pipeline-parallelism)
5. [Gradient Accumulation](#gradient-accumulation)
6. [Communication Strategies](#communication-strategies)
7. [Synchronous vs. Asynchronous Training](#synchronous-vs-asynchronous-training)
8. [Mixed Precision Training](#mixed-precision-training)
9. [Distributed Optimization](#distributed-optimization)
10. [Key Takeaways](#key-takeaways)

## Introduction

Distributed training enables training large models and processing large datasets by parallelizing computation across multiple devices. As models and datasets grow, distributed training becomes essential for practical deep learning.

This chapter covers distributed training strategies, from data and model parallelism to pipeline parallelism, communication optimization, and practical considerations for scaling training.

## Data Parallelism

Data parallelism replicates model across devices and splits data.

### Basic Algorithm

1. **Replicate Model**: Copy model to each device
2. **Split Data**: Divide batch across devices
3. **Forward Pass**: Each device processes its subset
4. **Backward Pass**: Compute gradients on each device
5. **All-Reduce**: Aggregate gradients across devices
6. **Update**: Update parameters (synchronized)

### Mathematical Formulation

For $K$ devices:

$$\mathbf{g} = \frac{1}{K} \sum_{k=1}^{K} \nabla_{\theta} \mathcal{L}(\theta, \mathcal{B}_k)$$

where $\mathcal{B}_k$ is batch on device $k$.

### Implementation

```python
import torch.nn as nn
import torch.distributed as dist

# Wrap model
model = nn.DataParallel(model)  # Single machine
# Or
model = nn.parallel.DistributedDataParallel(model)  # Multi-machine
```

### Advantages

- Linear speedup (ideal case)
- Easy to implement
- Works with any model
- Standard approach

### Limitations

- Memory: Each device holds full model
- Communication: Gradient synchronization overhead
- Batch size: Limited by single device memory

## Model Parallelism

Model parallelism splits model across devices.

### Basic Approach

Split layers across devices:

- **Device 1**: Layers 1-5
- **Device 2**: Layers 6-10
- **Device 3**: Layers 11-15

### Forward Pass

1. Input → Device 1
2. Device 1 output → Device 2
3. Device 2 output → Device 3
4. Device 3 output → Loss

### Backward Pass

1. Loss gradient → Device 3
2. Device 3 gradients → Device 2
3. Device 2 gradients → Device 1
4. Update parameters

### Use Cases

- **Large Models**: Don't fit on single device
- **Memory Constraints**: Exceed single device memory
- **Specialized Hardware**: Different devices for different layers

### Challenges

- **Communication Overhead**: Activations passed between devices
- **Load Balancing**: Ensure balanced computation
- **Complexity**: More complex than data parallelism

## Pipeline Parallelism

Pipeline parallelism combines data and model parallelism.

### Concept

Process multiple micro-batches in pipeline:

```
Time:  t0    t1    t2    t3    t4
Dev1:  [b0]  [b1]  [b2]  [b3]  [b4]
Dev2:       [b0]  [b1]  [b2]  [b3]
Dev3:            [b0]  [b1]  [b2]
```

### GPipe Algorithm

1. Split batch into micro-batches
2. Process micro-batches through pipeline
3. Aggregate gradients at end
4. Update parameters

### Benefits

- Better device utilization
- Enables very large models
- Combines data and model parallelism

### Challenges

- **Bubble Time**: Pipeline startup/teardown overhead
- **Memory**: Need to store activations
- **Synchronization**: Complex coordination

### Optimizations

**Gradient Checkpointing**: Trade computation for memory

**1F1B (One Forward One Backward)**: Overlap forward/backward

## Gradient Accumulation

Accumulate gradients over multiple mini-batches before updating.

### Algorithm

```python
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / num_accumulation_steps
    loss.backward()
    
    if (i + 1) % num_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Benefits

- **Effective Large Batch**: Simulate large batch with small batches
- **Memory Efficiency**: Process smaller batches
- **Gradient Stability**: More stable gradients

### Use Cases

- Limited GPU memory
- Want large effective batch size
- Single device training

## Communication Strategies

Efficient communication is crucial for distributed training.

### All-Reduce

Aggregate gradients across all devices:

$$\mathbf{g}_{\text{global}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{g}_k$$

**Ring All-Reduce**: Efficient algorithm using ring topology

**Tree All-Reduce**: Tree-based aggregation

### Gradient Compression

Reduce communication by compressing gradients:

**Quantization**: Reduce precision (e.g., FP32 → FP16)

**Sparsification**: Send only large gradients

**Top-K**: Send only top-K gradients

### Overlap Communication

Overlap gradient communication with computation:

- Compute gradients for next layer
- While communicating current gradients
- Reduces communication overhead

## Synchronous vs. Asynchronous Training

### Synchronous Training

All devices wait for all gradients:

- **All-Reduce**: Aggregate all gradients
- **Update**: Update parameters
- **Next Iteration**: All devices proceed together

**Advantages**:
- Deterministic
- Stable convergence
- Easier to debug

**Disadvantages**:
- Wait for slowest device
- Straggler problem

### Asynchronous Training

Devices update independently:

- **Compute Gradients**: On local data
- **Send to Parameter Server**: Asynchronously
- **Update Parameters**: Immediately
- **Pull Updated Parameters**: Before next iteration

**Advantages**:
- No waiting
- Better device utilization
- Faster (in some cases)

**Disadvantages**:
- Stale gradients
- Convergence issues
- More complex

### Hybrid Approaches

**Stale Synchronous Parallel (SSP)**:
- Allow bounded staleness
- Balance speed and consistency

## Mixed Precision Training

Use lower precision (FP16) to speed up training.

### Benefits

- **Speed**: 2x faster computation
- **Memory**: 2x less memory
- **Throughput**: Train larger models/batches

### Challenges

- **Numerical Stability**: Risk of underflow/overflow
- **Gradient Scaling**: Scale gradients to prevent underflow

### Implementation

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Best Practices

- Use FP32 for master weights
- Scale loss to prevent underflow
- Monitor for NaN/Inf
- Use dynamic loss scaling

## Distributed Optimization

Optimizers adapted for distributed training.

### Distributed SGD

Standard SGD with gradient aggregation:

$$\theta_{t+1} = \theta_t - \eta \frac{1}{K} \sum_{k=1}^{K} \nabla_{\theta} \mathcal{L}_k(\theta_t)$$

### Local SGD

Perform multiple local updates before synchronization:

- Update locally for $H$ steps
- Synchronize every $H$ steps
- Reduces communication

### Federated Averaging

Aggregate model parameters instead of gradients:

$$\theta_{t+1} = \frac{1}{K} \sum_{k=1}^{K} \theta_t^{(k)}$$

where $\theta_t^{(k)}$ is model after local updates on device $k$.

### Adaptive Methods

Distributed Adam, etc.:
- Aggregate first/second moments
- Update parameters
- Maintain optimizer state

## Key Takeaways

1. **Data Parallelism**: Replicates model across devices and splits data, providing linear speedup and being the standard approach for distributed training.

2. **Model Parallelism**: Splits model across devices, enabling training of models too large for single device but with communication overhead.

3. **Pipeline Parallelism**: Combines data and model parallelism by processing micro-batches in pipeline, enabling very large models with better device utilization.

4. **Gradient Accumulation**: Accumulates gradients over multiple mini-batches, simulating large batch size with limited memory.

5. **Communication Strategies**: Efficient all-reduce, gradient compression, and communication-computation overlap are crucial for distributed training performance.

6. **Synchronous Training**: All devices synchronize gradients, providing deterministic and stable training but waiting for slowest device.

7. **Asynchronous Training**: Devices update independently, avoiding waiting but introducing staleness and potential convergence issues.

8. **Mixed Precision**: Uses FP16 for computation and FP32 for master weights, providing 2x speedup and memory savings with careful gradient scaling.

9. **Distributed Optimization**: Adapts optimizers (SGD, Adam) for distributed settings, with techniques like local SGD and federated averaging reducing communication.

10. **Practical Considerations**: Choose parallelism strategy based on model size, data size, and hardware, with communication optimization and mixed precision essential for efficiency.
