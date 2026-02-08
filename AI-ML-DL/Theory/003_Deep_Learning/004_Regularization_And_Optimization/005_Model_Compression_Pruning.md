# Model Compression and Pruning

## Table of Contents

1. [Introduction](#introduction)
2. [Network Pruning](#network-pruning)
3. [Structured vs. Unstructured Pruning](#structured-vs-unstructured-pruning)
4. [Quantization](#quantization)
5. [Knowledge Distillation](#knowledge-distillation)
6. [Lottery Ticket Hypothesis](#lottery-ticket-hypothesis)
7. [Neural Architecture Search](#neural-architecture-search)
8. [Low-Rank Factorization](#low-rank-factorization)
9. [Practical Compression Pipelines](#practical-compression-pipelines)
10. [Key Takeaways](#key-takeaways)

## Introduction

Model compression reduces model size and computational requirements while maintaining performance, enabling deployment on resource-constrained devices. Techniques include pruning, quantization, knowledge distillation, and architecture optimization.

This chapter covers model compression techniques, from pruning and quantization to knowledge distillation and the lottery ticket hypothesis.

## Network Pruning

Pruning removes unnecessary parameters or connections.

### Magnitude-Based Pruning

Remove parameters with smallest magnitude:

**Algorithm**:
1. Train model to convergence
2. Remove smallest weights
3. Fine-tune remaining weights
4. Repeat if needed

**Criterion**: $|w_{ij}| < \theta$

### Gradient-Based Pruning

Remove parameters with smallest gradient impact:

$$\Delta \mathcal{L} \approx |w_{ij} \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}}|$$

### Importance Scoring

Score parameters by importance:

$$s_{ij} = |w_{ij}| \quad \text{(magnitude)}$$

$$s_{ij} = |w_{ij} \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}}| \quad \text{(gradient)}$$

$$s_{ij} = \frac{w_{ij}^2}{2} \quad \text{(optimal brain damage)}$$

### Iterative Pruning

Prune gradually:
1. Prune small percentage
2. Fine-tune
3. Repeat

**Typical Schedule**: 10-20% per iteration

### One-Shot Pruning

Prune once, then fine-tune:
- Faster
- May hurt performance more
- Simpler

## Structured vs. Unstructured Pruning

### Unstructured Pruning

Remove individual weights:
- **Sparsity**: High sparsity possible
- **Hardware**: Requires sparse operations
- **Speedup**: Limited without specialized hardware

**Example**: Remove 90% of weights → 90% sparse, but limited speedup

### Structured Pruning

Remove entire structures:
- **Channels**: Remove entire channels
- **Filters**: Remove entire filters
- **Layers**: Remove entire layers

**Benefits**:
- Hardware-friendly
- Actual speedup
- Easier to implement

**Drawbacks**:
- Less flexible
- May hurt performance more

### Channel Pruning

Remove entire channels:

**Criteria**:
- Channel importance
- L1 norm of filters
- BatchNorm scaling factors

**Process**:
1. Score channels
2. Remove least important
3. Fine-tune

### Filter Pruning

Remove entire filters (for next layer's input channels):

- Similar to channel pruning
- Prune filters in current layer
- Adjust next layer accordingly

## Quantization

Quantization reduces precision of weights and activations.

### Post-Training Quantization

Quantize after training:

**INT8 Quantization**:
- FP32 → INT8
- 4x memory reduction
- 2-4x speedup (with INT8 ops)

**Quantization Function**:

$$Q(x) = \text{clip}(\text{round}(x/s), -128, 127)$$

where $s$ is scale factor.

### Quantization-Aware Training

Train with quantization simulation:

1. Simulate quantization during training
2. Model learns quantized-friendly weights
3. Better performance than post-training

**Fake Quantization**:

$$\tilde{x} = Q(x) \cdot s$$

Use $\tilde{x}$ in forward pass, but backpropagate through $Q$.

### Mixed Precision

Use different precisions:
- **FP32**: Master weights, sensitive operations
- **FP16**: Most computations
- **INT8**: Inference

**Benefits**:
- Speedup from FP16
- Stability from FP32
- Best of both

### Dynamic vs. Static Quantization

**Static**: Scale computed once
**Dynamic**: Scale computed per input

Dynamic more accurate but slower.

## Knowledge Distillation

Transfer knowledge from large teacher to small student.

### Standard Distillation

$$\mathcal{L} = \alpha \mathcal{L}_{\text{CE}}(y, y_s) + (1-\alpha) \mathcal{L}_{\text{KL}}(y_t/T, y_s/T)$$

where:
- $y_t$: Teacher predictions
- $y_s$: Student predictions
- $T$: Temperature

### Feature Distillation

Match intermediate features:

$$\mathcal{L} = ||f_t(\mathbf{x}) - f_s(\mathbf{x})||^2$$

### Attention Transfer

Match attention maps:

$$\mathcal{L} = ||A_t - A_s||^2$$

### Self-Distillation

Student distills from itself:
- Different architectures
- Temporal ensembling
- Progressive distillation

## Lottery Ticket Hypothesis

Lottery ticket: Subnetwork that trains to good performance.

### Hypothesis

Dense networks contain subnetworks that:
- When trained in isolation
- From same initialization
- Achieve similar performance

### Finding Lottery Tickets

**Algorithm**:
1. Train network to convergence
2. Prune small weights
3. Reset remaining weights to initialization
4. Train pruned network

**Result**: Pruned network trains to similar performance

### Implications

- Initialization matters
- Pruning finds good architectures
- Early training important

### Applications

- Pruning strategies
- Architecture search
- Understanding training

## Neural Architecture Search

Automatically search for efficient architectures.

### Search Space

- Layer types
- Number of layers
- Width/depth
- Operations

### Search Strategies

**Reinforcement Learning**: Reward = accuracy/efficiency
**Evolutionary**: Evolve architectures
**Differentiable**: Gradient-based search

### Efficiency-Aware Search

Optimize for:
- Accuracy
- Model size
- Latency
- Energy

**Pareto Frontier**: Trade-off between objectives

## Low-Rank Factorization

Factorize weight matrices into products.

### Matrix Factorization

For weight matrix $W \in \mathbb{R}^{m \times n}$:

$$W \approx UV^T$$

where $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$, $r < \min(m,n)$.

**Compression**: $(m+n)r$ vs. $mn$ parameters

### SVD Decomposition

$$W = U \Sigma V^T$$

Keep top-$r$ singular values:

$$W \approx U_r \Sigma_r V_r^T$$

### Tensor Decomposition

For convolutional layers:
- CP decomposition
- Tucker decomposition
- More complex but better compression

## Practical Compression Pipelines

### Compression Workflow

1. **Train**: Train full model
2. **Prune**: Remove unimportant parameters
3. **Quantize**: Reduce precision
4. **Distill**: Transfer to smaller model
5. **Fine-tune**: Recover performance

### Combining Techniques

**Pipeline**:
1. Knowledge distillation
2. Pruning
3. Quantization
4. Fine-tuning

**Order Matters**: Distillation → Pruning → Quantization often best

### Evaluation Metrics

**Compression Ratio**: Size reduction
**Speedup**: Inference speed improvement
**Accuracy Drop**: Performance degradation
**Energy**: Energy consumption

### Target Deployment

Choose techniques based on:
- **Mobile**: Pruning + Quantization
- **Edge**: Structured pruning + INT8
- **Cloud**: May tolerate larger models

## Key Takeaways

1. **Network Pruning**: Removes unnecessary parameters through magnitude-based, gradient-based, or importance-based criteria, with iterative pruning typically outperforming one-shot.

2. **Structured vs. Unstructured**: Structured pruning (channels, filters) provides hardware-friendly speedup, while unstructured pruning achieves higher sparsity but requires specialized hardware.

3. **Quantization**: Reduces precision (FP32 → INT8) for 4x memory reduction and 2-4x speedup, with quantization-aware training outperforming post-training quantization.

4. **Knowledge Distillation**: Transfers knowledge from large teacher to small student using soft targets, feature matching, and attention transfer.

5. **Lottery Ticket Hypothesis**: Dense networks contain subnetworks that train well from same initialization, suggesting importance of initialization and early training.

6. **Neural Architecture Search**: Automatically searches for efficient architectures, optimizing for accuracy, size, and latency trade-offs.

7. **Low-Rank Factorization**: Factorizes weight matrices into products of smaller matrices, reducing parameters while maintaining structure.

8. **Compression Pipelines**: Combining techniques (distillation → pruning → quantization) provides best results, with order and fine-tuning crucial for performance recovery.

9. **Evaluation**: Measure compression ratio, speedup, accuracy drop, and energy consumption to assess compression effectiveness.

10. **Deployment Considerations**: Choose compression techniques based on target deployment (mobile, edge, cloud) and hardware capabilities.
