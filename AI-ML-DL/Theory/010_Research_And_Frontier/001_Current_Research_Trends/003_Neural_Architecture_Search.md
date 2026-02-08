# Neural Architecture Search

## Table of Contents

1. [Introduction](#introduction)
2. [NAS Problem Formulation](#nas-problem-formulation)
3. [Search Spaces](#search-spaces)
4. [Reinforcement Learning-Based NAS](#reinforcement-learning-based-nas)
5. [Evolutionary Algorithms for NAS](#evolutionary-algorithms-for-nas)
6. [Differentiable Architecture Search (DARTS)](#differentiable-architecture-search-darts)
7. [One-Shot NAS](#one-shot-nas)
8. [Efficiency-Aware NAS](#efficiency-aware-nas)
9. [Hardware-Aware NAS](#hardware-aware-nas)
10. [Key Takeaways](#key-takeaways)

## Introduction

Neural Architecture Search (NAS) automates the design of neural network architectures, traditionally a manual and expertise-intensive process. NAS methods explore large search spaces to discover architectures that achieve high performance on target tasks.

Early NAS methods required thousands of GPU days, but recent advances have dramatically reduced computational requirements while maintaining or improving performance. Modern NAS methods can discover architectures competitive with or superior to hand-designed ones.

Key challenges:
- How to define the search space?
- How to efficiently explore the search space?
- How to balance performance and efficiency?
- How to transfer architectures across tasks?

## NAS Problem Formulation

NAS can be formulated as an optimization problem over a discrete search space.

### Problem Definition

**Given**:
- Search space $\mathcal{A}$ (set of possible architectures)
- Dataset $\mathcal{D} = \{(x_i, y_i)\}$
- Performance metric $M(a, w_a)$ (e.g., accuracy)

**Find**: Architecture $a^* \in \mathcal{A}$ that maximizes performance:

$$a^* = \arg\max_{a \in \mathcal{A}} M(a, w_a^*)$$

where $w_a^*$ are optimal weights for architecture $a$:

$$w_a^* = \arg\min_w \mathcal{L}(a, w, \mathcal{D})$$

### Challenges

**Discrete optimization**: Search space is discrete and large
**Nested optimization**: Must train each architecture to evaluate it
**Computational cost**: Training many architectures is expensive
**Transfer**: Architectures may not transfer across tasks

### Evaluation Strategies

**Full training**: Train each architecture fully (accurate but expensive)
**Early stopping**: Stop training early (faster but less accurate)
**Weight sharing**: Share weights across architectures (efficient but may be inaccurate)
**Surrogate models**: Predict performance without training (fast but approximate)

## Search Spaces

The search space defines which architectures can be discovered.

### Cell-Based Search Spaces

**Macro-architecture**: Fixed overall structure (e.g., stack of cells)
**Micro-architecture**: Search for cell structure

**Cell structure**:
- Directed acyclic graph (DAG)
- Nodes: Feature maps
- Edges: Operations (conv, pool, etc.)

**Example**: NASNet search space
- Normal cell: Maintains spatial resolution
- Reduction cell: Reduces spatial resolution
- Operations: conv 3x3, conv 5x5, max pool, avg pool, identity, etc.

### Hierarchical Search Spaces

**Multiple levels**: Search at different granularities
- **Level 1**: Overall network structure
- **Level 2**: Cell structure
- **Level 3**: Operation choices

**Advantages**: More flexible, can discover diverse architectures
**Challenges**: Larger search space, more complex optimization

### Operation Search Spaces

**Convolutional operations**:
- Kernel size: 3x3, 5x5, 7x7
- Dilation: 1, 2, 3
- Groups: 1, 2, 4, 8
- Activation: ReLU, Swish, GELU

**Attention operations**:
- Self-attention variants
- Multi-head attention
- Sparse attention

**Normalization**:
- BatchNorm, LayerNorm, GroupNorm

### Progressive Search Spaces

**Start small**: Begin with simple architectures
**Gradually expand**: Add complexity over time
**Advantages**: More efficient exploration
**Example**: Progressive NAS

## Reinforcement Learning-Based NAS

RL-based NAS uses reinforcement learning to guide architecture search.

### Formulation

**State**: Current architecture (or partial architecture)
**Action**: Architecture modification (add operation, change connection, etc.)
**Reward**: Performance of completed architecture

**Policy**: $\pi(a | s)$: Probability of action $a$ given state $s$

**Objective**: Maximize expected reward:

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta} [R(a)]$$

where $R(a)$ is the reward (e.g., validation accuracy) of architecture $a$.

### Policy Gradient Methods

**REINFORCE**: 
$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_\theta} [R(a) \nabla_\theta \log \pi_\theta(a)]$$

**Training**:
1. Sample architectures from policy
2. Train each architecture
3. Evaluate performance
4. Update policy using REINFORCE

### NASNet

**Search space**: Cell-based (normal and reduction cells)
**RL method**: Policy gradient (REINFORCE)
**Controller**: RNN that generates architecture descriptions
**Performance**: Achieved state-of-the-art on ImageNet

**Limitations**: Requires thousands of GPU days

### ENAS: Efficient NAS

**Key innovation**: Weight sharing across architectures
**Controller**: RNN that samples architectures
**Training**: Train supernet with shared weights
**Evaluation**: Sample architectures, evaluate on shared weights

**Efficiency**: 1000x faster than NASNet
**Performance**: Competitive with NASNet

## Evolutionary Algorithms for NAS

Evolutionary algorithms evolve populations of architectures through mutation and selection.

### Genetic Algorithm Framework

**Population**: Set of architectures
**Fitness**: Performance of each architecture
**Selection**: Choose architectures based on fitness
**Mutation**: Modify architectures (add/remove operations, change connections)
**Crossover**: Combine two architectures

### Algorithm

```
1. Initialize population P
2. For generation = 1 to G:
   a. Evaluate fitness of all architectures in P
   b. Select parents from P (based on fitness)
   c. Create offspring through mutation and crossover
   d. Evaluate fitness of offspring
   e. Update population P (select best architectures)
3. Return best architecture
```

### AmoebaNet

**Evolutionary algorithm**: Regularized evolution
**Mutation**: Random modifications to architecture
**Selection**: Age-based (prefer younger architectures with good fitness)
**Performance**: Achieved state-of-the-art on ImageNet

**Advantages**: Simple, parallelizable
**Limitations**: Still requires many evaluations

### Progressive Evolution

**Start simple**: Begin with small architectures
**Evolve**: Gradually increase complexity
**Advantages**: More efficient exploration
**Example**: Progressive Evolution of Image Classifiers

## Differentiable Architecture Search (DARTS)

DARTS makes architecture search differentiable by relaxing the discrete search space.

### Key Idea

**Relaxation**: Replace discrete operation choice with continuous mixture
**Differentiation**: Use gradient descent instead of discrete optimization

### Formulation

**Mixed operation**: 
$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \alpha_o^{(i,j)} o(x)$$

where:
- $\mathcal{O}$: Set of candidate operations
- $\alpha_o^{(i,j)}$: Architecture weight for operation $o$ on edge $(i,j)$
- $\sum_o \alpha_o^{(i,j)} = 1$

**Architecture parameters**: $\alpha = \{\alpha_o^{(i,j)}\}$

### Bilevel Optimization

**Inner problem**: Train network weights $w$
$$\min_w \mathcal{L}_{train}(w, \alpha)$$

**Outer problem**: Optimize architecture $\alpha$
$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

where $w^*(\alpha)$ are optimal weights for architecture $\alpha$.

### Optimization

**Approximation**: Use one-step gradient descent for inner problem
$$w^*(\alpha) \approx w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha)$$

**Gradient**: 
$$\nabla_\alpha \mathcal{L}_{val} = \nabla_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) - \xi \nabla_{\alpha,w}^2 \mathcal{L}_{train}(w, \alpha) \nabla_w \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

**Practical**: Use first-order approximation (ignore second-order term)

### Discretization

After optimization, discretize:
- Keep top-$k$ operations per edge
- Or sample operations according to $\alpha$

### Advantages

**Efficiency**: Much faster than RL or evolutionary methods
**Differentiable**: Can use gradient-based optimization
**Performance**: Competitive with hand-designed architectures

### Limitations

**Memory**: Requires storing all operations in memory
**Approximation**: First-order approximation may be inaccurate
**Collapse**: May collapse to single operation

### Variants

**PC-DARTS**: Partial channel connections (reduce memory)
**GDAS**: Gumbel-based sampling (better discretization)
**FairDARTS**: Addresses operation collapse

## One-Shot NAS

One-shot NAS trains a single supernet that contains all architectures, then searches by evaluating architectures on the supernet.

### Supernet Training

**Supernet**: Single network containing all architectures in search space
**Weight sharing**: All architectures share weights
**Training**: Train supernet with various architectures sampled

**Loss function**:
$$\mathcal{L} = \mathbb{E}_{a \sim \mathcal{A}} [\mathcal{L}_{task}(a, w)]$$

where architectures $a$ are sampled from search space $\mathcal{A}$.

### Architecture Search

**After supernet training**:
1. Sample architectures from search space
2. Evaluate on supernet (no training needed)
3. Select best architecture

**Evaluation**: Can be very fast (forward pass only)

### Training Strategies

**Uniform sampling**: Sample architectures uniformly
**Fairness training**: Ensure all architectures trained equally
**Progressive shrinking**: Start with full supernet, gradually shrink

### Once-for-All

**Supernet**: Contains architectures of different sizes
**Training**: Train once, deploy many architectures
**Search**: Find best architecture for target hardware

**Advantages**: 
- Train once, use for multiple deployment scenarios
- Hardware-aware search without retraining

### Limitations

**Weight sharing**: Shared weights may not be optimal for individual architectures
**Ranking correlation**: Supernet ranking may not match fully trained ranking
**Training**: Supernet training can be challenging

## Efficiency-Aware NAS

Efficiency-aware NAS optimizes for both performance and efficiency (FLOPs, latency, memory).

### Multi-Objective Optimization

**Objectives**:
- Performance: $M(a)$ (e.g., accuracy)
- Efficiency: $E(a)$ (e.g., FLOPs, latency)

**Pareto optimal**: Architectures that cannot be improved in one objective without worsening the other

**Approaches**:
1. **Weighted sum**: $\max M(a) - \lambda E(a)$
2. **Constraint**: $\max M(a)$ s.t. $E(a) \leq \epsilon$
3. **Pareto search**: Find Pareto frontier

### FBNet

**Search space**: MobileNet-like architectures
**Objective**: Accuracy and latency
**Method**: Differentiable search with latency predictor
**Performance**: Efficient architectures for mobile devices

### MnasNet

**Multi-objective**: Accuracy and latency
**RL method**: Policy gradient with latency reward
**Latency**: Measured on real device
**Performance**: State-of-the-art mobile architectures

### EfficientNet

**Compound scaling**: Scale depth, width, and resolution together
**Base architecture**: Found by NAS
**Scaling**: Systematic scaling of base architecture
**Performance**: Efficient architectures across scales

## Hardware-Aware NAS

Hardware-aware NAS considers specific hardware characteristics during search.

### Hardware Metrics

**Latency**: Inference time on target hardware
**Energy**: Power consumption
**Memory**: Memory usage
**Throughput**: Examples per second

### Measurement

**Profiling**: Measure metrics on real hardware
**Predictors**: Train models to predict metrics
**Simulators**: Simulate hardware behavior

### HAT: Hardware-Aware Transformers

**Search space**: Transformer architectures
**Hardware**: Various devices (CPU, GPU, mobile)
**Method**: Differentiable search with hardware predictors
**Performance**: Efficient transformers for different hardware

### Once-for-All (OFA)

**Supernet**: Contains architectures for different hardware
**Training**: Train once
**Search**: Find best architecture for each hardware target
**Deployment**: Deploy different architectures for different devices

### Neural Architecture Transfer

**Source hardware**: Search on one hardware
**Target hardware**: Transfer to another hardware
**Challenge**: Architecture performance may not transfer
**Solution**: Hardware-aware search or fine-tuning

## Key Takeaways

1. **Neural Architecture Search** automates architecture design, discovering architectures competitive with hand-designed ones.

2. **Search spaces** define which architectures can be discovered, with cell-based, hierarchical, and progressive spaces being common approaches.

3. **Reinforcement learning-based NAS** uses RL to guide search, with NASNet achieving strong performance but requiring significant compute.

4. **Evolutionary algorithms** evolve populations of architectures, with AmoebaNet showing that simple evolutionary methods can be effective.

5. **DARTS** makes architecture search differentiable by relaxing discrete choices, enabling efficient gradient-based optimization.

6. **One-shot NAS** trains a single supernet containing all architectures, enabling fast architecture evaluation through weight sharing.

7. **Efficiency-aware NAS** optimizes for both performance and efficiency, discovering architectures suitable for resource-constrained deployments.

8. **Hardware-aware NAS** considers specific hardware characteristics, enabling deployment of efficient architectures on target devices.

9. **Weight sharing** is a key technique for efficiency, though it may introduce inaccuracies in architecture evaluation.

10. **Future directions** include improving search efficiency, better handling of multi-objective optimization, and developing methods that transfer across tasks and hardware.

## References

- Zoph, B., & Le, Q. V. (2016). "Neural Architecture Search with Reinforcement Learning." arXiv:1611.01578
- Real, E., et al. (2019). "Regularized Evolution for Image Classifier Architecture Search." AAAI 2019
- Pham, H., et al. (2018). "Efficient Neural Architecture Search via Parameters." ICML 2018
- Liu, H., et al. (2018). "DARTS: Differentiable Architecture Search." ICLR 2019
- Cai, H., et al. (2018). "Progressive Neural Architecture Search." ECCV 2018
- Wu, B., et al. (2019). "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search." CVPR 2019
- Tan, M., et al. (2019). "MnasNet: Platform-Aware Neural Architecture Search for Mobile." CVPR 2019
- Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019
- Cai, H., et al. (2019). "Once-for-All: Train One Network and Specialize it for Efficient Deployment." arXiv:1908.09791
- Wang, H., et al. (2020). "HAT: Hardware-Aware Transformers for Efficient Natural Language Processing." ACL 2020
