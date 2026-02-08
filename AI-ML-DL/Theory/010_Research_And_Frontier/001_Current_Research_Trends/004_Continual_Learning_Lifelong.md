# Continual Learning and Lifelong Learning

## Table of Contents

1. [Introduction](#introduction)
2. [The Catastrophic Forgetting Problem](#the-catastrophic-forgetting-problem)
3. [Experience Replay Methods](#experience-replay-methods)
4. [Elastic Weight Consolidation (EWC)](#elastic-weight-consolidation-ewc)
5. [Progressive Neural Networks](#progressive-neural-networks)
6. [Meta-Plasticity Approaches](#meta-plasticity-approaches)
7. [Task-Incremental vs Class-Incremental vs Domain-Incremental](#task-incremental-vs-class-incremental-vs-domain-incremental)
8. [Evaluation Protocols and Benchmarks](#evaluation-protocols-and-benchmarks)
9. [Theoretical Perspectives](#theoretical-perspectives)
10. [Key Takeaways](#key-takeaways)

## Introduction

Continual learning (also known as lifelong learning or incremental learning) aims to enable machine learning models to learn continuously from a stream of data, acquiring new knowledge while retaining previously learned information. This is a fundamental challenge for artificial intelligence systems that must operate in dynamic environments.

Unlike traditional machine learning, where models are trained on static datasets, continual learning systems must adapt to new tasks, classes, or domains over time. The primary challenge is catastrophic forgetting: the tendency of neural networks to overwrite previously learned knowledge when learning new information.

Key research questions:
- How to prevent catastrophic forgetting?
- How to balance stability (retaining old knowledge) and plasticity (learning new knowledge)?
- How to transfer knowledge across tasks?
- How to evaluate continual learning systems?

## The Catastrophic Forgetting Problem

Catastrophic forgetting is the dramatic loss of previously learned information when a neural network learns new information.

### Phenomenon

**Observation**: When a neural network trained on task A is then trained on task B, performance on task A drops significantly, often to near-random levels.

**Example**: 
- Train on MNIST (digits 0-9)
- Then train on Fashion-MNIST
- Performance on MNIST drops dramatically

### Causes

**Overlapping representations**: New task uses same weights as old task
**Weight updates**: Gradient updates for new task overwrite weights important for old task
**Lack of constraints**: No mechanism to protect important weights

### Mathematical Formulation

**Old task**: $\mathcal{D}_{old} = \{(x_i, y_i)\}_{i=1}^{N_{old}}$
**New task**: $\mathcal{D}_{new} = \{(x_i, y_i)\}_{i=1}^{N_{new}}$

**Standard training** (causes forgetting):
$$\theta^* = \arg\min_\theta \mathcal{L}(\theta, \mathcal{D}_{new})$$

**Ideal continual learning**:
$$\theta^* = \arg\min_\theta [\mathcal{L}(\theta, \mathcal{D}_{old}) + \mathcal{L}(\theta, \mathcal{D}_{new})]$$

But $\mathcal{D}_{old}$ is not available during training on new task.

### Measures of Forgetting

**Backward transfer**: Performance on old tasks after learning new tasks
**Forward transfer**: Performance on new tasks due to knowledge from old tasks
**Retention**: Ability to retain performance on old tasks

**Metrics**:
- **Accuracy**: Performance on each task
- **Forgetting measure**: $F_i = \max_{j \leq i} a_{j,i} - a_{j,T}$ where $a_{j,i}$ is accuracy on task $j$ after training on task $i$
- **Average accuracy**: $\frac{1}{T}\sum_{i=1}^T a_{i,T}$

## Experience Replay Methods

Experience replay stores and replays examples from previous tasks during training on new tasks.

### Basic Idea

**Buffer**: Maintain a memory buffer of examples from previous tasks
**Training**: When learning new task, interleave examples from buffer
**Objective**: Prevent forgetting by revisiting old examples

### Algorithm

```
Initialize buffer B = {}
For each task t:
  For each batch (x, y) from task t:
    Add (x, y) to buffer B (with replacement strategy)
    Sample batch from buffer B
    Update model on mixed batch
```

### Buffer Management

**Fixed size**: Buffer has fixed capacity
**Replacement strategies**:
- **Random**: Random replacement
- **Reservoir sampling**: Uniform sampling of seen examples
- **Ring buffer**: FIFO replacement
- **Importance-based**: Replace least important examples

### Advantages

**Simple**: Easy to implement
**Effective**: Strong performance on many benchmarks
**Flexible**: Can be combined with other methods

### Limitations

**Memory**: Requires storing examples (may be large)
**Privacy**: Storing raw data may violate privacy constraints
**Scalability**: May not scale to many tasks

### Variants

**GEM**: Gradient Episodic Memory - uses buffer to constrain gradients
**A-GEM**: Averaged GEM - more efficient version
**ER**: Experience Replay - simple and effective baseline

### GEM: Gradient Episodic Memory

**Key idea**: Use buffer to constrain gradient updates

**Constraint**: New gradient should not increase loss on old tasks:
$$g \cdot g_{ref} \geq 0$$

for all reference gradients $g_{ref}$ from old tasks.

**Update**: If constraint violated, project gradient:
$$g \leftarrow g - \frac{g \cdot g_{ref}}{g_{ref} \cdot g_{ref}} g_{ref}$$

**Advantages**: Strong theoretical guarantees
**Limitations**: Requires computing gradients on buffer

## Elastic Weight Consolidation (EWC)

EWC protects important weights for previous tasks by adding a regularization term.

### Key Idea

**Importance**: Some weights are more important for old tasks than others
**Protection**: Add penalty for changing important weights
**Regularization**: Elastic constraint (allows some change)

### Formulation

**Fisher Information Matrix**: Measures importance of parameters
$$F_i = \mathbb{E}_{x \sim \mathcal{D}_i} \left[\nabla_\theta \log p(y|x, \theta)^T \nabla_\theta \log p(y|x, \theta)\right]$$

**EWC loss**:
$$\mathcal{L}_{EWC} = \mathcal{L}_{new}(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta - \theta_i^*)^2$$

where:
- $\theta_i^*$: Optimal parameters for task $i$
- $F_i$: Fisher information matrix for task $i$
- $\lambda$: Regularization strength

### Interpretation

**Quadratic penalty**: Penalizes changes to important weights
**Elastic**: Allows changes but with cost
**Task-specific**: Different importance for different tasks

### Advantages

**No buffer**: Doesn't require storing examples
**Efficient**: Only need to store Fisher matrices
**Theoretical**: Based on Bayesian inference

### Limitations

**Diagonal approximation**: Often uses diagonal Fisher (loses correlations)
**Memory**: Still need to store Fisher matrices for each task
**Hyperparameter**: $\lambda$ needs tuning

### Variants

**Online EWC**: Updates Fisher matrix online
**SI**: Synaptic Intelligence - similar to EWC
**MAS**: Memory Aware Synapses - importance based on sensitivity

## Progressive Neural Networks

Progressive neural networks add new columns (subnetworks) for each new task while freezing old columns.

### Architecture

**Columns**: Each task gets its own column (subnetwork)
**Lateral connections**: New columns can use features from old columns
**Frozen weights**: Old columns are frozen (not updated)

**Structure**:
```
Task 1: Column 1 (frozen)
Task 2: Column 2 + lateral connections to Column 1
Task 3: Column 3 + lateral connections to Columns 1, 2
...
```

### Advantages

**No forgetting**: Old columns never change
**Transfer**: Lateral connections enable knowledge transfer
**Modular**: Each task has dedicated capacity

### Limitations

**Capacity**: Grows linearly with number of tasks
**No compression**: Cannot reduce model size
**Transfer**: Lateral connections may not be optimal

### Variants

**Progressive Networks**: Original formulation
**PackNet**: Progressive packing (more efficient)
**HAT**: Hard Attention to Tasks (learns which columns to use)

## Meta-Plasticity Approaches

Meta-plasticity methods learn how to learn, adapting the learning process itself.

### Concept

**Plasticity**: Ability to change (learn)
**Meta-plasticity**: Learning how to be plastic
**Adaptive**: Learning rule adapts based on experience

### Formulation

**Standard learning**: $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$

**Meta-plastic learning**: Learning rate $\alpha$ or update rule adapts:
$$\alpha \leftarrow f(\text{history}, \text{current state})$$

### Methods

**Learning to learn**: Meta-learn learning rules
**Adaptive learning rates**: Per-parameter learning rates
**Task-specific adaptation**: Different learning rules for different tasks

### Advantages

**Flexible**: Can adapt to different tasks
**General**: Applicable to various continual learning scenarios
**Efficient**: May reduce need for explicit memory

### Limitations

**Complexity**: More complex to implement and train
**Stability**: May be less stable than fixed methods
**Interpretability**: Harder to understand and debug

## Task-Incremental vs Class-Incremental vs Domain-Incremental

Different continual learning scenarios pose different challenges.

### Task-Incremental Learning

**Setting**: Each task has distinct set of classes
**Task ID**: Provided during training and inference
**Challenge**: Moderate (can use task-specific heads)

**Example**:
- Task 1: Classes {0, 1, 2, 3, 4}
- Task 2: Classes {5, 6, 7, 8, 9}
- Inference: Know which task (use corresponding head)

**Evaluation**: Accuracy on each task (with task ID)

### Class-Incremental Learning

**Setting**: New classes appear over time, old classes may reappear
**Task ID**: Not provided during inference
**Challenge**: High (must distinguish all classes)

**Example**:
- Task 1: Classes {0, 1, 2}
- Task 2: Classes {3, 4, 5}
- Inference: Must classify among {0, 1, 2, 3, 4, 5} without task ID

**Evaluation**: Accuracy on all classes seen so far

### Domain-Incremental Learning

**Setting**: Same task, different input distribution
**Task ID**: May or may not be provided
**Challenge**: Varies (distribution shift)

**Example**:
- Task 1: MNIST (handwritten digits)
- Task 2: SVHN (street view house numbers)
- Same classes, different domains

**Evaluation**: Performance on each domain

### Comparison

| Setting | Task ID | Challenge | Common Methods |
|---------|---------|-----------|----------------|
| Task-incremental | Yes | Low | Task-specific heads |
| Class-incremental | No | High | Replay, regularization |
| Domain-incremental | Maybe | Medium | Domain adaptation |

### Benchmarks

**Split MNIST**: MNIST split into tasks
**Permuted MNIST**: MNIST with pixel permutations
**CIFAR-100**: 100 classes split into tasks
**ImageNet-1000**: Large-scale class-incremental

## Evaluation Protocols and Benchmarks

Standardized evaluation is crucial for comparing continual learning methods.

### Evaluation Metrics

**Average accuracy**: $\frac{1}{T}\sum_{i=1}^T a_{i,T}$ where $a_{i,T}$ is accuracy on task $i$ after learning all $T$ tasks

**Forgetting measure**: $F_i = \max_{j \leq i} a_{j,i} - a_{j,T}$

**Backward transfer**: Performance on old tasks after learning new tasks

**Forward transfer**: Performance on new tasks due to knowledge from old tasks

**Learning efficiency**: How quickly model learns new tasks

### Benchmarks

**MNIST variants**:
- Split MNIST: 5 tasks, 2 classes each
- Permuted MNIST: 10 tasks, pixel permutations
- Rotated MNIST: 10 tasks, rotations

**CIFAR-100**: 100 classes, various splits
**ImageNet**: Large-scale, realistic scenarios
**CORe50**: Object recognition, 50 classes

### Protocols

**Disjoint tasks**: Tasks have no overlap
**Blurry tasks**: Tasks have some overlap
**Online learning**: One example at a time
**Offline learning**: Full access to each task

### Challenges in Evaluation

**Hyperparameter tuning**: Methods may need different hyperparameters
**Compute**: Some methods require more compute
**Fairness**: Ensure fair comparison across methods

## Theoretical Perspectives

Theoretical analysis provides insights into continual learning.

### Information-Theoretic Perspective

**Information bottleneck**: Trade-off between compression and prediction
**Continual learning**: Must compress old information while learning new

**Formulation**:
$$\min I(X_{old}; \theta) \text{ s.t. } I(X_{new}; Y_{new} | \theta) \geq C$$

### Stability-Plasticity Dilemma

**Stability**: Retain old knowledge
**Plasticity**: Learn new knowledge
**Trade-off**: Cannot maximize both simultaneously

**Optimal balance**: Depends on task similarity and data distribution

### Catastrophic Forgetting Bounds

**Theoretical bounds**: On amount of forgetting
**Conditions**: When forgetting is inevitable
**Guarantees**: When methods can prevent forgetting

### PAC-Bayes Analysis

**PAC-Bayes**: Probably Approximately Correct Bayesian framework
**Continual learning**: Bounds on generalization across tasks
**Transfer**: Conditions for positive transfer

### Gradient Alignment

**Observation**: Gradients from different tasks may conflict
**Analysis**: When gradients align vs conflict
**Methods**: Project gradients to reduce conflict

## Key Takeaways

1. **Continual learning** enables models to learn continuously from streams of data, acquiring new knowledge while retaining old knowledge.

2. **Catastrophic forgetting** is the primary challenge, where learning new information causes dramatic loss of previously learned information.

3. **Experience replay** methods store and replay examples from previous tasks, effectively preventing forgetting through revisiting old data.

4. **Elastic Weight Consolidation (EWC)** protects important weights by adding regularization based on Fisher information, preventing changes to critical parameters.

5. **Progressive neural networks** add new columns for each task while freezing old columns, ensuring no forgetting but growing model size.

6. **Meta-plasticity** approaches learn how to learn, adapting the learning process itself to better handle continual learning.

7. **Task scenarios** differ in difficulty: task-incremental (easiest), domain-incremental (medium), class-incremental (hardest).

8. **Evaluation** requires standardized protocols and metrics, including average accuracy, forgetting measures, and transfer metrics.

9. **Theoretical perspectives** provide insights through information theory, stability-plasticity trade-offs, and gradient alignment analysis.

10. **Future directions** include improving efficiency, handling more realistic scenarios, developing better theoretical understanding, and creating methods that scale to many tasks.

## References

- McCloskey, M., & Cohen, N. J. (1989). "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem." Psychology of Learning and Motivation
- Kirkpatrick, J., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." PNAS 2017
- Lopez-Paz, D., & Ranzato, M. (2017). "Gradient Episodic Memory for Continual Learning." NeurIPS 2017
- Chaudhry, A., et al. (2019). "On Tiny Episodic Memories in Continual Learning." arXiv:1902.10486
- Rusu, A. A., et al. (2016). "Progressive Neural Networks." arXiv:1606.04671
- Zenke, F., et al. (2017). "Continual Learning Through Synaptic Intelligence." ICML 2017
- Aljundi, R., et al. (2018). "Memory Aware Synapses: Learning what (not) to forget." ECCV 2018
- van de Ven, G. M., & Tolias, A. S. (2019). "Three Scenarios for Continual Learning." arXiv:1904.07734
- Farquhar, S., & Gal, Y. (2019). "Towards Robust Evaluations of Continual Learning." arXiv:1805.09733
- Lesort, T., et al. (2020). "Continual Learning for Robotics: A Survey." arXiv:1907.00182
