# Curriculum Learning and Multi-Task Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Curriculum Learning Fundamentals](#curriculum-learning-fundamentals)
3. [Self-Paced Learning](#self-paced-learning)
4. [Multi-Task Learning](#multi-task-learning)
5. [Auxiliary Tasks](#auxiliary-tasks)
6. [Hard Parameter Sharing](#hard-parameter-sharing)
7. [Soft Parameter Sharing](#soft-parameter-sharing)
8. [Task Relationships](#task-relationships)
9. [Practical Applications](#practical-applications)
10. [Key Takeaways](#key-takeaways)

## Introduction

Curriculum learning orders training examples from easy to hard, mimicking human learning. Multi-task learning trains models on multiple related tasks simultaneously, improving generalization through shared representations. Both approaches enhance learning efficiency and model performance.

This chapter covers curriculum learning strategies, multi-task learning architectures, and how these techniques improve learning and generalization.

## Curriculum Learning Fundamentals

### Concept

Train on easier examples first, gradually introducing harder ones:

1. **Easy Examples**: Learn basic patterns
2. **Medium Examples**: Build on basics
3. **Hard Examples**: Refine and generalize

### Motivation

- **Human Learning**: Humans learn easier concepts first
- **Stability**: Easier examples provide stable gradients
- **Generalization**: Gradual difficulty improves generalization
- **Convergence**: Faster convergence to better solutions

### Curriculum Design

**Difficulty Metrics**:
- Loss value
- Prediction confidence
- Human annotation
- Domain knowledge

**Scheduling**:
- When to introduce harder examples
- How many easy vs. hard examples
- Rate of curriculum progression

### Implementation

```python
class CurriculumSampler:
    def __init__(self, dataset, difficulty_fn, initial_ratio=0.1):
        self.dataset = dataset
        self.difficulty_fn = difficulty_fn
        self.current_ratio = initial_ratio
        self.difficulties = [difficulty_fn(x) for x in dataset]
    
    def sample_batch(self, batch_size):
        # Sort by difficulty
        sorted_indices = sorted(range(len(self.dataset)), 
                               key=lambda i: self.difficulties[i])
        
        # Sample easiest examples
        num_easy = int(batch_size * self.current_ratio)
        easy_indices = sorted_indices[:num_easy]
        hard_indices = sorted_indices[num_easy:]
        
        # Sample from both
        batch_indices = (random.sample(easy_indices, num_easy) + 
                        random.sample(hard_indices, batch_size - num_easy))
        
        return [self.dataset[i] for i in batch_indices]
    
    def update_curriculum(self, epoch):
        # Gradually increase ratio of hard examples
        self.current_ratio = min(1.0, 0.1 + 0.9 * epoch / max_epochs)
```

## Self-Paced Learning

Self-paced learning automatically determines curriculum from data.

### Algorithm

Learn both model parameters and curriculum:

$$\min_{\mathbf{w}, \mathbf{v}} \mathcal{L}(\mathbf{w}, \mathbf{v}) = \sum_{i=1}^{n} v_i \ell_i(\mathbf{w}) - \lambda \sum_{i=1}^{n} v_i$$

where:
- $\mathbf{w}$: Model parameters
- $\mathbf{v} \in \{0,1\}^n$: Sample selection (curriculum)
- $\ell_i$: Loss for example $i$
- $\lambda$: Difficulty parameter

### Optimization

Alternating optimization:

1. **Fix $\mathbf{v}$, optimize $\mathbf{w}$**: Standard training
2. **Fix $\mathbf{w}$, optimize $\mathbf{v}$**: Select easy examples

$$v_i^* = \begin{cases}
1 & \text{if } \ell_i(\mathbf{w}) < \lambda \\
0 & \text{otherwise}
\end{cases}$$

### Adaptive $\lambda$

Increase $\lambda$ over time:

- Start: Small $\lambda$ (only easiest examples)
- Gradually increase (include more examples)
- End: Large $\lambda$ (all examples)

### Benefits

- Automatic curriculum
- Data-driven difficulty
- Adapts to model state
- No manual curriculum design

## Multi-Task Learning

Multi-task learning trains on multiple tasks simultaneously.

### Motivation

- **Shared Representations**: Tasks share useful features
- **Data Efficiency**: Leverage data from multiple tasks
- **Regularization**: Shared parameters regularize each other
- **Transfer**: Knowledge transfers between tasks

### Formulation

Minimize combined loss:

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \lambda_t \mathcal{L}_t(\theta_{\text{shared}}, \theta_t)$$

where:
- $T$: Number of tasks
- $\lambda_t$: Task weight
- $\theta_{\text{shared}}$: Shared parameters
- $\theta_t$: Task-specific parameters

### Benefits

- Better generalization
- Data efficiency
- Feature learning
- Transfer between tasks

## Auxiliary Tasks

Auxiliary tasks help main task learning.

### Types

**1. Predictive Tasks**:
- Predict related quantities
- E.g., predict depth from RGB

**2. Consistency Tasks**:
- Enforce consistency
- E.g., rotation prediction

**3. Reconstruction Tasks**:
- Reconstruct inputs
- E.g., autoencoder loss

### Example: Depth Estimation

Main task: Depth estimation
Auxiliary tasks:
- Surface normal prediction
- Semantic segmentation
- Edge detection

All share encoder, different decoders.

### Benefits

- Better representations
- More robust features
- Improved main task performance

## Hard Parameter Sharing

Hard parameter sharing uses same layers for all tasks.

### Architecture

```
Input → Shared Layers → Task-Specific Heads
                      → Task 1 Head
                      → Task 2 Head
                      → Task 3 Head
```

### Implementation

```python
class MultiTaskModel(nn.Module):
    def __init__(self, shared_dim, task_dims):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU()
        )
        self.task_heads = nn.ModuleDict({
            f'task_{i}': nn.Linear(shared_dim, dim)
            for i, dim in enumerate(task_dims)
        })
    
    def forward(self, x, task_id):
        shared = self.shared_backbone(x)
        return self.task_heads[f'task_{task_id}'](shared)
```

### Advantages

- Parameter efficient
- Strong regularization
- Natural feature sharing
- Simple architecture

### Disadvantages

- Tasks must be related
- Negative transfer possible
- Less flexibility

## Soft Parameter Sharing

Soft parameter sharing uses separate models with regularization.

### Formulation

Separate parameters per task with similarity regularization:

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \mathcal{L}_t(\theta_t) + \lambda \sum_{i} ||\theta_i^{(1)} - \theta_i^{(2)}||^2$$

### Benefits

- More flexibility
- Can handle unrelated tasks
- Less negative transfer

### Disadvantages

- More parameters
- Weaker regularization
- More complex

## Task Relationships

Understanding task relationships is crucial for multi-task learning.

### Positive Transfer

Tasks help each other:
- Related tasks
- Shared features beneficial
- Improved performance

### Negative Transfer

Tasks hurt each other:
- Conflicting objectives
- Different optimal features
- Worse performance

### Task Grouping

Group related tasks:
- Cluster tasks by similarity
- Share parameters within groups
- Separate parameters between groups

### Task Weighting

Balance task contributions:

**Equal Weighting**: $\lambda_t = 1/T$

**Uncertainty Weighting**: Learn task weights:

$$\mathcal{L} = \sum_t \frac{1}{\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

**Gradient Norm Weighting**: Balance gradient magnitudes

## Practical Applications

### Computer Vision

**Object Detection + Segmentation**:
- Shared backbone
- Detection head + segmentation head
- Both benefit from shared features

**Depth + Normals + Segmentation**:
- Multi-task learning
- Shared encoder
- Multiple decoders

### Natural Language Processing

**Named Entity Recognition + POS Tagging**:
- Shared word embeddings
- Task-specific classifiers
- Both benefit from shared representations

**Translation + Language Modeling**:
- Shared encoder
- Different decoders
- Transfer between tasks

### Reinforcement Learning

**Multiple Environments**:
- Train on multiple tasks
- Shared policy network
- Better generalization

## Key Takeaways

1. **Curriculum Learning**: Orders training examples from easy to hard, mimicking human learning and improving convergence and generalization.

2. **Self-Paced Learning**: Automatically determines curriculum by selecting examples with loss below threshold, adapting difficulty as model improves.

3. **Multi-Task Learning**: Trains on multiple tasks simultaneously with shared representations, improving generalization and data efficiency.

4. **Auxiliary Tasks**: Additional tasks that help main task learning by encouraging better feature learning and more robust representations.

5. **Hard Parameter Sharing**: Uses same layers for all tasks with task-specific heads, providing strong regularization and parameter efficiency.

6. **Soft Parameter Sharing**: Uses separate models per task with regularization encouraging similarity, providing more flexibility than hard sharing.

7. **Task Relationships**: Understanding positive/negative transfer and task similarity is crucial for effective multi-task learning and avoiding negative transfer.

8. **Task Weighting**: Balancing task contributions through equal weighting, uncertainty weighting, or gradient norm weighting is important for optimal performance.

9. **Applications**: Multi-task learning excels in computer vision (detection+segmentation), NLP (NER+POS), and RL (multiple environments).

10. **Practical Considerations**: Curriculum learning and multi-task learning require careful design of curriculum, task selection, and architecture to achieve benefits while avoiding negative transfer.
