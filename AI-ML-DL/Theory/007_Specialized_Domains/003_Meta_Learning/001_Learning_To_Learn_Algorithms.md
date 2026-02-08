# Learning to Learn Algorithms

## Table of Contents

1. [Introduction to Meta-Learning](#introduction-to-meta-learning)
2. [MAML: Model-Agnostic Meta-Learning](#maml-model-agnostic-meta-learning)
3. [Reptile Algorithm](#reptile-algorithm)
4. [Gradient-Based Meta-Learning](#gradient-based-meta-learning)
5. [Second-Order Gradients and Approximations](#second-order-gradients-and-approximations)
6. [Task Distribution and Sampling](#task-distribution-and-sampling)
7. [Inner and Outer Loop Optimization](#inner-and-outer-loop-optimization)
8. [Meta-Learning Architectures](#meta-learning-architectures)
9. [Applications and Extensions](#applications-and-extensions)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Meta-Learning

Meta-learning, or "learning to learn," aims to develop algorithms that can quickly adapt to new tasks with limited data by leveraging experience from related tasks.

### Problem Formulation

**Standard Learning**: Learn from dataset $\mathcal{D}$:
$$\theta^* = \arg\min_\theta \mathcal{L}(\theta, \mathcal{D})$$

**Meta-Learning**: Learn to learn across tasks $\mathcal{T}_1, \ldots, \mathcal{T}_n$:
$$\phi^* = \arg\min_\phi \sum_{i=1}^{n} \mathcal{L}(\theta_i^*(\phi), \mathcal{D}_i^{\text{test}})$$

where $\theta_i^*(\phi)$ is learned for task $i$ using meta-parameters $\phi$.

### Few-Shot Learning Setup

**N-way K-shot**: 
- $N$ classes
- $K$ examples per class for training
- Goal: Classify new examples from these $N$ classes

**Support Set**: Training examples $\mathcal{S} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{NK}$

**Query Set**: Test examples $\mathcal{Q} = \{(\mathbf{x}_j, y_j)\}_{j=1}^{M}$

### Meta-Learning Approaches

1. **Metric-Based**: Learn distance metrics (Siamese networks, Prototypical networks)
2. **Model-Based**: Use memory-augmented networks (MANN, Neural Turing Machine)
3. **Optimization-Based**: Learn initialization or optimizer (MAML, Reptile)

### Evaluation Protocol

**Meta-Training**: Train on tasks $\mathcal{T}_{\text{train}}$

**Meta-Testing**: Evaluate on new tasks $\mathcal{T}_{\text{test}}$

**Episodic Training**: Each episode samples a task, support set, and query set.

---

## MAML: Model-Agnostic Meta-Learning

MAML learns a good parameter initialization that enables fast adaptation to new tasks with few gradient steps.

### MAML Objective

For task distribution $p(\mathcal{T})$, MAML optimizes:

$$\min_\phi \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[\mathcal{L}_{\mathcal{T}}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta, \mathcal{D}^{\text{tr}}))\right]$$

where:
- $\phi$ are meta-parameters (initialization)
- $\theta = \phi$ initially
- $\alpha$ is inner-loop learning rate
- $\mathcal{D}^{\text{tr}}$ is training set for task $\mathcal{T}$

### Algorithm

**MAML Algorithm**:

1. Sample task $\mathcal{T}_i \sim p(\mathcal{T})$
2. Sample support set $\mathcal{D}^{\text{tr}}$ and query set $\mathcal{D}^{\text{test}}$
3. **Inner Loop**: Compute adapted parameters:
   $$\theta_i' = \phi - \alpha \nabla_\phi \mathcal{L}_{\mathcal{T}_i}(\phi, \mathcal{D}^{\text{tr}})$$
4. **Outer Loop**: Update meta-parameters:
   $$\phi \leftarrow \phi - \beta \nabla_\phi \mathcal{L}_{\mathcal{T}_i}(\theta_i', \mathcal{D}^{\text{test}})$$

### First-Order MAML (FOMAML)

Approximate second-order gradients by ignoring second-order terms:

$$\nabla_\phi \mathcal{L}(\theta') \approx \nabla_{\theta'} \mathcal{L}(\theta')$$

**Advantage**: Faster computation, no Hessian needed.

**Disadvantage**: Less accurate, may reduce performance.

### MAML Variants

**MAML++**: Multiple inner-loop steps, learnable learning rates, batch normalization statistics.

**CAVIA**: Context adaptation via inner adaptation, only adapts subset of parameters.

**LEO**: Latent embedding optimization, operates in low-dimensional latent space.

---

## Reptile Algorithm

Reptile is a first-order meta-learning algorithm that is simpler than MAML but often performs comparably.

### Reptile Update

After $k$ inner-loop steps:

$$\phi \leftarrow \phi + \epsilon (\theta_k - \phi)$$

where $\theta_k$ is parameters after $k$ gradient steps from $\phi$.

### Algorithm

**Reptile Algorithm**:

1. Sample task $\mathcal{T}_i$
2. Initialize $\theta_0 = \phi$
3. **Inner Loop**: Update $k$ times:
   $$\theta_{t+1} = \theta_t - \alpha \nabla_{\theta_t} \mathcal{L}_{\mathcal{T}_i}(\theta_t, \mathcal{D}^{\text{tr}})$$
4. **Outer Loop**: Update meta-parameters:
   $$\phi \leftarrow \phi + \epsilon (\theta_k - \phi)$$

### Interpretation

Reptile moves initialization toward solutions of multiple tasks:

$$\phi \leftarrow \phi + \epsilon \mathbb{E}_\mathcal{T}[\theta_k^*(\phi) - \phi]$$

where $\theta_k^*(\phi)$ is solution after $k$ steps.

### Comparison with MAML

| Aspect | MAML | Reptile |
|--------|------|---------|
| Gradients | Second-order | First-order |
| Computation | Expensive | Cheap |
| Performance | Often better | Comparable |
| Simplicity | Complex | Simple |

### Advantages

- **Simplicity**: No need for second-order gradients
- **Efficiency**: Faster than MAML
- **Flexibility**: Works with any optimizer

---

## Gradient-Based Meta-Learning

Gradient-based meta-learning uses gradients to adapt to new tasks, learning how to learn effectively.

### Meta-Gradient Descent

Learn learning rates and update rules:

$$\theta_{t+1} = \theta_t - \alpha_\phi(\theta_t, \nabla_{\theta_t} \mathcal{L}) \odot \nabla_{\theta_t} \mathcal{L}$$

where $\alpha_\phi$ is learned learning rate function.

### Learned Optimizers

**LSTM Optimizer**: Use LSTM to predict parameter updates:

$$h_t = \text{LSTM}([\nabla_{\theta_t} \mathcal{L}, \mathcal{L}, \theta_t], h_{t-1})$$
$$\theta_{t+1} = \theta_t + f_\phi(h_t)$$

**Meta-Learned SGD**: Learn per-parameter learning rates:

$$\theta_{t+1}^{(i)} = \theta_t^{(i)} - \alpha_\phi^{(i)} \nabla_{\theta_t^{(i)}} \mathcal{L}$$

### Gradient-Based Hyperparameter Optimization

Learn hyperparameters via gradients:

$$\min_\lambda \mathbb{E}_{\mathcal{T}}[\mathcal{L}_{\mathcal{T}}(\theta^*(\lambda))]$$

where $\theta^*(\lambda)$ depends on hyperparameters $\lambda$.

### Implicit Differentiation

For fixed-point solutions:

$$\frac{\partial \theta^*}{\partial \phi} = \left(I - \frac{\partial^2 \mathcal{L}}{\partial \theta^2}\right)^{-1} \frac{\partial^2 \mathcal{L}}{\partial \theta \partial \phi}$$

---

## Second-Order Gradients and Approximations

MAML requires second-order gradients, which can be expensive. Various approximations trade accuracy for efficiency.

### Exact Second-Order Gradients

**Gradient of Gradient**:

$$\frac{\partial \mathcal{L}(\theta')}{\partial \phi} = \frac{\partial \mathcal{L}(\theta')}{\partial \theta'} \cdot \frac{\partial \theta'}{\partial \phi}$$

where:
$$\frac{\partial \theta'}{\partial \phi} = I - \alpha \frac{\partial^2 \mathcal{L}(\phi)}{\partial \phi^2}$$

**Hessian-Vector Products**: Compute without full Hessian:

$$H\mathbf{v} = \nabla_\phi (\mathbf{v}^T \nabla_\phi \mathcal{L}(\phi))$$

### First-Order Approximation

**FOMAML**: Ignore second-order terms:

$$\frac{\partial \mathcal{L}(\theta')}{\partial \phi} \approx \frac{\partial \mathcal{L}(\theta')}{\partial \theta'}$$

**Trade-off**: Faster but less accurate.

### Hessian-Free Methods

**Conjugate Gradients**: Solve linear system:

$$H\mathbf{x} = \mathbf{b}$$

**Finite Differences**: Approximate Hessian:

$$H_{ij} \approx \frac{\mathcal{L}(\phi + \epsilon \mathbf{e}_i + \epsilon \mathbf{e}_j) - \mathcal{L}(\phi + \epsilon \mathbf{e}_i) - \mathcal{L}(\phi + \epsilon \mathbf{e}_j) + \mathcal{L}(\phi)}{\epsilon^2}$$

### Truncated Backpropagation

**Truncated BPTT**: Only backpropagate through last $k$ steps:

$$\frac{\partial \mathcal{L}(\theta_k')}{\partial \phi} \approx \frac{\partial \mathcal{L}(\theta_k')}{\partial \theta_k'} \cdot \prod_{i=k-T}^{k-1} \frac{\partial \theta_{i+1}'}{\partial \theta_i'}$$

**Trade-off**: Reduces memory but may lose information.

---

## Task Distribution and Sampling

The task distribution $p(\mathcal{T})$ critically affects meta-learning performance.

### Task Definition

**Task** $\mathcal{T}_i = (\mathcal{D}_i^{\text{tr}}, \mathcal{D}_i^{\text{test}}, \mathcal{L}_i)$:
- Training data $\mathcal{D}_i^{\text{tr}}$
- Test data $\mathcal{D}_i^{\text{test}}$
- Loss function $\mathcal{L}_i$

### Task Sampling Strategies

**Uniform Sampling**: Sample tasks uniformly:
$$\mathcal{T}_i \sim \text{Uniform}(\mathcal{T}_{\text{train}})$$

**Curriculum Learning**: Start with easy tasks, gradually increase difficulty:
$$p(\mathcal{T}) = \begin{cases}
\text{Easy} & \text{if } t < T_1 \\
\text{Medium} & \text{if } T_1 \leq t < T_2 \\
\text{Hard} & \text{if } t \geq T_2
\end{cases}$$

**Hard Example Mining**: Focus on difficult tasks:
$$p(\mathcal{T}) \propto \exp(\lambda \mathcal{L}(\theta^*(\phi), \mathcal{T}))$$

### Task Diversity

**Importance**: Diverse tasks improve generalization:
- Different classes
- Different domains
- Different difficulty levels

**Task Augmentation**: Create new tasks via:
- Data augmentation
- Task transformation
- Synthetic tasks

### Domain Adaptation

**Domain Shift**: Tasks from different domains:
$$\mathcal{T}_{\text{source}} \sim p_{\text{source}}(\mathcal{T})$$
$$\mathcal{T}_{\text{target}} \sim p_{\text{target}}(\mathcal{T})$$

**Domain Adaptation Meta-Learning**: Learn to adapt across domains.

---

## Inner and Outer Loop Optimization

Meta-learning involves nested optimization: inner loop adapts to tasks, outer loop updates meta-parameters.

### Inner Loop

**Objective**: Adapt parameters to task:
$$\theta^* = \arg\min_\theta \mathcal{L}_{\mathcal{T}}(\theta, \mathcal{D}^{\text{tr}})$$

**Methods**:
- Gradient descent: $\theta_{t+1} = \theta_t - \alpha \nabla_{\theta_t} \mathcal{L}$
- Adam: Adaptive learning rates
- Natural gradients: Account for parameter space geometry

**Steps**: Typically 1-10 gradient steps.

### Outer Loop

**Objective**: Update meta-parameters:
$$\phi^* = \arg\min_\phi \mathbb{E}_{\mathcal{T}}[\mathcal{L}_{\mathcal{T}}(\theta^*(\phi), \mathcal{D}^{\text{test}})]$$

**Methods**:
- Gradient descent: $\phi \leftarrow \phi - \beta \nabla_\phi \mathcal{L}$
- Adam: Often used for stability
- Second-order methods: More accurate but expensive

**Learning Rate**: Typically smaller than inner loop: $\beta < \alpha$.

### Bi-Level Optimization

**Upper Level**: Meta-parameters $\phi$
**Lower Level**: Task parameters $\theta(\phi)$

**Challenge**: $\theta(\phi)$ depends on $\phi$ implicitly.

**Solution**: Use implicit differentiation or unrolled optimization.

### Convergence

**Inner Loop**: Should converge quickly (few steps).

**Outer Loop**: May require many iterations.

**Stability**: Balance between inner and outer learning rates critical.

---

## Meta-Learning Architectures

Different architectures enable different meta-learning capabilities.

### Memory-Augmented Networks

**Neural Turing Machine (NTM)**: External memory for few-shot learning.

**Memory-Augmented Neural Networks (MANN)**: Use memory to store and retrieve examples.

**Differentiable Neural Computer (DNC)**: Improved memory access.

### Attention-Based

**Matching Networks**: Use attention to match support and query examples:

$$p(y|\mathbf{x}, \mathcal{S}) = \sum_{i=1}^{|\mathcal{S}|} a(\mathbf{x}, \mathbf{x}_i) y_i$$

where $a$ is attention function.

**Prototypical Networks**: Learn prototypes for each class:

$$\mathbf{c}_k = \frac{1}{|\mathcal{S}_k|} \sum_{(\mathbf{x}_i, y_i) \in \mathcal{S}_k} f_\phi(\mathbf{x}_i)$$

### Hypernetworks

**Hypernetworks**: Generate weights for main network:

$$W = g_\psi(\mathbf{z})$$

where $\mathbf{z}$ encodes task information.

**Conditional Hypernetworks**: Condition on task:

$$W_\mathcal{T} = g_\psi(\text{encode}(\mathcal{T}))$$

### Learned Initializations

**MAML**: Learns initialization $\phi$.

**LEO**: Learns initialization in latent space:

$$\phi = h_\psi(\mathbf{z})$$

where $\mathbf{z}$ is low-dimensional latent.

---

## Applications and Extensions

### Few-Shot Classification

**Problem**: Classify with few examples per class.

**Approach**: Learn to quickly adapt to new classes.

**Results**: MAML achieves strong performance on MiniImagenet, Omniglot.

### Few-Shot Regression

**Problem**: Learn function from few examples.

**Approach**: Meta-learn to adapt regression models.

**Example**: Sinusoid regression, where each task is a different sinusoid.

### Reinforcement Learning

**Problem**: Learn policies quickly in new environments.

**Approach**: Meta-learn policy initialization or learning algorithm.

**MAML-RL**: Apply MAML to RL, adapt policies with few episodes.

### Continual Learning

**Problem**: Learn new tasks without forgetting old ones.

**Approach**: Meta-learn to balance new and old knowledge.

**Gradient Episodic Memory**: Store gradients from previous tasks.

### Domain Generalization

**Problem**: Generalize to unseen domains.

**Approach**: Meta-learn across multiple domains.

**MLDG**: Meta-learning for domain generalization.

### Neural Architecture Search

**Problem**: Find optimal architectures.

**Approach**: Meta-learn architecture search strategies.

**DARTS**: Differentiable architecture search.

---

## Key Takeaways

1. **Meta-Learning Goal**: Learn to learn by leveraging experience across tasks, enabling fast adaptation to new tasks with limited data through learned initialization or optimization strategies.

2. **MAML Framework**: Learns parameter initialization $\phi$ that enables fast adaptation via $\theta' = \phi - \alpha \nabla_\phi \mathcal{L}(\phi)$, requiring second-order gradients but achieving strong few-shot performance.

3. **Reptile Algorithm**: Simpler first-order alternative to MAML, updates $\phi \leftarrow \phi + \epsilon(\theta_k - \phi)$ after $k$ inner steps, often achieving comparable performance with lower computational cost.

4. **Gradient-Based Meta-Learning**: Uses gradients to adapt, including learned optimizers (LSTM optimizers), meta-learned learning rates, and gradient-based hyperparameter optimization.

5. **Second-Order Gradients**: MAML requires expensive second-order gradients. Approximations like FOMAML, Hessian-free methods, and truncated backpropagation trade accuracy for efficiency.

6. **Task Distribution**: Task sampling strategy (uniform, curriculum, hard mining) and diversity critically affect meta-learning performance and generalization.

7. **Nested Optimization**: Inner loop adapts to tasks (1-10 steps), outer loop updates meta-parameters (many iterations), requiring careful balance of learning rates for stability.

8. **Architectures**: Memory-augmented networks (MANN), attention-based (Matching/Prototypical networks), hypernetworks, and learned initializations enable different meta-learning capabilities.

9. **Applications**: Few-shot classification/regression, reinforcement learning, continual learning, domain generalization, and neural architecture search benefit from meta-learning.

10. **Challenges**: Computational cost (second-order gradients), task distribution design, stability (inner/outer loop balance), and generalization to truly novel tasks remain active research areas.
