# Meta-Learning and Few-Shot Reinforcement Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Meta-Learning Formulation](#meta-learning-formulation)
3. [Model-Agnostic Meta-Learning for RL](#model-agnostic-meta-learning-for-rl)
4. [RL^2: Learning to Reinforcement Learn](#rl2-learning-to-reinforcement-learn)
5. [Fast Adaptation Mechanisms](#fast-adaptation-mechanisms)
6. [Context-Based Meta-Learning](#context-based-meta-learning)
7. [Task Distribution Design](#task-distribution-design)
8. [Few-Shot RL Applications](#few-shot-rl-applications)
9. [Evaluation and Benchmarks](#evaluation-and-benchmarks)
10. [Key Takeaways](#key-takeaways)

## Introduction

Traditional reinforcement learning algorithms require extensive interaction with an environment to learn effective policies. In many real-world scenarios, we need agents that can quickly adapt to new tasks with minimal experience. **Meta-learning** (learning to learn) addresses this challenge by training agents on distributions of tasks, enabling them to rapidly adapt to new tasks from the same distribution.

**Few-shot reinforcement learning** refers to the ability to learn effective policies for new tasks using only a small number of episodes or interactions. This is crucial for:
- **Transfer learning**: Adapting to new environments or task variations
- **Personalization**: Customizing policies for individual users or contexts
- **Robustness**: Handling distribution shifts and novel situations
- **Sample efficiency**: Reducing the data requirements for new tasks

This chapter covers meta-learning formulations for RL, key algorithms including MAML and RL^2, and practical considerations for few-shot RL.

## Meta-Learning Formulation

### Problem Setup

In meta-learning, we have:
- **Meta-training**: Distribution of tasks $p(\mathcal{T})$ for training
- **Meta-testing**: New tasks $\mathcal{T}_{\text{test}} \sim p(\mathcal{T})$ for evaluation

Each task $\mathcal{T}_i$ is a Markov Decision Process (MDP):
$$\mathcal{T}_i = (\mathcal{S}_i, \mathcal{A}_i, P_i, R_i, \gamma, \rho_i)$$

Tasks may differ in:
- State/action spaces: $\mathcal{S}_i$, $\mathcal{A}_i$
- Dynamics: $P_i(s' | s, a)$
- Rewards: $R_i(s, a, s')$
- Initial state distribution: $\rho_i(s_0)$

### Meta-Learning Objective

The goal is to learn a learning algorithm or initialization that enables fast adaptation:

$$\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \left[ \mathcal{L}_{\mathcal{T}}(U^k_\theta(\mathcal{T})) \right]$$

where:
- $\theta$: Meta-parameters (e.g., policy initialization)
- $U^k_\theta(\mathcal{T})$: Adaptation procedure producing task-specific parameters $\phi$
- $\mathcal{L}_{\mathcal{T}}(\phi)$: Loss on task $\mathcal{T}$ with parameters $\phi$
- $k$: Number of adaptation steps

### Adaptation Procedures

**Gradient-based adaptation:**
$$\phi = U^k_\theta(\mathcal{T}) = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta)$$

After $k$ steps:
$$\phi_k = \theta - \alpha \sum_{i=0}^{k-1} \nabla_{\phi_i} \mathcal{L}_{\mathcal{T}}(\phi_i)$$

**Few-shot setting:**
- Training: $K$ episodes per task during meta-training
- Testing: $K$ episodes per task for adaptation (few-shot)
- Evaluation: Performance on adapted policy

### Types of Meta-Learning

**Optimization-based:**
- Learn good initialization $\theta$
- Fast adaptation via few gradient steps
- Examples: MAML, Reptile

**Model-based:**
- Learn recurrent model that accumulates task information
- Fast adaptation via hidden state updates
- Examples: RL^2, SNAIL

**Metric-based:**
- Learn embedding space and similarity metric
- Fast adaptation via nearest neighbors
- Less common in RL

## Model-Agnostic Meta-Learning for RL

Model-Agnostic Meta-Learning (MAML) learns parameter initializations that enable fast adaptation with gradient descent.

### MAML Algorithm

**Meta-objective:**
$$\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i})$$

where adapted parameters are:
$$\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{train}}(f_\theta)$$

**Meta-gradient:**
$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i}) = \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}^{\text{train}}(f_\theta)})$$

Using chain rule:
$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i}) = \nabla_{\phi_i} \mathcal{L}_{\mathcal{T}_i}(f_{\phi_i}) \cdot \nabla_\theta \phi_i$$

where:
$$\nabla_\theta \phi_i = I - \alpha \nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i}^{\text{train}}(f_\theta)$$

### MAML for Reinforcement Learning

**Policy gradient formulation:**

For task $\mathcal{T}_i$, policy gradient is:
$$\nabla_\theta J_{\mathcal{T}_i}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right]$$

**MAML update:**

1. **Inner loop** (adaptation on task $\mathcal{T}_i$):
   - Collect trajectories $\tau_1, \ldots, \tau_K$ using $\pi_\theta$
   - Compute policy gradient: $g_i = \nabla_\theta J_{\mathcal{T}_i}(\theta)$
   - Adapt: $\phi_i = \theta - \alpha g_i$

2. **Outer loop** (meta-update):
   - Collect trajectories $\tau'_1, \ldots, \tau'_K$ using $\pi_{\phi_i}$
   - Compute meta-gradient: $\nabla_\theta J_{\mathcal{T}_i}(\phi_i)$
   - Update: $\theta \leftarrow \theta - \beta \sum_i \nabla_\theta J_{\mathcal{T}_i}(\phi_i)$

**Algorithm:**

```
Initialize meta-parameters θ
for meta-iteration = 1 to M:
    Sample batch of tasks {T_i} ~ p(T)
    for each task T_i:
        # Collect trajectories with current policy
        τ_1, ..., τ_K ~ π_θ
        
        # Compute inner gradient
        g_i = ∇_θ J_{T_i}(θ) using {τ_1, ..., τ_K}
        
        # Adapt parameters
        φ_i = θ - α g_i
        
        # Collect trajectories with adapted policy
        τ'_1, ..., τ'_K ~ π_{φ_i}
        
        # Compute meta-gradient
        ∇_θ L_i = ∇_θ J_{T_i}(φ_i) using {τ'_1, ..., τ'_K}
    
    # Meta-update
    θ ← θ - β Σ_i ∇_θ L_i
```

### Second-Order Derivatives

MAML requires second-order derivatives:
$$\nabla_\theta \phi_i = I - \alpha \nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i}^{\text{train}}(f_\theta)$$

**First-order approximation (FOMAML):**
Ignore second-order term:
$$\nabla_\theta \phi_i \approx I$$

This simplifies computation but may reduce performance.

**Reptile:**
Alternative that avoids second-order derivatives:
$$\theta \leftarrow \theta + \epsilon (\phi_i - \theta)$$

where $\phi_i$ is adapted parameters. This is equivalent to:
$$\theta \leftarrow \theta + \epsilon \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

### Challenges in RL

**High variance:**
- Policy gradients have high variance
- Meta-gradients compound this variance
- Requires many samples per task

**Credit assignment:**
- Need to attribute performance to initialization vs adaptation
- Long episodes make credit assignment difficult

**Task distribution:**
- Tasks must be diverse but learnable
- Poor task distribution leads to poor meta-learning

## RL^2: Learning to Reinforcement Learn

RL^2 frames meta-learning as a partially observable Markov decision process (POMDP), where the hidden state is the current task.

### Formulation

**POMDP:**
- **Observations**: Environment states $s_t$
- **Hidden state**: Task $\mathcal{T}$
- **Actions**: Policy actions $a_t$
- **Rewards**: Task rewards $r_t$

**Recurrent policy:**
$$\pi_\theta(a_t | h_t, s_t)$$

where $h_t$ is hidden state encoding task information:
$$h_t = f_\theta(h_{t-1}, s_{t-1}, a_{t-1}, r_{t-1}, s_t)$$

### Architecture

**RNN-based agent:**
```
Input: (s_t, a_{t-1}, r_{t-1})
Hidden: h_t = RNN(h_{t-1}, [s_t, a_{t-1}, r_{t-1}])
Output: a_t ~ π(h_t, s_t)
```

The RNN accumulates information about the task through experience, enabling fast adaptation.

### Training Procedure

```
Initialize RNN parameters θ
for meta-iteration = 1 to M:
    Sample task T ~ p(T)
    Reset RNN hidden state h_0
    
    # Fast adaptation phase (K episodes)
    for episode = 1 to K:
        s_0 = initial state
        for t = 0 to T:
            a_t ~ π_θ(· | h_t, s_t)
            Execute a_t, observe r_t, s_{t+1}
            h_{t+1} = RNN(h_t, [s_t, a_t, r_t, s_{t+1}])
    
    # Evaluation phase
    for episode = K+1 to K+E:
        s_0 = initial state
        for t = 0 to T:
            a_t ~ π_θ(· | h_t, s_t)
            Execute a_t, observe r_t, s_{t+1}
            h_{t+1} = RNN(h_t, [s_t, a_t, r_t, s_{t+1}])
        
        # Update using policy gradient
        Compute returns G_t
        ∇_θ J = Σ_t ∇_θ log π_θ(a_t | h_t, s_t) G_t
        θ ← θ + α ∇_θ J
```

### Key Insights

**Task identification:**
- RNN learns to identify task from experience
- Hidden state encodes task-specific information
- Enables rapid adaptation without explicit task labels

**Few-shot learning:**
- $K$ episodes for fast adaptation
- Policy improves as RNN accumulates task information
- No explicit gradient-based adaptation needed

**End-to-end learning:**
- Single optimization objective
- No separate inner/outer loops
- Simpler than MAML in some respects

### Variants

**Variational Meta-RL:**
Adds task inference via variational autoencoder:
$$q_\phi(z | \tau_{1:K}) \approx p(z | \mathcal{T})$$

where $z$ is task embedding and $\tau_{1:K}$ are adaptation episodes.

**PEARL (Probabilistic Embeddings for Actor-Critic RL):**
- Learns probabilistic task embeddings
- Uses context set for task inference
- Combines with actor-critic methods

## Fast Adaptation Mechanisms

### Gradient-Based Adaptation

**MAML-style:**
$$\phi = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta)$$

**Multiple steps:**
$$\phi_k = \theta - \alpha \sum_{i=0}^{k-1} \nabla_{\phi_i} \mathcal{L}_{\mathcal{T}}(\phi_i)$$

**Learning rate adaptation:**
Learn per-parameter learning rates:
$$\phi = \theta - \alpha_\theta \odot \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta)$$

where $\alpha_\theta$ are learned meta-parameters.

### Context-Based Adaptation

**Context set:**
Collect context $C = \{(s_i, a_i, r_i, s'_i)\}$ from task.

**Attention mechanism:**
$$h_t = \text{Attention}(s_t, C)$$

Agent attends to relevant context for current state.

**Memory-augmented networks:**
- External memory stores task-specific information
- Read/write operations for fast adaptation
- Examples: Neural Turing Machine, Differentiable Neural Computer

### Hypernetwork Adaptation

**Hypernetworks:**
Learn to generate task-specific parameters:
$$\phi_i = g_\theta(\text{task\_embedding}_i)$$

where $g_\theta$ is a hypernetwork.

**Conditional batch normalization:**
Adapt normalization statistics per task:
$$\text{BN}(x) = \gamma_{\mathcal{T}} \frac{x - \mu_{\mathcal{T}}}{\sigma_{\mathcal{T}}} + \beta_{\mathcal{T}}$$

where $\gamma_{\mathcal{T}}, \beta_{\mathcal{T}}$ are task-specific.

## Context-Based Meta-Learning

Context-based methods use experience from a task to form a context representation that guides policy.

### Context Encoders

**Simple aggregation:**
$$c_{\mathcal{T}} = \frac{1}{K} \sum_{i=1}^K \text{encoder}(s_i, a_i, r_i, s'_i)$$

**Attention-based:**
$$c_{\mathcal{T}} = \sum_{i=1}^K \alpha_i \text{encoder}(s_i, a_i, r_i, s'_i)$$

where $\alpha_i = \text{softmax}(\text{query} \cdot \text{key}_i)$.

**Set encoders:**
Use permutation-invariant architectures:
- Deep Sets
- Set Transformers
- Graph Neural Networks

### Context Usage

**Conditional policies:**
$$\pi(a | s, c_{\mathcal{T}})$$

Policy conditions on context for task-specific behavior.

**Context-aware value functions:**
$$Q(s, a, c_{\mathcal{T}})$$

Value function uses context to estimate task-specific values.

### PEARL Algorithm

Probabilistic Embeddings for Actor-Critic RL:

1. **Task inference:**
   $$q_\phi(z | c) = \mathcal{N}(\mu_\phi(c), \sigma_\phi(c))$$
   
   where $c$ is context set from task.

2. **Context encoder:**
   $$c = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^K$$
   $$h_c = \text{SetEncoder}(c)$$

3. **Conditional policy:**
   $$\pi_\theta(a | s, z)$$
   
   where $z \sim q_\phi(z | c)$.

4. **Training:**
   - Maximize expected return
   - Minimize KL divergence: $D_{KL}(q_\phi(z | c) \| p(z))$
   - Use reparameterization trick for gradients

## Task Distribution Design

The task distribution $p(\mathcal{T})$ is crucial for meta-learning success.

### Task Diversity

**Requirements:**
- **Diversity**: Tasks should cover distribution of interest
- **Learnability**: Tasks should be solvable with few-shot adaptation
- **Relevance**: Tasks should be related (shared structure)

**Too diverse:**
- No shared structure to learn
- Meta-learning fails

**Too similar:**
- Trivial adaptation
- Poor generalization

### Task Generation Strategies

**Parameterized tasks:**
- Vary reward functions: $R(s, a; w)$ where $w \sim p(w)$
- Vary dynamics: $P(s' | s, a; \theta)$ where $\theta \sim p(\theta)$
- Vary initial states: $\rho_0(s; \phi)$ where $\phi \sim p(\phi)$

**Curriculum learning:**
- Start with easy tasks
- Gradually increase difficulty
- Improves meta-learning stability

**Task augmentation:**
- Apply transformations to base tasks
- Rotation, scaling, noise
- Increases diversity

### Domain Randomization

**Visual domain randomization:**
- Vary textures, lighting, colors
- Enables sim-to-real transfer
- Related to meta-learning

**Dynamics randomization:**
- Vary physics parameters
- Mass, friction, gravity
- Robust policies

### Task Sampling

**Uniform sampling:**
$$\mathcal{T}_i \sim \text{Uniform}(p(\mathcal{T}))$$

**Curriculum sampling:**
$$\mathcal{T}_i \sim p(\mathcal{T} | \text{difficulty})$$

**Hard example mining:**
Focus on tasks where current policy fails.

## Few-Shot RL Applications

### Sim-to-Real Transfer

**Problem:**
- Train in simulation
- Deploy in real world
- Domain shift

**Meta-learning solution:**
- Meta-train on diverse simulated tasks
- Few-shot adapt to real-world task
- Enables rapid deployment

### Personalization

**Problem:**
- Single policy for all users
- Individual preferences vary

**Meta-learning solution:**
- Meta-train on user distribution
- Few-shot adapt to new user
- Personalized policies

### Continual Learning

**Problem:**
- Learn sequence of tasks
- Avoid catastrophic forgetting

**Meta-learning solution:**
- Fast adaptation to new tasks
- Retain knowledge from previous tasks
- Continual adaptation

### Multi-Task Learning

**Problem:**
- Learn multiple related tasks
- Share knowledge across tasks

**Meta-learning solution:**
- Meta-learn shared structure
- Task-specific adaptation
- Better than independent learning

## Evaluation and Benchmarks

### Evaluation Protocol

**Meta-training:**
- Train on task distribution $p(\mathcal{T})$
- No evaluation on test tasks

**Meta-testing:**
1. Sample test task $\mathcal{T}_{\text{test}} \sim p(\mathcal{T})$
2. **Fast adaptation**: $K$ episodes for adaptation
3. **Evaluation**: $E$ episodes for performance measurement
4. Report average return

### Few-Shot Setting

**$K$-shot learning:**
- $K$ episodes for adaptation
- Typical: $K = 1, 5, 10$

**Performance metrics:**
- **Sample efficiency**: Episodes to reach target performance
- **Final performance**: Return after adaptation
- **Adaptation speed**: Improvement per episode

### Benchmarks

**ML10/ML45 (Meta-World):**
- 10/45 manipulation tasks
- Diverse but related
- Standard meta-RL benchmark

**Procgen:**
- Procedurally generated environments
- Infinite task distribution
- Tests generalization

**DMControl:**
- Continuous control tasks
- Vary dynamics/rewards
- Realistic benchmark

### Common Pitfalls

**Data leakage:**
- Ensure test tasks not seen during training
- Proper train/test splits

**Overfitting:**
- Meta-learners can overfit to training task distribution
- Need diverse test tasks

**Evaluation variance:**
- Few-shot evaluation has high variance
- Report confidence intervals
- Multiple random seeds

## Key Takeaways

1. **Meta-learning enables few-shot RL**: By learning to learn, agents can adapt to new tasks with minimal experience.

2. **MAML learns good initializations**: Gradient-based meta-learning finds parameter initializations that enable fast adaptation.

3. **RL^2 uses recurrent policies**: RNN-based agents accumulate task information, enabling adaptation without explicit gradients.

4. **Context is crucial**: Context-based methods use experience to form task representations that guide policies.

5. **Task distribution matters**: The distribution of meta-training tasks determines what can be learned and how well it generalizes.

6. **Fast adaptation mechanisms vary**: Gradient-based, context-based, and hypernetwork approaches each have strengths.

7. **Evaluation requires care**: Proper protocols, diverse test tasks, and statistical significance are essential.

8. **Applications are diverse**: Sim-to-real, personalization, continual learning all benefit from meta-learning.

9. **Sample efficiency is key advantage**: Few-shot RL reduces data requirements for new tasks.

10. **Active research area**: Meta-learning for RL continues to evolve with new architectures, algorithms, and applications.
