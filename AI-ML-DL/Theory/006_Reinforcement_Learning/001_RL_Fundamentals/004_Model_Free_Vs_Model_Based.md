# Model-Free Versus Model-Based Reinforcement Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Model-Free Methods](#model-free-methods)
3. [Model-Based Methods](#model-based-methods)
4. [The Dyna Architecture](#the-dyna-architecture)
5. [World Models and Learned Dynamics](#world-models-and-learned-dynamics)
6. [MuZero: Combining Planning and Learning](#muzero-combining-planning-and-learning)
7. [Sample Efficiency Comparison](#sample-efficiency-comparison)
8. [When to Use Each Approach](#when-to-use-each-approach)
9. [Hybrid Approaches](#hybrid-approaches)
10. [Key Takeaways](#key-takeaways)

## Introduction

Reinforcement learning algorithms can be broadly categorized into two paradigms: **model-free** and **model-based** methods. The fundamental distinction lies in whether the agent explicitly learns or uses a model of the environment's dynamics (transition probabilities and reward function) to make decisions.

In **model-free** approaches, the agent learns value functions or policies directly from experience without constructing an explicit model of the environment. These methods include Q-learning, SARSA, and policy gradient algorithms.

In **model-based** approaches, the agent learns or is given a model of the environment dynamics, which it then uses for planning. This model typically consists of:
- Transition model: $P(s' | s, a)$ - probability of transitioning to state $s'$ from state $s$ after taking action $a$
- Reward model: $R(s, a, s')$ - expected reward for transition $(s, a, s')$

The choice between these paradigms involves fundamental trade-offs in sample efficiency, computational complexity, and applicability to different problem domains.

## Model-Free Methods

### Q-Learning

Q-learning is a classic model-free algorithm that learns the optimal action-value function $Q^*(s, a)$ directly from experience. The update rule is:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

where $\alpha$ is the learning rate and $\gamma$ is the discount factor.

**Key characteristics:**
- No explicit model of environment dynamics
- Learns directly from $(s, a, r, s')$ tuples
- Off-policy algorithm (can learn optimal policy while following exploratory policy)
- Requires many samples to converge

### Policy Gradient Methods

Policy gradient methods parameterize the policy directly as $\pi_\theta(a | s)$ and optimize it using gradient ascent:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

where $J(\theta)$ is the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$

The policy gradient theorem provides:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right]$$

where $\hat{A}_t$ is an advantage estimator.

**Advantages:**
- Can handle continuous action spaces naturally
- Direct policy optimization
- Can learn stochastic policies

**Disadvantages:**
- High variance in gradient estimates
- Typically requires more samples than value-based methods
- Slower convergence

### Actor-Critic Methods

Actor-critic methods combine value functions and policy gradients:

- **Actor**: Policy $\pi_\theta(a | s)$ updated via policy gradient
- **Critic**: Value function $V_\phi(s)$ or $Q_\phi(s, a)$ used to reduce variance

The critic provides a baseline or advantage estimate:

$$\hat{A}_t = Q_\phi(s_t, a_t) - V_\phi(s_t)$$

## Model-Based Methods

### Learned Transition Models

In model-based RL, we learn approximations $\hat{P}(s' | s, a)$ and $\hat{R}(s, a, s')$ from data. This is typically framed as a supervised learning problem:

Given dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$, we train:
- Transition model: $s' \sim \hat{P}_\phi(s' | s, a)$
- Reward model: $\hat{r} = \hat{R}_\psi(s, a, s')$

The models can be deterministic or stochastic. For continuous state spaces, common choices include:
- Neural networks: $s' = f_\phi(s, a)$
- Gaussian models: $P(s' | s, a) = \mathcal{N}(\mu_\phi(s, a), \Sigma_\phi(s, a))$

### Planning with Learned Models

Once a model is learned, planning algorithms can be used:

**Value Iteration:**
$$V_{k+1}(s) = \max_a \sum_{s'} \hat{P}(s' | s, a) \left[ \hat{R}(s, a, s') + \gamma V_k(s') \right]$$

**Policy Iteration:**
1. Policy evaluation: Solve $V^\pi(s) = \sum_{s'} \hat{P}(s' | s, \pi(s)) [\hat{R}(s, \pi(s), s') + \gamma V^\pi(s')]$
2. Policy improvement: $\pi'(s) = \arg\max_a \sum_{s'} \hat{P}(s' | s, a) [\hat{R}(s, a, s') + \gamma V^\pi(s')]$

**Monte Carlo Tree Search (MCTS):**
- Uses model for forward simulation
- Balances exploration and exploitation via UCB
- Particularly effective in discrete action spaces

### Model Learning Challenges

**Model Bias:**
Learned models may be inaccurate, especially in regions of state-action space with limited data. This can lead to:
- Compounding errors during long-horizon planning
- Overfitting to training distribution
- Distributional shift between training and planning

**Model Uncertainty:**
Uncertainty quantification is crucial. Methods include:
- Ensemble models: $\{\hat{P}_1, \ldots, \hat{P}_K\}$
- Bayesian neural networks
- Probabilistic models with explicit uncertainty

## The Dyna Architecture

The Dyna architecture combines model-free learning with model-based planning, addressing limitations of both approaches.

### Architecture Overview

Dyna maintains:
1. **Direct RL component**: Model-free learning (e.g., Q-learning)
2. **Model learning component**: Learns $\hat{P}$ and $\hat{R}$
3. **Planning component**: Uses model for simulated experience

### Algorithm

```
Initialize Q(s, a), Model(s, a)
for each step:
    s = current state
    a = choose action using Q (e.g., epsilon-greedy)
    take action a, observe r, s'
    
    # Direct RL update
    Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
    
    # Model learning
    Model(s, a) ← (r, s')
    
    # Planning: n simulated updates
    for i = 1 to n:
        (s̃, ã) ← random previously observed state-action
        (r̃, s̃') ← Model(s̃, ã)
        Q(s̃, ã) ← Q(s̃, ã) + α[r̃ + γ max_ã' Q(s̃', ã') - Q(s̃, ã)]
```

### Advantages

- **Sample efficiency**: Planning updates provide additional learning signal
- **Flexibility**: Can adjust planning amount (parameter $n$) based on computational budget
- **Robustness**: Direct RL component ensures learning even if model is imperfect

### Variants

**Prioritized Sweeping:**
Prioritizes planning updates for states with large Bellman errors:

$$\text{priority}(s, a) = |r + \gamma \max_{a'} Q(s', a') - Q(s, a)|$$

**Dyna-Q+**: Adds exploration bonus for rarely visited state-action pairs.

## World Models and Learned Dynamics

World Models represent a paradigm where deep neural networks learn compact representations of environment dynamics.

### World Models Architecture

The World Model consists of three components:

1. **Vision Model (V)**: Encodes observations into latent states
   $$z_t = V(o_t)$$

2. **Memory Model (M)**: Predicts next latent state and reward
   $$z_{t+1}, r_t = M(z_t, a_t)$$

3. **Controller (C)**: Policy that acts in latent space
   $$a_t = C(z_t)$$

### Training Procedure

```
# Phase 1: Collect random rollouts
for episode in episodes:
    collect (o_t, a_t, r_t, o_{t+1}) tuples

# Phase 2: Train VAE (Vision Model)
train V to encode/decode observations

# Phase 3: Train RNN (Memory Model)
train M to predict z_{t+1}, r_t from z_t, a_t

# Phase 4: Train Controller
train C using CMA-ES or evolution strategies in latent space
```

### Advantages

- **Sample efficiency**: Training controller in compact latent space requires fewer environment interactions
- **Fast planning**: Can simulate many trajectories quickly in latent space
- **Transfer learning**: Learned representations may transfer across tasks

### Limitations

- **Representation quality**: Poor latent representations lead to poor policies
- **Distribution shift**: Latent space may not capture all relevant dynamics
- **Complexity**: Requires careful design of each component

## MuZero: Combining Planning and Learning

MuZero extends AlphaZero by learning a model that is optimized for planning rather than accurately predicting observations.

### Key Innovation

Instead of learning a model that predicts raw observations, MuZero learns:
- **Representation function**: $h_t = f_\theta(o_1, \ldots, o_t)$
- **Dynamics function**: $(h_{t+1}, r_t) = g_\theta(h_t, a_t)$
- **Prediction function**: $(p_t, v_t) = p_\theta(h_t)$

where $p_t$ is policy logits and $v_t$ is value estimate.

### Training Objective

MuZero optimizes three losses:

1. **Policy loss**: Cross-entropy between predicted policy and MCTS policy
   $$L_p = -\sum_t \pi_t \log p_t$$

2. **Value loss**: MSE between predicted value and observed return
   $$L_v = \sum_t (v_t - z_t)^2$$

3. **Reward loss**: MSE between predicted and observed rewards
   $$L_r = \sum_t (r_t - u_t)^2$$

Total loss: $L = L_p + L_v + L_r$

### Planning with MuZero

MCTS uses learned dynamics model:

```
def mcts_search(h, model):
    for simulation in range(num_simulations):
        s = h
        path = []
        while s in tree:
            a = select_action(s)  # UCB
            s, r = model.dynamics(s, a)
            path.append((s, a, r))
        expand_and_evaluate(s, model)
        backup(path)
    return policy_from_tree(root)
```

### Advantages

- **State-of-the-art performance**: Achieves superhuman performance in games
- **Model optimized for planning**: Not constrained to predict observations accurately
- **Handles partial observability**: Representation function aggregates history

## Sample Efficiency Comparison

### Theoretical Analysis

**Model-free methods:**
- Sample complexity: $O(|\mathcal{S}| \cdot |\mathcal{A}| / \epsilon^2)$ for tabular Q-learning
- Each sample used once for direct update

**Model-based methods:**
- Model learning: $O(|\mathcal{S}| \cdot |\mathcal{A}| / \epsilon_m^2)$ samples
- Planning: Can generate unlimited simulated samples
- Effective sample complexity depends on model accuracy

### Empirical Comparison

| Method | Samples to 50% Performance | Samples to 90% Performance | Final Performance |
|--------|---------------------------|----------------------------|------------------|
| Q-Learning | 10K | 100K | 95% |
| DQN | 50K | 500K | 98% |
| Model-Based (learned) | 5K | 20K | 85% |
| Dyna-Q | 3K | 15K | 92% |
| MuZero | 1K | 10K | 99% |

*Note: Numbers are illustrative and vary by domain*

### Factors Affecting Sample Efficiency

1. **Model accuracy**: Better models enable more effective planning
2. **Planning budget**: More planning steps improve sample efficiency but increase computation
3. **Exploration**: Model-based methods can plan exploration more effectively
4. **Task complexity**: Simple tasks favor model-free; complex tasks may benefit from models

## When to Use Each Approach

### Use Model-Free When:

- **Simple environments**: Low-dimensional state spaces, deterministic dynamics
- **Online learning**: Need to adapt quickly to changing environment
- **Computational constraints**: Limited resources for planning
- **Sufficient data**: Can afford many environment interactions
- **Stochastic policies**: Need explicit exploration in policy

### Use Model-Based When:

- **Sample efficiency critical**: Expensive or limited environment interactions
- **Safety constraints**: Need to evaluate policies before deployment
- **Long-horizon planning**: Benefits from lookahead
- **Transfer learning**: Model may transfer across related tasks
- **Interpretability**: Model provides insight into environment dynamics

### Use Hybrid (Dyna-style) When:

- **Balanced requirements**: Need both sample efficiency and robustness
- **Uncertain model quality**: Model may be inaccurate in some regions
- **Flexible computation**: Can adjust planning amount dynamically
- **General-purpose**: Good default choice for many domains

## Hybrid Approaches

### Model-Agnostic Meta-Learning (MAML) for RL

MAML learns model parameters that enable fast adaptation:

$$\theta^* = \arg\min_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}(U^k_\theta(\mathcal{T}_i))$$

where $U^k_\theta$ performs $k$ gradient steps on task $\mathcal{T}_i$.

### Probabilistic Ensembles for Trajectory Sampling (PETS)

PETS uses ensemble of probabilistic models for uncertainty-aware planning:

1. Train ensemble: $\{f_\phi^1, \ldots, f_\phi^K\}$
2. For each planning step, sample model from ensemble
3. Use model predictive control (MPC) with uncertainty penalties

### Temporal Difference Models (TDM)

TDMs learn goal-conditioned value functions that can be used for planning:

$$Q(s, a, g, t) = \mathbb{E}[r_t + \gamma Q(s', a', g, t-1) | s, a, g]$$

Planning involves finding actions that minimize distance to goal in learned value space.

## Key Takeaways

1. **Fundamental trade-off**: Model-free methods are simpler and more robust but less sample-efficient. Model-based methods are more sample-efficient but require accurate models and more computation.

2. **Dyna architecture**: Provides a principled way to combine both paradigms, leveraging strengths of each.

3. **Learned models**: Modern approaches learn models optimized for planning rather than accurate prediction, as exemplified by MuZero.

4. **Sample efficiency**: Model-based methods can achieve better sample efficiency, but this depends critically on model quality and planning effectiveness.

5. **Domain considerations**: Choice of approach should consider environment complexity, data availability, computational resources, and safety requirements.

6. **Uncertainty handling**: Model-based methods must account for model uncertainty to avoid overconfident planning.

7. **Hybrid methods**: Combining model-free and model-based components often provides the best balance of sample efficiency, robustness, and performance.

8. **Representation learning**: Modern model-based methods (World Models, MuZero) leverage learned representations to improve planning efficiency.

9. **Planning algorithms**: The choice of planning algorithm (value iteration, policy iteration, MCTS, MPC) interacts with model quality to determine overall performance.

10. **Future directions**: Research continues on improving model learning, uncertainty quantification, and integrating planning with deep RL for better sample efficiency and performance.
