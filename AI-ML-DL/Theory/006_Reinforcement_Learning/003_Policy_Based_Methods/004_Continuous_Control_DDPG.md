# Continuous Control with Deep Deterministic Policy Gradient

## Table of Contents

1. [Introduction](#introduction)
2. [Continuous Action Spaces](#continuous-action-spaces)
3. [Deep Deterministic Policy Gradient (DDPG)](#deep-deterministic-policy-gradient-ddpg)
4. [Twin Delayed DDPG (TD3)](#twin-delayed-ddpg-td3)
5. [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
6. [Exploration in Continuous Spaces](#exploration-in-continuous-spaces)
7. [Ornstein-Uhlenbeck Process](#ornstein-uhlenbeck-process)
8. [Implementation Considerations](#implementation-considerations)
9. [Comparison of Algorithms](#comparison-of-algorithms)
10. [Key Takeaways](#key-takeaways)

## Introduction

Continuous control tasks, where actions are real-valued vectors rather than discrete choices, present unique challenges for reinforcement learning. Traditional Q-learning and policy gradient methods designed for discrete actions require significant modifications to handle continuous action spaces effectively.

Deep Deterministic Policy Gradient (DDPG) addresses this challenge by combining:
- **Actor-critic architecture**: Separate networks for policy (actor) and value function (critic)
- **Deterministic policy**: Outputs continuous actions directly
- **Off-policy learning**: Uses experience replay for sample efficiency
- **Target networks**: Stabilizes learning through delayed updates

This chapter covers DDPG and its improvements: Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC), which address limitations in the original algorithm.

## Continuous Action Spaces

### Challenges

**Discrete vs Continuous:**
- Discrete: $a \in \{a_1, a_2, \ldots, a_n\}$ - finite set of actions
- Continuous: $a \in \mathbb{R}^d$ or $a \in [a_{\min}, a_{\max}]^d$ - infinite action space

**Key difficulties:**
1. **Argmax intractable**: Cannot compute $\arg\max_a Q(s, a)$ exactly
2. **Policy gradient variance**: Continuous spaces can lead to high variance in gradient estimates
3. **Exploration**: Need effective exploration strategies for continuous domains
4. **Action constraints**: Must respect physical limits and safety constraints

### Approaches to Continuous Control

**Discretization:**
- Divide continuous space into bins: $a \in \{a_1, a_2, \ldots, a_n\}$
- Problems: Curse of dimensionality, loss of precision

**Parameterized policies:**
- Gaussian policy: $\pi(a | s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$
- Beta policy: For bounded actions, use Beta distribution
- Deterministic policy: $a = \mu_\theta(s)$ (used in DDPG)

**Normalized advantage functions (NAF):**
- Restrict Q-function form to enable exact maximization
- Limited expressiveness

## Deep Deterministic Policy Gradient (DDPG)

### Algorithm Overview

DDPG is an actor-critic algorithm that learns:
- **Actor**: Deterministic policy $\mu_\theta(s)$ mapping states to actions
- **Critic**: Q-function $Q_\phi(s, a)$ estimating action values

### Deterministic Policy Gradient Theorem

For deterministic policy $\mu_\theta(s)$, the policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu} \left[ \nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a) \big|_{a=\mu_\theta(s)} \right]$$

This is simpler than stochastic policy gradients because:
- No expectation over actions
- Gradient flows directly through policy to Q-function
- Lower variance estimates

### DDPG Algorithm

```
Initialize:
    Actor network: μ_θ(s) with parameters θ
    Critic network: Q_φ(s, a) with parameters φ
    Target networks: μ_θ'(s), Q_φ'(s, a) with θ' ← θ, φ' ← φ
    Replay buffer: D

for episode = 1 to M:
    Initialize noise process N (e.g., OU process)
    s_0 = initial state
    
    for t = 0 to T:
        # Select action with exploration noise
        a_t = μ_θ(s_t) + N_t
        
        # Execute action, observe reward and next state
        Execute a_t, observe r_t, s_{t+1}
        Store (s_t, a_t, r_t, s_{t+1}) in D
        
        # Sample minibatch from replay buffer
        Sample {(s_i, a_i, r_i, s'_i)} from D
        
        # Update critic
        y_i = r_i + γ Q_φ'(s'_i, μ_θ'(s'_i))
        L_critic = (1/N) Σ_i (Q_φ(s_i, a_i) - y_i)^2
        φ ← φ - α_critic ∇_φ L_critic
        
        # Update actor
        L_actor = -(1/N) Σ_i Q_φ(s_i, μ_θ(s_i))
        θ ← θ - α_actor ∇_θ L_actor
        
        # Soft update target networks
        θ' ← τθ + (1-τ)θ'
        φ' ← τφ + (1-τ)φ'
```

### Key Components

**Experience Replay:**
- Stores transitions $(s_t, a_t, r_t, s_{t+1})$ in buffer
- Samples random minibatches to break correlation
- Enables off-policy learning

**Target Networks:**
- Separate networks $Q_{\phi'}$ and $\mu_{\theta'}$ updated slowly
- Target: $y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$
- Prevents instability from learning moving targets
- Soft update: $\phi' \leftarrow \tau \phi + (1-\tau) \phi'$ with $\tau \ll 1$

**Exploration:**
- Add noise to deterministic policy: $a = \mu_\theta(s) + \mathcal{N}$
- Common choices: Gaussian noise, Ornstein-Uhlenbeck process

### Loss Functions

**Critic Loss:**
$$L_{\text{critic}} = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( Q_\phi(s, a) - \left( r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s')) \right) \right)^2 \right]$$

**Actor Loss:**
$$L_{\text{actor}} = -\mathbb{E}_{s \sim D} \left[ Q_\phi(s, \mu_\theta(s)) \right]$$

The actor maximizes expected Q-value by following policy gradient.

### Limitations of DDPG

1. **Overestimation bias**: Q-function tends to overestimate values
2. **Hyperparameter sensitivity**: Sensitive to learning rates, noise parameters
3. **Sample efficiency**: May require many samples for complex tasks
4. **Exploration**: Simple noise injection may be insufficient

## Twin Delayed DDPG (TD3)

TD3 addresses overestimation bias and instability in DDPG through three key improvements.

### Overestimation Bias Problem

In DDPG, the target is:
$$y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$$

If $Q_{\phi'}$ overestimates, this bias propagates and compounds. TD3 mitigates this through:

### Key Improvements

**1. Twin Critic Networks:**
Maintain two Q-networks $Q_{\phi_1}$ and $Q_{\phi_2}$, take minimum for target:

$$y = r + \gamma \min_{i=1,2} Q_{\phi'_i}(s', \mu_{\theta'}(s'))$$

This reduces overestimation bias.

**2. Delayed Policy Updates:**
Update actor less frequently than critic (e.g., every 2 critic updates). This allows critic to be more accurate before policy changes.

**3. Target Policy Smoothing:**
Add small noise to target actions to regularize Q-function:

$$\tilde{a}' = \mu_{\theta'}(s') + \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$
$$y = r + \gamma Q_{\phi'}(s', \tilde{a}')$$

Prevents Q-function from overfitting to sharp peaks.

### TD3 Algorithm

```
Initialize:
    Twin critics: Q_φ₁(s, a), Q_φ₂(s, a)
    Actor: μ_θ(s)
    Target networks: Q_φ'₁, Q_φ'₂, μ_θ'
    Replay buffer: D

for episode = 1 to M:
    for t = 0 to T:
        a_t = μ_θ(s_t) + clip(N(0, σ), -c, c)
        Execute a_t, observe r_t, s_{t+1}
        Store (s_t, a_t, r_t, s_{t+1}) in D
        
        Sample minibatch {(s_i, a_i, r_i, s'_i)} from D
        
        # Update critics
        ã'_i = μ_θ'(s'_i) + clip(N(0, σ_target), -c_target, c_target)
        y_i = r_i + γ min(Q_φ'₁(s'_i, ã'_i), Q_φ'₂(s'_i, ã'_i))
        
        L_1 = (1/N) Σ_i (Q_φ₁(s_i, a_i) - y_i)^2
        L_2 = (1/N) Σ_i (Q_φ₂(s_i, a_i) - y_i)^2
        
        φ₁ ← φ₁ - α_critic ∇_φ₁ L_1
        φ₂ ← φ₂ - α_critic ∇_φ₂ L_2
        
        # Delayed actor update
        if t mod d:
            L_actor = -(1/N) Σ_i Q_φ₁(s_i, μ_θ(s_i))
            θ ← θ - α_actor ∇_θ L_actor
            
            # Soft update targets
            θ' ← τθ + (1-τ)θ'
            φ'₁ ← τφ₁ + (1-τ)φ'₁
            φ'₂ ← τφ₂ + (1-τ)φ'₂
```

### Performance Improvements

TD3 typically achieves:
- More stable learning
- Better final performance
- Reduced hyperparameter sensitivity
- Faster convergence in many domains

## Soft Actor-Critic (SAC)

SAC combines the benefits of actor-critic methods with maximum entropy reinforcement learning, encouraging exploration through entropy regularization.

### Maximum Entropy RL

Standard RL maximizes expected return:
$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t r(s_t, a_t) \right]$$

Maximum entropy RL adds entropy bonus:
$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$

where $\mathcal{H}(\pi(\cdot | s)) = -\mathbb{E}_{a \sim \pi} [\log \pi(a | s)]$ is policy entropy and $\alpha$ is temperature parameter.

**Benefits:**
- Encourages exploration
- More robust policies
- Better sample efficiency
- Natural handling of multiple near-optimal solutions

### SAC Algorithm

SAC uses:
- **Stochastic actor**: $\pi_\theta(a | s)$ (typically Gaussian)
- **Twin Q-networks**: $Q_{\phi_1}(s, a)$, $Q_{\phi_2}(s, a)$
- **Value function**: $V_\psi(s)$ (optional, can be derived from Q)

### Soft Policy Iteration

**Soft Policy Evaluation:**
$$Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p} \left[ V^\pi(s') \right]$$

where soft value function is:
$$V^\pi(s) = \mathbb{E}_{a \sim \pi} \left[ Q^\pi(s, a) - \alpha \log \pi(a | s) \right]$$

**Soft Policy Improvement:**
$$\pi_{\text{new}} = \arg\min_{\pi'} D_{KL} \left( \pi'(\cdot | s) \middle\| \frac{\exp(Q^{\pi_{\text{old}}}(s, \cdot) / \alpha)}{Z(s)} \right)$$

Solution is:
$$\pi_{\text{new}}(a | s) = \frac{\exp(Q^{\pi_{\text{old}}}(s, a) / \alpha)}{Z(s)}$$

### Practical SAC Implementation

```
Initialize:
    Actor: π_θ(a | s) (Gaussian with tanh squashing)
    Twin Q-networks: Q_φ₁(s, a), Q_φ₂(s, a)
    Target Q-networks: Q_φ'₁, Q_φ'₂
    Replay buffer: D
    Temperature: α (can be learned)

for episode = 1 to M:
    for t = 0 to T:
        # Sample action from stochastic policy
        a_t ~ π_θ(· | s_t)
        Execute a_t, observe r_t, s_{t+1}
        Store (s_t, a_t, r_t, s_{t+1}, done) in D
        
        if |D| > batch_size:
            Sample minibatch {(s_i, a_i, r_i, s'_i, d_i)} from D
            
            # Compute target Q-values
            ã'_i ~ π_θ(· | s'_i)
            Q_target = r_i + γ(1-d_i) min(Q_φ'₁(s'_i, ã'_i), 
                                          Q_φ'₂(s'_i, ã'_i)) 
                          - α log π_θ(ã'_i | s'_i)
            
            # Update Q-networks
            L_Q₁ = (1/N) Σ_i (Q_φ₁(s_i, a_i) - Q_target)^2
            L_Q₂ = (1/N) Σ_i (Q_φ₂(s_i, a_i) - Q_target)^2
            
            φ₁ ← φ₁ - α_Q ∇_φ₁ L_Q₁
            φ₂ ← φ₂ - α_Q ∇_φ₂ L_Q₂
            
            # Update policy
            ã_i ~ π_θ(· | s_i)  # Re-sample for gradient
            Q_π = min(Q_φ₁(s_i, ã_i), Q_φ₂(s_i, ã_i))
            L_π = (1/N) Σ_i [α log π_θ(ã_i | s_i) - Q_π]
            θ ← θ - α_π ∇_θ L_π
            
            # Update temperature (if learned)
            L_α = (1/N) Σ_i [-α (log π_θ(ã_i | s_i) + H_target)]
            α ← α - α_α ∇_α L_α
            
            # Soft update targets
            φ'₁ ← τφ₁ + (1-τ)φ'₁
            φ'₂ ← τφ₂ + (1-τ)φ'₂
```

### Key Features

**Stochastic Policy:**
- Typically Gaussian: $a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$
- Actions squashed to $[-1, 1]$ via $\tanh$
- Entropy naturally encourages exploration

**Automatic Temperature Tuning:**
- Can learn $\alpha$ to maintain target entropy $\bar{\mathcal{H}}$
- Prevents collapse to deterministic policy
- Adapts exploration automatically

**Clipped Double Q-Learning:**
- Uses minimum of twin Q-networks
- Reduces overestimation bias

## Exploration in Continuous Spaces

### Challenges

1. **Infinite action space**: Cannot enumerate all actions
2. **Smoothness**: Nearby actions may have similar values
3. **Curse of dimensionality**: High-dimensional action spaces

### Exploration Strategies

**Gaussian Noise:**
$$a = \mu_\theta(s) + \mathcal{N}(0, \sigma^2 I)$$

Simple but may not be optimal for correlated action dimensions.

**Ornstein-Uhlenbeck Process:**
Provides temporally correlated noise, better for physical systems with inertia.

**Parameter Space Noise:**
Add noise to policy parameters rather than actions:
$$\theta_{\text{noisy}} = \theta + \mathcal{N}(0, \sigma^2 I)$$

**Intrinsic Motivation:**
- Curiosity-driven exploration
- Count-based bonuses
- Prediction error bonuses

## Ornstein-Uhlenbeck Process

The Ornstein-Uhlenbeck (OU) process generates temporally correlated noise suitable for continuous control.

### Definition

The OU process is defined by the stochastic differential equation:

$$dx_t = \theta(\mu - x_t)dt + \sigma dW_t$$

where:
- $\theta > 0$: Mean reversion strength
- $\mu$: Long-term mean
- $\sigma > 0$: Volatility
- $dW_t$: Wiener process

### Discrete-Time Approximation

For use in RL, we use discrete approximation:

$$x_{t+1} = x_t + \theta(\mu - x_t)\Delta t + \sigma \sqrt{\Delta t} \mathcal{N}(0, 1)$$

Setting $\Delta t = 1$ and $\mu = 0$:

$$x_{t+1} = x_t + \theta(-x_t) + \sigma \mathcal{N}(0, 1) = (1-\theta)x_t + \sigma \mathcal{N}(0, 1)$$

### Properties

**Mean reversion:**
- Process tends toward $\mu$ over time
- Creates smooth, correlated noise

**Variance:**
- Stationary variance: $\text{Var}(x_\infty) = \frac{\sigma^2}{2\theta}$

**Autocorrelation:**
- $\text{Corr}(x_t, x_{t+k}) = e^{-\theta k}$

### Use in DDPG

```python
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
    
    def reset(self):
        self.state = np.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state
```

**Advantages:**
- Smooth exploration suitable for physical systems
- Better than independent Gaussian noise for correlated actions
- Natural for systems with momentum

**Disadvantages:**
- Additional hyperparameters ($\theta$, $\sigma$)
- May be unnecessary for some domains
- SAC's stochastic policy often eliminates need for explicit noise

## Implementation Considerations

### Network Architecture

**Actor Network:**
```
Input: state s
Hidden: [400, 300] (or similar)
Output: action mean μ(s)
        (optional) action std σ(s) for stochastic policies
```

**Critic Network:**
```
Input: state s, action a
Hidden: [400, 300]
Output: Q-value Q(s, a)
```

**Design choices:**
- Batch normalization can help but may not be necessary
- Layer normalization often works better
- Residual connections for deep networks

### Hyperparameters

**Learning rates:**
- Actor: Typically $10^{-4}$ to $10^{-3}$
- Critic: Typically $10^{-3}$ to $10^{-2}$
- Critic usually learns faster than actor

**Target network update:**
- $\tau = 0.001$ to $0.01$ (soft update)
- Or update every $C$ steps (hard update)

**Replay buffer:**
- Size: $10^5$ to $10^6$ transitions
- Batch size: 64 to 256

**Exploration:**
- Initial noise: High for exploration
- Decay noise: Gradually reduce over training
- OU process: $\theta = 0.15$, $\sigma = 0.2$ (domain-dependent)

### Action Normalization

For bounded action spaces $a \in [a_{\min}, a_{\max}]$:

**Tanh squashing:**
$$a = a_{\min} + \frac{a_{\max} - a_{\min}}{2} (1 + \tanh(\mu_\theta(s)))$$

**Clipping:**
$$a = \text{clip}(\mu_\theta(s), a_{\min}, a_{\max})$$

Tanh is preferred as it provides smooth gradients.

### Gradient Clipping

Prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
```

## Comparison of Algorithms

| Algorithm | Policy Type | Exploration | Key Features | Best For |
|-----------|-------------|-------------|--------------|----------|
| DDPG | Deterministic | Noise injection | Simple, baseline | Simple tasks |
| TD3 | Deterministic | Noise injection | Twin critics, delayed updates | Stable learning |
| SAC | Stochastic | Entropy bonus | Maximum entropy, robust | Complex tasks, sample efficiency |

### Performance Characteristics

**Sample Efficiency:**
- SAC typically most sample-efficient
- TD3 better than DDPG
- All benefit from experience replay

**Stability:**
- TD3 most stable (addresses overestimation)
- SAC stable with proper temperature tuning
- DDPG can be unstable

**Exploration:**
- SAC: Natural exploration via entropy
- TD3/DDPG: Require careful noise tuning

**Hyperparameter Sensitivity:**
- SAC: Moderate (temperature can be learned)
- TD3: Low (more robust than DDPG)
- DDPG: High

## Key Takeaways

1. **Continuous control requires specialized algorithms**: Standard discrete-action methods don't directly apply.

2. **DDPG provides foundation**: Actor-critic with deterministic policies and experience replay enables continuous control.

3. **TD3 improves stability**: Twin critics and delayed updates address overestimation bias and improve learning stability.

4. **SAC enables robust learning**: Maximum entropy framework provides natural exploration and better sample efficiency.

5. **Exploration is critical**: Continuous spaces require effective exploration strategies (noise injection, entropy regularization).

6. **Target networks stabilize learning**: Slow updates prevent instability from learning moving targets.

7. **Experience replay essential**: Off-policy learning with replay buffers improves sample efficiency.

8. **Stochastic vs deterministic**: Stochastic policies (SAC) provide better exploration; deterministic (DDPG/TD3) may be simpler.

9. **Hyperparameter tuning**: Learning rates, noise parameters, and update frequencies significantly affect performance.

10. **Algorithm choice**: Select based on task complexity, sample efficiency requirements, and stability needs.
