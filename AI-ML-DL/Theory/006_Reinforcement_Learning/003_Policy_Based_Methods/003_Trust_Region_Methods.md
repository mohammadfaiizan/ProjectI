# Trust Region Methods

## Table of Contents

1. [Introduction to Trust Region Methods](#introduction-to-trust-region-methods)
2. [Natural Policy Gradient](#natural-policy-gradient)
3. [Trust Region Policy Optimization (TRPO)](#trust-region-policy-optimization-trpo)
4. [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
5. [KL Divergence Constraint](#kl-divergence-constraint)
6. [Practical Implementation](#practical-implementation)
7. [Clipped PPO](#clipped-ppo)
8. [Adaptive KL Penalty PPO](#adaptive-kl-penalty-ppo)
9. [Comparison and Trade-offs](#comparison-and-trade-offs)
10. [Key Takeaways](#key-takeaways)

## Introduction to Trust Region Methods

**Trust Region Methods** constrain policy updates to stay within a "trust region" where the policy approximation is accurate. This prevents large, destructive updates that can occur in standard policy gradient methods.

The key idea is to ensure that the new policy $\pi_{\theta_{\text{new}}}$ is not too different from the old policy $\pi_{\theta_{\text{old}}}$, measured by a distance metric (typically KL divergence).

### Motivation

Standard policy gradient methods can suffer from:
1. **Large Updates**: May take steps that degrade performance
2. **Instability**: Policy can change drastically between updates
3. **Sample Inefficiency**: Need to collect new data after each update

Trust region methods address these by:
1. **Constraining Updates**: Limit how much the policy can change
2. **Stability**: Ensure monotonic improvement
3. **Sample Efficiency**: Can reuse data for multiple updates

## Natural Policy Gradient

The **Natural Policy Gradient** uses the Fisher information matrix to define a natural metric in policy space, leading to more stable updates.

### Fisher Information Matrix

The **Fisher Information Matrix** for a policy $\pi_\theta$ is:

$$F(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a | s) \nabla_\theta \log \pi_\theta(a | s)^\top]$$

This matrix captures the curvature of the policy distribution.

### Natural Gradient

The **natural gradient** is:

$$\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta)$$

This accounts for the geometry of the policy space, leading to more stable updates.

### Natural Policy Gradient Update

$$\theta_{\text{new}} = \theta_{\text{old}} + \alpha F(\theta)^{-1} \nabla_\theta J(\theta)$$

where $\alpha$ is the step size.

### Approximation

Computing $F(\theta)^{-1}$ is expensive. Instead, we can solve:

$$F(\theta) \Delta \theta = \nabla_\theta J(\theta)$$

using conjugate gradient methods.

## Trust Region Policy Optimization (TRPO)

**TRPO** constrains policy updates using KL divergence, ensuring monotonic improvement.

### Objective

TRPO maximizes:

$$\max_\theta \mathbb{E}_{s \sim \rho^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[\frac{\pi_\theta(a | s)}{\pi_{\text{old}}(a | s)} \hat{A}^{\pi_{\text{old}}}(s, a)\right]$$

subject to:

$$\mathbb{E}_{s \sim \rho^{\pi_{\text{old}}}} [D_{KL}(\pi_{\text{old}}(\cdot | s) || \pi_\theta(\cdot | s))] \leq \delta$$

where $\delta$ is the trust region size (typically 0.01).

### Surrogate Objective

The surrogate objective (first-order approximation):

$$L(\theta) = \mathbb{E}_{s \sim \rho^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}} \left[\frac{\pi_\theta(a | s)}{\pi_{\text{old}}(a | s)} \hat{A}^{\pi_{\text{old}}}(s, a)\right]$$

### KL Constraint

The KL divergence constraint:

$$\bar{D}_{KL}(\theta_{\text{old}} || \theta) = \mathbb{E}_{s \sim \rho^{\pi_{\text{old}}}} [D_{KL}(\pi_{\theta_{\text{old}}}(\cdot | s) || \pi_\theta(\cdot | s))] \leq \delta$$

### TRPO Algorithm

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

def trpo_update(policy_network, states, actions, advantages, 
                old_log_probs, max_kl=0.01, damping=0.1):
    """
    TRPO update
    
    Args:
        policy_network: Policy network
        states: States
        actions: Actions
        advantages: Advantage estimates
        old_log_probs: Old policy log probabilities
        max_kl: Maximum KL divergence
        damping: Damping coefficient for Fisher matrix
    """
    # Get current policy log probabilities
    action_probs = policy_network(states)
    dist = torch.distributions.Categorical(action_probs)
    log_probs = dist.log_prob(actions)
    
    # Compute importance weights
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Surrogate objective
    surrogate_loss = -(ratio * advantages).mean()
    
    # Compute gradient
    grads = torch.autograd.grad(
        surrogate_loss, policy_network.parameters(), 
        create_graph=True, retain_graph=True
    )
    flat_grad = torch.cat([g.view(-1) for g in grads])
    
    # Compute Fisher-vector product
    def fisher_vector_product(v):
        """Compute F * v where F is Fisher information matrix"""
        kl = compute_kl(policy_network, states, old_log_probs)
        grads = torch.autograd.grad(kl, policy_network.parameters(), 
                                   create_graph=True, retain_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads])
        
        grad_v = (flat_grads * v).sum()
        grad_grads = torch.autograd.grad(grad_v, policy_network.parameters(), 
                                        retain_graph=True)
        flat_grad_grads = torch.cat([g.contiguous().view(-1) 
                                    for g in grad_grads])
        return flat_grad_grads + damping * v
    
    # Solve for natural gradient using conjugate gradient
    natural_grad = conjugate_gradient(fisher_vector_product, flat_grad)
    
    # Compute step size
    step_size = torch.sqrt(2 * max_kl / (
        natural_grad @ fisher_vector_product(natural_grad) + 1e-8
    ))
    
    # Update parameters
    old_params = [p.clone() for p in policy_network.parameters()]
    
    with torch.no_grad():
        for param, ng in zip(policy_network.parameters(), 
                            natural_grad.view(-1, param.numel())):
            param += step_size * ng.view(param.shape)
    
    # Check KL constraint
    new_log_probs = dist.log_prob(actions)
    kl = compute_kl(policy_network, states, old_log_probs)
    
    if kl > max_kl:
        # Revert update if constraint violated
        for param, old_param in zip(policy_network.parameters(), old_params):
            param.data = old_param.data

def compute_kl(policy_network, states, old_log_probs):
    """Compute KL divergence"""
    action_probs = policy_network(states)
    dist = torch.distributions.Categorical(action_probs)
    log_probs = dist.log_prob(actions)
    
    kl = (old_log_probs - log_probs).mean()
    return kl

def conjugate_gradient(A, b, max_iter=10, tol=1e-10):
    """Conjugate gradient method"""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    
    for i in range(max_iter):
        Ap = A(p)
        alpha = (r @ r) / (p @ Ap + 1e-8)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        if torch.norm(r_new) < tol:
            break
        
        beta = (r_new @ r_new) / (r @ r + 1e-8)
        p = r_new + beta * p
        r = r_new
    
    return x
```

## Proximal Policy Optimization (PPO)

**PPO** is a simpler alternative to TRPO that uses clipping or adaptive KL penalty to constrain updates, avoiding the computational cost of conjugate gradient.

### Clipped Objective

PPO uses a clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the importance weight and $\epsilon$ is the clipping parameter (typically 0.2).

### Clipping Mechanism

The clipping prevents the policy from changing too much:
- If advantage is positive: don't increase ratio beyond $1+\epsilon$
- If advantage is negative: don't decrease ratio below $1-\epsilon$

## KL Divergence Constraint

The KL divergence measures the difference between two probability distributions:

$$D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

For policies:

$$D_{KL}(\pi_{\text{old}} || \pi_\theta) = \mathbb{E}_{s \sim \rho^{\pi_{\text{old}}}} \left[\sum_a \pi_{\text{old}}(a | s) \log \frac{\pi_{\text{old}}(a | s)}{\pi_\theta(a | s)}\right]$$

### Properties

1. **Non-negative**: $D_{KL}(P || Q) \geq 0$
2. **Zero when equal**: $D_{KL}(P || Q) = 0$ iff $P = Q$
3. **Asymmetric**: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$

### Computing KL Divergence

```python
def compute_kl_divergence(old_probs, new_probs):
    """
    Compute KL divergence between two policies
    
    Args:
        old_probs: Old policy probabilities
        new_probs: New policy probabilities
    
    Returns:
        KL divergence
    """
    kl = (old_probs * torch.log(old_probs / (new_probs + 1e-8))).sum(dim=-1)
    return kl.mean()
```

## Practical Implementation

### PPO with Clipping

```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, 
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        """
        PPO Agent
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lr: Learning rate
            clip_epsilon: Clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy regularization coefficient
        """
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters()),
            lr=lr
        )
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        """
        PPO update
        
        Args:
            states: States
            actions: Actions
            old_log_probs: Old policy log probabilities
            advantages: Advantage estimates
            returns: Returns
        """
        # Get current policy
        action_probs = self.policy_network(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                           1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss - 
                     self.entropy_coef * entropy)
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters()),
            0.5
        )
        self.optimizer.step()
        
        # Compute KL for monitoring
        with torch.no_grad():
            kl = (old_log_probs - log_probs).mean()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl': kl.item()
        }
```

## Clipped PPO

The clipped PPO objective prevents large policy updates:

```python
def clipped_ppo_loss(ratio, advantages, clip_epsilon=0.2):
    """
    Clipped PPO loss
    
    Args:
        ratio: Importance weight ratio
        advantages: Advantage estimates
        clip_epsilon: Clipping parameter
    
    Returns:
        Policy loss
    """
    # Unclipped objective
    surr1 = ratio * advantages
    
    # Clipped objective
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    
    # Take minimum (pessimistic)
    loss = -torch.min(surr1, surr2).mean()
    
    return loss
```

### Why Clipping Works

Clipping ensures that:
1. **Positive Advantages**: Policy doesn't increase too much
2. **Negative Advantages**: Policy doesn't decrease too much
3. **Stability**: Updates are bounded

## Adaptive KL Penalty PPO

**Adaptive KL Penalty PPO** uses a penalty instead of a hard constraint:

$$L^{\text{KLPEN}}(\theta) = \mathbb{E}_t \left[r_t(\theta) \hat{A}_t - \beta D_{KL}(\pi_{\theta_{\text{old}}} || \pi_\theta)\right]$$

The penalty coefficient $\beta$ is adapted based on the KL divergence:

```python
def adaptive_kl_ppo_loss(ratio, advantages, kl, target_kl=0.01, 
                         beta=1.0, beta_update_rate=1.5):
    """
    Adaptive KL penalty PPO loss
    
    Args:
        ratio: Importance weight ratio
        advantages: Advantage estimates
        kl: KL divergence
        target_kl: Target KL divergence
        beta: Current penalty coefficient
        beta_update_rate: Rate to update beta
    
    Returns:
        Policy loss and updated beta
    """
    # Policy loss with KL penalty
    policy_loss = -(ratio * advantages - beta * kl).mean()
    
    # Adapt beta
    if kl > target_kl * 1.5:
        beta *= beta_update_rate
    elif kl < target_kl / 1.5:
        beta /= beta_update_rate
    
    return policy_loss, beta
```

## Comparison and Trade-offs

### TRPO vs PPO

| Aspect | TRPO | PPO |
|-------|------|-----|
| **Constraint** | Hard KL constraint | Soft clipping/penalty |
| **Computation** | Expensive (conjugate gradient) | Cheap (simple clipping) |
| **Stability** | Very stable | Stable |
| **Implementation** | Complex | Simple |
| **Performance** | Slightly better | Slightly worse |

### When to Use Each

- **TRPO**: When stability is critical, computational cost acceptable
- **PPO**: When simplicity and speed are important, good default choice

## Key Takeaways

1. **Trust Region Methods** constrain policy updates to stay within a trust region, preventing large destructive updates.

2. **Natural Policy Gradient** uses the Fisher information matrix to define a natural metric in policy space, leading to more stable updates.

3. **TRPO** uses a hard KL divergence constraint to ensure monotonic improvement, but requires expensive conjugate gradient computation.

4. **PPO** is a simpler alternative that uses clipping or adaptive KL penalty, avoiding the computational cost of TRPO while maintaining stability.

5. **KL Divergence** measures the difference between policies and is used to constrain updates in trust region methods.

6. **Clipped PPO** prevents large policy updates by clipping the importance weight ratio, ensuring updates stay within a trust region.

7. **Adaptive KL Penalty PPO** uses a penalty term that adapts based on the KL divergence, providing a softer constraint than hard clipping.

8. **Practical Implementation** involves careful hyperparameter tuning, including clipping parameter, learning rates, and value/entropy coefficients.

9. **TRPO vs PPO** trade-off involves computational cost (TRPO) vs simplicity (PPO), with PPO being the more popular choice in practice.

10. **Trust Region Methods** form the foundation for stable policy optimization and are widely used in modern reinforcement learning applications.
