# Policy Gradient Methods

## Table of Contents

1. [Introduction to Policy Gradient Methods](#introduction-to-policy-gradient-methods)
2. [Policy Parameterization](#policy-parameterization)
3. [Policy Gradient Theorem](#policy-gradient-theorem)
4. [REINFORCE Algorithm](#reinforce-algorithm)
5. [Baseline Methods and Variance Reduction](#baseline-methods-and-variance-reduction)
6. [Log-Derivative Trick](#log-derivative-trick)
7. [Variance Reduction Techniques](#variance-reduction-techniques)
8. [Policy Gradient with Function Approximation](#policy-gradient-with-function-approximation)
9. [Advantages and Limitations](#advantages-and-limitations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Policy Gradient Methods

**Policy Gradient Methods** are a class of reinforcement learning algorithms that directly optimize the policy by computing gradients of the expected return with respect to policy parameters. Unlike value-based methods that learn value functions and derive policies, policy gradient methods parameterize the policy directly and optimize it using gradient ascent.

The key idea is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

where $\tau = (s_0, a_0, r_1, s_1, a_1, \ldots)$ is a trajectory and $R(\tau) = \sum_{t=0}^T \gamma^t r_t$ is the return.

### Advantages of Policy Gradients

1. **Direct Optimization**: Optimize policy directly without value functions
2. **Continuous Actions**: Naturally handle continuous action spaces
3. **Stochastic Policies**: Can learn stochastic policies
4. **Convergence**: Guaranteed convergence to at least local optimum

### Disadvantages

1. **High Variance**: Gradient estimates have high variance
2. **Sample Inefficiency**: May require many samples
3. **Local Optima**: May converge to local rather than global optimum
4. **Slow Learning**: Can be slower than value-based methods

## Policy Parameterization

Policies are parameterized as $\pi_\theta(a | s)$, where $\theta$ are the parameters to be learned.

### Discrete Actions

For discrete actions, use a softmax policy:

$$\pi_\theta(a | s) = \frac{e^{h_\theta(s, a)}}{\sum_{a'} e^{h_\theta(s, a')}}$$

where $h_\theta(s, a)$ is a parameterized function (e.g., neural network).

### Continuous Actions

For continuous actions, use a Gaussian policy:

$$\pi_\theta(a | s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))$$

where $\mu_\theta(s)$ and $\sigma_\theta(s)$ are parameterized functions.

### Policy Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False):
        """
        Policy network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            continuous: Whether actions are continuous
        """
        super(PolicyNetwork, self).__init__()
        
        self.continuous = continuous
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        if continuous:
            # Mean and standard deviation for continuous actions
            self.mean_head = nn.Linear(128, action_dim)
            self.std_head = nn.Linear(128, action_dim)
        else:
            # Logits for discrete actions
            self.action_head = nn.Linear(128, action_dim)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: State tensor
        
        Returns:
            Action distribution parameters
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.continuous:
            mean = self.mean_head(x)
            std = F.softplus(self.std_head(x)) + 1e-5  # Ensure positive
            return mean, std
        else:
            logits = self.action_head(x)
            return F.softmax(logits, dim=-1)
    
    def sample(self, state):
        """
        Sample action from policy
        
        Args:
            state: State tensor
        
        Returns:
            Action and log probability
        """
        if self.continuous:
            mean, std = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob
        else:
            probs = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
```

## Policy Gradient Theorem

The **Policy Gradient Theorem** provides the gradient of the expected return with respect to policy parameters.

### Theorem Statement

For a policy $\pi_\theta$, the gradient of the expected return is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau)\right]$$

where $R(\tau) = \sum_{t=0}^T \gamma^t r_t$ is the return.

### Derivation

Starting from the objective:

$$J(\theta) = \int p_\theta(\tau) R(\tau) d\tau$$

where $p_\theta(\tau) = p(s_0) \prod_{t=0}^T \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t)$ is the trajectory distribution.

Taking the gradient:

$$\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau$$

Using the log-derivative trick:

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$$

Substituting:

$$\nabla_\theta J(\theta) = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau$$

$$= \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log p_\theta(\tau) R(\tau)]$$

Expanding $\log p_\theta(\tau)$:

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t | s_t) + \sum_{t=0}^T \log p(s_{t+1} | s_t, a_t)$$

Only the policy terms depend on $\theta$:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t)$$

Therefore:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau)\right]$$

### Monte Carlo Estimate

The gradient can be estimated using Monte Carlo:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)})$$

where $\tau^{(i)}$ are sampled trajectories.

## REINFORCE Algorithm

**REINFORCE** (REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility) is the simplest policy gradient algorithm.

### Algorithm

```python
def reinforce(env, policy_network, num_episodes=1000, gamma=0.99, lr=0.001):
    """
    REINFORCE algorithm
    
    Args:
        env: Environment
        policy_network: Policy network
        num_episodes: Number of episodes
        gamma: Discount factor
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        # Collect trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        state = env.reset()
        done = False
        
        while not done:
            # Sample action
            action, log_prob = policy_network.sample(torch.FloatTensor(state))
            
            # Take action
            next_state, reward, done = env.step(action.item())
            
            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        # Normalize returns (optional)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G  # Negative because we maximize
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### REINFORCE Update Rule

The REINFORCE update:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau)$$

where $\alpha$ is the learning rate.

### Issues with REINFORCE

1. **High Variance**: Uses full return $R(\tau)$, which has high variance
2. **Sample Inefficiency**: Requires many episodes
3. **Slow Convergence**: May take many iterations

## Baseline Methods and Variance Reduction

A **baseline** $b(s)$ is subtracted from the return to reduce variance without introducing bias:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) (R(\tau) - b(s_t))\right]$$

### Unbiasedness

The baseline doesn't introduce bias because:

$$\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) b(s_t)\right] = 0$$

This follows from the policy gradient theorem and the fact that $b(s_t)$ doesn't depend on actions.

### Common Baselines

1. **Constant Baseline**: $b = \mathbb{E}[R(\tau)]$ (mean return)
2. **State-Value Baseline**: $b(s) = V^\pi(s)$
3. **Advantage Function**: $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$

### REINFORCE with Baseline

```python
def reinforce_with_baseline(env, policy_network, value_network, 
                           num_episodes=1000, gamma=0.99, lr=0.001):
    """
    REINFORCE with value function baseline
    
    Args:
        env: Environment
        policy_network: Policy network
        value_network: Value network for baseline
        num_episodes: Number of episodes
        gamma: Discount factor
        lr: Learning rate
    """
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        # Collect trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        state = env.reset()
        done = False
        
        while not done:
            action, log_prob = policy_network.sample(torch.FloatTensor(state))
            next_state, reward, done = env.step(action.item())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        states_tensor = torch.FloatTensor(states)
        
        # Compute baseline (value function)
        values = value_network(states_tensor).squeeze()
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = 0
        for log_prob, adv in zip(log_probs, advantages):
            policy_loss -= log_prob * adv
        
        # Value function loss
        value_loss = F.mse_loss(values, returns)
        
        # Update networks
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
```

## Log-Derivative Trick

The **log-derivative trick** is a fundamental technique used in policy gradients:

$$\nabla_\theta p_\theta(x) = p_\theta(x) \nabla_\theta \log p_\theta(x)$$

### Proof

Starting from:

$$\nabla_\theta \log p_\theta(x) = \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}$$

Multiplying both sides by $p_\theta(x)$:

$$p_\theta(x) \nabla_\theta \log p_\theta(x) = \nabla_\theta p_\theta(x)$$

### Application to Policy Gradients

In policy gradients, we need:

$$\nabla_\theta \mathbb{E}_{a \sim \pi_\theta} [f(a)] = \int \nabla_\theta \pi_\theta(a) f(a) da$$

Using the log-derivative trick:

$$= \int \pi_\theta(a) \nabla_\theta \log \pi_\theta(a) f(a) da$$

$$= \mathbb{E}_{a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a) f(a)]$$

This enables gradient estimation through sampling.

## Variance Reduction Techniques

### Actor-Critic Methods

**Actor-Critic** methods use a value function (critic) to estimate advantages, reducing variance:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t\right]$$

where $\hat{A}_t$ is an advantage estimate (e.g., TD error).

### Generalized Advantage Estimation (GAE)

**GAE** combines multiple n-step advantage estimates:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

### Reward Shaping

**Reward shaping** modifies rewards to reduce variance:

$$r'(s, a, s') = r(s, a, s') + \gamma \phi(s') - \phi(s)$$

where $\phi$ is a potential function. This doesn't change optimal policies but can reduce variance.

### Importance Sampling

**Importance sampling** can reduce variance when using off-policy data:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_{\text{old}}} \left[\frac{\pi_\theta(a | s)}{\pi_{\text{old}}(a | s)} \nabla_\theta \log \pi_\theta(a | s) \hat{A}_t\right]$$

## Policy Gradient with Function Approximation

With function approximation, policies are parameterized by neural networks.

### Deep Policy Networks

```python
class DeepPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Deep policy network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
        """
        super(DeepPolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head
        self.policy_head = nn.Linear(input_dim, action_dim)
    
    def forward(self, state):
        """Forward pass"""
        x = self.shared_layers(state)
        logits = self.policy_head(x)
        return F.softmax(logits, dim=-1)
    
    def get_log_prob(self, state, action):
        """Get log probability of action"""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(action)
```

### Training Deep Policies

```python
def train_deep_policy(env, policy_network, num_episodes=1000, 
                     batch_size=32, gamma=0.99, lr=0.001):
    """
    Train deep policy network
    
    Args:
        env: Environment
        policy_network: Deep policy network
        num_episodes: Number of episodes
        batch_size: Batch size for updates
        gamma: Discount factor
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        # Collect batch of trajectories
        trajectories = []
        
        for _ in range(batch_size):
            trajectory = collect_trajectory(env, policy_network)
            trajectories.append(trajectory)
        
        # Compute policy gradient
        total_loss = 0
        
        for trajectory in trajectories:
            states, actions, rewards = trajectory
            
            # Compute returns
            returns = compute_returns(rewards, gamma)
            
            # Compute log probabilities
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            log_probs = policy_network.get_log_prob(states_tensor, actions_tensor)
            
            # Policy gradient loss
            loss = -(log_probs * torch.FloatTensor(returns)).mean()
            total_loss += loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), 0.5)
        optimizer.step()
```

## Advantages and Limitations

### Advantages

1. **Direct Policy Optimization**: Optimizes policy directly
2. **Continuous Actions**: Handles continuous action spaces naturally
3. **Stochastic Policies**: Can learn stochastic policies
4. **Convergence**: Guaranteed convergence to local optimum
5. **No Value Function**: Doesn't require value function (though baselines help)

### Limitations

1. **High Variance**: Gradient estimates have high variance
2. **Sample Inefficiency**: Requires many samples
3. **Local Optima**: May converge to local rather than global optimum
4. **Slow Learning**: Can be slower than value-based methods
5. **Hyperparameter Sensitivity**: Sensitive to learning rates and other hyperparameters

### When to Use Policy Gradients

- Continuous action spaces
- Stochastic policies needed
- Value function hard to learn
- On-policy learning acceptable

## Key Takeaways

1. **Policy Gradient Methods** directly optimize policies by computing gradients of expected return with respect to policy parameters.

2. **The Policy Gradient Theorem** provides the foundation, expressing the gradient as an expectation over trajectories weighted by returns.

3. **REINFORCE** is the simplest policy gradient algorithm, using Monte Carlo estimates of returns but suffering from high variance.

4. **Baselines** reduce variance without introducing bias by subtracting state-dependent values from returns.

5. **The Log-Derivative Trick** enables gradient estimation through sampling by expressing gradients in terms of log probabilities.

6. **Variance Reduction** techniques include actor-critic methods, GAE, reward shaping, and importance sampling.

7. **Function Approximation** with neural networks enables policy gradients in high-dimensional state spaces.

8. **Policy Gradients** excel in continuous action spaces and when stochastic policies are needed, but suffer from high variance and sample inefficiency.

9. **Actor-Critic Methods** combine policy gradients with value functions to reduce variance and improve sample efficiency.

10. **Policy Gradient Methods** form the foundation for advanced algorithms like PPO, TRPO, and SAC, making them essential in modern reinforcement learning.
