# Actor-Critic Methods

## Table of Contents

1. [Introduction to Actor-Critic Methods](#introduction-to-actor-critic-methods)
2. [Actor-Critic Architecture](#actor-critic-architecture)
3. [Advantage Actor-Critic (A2C)](#advantage-actor-critic-a2c)
4. [Asynchronous Advantage Actor-Critic (A3C)](#asynchronous-advantage-actor-critic-a3c)
5. [Generalized Advantage Estimation (GAE)](#generalized-advantage-estimation-gae)
6. [Asynchronous Training](#asynchronous-training)
7. [Importance Sampling in Actor-Critic](#importance-sampling-in-actor-critic)
8. [V-trace Algorithm](#v-trace-algorithm)
9. [Implementation and Practical Considerations](#implementation-and-practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Actor-Critic Methods

**Actor-Critic Methods** combine the benefits of policy gradient methods (actor) and value function methods (critic). The actor learns the policy, while the critic learns the value function to provide a baseline, reducing variance in policy gradient estimates.

The key idea is to use the value function $V^\pi(s)$ as a baseline:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t\right]$$

where $\hat{A}_t$ is an advantage estimate computed using the value function.

### Advantages

1. **Reduced Variance**: Value function baseline reduces variance
2. **Online Learning**: Can update after each step
3. **Sample Efficiency**: More efficient than pure policy gradients
4. **Stability**: More stable than pure value-based methods

### Components

- **Actor**: Policy $\pi_\theta(a | s)$ that selects actions
- **Critic**: Value function $V_\phi(s)$ that estimates state values

## Actor-Critic Architecture

The actor-critic architecture consists of two networks:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Actor-Critic network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
        """
        super(ActorCritic, self).__init__()
        
        # Shared layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: State tensor
        
        Returns:
            Action probabilities and value estimate
        """
        x = self.shared_layers(state)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value
    
    def act(self, state):
        """
        Sample action from policy
        
        Args:
            state: State tensor
        
        Returns:
            Action, log probability, and value
        """
        action_probs, value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()
```

## Advantage Actor-Critic (A2C)

**Advantage Actor-Critic (A2C)** uses the advantage function $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ to reduce variance.

### Advantage Estimation

The advantage can be estimated using TD error:

$$\hat{A}_t = \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

Or using n-step returns:

$$\hat{A}_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n V(s_{t+n}) - V(s_t)$$

### A2C Algorithm

```python
def a2c_train_step(env, actor_critic, optimizer, gamma=0.99, 
                   n_steps=5, value_coef=0.5, entropy_coef=0.01):
    """
    A2C training step
    
    Args:
        env: Environment
        actor_critic: Actor-Critic network
        optimizer: Optimizer
        gamma: Discount factor
        n_steps: Number of steps for n-step return
        value_coef: Value loss coefficient
        entropy_coef: Entropy regularization coefficient
    """
    # Collect n-step trajectory
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    
    state = env.reset()
    done = False
    
    for step in range(n_steps):
        action, log_prob, value = actor_critic.act(torch.FloatTensor(state))
        next_state, reward, done = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)
        
        state = next_state
        if done:
            break
    
    # Compute returns and advantages
    returns = []
    advantages = []
    
    if done:
        R = 0
    else:
        _, _, R = actor_critic.act(torch.FloatTensor(state))
        R = R.item()
    
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R * (1 - dones[i])
        returns.insert(0, R)
        advantages.insert(0, R - values[i].item())
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    returns_tensor = torch.FloatTensor(returns)
    advantages_tensor = torch.FloatTensor(advantages)
    
    # Normalize advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
        advantages_tensor.std() + 1e-8
    )
    
    # Forward pass
    action_probs, values_pred = actor_critic(states_tensor)
    dist = torch.distributions.Categorical(action_probs)
    
    # Compute losses
    # Policy loss (negative because we maximize)
    log_probs_tensor = dist.log_prob(torch.LongTensor(actions))
    policy_loss = -(log_probs_tensor * advantages_tensor).mean()
    
    # Value loss
    value_loss = F.mse_loss(values_pred.squeeze(), returns_tensor)
    
    # Entropy loss (for exploration)
    entropy_loss = -dist.entropy().mean()
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    # Update
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'total_loss': total_loss.item()
    }
```

### A2C Training Loop

```python
def train_a2c(env, actor_critic, num_episodes=1000, lr=0.0003):
    """
    Train A2C agent
    
    Args:
        env: Environment
        actor_critic: Actor-Critic network
        num_episodes: Number of episodes
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        
        while not done:
            # Collect n-step trajectory
            loss_info = a2c_train_step(env, actor_critic, optimizer)
            
            episode_reward += sum(loss_info.get('rewards', []))
            done = loss_info.get('done', False)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

## Asynchronous Advantage Actor-Critic (A3C)

**A3C (Asynchronous Advantage Actor-Critic)** uses multiple parallel actors to collect experiences asynchronously, improving sample efficiency and exploration.

### Architecture

A3C maintains:
- **Global Network**: Shared actor-critic network
- **Local Networks**: Worker-specific copies for each parallel actor

### Algorithm

```python
import threading
import queue

class A3CWorker(threading.Thread):
    def __init__(self, worker_id, env, global_actor_critic, 
                 optimizer, gamma=0.99, t_max=5):
        """
        A3C Worker thread
        
        Args:
            worker_id: Worker identifier
            env: Environment
            global_actor_critic: Global network
            optimizer: Optimizer for global network
            gamma: Discount factor
            t_max: Maximum steps before update
        """
        super(A3CWorker, self).__init__()
        self.worker_id = worker_id
        self.env = env
        self.global_actor_critic = global_actor_critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.t_max = t_max
        
        # Local network (copy of global)
        self.local_actor_critic = ActorCritic(
            env.observation_space.shape[0],
            env.action_space.n
        )
        self.local_actor_critic.load_state_dict(
            self.global_actor_critic.state_dict()
        )
    
    def run(self):
        """Worker main loop"""
        while True:
            # Synchronize local network with global
            self.local_actor_critic.load_state_dict(
                self.global_actor_critic.state_dict()
            )
            
            # Collect trajectory
            states, actions, rewards, values, log_probs = [], [], [], [], []
            state = self.env.reset()
            done = False
            t = 0
            
            while not done and t < self.t_max:
                action, log_prob, value = self.local_actor_critic.act(
                    torch.FloatTensor(state)
                )
                next_state, reward, done = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                
                state = next_state
                t += 1
            
            # Compute returns
            if done:
                R = 0
            else:
                _, _, R = self.local_actor_critic.act(torch.FloatTensor(state))
                R = R.item()
            
            returns = []
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self.gamma * R
                returns.insert(0, R)
            
            # Compute advantages
            advantages = [r - v.item() for r, v in zip(returns, values)]
            
            # Normalize advantages
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in advantages]
            
            # Compute losses
            states_tensor = torch.FloatTensor(states)
            returns_tensor = torch.FloatTensor(returns)
            advantages_tensor = torch.FloatTensor(advantages)
            
            action_probs, values_pred = self.local_actor_critic(states_tensor)
            dist = torch.distributions.Categorical(action_probs)
            
            policy_loss = -(
                dist.log_prob(torch.LongTensor(actions)) * advantages_tensor
            ).mean()
            value_loss = F.mse_loss(values_pred.squeeze(), returns_tensor)
            entropy_loss = -dist.entropy().mean()
            
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # Update global network
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Copy gradients to global network
            for local_param, global_param in zip(
                self.local_actor_critic.parameters(),
                self.global_actor_critic.parameters()
            ):
                if global_param.grad is not None:
                    global_param.grad += local_param.grad
                else:
                    global_param.grad = local_param.grad.clone()
            
            self.optimizer.step()

def train_a3c(env_factory, num_workers=4, num_episodes=1000):
    """
    Train A3C with multiple workers
    
    Args:
        env_factory: Function that creates environments
        num_workers: Number of parallel workers
        num_episodes: Number of episodes per worker
    """
    # Global network
    global_actor_critic = ActorCritic(
        env_factory().observation_space.shape[0],
        env_factory().action_space.n
    )
    optimizer = torch.optim.Adam(global_actor_critic.parameters(), lr=0.0001)
    
    # Create workers
    workers = []
    for i in range(num_workers):
        worker = A3CWorker(i, env_factory(), global_actor_critic, optimizer)
        workers.append(worker)
        worker.start()
    
    # Wait for workers to finish
    for worker in workers:
        worker.join()
```

### Benefits of A3C

1. **Parallelization**: Multiple workers collect experiences simultaneously
2. **Exploration**: Different workers explore different parts of state space
3. **Stability**: Asynchronous updates provide natural exploration
4. **Efficiency**: Better sample efficiency than single-threaded A2C

## Generalized Advantage Estimation (GAE)

**Generalized Advantage Estimation (GAE)** combines multiple n-step advantage estimates using an exponential weighting scheme.

### GAE Formula

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

### Interpretation

- **$\lambda = 0$**: Uses only TD error (one-step)
- **$\lambda = 1$**: Uses Monte Carlo return (full episode)
- **$0 < \lambda < 1$**: Weighted combination

### Implementation

```python
def compute_gae(rewards, values, dones, gamma=0.99, lambda_param=0.95):
    """
    Compute Generalized Advantage Estimation
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        lambda_param: GAE lambda parameter
    
    Returns:
        Advantages and returns
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * values[t+1] - values[t]
            gae = delta + gamma * lambda_param * gae
        
        advantages.insert(0, gae)
    
    # Compute returns from advantages
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return advantages, returns
```

### Using GAE in Actor-Critic

```python
def a2c_with_gae(env, actor_critic, optimizer, gamma=0.99, lambda_param=0.95):
    """
    A2C with GAE
    
    Args:
        env: Environment
        actor_critic: Actor-Critic network
        optimizer: Optimizer
        gamma: Discount factor
        lambda_param: GAE lambda
    """
    # Collect trajectory
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    
    state = env.reset()
    done = False
    
    while not done:
        action, log_prob, value = actor_critic.act(torch.FloatTensor(state))
        next_state, reward, done = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value.item())
        log_probs.append(log_prob)
        dones.append(done)
        
        state = next_state
    
    # Add final value
    if not done:
        _, _, final_value = actor_critic.act(torch.FloatTensor(state))
        values.append(final_value.item())
    else:
        values.append(0)
    
    # Compute GAE
    advantages, returns = compute_gae(rewards, values, dones, gamma, lambda_param)
    
    # Normalize advantages
    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    returns_tensor = torch.FloatTensor(returns)
    advantages_tensor = torch.FloatTensor(advantages)
    
    # Compute losses and update
    action_probs, values_pred = actor_critic(states_tensor)
    dist = torch.distributions.Categorical(action_probs)
    
    policy_loss = -(
        dist.log_prob(torch.LongTensor(actions)) * advantages_tensor
    ).mean()
    value_loss = F.mse_loss(values_pred.squeeze(), returns_tensor)
    entropy_loss = -dist.entropy().mean()
    
    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
    optimizer.step()
```

## Asynchronous Training

Asynchronous training in A3C provides several benefits:

### Benefits

1. **Parallelization**: Multiple workers collect experiences simultaneously
2. **Diversity**: Different workers explore different trajectories
3. **Stability**: Asynchronous updates provide regularization
4. **Efficiency**: Better utilization of computational resources

### Challenges

1. **Synchronization**: Need to handle concurrent updates
2. **Staleness**: Local networks may be stale
3. **Complexity**: More complex implementation

## Importance Sampling in Actor-Critic

**Importance Sampling** enables off-policy learning in actor-critic methods by correcting for policy mismatch.

### Off-Policy Actor-Critic

For off-policy learning with behavior policy $\beta$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \beta} \left[\frac{\pi_\theta(a | s)}{\beta(a | s)} \nabla_\theta \log \pi_\theta(a | s) \hat{A}^\pi(s, a)\right]$$

### Implementation

```python
def off_policy_actor_critic(behavior_policy, target_policy, 
                           value_network, states, actions, rewards, 
                           next_states, dones, gamma=0.99):
    """
    Off-policy actor-critic update with importance sampling
    
    Args:
        behavior_policy: Policy used to collect data
        target_policy: Policy being learned
        value_network: Value function network
        states: States
        actions: Actions
        rewards: Rewards
        next_states: Next states
        dones: Done flags
        gamma: Discount factor
    """
    # Compute importance weights
    behavior_probs = behavior_policy.get_probs(states, actions)
    target_probs = target_policy.get_probs(states, actions)
    importance_weights = target_probs / (behavior_probs + 1e-8)
    
    # Compute advantages
    values = value_network(states)
    next_values = value_network(next_states)
    advantages = rewards + gamma * next_values * (1 - dones) - values
    
    # Weighted policy gradient
    policy_loss = -(
        target_policy.get_log_probs(states, actions) * 
        advantages * 
        importance_weights
    ).mean()
    
    # Value loss
    returns = rewards + gamma * next_values * (1 - dones)
    value_loss = F.mse_loss(values, returns)
    
    return policy_loss, value_loss
```

## V-trace Algorithm

**V-trace** is an off-policy actor-critic algorithm that uses importance sampling with clipping for stability.

### V-trace Update

The V-trace value update:

$$v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left(\prod_{i=s}^{t-1} c_i\right) \rho_t \delta_t V$$

where:
- $\rho_t = \min(\bar{\rho}, \frac{\pi(a_t | x_t)}{\mu(a_t | x_t)})$ is the clipped importance weight
- $c_i = \min(\bar{c}, \frac{\pi(a_i | x_i)}{\mu(a_i | x_i)})$ is the trace coefficient
- $\delta_t V = r_t + \gamma V(x_{t+1}) - V(x_t)$ is the TD error

### Implementation

```python
def v_trace_update(states, actions, rewards, behavior_probs, 
                  target_probs, value_network, gamma=0.99, 
                  rho_bar=1.0, c_bar=1.0):
    """
    V-trace update
    
    Args:
        states: States
        actions: Actions
        rewards: Rewards
        behavior_probs: Behavior policy probabilities
        target_probs: Target policy probabilities
        value_network: Value network
        gamma: Discount factor
        rho_bar: Clipping threshold for importance weights
        c_bar: Clipping threshold for trace coefficients
    """
    values = value_network(states)
    next_values = value_network(states[1:])
    
    # Compute importance weights
    importance_weights = target_probs / (behavior_probs + 1e-8)
    rho_t = torch.clamp(importance_weights, max=rho_bar)
    c_t = torch.clamp(importance_weights, max=c_bar)
    
    # Compute TD errors
    delta_t = rewards[:-1] + gamma * next_values - values[:-1]
    
    # Compute V-trace targets
    v_trace_targets = []
    v_s = values[-1]
    
    for t in reversed(range(len(delta_t))):
        v_s = values[t] + rho_t[t] * delta_t[t] + gamma * c_t[t] * (v_s - values[t+1])
        v_trace_targets.insert(0, v_s)
    
    # Compute advantages
    advantages = v_trace_targets - values[:-1]
    
    # Policy and value losses
    log_probs = torch.log(target_probs[:-1] + 1e-8)
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values[:-1], torch.stack(v_trace_targets))
    
    return policy_loss, value_loss
```

## Implementation and Practical Considerations

### Hyperparameter Tuning

Key hyperparameters:
- **Learning Rate**: Typically 0.0001 - 0.001
- **GAE Lambda**: 0.9 - 0.99
- **Value Coefficient**: 0.5
- **Entropy Coefficient**: 0.01
- **Gradient Clipping**: 0.5

### Network Architecture

- **Shared Layers**: Common practice to share early layers
- **Separate Heads**: Actor and critic heads typically separate
- **Normalization**: Batch normalization can help

### Training Tips

1. **Normalize Advantages**: Always normalize advantages
2. **Entropy Regularization**: Use entropy bonus for exploration
3. **Gradient Clipping**: Clip gradients for stability
4. **Learning Rate Scheduling**: Decay learning rate over time

## Key Takeaways

1. **Actor-Critic Methods** combine policy gradients (actor) with value functions (critic) to reduce variance and improve sample efficiency.

2. **A2C (Advantage Actor-Critic)** uses advantage estimates computed from value functions to reduce variance in policy gradient updates.

3. **A3C (Asynchronous A2C)** uses multiple parallel workers to collect experiences asynchronously, improving exploration and sample efficiency.

4. **Generalized Advantage Estimation (GAE)** combines multiple n-step advantage estimates using exponential weighting, balancing bias and variance.

5. **Asynchronous Training** in A3C provides natural exploration through parallel workers and improves computational efficiency.

6. **Importance Sampling** enables off-policy learning in actor-critic methods by correcting for policy mismatch between behavior and target policies.

7. **V-trace** is an off-policy actor-critic algorithm that uses clipped importance sampling for stable learning from off-policy data.

8. **Practical Considerations** include hyperparameter tuning, network architecture design, and training techniques like advantage normalization and entropy regularization.

9. **Actor-Critic Methods** provide a good balance between sample efficiency and stability, making them popular in practice.

10. **These methods** form the foundation for advanced algorithms like PPO and TRPO, which add trust region constraints for more stable learning.
