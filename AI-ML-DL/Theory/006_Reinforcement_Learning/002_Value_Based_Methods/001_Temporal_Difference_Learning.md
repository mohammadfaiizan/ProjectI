# Temporal Difference Learning

## Table of Contents

1. [Introduction to Temporal Difference Learning](#introduction-to-temporal-difference-learning)
2. [TD(0) Algorithm](#td0-algorithm)
3. [TD(λ) and Eligibility Traces](#tdλ-and-eligibility-traces)
4. [On-Policy vs Off-Policy Methods](#on-policy-vs-off-policy-methods)
5. [N-Step TD Methods](#n-step-td-methods)
6. [Monte Carlo vs Temporal Difference](#monte-carlo-vs-temporal-difference)
7. [Convergence and Stability](#convergence-and-stability)
8. [Function Approximation with TD](#function-approximation-with-td)
9. [Bias-Variance Trade-off in TD Learning](#bias-variance-trade-off-in-td-learning)
10. [Key Takeaways](#key-takeaways)

## Introduction to Temporal Difference Learning

**Temporal Difference (TD) Learning** is a class of model-free reinforcement learning algorithms that learn value functions by bootstrapping from estimates of future values. Unlike Monte Carlo methods that wait until the end of an episode, TD methods update estimates based on other learned estimates, enabling online learning and faster convergence.

The key insight of TD learning is the **temporal difference error**:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

This error measures the difference between the current estimate $V(S_t)$ and the better estimate $R_{t+1} + \gamma V(S_{t+1})$ based on the observed reward and next state value.

### Advantages of TD Learning

1. **Online Learning**: Updates after each step, not just at episode end
2. **Faster Convergence**: Bootstrapping accelerates learning
3. **Lower Variance**: Uses learned estimates instead of full returns
4. **Works in Continuing Tasks**: Doesn't require episodic termination

### Disadvantages of TD Learning

1. **Bias**: Bootstrapping introduces bias if value estimates are inaccurate
2. **Sensitivity**: More sensitive to initialization and learning rates
3. **Off-Policy Challenges**: Can diverge with function approximation

## TD(0) Algorithm

**TD(0)** is the simplest temporal difference algorithm, updating the value function after each step using the one-step return.

### Update Rule

The TD(0) update rule is:

$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

where:
- $\alpha$ is the learning rate
- $\gamma$ is the discount factor
- $R_{t+1} + \gamma V(S_{t+1})$ is the **TD target**
- $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error** $\delta_t$

### Algorithm Description

```python
def td0(env, policy, alpha=0.1, gamma=0.9, num_episodes=1000):
    """
    TD(0) algorithm for policy evaluation
    
    Args:
        env: Environment
        policy: Policy to evaluate
        alpha: Learning rate
        gamma: Discount factor
        num_episodes: Number of episodes
    
    Returns:
        Value function V
    """
    # Initialize value function
    V = {s: 0.0 for s in env.state_space}
    
    for episode in range(num_episodes):
        s = env.reset()
        
        while not env.is_terminal(s):
            # Select action according to policy
            a = policy(s)
            
            # Take action and observe reward and next state
            s_next, r = env.step(s, a)
            
            # TD(0) update
            if s_next not in V:
                V[s_next] = 0.0
            
            td_error = r + gamma * V[s_next] - V[s]
            V[s] = V[s] + alpha * td_error
            
            s = s_next
    
    return V
```

### Tabular TD(0)

For tabular settings with discrete states:

```python
class TabularTD0:
    def __init__(self, num_states, alpha=0.1, gamma=0.9):
        self.V = np.zeros(num_states)
        self.alpha = alpha
        self.gamma = gamma
    
    def update(self, state, reward, next_state, done=False):
        """
        Update value function using TD(0)
        
        Args:
            state: Current state index
            reward: Immediate reward
            next_state: Next state index
            done: Whether episode terminated
        """
        if done:
            target = reward  # Terminal state has value 0
        else:
            target = reward + self.gamma * self.V[next_state]
        
        td_error = target - self.V[state]
        self.V[state] += self.alpha * td_error
        
        return td_error
```

### Convergence Properties

Under certain conditions, TD(0) converges to $v_\pi$:

**Theorem**: For a fixed policy $\pi$, if the learning rate $\alpha$ satisfies:

$$\sum_t \alpha_t = \infty \quad \text{and} \quad \sum_t \alpha_t^2 < \infty$$

and all states are visited infinitely often, then $V_t \rightarrow v_\pi$ with probability 1.

## TD(λ) and Eligibility Traces

**TD(λ)** generalizes TD(0) by using **eligibility traces** to combine information from multiple time steps, where $\lambda \in [0,1]$ controls the trace decay.

### Eligibility Traces

An **eligibility trace** $e_t(s)$ tracks how "eligible" each state is for learning based on recent visits:

$$e_t(s) = \begin{cases}
\gamma \lambda e_{t-1}(s) + 1 & \text{if } s = S_t \\
\gamma \lambda e_{t-1}(s) & \text{otherwise}
\end{cases}$$

The trace accumulates when a state is visited and decays exponentially otherwise.

### TD(λ) Update Rule

TD(λ) updates all states in proportion to their eligibility:

$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \forall s$$

where $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the TD error.

### Forward View of TD(λ)

The **forward view** shows that TD(λ) averages n-step returns:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

where $G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$ is the n-step return.

### Backward View Implementation

```python
def td_lambda(env, policy, alpha=0.1, gamma=0.9, lambda_param=0.7, num_episodes=1000):
    """
    TD(λ) algorithm with eligibility traces
    
    Args:
        env: Environment
        policy: Policy to evaluate
        alpha: Learning rate
        gamma: Discount factor
        lambda_param: Trace decay parameter λ
        num_episodes: Number of episodes
    """
    V = {s: 0.0 for s in env.state_space}
    
    for episode in range(num_episodes):
        # Initialize eligibility traces
        e = {s: 0.0 for s in env.state_space}
        
        s = env.reset()
        
        while not env.is_terminal(s):
            a = policy(s)
            s_next, r = env.step(s, a)
            
            # Compute TD error
            if s_next not in V:
                V[s_next] = 0.0
            
            delta = r + gamma * V[s_next] - V[s]
            
            # Update eligibility trace for current state
            e[s] = gamma * lambda_param * e.get(s, 0.0) + 1
            
            # Update all states
            for state in V:
                V[state] += alpha * delta * e[state]
                e[state] = gamma * lambda_param * e[state]
            
            s = s_next
    
    return V
```

### Relationship to Other Methods

TD(λ) interpolates between methods:
- **λ = 0**: TD(0) - one-step bootstrapping
- **λ = 1**: Monte Carlo - full return
- **0 < λ < 1**: Weighted combination of n-step returns

## On-Policy vs Off-Policy Methods

### On-Policy Methods

**On-policy methods** evaluate or improve the policy that is used to generate behavior. The policy being learned is the same as the policy being followed.

Examples:
- **SARSA**: Uses $Q(S_{t+1}, A_{t+1})$ where $A_{t+1}$ is from the current policy
- **Expected SARSA**: Uses expected value over policy distribution

**SARSA Update**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

where $A_{t+1} \sim \pi(\cdot | S_{t+1})$.

### Off-Policy Methods

**Off-policy methods** can learn about one policy while following another. This enables:
- **Replay Buffers**: Learn from past experience
- **Exploration**: Follow exploratory policy while learning optimal policy
- **Transfer**: Learn from expert demonstrations

**Q-Learning Update** (off-policy):
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

Uses $\max_a Q(S_{t+1}, a)$ regardless of action taken.

### Importance Sampling

For off-policy TD learning, use **importance sampling** to correct for policy mismatch:

$$\rho_t = \frac{\pi(A_t | S_t)}{b(A_t | S_t)}$$

where $b$ is the behavior policy and $\pi$ is the target policy.

**Off-policy TD(0) with importance sampling**:
$$V(S_t) \leftarrow V(S_t) + \alpha \rho_t [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

### Comparison

| Aspect | On-Policy | Off-Policy |
|-------|-----------|------------|
| **Policy** | Same for learning and behavior | Can differ |
| **Convergence** | Generally more stable | Can diverge with function approximation |
| **Sample Efficiency** | Lower (needs on-policy samples) | Higher (can reuse data) |
| **Exploration** | Must balance exploration/exploitation | Can explore separately |

## N-Step TD Methods

**N-step TD methods** use returns that look $n$ steps into the future, bridging the gap between TD(0) and Monte Carlo.

### N-Step Return

The **n-step return** is:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

### N-Step TD Update

**N-step TD update**:
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t^{(n)} - V(S_t)]$$

### Algorithm

```python
def n_step_td(env, policy, n=3, alpha=0.1, gamma=0.9, num_episodes=1000):
    """
    N-step TD algorithm
    
    Args:
        env: Environment
        policy: Policy to evaluate
        n: Number of steps
        alpha: Learning rate
        gamma: Discount factor
        num_episodes: Number of episodes
    """
    V = {s: 0.0 for s in env.state_space}
    
    for episode in range(num_episodes):
        # Store trajectory
        states = []
        rewards = []
        
        s = env.reset()
        states.append(s)
        t = 0
        T = float('inf')
        
        while True:
            if t < T:
                # Take action
                a = policy(s)
                s_next, r = env.step(s, a)
                states.append(s_next)
                rewards.append(r)
                
                if env.is_terminal(s_next):
                    T = t + 1
            
            # Update state τ steps ago
            tau = t - n + 1
            if tau >= 0:
                # Compute n-step return
                G = sum(gamma**(i - tau) * rewards[i] 
                       for i in range(tau, min(tau + n, T)))
                
                if tau + n < T:
                    G += gamma**n * V[states[tau + n]]
                
                # Update value
                V[states[tau]] += alpha * (G - V[states[tau]])
            
            if tau == T - 1:
                break
            
            t += 1
            s = s_next
    
    return V
```

### Relationship to Other Methods

- **n = 1**: TD(0) - one-step bootstrapping
- **n = ∞**: Monte Carlo - full return
- **n = intermediate**: Balance between bias and variance

## Monte Carlo vs Temporal Difference

### Monte Carlo Methods

**Monte Carlo methods** learn from complete episodes:

$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

where $G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T$ is the actual return.

### Comparison

| Aspect | Monte Carlo | Temporal Difference |
|-------|-------------|---------------------|
| **Bootstrap** | No | Yes |
| **Bias** | Unbiased | Biased |
| **Variance** | High | Low |
| **Update Timing** | End of episode | Every step |
| **Works Online** | No | Yes |
| **Convergence** | Slower | Faster |
| **Works Continuing** | No | Yes |

### Bias-Variance Trade-off

- **Monte Carlo**: Unbiased but high variance (uses actual returns)
- **TD**: Biased but low variance (uses estimates)

The bias in TD comes from bootstrapping with inaccurate estimates, but this bias decreases as learning progresses.

### Example: Random Walk

Consider a random walk with states $\{0, 1, 2, \ldots, 10\}$, where state 0 is terminal with reward 0, and state 10 is terminal with reward 1. Monte Carlo and TD methods both converge to the true values, but TD converges faster due to bootstrapping.

## Convergence and Stability

### Convergence Conditions

For TD(0) to converge to $v_\pi$:

1. **Learning Rate**: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$
2. **Visitation**: All states visited infinitely often
3. **Bounded Rewards**: Rewards are bounded

### Stability with Function Approximation

With **function approximation**, TD learning can diverge:

**Counterexample**: Baird's counterexample shows that off-policy TD with linear function approximation can diverge even with small learning rates.

**Conditions for Stability**:
- **On-policy**: Generally stable
- **Off-policy**: Requires additional conditions (e.g., importance sampling, gradient corrections)

### Gradient TD Methods

**Gradient TD methods** use gradient descent on the projected Bellman error to ensure stability:

**GTD (Gradient Temporal Difference)**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [\delta_t \mathbf{x}_t - \gamma \mathbf{x}_{t+1} (\mathbf{x}_t^\top \mathbf{v}_t)]$$

**TDC (Temporal Difference with Correction)**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha [\delta_t \mathbf{x}_t - \gamma \mathbf{x}_{t+1} (\mathbf{x}_t^\top \mathbf{v}_t)]$$
$$\mathbf{v}_{t+1} = \mathbf{v}_t + \beta [\delta_t - \mathbf{x}_t^\top \mathbf{v}_t] \mathbf{x}_t$$

These methods are guaranteed to converge under off-policy learning with linear function approximation.

## Function Approximation with TD

### Linear Function Approximation

With **linear function approximation**, value function is:

$$V(s) = \mathbf{w}^\top \phi(s)$$

where $\phi(s)$ is a feature vector.

**TD(0) with linear approximation**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \phi(S_t)$$

where $\delta_t = R_{t+1} + \gamma \mathbf{w}_t^\top \phi(S_{t+1}) - \mathbf{w}_t^\top \phi(S_t)$.

### Neural Network Approximation

With **neural networks**, use gradient descent:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \delta_t \nabla_{\mathbf{w}} V(S_t; \mathbf{w}_t)$$

### TD with Deep Networks

Deep Q-Networks (DQN) combine TD learning with deep neural networks:

```python
class DQN:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
    
    def update(self, batch):
        """
        Update Q-network using TD learning
        
        Args:
            batch: Batch of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = batch
        
        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + (1 - dones) * gamma * next_q_values
        
        # Compute TD error
        td_error = targets - q_values.squeeze()
        loss = F.mse_loss(q_values.squeeze(), targets)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## Bias-Variance Trade-off in TD Learning

### Sources of Bias

1. **Bootstrapping Bias**: Using estimates instead of true values
2. **Function Approximation Bias**: Limited representational capacity
3. **Off-Policy Bias**: Policy mismatch in off-policy learning

### Sources of Variance

1. **Stochastic Environment**: Random transitions and rewards
2. **Stochastic Policy**: Random action selection
3. **Monte Carlo Variance**: Full return has high variance

### Trade-off Analysis

**TD(0)**:
- **Bias**: Moderate (one-step bootstrapping)
- **Variance**: Low (uses single estimate)

**Monte Carlo**:
- **Bias**: None (unbiased)
- **Variance**: High (full return)

**N-step TD**:
- **Bias**: Decreases with $n$
- **Variance**: Increases with $n$

**TD(λ)**:
- **Bias**: Decreases with $\lambda$
- **Variance**: Increases with $\lambda$

### Optimal Choice

The optimal method depends on:
- **Problem Characteristics**: Episodic vs continuing, reward sparsity
- **Function Approximation**: Linear vs nonlinear, capacity
- **Sample Availability**: Limited vs abundant samples
- **Computational Resources**: Online vs batch learning

## Key Takeaways

1. **Temporal Difference learning** bootstraps from value estimates, enabling online learning and faster convergence compared to Monte Carlo methods.

2. **TD(0)** is the simplest TD algorithm, updating values after each step using one-step returns, balancing bias and variance effectively.

3. **TD(λ) with eligibility traces** combines information from multiple time steps, interpolating between TD(0) and Monte Carlo methods based on $\lambda$.

4. **On-policy methods** (e.g., SARSA) learn the policy being followed, while **off-policy methods** (e.g., Q-learning) can learn different policies, enabling more flexible learning.

5. **N-step TD methods** bridge TD(0) and Monte Carlo by using n-step returns, with bias decreasing and variance increasing as $n$ increases.

6. **Monte Carlo methods** are unbiased but have high variance, while **TD methods** have bias but lower variance, creating a fundamental trade-off.

7. **Convergence** of TD learning is guaranteed under certain conditions, but **stability** with function approximation requires careful treatment, especially for off-policy learning.

8. **Function approximation** enables TD learning in large state spaces, with linear and neural network approximations being common approaches.

9. **The bias-variance trade-off** is central to TD learning, with different methods (TD(0), n-step TD, TD(λ)) offering different points on this trade-off.

10. **TD learning** forms the foundation for many advanced RL algorithms, including Q-learning, SARSA, and deep Q-networks, making it one of the most important concepts in reinforcement learning.
