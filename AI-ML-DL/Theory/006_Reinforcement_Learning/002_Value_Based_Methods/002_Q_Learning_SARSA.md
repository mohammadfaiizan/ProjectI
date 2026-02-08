# Q-Learning and SARSA

## Table of Contents

1. [Introduction to Q-Learning and SARSA](#introduction-to-q-learning-and-sarsa)
2. [Q-Learning Algorithm](#q-learning-algorithm)
3. [SARSA Algorithm](#sarsa-algorithm)
4. [Expected SARSA](#expected-sarsa)
5. [Convergence Proofs](#convergence-proofs)
6. [Tabular vs Function Approximation](#tabular-vs-function-approximation)
7. [Practical Implementation Considerations](#practical-implementation-considerations)
8. [Comparison and When to Use Each](#comparison-and-when-to-use-each)
9. [Extensions and Variants](#extensions-and-variants)
10. [Key Takeaways](#key-takeaways)

## Introduction to Q-Learning and SARSA

**Q-Learning** and **SARSA** are two fundamental temporal difference algorithms for learning action-value functions in reinforcement learning. Both are model-free methods that learn optimal or near-optimal policies through experience, but they differ in their update rules and policy characteristics.

- **Q-Learning**: Off-policy algorithm that learns the optimal action-value function $q_*$ by using the maximum over next-state actions
- **SARSA**: On-policy algorithm that learns the action-value function $q_\pi$ for the policy being followed

The key difference lies in how they compute the TD target:
- **Q-Learning**: $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ (off-policy)
- **SARSA**: $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ (on-policy)

## Q-Learning Algorithm

**Q-Learning** is an off-policy temporal difference algorithm that directly learns the optimal action-value function $q_*$ without requiring a model of the environment.

### Update Rule

The Q-Learning update rule is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

where:
- $\alpha$ is the learning rate
- $\gamma$ is the discount factor
- $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ is the TD target
- The action $A_{t+1}$ actually taken doesn't affect the update (off-policy)

### Algorithm Description

```python
def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
    """
    Q-Learning algorithm
    
    Args:
        env: Environment
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration parameter for epsilon-greedy
        num_episodes: Number of episodes
    
    Returns:
        Optimal Q-function Q_*
    """
    # Initialize Q-function
    Q = {}
    
    for episode in range(num_episodes):
        s = env.reset()
        
        while not env.is_terminal(s):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                a = env.random_action()
            else:
                a = argmax_q(Q, s)  # Greedy action
            
            # Take action and observe
            s_next, r = env.step(s, a)
            
            # Q-Learning update (off-policy)
            if (s, a) not in Q:
                Q[(s, a)] = 0.0
            if s_next not in [s for (s, _) in Q.keys()]:
                # Initialize Q for next state
                pass
            
            # Compute max Q-value for next state
            max_next_q = max([Q.get((s_next, a_prime), 0.0) 
                             for a_prime in env.actions])
            
            # Update Q-value
            td_target = r + gamma * max_next_q
            td_error = td_target - Q[(s, a)]
            Q[(s, a)] += alpha * td_error
            
            s = s_next
    
    return Q
```

### Optimality Property

Q-Learning converges to $q_*$ under the following conditions:

1. **Learning Rate**: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$
2. **Visitation**: All state-action pairs visited infinitely often
3. **Bounded Rewards**: Rewards are bounded

The key property is that Q-Learning learns $q_*$ **regardless of the policy being followed**, as long as all state-action pairs are visited sufficiently often.

### Tabular Q-Learning

For discrete state-action spaces:

```python
class TabularQLearning:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
    
    def update(self, state, action, reward, next_state, done=False):
        """
        Q-Learning update
        
        Args:
            state: Current state index
            action: Action taken
            reward: Immediate reward
            next_state: Next state index
            done: Whether episode terminated
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        return td_error
    
    def select_action(self, state, epsilon=0.1):
        """
        Epsilon-greedy action selection
        
        Args:
            state: Current state
            epsilon: Exploration probability
        
        Returns:
            Selected action
        """
        if random.random() < epsilon:
            return random.randint(0, self.Q.shape[1] - 1)
        else:
            return np.argmax(self.Q[state])
```

## SARSA Algorithm

**SARSA** (State-Action-Reward-State-Action) is an on-policy temporal difference algorithm that learns the action-value function $q_\pi$ for the policy being followed.

### Update Rule

The SARSA update rule is:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

where $A_{t+1}$ is the action selected by the current policy $\pi$ in state $S_{t+1}$.

### Algorithm Description

```python
def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
    """
    SARSA algorithm
    
    Args:
        env: Environment
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration parameter
        num_episodes: Number of episodes
    
    Returns:
        Q-function Q_pi for policy pi
    """
    Q = {}
    
    for episode in range(num_episodes):
        s = env.reset()
        
        # Select initial action using epsilon-greedy
        a = epsilon_greedy(Q, s, epsilon)
        
        while not env.is_terminal(s):
            # Take action and observe
            s_next, r = env.step(s, a)
            
            # Select next action using epsilon-greedy (on-policy)
            a_next = epsilon_greedy(Q, s_next, epsilon)
            
            # SARSA update (on-policy)
            if (s, a) not in Q:
                Q[(s, a)] = 0.0
            if (s_next, a_next) not in Q:
                Q[(s_next, a_next)] = 0.0
            
            td_target = r + gamma * Q[(s_next, a_next)]
            td_error = td_target - Q[(s, a)]
            Q[(s, a)] += alpha * td_error
            
            # Move to next state-action pair
            s = s_next
            a = a_next
    
    return Q
```

### Policy Dependency

SARSA learns $q_\pi$ for the policy $\pi$ being followed. If the policy changes (e.g., epsilon decreases), the Q-function adapts to the new policy. This makes SARSA more conservative than Q-Learning, as it accounts for the exploration in the policy.

### Tabular SARSA

```python
class TabularSARSA:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
    
    def update(self, state, action, reward, next_state, next_action, done=False):
        """
        SARSA update
        
        Args:
            state: Current state
            action: Current action
            reward: Immediate reward
            next_state: Next state
            next_action: Next action (from policy)
            done: Whether episode terminated
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
        
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        return td_error
```

## Expected SARSA

**Expected SARSA** is a variant that uses the expected value of Q over the policy distribution instead of a single sample, reducing variance.

### Update Rule

Expected SARSA update:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \sum_a \pi(a | S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]$$

### Algorithm

```python
def expected_sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000):
    """
    Expected SARSA algorithm
    
    Args:
        env: Environment
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration parameter
        num_episodes: Number of episodes
    """
    Q = {}
    
    for episode in range(num_episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon)
        
        while not env.is_terminal(s):
            s_next, r = env.step(s, a)
            
            # Compute expected Q-value over policy
            expected_q = 0.0
            for a_prime in env.actions:
                if (s_next, a_prime) not in Q:
                    Q[(s_next, a_prime)] = 0.0
                
                # Policy probability
                if a_prime == argmax_q(Q, s_next):
                    prob = 1 - epsilon + epsilon / len(env.actions)
                else:
                    prob = epsilon / len(env.actions)
                
                expected_q += prob * Q[(s_next, a_prime)]
            
            # Expected SARSA update
            if (s, a) not in Q:
                Q[(s, a)] = 0.0
            
            td_target = r + gamma * expected_q
            td_error = td_target - Q[(s, a)]
            Q[(s, a)] += alpha * td_error
            
            s = s_next
            a = epsilon_greedy(Q, s, epsilon)
    
    return Q
```

### Advantages

Expected SARSA:
- **Lower Variance**: Uses expectation instead of sample
- **More Stable**: Smoother learning
- **Off-Policy Capable**: Can use different target policy

## Convergence Proofs

### Q-Learning Convergence

**Theorem**: Under the following conditions, Q-Learning converges to $q_*$ with probability 1:

1. **Learning Rate**: $\sum_t \alpha_t(s, a) = \infty$ and $\sum_t \alpha_t^2(s, a) < \infty$ for all $(s, a)$
2. **Visitation**: All state-action pairs visited infinitely often
3. **Bounded Rewards**: $|R_t| \leq R_{\max}$ for all $t$

**Proof Sketch**:

The Q-Learning update can be written as:

$$Q_{t+1}(s, a) = Q_t(s, a) + \alpha_t [T_* Q_t(s, a) - Q_t(s, a) + w_t(s, a)]$$

where:
- $T_*$ is the Bellman optimality operator
- $w_t(s, a)$ is a noise term with zero mean

This is a stochastic approximation algorithm. Under the conditions above, it converges to the fixed point of $T_*$, which is $q_*$.

### SARSA Convergence

**Theorem**: Under similar conditions, SARSA converges to $q_\pi$ where $\pi$ is the policy being followed (assuming it converges to a fixed policy).

The convergence is to the action-value function of the policy being followed, not necessarily the optimal policy.

### Convergence Rates

- **Q-Learning**: Converges to optimal policy
- **SARSA**: Converges to policy being followed (may be suboptimal if exploration continues)
- **Expected SARSA**: Similar to SARSA but with lower variance

## Tabular vs Function Approximation

### Tabular Setting

In **tabular settings** with discrete, finite state-action spaces:

- **Storage**: $O(|S| \times |A|)$ table
- **Convergence**: Guaranteed under conditions
- **Optimality**: Q-Learning finds optimal policy

### Function Approximation

With **function approximation** (e.g., neural networks):

$$Q(s, a; \theta) \approx q_*(s, a)$$

**Challenges**:
- **Divergence**: Can diverge, especially off-policy
- **Approximation Error**: Limited representational capacity
- **Generalization**: Must generalize across states

**Deep Q-Networks (DQN)** address these challenges:
- Experience replay
- Target networks
- Gradient clipping

## Practical Implementation Considerations

### Learning Rate Scheduling

Use adaptive or decaying learning rates:

```python
def adaptive_learning_rate(episode, initial_alpha=0.1, decay=0.995):
    """Decay learning rate over episodes"""
    return initial_alpha * (decay ** episode)

def visit_based_learning_rate(state, action, visit_count, initial_alpha=1.0):
    """Learning rate based on visit count"""
    return initial_alpha / (1 + visit_count)
```

### Exploration Strategies

**Epsilon-Greedy**:
```python
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random_action()
    else:
        return argmax(Q[state])
```

**Epsilon Decay**:
```python
epsilon_t = max(epsilon_min, epsilon_0 * decay_rate ** episode)
```

**Upper Confidence Bound**:
```python
def ucb_action_selection(Q, state, visit_counts, c=2.0):
    """UCB action selection"""
    ucb_values = []
    total_visits = sum(visit_counts[state].values())
    
    for a in actions:
        if visit_counts[state][a] == 0:
            ucb = float('inf')
        else:
            ucb = Q[state, a] + c * sqrt(log(total_visits) / visit_counts[state][a])
        ucb_values.append(ucb)
    
    return argmax(ucb_values)
```

### Initialization

**Optimistic Initialization**:
```python
Q = np.full((num_states, num_actions), optimistic_value)
```

Encourages exploration by making all actions initially attractive.

## Comparison and When to Use Each

### Q-Learning Advantages

1. **Optimal Policy**: Learns optimal policy regardless of behavior policy
2. **Off-Policy**: Can learn from any policy's experience
3. **Flexible**: Can use replay buffers, expert demonstrations
4. **Convergence**: Guaranteed convergence to $q_*$

### Q-Learning Disadvantages

1. **Exploration**: May not explore enough if behavior policy is poor
2. **Function Approximation**: Can diverge with function approximation
3. **Overestimation**: Max operator can cause overestimation bias

### SARSA Advantages

1. **Safety**: More conservative, accounts for exploration
2. **On-Policy Stability**: More stable with function approximation
3. **Risk-Averse**: Better for risky environments (e.g., cliff walking)

### SARSA Disadvantages

1. **Suboptimal**: May converge to suboptimal policy if exploration continues
2. **Policy Dependency**: Q-function depends on current policy
3. **Sample Efficiency**: Less flexible with off-policy data

### When to Use Q-Learning

- Learning optimal policy
- Off-policy learning (replay buffers, demonstrations)
- Safe exploration not critical
- Tabular or stable function approximation

### When to Use SARSA

- Safety-critical environments
- On-policy learning
- Risk-averse policies desired
- Function approximation with stability concerns

## Extensions and Variants

### Double Q-Learning

**Double Q-Learning** addresses overestimation bias by maintaining two Q-functions:

```python
def double_q_learning(env, alpha=0.1, gamma=0.9, num_episodes=1000):
    """Double Q-Learning to reduce overestimation"""
    Q1 = {}
    Q2 = {}
    
    for episode in range(num_episodes):
        s = env.reset()
        
        while not env.is_terminal(s):
            a = epsilon_greedy_combined(Q1, Q2, s)
            s_next, r = env.step(s, a)
            
            # Randomly update Q1 or Q2
            if random.random() < 0.5:
                # Update Q1 using Q2 for target
                a_star = argmax(Q1, s_next)
                target = r + gamma * Q2.get((s_next, a_star), 0.0)
                Q1[(s, a)] = Q1.get((s, a), 0.0) + alpha * (target - Q1.get((s, a), 0.0))
            else:
                # Update Q2 using Q1 for target
                a_star = argmax(Q2, s_next)
                target = r + gamma * Q1.get((s_next, a_star), 0.0)
                Q2[(s, a)] = Q2.get((s, a), 0.0) + alpha * (target - Q2.get((s, a), 0.0))
            
            s = s_next
    
    # Combined Q-function
    Q = {key: (Q1.get(key, 0) + Q2.get(key, 0)) / 2 for key in set(Q1.keys()) | set(Q2.keys())}
    return Q
```

### Multi-Step Q-Learning

**Multi-step Q-Learning** uses n-step returns:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \max_a Q(S_{t+n}, a)$$

### Q(λ)

**Q(λ)** combines Q-Learning with eligibility traces:

```python
def q_lambda(env, alpha=0.1, gamma=0.9, lambda_param=0.7, num_episodes=1000):
    """Q(λ) with eligibility traces"""
    Q = {}
    
    for episode in range(num_episodes):
        e = {}  # Eligibility traces
        s = env.reset()
        a = epsilon_greedy(Q, s)
        
        while not env.is_terminal(s):
            s_next, r = env.step(s, a)
            a_next = epsilon_greedy(Q, s_next)
            
            # Compute TD error
            delta = r + gamma * Q.get((s_next, a_next), 0.0) - Q.get((s, a), 0.0)
            
            # Update eligibility trace
            e[(s, a)] = e.get((s, a), 0.0) + 1
            
            # Update all Q-values
            for (s_e, a_e) in e:
                Q[(s_e, a_e)] = Q.get((s_e, a_e), 0.0) + alpha * delta * e[(s_e, a_e)]
                e[(s_e, a_e)] *= gamma * lambda_param
            
            s, a = s_next, a_next
    
    return Q
```

## Key Takeaways

1. **Q-Learning** is an off-policy algorithm that learns the optimal action-value function $q_*$ by using the maximum over next-state actions, regardless of the policy being followed.

2. **SARSA** is an on-policy algorithm that learns the action-value function $q_\pi$ for the policy being followed, making it more conservative and safer in risky environments.

3. **Expected SARSA** reduces variance by using the expected value over the policy distribution instead of a single sample, providing a middle ground between SARSA and Q-Learning.

4. **Convergence guarantees** exist for both algorithms under certain conditions, with Q-Learning converging to optimal policy and SARSA converging to the policy being followed.

5. **Tabular implementations** are straightforward and guaranteed to converge, while **function approximation** introduces challenges like potential divergence, especially for off-policy methods.

6. **Practical considerations** include learning rate scheduling, exploration strategies, and initialization methods, all of which significantly impact performance.

7. **Q-Learning excels** when learning optimal policies and using off-policy data, while **SARSA is better** for safety-critical environments and on-policy learning scenarios.

8. **Extensions** like Double Q-Learning address overestimation bias, while multi-step and eligibility trace variants improve sample efficiency and learning speed.

9. **The choice between Q-Learning and SARSA** depends on the problem characteristics, safety requirements, and whether off-policy learning is beneficial.

10. **Both algorithms** form the foundation for advanced methods like Deep Q-Networks and are essential tools in the reinforcement learning toolkit.
