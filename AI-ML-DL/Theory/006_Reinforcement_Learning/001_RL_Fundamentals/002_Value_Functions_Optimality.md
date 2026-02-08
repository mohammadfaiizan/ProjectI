# Value Functions and Optimality

## Table of Contents

1. [Introduction to Value Functions](#introduction-to-value-functions)
2. [State-Value Function](#state-value-function)
3. [Action-Value Function](#action-value-function)
4. [Optimal Value Functions](#optimal-value-functions)
5. [Value Iteration Algorithm](#value-iteration-algorithm)
6. [Policy Iteration Algorithm](#policy-iteration-algorithm)
7. [Contraction Mapping Theorem](#contraction-mapping-theorem)
8. [Generalized Policy Iteration](#generalized-policy-iteration)
9. [Asynchronous Dynamic Programming](#asynchronous-dynamic-programming)
10. [Key Takeaways](#key-takeaways)

## Introduction to Value Functions

Value functions are fundamental to reinforcement learning, providing a way to evaluate and compare policies. They quantify the long-term value of being in a state or taking an action, enabling agents to make informed decisions about which actions to take.

The concept of value functions bridges the gap between immediate rewards and long-term goals. While rewards provide immediate feedback, value functions capture the cumulative expected return, allowing agents to reason about sequences of actions and their consequences.

Value functions serve multiple purposes:
- **Policy Evaluation**: Assess how good a policy is
- **Policy Improvement**: Guide the search for better policies
- **Optimal Control**: Find policies that maximize expected return
- **Planning**: Simulate future outcomes to make decisions

## State-Value Function

The **state-value function** $v_\pi(s)$ for policy $\pi$ is the expected return when starting in state $s$ and following policy $\pi$ thereafter:

$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \middle| S_t = s\right]$$

where $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$ is the return.

### Properties of State-Value Functions

The state-value function has several important properties:

1. **Uniqueness**: For a given policy $\pi$ and MDP, $v_\pi$ is unique
2. **Boundedness**: If rewards are bounded, $v_\pi(s)$ is bounded for all $s$
3. **Linearity**: Value functions are linear in the policy space
4. **Monotonicity**: If $\pi_1 \geq \pi_2$ (in terms of expected return), then $v_{\pi_1} \geq v_{\pi_2}$

### Bellman Equation for State Values

The state-value function satisfies the Bellman equation:

$$v_\pi(s) = \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

This recursive relationship expresses the value of a state as the expected immediate reward plus the discounted value of the next state.

### Matrix Form

For finite MDPs, the Bellman equation can be written in matrix form:

$$\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi \mathbf{v}_\pi$$

where:
- $\mathbf{v}_\pi$ is a vector of state values
- $\mathbf{r}_\pi$ is the expected immediate reward vector
- $\mathbf{P}_\pi$ is the state transition probability matrix under policy $\pi$

Solving for $\mathbf{v}_\pi$:

$$\mathbf{v}_\pi = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi$$

This direct solution requires matrix inversion, which is computationally expensive for large state spaces.

## Action-Value Function

The **action-value function** $q_\pi(s, a)$ for policy $\pi$ is the expected return when starting in state $s$, taking action $a$, and then following policy $\pi$:

$$q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### Relationship to State-Value Function

The action-value function is related to the state-value function:

$$v_\pi(s) = \sum_a \pi(a | s) q_\pi(s, a)$$

$$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

The state-value function averages over actions according to the policy, while the action-value function evaluates specific actions.

### Advantages of Action-Value Functions

Action-value functions are particularly useful because:

1. **Direct Policy Improvement**: Can construct greedy policies without a model
2. **Model-Free Learning**: Enable learning without knowing transition probabilities
3. **Action Comparison**: Directly compare the value of different actions

## Optimal Value Functions

An **optimal policy** $\pi_*$ achieves the maximum expected return in all states:

$$\pi_* = \arg\max_\pi v_\pi(s) \quad \forall s \in S$$

### Optimal State-Value Function

The **optimal state-value function** $v_*(s)$ is:

$$v_*(s) = \max_\pi v_\pi(s) = v_{\pi_*}(s)$$

It represents the maximum expected return achievable from state $s$.

### Optimal Action-Value Function

The **optimal action-value function** $q_*(s, a)$ is:

$$q_*(s, a) = \max_\pi q_\pi(s, a) = q_{\pi_*}(s, a)$$

It represents the maximum expected return achievable from state $s$ when taking action $a$.

### Relationship Between Optimal Value Functions

$$v_*(s) = \max_a q_*(s, a)$$

$$q_*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]$$

### Optimal Policy from Optimal Value Functions

Given $q_*(s, a)$, any policy that is greedy with respect to $q_*$ is optimal:

$$\pi_*(s) = \arg\max_a q_*(s, a)$$

If multiple actions achieve the maximum, any of them can be chosen (deterministic optimal policy). There may be multiple optimal policies, but they all share the same optimal value functions.

## Value Iteration Algorithm

**Value iteration** directly computes the optimal value function by iteratively applying the Bellman optimality equation:

$$v_{k+1}(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

### Algorithm Description

```python
def value_iteration(mdp, theta=1e-6, max_iterations=1000):
    """
    Value iteration algorithm for finding optimal value function
    
    Args:
        mdp: MDP with states S, actions A, transitions P, rewards R, discount gamma
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Optimal value function v_*
    """
    # Initialize value function
    V = {s: 0.0 for s in mdp.S}
    
    for iteration in range(max_iterations):
        V_new = {}
        delta = 0.0
        
        for s in mdp.S:
            # Compute value for each action
            action_values = []
            for a in mdp.A(s):
                action_value = sum(
                    mdp.P(s, a, s_prime) * 
                    (mdp.R(s, a, s_prime) + mdp.gamma * V[s_prime])
                    for s_prime in mdp.S
                )
                action_values.append(action_value)
            
            # Take maximum over actions
            V_new[s] = max(action_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        # Check convergence
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return V
```

### Convergence Properties

Value iteration converges to $v_*$ under the contraction mapping theorem:
- **Convergence Rate**: Linear with rate $\gamma$
- **Error Bound**: $||v_k - v_*||_\infty \leq \gamma^k ||v_0 - v_*||_\infty$
- **Stopping Criterion**: Stop when $||v_{k+1} - v_k||_\infty < \theta$

### Extracting Optimal Policy

After convergence, extract the optimal policy:

```python
def extract_policy(mdp, V):
    """
    Extract optimal policy from optimal value function
    
    Args:
        mdp: MDP
        V: Optimal value function v_*
    
    Returns:
        Optimal policy pi_*
    """
    pi = {}
    
    for s in mdp.S:
        # Find action that maximizes value
        best_action = None
        best_value = float('-inf')
        
        for a in mdp.A(s):
            action_value = sum(
                mdp.P(s, a, s_prime) * 
                (mdp.R(s, a, s_prime) + mdp.gamma * V[s_prime])
                for s_prime in mdp.S
            )
            if action_value > best_value:
                best_value = action_value
                best_action = a
        
        pi[s] = best_action
    
    return pi
```

## Policy Iteration Algorithm

**Policy iteration** alternates between policy evaluation and policy improvement until convergence.

### Policy Evaluation

Given a policy $\pi$, compute its value function by solving the linear system:

$$v_\pi = \mathbf{r}_\pi + \gamma \mathbf{P}_\pi v_\pi$$

Or iteratively:

$$v_{k+1}(s) = \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_k(s')]$$

```python
def policy_evaluation(mdp, pi, theta=1e-6, max_iterations=1000):
    """
    Evaluate policy by computing its value function
    
    Args:
        mdp: MDP
        pi: Policy to evaluate
        theta: Convergence threshold
        max_iterations: Maximum iterations
    
    Returns:
        Value function v_pi
    """
    V = {s: 0.0 for s in mdp.S}
    
    for iteration in range(max_iterations):
        V_new = {}
        delta = 0.0
        
        for s in mdp.S:
            # Compute value according to policy
            value = 0.0
            for a in mdp.A(s):
                action_prob = pi.get(a, 0.0) if isinstance(pi, dict) else (1.0 if pi(s) == a else 0.0)
                for s_prime in mdp.S:
                    transition_prob = mdp.P(s, a, s_prime)
                    reward = mdp.R(s, a, s_prime)
                    value += action_prob * transition_prob * (reward + mdp.gamma * V[s_prime])
            
            V_new[s] = value
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        if delta < theta:
            break
    
    return V
```

### Policy Improvement

Improve the policy by making it greedy with respect to the current value function:

$$\pi'(s) = \arg\max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

```python
def policy_improvement(mdp, V):
    """
    Improve policy by making it greedy with respect to value function
    
    Args:
        mdp: MDP
        V: Value function
    
    Returns:
        Improved policy pi
    """
    pi = {}
    
    for s in mdp.S:
        best_action = None
        best_value = float('-inf')
        
        for a in mdp.A(s):
            action_value = sum(
                mdp.P(s, a, s_prime) * 
                (mdp.R(s, a, s_prime) + mdp.gamma * V[s_prime])
                for s_prime in mdp.S
            )
            if action_value > best_value:
                best_value = action_value
                best_action = a
        
        pi[s] = best_action
    
    return pi
```

### Complete Policy Iteration

```python
def policy_iteration(mdp, theta=1e-6):
    """
    Policy iteration algorithm
    
    Args:
        mdp: MDP
        theta: Convergence threshold
    
    Returns:
        Optimal policy pi_*
    """
    # Initialize random policy
    pi = {s: random.choice(mdp.A(s)) for s in mdp.S}
    
    while True:
        # Policy evaluation
        V = policy_evaluation(mdp, pi, theta)
        
        # Policy improvement
        pi_new = policy_improvement(mdp, V)
        
        # Check if policy is stable
        if pi == pi_new:
            break
        
        pi = pi_new
    
    return pi
```

### Convergence Properties

Policy iteration converges to an optimal policy in a finite number of iterations:
- **Finite Convergence**: At most $|A|^{|S|}$ iterations
- **Monotonic Improvement**: Each iteration improves the policy
- **Optimality**: Converged policy is optimal

## Contraction Mapping Theorem

The **contraction mapping theorem** provides the theoretical foundation for convergence of value iteration and policy evaluation.

### Contraction Mapping

An operator $T$ on a metric space $(X, d)$ is a **contraction mapping** if there exists $\gamma \in [0, 1)$ such that:

$$d(T(x), T(y)) \leq \gamma d(x, y) \quad \forall x, y \in X$$

### Bellman Operator

The **Bellman operator** $T_\pi$ for policy $\pi$ is:

$$(T_\pi v)(s) = \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma v(s')]$$

The **Bellman optimality operator** $T_*$ is:

$$(T_* v)(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v(s')]$$

### Contraction Property

Both operators are contractions with respect to the supremum norm:

$$||T_\pi v_1 - T_\pi v_2||_\infty \leq \gamma ||v_1 - v_2||_\infty$$

$$||T_* v_1 - T_* v_2||_\infty \leq \gamma ||v_1 - v_2||_\infty$$

### Fixed Point Theorem

By the contraction mapping theorem:
- **Unique Fixed Point**: Each operator has a unique fixed point
- **Convergence**: Iteration $v_{k+1} = T v_k$ converges to the fixed point
- **Rate**: Linear convergence with rate $\gamma$

The fixed points are:
- $T_\pi v_\pi = v_\pi$ (policy evaluation)
- $T_* v_* = v_*$ (value iteration)

## Generalized Policy Iteration

**Generalized Policy Iteration (GPI)** is a general framework that alternates between policy evaluation and policy improvement, without requiring either to converge completely.

### GPI Framework

GPI maintains:
- An approximate value function $V \approx v_\pi$
- An approximate policy $\pi$ that is greedy with respect to $V$

The two processes interact:
- **Policy Evaluation**: Makes $V$ closer to $v_\pi$
- **Policy Improvement**: Makes $\pi$ better with respect to $V$

### Convergence

GPI converges to optimality even when:
- Policy evaluation is approximate (stopped early)
- Policy improvement is approximate (not fully greedy)
- Both processes are interleaved arbitrarily

### Examples of GPI

Many RL algorithms are instances of GPI:
- **Value Iteration**: One step of policy evaluation, then policy improvement
- **Policy Iteration**: Full policy evaluation, then policy improvement
- **Q-Learning**: Approximate policy evaluation, greedy policy improvement
- **SARSA**: Approximate policy evaluation, on-policy improvement

## Asynchronous Dynamic Programming

**Asynchronous Dynamic Programming** updates states in any order, not necessarily all at once, enabling:
- **In-place Updates**: Update states as they are visited
- **Prioritized Updates**: Update important states more frequently
- **Real-time Updates**: Update during agent-environment interaction

### Asynchronous Value Iteration

```python
def asynchronous_value_iteration(mdp, state_sequence):
    """
    Asynchronous value iteration
    
    Args:
        mdp: MDP
        state_sequence: Sequence of states to update
    
    Returns:
        Value function V
    """
    V = {s: 0.0 for s in mdp.S}
    
    for s in state_sequence:
        # Update only state s
        V[s] = max([
            sum(mdp.P(s, a, s_prime) * 
                (mdp.R(s, a, s_prime) + mdp.gamma * V[s_prime])
                for s_prime in mdp.S)
            for a in mdp.A(s)
        ])
    
    return V
```

### Prioritized Sweeping

**Prioritized sweeping** focuses updates on states whose values have changed significantly:

```python
def prioritized_sweeping(mdp, V, priority_queue, theta=1e-6):
    """
    Prioritized sweeping for value iteration
    
    Args:
        mdp: MDP
        V: Current value function
        priority_queue: Priority queue of states to update
        theta: Threshold for significant changes
    """
    while not priority_queue.empty():
        s = priority_queue.get()
        
        # Update value
        old_value = V[s]
        V[s] = max([
            sum(mdp.P(s, a, s_prime) * 
                (mdp.R(s, a, s_prime) + mdp.gamma * V[s_prime])
                for s_prime in mdp.S)
            for a in mdp.A(s)
        ])
        
        # Check if change is significant
        if abs(V[s] - old_value) > theta:
            # Add predecessors to queue
            for s_pred in mdp.predecessors(s):
                priority = abs(V[s] - old_value)
                priority_queue.put(s_pred, priority)
```

### Convergence

Asynchronous DP converges to optimality under certain conditions:
- **Fairness**: Each state is updated infinitely often
- **Contraction**: Update operator remains a contraction

## Key Takeaways

1. **Value functions** ($v_\pi$ and $q_\pi$) quantify the long-term expected return, enabling policy evaluation and comparison.

2. **Optimal value functions** ($v_*$ and $q_*$) represent the maximum achievable return and characterize optimal policies.

3. **Value iteration** directly computes optimal values through iterative application of the Bellman optimality equation, converging linearly with rate $\gamma$.

4. **Policy iteration** alternates between policy evaluation and improvement, converging in finite iterations with monotonic improvement.

5. **The contraction mapping theorem** provides theoretical guarantees for convergence, showing that Bellman operators have unique fixed points.

6. **Generalized Policy Iteration** is a flexible framework that allows approximate evaluation and improvement, forming the basis for many RL algorithms.

7. **Asynchronous dynamic programming** enables efficient updates by focusing computation on relevant states, supporting real-time and prioritized updates.

8. **State-value functions** average over actions according to the policy, while **action-value functions** evaluate specific actions, making them more useful for model-free learning.

9. **Optimal policies** can be extracted from optimal value functions by taking greedy actions, with deterministic optimal policies existing for finite MDPs.

10. **The relationship between value functions** enables efficient computation and provides multiple perspectives on policy evaluation and optimization.
