# Markov Decision Processes and Bellman Equations

## Table of Contents

1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Markov Decision Process Framework](#markov-decision-process-framework)
3. [Core Components: States, Actions, and Rewards](#core-components-states-actions-and-rewards)
4. [Transition Probabilities and Dynamics](#transition-probabilities-and-dynamics)
5. [Policies and Policy Types](#policies-and-policy-types)
6. [Return and Discount Factor](#return-and-discount-factor)
7. [Bellman Expectation Equations](#bellman-expectation-equations)
8. [Bellman Optimality Equations](#bellman-optimality-equations)
9. [Solving MDPs: Value Iteration and Policy Iteration](#solving-mdps-value-iteration-and-policy-iteration)
10. [Key Takeaways](#key-takeaways)

## Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a computational approach to learning from interaction. An agent learns to make decisions by taking actions in an environment and receiving feedback in the form of rewards or penalties. Unlike supervised learning, RL does not require labeled examples of correct behavior. Instead, the agent discovers optimal actions through trial and error, guided by a reward signal.

The fundamental challenge in RL is the **credit assignment problem**: determining which actions led to which outcomes, especially when rewards are delayed. This temporal aspect distinguishes RL from other machine learning paradigms and requires sophisticated mathematical frameworks to model sequential decision-making.

The mathematical foundation of RL is built upon **Markov Decision Processes (MDPs)**, which provide a formal framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker.

## Markov Decision Process Framework

A **Markov Decision Process (MDP)** is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker. An MDP is formally defined as a tuple:

$$M = (S, A, P, R, \gamma)$$

where:
- $S$ is a finite set of states
- $A$ is a finite set of actions
- $P: S \times A \times S \rightarrow [0,1]$ is the transition probability function
- $R: S \times A \times S \rightarrow \mathbb{R}$ is the reward function
- $\gamma \in [0,1]$ is the discount factor

The **Markov property** states that the future depends only on the present state and action, not on the history of states and actions:

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

This property is crucial because it allows us to make decisions based solely on the current state, without needing to remember the entire history of interactions.

### MDP Assumptions

Several key assumptions underlie the MDP framework:

1. **Fully Observable**: The agent has complete information about the current state
2. **Markovian**: Future states depend only on the current state and action
3. **Stationary**: Transition probabilities and rewards do not change over time
4. **Discrete Time**: Actions are taken at discrete time steps

These assumptions, while restrictive, provide a tractable framework for analysis and algorithm development. Many real-world problems can be approximated as MDPs, even when they don't strictly satisfy all assumptions.

## Core Components: States, Actions, and Rewards

### States

A **state** $s \in S$ represents a complete description of the environment at a particular time. The state should contain all information necessary to make optimal decisions. In practice, states can be:

- **Discrete**: Finite set of distinct states (e.g., positions on a grid)
- **Continuous**: Real-valued vectors (e.g., joint angles of a robot)
- **Partially Observable**: Agent only sees observations, not true states

The **state space** $S$ can be finite or infinite, countable or uncountable. For finite MDPs, we typically work with discrete state spaces.

### Actions

An **action** $a \in A$ represents a decision made by the agent. Actions can be:

- **Discrete**: Finite set of choices (e.g., move left, right, up, down)
- **Continuous**: Real-valued vectors (e.g., torque applied to joints)
- **State-dependent**: Available actions may vary by state: $A(s) \subseteq A$

The **action space** $A$ defines all possible actions the agent can take. In many problems, the set of available actions depends on the current state.

### Rewards

The **reward function** $R(s, a, s')$ specifies the immediate reward received when transitioning from state $s$ to state $s'$ after taking action $a$. Rewards serve multiple purposes:

1. **Signal**: Indicate what is good or bad
2. **Objective**: Define the goal of learning
3. **Shaping**: Guide exploration and learning

Rewards can be:
- **Deterministic**: $R(s, a, s')$ is a fixed value
- **Stochastic**: $R(s, a, s')$ is a random variable with distribution $p(r | s, a, s')$

The **expected reward** for taking action $a$ in state $s$ is:

$$r(s, a) = \mathbb{E}[R_{t+1} | S_t = s, A_t = a] = \sum_{s'} P(s' | s, a) R(s, a, s')$$

## Transition Probabilities and Dynamics

The **transition probability function** $P(s' | s, a)$ specifies the probability of transitioning to state $s'$ given that action $a$ is taken in state $s$:

$$P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)$$

Transition probabilities satisfy:
- **Non-negativity**: $P(s' | s, a) \geq 0$ for all $s, s', a$
- **Normalization**: $\sum_{s'} P(s' | s, a) = 1$ for all $s, a$

The **dynamics** of an MDP are completely specified by the transition probabilities and reward function. Together, they define the **one-step dynamics**:

$$p(s', r | s, a) = \Pr(S_{t+1} = s', R_{t+1} = r | S_t = s, A_t = a)$$

This function captures both the probability of transitioning to state $s'$ and receiving reward $r$ when taking action $a$ in state $s$.

### State Transition Diagrams

For small MDPs, state transition diagrams provide visual representations of the dynamics. Each node represents a state, edges represent transitions labeled with actions and probabilities, and rewards are shown on transitions.

## Policies and Policy Types

A **policy** $\pi$ is a mapping from states to probability distributions over actions. It defines the agent's behavior:

$$\pi(a | s) = \Pr(A_t = a | S_t = s)$$

Policies can be:

### Deterministic Policies

A **deterministic policy** selects a single action in each state:

$$\pi(s) = a$$

where $\pi: S \rightarrow A$ maps states directly to actions.

### Stochastic Policies

A **stochastic policy** assigns probabilities to actions:

$$\pi(a | s) \in [0, 1], \quad \sum_{a} \pi(a | s) = 1$$

Stochastic policies are useful for:
- **Exploration**: Maintaining randomness to discover new strategies
- **Robustness**: Handling uncertainty and avoiding deterministic patterns
- **Optimality**: Some optimal policies are inherently stochastic

### Policy Space

The set of all possible policies is denoted $\Pi$. For finite MDPs with $|S|$ states and $|A|$ actions per state, there are $|A|^{|S|}$ deterministic policies and infinitely many stochastic policies.

## Return and Discount Factor

The **return** $G_t$ is the total discounted reward from time step $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

where $\gamma \in [0,1]$ is the **discount factor**.

### Discount Factor

The discount factor $\gamma$ serves several purposes:

1. **Mathematical Convenience**: Ensures finite returns for infinite horizons
2. **Temporal Preference**: Values immediate rewards more than delayed rewards
3. **Uncertainty**: Accounts for uncertainty about future rewards
4. **Economic Interpretation**: Represents interest rate or time preference

When $\gamma = 0$, the agent is **myopic** and only cares about immediate rewards. When $\gamma = 1$, the agent values all future rewards equally (requires finite horizon or absorbing states).

### Episodic vs Continuing Tasks

- **Episodic Tasks**: Have a terminal state, episodes end after $T$ steps
  - Return: $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$
- **Continuing Tasks**: No terminal state, infinite horizon
  - Return: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ (requires $\gamma < 1$)

## Bellman Expectation Equations

The **state-value function** $v_\pi(s)$ gives the expected return starting from state $s$ and following policy $\pi$:

$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \middle| S_t = s\right]$$

The **action-value function** $q_\pi(s, a)$ gives the expected return starting from state $s$, taking action $a$, and then following policy $\pi$:

$$q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### Bellman Equation for State Values

The state-value function satisfies the **Bellman expectation equation**:

$$v_\pi(s) = \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

This equation expresses the value of a state as the expected immediate reward plus the discounted value of the next state, averaged over all actions and next states according to the policy.

### Bellman Equation for Action Values

Similarly, the action-value function satisfies:

$$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \sum_{a'} \pi(a' | s') q_\pi(s', a')]$$

### Relationship Between Value Functions

The state-value and action-value functions are related:

$$v_\pi(s) = \sum_a \pi(a | s) q_\pi(s, a)$$

$$q_\pi(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

## Bellman Optimality Equations

An **optimal policy** $\pi_*$ is one that achieves the maximum expected return in all states:

$$v_{\pi_*}(s) \geq v_\pi(s) \quad \forall s \in S, \forall \pi$$

The **optimal state-value function** is:

$$v_*(s) = \max_\pi v_\pi(s)$$

The **optimal action-value function** is:

$$q_*(s, a) = \max_\pi q_\pi(s, a)$$

### Bellman Optimality Equation for State Values

The optimal state-value function satisfies:

$$v_*(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]$$

This equation states that the value of a state under an optimal policy equals the maximum expected return achievable by taking the best action in that state.

### Bellman Optimality Equation for Action Values

The optimal action-value function satisfies:

$$q_*(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \max_{a'} q_*(s', a')]$$

### Optimal Policy from Optimal Value Functions

Given $q_*(s, a)$, an optimal deterministic policy is:

$$\pi_*(s) = \arg\max_a q_*(s, a)$$

Any policy that is greedy with respect to $q_*$ is optimal. If multiple actions achieve the maximum, any of them can be chosen.

## Solving MDPs: Value Iteration and Policy Iteration

### Value Iteration

**Value iteration** directly computes the optimal value function by iteratively applying the Bellman optimality equation:

```python
def value_iteration(mdp, theta=1e-6):
    """
    Value iteration algorithm
    
    Args:
        mdp: MDP with states S, actions A, transitions P, rewards R, discount gamma
        theta: Convergence threshold
    
    Returns:
        Optimal value function v_*
    """
    V = {s: 0 for s in mdp.S}  # Initialize value function
    
    while True:
        V_new = {}
        delta = 0
        
        for s in mdp.S:
            # Bellman optimality update
            V_new[s] = max([
                sum(mdp.P(s, a, s_prime) * 
                    (mdp.R(s, a, s_prime) + mdp.gamma * V[s_prime])
                    for s_prime in mdp.S)
                for a in mdp.A(s)
            ])
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        if delta < theta:
            break
    
    return V
```

Value iteration converges to $v_*$ under the contraction mapping theorem.

### Policy Iteration

**Policy iteration** alternates between policy evaluation and policy improvement:

```python
def policy_iteration(mdp, theta=1e-6):
    """
    Policy iteration algorithm
    
    Args:
        mdp: MDP with states S, actions A, transitions P, rewards R, discount gamma
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
        policy_stable = True
        for s in mdp.S:
            old_action = pi[s]
            pi[s] = argmax([
                sum(mdp.P(s, a, s_prime) * 
                    (mdp.R(s, a, s_prime) + mdp.gamma * V[s_prime])
                    for s_prime in mdp.S)
                for a in mdp.A(s)
            ])
            if old_action != pi[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return pi
```

Policy iteration typically converges faster than value iteration but requires solving linear systems.

### Convergence Properties

Both algorithms converge to optimal policies:
- **Value Iteration**: Converges to $v_*$ in the limit
- **Policy Iteration**: Converges to $\pi_*$ in finite iterations (at most $|A|^{|S|}$)

The convergence rate depends on the discount factor $\gamma$: smaller $\gamma$ leads to faster convergence.

## Key Takeaways

1. **MDPs provide the mathematical foundation** for sequential decision-making under uncertainty, formalizing states, actions, rewards, transitions, and policies.

2. **The Markov property** enables tractable analysis by ensuring decisions depend only on the current state, not history.

3. **Value functions** ($v_\pi$ and $q_\pi$) quantify the long-term value of states and actions under a given policy, enabling policy evaluation and comparison.

4. **Bellman expectation equations** express recursive relationships for value functions, forming the basis for dynamic programming and temporal difference learning.

5. **Bellman optimality equations** characterize optimal value functions and policies, providing the foundation for optimal control algorithms.

6. **Optimal policies** can be derived from optimal value functions by taking greedy actions with respect to $q_*$.

7. **Value iteration and policy iteration** are fundamental algorithms for solving MDPs when the model is known, with different trade-offs in convergence speed and computational cost.

8. **The discount factor** $\gamma$ balances immediate and future rewards, ensuring mathematical tractability and modeling temporal preferences.

9. **Policies can be deterministic or stochastic**, with stochastic policies often necessary for exploration and optimality in certain environments.

10. **The MDP framework** extends to continuous states/actions, partial observability, and multi-agent settings, forming the basis for advanced RL algorithms.
