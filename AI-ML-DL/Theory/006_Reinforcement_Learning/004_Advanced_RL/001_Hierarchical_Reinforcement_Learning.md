# Hierarchical Reinforcement Learning

## Table of Contents

1. [Introduction to Hierarchical RL](#introduction-to-hierarchical-rl)
2. [Options Framework](#options-framework)
3. [Semi-Markov Decision Processes](#semi-markov-decision-processes)
4. [Goal-Conditioned Reinforcement Learning](#goal-conditioned-reinforcement-learning)
5. [Feudal Networks](#feudal-networks)
6. [HIRO: Hierarchical Reinforcement Learning with Off-Policy Correction](#hiro-hierarchical-reinforcement-learning-with-off-policy-correction)
7. [Option Discovery and Learning](#option-discovery-and-learning)
8. [Skill Learning and Composition](#skill-learning-and-composition)
9. [Implementation and Practical Considerations](#implementation-and-practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Hierarchical Reinforcement Learning

**Hierarchical Reinforcement Learning (HRL)** decomposes complex tasks into simpler subtasks, enabling more efficient learning and better generalization. Instead of learning a single flat policy, HRL learns policies at multiple levels of abstraction.

### Motivation

Standard RL faces challenges:
1. **Sparse Rewards**: Rewards may be delayed or sparse
2. **Long Horizons**: Need to reason about long sequences of actions
3. **Sample Inefficiency**: Many samples needed for complex tasks
4. **Transfer**: Hard to transfer knowledge across tasks

HRL addresses these by:
1. **Abstraction**: Learning at multiple levels
2. **Temporal Abstraction**: Actions that last multiple steps
3. **Reusability**: Skills can be reused across tasks
4. **Transfer**: Hierarchical structure enables transfer

### Key Concepts

- **Options**: Temporally extended actions
- **Skills**: Reusable behaviors
- **Subgoals**: Intermediate goals
- **Hierarchy**: Multiple levels of policies

## Options Framework

**Options** are temporally extended actions that encapsulate policies over sequences of primitive actions.

### Option Definition

An option $o$ is a tuple $(I_o, \pi_o, \beta_o)$:
- **Initiation Set** $I_o \subseteq S$: States where option can start
- **Policy** $\pi_o: S \times A \rightarrow [0,1]$: Policy executed by option
- **Termination Condition** $\beta_o: S \rightarrow [0,1]$: Probability of terminating

### Option Execution

When an option $o$ is executed:
1. Start in state $s \in I_o$
2. Follow policy $\pi_o$ until termination
3. Terminate with probability $\beta_o(s)$

### Option Value Function

The value of an option:

$$Q(s, o) = \mathbb{E} \left[\sum_{t=0}^T \gamma^t r_{t+1} + \gamma^T Q(s_T, o') | s_0 = s, o\right]$$

where $T$ is the termination time and $o'$ is the next option.

### Option-Critic Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptionCritic(nn.Module):
    def __init__(self, state_dim, num_options, num_actions):
        """
        Option-Critic architecture
        
        Args:
            state_dim: State dimension
            num_options: Number of options
            num_actions: Number of primitive actions
        """
        super(OptionCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Option policy over options
        self.option_policy = nn.Linear(128, num_options)
        
        # Intra-option policies (one per option)
        self.intra_option_policies = nn.ModuleList([
            nn.Linear(128, num_actions) for _ in range(num_options)
        ])
        
        # Termination functions (one per option)
        self.termination_functions = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_options)
        ])
        
        # Q-function over options
        self.q_options = nn.Linear(128, num_options)
    
    def forward(self, state):
        """Forward pass"""
        features = self.feature_extractor(state)
        
        # Option policy
        option_logits = self.option_policy(features)
        option_probs = F.softmax(option_logits, dim=-1)
        
        # Intra-option policies
        intra_option_logits = [policy(features) 
                              for policy in self.intra_option_policies]
        intra_option_probs = [F.softmax(logits, dim=-1) 
                             for logits in intra_option_logits]
        
        # Termination probabilities
        termination_probs = [torch.sigmoid(term(features)) 
                             for term in self.termination_functions]
        
        # Q-values
        q_values = self.q_options(features)
        
        return {
            'option_probs': option_probs,
            'intra_option_probs': intra_option_probs,
            'termination_probs': termination_probs,
            'q_values': q_values
        }
```

## Semi-Markov Decision Processes

**Semi-MDPs** extend MDPs to handle temporally extended actions (options).

### SMDP Definition

A Semi-MDP is defined by:
- States $S$
- Options $O$ (instead of primitive actions)
- Transition probabilities $P(s' | s, o)$
- Rewards $R(s, o)$
- Discount factor $\gamma$

### SMDP Bellman Equations

The value function for options:

$$V(s) = \max_{o \in O_s} \left[R(s, o) + \sum_{s'} P(s' | s, o) \gamma^{\tau(s, o)} V(s')\right]$$

where $\tau(s, o)$ is the duration of option $o$ in state $s$.

### Option Value Function

$$Q(s, o) = R(s, o) + \sum_{s'} P(s' | s, o) \gamma^{\tau(s, o)} \max_{o'} Q(s', o')$$

## Goal-Conditioned Reinforcement Learning

**Goal-Conditioned RL** learns policies that achieve specific goals, enabling skill composition and transfer.

### Goal-Conditioned Policy

A goal-conditioned policy $\pi(a | s, g)$ takes both state $s$ and goal $g$ as input:

```python
class GoalConditionedPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        """
        Goal-conditioned policy
        
        Args:
            state_dim: State dimension
            goal_dim: Goal dimension
            action_dim: Action dimension
        """
        super(GoalConditionedPolicy, self).__init__()
        
        # Concatenate state and goal
        input_dim = state_dim + goal_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, goal):
        """Forward pass"""
        state_goal = torch.cat([state, goal], dim=-1)
        return self.network(state_goal)
```

### Goal-Conditioned Value Function

$$Q(s, a, g) = \mathbb{E} \left[\sum_{t=0}^T \gamma^t r_t | s_0 = s, a_0 = a, g\right]$$

where the reward depends on achieving goal $g$.

### Universal Value Function Approximators (UVFA)

UVFAs approximate value functions for any goal:

$$Q(s, a, g) \approx Q_\theta(s, a, g)$$

This enables generalization across goals.

## Feudal Networks

**Feudal Networks** use a manager-worker hierarchy where:
- **Manager**: Sets subgoals at lower temporal resolution
- **Worker**: Achieves subgoals at higher temporal resolution

### Architecture

```python
class FeudalNetwork(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, 
                 manager_steps=10):
        """
        Feudal Network
        
        Args:
            state_dim: State dimension
            goal_dim: Goal dimension
            action_dim: Action dimension
            manager_steps: Steps between manager updates
        """
        super(FeudalNetwork, self).__init__()
        
        # Manager network (sets subgoals)
        self.manager = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, goal_dim)
        )
        
        # Worker network (achieves subgoals)
        self.worker = GoalConditionedPolicy(state_dim, goal_dim, action_dim)
        
        # Internal state for manager
        self.manager_state = None
        self.manager_steps = manager_steps
        self.step_count = 0
    
    def forward(self, state):
        """Forward pass"""
        # Manager sets subgoal every manager_steps
        if self.step_count % self.manager_steps == 0:
            self.manager_state = self.manager(state)
        
        # Worker uses current subgoal
        action = self.worker(state, self.manager_state)
        
        self.step_count += 1
        return action
```

## HIRO: Hierarchical Reinforcement Learning with Off-Policy Correction

**HIRO** learns hierarchical policies with off-policy correction to handle non-stationarity in hierarchical learning.

### Problem

In hierarchical RL, the high-level policy's actions (subgoals) change the environment for the low-level policy, making the low-level policy's experience non-stationary.

### Solution

HIRO uses **off-policy correction** to relabel subgoals in past experiences:

```python
def hiro_relabel(high_level_transition, low_level_policy, 
                state_space, goal_space):
    """
    HIRO relabeling for off-policy correction
    
    Args:
        high_level_transition: (s, g, s', g', r)
        low_level_policy: Low-level policy
        state_space: State space
        goal_space: Goal space
    """
    s, g, s_next, g_next, r = high_level_transition
    
    # Compute achieved goal
    achieved_goal = compute_achieved_goal(s, s_next)
    
    # Relabel subgoal to be closer to achieved goal
    # This makes the experience more on-policy
    relabeled_goal = relabel_subgoal(
        g, achieved_goal, low_level_policy, state_space, goal_space
    )
    
    return (s, relabeled_goal, s_next, g_next, r)

def relabel_subgoal(original_goal, achieved_goal, policy, 
                   state_space, goal_space):
    """
    Relabel subgoal to be achievable
    
    Args:
        original_goal: Original subgoal
        achieved_goal: Actually achieved goal
        policy: Low-level policy
        state_space: State space
        goal_space: Goal space
    """
    # Find subgoal that is:
    # 1. Close to achieved goal (achievable)
    # 2. Makes progress toward original goal
    # 3. Is on-policy for current low-level policy
    
    # Simplified: use achieved goal if close enough
    if distance(original_goal, achieved_goal) < threshold:
        return achieved_goal
    else:
        # Interpolate between original and achieved
        return 0.5 * original_goal + 0.5 * achieved_goal
```

### HIRO Algorithm

```python
class HIRO:
    def __init__(self, state_dim, goal_dim, action_dim):
        """
        HIRO agent
        
        Args:
            state_dim: State dimension
            goal_dim: Goal dimension
            action_dim: Action dimension
        """
        # High-level policy (manager)
        self.high_level_policy = GoalConditionedPolicy(
            state_dim, goal_dim, goal_dim  # Outputs subgoals
        )
        
        # Low-level policy (worker)
        self.low_level_policy = GoalConditionedPolicy(
            state_dim, goal_dim, action_dim
        )
        
        # Replay buffers
        self.high_level_buffer = ReplayBuffer()
        self.low_level_buffer = ReplayBuffer()
    
    def train(self, batch_size=64):
        """Train both levels"""
        # Train low-level policy
        low_batch = self.low_level_buffer.sample(batch_size)
        self.train_low_level(low_batch)
        
        # Train high-level policy with relabeling
        high_batch = self.high_level_buffer.sample(batch_size)
        relabeled_batch = [hiro_relabel(t, self.low_level_policy) 
                          for t in high_batch]
        self.train_high_level(relabeled_batch)
```

## Option Discovery and Learning

**Option Discovery** automatically discovers useful options/skills without manual specification.

### Diversity-Based Discovery

Encourage diverse behaviors:

```python
def diversity_based_discovery(policy, state, num_options=10):
    """
    Discover options by encouraging diversity
    
    Args:
        policy: Policy network
        state: Current state
        num_options: Number of options to discover
    """
    options = []
    
    for i in range(num_options):
        # Sample diverse option
        option = sample_diverse_option(policy, state, options)
        options.append(option)
    
    return options

def sample_diverse_option(policy, state, existing_options):
    """Sample option different from existing ones"""
    # Maximize diversity while maintaining performance
    option = None
    max_diversity = -float('inf')
    
    for _ in range(100):  # Sample candidates
        candidate = policy.sample_option(state)
        diversity = compute_diversity(candidate, existing_options)
        
        if diversity > max_diversity:
            max_diversity = diversity
            option = candidate
    
    return option
```

### Skill Learning

Learn skills that are:
1. **Useful**: Achieve subgoals
2. **Diverse**: Cover different behaviors
3. **Composable**: Can be combined

```python
class SkillLearner:
    def __init__(self, state_dim, skill_dim, action_dim):
        """
        Skill learning
        
        Args:
            state_dim: State dimension
            skill_dim: Skill embedding dimension
            action_dim: Action dimension
        """
        # Skill-conditioned policy
        self.skill_policy = GoalConditionedPolicy(
            state_dim, skill_dim, action_dim
        )
        
        # Skill discriminator (for diversity)
        self.skill_discriminator = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, skill_dim)
        )
    
    def learn_skills(self, states, actions, num_skills=10):
        """
        Learn diverse skills
        
        Args:
            states: States
            actions: Actions
            num_skills: Number of skills
        """
        # Maximize mutual information between skills and states
        # while maintaining diversity
        
        for skill_id in range(num_skills):
            # Learn skill that is:
            # 1. Predictive of state distribution
            # 2. Different from other skills
            
            skill_embedding = self.sample_skill_embedding()
            
            # Train skill policy
            skill_loss = self.train_skill_policy(
                skill_embedding, states, actions
            )
            
            # Train discriminator for diversity
            discriminator_loss = self.train_discriminator(
                skill_embedding, states
            )
```

## Skill Learning and Composition

### Skill Composition

Compose skills to solve complex tasks:

```python
def compose_skills(skills, task):
    """
    Compose skills to solve task
    
    Args:
        skills: List of learned skills
        task: Task to solve
    """
    # Plan sequence of skills
    skill_sequence = plan_skill_sequence(skills, task)
    
    # Execute skills in sequence
    state = task.initial_state
    
    for skill in skill_sequence:
        # Execute skill until completion
        while not skill.is_complete(state):
            action = skill.policy(state)
            state = task.step(action)
        
        # Check if task is solved
        if task.is_solved(state):
            break
    
    return state
```

### Skill Transfer

Transfer skills across tasks:

```python
def transfer_skills(source_task, target_task, skills):
    """
    Transfer skills from source to target task
    
    Args:
        source_task: Source task
        target_task: Target task
        skills: Learned skills
    """
    # Map skills to target task
    transferred_skills = []
    
    for skill in skills:
        # Adapt skill to target task
        adapted_skill = adapt_skill(skill, source_task, target_task)
        transferred_skills.append(adapted_skill)
    
    return transferred_skills
```

## Implementation and Practical Considerations

### Hierarchical Training

Train multiple levels:
1. **Bottom-Up**: Learn low-level skills first
2. **Top-Down**: Learn high-level policy using skills
3. **Joint**: Train both levels simultaneously

### Challenges

1. **Non-Stationarity**: Low-level policy sees non-stationary environment
2. **Credit Assignment**: Which level gets credit?
3. **Option Discovery**: How to discover useful options?
4. **Composition**: How to compose skills?

### Best Practices

1. **Start Simple**: Begin with hand-designed options
2. **Gradual Complexity**: Increase complexity gradually
3. **Monitor**: Track performance at each level
4. **Transfer**: Leverage learned skills across tasks

## Key Takeaways

1. **Hierarchical RL** decomposes complex tasks into simpler subtasks, enabling more efficient learning and better generalization.

2. **Options Framework** provides temporally extended actions that encapsulate policies over sequences of primitive actions.

3. **Semi-MDPs** extend MDPs to handle options, with modified Bellman equations accounting for option duration.

4. **Goal-Conditioned RL** learns policies that achieve specific goals, enabling skill composition and transfer through universal value function approximators.

5. **Feudal Networks** use a manager-worker hierarchy where the manager sets subgoals and the worker achieves them.

6. **HIRO** addresses non-stationarity in hierarchical learning through off-policy correction by relabeling subgoals in past experiences.

7. **Option Discovery** automatically discovers useful options through diversity-based methods and skill learning techniques.

8. **Skill Composition** enables solving complex tasks by combining simpler skills, with transfer learning enabling skills to be reused across tasks.

9. **Implementation** requires careful handling of non-stationarity, credit assignment, and option discovery, with best practices including gradual complexity increase and performance monitoring.

10. **Hierarchical RL** provides a powerful framework for tackling complex, long-horizon tasks by learning reusable skills and hierarchical policies.
