# Multi-Agent Reinforcement Learning

## Table of Contents

1. [Introduction to Multi-Agent RL](#introduction-to-multi-agent-rl)
2. [MARL Formulation](#marl-formulation)
3. [Cooperative vs Competitive Settings](#cooperative-vs-competitive-settings)
4. [Independent Learners](#independent-learners)
5. [Centralized Training with Decentralized Execution](#centralized-training-with-decentralized-execution)
6. [QMIX: Value Function Factorization](#qmix-value-function-factorization)
7. [MAPPO: Multi-Agent PPO](#mappo-multi-agent-ppo)
8. [Communication in Multi-Agent Systems](#communication-in-multi-agent-systems)
9. [Emergent Behavior and Coordination](#emergent-behavior-and-coordination)
10. [Key Takeaways](#key-takeaways)

## Introduction to Multi-Agent Reinforcement Learning

**Multi-Agent Reinforcement Learning (MARL)** extends RL to settings with multiple agents interacting in a shared environment. Each agent learns a policy to maximize its own (or shared) reward, leading to complex dynamics including cooperation, competition, and coordination.

### Challenges

1. **Non-Stationarity**: Environment changes as other agents learn
2. **Credit Assignment**: Which agent deserves credit for outcomes?
3. **Coordination**: How to coordinate actions?
4. **Scalability**: Complexity grows with number of agents
5. **Communication**: How to share information?

### Applications

- **Robotics**: Multi-robot coordination
- **Game AI**: Team-based games
- **Autonomous Vehicles**: Traffic management
- **Economics**: Market dynamics
- **Distributed Systems**: Resource allocation

## MARL Formulation

### Multi-Agent MDP

A **Multi-Agent MDP (MAMDP)** extends MDPs:

$$M = (N, S, \{A_i\}_{i=1}^N, P, \{R_i\}_{i=1}^N, \gamma)$$

where:
- $N$ is the number of agents
- $S$ is the state space
- $A_i$ is the action space for agent $i$
- $P: S \times A_1 \times \cdots \times A_N \times S \rightarrow [0,1]$ is the transition function
- $R_i: S \times A_1 \times \cdots \times A_N \times S \rightarrow \mathbb{R}$ is the reward for agent $i$

### Joint Action Space

The **joint action space** is:

$$\mathbf{A} = A_1 \times A_2 \times \cdots \times A_N$$

A **joint action** is $\mathbf{a} = (a_1, a_2, \ldots, a_N)$.

### Policies

Each agent $i$ has a policy $\pi_i: S \times A_i \rightarrow [0,1]$. The **joint policy** is:

$$\boldsymbol{\pi}(\mathbf{a} | s) = \prod_{i=1}^N \pi_i(a_i | s)$$

## Cooperative vs Competitive Settings

### Cooperative MARL

In **cooperative settings**, agents share a common reward:

$$R_1(s, \mathbf{a}, s') = R_2(s, \mathbf{a}, s') = \cdots = R_N(s, \mathbf{a}, s') = R(s, \mathbf{a}, s')$$

Goal: Maximize shared return.

### Competitive MARL

In **competitive settings**, agents have opposing objectives:

$$\sum_{i=1}^N R_i(s, \mathbf{a}, s') = 0$$

This is a **zero-sum game**.

### Mixed Settings

**Mixed settings** combine cooperation and competition:
- **Coalitions**: Groups of cooperating agents
- **Team Competition**: Teams compete against each other
- **General-Sum Games**: Arbitrary reward structures

## Independent Learners

**Independent Learners** treat other agents as part of the environment, learning independently.

### Independent Q-Learning (IQL)

Each agent learns its own Q-function:

$$Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha [r_i + \gamma \max_{a_i'} Q_i(s', a_i') - Q_i(s, a_i)]$$

### Issues

1. **Non-Stationarity**: Other agents' policies change
2. **Partial Observability**: May not observe other agents' actions
3. **Coordination**: No explicit coordination mechanism

### Implementation

```python
class IndependentQLearning:
    def __init__(self, num_agents, state_dim, action_dims, lr=0.001):
        """
        Independent Q-Learning
        
        Args:
            num_agents: Number of agents
            state_dim: State dimension
            action_dims: List of action dimensions per agent
            lr: Learning rate
        """
        self.num_agents = num_agents
        self.agents = []
        
        for i in range(num_agents):
            agent = QLearningAgent(state_dim, action_dims[i], lr)
            self.agents.append(agent)
    
    def select_actions(self, state):
        """Select actions for all agents"""
        actions = []
        for agent in self.agents:
            action = agent.select_action(state)
            actions.append(action)
        return actions
    
    def update(self, state, actions, rewards, next_state, done):
        """Update all agents"""
        for i, agent in enumerate(self.agents):
            agent.update(state, actions[i], rewards[i], next_state, done)
```

## Centralized Training with Decentralized Execution

**CTDE (Centralized Training with Decentralized Execution)** uses centralized information during training but decentralized policies during execution.

### Centralized Critic

During training, use a **centralized critic** that sees all agents' observations and actions:

$$Q_i^{\text{tot}}(s, \mathbf{a}) = f(Q_1(s_1, a_1), \ldots, Q_N(s_N, a_N))$$

During execution, each agent uses only its local observation.

### MADDPG

**MADDPG (Multi-Agent DDPG)** extends DDPG to multi-agent settings:

```python
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, agent_id, num_agents):
        """
        MADDPG Agent
        
        Args:
            state_dim: Local state dimension
            action_dim: Action dimension
            agent_id: Agent identifier
            num_agents: Total number of agents
        """
        # Actor (decentralized)
        self.actor = Actor(state_dim, action_dim)
        
        # Critic (centralized, sees all observations and actions)
        critic_input_dim = num_agents * (state_dim + action_dim)
        self.critic = Critic(critic_input_dim, 1)
        
        # Target networks
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(critic_input_dim, 1)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def select_action(self, state):
        """Select action using actor (decentralized)"""
        return self.actor(state)
    
    def update(self, states, actions, rewards, next_states, dones):
        """
        Update using centralized critic
        
        Args:
            states: All agents' states
            actions: All agents' actions
            rewards: All agents' rewards
            next_states: All agents' next states
            dones: Done flags
        """
        # Flatten for centralized critic
        critic_input = torch.cat(states + actions, dim=1)
        next_critic_input = torch.cat(next_states + [
            self.actor_target(s) for s in next_states
        ], dim=1)
        
        # Critic update
        with torch.no_grad():
            target_q = rewards[self.agent_id] + self.gamma * (
                1 - dones
            ) * self.critic_target(next_critic_input)
        
        current_q = self.critic(critic_input)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Actor update
        actor_loss = -self.critic(critic_input).mean()
        
        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
```

## QMIX: Value Function Factorization

**QMIX** factors the joint Q-function into individual Q-functions with a monotonic mixing function.

### Factorization

$$Q_{\text{tot}}(s, \mathbf{a}) = f(Q_1(s_1, a_1), \ldots, Q_N(s_N, a_N))$$

where $f$ is a **monotonic mixing function**:

$$\frac{\partial f}{\partial Q_i} \geq 0 \quad \forall i$$

This ensures that:

$$\arg\max_{\mathbf{a}} Q_{\text{tot}}(s, \mathbf{a}) = (\arg\max_{a_1} Q_1(s_1, a_1), \ldots, \arg\max_{a_N} Q_N(s_N, a_N))$$

### Mixing Network

```python
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim=64):
        """
        QMIX Mixing Network
        
        Args:
            num_agents: Number of agents
            state_dim: Global state dimension
            hidden_dim: Hidden dimension
        """
        super(MixingNetwork, self).__init__()
        
        self.num_agents = num_agents
        
        # Hypernetworks for mixing weights
        self.hyper_w1 = nn.Linear(state_dim, num_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        
        # Bias
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values, state):
        """
        Mix Q-values
        
        Args:
            q_values: Individual Q-values (batch_size, num_agents)
            state: Global state (batch_size, state_dim)
        
        Returns:
            Mixed Q-value (batch_size, 1)
        """
        # First layer
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(-1, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(state)
        b1 = b1.view(-1, 1, self.hidden_dim)
        
        # Mix
        q_values = q_values.view(-1, 1, self.num_agents)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(-1, self.hidden_dim, 1)
        b2 = self.hyper_b2(state)
        
        # Output
        q_tot = torch.bmm(hidden, w2) + b2.view(-1, 1, 1)
        
        return q_tot.squeeze()

class QMIX:
    def __init__(self, num_agents, state_dims, action_dims, 
                 global_state_dim):
        """
        QMIX algorithm
        
        Args:
            num_agents: Number of agents
            state_dims: List of state dimensions per agent
            action_dims: List of action dimensions per agent
            global_state_dim: Global state dimension
        """
        # Individual Q-networks
        self.q_networks = nn.ModuleList([
            QNetwork(state_dims[i], action_dims[i]) 
            for i in range(num_agents)
        ])
        
        # Mixing network
        self.mixing_network = MixingNetwork(
            num_agents, global_state_dim
        )
        
        # Target networks
        self.target_q_networks = nn.ModuleList([
            QNetwork(state_dims[i], action_dims[i]) 
            for i in range(num_agents)
        ])
        self.target_mixing_network = MixingNetwork(
            num_agents, global_state_dim
        )
        
        for i in range(num_agents):
            self.target_q_networks[i].load_state_dict(
                self.q_networks[i].state_dict()
            )
        self.target_mixing_network.load_state_dict(
            self.mixing_network.state_dict()
        )
    
    def compute_q_tot(self, states, actions, global_state, use_target=False):
        """Compute total Q-value"""
        q_networks = self.target_q_networks if use_target else self.q_networks
        mixing_network = (self.target_mixing_network if use_target 
                        else self.mixing_network)
        
        # Individual Q-values
        q_values = []
        for i, (state, action) in enumerate(zip(states, actions)):
            q = q_networks[i](state)
            q = q.gather(1, action.unsqueeze(1))
            q_values.append(q)
        
        q_values = torch.cat(q_values, dim=1)
        
        # Mix
        q_tot = mixing_network(q_values, global_state)
        
        return q_tot
```

## MAPPO: Multi-Agent PPO

**MAPPO** extends PPO to multi-agent settings with centralized value functions.

### Architecture

```python
class MAPPOAgent:
    def __init__(self, state_dim, action_dim, agent_id, num_agents):
        """
        MAPPO Agent
        
        Args:
            state_dim: Local state dimension
            action_dim: Action dimension
            agent_id: Agent identifier
            num_agents: Total number of agents
        """
        # Actor (decentralized)
        self.actor = PolicyNetwork(state_dim, action_dim)
        
        # Critic (centralized)
        critic_input_dim = num_agents * state_dim
        self.critic = ValueNetwork(critic_input_dim)
    
    def update(self, states, actions, old_log_probs, 
              advantages, returns, global_states):
        """
        MAPPO update
        
        Args:
            states: Local states
            actions: Actions
            old_log_probs: Old log probabilities
            advantages: Advantages
            returns: Returns
            global_states: Global states (for critic)
        """
        # Actor update (decentralized)
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic update (centralized)
        values = self.critic(global_states)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

## Communication in Multi-Agent Systems

**Communication** enables agents to share information and coordinate.

### Communication Protocols

1. **Fixed Protocol**: Predefined communication structure
2. **Learned Protocol**: Agents learn what to communicate
3. **Attention-Based**: Use attention to focus on relevant information

### CommNet

**CommNet** uses learned communication:

```python
class CommNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_agents):
        """
        CommNet with learned communication
        
        Args:
            input_dim: Input dimension per agent
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_agents: Number of agents
        """
        super(CommNet, self).__init__()
        
        self.num_agents = num_agents
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Communication layers
        self.comm_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, inputs):
        """
        Forward pass with communication
        
        Args:
            inputs: Inputs for each agent (batch_size, num_agents, input_dim)
        
        Returns:
            Outputs for each agent (batch_size, num_agents, output_dim)
        """
        # Encode
        hidden = F.relu(self.encoder(inputs))
        
        # Communication rounds
        for comm_layer in self.comm_layers:
            # Average messages from other agents
            messages = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            hidden = F.relu(comm_layer(hidden + messages))
        
        # Decode
        outputs = self.decoder(hidden)
        
        return outputs
```

## Emergent Behavior and Coordination

### Emergent Behaviors

Multi-agent systems can exhibit **emergent behaviors**:
- **Formation**: Agents form patterns
- **Division of Labor**: Specialization
- **Coordination**: Synchronized actions
- **Altruism**: Self-sacrifice for team

### Coordination Mechanisms

1. **Explicit Communication**: Direct message passing
2. **Implicit Communication**: Through actions
3. **Shared Representations**: Common knowledge
4. **Reward Shaping**: Encourage coordination

### Example: Emergent Communication

```python
class EmergentCommunication:
    def __init__(self, num_agents, vocab_size=10):
        """
        Agents learn to communicate
        
        Args:
            num_agents: Number of agents
            vocab_size: Size of communication vocabulary
        """
        # Communication networks
        self.speakers = nn.ModuleList([
            nn.Linear(state_dim, vocab_size) for _ in range(num_agents)
        ])
        
        self.listeners = nn.ModuleList([
            nn.Linear(state_dim + vocab_size, action_dim) 
            for _ in range(num_agents)
        ])
    
    def communicate(self, states):
        """Agents communicate and act"""
        messages = []
        
        # Generate messages
        for i, speaker in enumerate(self.speakers):
            message_logits = speaker(states[i])
            message = F.gumbel_softmax(message_logits, hard=True)
            messages.append(message)
        
        # Use messages for actions
        actions = []
        for i, listener in enumerate(self.listeners):
            # Concatenate state and received messages
            other_messages = torch.cat([
                m for j, m in enumerate(messages) if j != i
            ], dim=-1)
            listener_input = torch.cat([states[i], other_messages], dim=-1)
            action = listener(listener_input)
            actions.append(action)
        
        return actions, messages
```

## Key Takeaways

1. **Multi-Agent RL** extends RL to settings with multiple agents, introducing challenges like non-stationarity and coordination.

2. **MARL Formulation** extends MDPs to multi-agent settings with joint action spaces and individual or shared rewards.

3. **Cooperative vs Competitive** settings determine whether agents share rewards (cooperative) or have opposing objectives (competitive).

4. **Independent Learners** treat other agents as part of the environment but suffer from non-stationarity and lack coordination.

5. **CTDE (Centralized Training with Decentralized Execution)** uses centralized information during training but decentralized policies during execution, enabling better coordination.

6. **QMIX** factors the joint Q-function into individual Q-functions with a monotonic mixing function, ensuring optimal individual actions lead to optimal joint actions.

7. **MAPPO** extends PPO to multi-agent settings with centralized value functions while maintaining decentralized policies.

8. **Communication** enables agents to share information and coordinate, with learned communication protocols emerging from training.

9. **Emergent Behavior** arises from multi-agent interactions, including formation, division of labor, and coordination.

10. **Multi-Agent RL** provides a framework for understanding and designing systems with multiple interacting agents, with applications in robotics, game AI, and distributed systems.
