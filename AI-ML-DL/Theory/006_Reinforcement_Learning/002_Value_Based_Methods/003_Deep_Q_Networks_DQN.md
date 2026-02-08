# Deep Q-Networks (DQN)

## Table of Contents

1. [Introduction to Deep Q-Networks](#introduction-to-deep-q-networks)
2. [DQN Architecture and Components](#dqn-architecture-and-components)
3. [Experience Replay](#experience-replay)
4. [Target Networks](#target-networks)
5. [Double DQN](#double-dqn)
6. [Dueling DQN](#dueling-dqn)
7. [Prioritized Experience Replay](#prioritized-experience-replay)
8. [Training and Implementation Details](#training-and-implementation-details)
9. [Challenges and Limitations](#challenges-and-limitations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Deep Q-Networks

**Deep Q-Networks (DQN)** combine Q-Learning with deep neural networks to handle high-dimensional state spaces. Introduced by Mnih et al. in 2015, DQN achieved human-level performance on many Atari games using only raw pixel inputs.

The key innovation is using a deep convolutional neural network to approximate the Q-function:

$$Q(s, a; \theta) \approx q_*(s, a)$$

where $\theta$ are the network parameters.

### Challenges Addressed

DQN addresses several challenges in combining deep learning with RL:

1. **Correlated Samples**: Sequential samples are highly correlated
2. **Non-Stationary Targets**: Q-function targets change during learning
3. **Scalability**: Need to handle high-dimensional inputs (e.g., images)
4. **Stability**: Deep RL is prone to instability and divergence

### Key Contributions

DQN introduced two critical techniques:
- **Experience Replay**: Store and sample from past experiences
- **Target Networks**: Use separate network for computing targets

## DQN Architecture and Components

### Network Architecture

The DQN architecture for Atari games:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        DQN network architecture
        
        Args:
            input_shape: Shape of input (C, H, W) for images
            num_actions: Number of possible actions
        """
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Compute feature size
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def _get_conv_out_size(self, shape):
        """Compute convolutional output size"""
        x = torch.zeros(1, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input state (batch_size, C, H, W)
        
        Returns:
            Q-values for each action (batch_size, num_actions)
        """
        x = x / 255.0  # Normalize pixel values
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
```

### Input Preprocessing

For Atari games, preprocessing includes:

```python
def preprocess_frame(frame):
    """
    Preprocess Atari frame
    
    Args:
        frame: Raw frame (210, 160, 3)
    
    Returns:
        Preprocessed frame (84, 84, 1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84))
    
    # Normalize
    normalized = resized / 255.0
    
    return normalized

def stack_frames(frames, num_frames=4):
    """
    Stack multiple frames for temporal information
    
    Args:
        frames: List of frames
        num_frames: Number of frames to stack
    
    Returns:
        Stacked frames (num_frames, 84, 84)
    """
    if len(frames) < num_frames:
        # Pad with first frame
        frames = [frames[0]] * (num_frames - len(frames)) + frames
    
    return np.stack(frames[-num_frames:], axis=0)
```

## Experience Replay

**Experience Replay** stores agent experiences in a replay buffer and samples batches randomly for training, breaking correlations between consecutive samples.

### Replay Buffer

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        Experience replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample random batch from buffer
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Batch of experiences
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
```

### Benefits of Experience Replay

1. **Data Efficiency**: Each experience used multiple times
2. **Reduced Correlation**: Random sampling breaks temporal correlations
3. **Stability**: More stable learning from diverse experiences
4. **Off-Policy Learning**: Can learn from past policies' experiences

### Algorithm with Experience Replay

```python
def dqn_with_replay(env, q_network, target_network, replay_buffer, 
                    optimizer, batch_size=32, gamma=0.99):
    """
    DQN training with experience replay
    
    Args:
        env: Environment
        q_network: Main Q-network
        target_network: Target Q-network
        replay_buffer: Experience replay buffer
        optimizer: Optimizer for q_network
        batch_size: Batch size for training
        gamma: Discount factor
    """
    # Collect experience
    state = env.reset()
    action = select_action(q_network, state)
    next_state, reward, done = env.step(action)
    
    # Store in replay buffer
    replay_buffer.push(state, action, reward, next_state, done)
    
    # Sample batch from replay buffer
    if len(replay_buffer) >= batch_size:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Compute current Q-values
        current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Target Networks

**Target Networks** are separate copies of the Q-network used to compute targets, updated less frequently to stabilize learning.

### Why Target Networks?

Without target networks, the Q-learning update is:

$$Q(s, a; \theta_t) \leftarrow Q(s, a; \theta_t) + \alpha [r + \gamma \max_{a'} Q(s', a'; \theta_t) - Q(s, a; \theta_t)]$$

The target $r + \gamma \max_{a'} Q(s', a'; \theta_t)$ changes at every step as $\theta_t$ changes, making learning unstable.

### Target Network Solution

Use a separate target network with parameters $\theta^-$:

$$Q(s, a; \theta_t) \leftarrow Q(s, a; \theta_t) + \alpha [r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta_t)]$$

Update target network periodically by copying main network:

$$\theta^- \leftarrow \theta_t \quad \text{(every C steps)}$$

### Implementation

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, 
                 target_update_freq=1000):
        """
        DQN Agent with target network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            lr: Learning rate
            gamma: Discount factor
            target_update_freq: Frequency of target network updates
        """
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.steps = 0
    
    def update_target_network(self):
        """Copy main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self, batch):
        """
        Single training step
        
        Args:
            batch: Batch of experiences
        """
        states, actions, rewards, next_states, dones = batch
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values using target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
```

## Double DQN

**Double DQN** addresses the overestimation bias in Q-Learning by using the main network to select actions and the target network to evaluate them.

### Overestimation Bias

Standard DQN uses:

$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$

The $\max$ operator causes overestimation because it always picks the maximum, which may be an overestimate.

### Double DQN Solution

Double DQN decouples action selection from evaluation:

$$Y_t^{\text{Double DQN}} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta_t); \theta^-)$$

- Use main network $\theta_t$ to select action
- Use target network $\theta^-$ to evaluate action

### Implementation

```python
class DoubleDQNAgent(DQNAgent):
    def train_step(self, batch):
        """
        Double DQN training step
        
        Args:
            batch: Batch of experiences
        """
        states, actions, rewards, next_states, dones = batch
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target
        with torch.no_grad():
            # Select actions using main network
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            
            # Evaluate using target network
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q.squeeze()
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
```

## Dueling DQN

**Dueling DQN** separates the Q-function into state value $V(s)$ and advantage $A(s, a)$:

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')$$

### Architecture

```python
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Dueling DQN architecture
        
        Args:
            input_shape: Input shape
            num_actions: Number of actions
        """
        super(DuelingDQN, self).__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Shared fully connected layer
        self.fc = nn.Linear(conv_out_size, 512)
        
        # Value stream
        self.value_stream = nn.Linear(512, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(512, num_actions)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input state
        
        Returns:
            Q-values
        """
        x = x / 255.0
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared fully connected
        x = F.relu(self.fc(x))
        
        # Value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(1, keepdim=True))
        
        return q_values
```

### Benefits

1. **Better Value Estimation**: Separates state value from action advantage
2. **Faster Learning**: Value function learns faster
3. **Robustness**: More robust to irrelevant actions

## Prioritized Experience Replay

**Prioritized Experience Replay** samples experiences with probability proportional to their TD error, focusing learning on important transitions.

### Prioritization

Priority of transition $i$:

$$p_i = |\delta_i| + \epsilon$$

where $\delta_i$ is the TD error and $\epsilon$ is a small constant.

Sampling probability:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

where $\alpha$ controls prioritization ($\alpha = 0$ is uniform).

### Importance Sampling

To correct for bias, use importance sampling weights:

$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^\beta$$

where $\beta$ controls correction ($\beta = 1$ is full correction).

### Implementation

```python
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Prioritized experience replay buffer
        
        Args:
            capacity: Buffer capacity
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch with priorities"""
        if len(self.buffer) < batch_size:
            return None
        
        # Compute sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
```

## Training and Implementation Details

### Complete DQN Training Loop

```python
def train_dqn(env, agent, replay_buffer, num_episodes=1000, 
              max_steps=10000, epsilon_start=1.0, epsilon_end=0.01, 
              epsilon_decay=0.995, batch_size=32, update_freq=4):
    """
    Complete DQN training loop
    
    Args:
        env: Environment
        agent: DQN agent
        replay_buffer: Replay buffer
        num_episodes: Number of episodes
        max_steps: Maximum steps per episode
        epsilon_start: Starting epsilon
        epsilon_end: Final epsilon
        epsilon_decay: Epsilon decay rate
        batch_size: Batch size
        update_freq: Network update frequency
    """
    epsilon = epsilon_start
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.random_action()
            else:
                with torch.no_grad():
                    q_values = agent.q_network(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            if len(replay_buffer) >= batch_size and total_steps % update_freq == 0:
                batch = replay_buffer.sample(batch_size)
                agent.train_step(batch)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
```

### Hyperparameters

Typical hyperparameters for DQN:

- **Learning Rate**: 0.0001 - 0.00025
- **Discount Factor**: 0.99
- **Replay Buffer Size**: 1,000,000
- **Batch Size**: 32
- **Target Update Frequency**: 10,000 steps
- **Epsilon Decay**: 0.995 per episode
- **Optimizer**: Adam or RMSprop

## Challenges and Limitations

### Overestimation Bias

Q-Learning's max operator causes overestimation. Solutions:
- Double DQN
- Clipped Double Q-Learning
- Distributional RL

### Sample Efficiency

DQN requires many samples. Improvements:
- Prioritized experience replay
- Multi-step returns
- Better exploration strategies

### Stability

Deep RL can be unstable. Mitigations:
- Target networks
- Gradient clipping
- Learning rate scheduling

### Function Approximation Limitations

- **Extrapolation**: Poor generalization to unseen states
- **Catastrophic Forgetting**: Forgets past experiences
- **Distributional Shift**: Training and test distributions differ

## Key Takeaways

1. **Deep Q-Networks (DQN)** combine Q-Learning with deep neural networks to handle high-dimensional state spaces, achieving breakthrough results on Atari games.

2. **Experience Replay** stores past experiences and samples randomly for training, breaking temporal correlations and improving data efficiency.

3. **Target Networks** use separate networks for computing targets, updated less frequently to stabilize learning and prevent target non-stationarity.

4. **Double DQN** addresses overestimation bias by decoupling action selection (main network) from action evaluation (target network).

5. **Dueling DQN** separates state value and action advantage, leading to better value estimation and faster learning.

6. **Prioritized Experience Replay** samples transitions with probability proportional to TD error, focusing learning on important experiences.

7. **Training DQN** requires careful hyperparameter tuning, including learning rates, exploration schedules, and network update frequencies.

8. **Challenges** include overestimation bias, sample inefficiency, stability issues, and limitations of function approximation in RL settings.

9. **DQN variants** (Double DQN, Dueling DQN, Prioritized Replay) can be combined for improved performance, as demonstrated in Rainbow DQN.

10. **DQN** forms the foundation for many advanced value-based methods and remains a fundamental algorithm in deep reinforcement learning.
