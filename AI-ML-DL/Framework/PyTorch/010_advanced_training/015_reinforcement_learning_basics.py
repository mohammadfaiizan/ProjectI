import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from typing import List, Tuple
import matplotlib.pyplot as plt

# Simple Grid World Environment
class GridWorldEnv:
    """Simple grid world environment for RL"""
    
    def __init__(self, size=5, goal_reward=10, step_penalty=-0.1):
        self.size = size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = 4
        self.state_space = size * size
        
        # Goal position (bottom-right corner)
        self.goal_pos = (size - 1, size - 1)
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Start at top-left corner
        self.agent_pos = (0, 0)
        return self.get_state()
    
    def get_state(self):
        """Get current state as integer"""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        """Take action and return (next_state, reward, done, info)"""
        row, col = self.agent_pos
        
        # Execute action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        
        self.agent_pos = (row, col)
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_penalty
            done = False
        
        return self.get_state(), reward, done, {}

# REINFORCE (Policy Gradient) Algorithm
class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class REINFORCE:
    """REINFORCE algorithm implementation"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for episode
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Select action using policy network"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state)
        
        # Sample action from probability distribution
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        
        # Store log probability for later
        self.log_probs.append(m.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def update(self):
        """Update policy using collected episode data"""
        if len(self.rewards) == 0:
            return
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()

# Deep Q-Network (DQN)
class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=32):
        """Update Q-network using batch of experiences"""
        if len(self.replay_buffer) < batch_size:
            return 0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

# Actor-Critic Algorithm
class ActorNetwork(nn.Module):
    """Actor network for Actor-Critic"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class CriticNetwork(nn.Module):
    """Critic network for Actor-Critic"""
    
    def __init__(self, state_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ActorCritic:
    """Actor-Critic algorithm implementation"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Networks
        self.actor = ActorNetwork(state_size, action_size)
        self.critic = CriticNetwork(state_size)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Storage
        self.log_probs = []
        self.values = []
        self.rewards = []
    
    def select_action(self, state):
        """Select action and estimate value"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities and value
        action_probs = self.actor(state)
        value = self.critic(state)
        
        # Sample action
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        
        # Store for later
        self.log_probs.append(m.log_prob(action))
        self.values.append(value)
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward"""
        self.rewards.append(reward)
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.rewards) == 0:
            return 0, 0
        
        # Calculate returns and advantages
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        values = torch.cat(self.values)
        
        # Calculate advantages
        advantages = returns - values.squeeze()
        
        # Actor loss (policy gradient with advantage)
        actor_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            actor_loss.append(-log_prob * advantage.detach())
        actor_loss = torch.cat(actor_loss).sum()
        
        # Critic loss (value function approximation)
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return actor_loss.item(), critic_loss.item()

# Training Functions
def train_reinforce(env, agent, episodes=1000):
    """Train REINFORCE agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state_one_hot = np.zeros(env.state_space)
        state_one_hot[state] = 1
        
        episode_reward = 0
        
        while True:
            action = agent.select_action(state_one_hot)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_reward(reward)
            episode_reward += reward
            
            if done:
                break
            
            state = next_state
            state_one_hot = np.zeros(env.state_space)
            state_one_hot[state] = 1
        
        # Update policy
        loss = agent.update()
        scores.append(episode_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {episode}, Average Score: {avg_score:.2f}')
    
    return scores

def train_dqn(env, agent, episodes=1000, update_target_freq=100):
    """Train DQN agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state_one_hot = np.zeros(env.state_space)
        state_one_hot[state] = 1
        
        episode_reward = 0
        
        while True:
            action = agent.select_action(state_one_hot)
            next_state, reward, done, _ = env.step(action)
            
            next_state_one_hot = np.zeros(env.state_space)
            next_state_one_hot[next_state] = 1
            
            # Store experience
            agent.store_experience(state_one_hot, action, reward, next_state_one_hot, done)
            
            # Update network
            loss = agent.update()
            
            episode_reward += reward
            
            if done:
                break
            
            state_one_hot = next_state_one_hot
        
        # Update target network periodically
        if episode % update_target_freq == 0:
            agent.update_target_network()
        
        scores.append(episode_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}')
    
    return scores

def train_actor_critic(env, agent, episodes=1000):
    """Train Actor-Critic agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state_one_hot = np.zeros(env.state_space)
        state_one_hot[state] = 1
        
        episode_reward = 0
        
        while True:
            action = agent.select_action(state_one_hot)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_reward(reward)
            episode_reward += reward
            
            if done:
                break
            
            state = next_state
            state_one_hot = np.zeros(env.state_space)
            state_one_hot[state] = 1
        
        # Update networks
        actor_loss, critic_loss = agent.update()
        scores.append(episode_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {episode}, Average Score: {avg_score:.2f}')
    
    return scores

# Evaluation Function
def evaluate_agent(env, agent, episodes=100, algorithm='dqn'):
    """Evaluate trained agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state_one_hot = np.zeros(env.state_space)
        state_one_hot[state] = 1
        
        episode_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite loops
        
        while steps < max_steps:
            if algorithm == 'dqn':
                action = agent.select_action(state_one_hot, training=False)
            elif algorithm == 'reinforce':
                action = agent.select_action(state_one_hot)
            elif algorithm == 'actor_critic':
                action = agent.select_action(state_one_hot)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
            state_one_hot = np.zeros(env.state_space)
            state_one_hot[state] = 1
        
        scores.append(episode_reward)
    
    return scores

if __name__ == "__main__":
    print("Reinforcement Learning Basics")
    print("=" * 35)
    
    # Create environment
    env = GridWorldEnv(size=4, goal_reward=10, step_penalty=-0.1)
    
    print(f"Environment: {env.size}x{env.size} Grid World")
    print(f"State space: {env.state_space}")
    print(f"Action space: {env.action_space}")
    print(f"Goal position: {env.goal_pos}")
    
    # Test REINFORCE
    print("\n1. Training REINFORCE Agent")
    print("-" * 30)
    
    reinforce_agent = REINFORCE(env.state_space, env.action_space, lr=1e-2)
    reinforce_scores = train_reinforce(env, reinforce_agent, episodes=500)
    
    print("REINFORCE training completed!")
    print(f"Final 100-episode average: {np.mean(reinforce_scores[-100:]):.2f}")
    
    # Test DQN
    print("\n2. Training DQN Agent")
    print("-" * 25)
    
    dqn_agent = DQNAgent(env.state_space, env.action_space, lr=1e-3)
    dqn_scores = train_dqn(env, dqn_agent, episodes=500)
    
    print("DQN training completed!")
    print(f"Final 100-episode average: {np.mean(dqn_scores[-100:]):.2f}")
    
    # Test Actor-Critic
    print("\n3. Training Actor-Critic Agent")
    print("-" * 35)
    
    ac_agent = ActorCritic(env.state_space, env.action_space, lr=1e-2)
    ac_scores = train_actor_critic(env, ac_agent, episodes=500)
    
    print("Actor-Critic training completed!")
    print(f"Final 100-episode average: {np.mean(ac_scores[-100:]):.2f}")
    
    # Evaluate all agents
    print("\n4. Evaluation")
    print("-" * 15)
    
    print("Evaluating agents on 100 episodes...")
    
    reinforce_eval = evaluate_agent(env, reinforce_agent, episodes=100, algorithm='reinforce')
    dqn_eval = evaluate_agent(env, dqn_agent, episodes=100, algorithm='dqn')
    ac_eval = evaluate_agent(env, ac_agent, episodes=100, algorithm='actor_critic')
    
    print(f"REINFORCE - Average Score: {np.mean(reinforce_eval):.2f} ± {np.std(reinforce_eval):.2f}")
    print(f"DQN - Average Score: {np.mean(dqn_eval):.2f} ± {np.std(dqn_eval):.2f}")
    print(f"Actor-Critic - Average Score: {np.mean(ac_eval):.2f} ± {np.std(ac_eval):.2f}")
    
    # Test optimal policy (for comparison)
    print("\n5. Random Policy Baseline")
    print("-" * 30)
    
    random_scores = []
    for _ in range(100):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 100:
            action = random.randint(0, env.action_space - 1)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
            
            state = next_state
        
        random_scores.append(episode_reward)
    
    print(f"Random Policy - Average Score: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    
    # Performance comparison
    print("\n6. Performance Summary")
    print("-" * 25)
    
    algorithms = ['Random', 'REINFORCE', 'DQN', 'Actor-Critic']
    avg_scores = [
        np.mean(random_scores),
        np.mean(reinforce_eval),
        np.mean(dqn_eval),
        np.mean(ac_eval)
    ]
    
    print("Algorithm Performance Ranking:")
    sorted_results = sorted(zip(algorithms, avg_scores), key=lambda x: x[1], reverse=True)
    
    for i, (algo, score) in enumerate(sorted_results):
        print(f"{i+1}. {algo}: {score:.2f}")
    
    # Demonstrate policy
    print("\n7. Demonstrating Best Policy")
    print("-" * 35)
    
    best_agent = dqn_agent  # Assuming DQN performed well
    state = env.reset()
    
    print(f"Grid size: {env.size}x{env.size}")
    print(f"Start position: (0, 0)")
    print(f"Goal position: {env.goal_pos}")
    print("\nAgent's path:")
    
    state_one_hot = np.zeros(env.state_space)
    state_one_hot[state] = 1
    
    path = [(0, 0)]
    steps = 0
    
    while steps < 20:  # Limit steps for demonstration
        action = best_agent.select_action(state_one_hot, training=False)
        next_state, reward, done, _ = env.step(action)
        
        # Convert state to position
        row, col = divmod(next_state, env.size)
        path.append((row, col))
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Step {steps + 1}: Action = {action_names[action]}, Position = ({row}, {col}), Reward = {reward}")
        
        if done:
            print("Goal reached!")
            break
        
        state = next_state
        state_one_hot = np.zeros(env.state_space)
        state_one_hot[state] = 1
        steps += 1
    
    print(f"\nTotal steps to reach goal: {len(path) - 1}")
    print("Path taken:", " -> ".join([str(pos) for pos in path]))
    
    print("\nReinforcement Learning demonstrations completed!") 