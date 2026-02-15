# Continual, Multi-Task, Active Learning, and Reinforcement Learning

## Table of Contents
1. [Overview](#overview)
2. [Continual Learning](#continual-learning)
3. [Neural Architecture Search](#neural-architecture-search)
4. [Multi-Task Learning](#multi-task-learning)
5. [Active Learning](#active-learning)
6. [Reinforcement Learning Basics with PyTorch](#reinforcement-learning-basics-with-pytorch)

---

## Overview

**Continual learning** trains on sequential tasks without forgetting. **Neural architecture search (NAS)** automates architecture design. **Multi-task learning** shares representations across tasks. **Active learning** selects which samples to label. **Reinforcement learning** learns from rewards in sequential decision-making. All extend standard supervised learning paradigms.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

---

## Continual Learning

### Catastrophic Forgetting

**Catastrophic forgetting** occurs when training on a new task overwrites weights that were important for previous tasks. The model loses performance on earlier tasks.

### EWC (Elastic Weight Consolidation)

**EWC** penalizes changes to parameters important for previous tasks. Importance is estimated via **Fisher information** (squared gradients).

```python
class EWC:
    def __init__(self, model, dataset_loader, device='cuda', importance=1000):
        self.model = model
        self.importance = importance
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher_information(dataset_loader)

    def _compute_fisher_information(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            loss = F.log_softmax(outputs, dim=1)[range(targets.size(0)), targets].sum()
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        for n in fisher:
            fisher[n] /= len(dataloader.dataset)
        return fisher

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.importance * loss
```

### Replay Buffers

**Experience replay** stores a subset of past task data and interleaves it with current task training. Reduces forgetting by rehearsing old examples.

```python
class ExperienceReplay:
    def __init__(self, memory_size=1000):
        self.memory_size = memory_size
        self.memory = []
        self.position = 0

    def add(self, data, targets, task_id):
        experience = (data.cpu(), targets.cpu(), task_id)
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size, device='cuda'):
        indices = np.random.choice(len(self.memory), min(batch_size, len(self.memory)), replace=False)
        batch_data = [self.memory[i] for i in indices]
        return (torch.stack([b[0] for b in batch_data]).to(device),
                torch.stack([b[1] for b in batch_data]).to(device),
                [b[2] for b in batch_data])
```

### Progressive Networks

**Progressive networks** add a new column (sub-network) per task. Lateral connections from previous columns allow transfer. No forgetting by design; parameters grow with tasks.

```python
class ProgressiveColumn(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, prev_columns=None):
        super().__init__()
        self.prev_columns = prev_columns or []
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            layer_input_size = sizes[i]
            if i > 0:
                for prev_col in self.prev_columns:
                    if i - 1 < len(prev_col.layers):
                        layer_input_size += hidden_sizes[i-1]
            self.layers.append(nn.Linear(layer_input_size, sizes[i + 1]))

    def forward(self, x, prev_activations=None):
        activations = []
        current = x
        for i, layer in enumerate(self.layers):
            if i > 0 and prev_activations:
                lateral = [pa[i-1] for pa in prev_activations if i-1 < len(pa)]
                if lateral:
                    current = torch.cat([current] + lateral, dim=1)
            current = layer(current)
            if i < len(self.layers) - 1:
                current = F.relu(current)
            activations.append(current)
        return current, activations
```

### Learning without Forgetting (LwF)

**LwF** distills knowledge from the old model (before training on new task) to the current model. No access to old task data required.

```python
def train_task(self, train_loader, epochs=10):
    for epoch in range(epochs):
        for data, targets in train_loader:
            outputs = self.model(data)
            ce_loss = F.cross_entropy(outputs, targets)
            total_loss = ce_loss
            if self.old_model is not None:
                with torch.no_grad():
                    old_outputs = self.old_model(data)
                distillation_loss = F.kl_div(
                    F.log_softmax(outputs / self.T, dim=1),
                    F.softmax(old_outputs / self.T, dim=1),
                    reduction='batchmean'
                ) * (self.T ** 2)
                total_loss += self.alpha * distillation_loss
            total_loss.backward()
            self.optimizer.step()
    self.old_model = copy.deepcopy(self.model)
```

---

## Neural Architecture Search

### Search Space

The **search space** defines allowed operations and connections. Common primitives: conv 3x3, conv 5x5, separable conv, dilated conv, skip connect, pooling, none.

```python
ops_list = ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
            'avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'none']
```

### DARTS (Differentiable Architecture Search)

**DARTS** relaxes the discrete search to continuous mixture weights. Architecture parameters \(\alpha\) are optimized jointly with weights via bilevel optimization.

```python
class MixedOp(nn.Module):
    def __init__(self, C, stride, ops_list):
        super().__init__()
        self.ops = nn.ModuleList([self._create_op(op, C, stride) for op in ops_list])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))
```

### Bilevel Optimization

**DARTS** alternates: (1) update architecture parameters on validation loss; (2) update weights on training loss.

```python
def train_step(self, train_data, train_targets, valid_data, valid_targets):
    self.arch_optimizer.zero_grad()
    logits = self.model(valid_data)
    arch_loss = self.criterion(logits, valid_targets)
    arch_loss.backward()
    self.arch_optimizer.step()

    self.model_optimizer.zero_grad()
    logits = self.model(train_data)
    model_loss = self.criterion(logits, train_targets)
    model_loss.backward()
    self.model_optimizer.step()
    return model_loss.item(), arch_loss.item()
```

### One-Shot NAS

**One-shot NAS** trains a supernet once; architectures are evaluated by inheriting weights from the supernet. No separate training per architecture.

### Random Search Baseline

**Random search** samples architectures from the search space and trains each. Simple but effective baseline.

```python
def sample_architecture(self):
    arch = {}
    for layer_name, ops in self.search_space.items():
        arch[layer_name] = np.random.choice(ops)
    return arch
```

---

## Multi-Task Learning

### Shared Encoder and Task-Specific Heads

A **shared backbone** extracts features; **task-specific heads** produce task outputs. Parameters are shared across tasks for transfer.

```python
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, shared_dim=128, task_configs=None):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                self.task_heads[task_name] = nn.Linear(shared_dim, config['num_classes'])
            elif config['type'] == 'regression':
                self.task_heads[task_name] = nn.Linear(shared_dim, config['output_dim'])

    def forward(self, x, task_name=None):
        shared = self.shared_backbone(x)
        if task_name:
            return self.task_heads[task_name](shared)
        return {t: self.task_heads[t](shared) for t in self.task_heads}
```

### Loss Weighting

**Equal weighting** sums task losses. **Uncertainty weighting** learns \(\sigma_i\) per task: \( \frac{1}{2\sigma_i^2} L_i + \log \sigma_i \).

```python
if self.uncertainty_weights:
    precision = torch.exp(-self.log_vars[i])
    weighted_loss = precision * loss + self.log_vars[i]
else:
    weighted_loss = self.task_weights[i] * loss
```

### Gradient Balancing (GradNorm)

**GradNorm** adjusts task weights so gradient norms are balanced. Targets gradient norms proportional to inverse training rate.

```python
def compute_gradnorm_loss(self, losses, shared_parameters):
    grad_norms = []
    for loss in losses.values():
        grads = torch.autograd.grad(loss, shared_parameters, retain_graph=True, create_graph=True)
        grad_norm = sum(g.norm()**2 for g in grads)**0.5
        grad_norms.append(grad_norm)
    grad_norms = torch.stack(grad_norms)
    loss_ratios = torch.tensor([l.item()/self.initial_losses[t] for t, l in losses.items()])
    inverse_rates = loss_ratios / loss_ratios.mean()
    target_norms = grad_norms.mean() * (inverse_rates ** self.alpha)
    return F.l1_loss(grad_norms, target_norms)
```

---

## Active Learning

### Uncertainty Sampling

**Uncertainty sampling** selects samples the model is most uncertain about. Metrics: **entropy**, **least confidence**, **margin**.

```python
class UncertaintySampling:
    def __init__(self, strategy='entropy'):
        self.strategy = strategy

    def query(self, model, unlabeled_loader, n_samples, device='cuda'):
        model.eval()
        uncertainties, indices = [], []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(device)
                probs = F.softmax(model(data), dim=1)
                if self.strategy == 'entropy':
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    uncertainties.extend(entropy.cpu().tolist())
                elif self.strategy == 'least_confidence':
                    uncertainty = 1 - probs.max(dim=1)[0]
                    uncertainties.extend(uncertainty.cpu().tolist())
                elif self.strategy == 'margin':
                    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                    uncertainty = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])
                    uncertainties.extend(uncertainty.cpu().tolist())
                start = batch_idx * unlabeled_loader.batch_size
                indices.extend(range(start, start + data.size(0)))
        scores = sorted(zip(indices, uncertainties), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:n_samples]]
```

### Query Strategies

| Strategy | Idea |
|----------|------|
| Entropy | High entropy = uncertain |
| Least confidence | 1 - max prob |
| Margin | Small gap between top-2 probs |
| Query by committee | Disagreement among models |
| Diversity | K-means, farthest-first |

### Pool-Based Active Learning

**Pool-based** setting: large unlabeled pool, small labeled set. Each round: train on labeled set, query batch from pool, add to labeled set.

```python
def active_learning_loop(self, labeled_dataset, unlabeled_dataset, test_dataset,
                         query_strategy, n_queries=100, n_rounds=5):
    labeled_indices = list(range(len(labeled_dataset)))
    unlabeled_indices = list(range(len(unlabeled_dataset)))
    for round_num in range(n_rounds):
        labeled_subset = Subset(labeled_dataset, labeled_indices)
        loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
        for epoch in range(5):
            self.train_epoch(loader)
        test_acc = self.evaluate(DataLoader(test_dataset, batch_size=32))
        if round_num < n_rounds - 1 and unlabeled_indices:
            unlabeled_subset = Subset(unlabeled_dataset, unlabeled_indices)
            unlabeled_loader = DataLoader(unlabeled_subset, batch_size=32)
            queried = query_strategy.query(self.model, unlabeled_loader, n_queries, self.device)
            actual = [unlabeled_indices[i] for i in queried if i < len(unlabeled_indices)]
            labeled_indices.extend(actual)
            for idx in actual:
                unlabeled_indices.remove(idx)
```

### Acquisition Functions

**Acquisition functions** score unlabeled samples. Common choices: entropy, margin, BALD (Bayesian), expected improvement.

---

## Reinforcement Learning Basics with PyTorch

### Policy Networks

A **policy network** maps state to action probabilities. Output is passed through softmax for discrete actions.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
```

### REINFORCE

**REINFORCE** is a policy gradient method. Collect trajectory, compute returns \(G_t\), and maximize \(\mathbb{E}[\log \pi(a|s) G_t]\).

```python
def update(self):
    returns = []
    G = 0
    for reward in reversed(self.rewards):
        G = reward + self.gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    policy_loss = torch.cat([-lp * G for lp, G in zip(self.log_probs, returns)]).sum()
    self.optimizer.zero_grad()
    policy_loss.backward()
    self.optimizer.step()
    self.log_probs, self.rewards = [], []
```

### Value Functions

A **value function** \(V(s)\) estimates expected return from state \(s\). Used as baseline to reduce variance in policy gradients (e.g., Actor-Critic).

```python
class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### DQN and Experience Replay

**DQN** learns \(Q(s,a)\) with a neural network. **Experience replay** stores transitions \((s, a, r, s')\) and samples random batches for training.

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        return states, actions, rewards, next_states, dones
```

### Simple Grid World Environment

A minimal environment for testing RL algorithms.

```python
class GridWorldEnv:
    def __init__(self, size=5, goal_reward=10, step_penalty=-0.1):
        self.size = size
        self.action_space = 4
        self.state_space = size * size
        self.goal_pos = (size - 1, size - 1)

    def reset(self):
        self.agent_pos = (0, 0)
        return self.get_state()

    def get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def step(self, action):
        row, col = self.agent_pos
        if action == 0: row = max(0, row - 1)
        elif action == 1: row = min(self.size - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(self.size - 1, col + 1)
        self.agent_pos = (row, col)
        reward = self.goal_reward if self.agent_pos == self.goal_pos else self.step_penalty
        done = self.agent_pos == self.goal_pos
        return self.get_state(), reward, done, {}
```

### Actor-Critic

**Actor-Critic** combines a policy (actor) and value function (critic). Advantage \(A = G - V(s)\) reduces variance; critic learns \(V\).

```python
def update(self):
    returns = []
    G = 0
    for r in reversed(self.rewards):
        G = r + self.gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    values = torch.cat(self.values)
    advantages = returns - values.squeeze()
    actor_loss = torch.cat([-lp * adv.detach() for lp, adv in zip(self.log_probs, advantages)]).sum()
    critic_loss = F.mse_loss(values.squeeze(), returns)
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
```
