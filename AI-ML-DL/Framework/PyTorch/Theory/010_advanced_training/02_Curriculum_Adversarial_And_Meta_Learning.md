# Curriculum, Adversarial, and Meta-Learning

## Table of Contents
1. [Overview](#overview)
2. [Curriculum Learning](#curriculum-learning)
3. [Adversarial Training](#adversarial-training)
4. [Meta-Learning (MAML)](#meta-learning-maml)
5. [Few-Shot Learning](#few-shot-learning)

---

## Overview

**Curriculum learning** orders training data by difficulty. **Adversarial training** improves robustness by training on perturbed inputs. **Meta-learning** learns to learn across tasks. **Few-shot learning** generalizes from few examples per class. All extend standard supervised training with structured data or task design.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
```

---

## Curriculum Learning

### Difficulty Scoring

**Difficulty scores** assign a scalar to each sample. Heuristics include target class, data variance, or model-based loss. Scores determine ordering for curriculum progression.

```python
def _calculate_difficulty_scores(self):
    difficulty_scores = []
    for i in range(len(self.base_dataset)):
        _, target = self.base_dataset[i]
        difficulty = float(target) if isinstance(target, (int, float)) else float(target.item())
        difficulty_scores.append(difficulty)
    return np.array(difficulty_scores)
```

### Pacing Functions

**Pacing functions** control how much data is exposed over time. Common strategies: linear, exponential, root, or step.

```python
def update_curriculum(self, epoch, total_epochs):
    progress = epoch / total_epochs
    if self.curriculum_strategy == 'linear':
        self.current_data_ratio = min(0.1 + 0.9 * progress, 1.0)
    elif self.curriculum_strategy == 'exponential':
        self.current_data_ratio = min(0.1 + 0.9 * (progress ** 2), 1.0)
    elif self.curriculum_strategy == 'root':
        self.current_data_ratio = min(0.1 + 0.9 * (progress ** 0.5), 1.0)
    elif self.curriculum_strategy == 'step':
        if progress < 0.25:
            self.current_data_ratio = 0.25
        elif progress < 0.5:
            self.current_data_ratio = 0.5
        elif progress < 0.75:
            self.current_data_ratio = 0.75
        else:
            self.current_data_ratio = 1.0
```

| Strategy | Behavior |
|----------|----------|
| linear | Gradual increase |
| exponential | Slow start, fast finish |
| root | Fast start, slow finish |
| step | Discrete jumps |

### Self-Paced Learning

**Self-paced learning** weights samples by loss. Samples with loss below a threshold \(\lambda\) get weight 1; others get 0. \(\lambda\) increases over training.

```python
def calculate_sample_weights(self, losses):
    weights = (losses <= self.lambda_sp).float()
    return weights

def self_paced_loss(self, outputs, targets):
    individual_losses = F.cross_entropy(outputs, targets, reduction='none')
    weights = self.calculate_sample_weights(individual_losses)
    weighted_loss = torch.sum(weights * individual_losses) / batch_size
    regularization = -self.lambda_sp * torch.sum(weights) / batch_size
    return weighted_loss + regularization
```

### Data Scheduling and Curriculum Sampler

A **CurriculumSampler** selects indices based on difficulty and epoch.

```python
class CurriculumSampler(Sampler):
    def __init__(self, dataset, curriculum_type='easy_first', batch_size=32):
        self.dataset = dataset
        self.curriculum_type = curriculum_type
        self.epoch = 0

    def __iter__(self):
        sorted_pairs = sorted(zip(self.indices, self.difficulty_scores), key=lambda x: x[1])
        progress = min(self.epoch / 50.0, 1.0)
        num_samples = int(len(self.indices) * progress)
        if self.curriculum_type == 'easy_first':
            selected = [p[0] for p in sorted_pairs[:num_samples]]
        elif self.curriculum_type == 'hard_first':
            selected = [p[0] for p in sorted_pairs[-num_samples:]]
        random.shuffle(selected)
        return iter(selected)
```

---

## Adversarial Training

### FGSM (Fast Gradient Sign Method)

**FGSM** generates adversarial examples in one step: \( x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L) \).

```python
class FGSMAttack:
    def __init__(self, epsilon=0.3):
        self.epsilon = epsilon

    def generate(self, model, data, targets, device='cuda'):
        data = data.to(device).requires_grad_(True)
        targets = targets.to(device)
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        model.zero_grad()
        loss.backward()
        perturbed = data + self.epsilon * data.grad.sign()
        return torch.clamp(perturbed, 0, 1).detach()
```

### PGD (Projected Gradient Descent)

**PGD** iteratively applies FGSM and projects back to an \(\epsilon\)-ball. Stronger than FGSM.

```python
class PGDAttack:
    def __init__(self, epsilon=0.3, alpha=0.01, num_steps=40):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate(self, model, data, targets, device='cuda'):
        perturbed = data + torch.empty_like(data).uniform_(-self.epsilon, self.epsilon)
        perturbed = torch.clamp(perturbed, 0, 1)
        for _ in range(self.num_steps):
            perturbed.requires_grad_(True)
            outputs = model(perturbed)
            loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            loss.backward()
            perturbed = perturbed + self.alpha * perturbed.grad.sign()
            eta = torch.clamp(perturbed - data, -self.epsilon, self.epsilon)
            perturbed = torch.clamp(data + eta, 0, 1).detach()
        return perturbed
```

### Adversarial Training Loop and AT Loss

Train on a mix of clean and adversarial examples. **AT loss** is standard cross-entropy on the combined batch.

```python
def train_step(self, data, targets, adv_ratio=0.5):
    adv_size = int(data.size(0) * adv_ratio)
    if adv_size > 0:
        adv_data = self.attack.generate(self.model, data[:adv_size], targets[:adv_size], self.device)
        combined_data = torch.cat([data[adv_size:], adv_data], dim=0)
    else:
        combined_data = data
    self.optimizer.zero_grad()
    outputs = self.model(combined_data)
    loss = self.criterion(outputs, targets)
    loss.backward()
    self.optimizer.step()
    return loss.item()
```

### TRADES Loss

**TRADES** minimizes natural loss plus KL divergence between natural and adversarial predictions.

```python
def trades_loss(self, data, targets):
    logits = self.model(data)
    natural_loss = F.cross_entropy(logits, targets)
    adv_data = self._generate_adversarial(data)
    logits_adv = self.model(adv_data)
    robust_loss = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits, dim=1),
        reduction='batchmean'
    )
    return natural_loss + self.beta * robust_loss
```

### Virtual Adversarial Training (VAT)

**VAT** finds a perturbation that maximizes KL divergence from the original prediction, then trains to minimize that divergence.

```python
def virtual_adversarial_loss(self, data, epsilon=1.0, xi=1e-6, ip=1):
    d = torch.randn_like(data)
    d = self._l2_normalize(d)
    for _ in range(ip):
        d.requires_grad_(True)
        kl_loss = F.kl_div(
            F.log_softmax(self.model(data + xi * d), dim=1),
            F.softmax(self.model(data).detach(), dim=1),
            reduction='batchmean'
        )
        kl_loss.backward()
        d = self._l2_normalize(d.grad.detach())
    r_vadv = epsilon * d
    vat_loss = F.kl_div(
        F.log_softmax(self.model(data + r_vadv), dim=1),
        F.softmax(self.model(data).detach(), dim=1),
        reduction='batchmean'
    )
    return vat_loss
```

---

## Meta-Learning (MAML)

### Inner and Outer Loop

**MAML** (Model-Agnostic Meta-Learning) has an **inner loop** that adapts to a task's support set and an **outer loop** that updates initial parameters to minimize query loss after adaptation.

```python
def inner_update(self, support_data, support_labels, model_params=None):
    if model_params is None:
        model_params = list(self.model.parameters())
    for step in range(self.inner_steps):
        predictions = self._forward_with_params(support_data, model_params)
        support_loss = self.criterion(predictions, support_labels)
        grads = torch.autograd.grad(
            support_loss, model_params,
            create_graph=not self.first_order,
            retain_graph=True, allow_unused=True
        )
        updated_params = []
        for param, grad in zip(model_params, grads):
            if grad is not None:
                updated_param = param - self.inner_lr * grad
            else:
                updated_param = param
            updated_params.append(updated_param)
        model_params = updated_params
    return updated_params

def meta_update(self, tasks_batch):
    meta_loss = 0.0
    for task in tasks_batch:
        support_data, support_labels, query_data, query_labels = task
        adapted_params = self.inner_update(support_data, support_labels)
        query_predictions = self._forward_with_params(query_data, adapted_params)
        meta_loss += self.criterion(query_predictions, query_labels)
    meta_loss = meta_loss / len(tasks_batch)
    self.meta_optimizer.zero_grad()
    meta_loss.backward()
    self.meta_optimizer.step()
    return meta_loss.item()
```

### Task Distribution

Sample tasks with **N-way K-shot** structure: N classes, K support examples per class, plus query examples.

```python
class TaskGenerator:
    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.class_to_indices = self._organize_by_class()

    def sample_task(self):
        selected_classes = random.sample(self.classes, self.n_way)
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        for class_idx, class_label in enumerate(selected_classes):
            indices = random.sample(self.class_to_indices[class_label], self.k_shot + self.q_query)
            support_indices = indices[:self.k_shot]
            query_indices = indices[self.k_shot:]
            for idx in support_indices:
                data, _ = self.dataset[idx]
                support_data.append(data)
                support_labels.append(class_idx)
            for idx in query_indices:
                data, _ = self.dataset[idx]
                query_data.append(data)
                query_labels.append(class_idx)
        return (torch.stack(support_data), torch.tensor(support_labels),
                torch.stack(query_data), torch.tensor(query_labels))
```

### First-Order MAML (FOMAML)

**FOMAML** drops second-order gradients for efficiency. Use `create_graph=False` in inner-loop gradient computation.

```python
class FOMAML(MAML):
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01, inner_steps=5, device='cuda'):
        super().__init__(model, meta_lr, inner_lr, inner_steps, first_order=True, device=device)
```

### Reptile

**Reptile** performs multiple inner updates and moves initial parameters toward the adapted parameters.

```python
def meta_update(self, task):
    support_data, support_labels, _, _ = task
    initial_params = [p.clone() for p in self.model.parameters()]
    inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
    for _ in range(self.inner_steps):
        inner_optimizer.zero_grad()
        loss = self.criterion(self.model(support_data), support_labels)
        loss.backward()
        inner_optimizer.step()
    for init_p, curr_p in zip(initial_params, self.model.parameters()):
        init_p.data += self.meta_lr * (curr_p.data - init_p.data)
        curr_p.data = init_p.data
```

---

## Few-Shot Learning

### Support and Query Sets

**Support set**: few labeled examples per class. **Query set**: examples to classify. Episodes are sampled as (support, query) pairs.

```python
support_data, support_labels, query_data, query_labels = task_generator.sample_task()
logits = model(support_data, support_labels, query_data, n_way, k_shot)
loss = F.cross_entropy(logits, query_labels)
```

### Prototypical Networks

**Prototypical networks** compute a prototype per class as the mean of support embeddings. Query embeddings are classified by nearest prototype (negative distance as logits).

```python
def compute_prototypes(self, support_embeddings, support_labels, n_way):
    prototypes = torch.zeros(n_way, support_embeddings.size(1)).to(support_embeddings.device)
    for class_idx in range(n_way):
        mask = (support_labels == class_idx)
        if mask.sum() > 0:
            prototypes[class_idx] = support_embeddings[mask].mean(dim=0)
    return prototypes

def forward(self, support_data, support_labels, query_data, n_way, k_shot):
    support_emb = self.encoder(support_data)
    query_emb = self.encoder(query_data)
    prototypes = self.compute_prototypes(support_emb, support_labels, n_way)
    distances = torch.cdist(query_emb, prototypes, p=2)
    return -distances
```

### Matching Networks

**Matching networks** use attention over support embeddings to predict query labels. Similarity (e.g., cosine) becomes attention weights; weighted sum of one-hot support labels gives predictions.

```python
def forward(self, support_data, support_labels, query_data, n_way, k_shot):
    support_emb = self.encoder(support_data)
    query_emb = self.encoder(query_data)
    similarities = F.linear(F.normalize(query_emb), F.normalize(support_emb))
    attention = F.softmax(similarities, dim=1)
    support_one_hot = F.one_hot(support_labels, num_classes=n_way).float()
    predictions = torch.mm(attention, support_one_hot)
    return predictions
```

### N-Way K-Shot

**N-way K-shot**: N classes, K support examples per class. Common setups: 5-way 1-shot, 5-way 5-shot.

| Setup | Support Size | Use Case |
|-------|--------------|----------|
| 5-way 1-shot | 5 | Minimal data |
| 5-way 5-shot | 25 | More stable |
| 20-way 1-shot | 20 | Harder discrimination |

### Relation and Siamese Networks

**Relation networks** learn a relation module that scores (query, class_representation) pairs. **Siamese networks** use contrastive or triplet loss on pairs.

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, labels):
        distances = F.pairwise_distance(embedding1, embedding2, p=2)
        loss_positive = labels * distances.pow(2)
        loss_negative = (1 - labels) * F.relu(self.margin - distances).pow(2)
        return 0.5 * (loss_positive + loss_negative).mean()
```
