# Knowledge Distillation and Self-Supervised Learning

## Table of Contents
1. [Overview](#overview)
2. [Knowledge Distillation](#knowledge-distillation)
3. [Self-Supervised Learning](#self-supervised-learning)
4. [Contrastive Learning](#contrastive-learning)

---

## Overview

**Knowledge distillation** transfers knowledge from a large teacher to a small student. **Self-supervised learning** learns representations from unlabeled data via pretext tasks. **Contrastive learning** learns by contrasting positive pairs against negatives. All enable learning without full supervision or with limited labeled data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
```

---

## Knowledge Distillation

### Teacher-Student Framework

A **teacher** model (large, pre-trained) produces soft targets. A **student** model (small) is trained to match both hard labels and soft targets. The teacher is frozen during distillation.

```python
teacher = TeacherModel(num_classes=10)
student = StudentModel(num_classes=10)

for param in teacher.parameters():
    param.requires_grad = False
teacher.eval()
```

### Soft Targets and Temperature Scaling

**Soft targets** are class probabilities from the teacher. **Temperature** \(T\) softens the distribution: \( p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \). Higher \(T\) yields softer, more informative distributions.

```python
temperature = 4.0
teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
student_soft = F.log_softmax(student_outputs / temperature, dim=1)
```

| Temperature | Effect |
|-------------|--------|
| T=1 | Standard softmax |
| T>1 | Softer, reveals dark knowledge |
| T>>1 | Nearly uniform |

### KL Divergence Loss

The **distillation loss** is KL divergence between student and teacher soft outputs, scaled by \(T^2\) to preserve gradient magnitude.

```python
soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
soft_loss = soft_loss * (temperature ** 2)
```

### Combined Distillation Loss

Combine **soft loss** (KL) and **hard loss** (cross-entropy with labels) via weight \(\alpha\).

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        hard_loss = self.criterion(student_outputs, targets)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### Feature Distillation and Intermediate Layer Matching

**Feature distillation** matches intermediate representations. **Attention transfer** aligns attention maps (e.g., sum of squared activations over channels).

```python
class AttentionTransfer(nn.Module):
    def __init__(self, beta=1000):
        super().__init__()
        self.beta = beta

    def attention_map(self, feature_map):
        attention = torch.sum(feature_map ** 2, dim=1, keepdim=True)
        attention = F.normalize(attention.view(attention.size(0), -1), p=2, dim=1)
        return attention.view(attention.size(0), 1, feature_map.size(2), feature_map.size(3))

    def forward(self, student_features, teacher_features):
        loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_att = self.attention_map(s_feat)
            t_att = self.attention_map(t_feat)
            loss += F.mse_loss(s_att, t_att)
        return self.beta * loss
```

### Training Loop with Distillation

```python
def train_epoch(self, dataloader, use_attention_transfer=False):
    self.student.train()
    for data, targets in dataloader:
        data, targets = data.to(self.device), targets.to(self.device)
        with torch.no_grad():
            teacher_outputs, teacher_features = self.teacher(data, return_features=True)
        student_outputs, student_features = self.student(data, return_features=True)
        dist_loss = self.distillation_loss(student_outputs, teacher_outputs, targets)
        if use_attention_transfer:
            dist_loss += self.attention_transfer(student_features, teacher_features)
        self.optimizer.zero_grad()
        dist_loss.backward()
        self.optimizer.step()
```

---

## Self-Supervised Learning

### Pretext Tasks

**Pretext tasks** are auxiliary objectives that require no labels. The encoder learns useful representations; a linear probe evaluates quality on downstream tasks.

### Rotation Prediction

Predict the rotation angle (0, 90, 180, 270) applied to the image. The encoder must understand structure to solve this.

```python
class RotationPredictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.rotation_classifier = nn.Linear(encoder_dim, 4)

    def rotate_image(self, x, rotation):
        if rotation == 0:
            return x
        elif rotation == 90:
            return torch.rot90(x, k=1, dims=[-2, -1])
        elif rotation == 180:
            return torch.rot90(x, k=2, dims=[-2, -1])
        elif rotation == 270:
            return torch.rot90(x, k=3, dims=[-2, -1])

    def forward(self, x):
        rotations = [0, 90, 180, 270]
        all_images = []
        labels = []
        for i, r in enumerate(rotations):
            rotated = self.rotate_image(x, r)
            all_images.append(rotated)
            labels.extend([i] * x.size(0))
        all_images = torch.cat(all_images, dim=0)
        labels = torch.tensor(labels).to(x.device)
        features = self.encoder(all_images)
        return self.rotation_classifier(features), labels
```

### Jigsaw Puzzle

Shuffle image patches; predict the permutation. Encourages spatial reasoning.

```python
class JigsawPuzzle(nn.Module):
    def __init__(self, encoder, grid_size=3):
        super().__init__()
        self.encoder = encoder
        self.grid_size = grid_size
        self.permutations = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 6, 7, 8, 3, 4, 5],
            [2, 1, 0, 5, 4, 3, 8, 7, 6],
        ]
        self.permutation_classifier = nn.Linear(encoder_dim, len(self.permutations))

    def create_patches(self, x):
        patches = []
        h, w = x.shape[-2] // self.grid_size, x.shape[-1] // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = x[:, :, i*h:(i+1)*h, j*w:(j+1)*w]
                patches.append(patch)
        return patches

    def reconstruct_from_patches(self, patches, permutation):
        reordered = [patches[permutation[i]] for i in range(len(patches))]
        rows = [torch.cat(reordered[i*self.grid_size:(i+1)*self.grid_size], dim=-1)
                for i in range(self.grid_size)]
        return torch.cat(rows, dim=-2)
```

### Masked Image Modeling

**Masked image modeling** (e.g., MAE) masks patches and predicts the masked content. Requires understanding of local and global structure.

---

## Contrastive Learning

### SimCLR

**SimCLR** uses two augmented views of each image. Positive pairs are (x1, x2) from the same image; negatives are other samples in the batch. A **projection head** maps encoder output to a contrastive space.

```python
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        encoder_dim = self._get_encoder_dim()
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = F.normalize(self.projection_head(h1), dim=1)
        z2 = F.normalize(self.projection_head(h2), dim=1)
        return z1, z2
```

### NT-Xent Loss

**NT-Xent** (Normalized Temperature-scaled Cross Entropy) treats (z1_i, z2_i) as positive and all other pairs as negative. Similarity matrix is computed; labels indicate positive indices.

```python
def contrastive_loss(self, z1, z2):
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
    mask = torch.eye(2 * batch_size, dtype=bool).to(z1.device)
    similarity_matrix.masked_fill_(mask, -9e15)
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z1.device)
    return F.cross_entropy(similarity_matrix, labels)
```

### MoCo Concepts

**MoCo** (Momentum Contrast) maintains a queue of negative keys and a momentum-updated key encoder. Reduces need for large batch sizes.

```python
@torch.no_grad()
def _momentum_update_key_encoder(self):
    for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

@torch.no_grad()
def _dequeue_and_enqueue(self, keys):
    batch_size = keys.shape[0]
    ptr = int(self.queue_ptr)
    self.queue[:, ptr:ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % self.K
    self.queue_ptr[0] = ptr
```

### Projection Head

A **projection head** (MLP) maps encoder output to a lower-dimensional space where contrastive loss is applied. The encoder is used for downstream tasks; the head is often discarded.

```python
self.projection_head = nn.Sequential(
    nn.Linear(encoder_dim, encoder_dim),
    nn.ReLU(),
    nn.Linear(encoder_dim, projection_dim)
)
```

### Positive and Negative Pairs

| Pair Type | Definition |
|-----------|-------------|
| Positive | Two augmented views of the same image |
| Negative | Different images (or different augmented views of different images) |

In SimCLR, positives come from the same sample; negatives are all other samples in the batch (both views).

### InfoNCE Loss

**InfoNCE** is a general form of contrastive loss. For anchor \(a\), positive \(p\), and negatives \(n_1, \ldots, n_K\):

\[
\mathcal{L} = -\log \frac{\exp(\text{sim}(a,p)/\tau)}{\exp(\text{sim}(a,p)/\tau) + \sum_i \exp(\text{sim}(a,n_i)/\tau)}
\]

```python
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)
```

### Data Augmentation for Contrastive Learning

Strong augmentation is critical: random crop, color jitter, blur, horizontal flip. Two views are sampled independently.

```python
class ContrastiveAugmentation:
    def __call__(self, x):
        x = self.random_crop_and_resize(x)
        x = self.random_horizontal_flip(x)
        if random.random() > 0.2:
            x = self.color_jitter(x)
        if random.random() > 0.5:
            x = self.gaussian_blur(x)
        return x

x1 = augmentation(x)
x2 = augmentation(x)
z1, z2 = model(x1, x2)
loss = model.contrastive_loss(z1, z2)
```

### Linear Evaluation Protocol

Freeze the encoder and train a linear classifier on top. Accuracy measures representation quality.

```python
def linear_evaluation(encoder, train_dataset, test_dataset, num_classes=10, device='cuda'):
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    with torch.no_grad():
        dummy = torch.randn(1, 3, 32, 32).to(device)
        feature_dim = encoder(dummy).size(1)
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for epoch in range(10):
        for data, targets in DataLoader(train_dataset, batch_size=32, shuffle=True):
            data, targets = data.to(device), targets.to(device)
            with torch.no_grad():
                features = encoder(data)
            outputs = classifier(features)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in DataLoader(test_dataset, batch_size=32):
            data, targets = data.to(device), targets.to(device)
            features = encoder(data)
            _, predicted = classifier(features).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100. * correct / total
```
