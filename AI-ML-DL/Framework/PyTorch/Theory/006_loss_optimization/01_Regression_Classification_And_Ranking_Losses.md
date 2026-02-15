# Regression, Classification, and Ranking Losses

## Table of Contents
1. [Overview](#overview)
2. [Regression Losses](#regression-losses)
3. [Classification Losses](#classification-losses)
4. [Ranking and Metric Learning Losses](#ranking-and-metric-learning-losses)
5. [KLDivLoss and Related Losses](#kldivloss-and-related-losses)
6. [Contrastive and Hinge Embedding Losses](#contrastive-and-hinge-embedding-losses)

---

## Overview

Loss functions measure the discrepancy between model predictions and ground truth. PyTorch provides built-in losses in `torch.nn` with functional counterparts in `torch.nn.functional`. Choosing the right loss is critical for training success.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Regression Losses

### MSELoss (Mean Squared Error)

**MSELoss** penalizes squared differences. Formula: \( L = \frac{1}{n}\sum_{i}(y_i - \hat{y}_i)^2 \). Sensitive to outliers; use when large errors must be heavily penalized.

```python
mse_loss = nn.MSELoss()
mse_loss_no_reduction = nn.MSELoss(reduction='none')
mse_loss_sum = nn.MSELoss(reduction='sum')

predictions = torch.randn(10, 5)
targets = torch.randn(10, 5)

mse_mean = mse_loss(predictions, targets)
mse_none = mse_loss_no_reduction(predictions, targets)
mse_manual = torch.mean((predictions - targets) ** 2)
mse_functional = F.mse_loss(predictions, targets)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `reduction` | 'mean' | 'none', 'mean', or 'sum' |

### L1Loss (Mean Absolute Error)

**L1Loss** uses absolute differences. Formula: \( L = \frac{1}{n}\sum_{i}|y_i - \hat{y}_i| \). Robust to outliers; gradients are constant in magnitude.

```python
mae_loss = nn.L1Loss()
mae_mean = mae_loss(predictions, targets)
mae_manual = torch.mean(torch.abs(predictions - targets))
mae_functional = F.l1_loss(predictions, targets)
```

### SmoothL1Loss (Huber-like)

**SmoothL1Loss** combines MSE and MAE: quadratic for small errors, linear for large. Formula: \( L = \frac{1}{n}\sum_{i} \begin{cases} 0.5(x_i - y_i)^2/\beta & \text{if } |x_i - y_i| < \beta \\ |x_i - y_i| - 0.5\beta & \text{otherwise} \end{cases} \)

```python
smooth_l1_loss = nn.SmoothL1Loss()
smooth_l1_loss_beta = nn.SmoothL1Loss(beta=2.0)
smooth_l1_mean = smooth_l1_loss(predictions, targets)
smooth_l1_functional = F.smooth_l1_loss(predictions, targets)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `beta` | 1.0 | Threshold between quadratic and linear |
| `reduction` | 'mean' | 'none', 'mean', or 'sum' |

### HuberLoss

**HuberLoss** is similar to SmoothL1 with parameter \(\delta\). PyTorch does not provide it built-in; implement as:

```python
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input_tensor, target):
        diff = torch.abs(input_tensor - target)
        loss = torch.where(diff <= self.delta,
                          0.5 * diff ** 2,
                          self.delta * (diff - 0.5 * self.delta))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss
```

### Use Case Summary

| Loss | Use Case |
|------|----------|
| MSE | Penalize large errors heavily; Gaussian noise assumption |
| MAE | Robust to outliers; sparse gradients acceptable |
| SmoothL1/Huber | Balance MSE and MAE; object detection (bounding boxes) |

---

## Classification Losses

### CrossEntropyLoss

**CrossEntropyLoss** combines LogSoftmax and NLLLoss. Expects **raw logits** and integer class indices. Formula: \( L = -\log\frac{\exp(x_{class})}{\sum_j \exp(x_j)} \)

```python
ce_loss = nn.CrossEntropyLoss()
ce_loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
ce_loss_weighted = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

batch_size, num_classes = 10, 5
logits = torch.randn(batch_size, num_classes)
targets = torch.randint(0, num_classes, (batch_size,))

ce_mean = ce_loss(logits, targets)
ce_smooth = ce_loss_smooth(logits, targets)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `weight` | None | Per-class weights for imbalanced data |
| `ignore_index` | -100 | Index to exclude from loss |
| `reduction` | 'mean' | 'none', 'mean', or 'sum' |
| `label_smoothing` | 0.0 | Smoothing factor (0 to 1) |

### NLLLoss

**NLLLoss** expects **log-probabilities** (output of LogSoftmax). Equivalent to CrossEntropy when used with log_softmax.

```python
nll_loss = nn.NLLLoss()
log_probs = F.log_softmax(logits, dim=1)
nll_mean = nll_loss(log_probs, targets)
ce_manual = F.nll_loss(log_probs, targets)
```

### BCEWithLogitsLoss

**BCEWithLogitsLoss** applies sigmoid internally for numerical stability. Preferred over BCELoss for binary classification.

```python
bce_logits_loss = nn.BCEWithLogitsLoss()
binary_logits = torch.randn(10, 1)
binary_targets = torch.randint(0, 2, (10, 1)).float()
bce_mean = bce_logits_loss(binary_logits, binary_targets)
```

### BCELoss

**BCELoss** requires probabilities in [0, 1]. Use BCEWithLogitsLoss instead for stability.

```python
bce_loss = nn.BCELoss()
sigmoid_probs = torch.sigmoid(binary_logits)
bce_manual = F.binary_cross_entropy(sigmoid_probs, binary_targets)
```

### Multi-Label Classification

For multiple independent binary labels per sample:

```python
multi_label_logits = torch.randn(10, 5)
multi_label_targets = torch.randint(0, 2, (10, 5)).float()
multi_bce = bce_logits_loss(multi_label_logits, multi_label_targets)
builtin_ml_loss = nn.MultiLabelSoftMarginLoss()
builtin_result = builtin_ml_loss(multi_label_logits, multi_label_targets)
```

---

## Ranking and Metric Learning Losses

### TripletMarginLoss

**TripletMarginLoss** encourages anchor-positive distance to be smaller than anchor-negative by margin. Formula: \( L = \max(0, d(a,p) - d(a,n) + \text{margin}) \)

```python
triplet_loss = nn.TripletMarginLoss(margin=1.0)
anchor = torch.randn(10, 64)
positive = torch.randn(10, 64)
negative = torch.randn(10, 64)
triplet_result = triplet_loss(anchor, positive, negative)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `margin` | 1.0 | Minimum distance margin |
| `p` | 2 | Norm for distance (2 for L2) |
| `swap` | False | Use swapped distances |
| `reduction` | 'mean' | 'none', 'mean', or 'sum' |

### MarginRankingLoss

**MarginRankingLoss** for pairwise ranking: \( L = \max(0, -\text{target} \cdot (x_1 - x_2) + \text{margin}) \). Target: 1 if x1 > x2, -1 if x2 > x1.

```python
margin_loss = nn.MarginRankingLoss(margin=1.0)
x1 = torch.randn(10)
x2 = torch.randn(10)
y = torch.randint(-1, 2, (10,), dtype=torch.float)
margin_result = margin_loss(x1, x2, y)
```

### CosineEmbeddingLoss

**CosineEmbeddingLoss** measures cosine similarity. Use for learning embeddings where direction matters.

```python
cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
input1 = torch.randn(10, 64)
input2 = torch.randn(10, 64)
targets_cosine = torch.randint(-1, 2, (10,)).float()
cosine_result = cosine_loss(input1, input2, targets_cosine)
```

### Custom Triplet with Hard Mining

```python
class TripletLossWithHardMining(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings, labels):
        distances = torch.cdist(embeddings, embeddings, p=2)
        batch_size = embeddings.size(0)
        losses = []
        for i in range(batch_size):
            positive_mask = (labels == labels[i]) & (torch.arange(batch_size) != i)
            negative_mask = labels != labels[i]
            if positive_mask.any() and negative_mask.any():
                hard_pos_dist = distances[i][positive_mask].max()
                hard_neg_dist = distances[i][negative_mask].min()
                loss = F.relu(hard_pos_dist - hard_neg_dist + self.margin)
                losses.append(loss)
        if losses:
            total = torch.stack(losses)
            return total.mean() if self.reduction == 'mean' else total.sum()
        return torch.tensor(0.0, requires_grad=True)
```

---

## KLDivLoss and Related Losses

### KLDivLoss

**KLDivLoss** measures Kullback-Leibler divergence. Inputs must be log-probabilities and target probabilities.

```python
kl_loss = nn.KLDivLoss(reduction='batchmean')
log_probs = F.log_softmax(logits, dim=1)
target_probs = F.softmax(torch.randn_like(logits), dim=1)
kl_result = kl_loss(log_probs, target_probs)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `reduction` | 'mean' | 'none', 'batchmean', 'sum', 'mean' |

### ListNet (KL-based Ranking)

```python
class ListNetLoss(nn.Module):
    def forward(self, predicted_scores, target_scores):
        pred_probs = F.softmax(predicted_scores, dim=-1)
        target_probs = F.softmax(target_scores, dim=-1)
        loss = F.kl_div(torch.log(pred_probs + 1e-8), target_probs, reduction='none')
        return loss.sum(dim=-1).mean()
```

---

## Contrastive and Hinge Embedding Losses

### Contrastive Loss

For siamese networks: similar pairs minimize distance; dissimilar pairs maximize distance up to margin.

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embedding1, embedding2, labels):
        distances = F.pairwise_distance(embedding1, embedding2, p=2)
        pos_loss = labels * distances.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        loss = pos_loss + neg_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()
```

### HingeEmbeddingLoss

**HingeEmbeddingLoss** for learning embeddings: \( L = \begin{cases} x & \text{if } y == 1 \\ \max(0, \text{margin} - x) & \text{if } y == -1 \end{cases} \)

```python
hinge_loss = nn.HingeEmbeddingLoss(margin=1.0)
input_tensor = torch.randn(10, 64)
labels_hinge = torch.randint(-1, 2, (10,)).float()
hinge_result = hinge_loss(input_tensor, labels_hinge)
```

### Summary Table

| Loss | Domain | Key Parameters |
|------|--------|-----------------|
| TripletMarginLoss | Metric learning | margin, p |
| MarginRankingLoss | Pairwise ranking | margin |
| CosineEmbeddingLoss | Embeddings | margin |
| ContrastiveLoss | Siamese | margin |
| HingeEmbeddingLoss | Embeddings | margin |
