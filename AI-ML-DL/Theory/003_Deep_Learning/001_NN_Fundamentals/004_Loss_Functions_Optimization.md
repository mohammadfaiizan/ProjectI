# Loss Functions and Optimization

## Table of Contents

1. [Introduction](#introduction)
2. [Regression Loss Functions](#regression-loss-functions)
3. [Classification Loss Functions](#classification-loss-functions)
4. [Specialized Loss Functions](#specialized-loss-functions)
5. [Contrastive and Metric Learning Losses](#contrastive-and-metric-learning-losses)
6. [Loss Function Properties](#loss-function-properties)
7. [Multi-Task Learning Losses](#multi-task-learning-losses)
8. [Loss Function Selection](#loss-function-selection)
9. [Optimization Considerations](#optimization-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Loss functions quantify the discrepancy between model predictions and true targets, providing the objective that optimization algorithms minimize during training. The choice of loss function fundamentally shapes what the model learns, how it generalizes, and its behavior on different types of errors. Understanding loss functions is crucial for effective model design and training.

This chapter covers the mathematical foundations, properties, and applications of major loss functions used in deep learning, from fundamental regression and classification losses to advanced techniques like focal loss, contrastive learning, and triplet loss.

## Regression Loss Functions

### Mean Squared Error (MSE)

The most common regression loss function:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

For vector predictions:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} ||\mathbf{y} - \hat{\mathbf{y}}||^2_2$$

**Properties**:
- Differentiable everywhere
- Penalizes large errors quadratically
- Sensitive to outliers
- Assumes Gaussian noise distribution

**Gradient**:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}_i} = -\frac{2}{n}(y_i - \hat{y}_i)$$

**Use Cases**:
- Continuous target prediction
- When errors should be penalized quadratically
- When data follows normal distribution

### Mean Absolute Error (MAE)

Also called L1 loss:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Properties**:
- Less sensitive to outliers than MSE
- Linear penalty for errors
- Non-differentiable at zero
- Assumes Laplacian noise distribution

**Gradient**:

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}_i} = \begin{cases}
-\frac{1}{n} & \text{if } y_i > \hat{y}_i \\
+\frac{1}{n} & \text{if } y_i < \hat{y}_i \\
0 & \text{if } y_i = \hat{y}_i \text{ (subgradient)}
\end{cases}$$

**Use Cases**:
- Robust regression (outlier resistance)
- When uniform penalty is desired
- Sparse regression problems

### Huber Loss

Combines MSE and MAE, robust to outliers:

$$\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{if } |y - \hat{y}| > \delta
\end{cases}$$

where $\delta$ is a hyperparameter (typically $1.0$).

**Properties**:
- Smooth transition between quadratic and linear
- Robust to outliers (like MAE)
- Differentiable everywhere
- Combines benefits of MSE and MAE

**Gradient**:

$$\frac{\partial \mathcal{L}_{\text{Huber}}}{\partial \hat{y}} = \begin{cases}
-(\hat{y} - y) & \text{if } |y - \hat{y}| \leq \delta \\
-\delta \cdot \text{sign}(\hat{y} - y) & \text{if } |y - \hat{y}| > \delta
\end{cases}$$

**Use Cases**:
- Regression with outliers
- When both accuracy and robustness matter
- Noisy data scenarios

### Smooth L1 Loss

Used in object detection (Faster R-CNN):

$$\mathcal{L}_{\text{SmoothL1}}(y, \hat{y}) = \begin{cases}
0.5(y - \hat{y})^2 / \beta & \text{if } |y - \hat{y}| < \beta \\
|y - \hat{y}| - 0.5\beta & \text{otherwise}
\end{cases}$$

where $\beta$ is typically $1.0$ (same as Huber with $\delta = \beta$).

## Classification Loss Functions

### Binary Cross-Entropy

For binary classification:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

where $\hat{y}_i \in (0,1)$ is the predicted probability.

**Properties**:
- Proper scoring rule
- Maximum likelihood under Bernoulli distribution
- Penalizes confident wrong predictions heavily
- Requires sigmoid activation

**Gradient**:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1-y_i}{1-\hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$$

**Use Cases**:
- Binary classification
- Multi-label classification (independent binary problems)

### Categorical Cross-Entropy

For multi-class classification:

$$\mathcal{L}_{\text{CCE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

where $C$ is the number of classes, and $\mathbf{y}_i$ is a one-hot vector.

**Properties**:
- Maximum likelihood under multinomial distribution
- Proper scoring rule
- Requires softmax activation
- Sum of probabilities equals 1

**Gradient**:

$$\frac{\partial \mathcal{L}_{\text{CCE}}}{\partial \hat{y}_{i,c}} = -\frac{y_{i,c}}{\hat{y}_{i,c}}$$

**Use Cases**:
- Multi-class classification
- When classes are mutually exclusive

### Sparse Categorical Cross-Entropy

When labels are integers (not one-hot):

$$\mathcal{L}_{\text{SparseCCE}} = -\frac{1}{n} \sum_{i=1}^{n} \log(\hat{y}_{i,y_i})$$

where $y_i \in \{0, 1, \ldots, C-1\}$ is the class index.

**Advantages**:
- More memory efficient (no one-hot encoding)
- Faster computation
- Same optimization objective as categorical cross-entropy

### Hinge Loss

Used in support vector machines:

$$\mathcal{L}_{\text{Hinge}} = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \hat{y}_i)$$

where $y_i \in \{-1, +1\}$.

**Properties**:
- Encourages margin maximization
- Non-differentiable at margin boundary
- Less sensitive to outliers than cross-entropy
- Used in SVMs

**Gradient**:

$$\frac{\partial \mathcal{L}_{\text{Hinge}}}{\partial \hat{y}_i} = \begin{cases}
-y_i & \text{if } y_i \hat{y}_i < 1 \\
0 & \text{if } y_i \hat{y}_i \geq 1
\end{cases}$$

**Use Cases**:
- Maximum margin classification
- When robustness to outliers is important

### Squared Hinge Loss

Smooth version of hinge loss:

$$\mathcal{L}_{\text{SquaredHinge}} = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \hat{y}_i)^2$$

**Properties**:
- Differentiable everywhere
- Stronger penalty than hinge loss
- Smoother optimization landscape

## Specialized Loss Functions

### Focal Loss

Addresses class imbalance by down-weighting easy examples:

$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

where:
- $p_t = \begin{cases} \hat{y} & \text{if } y=1 \\ 1-\hat{y} & \text{if } y=0 \end{cases}$
- $\alpha_t$ is a weighting factor (typically $\alpha$ for class 1, $1-\alpha$ for class 0)
- $\gamma$ is the focusing parameter (typically $2.0$)

**Properties**:
- Reduces contribution of easy examples
- Focuses learning on hard examples
- Effective for imbalanced datasets
- Used in object detection (RetinaNet)

**Gradient**:

More complex due to $(1-p_t)^\gamma$ term, but computable via chain rule.

**Use Cases**:
- Highly imbalanced datasets
- Object detection with many background examples
- When hard examples are more informative

### Dice Loss

Used in segmentation tasks:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2|X \cap Y| + \epsilon}{|X| + |Y| + \epsilon} = 1 - \frac{2\sum \hat{y}_i y_i + \epsilon}{\sum \hat{y}_i + \sum y_i + \epsilon}$$

where $\epsilon$ is a small constant for numerical stability.

**Properties**:
- Handles class imbalance naturally
- Focuses on overlap between prediction and ground truth
- Range: $[0, 1]$ (0 = perfect match)

**Use Cases**:
- Image segmentation
- When overlap is more important than pixel-wise accuracy

### Intersection over Union (IoU) Loss

For object detection and segmentation:

$$\mathcal{L}_{\text{IoU}} = 1 - \frac{|X \cap Y|}{|X \cup Y|} = 1 - \frac{\sum \min(\hat{y}_i, y_i)}{\sum \max(\hat{y}_i, y_i)}$$

**Properties**:
- Directly optimizes IoU metric
- Scale-invariant
- Non-differentiable at boundaries (requires smooth approximation)

## Contrastive and Metric Learning Losses

### Contrastive Loss

Learns representations by pulling similar examples together and pushing dissimilar ones apart:

$$\mathcal{L}_{\text{Contrastive}} = \frac{1}{2N} \sum_{i=1}^{N} [y_i d_i^2 + (1-y_i) \max(0, m - d_i)^2]$$

where:
- $d_i = ||f(\mathbf{x}_i^{(1)}) - f(\mathbf{x}_i^{(2)})||_2$ is the distance
- $y_i \in \{0,1\}$ indicates if pair is similar
- $m$ is the margin

**Properties**:
- Encourages small distances for similar pairs
- Enforces margin for dissimilar pairs
- Used in Siamese networks

### Triplet Loss

Uses triplets (anchor, positive, negative):

$$\mathcal{L}_{\text{Triplet}} = \frac{1}{N} \sum_{i=1}^{N} \max(0, d(a_i, p_i) - d(a_i, n_i) + m)$$

where:
- $a_i$ is the anchor
- $p_i$ is a positive example (same class)
- $n_i$ is a negative example (different class)
- $m$ is the margin
- $d(\cdot, \cdot)$ is distance metric

**Properties**:
- Relative comparison (not absolute distances)
- Requires careful triplet mining
- Effective for face recognition, metric learning

**Triplet Mining Strategies**:
- **Random**: Random triplets (inefficient)
- **Hard Negative**: Hardest negative for each anchor-positive pair
- **Semi-Hard**: Negatives that violate margin but are not hardest
- **Hard Positive**: Also mine hard positives

### N-Pair Loss

Generalization of triplet loss to multiple negatives:

$$\mathcal{L}_{\text{NPair}} = -\log \frac{\exp(f(\mathbf{x})^T f(\mathbf{x}^+))}{\exp(f(\mathbf{x})^T f(\mathbf{x}^+)) + \sum_{i=1}^{N-1} \exp(f(\mathbf{x})^T f(\mathbf{x}_i^-))}$$

**Properties**:
- More efficient than multiple triplet losses
- Uses all negatives simultaneously
- Better gradient signal

### InfoNCE Loss

Used in contrastive learning (SimCLR, MoCo):

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j^+) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

where:
- $\mathbf{z}_i, \mathbf{z}_j^+$ are positive pair embeddings
- $\text{sim}(\cdot, \cdot)$ is similarity (e.g., cosine)
- $\tau$ is temperature parameter
- Sum includes one positive and $N-1$ negatives

**Properties**:
- Maximizes mutual information
- Temperature controls concentration
- Foundation of modern self-supervised learning

## Loss Function Properties

### Convexity

**Convex Losses**:
- MSE (convex in predictions)
- Cross-entropy (convex in logits)
- Hinge loss (convex)

**Non-Convex Losses**:
- Most losses become non-convex when composed with neural networks
- Local minima exist in practice

### Differentiability

**Smooth Losses**:
- MSE, cross-entropy, Huber loss
- Enable gradient-based optimization

**Non-Smooth Losses**:
- MAE (non-differentiable at zero)
- Hinge loss (non-differentiable at margin)
- Require subgradients or smoothing

### Robustness

**Robust to Outliers**:
- MAE, Huber loss
- Less sensitive to extreme errors

**Sensitive to Outliers**:
- MSE (quadratic penalty)
- Cross-entropy (for extreme probabilities)

### Calibration

**Proper Scoring Rules**:
- Cross-entropy (log loss)
- Brier score
- Encourage calibrated probabilities

**Non-Proper**:
- Accuracy (not a loss, but not proper)
- Some custom losses

## Multi-Task Learning Losses

### Weighted Sum

Simple combination:

$$\mathcal{L}_{\text{Total}} = \sum_{t=1}^{T} \lambda_t \mathcal{L}_t$$

where $\lambda_t$ are task weights.

**Challenges**:
- Manual weight tuning
- Tasks may have different scales
- Tasks may conflict

### Uncertainty Weighting

Learns task weights:

$$\mathcal{L}_{\text{Total}} = \sum_{t=1}^{T} \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

where $\sigma_t$ are learnable uncertainty parameters.

**Advantages**:
- Automatic weight balancing
- Accounts for task uncertainty

### GradNorm

Balances gradients across tasks:

$$\mathcal{L}_{\text{GradNorm}} = \sum_{t} ||\nabla_{\mathbf{w}} \lambda_t \mathcal{L}_t|| - \bar{G}||_1$$

where $\bar{G}$ is the average gradient norm.

## Loss Function Selection

### Task Type Guidelines

**Regression**:
- MSE: Default, assumes Gaussian noise
- MAE: Robust to outliers
- Huber: Balanced robustness and smoothness

**Binary Classification**:
- Binary cross-entropy: Default
- Focal loss: Imbalanced data
- Hinge loss: Maximum margin

**Multi-Class Classification**:
- Categorical cross-entropy: Default
- Focal loss: Imbalanced classes

**Segmentation**:
- Dice loss: Overlap-focused
- Cross-entropy + Dice: Combined
- IoU loss: Direct metric optimization

**Object Detection**:
- Smooth L1: Bounding box regression
- Focal loss: Classification (imbalanced)
- IoU loss: Box overlap

### Data Characteristics

**Balanced Data**: Standard cross-entropy or MSE

**Imbalanced Data**: Focal loss, weighted cross-entropy, Dice loss

**Noisy Data**: Robust losses (MAE, Huber)

**Sparse Labels**: Focal loss, label smoothing

### Implementation Considerations

```python
import torch
import torch.nn as nn

# Regression
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.SmoothL1Loss()

# Classification
bce_loss = nn.BCELoss()
cce_loss = nn.CrossEntropyLoss()
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Custom loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

## Optimization Considerations

### Loss Landscape

Different losses create different optimization landscapes:
- Smooth losses: Easier optimization
- Non-smooth losses: May require special optimizers

### Gradient Properties

**Well-Behaved Gradients**:
- Cross-entropy: Stable gradients
- MSE: Linear gradients

**Challenging Gradients**:
- Focal loss: Can have large gradients for hard examples
- Contrastive losses: Require careful learning rate

### Learning Rate Interaction

Loss function choice affects optimal learning rate:
- Focal loss: May require lower learning rate
- Contrastive losses: Often need careful scheduling

### Regularization

Loss functions can include regularization:
- L2 regularization: $\mathcal{L} + \lambda ||\mathbf{w}||^2$
- Label smoothing: Soften one-hot targets
- Mixup: Interpolate between examples

## Key Takeaways

1. **Task-Specific Selection**: Loss function should match the task (regression vs. classification) and data characteristics (balanced vs. imbalanced).

2. **MSE and Cross-Entropy**: Fundamental losses for regression and classification, respectively, with strong theoretical foundations.

3. **Robust Losses**: MAE and Huber loss provide robustness to outliers, important for noisy real-world data.

4. **Focal Loss**: Powerful tool for imbalanced datasets, down-weighting easy examples to focus on hard cases.

5. **Contrastive Learning**: Triplet loss, InfoNCE, and related losses enable effective representation learning without explicit labels.

6. **Segmentation Losses**: Dice and IoU losses directly optimize overlap metrics, often more appropriate than pixel-wise losses.

7. **Loss Properties**: Understanding convexity, differentiability, and robustness helps select appropriate losses.

8. **Multi-Task Learning**: Combining multiple losses requires careful weighting, with uncertainty weighting providing automatic balancing.

9. **Optimization Interaction**: Loss choice affects optimization dynamics, requiring appropriate learning rates and optimizers.

10. **Empirical Validation**: While theory guides selection, empirical performance on validation data should drive final choices.
