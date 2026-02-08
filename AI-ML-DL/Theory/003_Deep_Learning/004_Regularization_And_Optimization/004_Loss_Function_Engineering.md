# Loss Function Engineering

## Table of Contents

1. [Introduction](#introduction)
2. [Custom Loss Design Principles](#custom-loss-design-principles)
3. [Metric Learning Losses](#metric-learning-losses)
4. [Contrastive Learning Losses](#contrastive-learning-losses)
5. [Knowledge Distillation Loss](#knowledge-distillation-loss)
6. [Auxiliary Loss Functions](#auxiliary-loss-functions)
7. [Multi-Task Loss Combination](#multi-task-loss-combination)
8. [Loss Function Selection](#loss-function-selection)
9. [Implementation Best Practices](#implementation-best-practices)
10. [Key Takeaways](#key-takeaways)

## Introduction

Loss functions define what models learn and how they optimize. Engineering effective loss functions is crucial for achieving desired model behavior, from standard classification to complex tasks like metric learning and knowledge distillation.

This chapter covers principles of loss function design, specialized losses for metric learning and contrastive learning, and practical considerations for implementing custom losses.

## Custom Loss Design Principles

### Alignment with Objective

Loss should match actual goal:
- **Classification**: Cross-entropy, focal loss
- **Regression**: MSE, MAE, Huber
- **Ranking**: Ranking losses
- **Detection**: IoU loss, focal loss

### Differentiability

Loss must be differentiable (almost everywhere):
- Enables gradient-based optimization
- Smooth gradients preferred
- Handle edge cases

### Numerical Stability

Prevent numerical issues:
- Avoid log(0)
- Clip extreme values
- Use stable implementations

### Scale and Magnitude

Loss scale affects:
- Learning rate sensitivity
- Gradient magnitudes
- Optimization dynamics

### Interpretability

Loss should be interpretable:
- Understand what it optimizes
- Relate to metrics
- Debug easily

## Metric Learning Losses

Metric learning learns distance functions.

### Triplet Loss

Enforces relative distances:

$$\mathcal{L}_{\text{triplet}} = \max(0, d(a,p) - d(a,n) + m)$$

where:
- $a$: Anchor
- $p$: Positive (same class)
- $n$: Negative (different class)
- $m$: Margin

**Properties**:
- Relative comparison
- Requires triplet mining
- Effective for face recognition

### Contrastive Loss

For pairs:

$$\mathcal{L}_{\text{contrastive}} = y \cdot d^2 + (1-y) \cdot \max(0, m-d)^2$$

where $y \in \{0,1\}$ indicates if pair is similar.

### N-Pair Loss

Generalizes triplet to multiple negatives:

$$\mathcal{L} = -\log \frac{\exp(f(x)^T f(x^+))}{\exp(f(x)^T f(x^+)) + \sum_{i=1}^{N-1} \exp(f(x)^T f(x_i^-))}$$

### Center Loss

Pulls examples to class centers:

$$\mathcal{L}_{\text{center}} = \frac{1}{2} \sum_{i=1}^{m} ||\mathbf{x}_i - \mathbf{c}_{y_i}||_2^2$$

where $\mathbf{c}_{y_i}$ is center of class $y_i$.

## Contrastive Learning Losses

### InfoNCE Loss

Maximizes mutual information:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j^+) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

where $\tau$ is temperature.

**Properties**:
- Foundation of self-supervised learning
- Temperature controls concentration
- Used in SimCLR, MoCo

### SimCLR Loss

Contrastive learning for visual representations:

- Positive pairs: Augmented versions of same image
- Negative pairs: Different images
- InfoNCE loss

### MoCo Loss

Momentum contrast:

- Maintains queue of negatives
- Momentum-updated encoder
- More stable training

### SwAV Loss

Swapped assignments:

- Cluster assignments
- Swap predictions
- No negative samples needed

## Knowledge Distillation Loss

Knowledge distillation transfers knowledge from teacher to student.

### Standard Distillation

$$\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}}(y_{\text{true}}, y_{\text{student}}) + (1-\alpha) \mathcal{L}_{\text{KL}}(y_{\text{teacher}} / T, y_{\text{student}} / T)$$

where:
- $T$: Temperature
- $\alpha$: Weighting factor
- $\mathcal{L}_{\text{KL}}$: KL divergence

### Temperature Scaling

Higher temperature softens probabilities:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**Benefits**:
- Reveals dark knowledge
- Smoother distributions
- Better transfer

### Feature Distillation

Match intermediate features:

$$\mathcal{L}_{\text{feat}} = ||f_{\text{teacher}}(\mathbf{x}) - f_{\text{student}}(\mathbf{x})||^2$$

### Attention Transfer

Match attention maps:

$$\mathcal{L}_{\text{attn}} = ||A_{\text{teacher}} - A_{\text{student}}||^2$$

## Auxiliary Loss Functions

Auxiliary losses help main task learning.

### Auxiliary Classification

Predict related quantities:

$$\mathcal{L}_{\text{aux}} = \mathcal{L}_{\text{main}} + \lambda \mathcal{L}_{\text{aux}}$$

**Example**: Predict depth + surface normals

### Consistency Loss

Enforce consistency:

$$\mathcal{L}_{\text{consistency}} = ||f(\mathbf{x}) - f(\text{augment}(\mathbf{x}))||^2$$

### Reconstruction Loss

Reconstruct inputs:

$$\mathcal{L}_{\text{recon}} = ||\mathbf{x} - \text{decode}(\text{encode}(\mathbf{x}))||^2$$

### Regularization Terms

Add regularization:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{L2}} + \lambda_2 \mathcal{L}_{\text{aux}}$$

## Multi-Task Loss Combination

### Weighted Sum

Simple combination:

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \lambda_t \mathcal{L}_t$$

**Challenges**:
- Manual weight tuning
- Tasks may have different scales
- Tasks may conflict

### Uncertainty Weighting

Learn task weights:

$$\mathcal{L} = \sum_{t=1}^{T} \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t$$

where $\sigma_t$ are learnable.

**Benefits**:
- Automatic balancing
- Accounts for uncertainty

### GradNorm

Balance gradient norms:

$$\mathcal{L}_{\text{grad}} = \sum_{t} ||\nabla_{\mathbf{w}} \lambda_t \mathcal{L}_t|| - \bar{G}||_1$$

where $\bar{G}$ is average gradient norm.

### Dynamic Weight Average

Adaptive weighting based on relative improvement.

## Loss Function Selection

### Task-Specific Guidelines

**Classification**:
- Cross-entropy: Standard
- Focal loss: Imbalanced data
- Label smoothing: Overconfidence

**Detection**:
- Smooth L1: Bounding boxes
- Focal loss: Classification
- IoU loss: Overlap

**Segmentation**:
- Cross-entropy: Standard
- Dice loss: Overlap-focused
- Focal + Dice: Combined

**Generation**:
- Adversarial loss: GANs
- Reconstruction loss: VAEs
- Perceptual loss: Quality

### Data Characteristics

**Balanced**: Standard losses
**Imbalanced**: Focal loss, weighted losses
**Noisy**: Robust losses (Huber, MAE)
**Sparse**: Focal loss, label smoothing

### Model Architecture

**CNNs**: Standard losses
**RNNs**: May need sequence losses
**Transformers**: Standard + auxiliary
**GANs**: Adversarial losses

## Implementation Best Practices

### Numerical Stability

```python
# Stable cross-entropy
def stable_cross_entropy(logits, targets):
    log_probs = logits - logits.max(dim=-1, keepdim=True)[0]
    log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
    return -log_probs.gather(1, targets.unsqueeze(1)).mean()
```

### Efficient Computation

- Vectorize operations
- Use built-in functions
- Avoid loops when possible

### Gradient Checking

Verify gradients:

```python
def check_gradient(loss_fn, params, eps=1e-7):
    numerical_grad = []
    for param in params:
        grad = torch.zeros_like(param)
        for i in range(param.numel()):
            param_flat = param.flatten()
            param_flat[i] += eps
            loss_plus = loss_fn()
            param_flat[i] -= 2*eps
            loss_minus = loss_fn()
            grad.flat[i] = (loss_plus - loss_minus) / (2*eps)
            param_flat[i] += eps
        numerical_grad.append(grad)
    return numerical_grad
```

### Monitoring

- Track loss components separately
- Visualize loss curves
- Monitor gradients
- Check for NaN/Inf

## Key Takeaways

1. **Design Principles**: Loss functions should align with objectives, be differentiable, numerically stable, and interpretable.

2. **Metric Learning**: Triplet loss, contrastive loss, and N-pair loss learn distance functions for similarity learning tasks.

3. **Contrastive Learning**: InfoNCE loss and variants enable self-supervised learning by contrasting positive and negative pairs.

4. **Knowledge Distillation**: Transfers knowledge from teacher to student using temperature-scaled soft targets and feature matching.

5. **Auxiliary Losses**: Additional losses for related tasks, consistency, or reconstruction help main task learning through shared representations.

6. **Multi-Task Combination**: Weighted sum, uncertainty weighting, and GradNorm balance multiple task losses effectively.

7. **Task-Specific Selection**: Choose losses based on task (classification, detection, segmentation) and data characteristics (balanced, imbalanced, noisy).

8. **Numerical Stability**: Implement losses with care for numerical stability, avoiding log(0) and extreme values.

9. **Monitoring**: Track loss components separately, visualize curves, and monitor gradients to understand training dynamics.

10. **Custom Design**: Effective loss engineering requires understanding task requirements, data characteristics, and optimization dynamics to design losses that guide models toward desired behavior.
