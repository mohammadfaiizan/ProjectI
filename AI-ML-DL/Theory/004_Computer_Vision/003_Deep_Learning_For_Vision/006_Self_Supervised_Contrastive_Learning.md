# Self-Supervised Contrastive Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Contrastive Learning Framework](#contrastive-learning-framework)
3. [SimCLR: A Simple Framework](#simclr-a-simple-framework)
4. [MoCo: Momentum Contrast](#moco-momentum-contrast)
5. [BYOL: Bootstrap Your Own Latent](#byol-bootstrap-your-own-latent)
6. [SwAV: Swapping Assignments](#swav-swapping-assignments)
7. [DINO: Knowledge Distillation](#dino-knowledge-distillation)
8. [Contrastive Loss: InfoNCE](#contrastive-loss-infonce)
9. [Pretext Tasks and Evaluation](#pretext-tasks-and-evaluation)
10. [Key Takeaways](#key-takeaways)

## Introduction

Self-supervised contrastive learning has revolutionized representation learning by enabling models to learn powerful visual representations without manual labels. These methods learn by contrasting positive pairs (different views of the same image) against negative pairs (views of different images), creating representations that capture semantic similarity.

The core idea is to learn an encoder $f_\theta$ that maps images to representations such that:

$$f_\theta(\mathbf{x}_i) \cdot f_\theta(\mathbf{x}_j) \approx \begin{cases}
\text{large} & \text{if } \mathbf{x}_i \text{ and } \mathbf{x}_j \text{ are similar} \\
\text{small} & \text{if } \mathbf{x}_i \text{ and } \mathbf{x}_j \text{ are dissimilar}
\end{cases}$$

Self-supervised learning creates supervision signals from data itself, typically through:
- **Data augmentation**: Different views of the same image are positives
- **Temporal consistency**: Video frames close in time are positives
- **Spatial consistency**: Nearby patches are positives

## Contrastive Learning Framework

### General Framework

Contrastive learning follows this pattern:

1. **Data augmentation**: Create two views $\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}_i'$ of image $\mathbf{x}_i$
2. **Encoding**: $\mathbf{z}_i = f_\theta(\tilde{\mathbf{x}}_i)$, $\mathbf{z}_i' = f_\theta(\tilde{\mathbf{x}}_i')$
3. **Projection**: $\mathbf{h}_i = g_\phi(\mathbf{z}_i)$, $\mathbf{h}_i' = g_\phi(\mathbf{z}_i')$ (optional)
4. **Contrastive loss**: Maximize similarity of positive pairs, minimize similarity of negative pairs

### Positive and Negative Pairs

- **Positive pairs**: $(\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}_i')$ - two augmented views of the same image
- **Negative pairs**: $(\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}_j)$ for $j \neq i$ - views of different images

### Key Components

**Encoder**: $f_\theta: \mathcal{X} \to \mathbb{R}^d$ maps images to representations
- Typically ResNet or Vision Transformer
- Output dimension $d = 2048$ or $512$

**Projection head**: $g_\phi: \mathbb{R}^d \to \mathbb{R}^{d_p}$ maps to projection space
- Small MLP: $d \to d \to d_p$ with ReLU
- Only used during pre-training, discarded for downstream tasks
- Output dimension $d_p = 128$ or $256$

**Augmentation strategy**: Creates diverse views
- Random crop and resize
- Random color jittering
- Random Gaussian blur
- Random horizontal flip
- Sometimes: grayscale conversion, solarization

## SimCLR: A Simple Framework

SimCLR (Simple Contrastive Learning of Representations) demonstrates that contrastive learning can achieve strong performance with a simple framework.

### Architecture

1. **Data augmentation**: Sample two augmentations $\mathcal{T}, \mathcal{T}'$ from augmentation family $\mathcal{A}$
2. **Encoding**: $\mathbf{z}_i = f_\theta(\mathcal{T}(\mathbf{x}_i))$, $\mathbf{z}_i' = f_\theta(\mathcal{T}'(\mathbf{x}_i))$
3. **Projection**: $\mathbf{h}_i = g_\phi(\mathbf{z}_i)$, $\mathbf{h}_i' = g_\phi(\mathbf{z}_i')$
4. **Loss**: InfoNCE loss on normalized projections

### Key Design Choices

**Large batch size**: Uses large batches (4096 or 8192) to provide many negatives
- More negatives improve contrastive learning
- Requires significant computational resources

**Strong augmentation**: Crucial for performance
- Color distortion (color jitter + color drop)
- Gaussian blur
- Random crop and resize

**Nonlinear projection head**: MLP with one hidden layer
- Improves representation quality
- Projection head is discarded after pre-training

**Normalized embeddings**: L2-normalize before computing similarity
- Prevents collapse to trivial solution
- Makes cosine similarity equivalent to dot product

### Training Objective

For a batch of $N$ images, creating $2N$ augmented views:

$$\mathcal{L}_{\text{SimCLR}} = -\frac{1}{2N}\sum_{i=1}^{N} \left[\log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_i') / \tau)}{\sum_{j=1}^{2N} \mathbb{1}_{j \neq i} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_j) / \tau)}\right]$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T\mathbf{v} / (||\mathbf{u}|| \cdot ||\mathbf{v}||)$ is cosine similarity and $\tau$ is temperature.

### Key Findings

- **Augmentation composition matters**: Combining multiple augmentations is crucial
- **Projection head helps**: Nonlinear projection improves learned representations
- **Larger models benefit more**: Contrastive learning scales well with model size
- **More training helps**: Longer training improves performance

## MoCo: Momentum Contrast

MoCo addresses the need for large batches by maintaining a queue of negative examples updated via momentum.

### Motivation

SimCLR requires large batches for many negatives. MoCo enables contrastive learning with small batches by:
- Maintaining a queue of negative representations
- Updating encoder via momentum to keep representations consistent

### Architecture

**Query encoder**: $f_q$ (with gradients, updated via backpropagation)
**Key encoder**: $f_k$ (momentum-updated, no gradients)

**Momentum update**:
$$\theta_k \leftarrow m \theta_k + (1-m) \theta_q$$

where $m \in [0, 1)$ is momentum coefficient (typically $m = 0.999$).

**Queue**: Maintains a queue of $K$ key representations (typically $K = 65536$)

### Algorithm

```
For each batch:
    1. Encode queries: q = f_q(aug(x))
    2. Encode keys: k = f_k(aug(x))
    3. Compute contrastive loss with queue negatives
    4. Update f_q via backpropagation
    5. Update f_k via momentum: θ_k ← m·θ_k + (1-m)·θ_q
    6. Enqueue current keys, dequeue oldest keys
```

### Advantages

- **Small batch size**: Works with batches as small as 256
- **Many negatives**: Queue provides thousands of negatives
- **Consistent representations**: Momentum keeps key encoder stable
- **Memory efficient**: Queue stored in memory, no need for large batches

### MoCo v2 Improvements

- Added projection head (like SimCLR)
- Stronger augmentation
- Cosine learning rate schedule
- Improved performance, especially with small batches

## BYOL: Bootstrap Your Own Latent

BYOL learns representations without negative examples by predicting one augmented view from another.

### Key Innovation

BYOL eliminates the need for negative examples, avoiding the contrastive learning requirement of many negatives.

### Architecture

**Online network**: $f_\theta, g_\theta, q_\theta$ (updated via gradients)
**Target network**: $f_\xi, g_\xi$ (updated via exponential moving average)

**Target network update**:
$$\xi \leftarrow \tau \xi + (1-\tau) \theta$$

where $\tau \in [0, 1]$ is target decay rate (typically $\tau = 0.996$).

### Prediction Task

Online network predicts target network's representation:

$$\mathbf{q}_\theta = q_\theta(g_\theta(f_\theta(\tilde{\mathbf{x}})))$$
$$\mathbf{z}_\xi' = g_\xi(f_\xi(\tilde{\mathbf{x}}'))$$

**Loss**: Mean squared error between normalized predictions:

$$\mathcal{L}_{\text{BYOL}} = ||\mathbf{q}_\theta / ||\mathbf{q}_\theta|| - \mathbf{z}_\xi' / ||\mathbf{z}_\xi'|| ||_2^2$$

Symmetrized: also predict $\mathbf{z}_\xi$ from $\mathbf{q}_\theta'$.

### Why It Works

- **Stop-gradient**: Target network provides stable targets (no gradients through $\xi$)
- **Momentum update**: Keeps target network slowly evolving
- **Prevents collapse**: Asymmetric architecture prevents trivial solution

### Advantages

- **No negatives**: Simpler, no need for large batches or queues
- **Simpler loss**: MSE instead of contrastive loss
- **Good performance**: Competitive with contrastive methods

## SwAV: Swapping Assignments

SwAV uses clustering to create assignments that are swapped between views, learning representations by predicting cluster assignments.

### Core Idea

Instead of contrasting embeddings directly, SwAV:
1. Computes cluster assignments (codes) for one view
2. Predicts these assignments from the other view
3. Swaps and predicts in both directions

### Architecture

**Prototypes**: $C$ learnable prototypes $\{\mathbf{c}_1, \ldots, \mathbf{c}_C\}$ (typically $C = 3000$)

**Codes**: Soft assignments $\mathbf{q}_i \in \Delta^C$ (probability simplex)

**Sinkhorn-Knopp**: Iterative algorithm to compute codes:
$$\mathbf{q}_i^{(t+1)} = \text{normalize}(\mathbf{q}_i^{(t)} \odot \exp(\mathbf{z}_i^T \mathbf{C} / \epsilon))$$

where $\mathbf{C}$ is prototype matrix and $\epsilon$ is temperature.

### Loss Function

Predict code of one view from embedding of other view:

$$\mathcal{L}_{\text{SwAV}} = \ell(\mathbf{z}_i, \mathbf{q}_j') + \ell(\mathbf{z}_j', \mathbf{q}_i)$$

where:
$$\ell(\mathbf{z}, \mathbf{q}) = -\sum_{k=1}^{C} q_k \log \frac{\exp(\mathbf{z}^T \mathbf{c}_k / \tau)}{\sum_{l=1}^{C} \exp(\mathbf{z}^T \mathbf{c}_l / \tau)}$$

### Multi-Crop

SwAV uses multiple crops (e.g., 2 large + 6 small) to increase data efficiency:
- Large crops: full resolution
- Small crops: lower resolution, more views per image

### Advantages

- **No negatives**: Clustering-based, no need for negative pairs
- **Data efficient**: Multi-crop increases views per image
- **Interpretable**: Prototypes can be visualized
- **Scalable**: Works well with various batch sizes

## DINO: Knowledge Distillation

DINO (Knowledge Distillation with No Labels) uses self-distillation with centering and sharpening to learn representations.

### Architecture

**Student network**: $f_\theta$ (updated via gradients)
**Teacher network**: $f_\xi$ (updated via exponential moving average)

**Centering**: Prevents collapse by centering teacher outputs:
$$\mathbf{c}_t \leftarrow m \mathbf{c}_{t-1} + (1-m) \frac{1}{B}\sum_{i=1}^{B} g_\xi(f_\xi(\mathbf{x}_i))$$

Centered teacher output: $g_\xi(f_\xi(\mathbf{x})) - \mathbf{c}_t$

**Sharpening**: Temperature-scaled softmax for teacher:
$$P_\xi(\mathbf{x}) = \text{softmax}((g_\xi(f_\xi(\mathbf{x})) - \mathbf{c}_t) / \tau_t)$$

where $\tau_t$ is teacher temperature (small, e.g., 0.04).

### Loss Function

Cross-entropy between student and teacher distributions:

$$\mathcal{L}_{\text{DINO}} = -\sum_{\mathbf{x} \in \{\tilde{\mathbf{x}}, \tilde{\mathbf{x}}'\}} P_\xi(\mathbf{x}) \log P_\theta(\mathbf{x})$$

where $P_\theta(\mathbf{x}) = \text{softmax}(g_\theta(f_\theta(\mathbf{x})) / \tau_s)$ with student temperature $\tau_s$ (larger, e.g., 0.1).

### Multi-Crop

Uses multiple views: 2 global crops + several local crops
- Global: full image
- Local: small patches

### Advantages

- **No negatives**: Self-distillation framework
- **Vision Transformers**: Works particularly well with ViT
- **Segmentation**: Produces features useful for dense prediction
- **Attention visualization**: Self-attention maps reveal object boundaries

## Contrastive Loss: InfoNCE

InfoNCE (Information Noise Contrastive Estimation) is the standard contrastive loss function.

### Formulation

For positive pair $(\mathbf{x}, \mathbf{x}^+)$ and negatives $\{\mathbf{x}_i^-\}_{i=1}^{N}$:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{z}, \mathbf{z}^+) / \tau)}{\exp(\text{sim}(\mathbf{z}, \mathbf{z}^+) / \tau) + \sum_{i=1}^{N} \exp(\text{sim}(\mathbf{z}, \mathbf{z}_i^-) / \tau)}$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T\mathbf{v}$ (after normalization) and $\tau$ is temperature.

### Information-Theoretic Interpretation

InfoNCE lower-bounds mutual information:

$$I(\mathbf{x}; \mathbf{x}^+) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$$

Maximizing InfoNCE maximizes mutual information between positive pairs.

### Temperature Parameter

Temperature $\tau$ controls concentration:
- **Small $\tau$**: Sharper distribution, harder negatives matter more
- **Large $\tau$**: Softer distribution, easier optimization

Typical values: $\tau \in [0.05, 0.2]$, often $\tau = 0.07$ or $0.1$.

### Variants

**NT-Xent** (Normalized Temperature-scaled Cross Entropy): Same as InfoNCE with normalization

**Triplet loss**: Older contrastive loss:
$$\mathcal{L}_{\text{triplet}} = \max(0, \text{sim}(\mathbf{z}, \mathbf{z}^-) - \text{sim}(\mathbf{z}, \mathbf{z}^+) + \text{margin})$$

## Pretext Tasks and Evaluation

### Pretext Tasks

Self-supervised learning creates supervision through pretext tasks:

**Instance discrimination**: Each image is its own class
- Contrastive learning (SimCLR, MoCo)
- Non-contrastive (BYOL, SwAV, DINO)

**Jigsaw puzzle**: Reassemble shuffled image patches

**Rotation prediction**: Predict rotation angle applied to image

**Colorization**: Predict color from grayscale

**Inpainting**: Predict masked regions

**Relative patch location**: Predict spatial relationship between patches

### Evaluation: Linear Probing

**Frozen features**: Freeze encoder $f_\theta$, train linear classifier on top:

$$\min_{\mathbf{W}} \sum_{i=1}^{n} \ell(\mathbf{W}^T f_\theta(\mathbf{x}_i), y_i)$$

**Metrics**: Top-1 accuracy on ImageNet validation set

**Rationale**: Good features should enable simple linear separation

### Evaluation: Fine-Tuning

**Full fine-tuning**: Update all parameters:

$$\min_{\theta, \mathbf{W}} \sum_{i=1}^{n} \ell(\mathbf{W}^T f_\theta(\mathbf{x}_i), y_i)$$

**Metrics**: Top-1 accuracy, often higher than linear probing

### Evaluation: Transfer Learning

Evaluate on downstream tasks:
- **Object detection**: COCO, Pascal VOC
- **Semantic segmentation**: ADE20K, Cityscapes
- **Few-shot learning**: $k$-shot classification

### Representation Quality Metrics

**Linear separability**: Train linear classifier, measure accuracy

**Nearest neighbor retrieval**: Find similar images in embedding space

**Clustering**: Apply K-means, measure cluster quality

**Visualization**: t-SNE or UMAP of representations

## Key Takeaways

1. **Self-supervised contrastive learning** learns representations by contrasting positive pairs (augmented views of same image) against negative pairs (views of different images), eliminating need for manual labels.

2. **SimCLR** demonstrates strong performance with simple framework: strong augmentation, large batches, nonlinear projection head, and InfoNCE loss, showing that composition of augmentations is crucial.

3. **MoCo** enables contrastive learning with small batches by maintaining a queue of negative examples and updating the key encoder via momentum, providing consistent representations without large batch requirements.

4. **BYOL** eliminates negative examples entirely by predicting one augmented view from another using online and target networks, with stop-gradient preventing collapse to trivial solutions.

5. **SwAV** uses clustering to create swap assignments between views, learning by predicting cluster codes, and employs multi-crop strategy to increase data efficiency without requiring negatives.

6. **DINO** uses self-distillation with centering and sharpening to prevent collapse, works particularly well with Vision Transformers, and produces features useful for dense prediction tasks like segmentation.

7. **InfoNCE loss** maximizes mutual information between positive pairs, with temperature parameter controlling concentration, and serves as the standard contrastive learning objective.

8. **Pretext tasks** create supervision signals from data itself (instance discrimination, jigsaw, rotation prediction), with contrastive methods focusing on instance discrimination through data augmentation.

9. **Evaluation** typically uses linear probing (frozen features + linear classifier) and fine-tuning (updating all parameters) on ImageNet and downstream tasks to measure representation quality.

10. **Self-supervised learning** has achieved performance competitive with supervised pre-training, enabling learning from unlabeled data at scale and transferring to diverse downstream tasks, revolutionizing representation learning in computer vision.
