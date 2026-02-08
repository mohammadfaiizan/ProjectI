# Self-Supervised Learning Trends

## Table of Contents

1. [Introduction](#introduction)
2. [Contrastive Learning Fundamentals](#contrastive-learning-fundamentals)
3. [SimCLR: Simple Framework for Contrastive Learning](#simclr-simple-framework-for-contrastive-learning)
4. [MoCo: Momentum Contrast](#moco-momentum-contrast)
5. [BYOL: Bootstrap Your Own Latent](#byol-bootstrap-your-own-latent)
6. [Masked Image Modeling](#masked-image-modeling)
7. [Masked Language Modeling](#masked-language-modeling)
8. [Predictive Coding and JEPA](#predictive-coding-and-jepa)
9. [Representation Quality Metrics](#representation-quality-metrics)
10. [Key Takeaways](#key-takeaways)

## Introduction

Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations from unlabeled data. By designing pretext tasks that leverage the structure inherent in data, SSL methods can learn rich representations that transfer well to downstream tasks.

Recent advances in SSL have achieved performance competitive with or exceeding supervised learning, particularly in computer vision and natural language processing. Key trends include contrastive learning, masked modeling, and predictive coding approaches.

Key research directions:
- How to design effective pretext tasks?
- How to learn robust representations without negative samples?
- How to scale SSL to larger models and datasets?
- How to evaluate representation quality?

## Contrastive Learning Fundamentals

Contrastive learning learns representations by contrasting positive and negative pairs of examples.

### Core Principle

**Goal**: Learn representations where similar examples are close and dissimilar examples are far apart in the representation space.

**Positive pairs**: Examples that should be similar (e.g., different augmentations of the same image)
**Negative pairs**: Examples that should be dissimilar (e.g., different images)

### Contrastive Loss

Given a query $q$ and positive key $k^+$ and negative keys $k^-_i$:

$$\mathcal{L}_{contrast} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_i \exp(q \cdot k^-_i / \tau)}$$

where $\tau$ is a temperature parameter.

### Key Components

**Data augmentation**: Creates positive pairs
- **Image**: Random crop, color jitter, rotation, blur
- **Text**: Masking, permutation, back-translation

**Encoder**: Maps inputs to representations
- **Architecture**: ResNet, Vision Transformer, etc.
- **Output**: Normalized feature vectors

**Projection head**: Maps representations to contrastive space
- **Purpose**: Improve contrastive learning
- **Architecture**: MLP with normalization

### Challenges

**Negative sampling**: Need many negative examples
**Collapse**: Model may collapse to trivial solution
**Computational cost**: Contrasting with many negatives is expensive

## SimCLR: Simple Framework for Contrastive Learning

SimCLR (Chen et al., 2020) introduced a simple yet effective framework for contrastive learning of visual representations.

### Framework

**Components**:
1. **Data augmentation**: Random crop + resize, color distortion, Gaussian blur
2. **Base encoder**: ResNet or other CNN
3. **Projection head**: MLP with one hidden layer
4. **Contrastive loss**: NT-Xent (normalized temperature-scaled cross-entropy)

### Architecture

```
Input x → Augment → x_i, x_j → Encoder f(·) → h_i, h_j → Projection g(·) → z_i, z_j
```

**Loss function**:
$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where $\text{sim}(u, v) = u^T v / ||u|| ||v||$ is cosine similarity.

### Key Findings

**Data augmentation is critical**:
- Composition of augmentations matters
- Color distortion is particularly important
- Stronger augmentation improves performance

**Projection head helps**:
- Learning in projection space improves representation quality
- Can discard projection head for downstream tasks

**Larger batch sizes help**:
- More negatives improve contrastive learning
- Batch size of 4096+ works well

**Larger models benefit more**:
- Larger encoders show larger improvements from SSL

### Training Details

- **Batch size**: 4096 (or larger)
- **Learning rate**: 0.3 × batch_size / 256
- **Temperature**: $\tau = 0.5$
- **Epochs**: 100-1000

### Performance

- **ImageNet**: 76.5% top-1 accuracy (linear evaluation)
- **Transfer**: Strong performance on many downstream tasks
- **Efficiency**: Competitive with supervised pre-training

## MoCo: Momentum Contrast

MoCo (He et al., 2020) addresses the challenge of maintaining a large and consistent dictionary of negative examples for contrastive learning.

### Motivation

**Problem**: Contrastive learning needs many negatives, but:
- End-to-end backpropagation limits batch size
- Memory constraints prevent large batches
- Inconsistent negatives hurt learning

**Solution**: Maintain a momentum-updated queue of negative examples.

### Architecture

**Key components**:
1. **Query encoder**: $f_q$ (updated by backpropagation)
2. **Key encoder**: $f_k$ (updated by momentum)
3. **Queue**: Stores encoded keys as negatives
4. **Momentum update**: $f_k \leftarrow m f_k + (1-m) f_q$

**Momentum coefficient**: $m \in [0, 1)$ (typically 0.999)

### Loss Function

$$\mathcal{L}_{MoCo} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{k^- \in \text{queue}} \exp(q \cdot k^- / \tau)}$$

### Advantages

**Large dictionary**: Queue can store many negatives (e.g., 65536)
**Consistent negatives**: Momentum update provides stable negatives
**Memory efficient**: Only need to encode current batch

### MoCo v2 Improvements

- **MLP projection head**: Improves representation quality
- **Stronger augmentation**: Better data augmentation
- **Cosine learning rate schedule**: Improves training

### Performance

- **ImageNet**: 71.1% top-1 accuracy (MoCo v2)
- **Transfer**: Strong performance on detection and segmentation
- **Efficiency**: More efficient than SimCLR (smaller batches)

## BYOL: Bootstrap Your Own Latent

BYOL (Grill et al., 2020) learns representations without negative examples, avoiding the need for large batches or negative sampling.

### Key Innovation

**No negatives**: Learn by predicting one augmented view from another
**Bootstrap**: Online network learns from target network
**Momentum update**: Target network updated slowly

### Architecture

**Two networks**:
- **Online network**: $f_\theta$ (encoder + predictor)
- **Target network**: $f_\xi$ (encoder only, momentum-updated)

**Loss function**:
$$\mathcal{L}_{BYOL} = ||q_\theta(z_\theta) - \bar{z}_\xi||_2^2$$

where:
- $z_\theta = f_\theta(x)$: Online representation
- $\bar{z}_\xi = f_\xi(x')$: Target representation (from augmented view)
- $q_\theta$: Predictor (only in online network)

**Symmetry**: Also predict $q_\theta(z'_\theta)$ from $\bar{z}_\xi$.

### Why It Works

**Prevention of collapse**:
- **Stop-gradient**: Prevents trivial solution
- **Momentum update**: Provides stable target
- **Predictor**: Prevents direct copying

**Theoretical understanding**:
- BYOL minimizes distance between representations
- Stop-gradient prevents collapse
- Augmentation provides diversity

### Advantages

**No negatives**: Simpler, more efficient
**Smaller batches**: Works with batch size 256
**Better performance**: Often outperforms contrastive methods

### Limitations

**Sensitivity**: Can be sensitive to hyperparameters
**Theoretical gaps**: Why it works is not fully understood
**Augmentation dependence**: Relies heavily on data augmentation

## Masked Image Modeling

Masked image modeling (MIM) learns representations by predicting masked patches of images, similar to masked language modeling.

### Motivation

**Success in NLP**: Masked language modeling (BERT) is highly effective
**Natural extension**: Apply similar approach to vision
**Rich supervision**: Predicting pixels/patches provides rich learning signal

### Approaches

**Pixel-level prediction**: Predict raw pixel values
**Token-level prediction**: Predict discrete tokens
**Feature-level prediction**: Predict features from teacher model

### MAE: Masked Autoencoders

MAE (He et al., 2021) uses an asymmetric encoder-decoder architecture.

**Architecture**:
- **Encoder**: Vision Transformer (ViT), only processes visible patches
- **Decoder**: Lightweight decoder, reconstructs all patches
- **Masking**: High masking ratio (75%)

**Loss function**:
$$\mathcal{L}_{MAE} = \sum_{i \in \text{masked}} ||\text{decoder}(z, \text{mask}_i) - x_i||_2^2$$

**Key design choices**:
- **High masking ratio**: 75% (vs 15% in BERT)
- **Asymmetric architecture**: Encoder only sees visible patches
- **Simple decoder**: Lightweight, can be discarded

### BEiT: BERT Pre-Training of Image Transformers

BEiT (Bao et al., 2021) predicts discrete visual tokens.

**Two-stage process**:
1. **Tokenizer**: Learn discrete visual tokens (e.g., dVAE)
2. **Pre-training**: Predict masked tokens

**Loss function**:
$$\mathcal{L}_{BEiT} = -\sum_{i \in \text{masked}} \log p(\text{token}_i | \text{context})$$

**Advantages**:
- Discrete tokens provide better learning signal
- Similar to BERT, well-understood

### iBOT: Image BERT Pre-Training

iBOT (Zhou et al., 2021) combines masked image modeling with contrastive learning.

**Two objectives**:
1. **Masked image modeling**: Predict masked patches
2. **Contrastive learning**: Contrastive loss on global views

**Benefits**: Combines benefits of both approaches

## Masked Language Modeling

Masked language modeling (MLM) is the pre-training objective used in BERT and many subsequent language models.

### BERT Objective

**Masked language modeling**: Predict masked tokens from context
**Next sentence prediction**: Predict if sentence B follows sentence A (removed in later models)

**Masking strategy**:
- 15% of tokens are masked
- 80% replaced with [MASK]
- 10% replaced with random token
- 10% unchanged

**Loss function**:
$$\mathcal{L}_{MLM} = -\sum_{i \in \text{masked}} \log p(x_i | x_{\backslash i})$$

### Variations

**RoBERTa**: Removes NSP, uses dynamic masking, longer training
**ALBERT**: Factorized embedding, parameter sharing, sentence-order prediction
**ELECTRA**: Replaced token detection (more efficient)

### Advantages

**Bidirectional**: Can use both left and right context
**Efficient**: Only predicts masked tokens (15% of tokens)
**Effective**: Strong performance on many NLP tasks

### Limitations

**Pre-training/inference mismatch**: [MASK] token not seen during fine-tuning
**Position bias**: May learn position-specific patterns

## Predictive Coding and JEPA

Predictive coding learns representations by predicting future or missing information.

### Predictive Coding

**Core idea**: Learn representations that enable prediction
**Applications**: Video prediction, future frame prediction, temporal modeling

**Loss function**:
$$\mathcal{L}_{predictive} = \sum_t ||\text{predict}(x_{t+1} | x_{\leq t}) - x_{t+1}||_2^2$$

### JEPA: Joint-Embedding Predictive Architecture

JEPA (LeCun, 2022) predicts representations rather than pixels.

**Key components**:
1. **Encoder**: Maps inputs to representations
2. **Predictor**: Predicts target representation from context
3. **Target encoder**: Provides target representation (momentum-updated)

**Loss function**:
$$\mathcal{L}_{JEPA} = ||\text{predictor}(z_{\text{context}}) - z_{\text{target}}||_2^2$$

**Advantages**:
- Predicts abstract representations (not pixels)
- More efficient and scalable
- Better generalization

### I-JEPA

I-JEPA (Assran et al., 2023) applies JEPA to images.

**Architecture**:
- **Context encoder**: Processes visible patches
- **Target encoder**: Processes target patches (momentum-updated)
- **Predictor**: Predicts target representations

**Masking**: Predicts multiple target blocks from context

**Performance**: Strong performance with efficient training

## Representation Quality Metrics

Evaluating representation quality is crucial for understanding and improving SSL methods.

### Linear Evaluation

**Protocol**:
1. Freeze encoder
2. Train linear classifier on frozen features
3. Evaluate on downstream task

**Advantages**: Simple, fast, standard
**Limitations**: May not capture all representation quality

### Fine-Tuning Evaluation

**Protocol**:
1. Fine-tune entire model on downstream task
2. Evaluate performance

**Advantages**: More comprehensive evaluation
**Limitations**: More expensive, hyperparameter sensitive

### Transfer Learning

**Protocol**: Evaluate on multiple downstream tasks
**Metrics**: Average performance, number of tasks improved

### Representation Analysis

**Nearest neighbors**: Visualize nearest neighbors in representation space
**t-SNE/UMAP**: Visualize representation space
**CKA**: Centered kernel alignment (measures similarity)

### Downstream Tasks

**Image classification**: ImageNet, CIFAR-10/100
**Object detection**: COCO, Pascal VOC
**Semantic segmentation**: ADE20K, Cityscapes
**Few-shot learning**: Few-shot benchmarks

### Metrics

**Top-1/Top-5 accuracy**: Classification accuracy
**mAP**: Mean average precision (detection)
**mIoU**: Mean intersection over union (segmentation)
**FID**: Fréchet Inception Distance (generation)

## Key Takeaways

1. **Self-supervised learning** enables learning rich representations from unlabeled data, achieving performance competitive with supervised learning.

2. **Contrastive learning** learns representations by contrasting positive and negative pairs, with SimCLR showing the importance of data augmentation and large batches.

3. **MoCo** addresses contrastive learning challenges by maintaining a momentum-updated queue of negatives, enabling large dictionaries with smaller batches.

4. **BYOL** demonstrates that negative examples are not necessary, learning representations by predicting augmented views through bootstrap learning.

5. **Masked image modeling** applies the success of masked language modeling to vision, with MAE and BEiT showing strong performance through high masking ratios and discrete tokens.

6. **Masked language modeling** remains a highly effective pre-training objective for language models, enabling bidirectional context understanding.

7. **Predictive coding and JEPA** learn representations by predicting future or abstract information, providing efficient and scalable alternatives to pixel-level prediction.

8. **Representation quality** is evaluated through linear evaluation, fine-tuning, transfer learning, and representation analysis, with different metrics capturing different aspects.

9. **Data augmentation** is critical for SSL success, with composition and strength of augmentations significantly impacting performance.

10. **Future directions** include understanding why SSL works, improving efficiency, scaling to larger models, and developing better evaluation methods.

## References

- Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020
- He, K., et al. (2020). "Momentum Contrast for Unsupervised Visual Representation Learning." CVPR 2020
- Grill, J.-B., et al. (2020). "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." NeurIPS 2020
- He, K., et al. (2021). "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022
- Bao, H., et al. (2021). "BEiT: BERT Pre-Training of Image Transformers." ICLR 2022
- Zhou, J., et al. (2021). "iBOT: Image BERT Pre-Training with Online Tokenizer." ICLR 2022
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019
- LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence." OpenReview
- Assran, M., et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR 2023
