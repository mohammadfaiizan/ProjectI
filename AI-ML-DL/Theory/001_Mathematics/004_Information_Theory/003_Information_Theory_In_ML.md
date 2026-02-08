# Information Theory in Machine Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Cross-Entropy Loss Derivation](#cross-entropy-loss-derivation)
3. [Information Bottleneck Method](#information-bottleneck-method)
4. [Rate-Distortion Theory Applications](#rate-distortion-theory-applications)
5. [Mutual Information Neural Estimation](#mutual-information-neural-estimation)
6. [Information-Theoretic Feature Selection](#information-theoretic-feature-selection)
7. [Contrastive Learning](#contrastive-learning)
8. [Information-Theoretic Regularization](#information-theoretic-regularization)
9. [Causal Inference](#causal-inference)
10. [Key Takeaways](#key-takeaways)

## Introduction

Information theory provides powerful principles and tools for machine learning, from loss function design to understanding learned representations. This document covers information-theoretic approaches in ML: cross-entropy loss derivation, information bottleneck, mutual information estimation, contrastive learning, and regularization. These concepts unify many ML algorithms under an information-theoretic framework.

## Cross-Entropy Loss Derivation

### Maximum Likelihood Perspective

**Classification**: Predict $p(y | \mathbf{x}; \boldsymbol{\theta})$ for classes $y \in \{1, \ldots, K\}$.

**MLE**: Maximize likelihood:

$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^n \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})$$

**One-hot encoding**: True distribution $p^*(y | \mathbf{x}_i) = \mathbf{1}[y = y_i]$.

**Cross-entropy**: 

$$H(p^*, p) = -\sum_y p^*(y) \log p(y | \mathbf{x}) = -\log p(y_i | \mathbf{x}_i)$$

**Equivalence**: Maximizing likelihood = minimizing cross-entropy.

### KL Divergence Connection

**Decomposition**:

$$H(p^*, p) = H(p^*) + D_{\text{KL}}(p^* \| p)$$

Since $H(p^*) = 0$ for one-hot (deterministic), minimizing cross-entropy = minimizing KL divergence.

**General case**: For soft labels $p^*(y | \mathbf{x})$:

$$\min_p H(p^*, p) = \min_p D_{\text{KL}}(p^* \| p)$$

Optimal $p = p^*$ (achieves zero KL).

### Information-Theoretic Interpretation

**Information content**: $-\log p(y | \mathbf{x})$ measures "surprise" of prediction.

**Cross-entropy**: Average surprise when true distribution is $p^*$ but we use $p$.

**Minimization**: Find $p$ that minimizes expected surprise.

## Information Bottleneck Method

### Principle

**Information bottleneck**: Compress input $X$ into representation $Z$ while preserving information about target $Y$:

$$\min_{p(z|x)} I(X; Z) - \beta I(Z; Y)$$

**Tradeoff**:
- **Compression**: Minimize $I(X; Z)$ (make $Z$ independent of $X$)
- **Relevance**: Maximize $I(Z; Y)$ (preserve information about $Y$)
- **$\beta$**: Controls tradeoff

### Variational Bound

**Challenge**: $I(X; Z)$ and $I(Z; Y)$ are hard to compute.

**Variational IB**: Introduce variational approximations:

$$I(X; Z) = \mathbb{E}_{p(x,z)}\left[\log\frac{p(z|x)}{p(z)}\right]$$

$$I(Z; Y) = \mathbb{E}_{p(z,y)}\left[\log\frac{p(y|z)}{p(y)}\right]$$

**Bound**: 

$$\mathcal{L}_{\text{IB}} = \mathbb{E}_{p(x,y,z)}\left[\log\frac{p(z|x)}{q(z)} - \beta\log p(y|z)\right]$$

where $q(z)$ approximates $p(z)$.

### Connection to Deep Learning

**Neural networks**: Encoder $p(z|x)$ compresses input, decoder $p(y|z)$ predicts target.

**IB interpretation**: 
- Encoder minimizes $I(X; Z)$ (compression)
- Decoder maximizes $I(Z; Y)$ (relevance)
- Tradeoff controlled by architecture and training

**Layers**: Each layer implements information bottleneck, progressively compressing while preserving task-relevant information.

### Optimal Representation

**IB curve**: Plot $I(Z; Y)$ vs. $I(X; Z)$ for different $\beta$.

**Pareto frontier**: Optimal tradeoff curve.

**Phase transition**: At critical $\beta$, representation quality changes dramatically.

## Rate-Distortion Theory Applications

### Variational Autoencoders

**VAE objective** (ELBO):

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**Rate-distortion interpretation**:
- **Rate**: $D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ (compression cost)
- **Distortion**: $-\mathbb{E}[\log p(\mathbf{x}|\mathbf{z})]$ (reconstruction error)

**$\beta$-VAE**: Weighted version:

$$\mathcal{L} = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - \beta D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

Larger $\beta$ → more compression, potentially better disentanglement.

### Generative Models

**GANs**: Minimize JS divergence:

$$\min_G D_{\text{JS}}(p_{\text{data}} \| p_g)$$

**VAEs**: Minimize KL divergence (different from GANs).

**Rate-distortion**: Both can be viewed through rate-distortion lens.

### Compression in Neural Networks

**Low-rank factorization**: $\mathbf{W} \approx \mathbf{U}\mathbf{V}^T$

**Rate**: Number of parameters (bits to store)

**Distortion**: Approximation error $\|\mathbf{W} - \mathbf{U}\mathbf{V}^T\|_F$

**Optimal**: Find factorization minimizing distortion for given rate.

## Mutual Information Neural Estimation

### Problem

**Challenge**: Computing $I(X; Y) = \mathbb{E}_{p(x,y)}\left[\log\frac{p(x,y)}{p(x)p(y)}\right]$ requires knowing distributions.

**Solution**: Estimate using neural networks.

### MINE (Mutual Information Neural Estimation)

**Donsker-Varadhan representation**:

$$I(X; Y) = \sup_{T: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}} \mathbb{E}_{p(x,y)}[T(x,y)] - \log\mathbb{E}_{p(x)p(y)}[e^{T(x,y)}]$$

**Neural estimator**: Parameterize $T$ as neural network, maximize:

$$\hat{I}(X; Y) = \max_\theta \mathbb{E}_{p(x,y)}[T_\theta(x,y)] - \log\mathbb{E}_{p(x)p(y)}[e^{T_\theta(x,y)}]$$

**Gradient**: Use samples to estimate expectations.

### Applications

**Feature selection**: Estimate $I(X_i; Y)$ to rank features.

**Representation learning**: Maximize $I(Z; Y)$ for learned representation $Z$.

**Contrastive learning**: Lower bound on $I(\text{enc}(\mathbf{x}); \text{enc}(\mathbf{x}^+))$.

## Information-Theoretic Feature Selection

### Mutual Information Criterion

**mRMR** (minimum Redundancy Maximum Relevance):

$$\max_{i} \left[I(X_i; Y) - \frac{1}{|S|}\sum_{j \in S} I(X_i; X_j)\right]$$

**Interpretation**: 
- Maximize relevance: $I(X_i; Y)$
- Minimize redundancy: Average $I(X_i; X_j)$ with selected features

**Greedy selection**: Iteratively add feature maximizing criterion.

### Conditional Mutual Information

**CMI-based**: Select features maximizing:

$$I(X_i; Y | X_S)$$

Information about $Y$ given already selected features $X_S$.

**Advantage**: Accounts for interactions between features.

### Information Gain

**Decision trees**: Split on feature maximizing:

$$\text{IG}(Y; X_i) = H(Y) - H(Y | X_i) = I(Y; X_i)$$

**Gain ratio**: Normalize by $H(X_i)$ to avoid bias toward high-cardinality features.

## Contrastive Learning

### Principle

**Contrastive learning**: Learn representations by contrasting positive and negative pairs.

**Positive pairs**: Similar examples $(\mathbf{x}, \mathbf{x}^+)$

**Negative pairs**: Dissimilar examples $(\mathbf{x}, \mathbf{x}^-)$

### InfoNCE Loss

**InfoNCE** (Information Noise Contrastive Estimation):

$$\mathcal{L} = -\mathbb{E}\left[\log\frac{\exp(f(\mathbf{x}, \mathbf{x}^+))}{\exp(f(\mathbf{x}, \mathbf{x}^+)) + \sum_{i=1}^{N-1} \exp(f(\mathbf{x}, \mathbf{x}_i^-))}\right]$$

**Lower bound**: InfoNCE lower bounds mutual information:

$$I(\text{enc}(\mathbf{x}); \text{enc}(\mathbf{x}^+)) \geq \log N - \mathcal{L}$$

where $N$ is number of negatives.

**Interpretation**: Maximizing InfoNCE maximizes mutual information between representations of positive pairs.

### SimCLR

**Framework**:
1. Data augmentation: $\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}_j$ from same $\mathbf{x}$
2. Encoder: $\mathbf{z}_i = f(\tilde{\mathbf{x}}_i)$
3. Projection: $\mathbf{h}_i = g(\mathbf{z}_i)$
4. Contrastive loss: InfoNCE on $(\mathbf{h}_i, \mathbf{h}_j)$

**Information**: Maximizes $I(\mathbf{h}_i; \mathbf{h}_j)$ for augmented views.

## Information-Theoretic Regularization

### Entropy Regularization

**Maximum entropy**: Regularize to prefer high-entropy distributions:

$$\mathcal{L} = \text{loss} - \lambda H(p(y | \mathbf{x}))$$

**Effect**: Encourages uniform/uncertain predictions (useful for exploration).

### Mutual Information Regularization

**Feature decorrelation**: Minimize $I(Z_i; Z_j)$ between features:

$$\mathcal{L} = \text{loss} + \lambda \sum_{i \neq j} I(Z_i; Z_j)$$

**Goal**: Learn independent/disentangled features.

### Information Bottleneck Regularization

**IB regularization**: Add IB term to loss:

$$\mathcal{L} = \text{task loss} + \beta I(X; Z) - \gamma I(Z; Y)$$

**Interpretation**: 
- Compress representation ($I(X; Z)$)
- Preserve task information ($I(Z; Y)$)

## Causal Inference

### Information Flow

**Causal graph**: Directed acyclic graph representing causal relationships.

**Information**: $I(X; Y)$ measures association, not necessarily causation.

**Conditional MI**: $I(X; Y | Z)$ measures direct association controlling for $Z$.

### Causal Discovery

**PC algorithm**: Uses conditional independence tests (related to mutual information) to infer causal structure.

**Information-theoretic**: Methods based on information measures to identify causal directions.

### Invariant Representations

**Goal**: Learn representations invariant to spurious correlations.

**Information**: Minimize $I(Z; S)$ where $S$ is spurious variable, maximize $I(Z; Y)$.

**IRM** (Invariant Risk Minimization): Uses information-theoretic principles.

## Key Takeaways

1. **Cross-entropy loss** minimizes KL divergence between true and predicted distributions, derived from maximum likelihood.

2. **Information bottleneck** provides framework for understanding compression and relevance tradeoff in representation learning.

3. **Rate-distortion theory** unifies VAE, compression, and generative modeling under single framework.

4. **Mutual information estimation** enables information-theoretic methods when distributions are unknown.

5. **Feature selection** uses mutual information to identify informative features while avoiding redundancy.

6. **Contrastive learning** maximizes mutual information between representations of similar examples via InfoNCE.

7. **Information-theoretic regularization** controls model behavior via entropy and mutual information penalties.

8. **Causal inference** benefits from information-theoretic measures of conditional independence.

9. **Information theory** provides unified framework for understanding many ML algorithms and designing new ones.

10. **Information measures** (entropy, MI, KL) appear throughout ML: loss functions, regularization, feature selection, representation learning.
