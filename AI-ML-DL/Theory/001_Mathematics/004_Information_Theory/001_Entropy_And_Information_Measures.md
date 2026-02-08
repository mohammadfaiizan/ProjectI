# Entropy and Information Measures

## Table of Contents

1. [Introduction](#introduction)
2. [Shannon Entropy](#shannon-entropy)
3. [Conditional Entropy](#conditional-entropy)
4. [Mutual Information](#mutual-information)
5. [KL Divergence](#kl-divergence)
6. [Jensen-Shannon Divergence](#jensen-shannon-divergence)
7. [Properties of Information Measures](#properties-of-information-measures)
8. [Continuous Case](#continuous-case)
9. [Information-Theoretic Inequalities](#information-theoretic-inequalities)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Information theory, founded by Claude Shannon, quantifies information, uncertainty, and communication. In machine learning, information-theoretic concepts appear in loss functions (cross-entropy), feature selection (mutual information), representation learning (information bottleneck), and understanding model behavior. This document covers entropy, conditional entropy, mutual information, KL divergence, and their properties, with emphasis on ML applications.

## Shannon Entropy

### Definition

**Shannon entropy** of discrete random variable $X$ with PMF $p(x)$:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = \mathbb{E}_p[-\log p(X)]$$

**Convention**: $0 \log 0 = 0$ (by continuity).

**Base**: Typically base 2 (bits) or natural log (nats). We use natural log unless specified.

**Interpretation**: 
- Average "surprise" or "information content"
- Minimum average number of bits needed to encode outcomes
- Measure of uncertainty

### Examples

**Uniform distribution**: $X \sim \text{Unif}(\{1, \ldots, n\})$:

$$H(X) = -\sum_{i=1}^n \frac{1}{n}\log\frac{1}{n} = \log n$$

Maximum entropy for $n$ outcomes.

**Deterministic**: $P(X = x_0) = 1$:

$$H(X) = -1 \cdot \log 1 = 0$$

No uncertainty, zero entropy.

**Bernoulli**: $X \sim \text{Bernoulli}(p)$:

$$H(X) = -p\log p - (1-p)\log(1-p) = H(p)$$

Maximum at $p = 1/2$: $H(1/2) = \log 2$.

### Properties

**Non-negativity**: $H(X) \geq 0$ with equality iff $X$ is deterministic.

**Upper bound**: $H(X) \leq \log|\mathcal{X}|$ with equality iff $X$ is uniform.

**Proof**: Use Jensen's inequality with concave function $-\log$.

**Additivity**: For independent $X$ and $Y$:

$$H(X, Y) = H(X) + H(Y)$$

**General**: $H(X, Y) = H(X) + H(Y | X) \leq H(X) + H(Y)$.

## Conditional Entropy

### Definition

**Conditional entropy** of $Y$ given $X$:

$$H(Y | X) = -\sum_{x,y} p(x, y) \log p(y | x) = \mathbb{E}_{p(x,y)}[-\log p(Y | X)]$$

**Interpretation**: Remaining uncertainty in $Y$ after observing $X$.

### Chain Rule

**Chain rule for entropy**:

$$H(X, Y) = H(X) + H(Y | X)$$

**Generalization**: 

$$H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i | X_1, \ldots, X_{i-1})$$

**Proof**: 

$$H(X, Y) = -\sum_{x,y} p(x,y)\log p(x,y) = -\sum_{x,y} p(x,y)\log[p(x)p(y|x)]$$
$$= -\sum_x p(x)\log p(x) - \sum_{x,y} p(x,y)\log p(y|x) = H(X) + H(Y|X)$$

### Properties

**Reduction**: $H(Y | X) \leq H(Y)$ (conditioning reduces entropy, on average)

**Equality**: $H(Y | X) = H(Y)$ iff $X$ and $Y$ are independent

**Note**: $H(Y | X = x)$ can be larger than $H(Y)$ for specific $x$, but average $H(Y | X) \leq H(Y)$.

## Mutual Information

### Definition

**Mutual information** between $X$ and $Y$:

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} = \mathbb{E}_{p(x,y)}\left[\log\frac{p(X,Y)}{p(X)p(Y)}\right]$$

**Alternative expressions**:
- $I(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)$
- $I(X; Y) = H(X) + H(Y) - H(X, Y)$

**Interpretation**: 
- Amount of information $X$ provides about $Y$ (and vice versa)
- Reduction in uncertainty about $Y$ after observing $X$
- Measure of dependence

### Properties

**Symmetry**: $I(X; Y) = I(Y; X)$

**Non-negativity**: $I(X; Y) \geq 0$ with equality iff $X$ and $Y$ are independent

**Upper bounds**: 
- $I(X; Y) \leq H(X)$, $I(X; Y) \leq H(Y)$
- $I(X; Y) \leq \min(H(X), H(Y))$

**Additivity**: For independent pairs $(X_1, Y_1)$ and $(X_2, Y_2)$:

$$I(X_1, X_2; Y_1, Y_2) = I(X_1; Y_1) + I(X_2; Y_2)$$

### Conditional Mutual Information

**Conditional MI**: 

$$I(X; Y | Z) = H(X | Z) - H(X | Y, Z) = H(Y | Z) - H(Y | X, Z)$$

**Chain rule**:

$$I(X_1, \ldots, X_n; Y) = \sum_{i=1}^n I(X_i; Y | X_1, \ldots, X_{i-1})$$

## KL Divergence

### Definition

**Kullback-Leibler divergence** (relative entropy) from $Q$ to $P$:

$$D_{\text{KL}}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_P\left[\log\frac{P(X)}{Q(X)}\right]$$

**Notation**: $D_{\text{KL}}(P \| Q)$ or $D(P \| Q)$.

**Interpretation**: 
- Measure of "distance" between distributions (not symmetric)
- Extra bits needed to encode $P$ using code optimized for $Q$
- Measure of inefficiency of using wrong distribution

### Properties

**Non-negativity**: $D_{\text{KL}}(P \| Q) \geq 0$ with equality iff $P = Q$ (almost everywhere)

**Proof**: Use Jensen's inequality with convex function $-\log$:

$$D_{\text{KL}}(P \| Q) = \mathbb{E}_P\left[\log\frac{P}{Q}\right] = -\mathbb{E}_P\left[\log\frac{Q}{P}\right] \geq -\log\mathbb{E}_P\left[\frac{Q}{P}\right] = -\log 1 = 0$$

**Asymmetry**: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$ in general

**Not a metric**: Doesn't satisfy triangle inequality

### Chain Rule

**Chain rule for KL divergence**:

$$D_{\text{KL}}(P(X,Y) \| Q(X,Y)) = D_{\text{KL}}(P(X) \| Q(X)) + D_{\text{KL}}(P(Y | X) \| Q(Y | X) | P(X))$$

where conditional KL: $D_{\text{KL}}(P(Y|X) \| Q(Y|X) | P(X)) = \sum_x p(x) D_{\text{KL}}(P(Y|x) \| Q(Y|x))$.

### Relationship to Mutual Information

**Connection**: 

$$I(X; Y) = D_{\text{KL}}(P(X,Y) \| P(X)P(Y))$$

Mutual information is KL divergence from joint to product of marginals.

## Jensen-Shannon Divergence

### Definition

**Jensen-Shannon divergence**:

$$D_{\text{JS}}(P \| Q) = \frac{1}{2}D_{\text{KL}}(P \| M) + \frac{1}{2}D_{\text{KL}}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$ is mixture distribution.

### Properties

**Symmetry**: $D_{\text{JS}}(P \| Q) = D_{\text{JS}}(Q \| P)$

**Bounded**: $0 \leq D_{\text{JS}}(P \| Q) \leq \log 2$ (unlike KL divergence)

**Square root**: $\sqrt{D_{\text{JS}}}$ is a metric

**Relationship**: 

$$D_{\text{JS}}(P \| Q) = H(M) - \frac{1}{2}[H(P) + H(Q)]$$

## Properties of Information Measures

### Data Processing Inequality

**DPI**: For Markov chain $X \to Y \to Z$:

$$I(X; Z) \leq I(X; Y)$$

**Interpretation**: Processing cannot increase information.

**Application**: Feature transformations cannot increase mutual information with target.

### Fano's Inequality

**Fano**: For estimator $\hat{X}$ of $X$:

$$H(X | \hat{X}) \leq H(P_e) + P_e \log(|\mathcal{X}| - 1)$$

where $P_e = P(X \neq \hat{X})$ is error probability.

**Lower bound on error**: 

$$P_e \geq \frac{H(X | \hat{X}) - 1}{\log|\mathcal{X}|}$$

### Pinsker's Inequality

**Pinsker**: Relates KL divergence to total variation:

$$\delta(P, Q) \leq \sqrt{\frac{1}{2}D_{\text{KL}}(P \| Q)}$$

where $\delta(P, Q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ is total variation distance.

## Continuous Case

### Differential Entropy

**Differential entropy** for continuous $X$ with PDF $f(x)$:

$$h(X) = -\int f(x) \log f(x) dx = \mathbb{E}_f[-\log f(X)]$$

**Note**: Can be negative (unlike discrete entropy).

**Example**: $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$h(X) = \frac{1}{2}\log(2\pi e\sigma^2)$$

Increases with variance.

### Mutual Information (Continuous)

$$I(X; Y) = \int\int f(x,y) \log\frac{f(x,y)}{f(x)f(y)} dx dy = h(X) + h(Y) - h(X,Y)$$

### KL Divergence (Continuous)

$$D_{\text{KL}}(P \| Q) = \int p(x) \log\frac{p(x)}{q(x)} dx$$

## Information-Theoretic Inequalities

### Jensen's Inequality

**Jensen**: For convex $\phi$:

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)]$$

**Application**: $-\log$ is convex, so:

$$H(X) = \mathbb{E}[-\log p(X)] \geq -\log\mathbb{E}[p(X)] = -\log 1 = 0$$

### Log-Sum Inequality

**Log-sum**: For nonnegative $a_i, b_i$:

$$\sum_i a_i \log\frac{a_i}{b_i} \geq \left(\sum_i a_i\right)\log\frac{\sum_i a_i}{\sum_i b_i}$$

**Application**: Proving non-negativity of KL divergence.

### Entropy Power Inequality

**EPI**: For independent $X$ and $Y$:

$$e^{2h(X+Y)} \geq e^{2h(X)} + e^{2h(Y)}$$

**Gaussian case**: Equality when $X$ and $Y$ are Gaussian.

## Machine Learning Applications

### Cross-Entropy Loss

**Cross-entropy**: For true distribution $p^*(y)$ and predicted $p(y | \mathbf{x})$:

$$H(p^*, p) = -\sum_y p^*(y) \log p(y | \mathbf{x})$$

**Relationship**: $H(p^*, p) = H(p^*) + D_{\text{KL}}(p^* \| p)$

**Minimization**: Minimizing cross-entropy is equivalent to minimizing KL divergence (since $H(p^*)$ is constant).

**Classification**: With one-hot $p^*$ (true label):

$$H(p^*, p) = -\log p(y_{\text{true}} | \mathbf{x})$$

Standard cross-entropy loss.

### Feature Selection

**Mutual information criterion**: Select features maximizing $I(X_i; Y)$.

**mRMR**: Maximize relevance $I(X_i; Y)$ while minimizing redundancy $\sum_{j \in S} I(X_i; X_j)$:

$$\max_{i} \left[I(X_i; Y) - \frac{1}{|S|}\sum_{j \in S} I(X_i; X_j)\right]$$

### Information Bottleneck

**Principle**: Compress $X$ into representation $Z$ while preserving information about $Y$:

$$\min_{p(z|x)} I(X; Z) - \beta I(Z; Y)$$

**Tradeoff**: Compression vs. prediction (controlled by $\beta$).

**Application**: Understanding learned representations in deep learning.

### Variational Autoencoders

**ELBO**: 

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**Interpretation**: Maximize reconstruction while keeping posterior close to prior.

### GAN Training

**Objective**: Minimize JS divergence between data and generator distributions:

$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1-D(G(\mathbf{z})))]$$

Optimal $D$ gives $D_{\text{JS}}(p_{\text{data}} \| p_g)$.

### Maximum Entropy Principle

**Principle**: Choose distribution maximizing entropy subject to constraints.

**Application**: 
- Exponential family distributions maximize entropy given moment constraints
- Regularization via entropy penalty

### Information Gain

**Decision trees**: Split on feature maximizing information gain:

$$\text{IG}(Y; X_i) = H(Y) - H(Y | X_i) = I(Y; X_i)$$

### Contrastive Learning

**InfoNCE loss**: Maximize mutual information between positive pairs:

$$\mathcal{L} = -\log\frac{\exp(f(\mathbf{x}, \mathbf{x}^+))}{\exp(f(\mathbf{x}, \mathbf{x}^+)) + \sum_{\mathbf{x}^-} \exp(f(\mathbf{x}, \mathbf{x}^-))}$$

Lower bound on $I(\text{enc}(\mathbf{x}); \text{enc}(\mathbf{x}^+))$.

## Key Takeaways

1. **Shannon entropy** measures uncertainty/information content, fundamental to information theory.

2. **Conditional entropy** measures remaining uncertainty after conditioning, always $\leq$ unconditional entropy.

3. **Mutual information** quantifies dependence between variables, symmetric and non-negative.

4. **KL divergence** measures "distance" between distributions (asymmetric), appears in many ML objectives.

5. **Jensen-Shannon divergence** is symmetric version of KL, bounded and metric-like.

6. **Cross-entropy loss** minimizes KL divergence between true and predicted distributions.

7. **Feature selection** uses mutual information to identify informative features.

8. **Information bottleneck** provides framework for understanding representation learning.

9. **VAEs** use KL divergence to regularize latent representations.

10. **Information theory** provides principled framework for understanding and designing ML algorithms, quantifying information flow, and measuring model quality.
