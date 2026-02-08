# Statistical Learning Theory

## Table of Contents

1. [Introduction](#introduction)
2. [Learning Problem Formulation](#learning-problem-formulation)
3. [PAC Learning Framework](#pac-learning-framework)
4. [VC Dimension](#vc-dimension)
5. [Generalization Bounds](#generalization-bounds)
6. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
7. [Rademacher Complexity](#rademacher-complexity)
8. [Concentration Inequalities](#concentration-inequalities)
9. [Structural Risk Minimization](#structural-risk-minimization)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Statistical learning theory provides mathematical foundations for understanding when and why machine learning algorithms generalize from training data to unseen examples. It addresses fundamental questions: How much data is needed? What is the relationship between training error and test error? How complex should models be? This document covers PAC learning, VC dimension, generalization bounds, bias-variance tradeoff, and Rademacher complexity, providing theoretical guarantees for ML algorithms.

## Learning Problem Formulation

### Supervised Learning Setup

**Data**: $\mathcal{D} = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ i.i.d. from distribution $P(\mathbf{X}, Y)$

**Hypothesis class**: $\mathcal{H} = \{h: \mathcal{X} \to \mathcal{Y}\}$ (set of possible models)

**Loss function**: $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$ (e.g., 0-1 loss, squared loss)

**Goal**: Find $h \in \mathcal{H}$ minimizing **risk** (expected loss):

$$R(h) = \mathbb{E}_{(\mathbf{X},Y) \sim P}[\ell(h(\mathbf{X}), Y)]$$

### Empirical Risk Minimization

**Empirical risk** (training error):

$$\hat{R}_n(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(\mathbf{x}_i), y_i)$$

**ERM principle**: Choose $\hat{h}_n = \arg\min_{h \in \mathcal{H}} \hat{R}_n(h)$

**Generalization gap**: $R(\hat{h}_n) - \hat{R}_n(\hat{h}_n)$ (difference between test and training error)

## PAC Learning Framework

### Probably Approximately Correct

**PAC learning**: Algorithm learns concept class $\mathcal{C}$ if for all $c \in \mathcal{C}$, distributions $P$, $\epsilon > 0$, $\delta > 0$, with probability at least $1-\delta$ over training samples of size $n \geq n_0(\epsilon, \delta)$, algorithm outputs $h$ with:

$$R(h) \leq \epsilon$$

**Interpretation**: With high probability ($1-\delta$), error is small ($\leq \epsilon$).

### Sample Complexity

**Sample complexity**: Minimum $n$ needed to achieve $(\epsilon, \delta)$-PAC learning.

**Key question**: How does $n$ depend on $\epsilon$, $\delta$, and complexity of $\mathcal{H}$?

### Agnostic PAC Learning

**Agnostic setting**: No assumption that true concept is in $\mathcal{H}$.

**Goal**: Find $h$ with:

$$R(h) \leq \min_{h' \in \mathcal{H}} R(h') + \epsilon$$

(Within $\epsilon$ of best possible in $\mathcal{H}$)

## VC Dimension

### Shattering

**Shattering**: Hypothesis class $\mathcal{H}$ **shatters** set $\{\mathbf{x}_1, \ldots, \mathbf{x}_d\}$ if $\mathcal{H}$ can realize all $2^d$ possible labelings.

**Example**: For $\mathcal{H} = \{\text{thresholds on } \mathbb{R}\}$ and points $x_1 < x_2$, can realize $(0,0)$, $(0,1)$, $(1,0)$, $(1,1)$? Yes if $x_1 < x_2$, so shatters 2 points. Cannot shatter 3 points (cannot get $(1,0,1)$).

### VC Dimension Definition

**VC dimension**: $d_{\text{VC}}(\mathcal{H}) = \max\{d : \mathcal{H} \text{ shatters some set of size } d\}$

**Interpretation**: Maximum number of points that can be shattered.

**Examples**:
- **Intervals on $\mathbb{R}$**: $d_{\text{VC}} = 2$
- **Half-spaces in $\mathbb{R}^d$**: $d_{\text{VC}} = d+1$
- **Axis-aligned rectangles in $\mathbb{R}^2$**: $d_{\text{VC}} = 4$

### VC Dimension and Generalization

**Fundamental theorem**: For binary classification with 0-1 loss:

$$R(\hat{h}_n) \leq \hat{R}_n(\hat{h}_n) + O\left(\sqrt{\frac{d_{\text{VC}}(\mathcal{H})\log n}{n}}\right)$$

with high probability.

**Interpretation**: Generalization gap scales as $\sqrt{d_{\text{VC}}/n}$.

## Generalization Bounds

### Uniform Convergence

**Uniform bound**: With probability at least $1-\delta$:

$$\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_n(h)| \leq \epsilon(n, \delta)$$

**Implication**: For ERM $\hat{h}_n$:

$$R(\hat{h}_n) \leq \hat{R}_n(\hat{h}_n) + \epsilon(n, \delta) \leq \min_{h \in \mathcal{H}} R(h) + 2\epsilon(n, \delta)$$

### VC Bound

**Theorem**: For binary classification with VC dimension $d$:

$$R(\hat{h}_n) \leq \hat{R}_n(\hat{h}_n) + \sqrt{\frac{2d\log(2en/d)}{n}} + \sqrt{\frac{\log(2/\delta)}{2n}}$$

with probability at least $1-\delta$.

**Rate**: $O(\sqrt{d\log n/n})$ (ignoring $\delta$ dependence).

### Occam's Razor

**Interpretation**: Simpler models (smaller $d_{\text{VC}}$) generalize better for fixed $n$.

**Tradeoff**: Need to balance model complexity (capacity) with data size.

## Bias-Variance Tradeoff

### Decomposition

For squared loss, expected prediction error decomposes:

$$\mathbb{E}[(Y - \hat{h}(\mathbf{X}))^2] = \underbrace{\text{Var}(Y | \mathbf{X})}_{\text{irreducible}} + \underbrace{(\mathbb{E}[\hat{h}(\mathbf{X})] - \mathbb{E}[Y | \mathbf{X}])^2}_{\text{bias}^2} + \underbrace{\text{Var}(\hat{h}(\mathbf{X}))}_{\text{variance}}$$

**Bias**: $(\mathbb{E}[\hat{h}(\mathbf{X})] - \mathbb{E}[Y | \mathbf{X}])^2$ (systematic error)

**Variance**: $\text{Var}(\hat{h}(\mathbf{X}))$ (sensitivity to training data)

**Tradeoff**: 
- **Simple models** (small $\mathcal{H}$): Low variance, high bias (underfitting)
- **Complex models** (large $\mathcal{H}$): Low bias, high variance (overfitting)

### Model Selection

**Goal**: Choose $\mathcal{H}$ balancing bias and variance to minimize total error.

**Methods**:
- Cross-validation
- Regularization (implicitly restricts $\mathcal{H}$)
- Early stopping

## Rademacher Complexity

### Definition

**Rademacher complexity** of $\mathcal{H}$:

$$\mathcal{R}_n(\mathcal{H}) = \mathbb{E}_{\boldsymbol{\sigma}, \mathcal{D}}\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(\mathbf{x}_i)\right]$$

where $\sigma_i \sim \text{Unif}(\{-1, +1\})$ are Rademacher random variables.

**Interpretation**: Measures ability of $\mathcal{H}$ to fit random noise.

### Generalization Bound

**Theorem**: With probability at least $1-\delta$:

$$R(\hat{h}_n) \leq \hat{R}_n(\hat{h}_n) + 2\mathcal{R}_n(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

**Advantage**: Data-dependent (depends on actual $\mathcal{D}$), often tighter than VC bounds.

### Examples

**Linear functions**: $\mathcal{H} = \{\mathbf{x} \mapsto \mathbf{w}^T\mathbf{x} : \|\mathbf{w}\| \leq B\}$

$$\mathcal{R}_n(\mathcal{H}) \leq \frac{B \max_i \|\mathbf{x}_i\|}{\sqrt{n}}$$

**Neural networks**: Bounds depend on depth, width, weight norms.

## Concentration Inequalities

### Markov's Inequality

**Markov**: For nonnegative $X$ and $a > 0$:

$$P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}$$

### Chebyshev's Inequality

**Chebyshev**: For $X$ with mean $\mu$ and variance $\sigma^2$:

$$P(|X - \mu| \geq t) \leq \frac{\sigma^2}{t^2}$$

### Hoeffding's Inequality

**Hoeffding**: For bounded i.i.d. $X_1, \ldots, X_n$ with $a \leq X_i \leq b$:

$$P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mathbb{E}[X]\right| \geq t\right) \leq 2\exp\left(-\frac{2nt^2}{(b-a)^2}\right)$$

**Application**: Bounding $|\hat{R}_n(h) - R(h)|$ for fixed $h$.

### McDiarmid's Inequality

**McDiarmid**: For function $f(X_1, \ldots, X_n)$ with bounded differences $|f(\mathbf{x}) - f(\mathbf{x}')| \leq c_i$ when only $x_i$ changes:

$$P(f(\mathbf{X}) - \mathbb{E}[f(\mathbf{X})] \geq t) \leq \exp\left(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\right)$$

**Application**: Uniform bounds over $\mathcal{H}$.

## Structural Risk Minimization

### Principle

**SRM**: Choose $\mathcal{H}$ and $h \in \mathcal{H}$ to minimize:

$$R(h) + \text{complexity penalty}(\mathcal{H})$$

**Penalty**: Increases with model complexity (e.g., VC dimension).

**Example**: Nested classes $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \cdots$ with increasing complexity.

### Regularization

**Regularized ERM**:

$$\hat{h}_n = \arg\min_{h \in \mathcal{H}} \left[\hat{R}_n(h) + \lambda \Omega(h)\right]$$

where $\Omega(h)$ is complexity measure (e.g., $\|\mathbf{w}\|^2$ for linear models).

**Interpretation**: Implicitly restricts to $\{h : \Omega(h) \leq C\}$ for some $C$.

## Machine Learning Applications

### Model Selection

**Cross-validation**: Estimate generalization error to choose model complexity.

**Theoretical guidance**: VC dimension or Rademacher complexity suggest appropriate model size.

### Regularization

**L2 regularization**: $\Omega(h) = \|\mathbf{w}\|^2$ restricts to ball $\{\mathbf{w} : \|\mathbf{w}\| \leq C\}$.

**L1 regularization**: Promotes sparsity, effectively reduces model complexity.

### Early Stopping

**Early stopping**: Stop training before convergence to prevent overfitting.

**Interpretation**: Implicit regularization via limiting number of gradient steps.

### Double Descent

**Phenomenon**: Test error decreases, increases (classical), then decreases again as model size increases beyond interpolation threshold.

**Theoretical challenge**: Traditional bounds predict monotonic increase, but modern deep learning shows different behavior.

### Generalization in Deep Learning

**Empirical observation**: Deep networks generalize well despite:
- Very large capacity ($d_{\text{VC}}$)
- Small training sets
- Fitting training data perfectly (zero training error)

**Possible explanations**:
- Implicit regularization (optimization algorithm)
- Data structure
- Need new theoretical frameworks beyond VC dimension

### PAC-Bayes Bounds

**PAC-Bayes**: Bounds depend on prior $P$ and posterior $Q$ over $\mathcal{H}$:

$$R(Q) \leq \hat{R}_n(Q) + \sqrt{\frac{\text{KL}(Q\|P) + \log(2n/\delta)}{2(n-1)}}$$

where $R(Q) = \mathbb{E}_{h \sim Q}[R(h)]$ is expected risk.

**Application**: Provides bounds for Bayesian methods, stochastic neural networks.

### Margin Theory

**Margin**: Distance from example to decision boundary.

**Large margin**: Better generalization (SVM theory).

**Bound**: Generalization error $\leq \frac{R^2\|\mathbf{w}\|^2}{n}$ where $R$ is radius of data.

## Key Takeaways

1. **PAC learning** provides framework for understanding when algorithms learn with high probability and low error.

2. **VC dimension** measures model complexity, directly related to generalization gap.

3. **Generalization bounds** relate training error, test error, and model complexity.

4. **Bias-variance tradeoff** explains underfitting vs overfitting: simple models have high bias/low variance, complex models have low bias/high variance.

5. **Rademacher complexity** provides data-dependent measure of model complexity, often tighter than VC bounds.

6. **Concentration inequalities** (Hoeffding, McDiarmid) enable probabilistic bounds on generalization gap.

7. **Structural risk minimization** balances empirical risk and model complexity to minimize total error.

8. **Regularization** implicitly restricts hypothesis class, reducing overfitting.

9. **Deep learning** challenges traditional theory: very large models generalize despite large capacity.

10. **Statistical learning theory** provides principled framework for understanding generalization, guiding model selection, and designing learning algorithms.
