# Density Estimation Methods

## Table of Contents

1. [Introduction to Density Estimation](#introduction-to-density-estimation)
2. [Parametric Methods](#parametric-methods)
3. [Non-Parametric Methods](#non-parametric-methods)
4. [Kernel Density Estimation](#kernel-density-estimation)
5. [Bandwidth Selection](#bandwidth-selection)
6. [Multivariate KDE](#multivariate-kde)
7. [Histogram-Based Methods](#histogram-based-methods)
8. [Mixture Models](#mixture-models)
9. [Applications](#applications)
10. [Key Takeaways](#key-takeaways)

## Introduction to Density Estimation

Density estimation aims to estimate the probability density function $p(\mathbf{x})$ from observed data samples.

### Problem Formulation

Given independent samples $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$ from unknown distribution $p(\mathbf{x})$, estimate $\hat{p}(\mathbf{x})$.

### Why Density Estimation?

- **Understanding Data**: Characterize underlying distribution
- **Generative Modeling**: Generate new samples
- **Anomaly Detection**: Identify low-probability regions
- **Classification**: Estimate class-conditional densities for Bayes classifier
- **Clustering**: Model-based clustering (GMM)

### Types of Methods

**Parametric**: Assume specific distribution family (e.g., Gaussian)
- Few parameters to estimate
- Fast and efficient
- May be too restrictive

**Non-Parametric**: Make minimal assumptions
- Flexible, adapts to data
- More parameters (or bandwidth)
- Slower, requires more data

**Semi-Parametric**: Combine both approaches (e.g., mixture models)

## Parametric Methods

Parametric methods assume data follows a specific parametric distribution.

### Maximum Likelihood Estimation

Given parametric family $p(\mathbf{x} | \boldsymbol{\theta})$, find:

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n p(\mathbf{x}_i | \boldsymbol{\theta})$$

### Gaussian Distribution

**Univariate**:
$$p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**MLE Estimates**:
$$\hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i$$

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$$

**Multivariate**:
$$p(\mathbf{x} | \boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

**MLE Estimates**:
$$\hat{\boldsymbol{\mu}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$$

$$\hat{\Sigma} = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^T$$

### Other Parametric Distributions

**Exponential**: $p(x | \lambda) = \lambda e^{-\lambda x}$ for $x \geq 0$

**Gamma**: $p(x | \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$

**Beta**: $p(x | \alpha, \beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1}(1-x)^{\beta-1}$ for $x \in [0,1]$

### Advantages

- Efficient (few parameters)
- Fast estimation and evaluation
- Good when distributional assumption holds

### Limitations

- May be too restrictive
- Poor fit if assumption violated
- Requires domain knowledge to choose distribution

## Non-Parametric Methods

Non-parametric methods make minimal assumptions about the distribution.

### Histogram

Simplest non-parametric method:
1. Divide range into bins
2. Count samples in each bin
3. Normalize: $\hat{p}(x) = \frac{\text{count in bin}}{n \times \text{bin width}}$

**Issues**:
- Discontinuous
- Sensitive to bin placement
- High-dimensional curse

### Kernel Density Estimation (KDE)

Smooth alternative to histogram using kernel functions.

## Kernel Density Estimation

KDE places a kernel (smoothing function) at each data point and sums them.

### Formulation

$$\hat{p}(x) = \frac{1}{nh}\sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$$

where:
- $K$: Kernel function
- $h$: Bandwidth (smoothing parameter)
- $n$: Number of samples

### Kernel Functions

Kernels are symmetric, non-negative functions integrating to 1.

**Gaussian Kernel**:
$$K(u) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{u^2}{2}\right)$$

**Epanechnikov Kernel**:
$$K(u) = \frac{3}{4}(1 - u^2) \mathbb{1}(|u| \leq 1)$$

**Uniform Kernel**:
$$K(u) = \frac{1}{2} \mathbb{1}(|u| \leq 1)$$

**Triangular Kernel**:
$$K(u) = (1 - |u|) \mathbb{1}(|u| \leq 1)$$

### Properties

- **Smooth**: Continuous and differentiable (with smooth kernels)
- **Asymptotically Unbiased**: $E[\hat{p}(x)] \to p(x)$ as $n \to \infty$
- **Consistent**: $\hat{p}(x) \to p(x)$ in probability

### Effect of Bandwidth

**Small $h$**:
- More detail, follows data closely
- High variance, noisy
- Risk of overfitting

**Large $h$**:
- Smoother estimate
- Lower variance but higher bias
- May oversmooth, miss features

**Optimal $h$**: Balances bias and variance

## Bandwidth Selection

Choosing optimal bandwidth $h$ is crucial for KDE performance.

### Rule of Thumb (Silverman's Rule)

For Gaussian kernel:

$$h = 1.06 \hat{\sigma} n^{-1/5}$$

where $\hat{\sigma}$ is sample standard deviation.

**Robust Version** (uses IQR):
$$h = 0.9 \min(\hat{\sigma}, \frac{\text{IQR}}{1.34}) n^{-1/5}$$

### Cross-Validation

**Leave-One-Out Cross-Validation**:

Maximize log-likelihood:
$$CV(h) = \sum_{i=1}^n \log \hat{p}_{-i}(x_i)$$

where $\hat{p}_{-i}$ is KDE using all points except $x_i$.

**Grid Search**: Try different $h$ values, choose maximizing $CV(h)$.

### Plug-in Methods

Estimate optimal bandwidth by estimating derivatives of $p(x)$:

$$h_{\text{opt}} = \left(\frac{R(K)}{\mu_2(K)^2 R(p'') n}\right)^{1/5}$$

where:
- $R(K) = \int K(u)^2 du$
- $\mu_2(K) = \int u^2 K(u) du$
- $R(p'') = \int p''(x)^2 dx$ (estimated from data)

### Adaptive Bandwidth

Use different bandwidths in different regions:

$$h(x) = h_0 \cdot \left(\frac{\hat{p}(x)}{g}\right)^{-\alpha}$$

where $g$ is geometric mean of pilot estimate, $\alpha \in [0,1]$ controls adaptivity.

## Multivariate KDE

Extend KDE to multiple dimensions.

### Formulation

$$\hat{p}(\mathbf{x}) = \frac{1}{n|\mathbf{H}|^{1/2}} \sum_{i=1}^n K(\mathbf{H}^{-1/2}(\mathbf{x} - \mathbf{x}_i))$$

where $\mathbf{H}$ is bandwidth matrix.

### Bandwidth Matrix

**Scalar Bandwidth**: $\mathbf{H} = h^2 \mathbf{I}$ (same bandwidth in all directions)

**Diagonal Bandwidth**: $\mathbf{H} = \text{diag}(h_1^2, \ldots, h_d^2)$ (different bandwidths per dimension)

**Full Bandwidth**: $\mathbf{H}$ is full matrix (accounts for correlations)

### Product Kernel

For computational efficiency, use product of univariate kernels:

$$K(\mathbf{u}) = \prod_{j=1}^d K(u_j)$$

$$\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n \prod_{j=1}^d \frac{1}{h_j} K\left(\frac{x_j - x_{ij}}{h_j}\right)$$

### Curse of Dimensionality

KDE performance degrades in high dimensions:
- Need exponentially more data
- Bandwidth selection becomes difficult
- May require dimension reduction first

## Histogram-Based Methods

Histograms provide simple density estimates but have limitations.

### Fixed Bin Width

Divide range into $m$ equal-width bins:

$$\hat{p}(x) = \frac{n_j}{n \Delta}$$

where $n_j$ is count in bin containing $x$, $\Delta$ is bin width.

**Optimal Bin Width** (for Gaussian data):
$$\Delta = 3.49 \hat{\sigma} n^{-1/3}$$

### Variable Bin Width

Adapt bin widths to data density:
- Narrow bins in dense regions
- Wide bins in sparse regions

### Advantages

- Simple and fast
- No bandwidth selection needed
- Interpretable

### Disadvantages

- Discontinuous
- Sensitive to bin placement
- Poor in high dimensions
- May miss smooth structure

## Mixture Models

Mixture models combine parametric components for flexibility.

### Gaussian Mixture Model

$$p(\mathbf{x}) = \sum_{j=1}^k \pi_j \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_j, \Sigma_j)$$

where $\pi_j$ are mixing coefficients ($\sum_j \pi_j = 1$).

### Estimation via EM

**E-step**: Compute responsibilities
$$\gamma_{ij} = \frac{\pi_j \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_j, \Sigma_j)}{\sum_{l=1}^k \pi_l \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_l, \Sigma_l)}$$

**M-step**: Update parameters
$$\boldsymbol{\mu}_j = \frac{\sum_{i=1}^n \gamma_{ij} \mathbf{x}_i}{\sum_{i=1}^n \gamma_{ij}}$$

$$\Sigma_j = \frac{\sum_{i=1}^n \gamma_{ij} (\mathbf{x}_i - \boldsymbol{\mu}_j)(\mathbf{x}_i - \boldsymbol{\mu}_j)^T}{\sum_{i=1}^n \gamma_{ij}}$$

$$\pi_j = \frac{1}{n}\sum_{i=1}^n \gamma_{ij}$$

### Advantages

- Combines flexibility of non-parametric with efficiency of parametric
- Can model multi-modal distributions
- Probabilistic framework

### Model Selection

Choose number of components $k$:
- Cross-validation
- Information criteria (AIC, BIC)
- Bayesian methods

## Applications

### Anomaly Detection

Identify points with low density:
- Set threshold $\tau$
- Flag points where $\hat{p}(\mathbf{x}) < \tau$

### Classification

Estimate class-conditional densities for Bayes classifier:

$$P(y | \mathbf{x}) = \frac{p(\mathbf{x} | y) P(y)}{p(\mathbf{x})}$$

### Generative Modeling

Sample from estimated density:
- Parametric: Sample from fitted distribution
- KDE: Sample data point, add noise from kernel
- GMM: Sample component, then sample from component

### Data Visualization

Visualize estimated density:
- 1D: Plot $\hat{p}(x)$
- 2D: Contour plot
- Higher-D: Projections or slices

### Hypothesis Testing

Test if data follows specific distribution:
- Compare estimated density to theoretical
- Use goodness-of-fit tests

## Key Takeaways

1. **Density Estimation** estimates probability density $p(\mathbf{x})$ from samples, used for understanding data, generative modeling, anomaly detection, and classification.

2. **Parametric Methods** assume specific distribution (e.g., Gaussian), estimating parameters via MLE: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$, efficient but restrictive.

3. **Kernel Density Estimation** provides smooth non-parametric estimate: $\hat{p}(x) = \frac{1}{nh}\sum_i K(\frac{x-x_i}{h})$, placing kernels at each data point with bandwidth $h$ controlling smoothness.

4. **Bandwidth Selection** is crucial: Silverman's rule $h = 1.06\hat{\sigma}n^{-1/5}$, cross-validation maximizes log-likelihood, with small $h$ giving detail but high variance, large $h$ smoothing but higher bias.

5. **Multivariate KDE** extends to $d$ dimensions using bandwidth matrix $\mathbf{H}$, with product kernels $\prod_j K(u_j)$ for efficiency, though suffering from curse of dimensionality.

6. **Histogram Methods** divide range into bins and normalize counts, simple but discontinuous and sensitive to bin placement, with optimal bin width $\Delta = 3.49\hat{\sigma}n^{-1/3}$ for Gaussian data.

7. **Mixture Models** combine parametric components (e.g., GMM: $\sum_j \pi_j \mathcal{N}(\boldsymbol{\mu}_j, \Sigma_j)$), estimated via EM algorithm, providing flexibility while maintaining efficiency.

8. **Kernel Choice** (Gaussian, Epanechnikov, uniform, triangular) matters less than bandwidth selection, with Gaussian being most commonly used for smoothness.

9. **Applications** include anomaly detection (low-density regions), classification (class-conditional densities), generative modeling (sampling), visualization, and hypothesis testing.

10. **Method Selection** depends on data characteristics: parametric for known distributions, KDE for flexibility with sufficient data, histograms for quick estimates, mixture models for multi-modal data, with bandwidth/model selection critical for performance.
