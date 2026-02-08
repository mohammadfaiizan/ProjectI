# Multivariate Statistics and Distributions

## Table of Contents

1. [Introduction](#introduction)
2. [Multivariate Random Variables](#multivariate-random-variables)
3. [Multivariate Gaussian Distribution](#multivariate-gaussian-distribution)
4. [Covariance and Correlation](#covariance-and-correlation)
5. [Joint and Marginal Distributions](#joint-and-marginal-distributions)
6. [Conditional Distributions](#conditional-distributions)
7. [Multivariate Transformations](#multivariate-transformations)
8. [Gaussian Mixture Models](#gaussian-mixture-models)
9. [Bayesian Networks](#bayesian-networks)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Multivariate statistics extends probability theory to multiple random variables, enabling analysis of relationships, dependencies, and joint behavior. In machine learning, data is inherently multivariate (multiple features), and understanding multivariate distributions is essential for modeling, dimensionality reduction, and probabilistic inference. This document covers multivariate Gaussian distributions, covariance structures, conditional distributions, Gaussian mixture models, and Bayesian networks.

## Multivariate Random Variables

### Random Vectors

A **random vector** $\mathbf{X} = (X_1, \ldots, X_d)^T$ is a collection of random variables.

**Joint CDF**: $F_{\mathbf{X}}(\mathbf{x}) = P(X_1 \leq x_1, \ldots, X_d \leq x_d)$

**Joint PMF** (discrete): $p_{\mathbf{X}}(\mathbf{x}) = P(X_1 = x_1, \ldots, X_d = x_d)$

**Joint PDF** (continuous): $f_{\mathbf{X}}(\mathbf{x})$ such that:

$$P(\mathbf{X} \in A) = \int_A f_{\mathbf{X}}(\mathbf{x}) d\mathbf{x}$$

### Marginal Distributions

**Marginal PDF**: For continuous $\mathbf{X} = (X_1, X_2)^T$:

$$f_{X_1}(x_1) = \int_{-\infty}^{\infty} f_{X_1,X_2}(x_1, x_2) dx_2$$

**General**: $f_{X_i}(x_i) = \int f_{\mathbf{X}}(\mathbf{x}) d\mathbf{x}_{-i}$ where integration is over all variables except $X_i$.

### Independence

Random variables $X_1, \ldots, X_d$ are **mutually independent** if:

$$f_{\mathbf{X}}(\mathbf{x}) = \prod_{i=1}^d f_{X_i}(x_i)$$

**Pairwise independence** doesn't imply mutual independence.

## Multivariate Gaussian Distribution

### Definition

Random vector $\mathbf{X} \in \mathbb{R}^d$ follows **multivariate Gaussian** (multivariate normal) distribution if:

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

**Notation**: $\mathbf{X} \sim \mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

**Parameters**:
- **Mean vector**: $\boldsymbol{\mu} \in \mathbb{R}^d$
- **Covariance matrix**: $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ (symmetric, positive semidefinite)

### Properties

**Mean**: $\mathbb{E}[\mathbf{X}] = \boldsymbol{\mu}$

**Covariance**: $\text{Cov}(\mathbf{X}) = \boldsymbol{\Sigma}$

**Characteristic function**: $\phi_{\mathbf{X}}(\mathbf{t}) = \exp(i\boldsymbol{\mu}^T\mathbf{t} - \frac{1}{2}\mathbf{t}^T\boldsymbol{\Sigma}\mathbf{t})$

**Affine transformation**: If $\mathbf{X} \sim \mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:

$$\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}_m(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

### Standard Multivariate Gaussian

**Standard Gaussian**: $\mathbf{Z} \sim \mathcal{N}_d(\mathbf{0}, \mathbf{I})$ (zero mean, identity covariance)

**General Gaussian**: $\mathbf{X} = \boldsymbol{\Sigma}^{1/2}\mathbf{Z} + \boldsymbol{\mu} \sim \mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

### Mahalanobis Distance

**Mahalanobis distance** from $\mathbf{x}$ to $\boldsymbol{\mu}$:

$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$$

**Interpretation**: Accounts for covariance structure (unitless, scale-invariant).

**Contours**: Constant density surfaces are ellipsoids: $(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) = c$.

## Covariance and Correlation

### Covariance Matrix

**Covariance matrix**: $\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X})$ with entries:

$$\Sigma_{ij} = \text{Cov}(X_i, X_j) = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$$

**Properties**:
- Symmetric: $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$
- Positive semidefinite: $\mathbf{v}^T\boldsymbol{\Sigma}\mathbf{v} \geq 0$ for all $\mathbf{v}$
- Diagonal entries: $\Sigma_{ii} = \text{Var}(X_i)$

### Correlation Matrix

**Correlation matrix**: $\mathbf{R}$ with entries:

$$R_{ij} = \rho_{X_i,X_j} = \frac{\text{Cov}(X_i, X_j)}{\sqrt{\text{Var}(X_i)\text{Var}(X_j)}} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}$$

**Properties**:
- $|R_{ij}| \leq 1$
- $R_{ii} = 1$
- If $\mathbf{D} = \text{diag}(\sqrt{\Sigma_{11}}, \ldots, \sqrt{\Sigma_{dd}})$, then $\mathbf{R} = \mathbf{D}^{-1}\boldsymbol{\Sigma}\mathbf{D}^{-1}$

### Sample Covariance

**Sample covariance matrix**: For data $\mathbf{X} = [\mathbf{x}_1 | \cdots | \mathbf{x}_n] \in \mathbb{R}^{d \times n}$:

$$\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T = \frac{1}{n-1}(\mathbf{X} - \bar{\mathbf{x}}\mathbf{1}^T)(\mathbf{X} - \bar{\mathbf{x}}\mathbf{1}^T)^T$$

where $\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$ is sample mean.

**Matrix form**: $\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}\tilde{\mathbf{X}}\tilde{\mathbf{X}}^T$ where $\tilde{\mathbf{X}}$ has centered columns.

## Joint and Marginal Distributions

### Marginal of Multivariate Gaussian

**Theorem**: If $\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}\right)$

Then marginal: $\mathbf{X}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})$

**Proof**: Integrate joint PDF over $\mathbf{X}_2$.

### Block Matrix Notation

Partition mean and covariance:

$$\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$$

where $\boldsymbol{\Sigma}_{12} = \boldsymbol{\Sigma}_{21}^T$ (cross-covariance).

## Conditional Distributions

### Conditional Gaussian

**Theorem**: For partitioned $\mathbf{X} = (\mathbf{X}_1^T, \mathbf{X}_2^T)^T \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$

where:
- **Conditional mean**: $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$
- **Conditional covariance**: $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$

**Key insight**: Conditional distribution is Gaussian with mean that depends linearly on conditioning variable.

### Interpretation

**Conditional mean**: $\mathbb{E}[\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2] = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$

- If $\boldsymbol{\Sigma}_{12} = \mathbf{0}$ (uncorrelated), then $\mathbb{E}[\mathbf{X}_1 | \mathbf{X}_2] = \boldsymbol{\mu}_1$ (independent)
- Otherwise, mean shifts based on $\mathbf{x}_2$

**Conditional covariance**: $\text{Cov}(\mathbf{X}_1 | \mathbf{X}_2) = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} \preceq \boldsymbol{\Sigma}_{11}$

- Conditioning reduces uncertainty (covariance decreases)
- Reduction depends on correlation strength

### Schur Complement

**Schur complement**: $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ appears in matrix inversion:

$$\boldsymbol{\Sigma}^{-1} = \begin{pmatrix} \boldsymbol{\Sigma}_{1|2}^{-1} & -\boldsymbol{\Sigma}_{1|2}^{-1}\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1} \\ -\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}\boldsymbol{\Sigma}_{1|2}^{-1} & \boldsymbol{\Sigma}_{22}^{-1} + \boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}\boldsymbol{\Sigma}_{1|2}^{-1}\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1} \end{pmatrix}$$

## Multivariate Transformations

### Linear Transformations

**Affine transformation**: $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$

If $\mathbf{X} \sim \mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:

$$\mathbf{Y} \sim \mathcal{N}_m(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

**Special cases**:
- **Rotation**: $\mathbf{A}$ orthogonal, $\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$ rotates covariance ellipsoid
- **Scaling**: $\mathbf{A}$ diagonal, scales variances
- **Whitening**: $\mathbf{A} = \boldsymbol{\Sigma}^{-1/2}$ gives $\mathbf{Y} \sim \mathcal{N}(\boldsymbol{\Sigma}^{-1/2}\boldsymbol{\mu}, \mathbf{I})$

### Change of Variables

For transformation $\mathbf{Y} = g(\mathbf{X})$ with invertible $g$:

$$f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(g^{-1}(\mathbf{y})) \left|\det\left(\frac{\partial g^{-1}}{\partial \mathbf{y}}\right)\right|$$

where $\frac{\partial g^{-1}}{\partial \mathbf{y}}$ is Jacobian matrix.

## Gaussian Mixture Models

### Definition

**Gaussian Mixture Model (GMM)**:

$$f(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where:
- $\pi_k \geq 0$ are mixing weights with $\sum_{k=1}^K \pi_k = 1$
- Each component is multivariate Gaussian

**Latent variable**: $Z \in \{1, \ldots, K\}$ indicates which component:

$$P(Z = k) = \pi_k$$
$$\mathbf{X} | Z = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

### Parameter Estimation

**MLE**: Maximize log-likelihood:

$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^n \log \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**EM algorithm**: Iteratively:
1. **E-step**: Compute responsibilities $\gamma_{ik} = P(Z_i = k | \mathbf{x}_i)$
2. **M-step**: Update parameters given responsibilities

**Updates**:
- $\pi_k = \frac{1}{n}\sum_{i=1}^n \gamma_{ik}$
- $\boldsymbol{\mu}_k = \frac{\sum_{i=1}^n \gamma_{ik}\mathbf{x}_i}{\sum_{i=1}^n \gamma_{ik}}$
- $\boldsymbol{\Sigma}_k = \frac{\sum_{i=1}^n \gamma_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_{i=1}^n \gamma_{ik}}$

### Applications

- **Clustering**: Each component represents a cluster
- **Density estimation**: Flexible non-parametric density model
- **Anomaly detection**: Low density regions indicate anomalies

## Bayesian Networks

### Definition

**Bayesian network** (directed acyclic graph) represents conditional independence structure:

$$p(\mathbf{X}) = \prod_{i=1}^d p(X_i | \text{Pa}(X_i))$$

where $\text{Pa}(X_i)$ are parents of $X_i$ in graph.

**Markov property**: Each variable is independent of non-descendants given parents.

### Gaussian Bayesian Networks

**Linear Gaussian model**: Each variable depends linearly on parents:

$$X_i | \text{Pa}(X_i) \sim \mathcal{N}\left(\sum_{j \in \text{Pa}(i)} w_{ij}X_j + b_i, \sigma_i^2\right)$$

**Joint distribution**: Multivariate Gaussian with covariance determined by graph structure.

### Inference

**Exact inference**: Variable elimination, belief propagation

**Approximate inference**: Variational methods, MCMC

## Machine Learning Applications

### Multivariate Data Modeling

**Data representation**: $\mathbf{X} \in \mathbb{R}^{n \times d}$ where rows are samples, columns are features.

**Assumption**: Each sample $\mathbf{x}_i \sim \mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ (multivariate Gaussian).

### Principal Component Analysis

**PCA** finds principal directions (eigenvectors of covariance matrix):

$$\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$$

**Projection**: $\mathbf{z}_i = \mathbf{V}_k^T(\mathbf{x}_i - \boldsymbol{\mu})$ onto top $k$ principal components.

**Probabilistic PCA**: Assumes $\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$.

### Linear Discriminant Analysis

**LDA** assumes classes have Gaussian distributions with shared covariance:

$$p(\mathbf{x} | Y = k) = \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$$

**Decision boundary**: Linear (due to shared covariance).

**Classification**: $P(Y = k | \mathbf{x}) \propto \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma})$

### Multivariate Regression

**Multivariate regression**: $\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}$ where:

- $\mathbf{Y} \in \mathbb{R}^{n \times q}$: multiple outputs
- $\mathbf{B} \in \mathbb{R}^{d \times q}$: coefficient matrix
- $\mathbf{E} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_n \otimes \boldsymbol{\Sigma})$: error matrix

**MLE**: $\hat{\mathbf{B}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$

### Factor Analysis

**Factor model**: $\mathbf{x} = \boldsymbol{\Lambda}\mathbf{f} + \boldsymbol{\mu} + \boldsymbol{\epsilon}$

where:
- $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: latent factors
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi})$: noise (diagonal)
- $\boldsymbol{\Lambda}$: factor loadings

**Covariance**: $\boldsymbol{\Sigma} = \boldsymbol{\Lambda}\boldsymbol{\Lambda}^T + \boldsymbol{\Psi}$

### Kalman Filtering

**State-space model**: 

$$\mathbf{x}_t = \mathbf{F}_t\mathbf{x}_{t-1} + \mathbf{w}_t$$
$$\mathbf{y}_t = \mathbf{H}_t\mathbf{x}_t + \mathbf{v}_t$$

where $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)$, $\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)$.

**Kalman filter**: Recursive Bayesian inference for state $\mathbf{x}_t$ given observations $\mathbf{y}_{1:t}$.

### Variational Autoencoders

**VAE** assumes latent $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and:

$$p(\mathbf{x} | \mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{z}), \boldsymbol{\sigma}_\theta^2(\mathbf{z})\mathbf{I})$$

**Encoder**: $q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\Sigma}_\phi(\mathbf{x}))$

**Training**: Maximize ELBO using reparameterization trick.

### Gaussian Processes

**GP**: $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ where any finite collection is multivariate Gaussian:

$$(f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n))^T \sim \mathcal{N}(\mathbf{m}, \mathbf{K})$$

where $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.

**Prediction**: Posterior $p(f(\mathbf{x}_*) | \mathbf{y})$ is Gaussian with closed-form mean and covariance.

## Key Takeaways

1. **Multivariate Gaussian** is fundamental distribution for multivariate data, with elegant properties for conditioning and marginalization.

2. **Covariance matrix** captures pairwise relationships between variables, essential for understanding dependencies.

3. **Conditional distributions** of multivariate Gaussian are Gaussian, with mean depending linearly on conditioning variables.

4. **Marginal distributions** are also Gaussian, enabling tractable inference in high dimensions.

5. **Gaussian Mixture Models** provide flexible density estimation via weighted combination of Gaussians.

6. **Bayesian networks** represent conditional independence structure, enabling efficient probabilistic inference.

7. **PCA** uses eigendecomposition of covariance matrix to find principal directions of variation.

8. **Multivariate regression** extends univariate regression to multiple outputs with correlated errors.

9. **Factor analysis** models observed variables as linear combinations of latent factors plus noise.

10. **Multivariate statistics** provides foundation for understanding high-dimensional data, relationships between features, and probabilistic modeling in ML.
