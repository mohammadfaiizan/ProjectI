# Dimensionality Reduction PCA ICA

## Table of Contents

1. [Introduction to Dimensionality Reduction](#introduction-to-dimensionality-reduction)
2. [Principal Component Analysis](#principal-component-analysis)
3. [Kernel PCA](#kernel-pca)
4. [Independent Component Analysis](#independent-component-analysis)
5. [t-SNE](#t-sne)
6. [UMAP](#umap)
7. [Manifold Learning](#manifold-learning)
8. [Linear vs Nonlinear Methods](#linear-vs-nonlinear-methods)
9. [Applications and Use Cases](#applications-and-use-cases)
10. [Key Takeaways](#key-takeaways)

## Introduction to Dimensionality Reduction

Dimensionality reduction transforms high-dimensional data into a lower-dimensional representation while preserving important information.

### Motivation

**Curse of Dimensionality**: As dimensionality increases:
- Data becomes sparse
- Distances become less meaningful
- Computational cost increases
- Overfitting risk increases

**Benefits**:
- **Visualization**: Project to 2D/3D for visualization
- **Compression**: Reduce storage and computation
- **Noise Reduction**: Remove irrelevant dimensions
- **Feature Extraction**: Discover underlying structure

### Types of Dimensionality Reduction

**Linear Methods**: 
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Linear Discriminant Analysis (LDA)

**Nonlinear Methods**:
- Kernel PCA
- t-SNE
- UMAP
- Manifold learning (Isomap, LLE)

**Supervised vs Unsupervised**:
- **Unsupervised**: PCA, ICA, t-SNE (no labels)
- **Supervised**: LDA (uses class labels)

## Principal Component Analysis

PCA finds orthogonal directions of maximum variance in the data.

### Problem Formulation

Given data matrix $X \in \mathbb{R}^{n \times d}$ (centered: mean zero), find projection matrix $W \in \mathbb{R}^{d \times k}$ ($k < d$) that maximizes variance of projected data.

### Optimization Problem

**Variance Maximization**:
$$\max_{W} \text{tr}(W^T X^T X W) \quad \text{subject to} \quad W^T W = I$$

The constraint ensures orthonormal columns (principal components).

### Solution via Eigendecomposition

The solution is given by eigenvectors of covariance matrix $C = \frac{1}{n}X^T X$:

$$C \mathbf{w}_j = \lambda_j \mathbf{w}_j$$

where:
- $\mathbf{w}_j$: $j$-th principal component (eigenvector)
- $\lambda_j$: $j$-th eigenvalue (variance along $\mathbf{w}_j$)

**Principal Components**: Eigenvectors ordered by decreasing eigenvalues.

### Projection

Project data onto first $k$ principal components:

$$Z = X W_k$$

where $W_k = [\mathbf{w}_1, \ldots, \mathbf{w}_k]$ contains first $k$ eigenvectors.

### Reconstruction

Reconstruct original data:

$$\hat{X} = Z W_k^T = X W_k W_k^T$$

**Reconstruction Error**: 
$$\|X - \hat{X}\|_F^2 = \sum_{j=k+1}^d \lambda_j$$

Error equals sum of discarded eigenvalues.

### Explained Variance

**Proportion of Variance Explained**:
$$\text{PVE}_j = \frac{\lambda_j}{\sum_{i=1}^d \lambda_i}$$

**Cumulative PVE**:
$$\text{Cumulative PVE}_k = \frac{\sum_{j=1}^k \lambda_j}{\sum_{i=1}^d \lambda_i}$$

Choose $k$ such that cumulative PVE exceeds threshold (e.g., 0.95).

### SVD Formulation

PCA can be computed via Singular Value Decomposition (SVD):

$$X = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{n \times n}$: Left singular vectors
- $\Sigma \in \mathbb{R}^{n \times d}$: Singular values (diagonal)
- $V \in \mathbb{R}^{d \times d}$: Right singular vectors (principal components)

**Principal Components**: Columns of $V$ (right singular vectors)
**Variances**: $\sigma_j^2/n$ where $\sigma_j$ are singular values

### Properties

- **Orthogonal**: Principal components are orthogonal
- **Uncorrelated**: Projected data has zero correlation
- **Optimal Reconstruction**: Minimizes mean squared reconstruction error
- **Scale Sensitive**: Requires feature scaling

### Limitations

- **Linear**: Assumes linear relationships
- **Variance-Based**: May not preserve important structure
- **Global**: Considers all data points equally
- **Gaussian Assumption**: Optimal for Gaussian data

## Kernel PCA

Kernel PCA extends PCA to nonlinear relationships using the kernel trick.

### Idea

Map data to high-dimensional feature space $\phi(\mathbf{x})$, then apply PCA:

$$C = \frac{1}{n}\sum_{i=1}^n \phi(\mathbf{x}_i) \phi(\mathbf{x}_i)^T$$

### Kernel Trick

Eigenvalue problem becomes:

$$K \boldsymbol{\alpha} = n\lambda \boldsymbol{\alpha}$$

where $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$ is the kernel matrix.

**Projection**:
$$z_j = \sum_{i=1}^n \alpha_{ij} k(\mathbf{x}_i, \mathbf{x})$$

### Kernel Functions

**Polynomial**: $k(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$

**RBF**: $k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$

**Sigmoid**: $k(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$

### Advantages

- Captures nonlinear structure
- Can work in infinite-dimensional spaces
- Same computational complexity as linear PCA (after kernel computation)

### Disadvantages

- Requires storing kernel matrix ($O(n^2)$ memory)
- Slower for large datasets
- Kernel selection is crucial

## Independent Component Analysis

ICA finds statistically independent components, useful for blind source separation.

### Problem Formulation

Assume data is linear mixture of independent sources:

$$\mathbf{x} = A \mathbf{s}$$

where:
- $\mathbf{x}$: Observed data
- $\mathbf{s}$: Independent sources
- $A$: Mixing matrix

**Goal**: Recover sources $\mathbf{s}$ and mixing matrix $A$ (or unmixing matrix $W = A^{-1}$).

### Independence vs Uncorrelation

**Uncorrelation**: $\text{Cov}(s_i, s_j) = 0$ (weaker)

**Independence**: $p(s_i, s_j) = p(s_i)p(s_j)$ (stronger)

PCA finds uncorrelated components; ICA finds independent components.

### Assumptions

1. **Independence**: Sources are statistically independent
2. **Non-Gaussian**: At most one source can be Gaussian
3. **Linear Mixing**: Observed data is linear combination of sources

### Objective Function

Maximize non-Gaussianity (independence implies non-Gaussianity for mixed signals):

**Negentropy**:
$$J(y) = H(y_{\text{Gauss}}) - H(y)$$

where $H$ is entropy. Higher negentropy implies more non-Gaussian.

**Kurtosis**:
$$\text{kurt}(y) = E[y^4] - 3(E[y^2])^2$$

Maximize absolute kurtosis.

### FastICA Algorithm

1. Center and whiten data: $\mathbf{z} = V \mathbf{x}$ where $V$ whitening matrix
2. Initialize random weight vector $\mathbf{w}$
3. Update: $\mathbf{w} \leftarrow E[\mathbf{z} g(\mathbf{w}^T \mathbf{z})] - E[g'(\mathbf{w}^T \mathbf{z})] \mathbf{w}$
   where $g$ is nonlinear function (e.g., $\tanh$)
4. Normalize: $\mathbf{w} \leftarrow \mathbf{w} / \|\mathbf{w}\|$
5. Repeat until convergence
6. Extract multiple components using deflation

### Applications

- **Blind Source Separation**: Separate audio signals, images
- **Feature Extraction**: Find independent features
- **Signal Processing**: Remove artifacts, denoising

### Limitations

- **Identifiability**: Order and scale of components ambiguous
- **Non-Gaussian Requirement**: Fails if all sources Gaussian
- **Linear Assumption**: Assumes linear mixing

## t-SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) preserves local neighborhoods in low-dimensional space.

### Idea

- **High-Dimensional**: Compute similarities between points
- **Low-Dimensional**: Embed points preserving these similarities
- **t-Distribution**: Use heavy-tailed distribution in low-dim space to avoid crowding

### Algorithm

**High-Dimensional Similarities**:
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**Low-Dimensional Similarities** (using t-distribution):
$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**Objective**: Minimize KL divergence:
$$KL(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### Properties

- **Local Structure**: Preserves local neighborhoods well
- **Nonlinear**: Captures nonlinear manifolds
- **Crowding Problem**: t-distribution prevents points from crowding in center
- **Stochastic**: Results vary with different initializations

### Limitations

- **Computational Cost**: $O(n^2)$ complexity
- **Hyperparameters**: Perplexity parameter affects results
- **Global Structure**: May not preserve global structure
- **Non-Deterministic**: Different runs yield different results

### Perplexity

Controls number of effective neighbors:
- Low perplexity: Focus on local structure
- High perplexity: Consider more neighbors
- Typical: 5-50

## UMAP

Uniform Manifold Approximation and Projection (UMAP) preserves both local and global structure.

### Theoretical Foundation

Based on manifold learning and topological data analysis:
- Assumes data lies on Riemannian manifold
- Preserves topological structure (neighborhoods, connectivity)

### Algorithm

**High-Dimensional Graph**:
1. Find $k$ nearest neighbors for each point
2. Compute fuzzy simplicial set (weighted graph)
3. Weights: $w_{ij} = \exp(-\max(0, d_{ij} - \rho_i) / \sigma_i)$
   where $\rho_i$ is distance to nearest neighbor, $\sigma_i$ chosen to match desired neighbors

**Low-Dimensional Graph**:
1. Initialize points (e.g., via spectral embedding)
2. Optimize embedding to match high-dim graph structure
3. Use cross-entropy as objective

### Advantages

- **Scalable**: More efficient than t-SNE ($O(n^{1.14})$ vs $O(n^2)$)
- **Global Structure**: Preserves both local and global structure
- **Deterministic**: Reproducible results
- **Dimensionality Reduction**: Can reduce to any dimension (not just 2D/3D)

### Parameters

- **$n_{\text{neighbors}}$**: Number of neighbors (typically 15-100)
- **$min_{\text{dist}}$**: Minimum distance in low-dim space (typically 0.1-0.5)
- **$n_{\text{components}}$**: Target dimensionality

### Comparison with t-SNE

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| Local Structure | Excellent | Excellent |
| Global Structure | Poor | Good |
| Speed | Slow | Faster |
| Scalability | Poor | Better |
| Deterministic | No | Yes |

## Manifold Learning

Manifold learning assumes data lies on a low-dimensional manifold embedded in high-dimensional space.

### Isomap

Extends MDS to nonlinear manifolds:
1. Construct neighborhood graph (k-NN or $\epsilon$-ball)
2. Compute geodesic distances (shortest paths on graph)
3. Apply MDS to geodesic distance matrix

**Geodesic Distance**: Distance along manifold (approximated by graph shortest path)

### Locally Linear Embedding (LLE)

Preserves local linear structure:
1. For each point, find $k$ nearest neighbors
2. Compute weights $W$ that reconstruct point from neighbors: $\min \|\mathbf{x}_i - \sum_j W_{ij} \mathbf{x}_j\|^2$
3. Find low-dim embedding preserving these weights: $\min \sum_i \|\mathbf{y}_i - \sum_j W_{ij} \mathbf{y}_j\|^2$

**Properties**: Preserves local geometry, invariant to rotations/translations

### Laplacian Eigenmaps

Uses graph Laplacian:
1. Construct neighborhood graph
2. Compute graph Laplacian $L = D - W$
3. Find eigenvectors of $L$ (smallest eigenvalues)
4. Use as embedding coordinates

**Properties**: Preserves local structure, smooth embedding

## Linear vs Nonlinear Methods

### Linear Methods

**PCA, ICA, Factor Analysis**:
- Fast computation
- Interpretable (linear transformation)
- Global structure
- Limited to linear relationships

**When to Use**: 
- Data has linear structure
- Need interpretability
- Large datasets
- Preprocessing step

### Nonlinear Methods

**Kernel PCA, t-SNE, UMAP, Manifold Learning**:
- Captures nonlinear structure
- Better for complex manifolds
- More flexible
- Computationally expensive
- Less interpretable

**When to Use**:
- Nonlinear structure suspected
- Visualization needed
- Complex relationships
- Smaller datasets

## Applications and Use Cases

### Visualization

- **t-SNE/UMAP**: Visualize high-dim data in 2D/3D
- **PCA**: Quick linear visualization
- **Applications**: Exploratory data analysis, presentation

### Feature Extraction

- **PCA**: Reduce dimensionality before classification/regression
- **ICA**: Extract independent features
- **Applications**: Image processing, signal processing

### Compression

- **PCA**: Compress data while preserving variance
- **Applications**: Image compression, data storage

### Noise Reduction

- **PCA**: Remove dimensions with low variance (often noise)
- **Applications**: Signal denoising, preprocessing

### Preprocessing

- **PCA**: Reduce dimensionality before other algorithms
- **Benefits**: Faster computation, reduced overfitting
- **Applications**: Before classification, regression, clustering

## Key Takeaways

1. **Dimensionality Reduction** transforms high-dimensional data to lower dimensions, addressing curse of dimensionality through visualization, compression, noise reduction, and feature extraction.

2. **Principal Component Analysis** finds orthogonal directions of maximum variance via eigendecomposition of covariance matrix, with principal components as eigenvectors ordered by eigenvalues.

3. **PCA Properties** include optimal reconstruction (minimizes MSE), explained variance $\text{PVE}_j = \lambda_j / \sum \lambda_i$, and can be computed via SVD: $X = U\Sigma V^T$ where $V$ contains principal components.

4. **Kernel PCA** extends PCA to nonlinear relationships using kernel trick, computing eigenvectors of kernel matrix $K$ instead of covariance matrix, enabling nonlinear dimensionality reduction.

5. **Independent Component Analysis** finds statistically independent components for blind source separation, maximizing non-Gaussianity (negentropy/kurtosis) rather than just uncorrelation like PCA.

6. **t-SNE** preserves local neighborhoods using t-distribution in low-dim space to prevent crowding, minimizing KL divergence between high-dim and low-dim similarity distributions, excellent for visualization.

7. **UMAP** preserves both local and global structure based on manifold learning theory, more scalable than t-SNE with $O(n^{1.14})$ complexity, deterministic and suitable for any target dimensionality.

8. **Manifold Learning** (Isomap, LLE, Laplacian Eigenmaps) assumes data on low-dim manifold, preserving geodesic distances (Isomap) or local linear structure (LLE) through graph-based methods.

9. **Linear vs Nonlinear**: Linear methods (PCA, ICA) are fast and interpretable but limited to linear relationships; nonlinear methods (Kernel PCA, t-SNE, UMAP) capture complex structure but are computationally expensive.

10. **Method Selection** depends on data structure (linear vs nonlinear), purpose (visualization vs feature extraction), dataset size, and interpretability needs, with PCA for linear/quick preprocessing, t-SNE for visualization, UMAP for scalable nonlinear reduction, and ICA for source separation.
