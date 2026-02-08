# Advanced Topics: Singular Value Decomposition and Principal Component Analysis

## Table of Contents

1. [Introduction](#introduction)
2. [Singular Value Decomposition](#singular-value-decomposition)
3. [SVD Derivation and Properties](#svd-derivation-and-properties)
4. [Low-Rank Approximation](#low-rank-approximation)
5. [Eckart-Young Theorem](#eckart-young-theorem)
6. [Principal Component Analysis](#principal-component-analysis)
7. [PCA Derivation from SVD](#pca-derivation-from-svd)
8. [Non-Negative Matrix Factorization](#non-negative-matrix-factorization)
9. [Applications](#applications)
10. [Key Takeaways](#key-takeaways)

## Introduction

Singular Value Decomposition (SVD) and Principal Component Analysis (PCA) are among the most powerful tools in linear algebra for dimensionality reduction, data compression, and feature extraction. SVD provides a fundamental decomposition of any matrix into constituent parts that reveal its structure, while PCA leverages this decomposition to find optimal low-dimensional representations of data. These techniques form the mathematical foundation for many modern machine learning algorithms.

SVD generalizes the eigendecomposition to non-square matrices and provides a stable numerical method for matrix factorization. PCA, derived from SVD, finds the directions of maximum variance in data, enabling efficient dimensionality reduction while preserving the most important information.

## Singular Value Decomposition

The Singular Value Decomposition of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ factorizes it into three matrices:

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$ is an orthogonal matrix (left singular vectors)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ is a diagonal matrix with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$ on the diagonal, where $r = \min(m, n)$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ is an orthogonal matrix (right singular vectors)

The singular values are ordered: $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$.

### Compact SVD

For a matrix of rank $k \leq r$, we can write the compact SVD:

$$\mathbf{A} = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

where $\mathbf{U}_k \in \mathbb{R}^{m \times k}$ contains the first $k$ columns of $\mathbf{U}$, $\mathbf{\Sigma}_k \in \mathbb{R}^{k \times k}$ contains the top $k$ singular values, and $\mathbf{V}_k \in \mathbb{R}^{n \times k}$ contains the first $k$ columns of $\mathbf{V}$.

### Connection to Eigendecomposition

SVD is related to eigendecomposition through:

$$\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{V}^T$$

$$\mathbf{AA}^T = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^T\mathbf{U}^T$$

The columns of $\mathbf{V}$ are eigenvectors of $\mathbf{A}^T\mathbf{A}$, and the columns of $\mathbf{U}$ are eigenvectors of $\mathbf{AA}^T$. The singular values are the square roots of the eigenvalues of $\mathbf{A}^T\mathbf{A}$ (or $\mathbf{AA}^T$).

## SVD Derivation and Properties

### Existence Theorem

Every matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ has a singular value decomposition. This follows from the spectral theorem applied to the symmetric positive semidefinite matrices $\mathbf{A}^T\mathbf{A}$ and $\mathbf{AA}^T$.

### Derivation via Eigendecomposition

To derive SVD:

1. Compute $\mathbf{A}^T\mathbf{A}$, which is symmetric and positive semidefinite
2. Find its eigendecomposition: $\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$
3. The singular values are $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are eigenvalues
4. The right singular vectors are the eigenvectors $\mathbf{V}$
5. The left singular vectors are: $\mathbf{u}_i = \frac{1}{\sigma_i}\mathbf{Av}_i$ for $\sigma_i > 0$

### Properties

**Uniqueness**: SVD is unique up to:
- Sign flips of corresponding columns in $\mathbf{U}$ and $\mathbf{V}$
- Permutations when singular values are equal

**Rank**: The rank of $\mathbf{A}$ equals the number of nonzero singular values:
$$\text{rank}(\mathbf{A}) = |\{i : \sigma_i > 0\}|$$

**Frobenius Norm**: The Frobenius norm can be expressed in terms of singular values:
$$||\mathbf{A}||_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\sum_{i=1}^{r} \sigma_i^2}$$

**Spectral Norm**: The operator (spectral) norm equals the largest singular value:
$$||\mathbf{A}||_2 = \sigma_1$$

**Condition Number**: The condition number of $\mathbf{A}$ is:
$$\kappa(\mathbf{A}) = \frac{\sigma_1}{\sigma_r}$$

### Numerical Computation

SVD is computed using iterative algorithms such as:
- Golub-Reinsch algorithm (QR-based)
- Divide-and-conquer methods
- Randomized SVD for large matrices

Modern implementations use Householder reflections and Givens rotations for numerical stability.

## Low-Rank Approximation

One of the most important applications of SVD is low-rank matrix approximation. Given a matrix $\mathbf{A}$ with SVD $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, the best rank-$k$ approximation is:

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

This approximation minimizes the reconstruction error under both Frobenius and spectral norms.

### Approximation Error

The error of the rank-$k$ approximation is:

$$||\mathbf{A} - \mathbf{A}_k||_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

$$||\mathbf{A} - \mathbf{A}_k||_2 = \sigma_{k+1}$$

The fraction of variance explained by the top $k$ components is:

$$\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}$$

### Applications

Low-rank approximation is used for:
- **Data compression**: Representing high-dimensional data with fewer parameters
- **Noise reduction**: Removing components corresponding to small singular values
- **Collaborative filtering**: Recommender systems factorize user-item matrices
- **Image compression**: Representing images with fewer components

## Eckart-Young Theorem

The Eckart-Young-Mirsky theorem provides the theoretical foundation for low-rank approximation:

**Theorem**: For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ with SVD $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, and for any $k < \text{rank}(\mathbf{A})$, the best rank-$k$ approximation under the Frobenius norm is:

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

with error:
$$\min_{\text{rank}(\mathbf{B}) \leq k} ||\mathbf{A} - \mathbf{B}||_F = ||\mathbf{A} - \mathbf{A}_k||_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

The same result holds for the spectral norm:
$$\min_{\text{rank}(\mathbf{B}) \leq k} ||\mathbf{A} - \mathbf{B}||_2 = \sigma_{k+1}$$

### Proof Sketch

The proof relies on showing that any rank-$k$ matrix $\mathbf{B}$ must have its $k$ largest singular values bounded by those of $\mathbf{A}$, and the optimal choice sets them equal. The key insight is that the singular values of $\mathbf{A} - \mathbf{B}$ are minimized when $\mathbf{B}$ captures the top $k$ components of $\mathbf{A}$.

## Principal Component Analysis

Principal Component Analysis is a dimensionality reduction technique that finds orthogonal directions of maximum variance in data. Given a data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ with $n$ samples and $d$ features, PCA seeks to:

1. Center the data: $\tilde{\mathbf{X}} = \mathbf{X} - \boldsymbol{\mu}$ where $\boldsymbol{\mu}$ is the mean vector
2. Find principal components (directions of maximum variance)
3. Project data onto the top $k$ principal components

### Variance Maximization Formulation

The first principal component $\mathbf{w}_1$ maximizes:

$$\mathbf{w}_1 = \arg\max_{||\mathbf{w}||=1} \text{Var}(\mathbf{X}\mathbf{w}) = \arg\max_{||\mathbf{w}||=1} \mathbf{w}^T\mathbf{\Sigma}\mathbf{w}$$

where $\mathbf{\Sigma} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$ is the covariance matrix.

Subsequent components $\mathbf{w}_i$ maximize variance subject to orthogonality:
$$\mathbf{w}_i = \arg\max_{||\mathbf{w}||=1, \mathbf{w}^T\mathbf{w}_j=0 \forall j<i} \mathbf{w}^T\mathbf{\Sigma}\mathbf{w}$$

### Geometric Interpretation

PCA finds the axes that best fit the data cloud. The first principal component points in the direction of maximum spread, the second in the direction of maximum remaining spread orthogonal to the first, and so on.

## PCA Derivation from SVD

PCA can be derived directly from SVD. For mean-centered data $\tilde{\mathbf{X}} \in \mathbb{R}^{n \times d}$:

1. Compute SVD: $\tilde{\mathbf{X}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$

2. The principal components are the columns of $\mathbf{V}$ (right singular vectors)

3. The principal component scores are: $\mathbf{Y} = \tilde{\mathbf{X}}\mathbf{V}_k = \mathbf{U}_k\mathbf{\Sigma}_k$

4. The explained variance by component $i$ is: $\frac{\sigma_i^2}{\sum_{j=1}^{r} \sigma_j^2}$

### Covariance Matrix Connection

The covariance matrix is:
$$\mathbf{\Sigma} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} = \frac{1}{n-1}\mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^T$$

The eigendecomposition of $\mathbf{\Sigma}$ gives:
- Eigenvectors = principal components (columns of $\mathbf{V}$)
- Eigenvalues = $\frac{\sigma_i^2}{n-1}$ = variances along principal components

### Dimensionality Reduction

To reduce dimensionality from $d$ to $k$:
1. Select top $k$ principal components: $\mathbf{V}_k \in \mathbb{R}^{d \times k}$
2. Project data: $\mathbf{Y} = \tilde{\mathbf{X}}\mathbf{V}_k \in \mathbb{R}^{n \times k}$
3. Reconstruct: $\hat{\mathbf{X}} = \mathbf{Y}\mathbf{V}_k^T + \boldsymbol{\mu}$

The reconstruction error is minimized by the Eckart-Young theorem.

## Non-Negative Matrix Factorization

Non-Negative Matrix Factorization (NMF) factorizes a non-negative matrix $\mathbf{X} \in \mathbb{R}_{\geq 0}^{m \times n}$ into:

$$\mathbf{X} \approx \mathbf{W}\mathbf{H}$$

where $\mathbf{W} \in \mathbb{R}_{\geq 0}^{m \times k}$ and $\mathbf{H} \in \mathbb{R}_{\geq 0}^{k \times n}$ are non-negative matrices.

### Optimization Problem

NMF solves:
$$\min_{\mathbf{W} \geq 0, \mathbf{H} \geq 0} ||\mathbf{X} - \mathbf{WH}||_F^2$$

This is a non-convex optimization problem typically solved using:
- Multiplicative update rules
- Alternating least squares
- Gradient descent with projection

### Properties

- **Parts-based representation**: NMF produces additive, parts-based representations unlike PCA's holistic components
- **Sparsity**: Often produces sparse factors
- **Interpretability**: Non-negativity constraint improves interpretability

### Applications

- Topic modeling: $\mathbf{W}$ represents topics, $\mathbf{H}$ represents document-topic assignments
- Image analysis: Parts-based decomposition
- Audio source separation
- Recommender systems with non-negative constraints

## Applications

### Image Compression

SVD-based image compression represents an image matrix $\mathbf{I} \in \mathbb{R}^{m \times n}$ as:

$$\mathbf{I} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

Storage requirements:
- Original: $mn$ values
- Compressed: $k(m + n + 1)$ values
- Compression ratio: $\frac{mn}{k(m+n+1)}$

For $k \ll \min(m,n)$, significant compression is achieved with minimal quality loss.

### Recommender Systems

Collaborative filtering factorizes the user-item rating matrix $\mathbf{R} \in \mathbb{R}^{m \times n}$:

$$\mathbf{R} \approx \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

where:
- $\mathbf{U}$ captures user preferences
- $\mathbf{V}$ captures item characteristics
- Missing ratings are predicted via: $\hat{r}_{ij} = \mathbf{u}_i^T \mathbf{\Sigma} \mathbf{v}_j$

### Latent Semantic Analysis

In natural language processing, term-document matrices are factorized to discover latent topics:

$$\mathbf{T} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

where $\mathbf{T}_{ij}$ represents the frequency of term $i$ in document $j$. The columns of $\mathbf{V}_k$ represent document-topic associations.

### Face Recognition

Eigenfaces use PCA to represent faces in a low-dimensional space:
1. Vectorize face images into columns of $\mathbf{X}$
2. Compute PCA: $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$
3. Represent faces as coefficients: $\mathbf{y} = \mathbf{V}_k^T(\mathbf{x} - \boldsymbol{\mu})$
4. Recognition via nearest neighbor in coefficient space

### Dimensionality Reduction for Visualization

PCA reduces high-dimensional data to 2D or 3D for visualization:
- Preserves maximum variance
- Enables exploration of data structure
- Useful for understanding clusters and patterns

### Noise Reduction

Small singular values often correspond to noise. Truncating SVD removes noise:
$$\mathbf{A}_{\text{denoised}} = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where $k$ is chosen to retain signal while discarding noise.

## Key Takeaways

1. **SVD provides a universal decomposition** of any matrix into orthogonal components and singular values, generalizing eigendecomposition to non-square matrices.

2. **Low-rank approximation via SVD** is optimal under Frobenius and spectral norms, as guaranteed by the Eckart-Young theorem.

3. **PCA is derived from SVD** of the mean-centered data matrix, with principal components being right singular vectors and explained variance proportional to squared singular values.

4. **The rank-$k$ approximation error** equals the sum of squares of discarded singular values, providing a principled way to choose dimensionality.

5. **NMF provides parts-based factorization** with non-negativity constraints, offering interpretable decompositions for applications like topic modeling.

6. **SVD enables efficient compression** by representing matrices with fewer parameters while minimizing reconstruction error.

7. **Principal components maximize variance** sequentially while maintaining orthogonality, providing the optimal linear dimensionality reduction.

8. **Applications span many domains**: image compression, recommender systems, topic modeling, face recognition, and noise reduction all leverage SVD/PCA.

9. **Numerical stability**: SVD is more numerically stable than eigendecomposition for near-singular matrices, making it preferred in practice.

10. **Modern extensions**: Randomized SVD enables efficient computation for very large matrices, while kernel PCA extends PCA to nonlinear manifolds.
