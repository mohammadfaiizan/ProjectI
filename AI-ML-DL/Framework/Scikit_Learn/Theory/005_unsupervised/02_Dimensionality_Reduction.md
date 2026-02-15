# Dimensionality Reduction

---

## Table of Contents

- [Overview](#overview)
- [PCA](#pca)
- [Kernel PCA](#kernel-pca)
- [Incremental PCA](#incremental-pca)
- [Sparse PCA](#sparse-pca)
- [Truncated SVD and NMF](#truncated-svd-and-nmf)
- [t-SNE and UMAP](#t-sne-and-umap)
- [Manifold Learning](#manifold-learning)
- [Choosing a Method](#choosing-a-method)
- [Best Practices](#best-practices)

---

## Overview

**Dimensionality reduction** projects high-dimensional data into a lower-dimensional space. Goals include visualization, noise reduction, compression, and faster downstream modeling. Methods differ in linear vs non-linear, supervised vs unsupervised, and scalability.

| Method | Type | Use Case |
|--------|------|----------|
| **PCA** | Linear | General purpose, fast |
| **Kernel PCA** | Non-linear | Non-linear structure |
| **Truncated SVD** | Linear | Sparse data, LSA |
| **NMF** | Linear, non-negative | Non-negative data |
| **t-SNE** | Non-linear | Visualization (2D/3D) |
| **MDS, Isomap, LLE** | Manifold | Non-linear structure |

---

## PCA

**PCA** (Principal Component Analysis) finds orthogonal directions of maximum variance. Projects data onto these principal components. Assumes linear structure.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_components** | Number of components (int, float, or "mle") |
| **svd_solver** | 'auto', 'full', 'arpack', 'randomized' |
| **whiten** | Scale components to unit variance |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| **components_** | Principal axes (eigenvectors) |
| **explained_variance_** | Variance per component |
| **explained_variance_ratio_** | Fraction of total variance |
| **mean_** | Per-feature mean |

### Usage

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
```

### Variance Retention

```python
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_reduced = pca.fit_transform(X)
print(pca.n_components_)  # Actual number chosen
```

---

## Kernel PCA

**Kernel PCA** applies PCA in a kernel-induced feature space. Captures non-linear structure via RBF, polynomial, or other kernels.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_components** | Number of components |
| **kernel** | 'rbf', 'poly', 'sigmoid', 'cosine', 'precomputed' |
| **gamma** | RBF/poly/sigmoid kernel coefficient |
| **degree** | Polynomial kernel degree |

### Usage

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = kpca.fit_transform(X)
```

### Inverse Transform

Kernel PCA does not provide exact inverse by default. Use **fit_inverse_transform=True** for approximate reconstruction.

---

## Incremental PCA

**Incremental PCA** processes data in batches. Suitable for data that does not fit in memory.

### Usage

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=2, batch_size=100)
for batch in np.array_split(X, 10):
    ipca.partial_fit(batch)
X_reduced = ipca.transform(X)
```

---

## Sparse PCA

**Sparse PCA** finds sparse components (many zeros). Improves interpretability when each component should involve few features.

### Usage

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=5, alpha=1.0)
X_reduced = spca.fit_transform(X)
print(spca.components_.shape)
```

---

## Truncated SVD and NMF

**TruncatedSVD** computes the truncated SVD (no centering). Used for sparse matrices (e.g., document-term matrices in LSA).

**NMF** (Non-negative Matrix Factorization) constrains factors to be non-negative. For data that is naturally non-negative (e.g., counts, images).

### Truncated SVD

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5)
X_reduced = svd.fit_transform(X_sparse)
```

### NMF

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=5, init="nndsvda", random_state=42)
X_reduced = nmf.fit_transform(X_nonnegative)
```

---

## t-SNE and UMAP

**t-SNE** (t-Distributed Stochastic Neighbor Embedding) emphasizes local structure. Excellent for 2D/3D visualization. Not for feature extraction (non-deterministic, no transform for new points in standard API).

**UMAP** is not in scikit-learn but is a popular alternative. Use the **umap-learn** package.

### t-SNE

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)
# Use for visualization only
```

### Important Notes

- **perplexity**: Typically 5-50; higher for larger datasets
- **fit_transform** only; no separate transform for new data
- Results vary with random_state; run multiple times for stability

---

## Manifold Learning

**Manifold learning** assumes data lies on a lower-dimensional manifold. Methods include **Isomap**, **LLE**, **MDS**, **SpectralEmbedding**.

### Isomap

```python
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2, n_neighbors=5)
X_embedded = isomap.fit_transform(X)
```

### LLE (Locally Linear Embedding)

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_embedded = lle.fit_transform(X)
```

### MDS (Multidimensional Scaling)

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42)
X_embedded = mds.fit_transform(X)
```

---

## Choosing a Method

| Goal | Recommended Method |
|------|-------------------|
| General linear reduction | PCA |
| Sparse data | Truncated SVD |
| Non-negative data | NMF |
| Non-linear structure | Kernel PCA, Isomap, LLE |
| 2D/3D visualization | t-SNE |
| Large data, memory limit | Incremental PCA |
| Interpretable components | Sparse PCA |

---

## Best Practices

| Practice | Reason |
|----------|--------|
| **Scale** data before PCA | Variance-based; scale-sensitive |
| Use **explained_variance_ratio_** to choose n_components | Retain sufficient variance |
| Use **Incremental PCA** for large data | Memory efficiency |
| Use **t-SNE** for visualization only | Not for feature extraction |
| Use **random_state** for reproducibility | t-SNE, NMF, etc. |
