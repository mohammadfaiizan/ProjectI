# Clustering Algorithms

---

## Table of Contents

- [Overview](#overview)
- [KMeans and MiniBatchKMeans](#kmeans-and-minibatchkmeans)
- [DBSCAN](#dbscan)
- [Hierarchical Clustering](#hierarchical-clustering)
- [Spectral Clustering](#spectral-clustering)
- [Gaussian Mixture Models](#gaussian-mixture-models)
- [MeanShift](#meanshift)
- [OPTICS](#optics)
- [Birch](#birch)
- [HDBSCAN](#hdbscan)

---

## Overview

Clustering groups data points into clusters without using labels. Algorithms differ in assumptions about cluster shape, need for pre-specified **n_clusters**, and scalability.

---

## KMeans and MiniBatchKMeans

**KMeans** partitions data into k clusters by minimizing within-cluster variance (inertia). Assumes spherical, similar-sized clusters.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_clusters** | Number of clusters k |
| **init** | `k-means++`, `random` |
| **max_iter** | Maximum iterations |
| **n_init** | Runs with different seeds; keeps best |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| **inertia_** | Sum of squared distances to nearest center |
| **labels_** | Cluster index per sample |
| **cluster_centers_** | Coordinates of cluster centers |

### Code Example

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(kmeans.inertia_)
```

### MiniBatchKMeans

**MiniBatchKMeans** uses random mini-batches for scalability. Slightly lower quality but much faster on large data.

```python
from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(n_clusters=4, batch_size=100)
mbk.fit(X)
```

---

## DBSCAN

**DBSCAN** (Density-Based Spatial Clustering) finds clusters of arbitrary shape. Does not require **n_clusters**; identifies noise as label **-1**.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **eps** | Maximum distance between neighbors |
| **min_samples** | Min points to form a core point |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| **labels_** | Cluster index; -1 = noise |
| **core_sample_indices_** | Indices of core points |

### Code Example

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X)
n_clusters = len(set(db.labels_) - {-1})
n_noise = (db.labels_ == -1).sum()
```

---

## Hierarchical Clustering

**AgglomerativeClustering** builds a hierarchy by merging nearest clusters. Can use **n_clusters** or **distance_threshold** for flat output.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_clusters** | Number of clusters (or None) |
| **linkage** | `ward`, `complete`, `average`, `single` |
| **distance_threshold** | Stop merging above this distance |

### Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, method='ward')
dendrogram(Z)
```

---

## Spectral Clustering

**SpectralClustering** uses the spectrum of a similarity matrix. Good for non-convex clusters (e.g., moons, circles).

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_clusters** | Number of clusters |
| **affinity** | `rbf`, `nearest_neighbors` |
| **assign_labels** | `kmeans`, `discretize` |

```python
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=2, affinity='rbf')
labels = sc.fit_predict(X)
```

---

## Gaussian Mixture Models

**GaussianMixture** models data as a mixture of Gaussians. Soft clustering via **predict_proba**; supports **BIC**/ **AIC** for model selection.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_components** | Number of mixture components |
| **covariance_type** | `full`, `tied`, `diag` |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| **means_** | Component means |
| **weights_** | Mixing coefficients |
| **converged_** | Whether EM converged |

### BIC and AIC

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type='full')
gmm.fit(X)
print(gmm.bic(X), gmm.aic(X))
```

### BayesianGaussianMixture

**BayesianGaussianMixture** infers the number of components via Dirichlet prior; unused components get near-zero weight.

---

## MeanShift

**MeanShift** finds modes by iteratively shifting points toward density maxima. No **n_clusters**; bandwidth controls scale.

```python
from sklearn.cluster import MeanShift

ms = MeanShift()
ms.fit(X)
n_clusters = ms.labels_.max() + 1
```

---

## OPTICS

**OPTICS** (Ordering Points To Identify Clustering Structure) extends DBSCAN with variable density. Produces **reachability_** ordering; extract clusters via **xi** or **min_cluster_size**.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **min_samples** | Core point definition |
| **xi** | Minimum steepness for cluster extraction |
| **min_cluster_size** | Minimum cluster size (fraction or int) |

```python
from sklearn.cluster import OPTICS

opt = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
opt.fit(X)
```

---

## Birch

**Birch** (Balanced Iterative Reducing and Clustering using Hierarchies) builds a CF-tree for scalable clustering. Good for large datasets.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_clusters** | Final number of clusters |
| **threshold** | Radius for subclusters |

```python
from sklearn.cluster import Birch

birch = Birch(n_clusters=4, threshold=0.5)
birch.fit(X)
```

---

## HDBSCAN

**HDBSCAN** (Hierarchical DBSCAN) finds clusters of varying density. Not in scikit-learn; use the **hdbscan** package.

```python
from hdbscan import HDBSCAN

hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
labels = hdb.fit_predict(X)
```

---

## Algorithm Comparison

| Algorithm | n_clusters | Cluster Shape | Scalability | Noise |
|-----------|------------|---------------|-------------|-------|
| KMeans | Required | Spherical | High | No |
| DBSCAN | No | Arbitrary | Medium | Yes |
| Hierarchical | Optional | Any | Low | No |
| Spectral | Required | Non-convex | Low | No |
| GMM | Required | Elliptical | Medium | No |
| MeanShift | No | Arbitrary | Low | No |
| OPTICS | No | Arbitrary | Medium | Yes |
| Birch | Optional | Spherical | High | No |
