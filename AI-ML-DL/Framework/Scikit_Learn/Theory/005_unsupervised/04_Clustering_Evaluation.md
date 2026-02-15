# Clustering Evaluation

---

## Table of Contents

- [Overview](#overview)
- [Internal Metrics](#internal-metrics)
- [External Metrics](#external-metrics)
- [Metric Details](#metric-details)
- [Choosing k and Comparing Algorithms](#choosing-k-and-comparing-algorithms)

---

## Overview

Clustering evaluation uses **internal** metrics (no ground truth) and **external** metrics (with ground truth). Internal metrics assess compactness and separation; external metrics compare to known labels.

---

## Internal Metrics

Internal metrics use only **X** and **labels**. No true labels required.

### silhouette_score

**Silhouette coefficient** measures how similar a point is to its own cluster vs others. Range [-1, 1]; higher is better.

| Value | Interpretation |
|-------|----------------|
| 1 | Well-matched to cluster |
| 0 | On boundary |
| -1 | Likely in wrong cluster |

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
```

### calinski_harabasz_score

**Calinski-Harabasz index** (variance ratio criterion): ratio of between-cluster to within-cluster dispersion. Higher is better. No upper bound.

```python
from sklearn.metrics import calinski_harabasz_score

ch = calinski_harabasz_score(X, labels)
```

### davies_bouldin_score

**Davies-Bouldin index** measures average similarity between each cluster and its most similar cluster. Lower is better. Minimum is 0.

```python
from sklearn.metrics import davies_bouldin_score

db = davies_bouldin_score(X, labels)
```

---

## External Metrics

External metrics compare **labels** to **y_true**. Require ground truth.

### adjusted_rand_score

**Adjusted Rand Index (ARI)** measures agreement between two labelings, corrected for chance. Range [-1, 1]; 1 = perfect match, 0 = random.

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, labels)
```

### Other External Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **adjusted_rand_score** | [-1, 1] | Pairwise agreement, chance-corrected |
| **normalized_mutual_info_score** | [0, 1] | Mutual information, normalized |
| **homogeneity_score** | [0, 1] | Clusters contain single class |
| **completeness_score** | [0, 1] | Class members in same cluster |
| **v_measure_score** | [0, 1] | Harmonic mean of homogeneity and completeness |

```python
from sklearn.metrics import (
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

nmi = normalized_mutual_info_score(y_true, labels)
h = homogeneity_score(y_true, labels)
c = completeness_score(y_true, labels)
v = v_measure_score(y_true, labels)
```

---

## Metric Details

### Silhouette per Sample

```python
from sklearn.metrics import silhouette_samples

samples = silhouette_samples(X, labels)
```

### When to Use Which

| Scenario | Recommended Metric |
|----------|--------------------|
| No ground truth, choose k | silhouette_score, davies_bouldin_score |
| Compare algorithms (no labels) | silhouette_score |
| Ground truth available | adjusted_rand_score, v_measure_score |
| Per-point analysis | silhouette_samples |

---

## Choosing k and Comparing Algorithms

### Elbow Method (KMeans)

Plot **inertia** vs k; look for elbow.

```python
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
```

### Silhouette Analysis

```python
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    print(f"k={k}: silhouette={sil:.4f}, davies_bouldin={db:.4f}")
```

### BIC/AIC (GMM)

For **GaussianMixture**, use BIC or AIC to select **n_components**.

```python
from sklearn.mixture import GaussianMixture

bics = []
for k in range(2, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    bics.append(gmm.bic(X))
best_k = np.argmin(bics) + 2
```

### Algorithm Comparison

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

algorithms = {
    "KMeans": KMeans(n_clusters=4, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=4),
}
for name, model in algorithms.items():
    if hasattr(model, "fit_predict"):
        labels = model.fit_predict(X)
    else:
        model.fit(X)
        labels = model.labels_
    sil = silhouette_score(X, labels)
    print(f"{name}: silhouette={sil:.4f}")
```

---

## Summary Table

| Metric | Type | Range | Better |
|--------|------|-------|--------|
| **silhouette_score** | Internal | [-1, 1] | Higher |
| **calinski_harabasz_score** | Internal | [0, inf) | Higher |
| **davies_bouldin_score** | Internal | [0, inf) | Lower |
| **adjusted_rand_score** | External | [-1, 1] | Higher |
| **normalized_mutual_info_score** | External | [0, 1] | Higher |
| **v_measure_score** | External | [0, 1] | Higher |
