# Clustering Algorithms KMeans

## Table of Contents

1. [Introduction to Clustering](#introduction-to-clustering)
2. [K-Means Algorithm](#k-means-algorithm)
3. [K-Medoids and PAM](#k-medoids-and-pam)
4. [Hierarchical Clustering](#hierarchical-clustering)
5. [DBSCAN](#dbscan)
6. [Gaussian Mixture Models](#gaussian-mixture-models)
7. [Cluster Validation](#cluster-validation)
8. [Choosing Number of Clusters](#choosing-number-of-clusters)
9. [Comparison of Clustering Methods](#comparison-of-clustering-methods)
10. [Key Takeaways](#key-takeaways)

## Introduction to Clustering

Clustering is an unsupervised learning task that groups similar data points together without labeled examples.

### What is Clustering?

Clustering partitions data into groups (clusters) such that:
- **Intra-cluster Similarity**: Points within the same cluster are similar
- **Inter-cluster Dissimilarity**: Points in different clusters are dissimilar

### Objectives

- **Discover Hidden Patterns**: Find natural groupings in data
- **Data Compression**: Represent data by cluster centers
- **Anomaly Detection**: Identify outliers as points far from clusters
- **Preprocessing**: Reduce data complexity for downstream tasks

### Types of Clustering

**Partitioning Methods**: Divide data into $k$ non-overlapping clusters
- K-means, K-medoids

**Hierarchical Methods**: Create tree of clusters (dendrogram)
- Agglomerative, Divisive

**Density-Based Methods**: Find clusters as dense regions
- DBSCAN, OPTICS

**Model-Based Methods**: Assume data generated from mixture of distributions
- Gaussian Mixture Models, Expectation-Maximization

### Clustering Challenges

- **No Ground Truth**: Cannot directly evaluate quality
- **Number of Clusters**: Often unknown a priori
- **Cluster Shape**: Assumptions about cluster shape (spherical, arbitrary)
- **Noise and Outliers**: Handling noisy data
- **Scalability**: Efficient algorithms for large datasets

## K-Means Algorithm

K-means is the most popular partitioning clustering algorithm, aiming to minimize within-cluster sum of squares.

### Problem Formulation

Given data $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$ and number of clusters $k$, find:
- Cluster centers $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k$
- Cluster assignments $c_i \in \{1, \ldots, k\}$

that minimize:

$$J = \sum_{i=1}^n \sum_{j=1}^k \mathbb{1}(c_i = j) \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$

where $\mathbb{1}(\cdot)$ is the indicator function.

### Algorithm

**Initialize**: Randomly choose $k$ cluster centers $\boldsymbol{\mu}_1^{(0)}, \ldots, \boldsymbol{\mu}_k^{(0)}$

**Repeat until convergence**:
1. **Assignment Step**: Assign each point to nearest center
   $$c_i^{(t)} = \arg\min_{j} \|\mathbf{x}_i - \boldsymbol{\mu}_j^{(t)}\|^2$$

2. **Update Step**: Update centers to mean of assigned points
   $$\boldsymbol{\mu}_j^{(t+1)} = \frac{1}{|\mathcal{C}_j|} \sum_{i \in \mathcal{C}_j} \mathbf{x}_i$$
   where $\mathcal{C}_j = \{i : c_i^{(t)} = j\}$ is cluster $j$

**Convergence**: When assignments don't change or objective $J$ stops decreasing.

### Properties

**Convergence**: Algorithm converges to local minimum (not necessarily global)

**Complexity**: 
- Per iteration: $O(nkd)$ where $d$ is dimensionality
- Number of iterations: Typically small (10-50)

**Optimality**: Assignment step minimizes $J$ given centers, update step minimizes $J$ given assignments.

### Initialization Methods

**Random Initialization**: 
- Simple but can lead to poor local minima
- Run multiple times, choose best result

**K-means++**: 
- Choose first center randomly
- Choose subsequent centers with probability proportional to distance squared from nearest existing center
- Better initialization, often finds better solutions

**Forgy Method**: 
- Randomly choose $k$ data points as initial centers

### Limitations

- **Assumes Spherical Clusters**: Works best for spherical, similar-sized clusters
- **Sensitive to Initialization**: Different initializations yield different results
- **Requires $k$**: Number of clusters must be specified
- **Sensitive to Outliers**: Outliers can significantly affect cluster centers
- **Local Optima**: May converge to poor local minimum

### Variants

**Fuzzy K-means (C-means)**: Points belong to clusters with membership weights

**K-medians**: Uses median instead of mean (more robust to outliers)

**Mini-batch K-means**: Uses random subsets for faster computation on large datasets

## K-Medoids and PAM

K-medoids uses actual data points (medoids) as cluster centers instead of means.

### K-Medoids

**Medoid**: The most centrally located point in a cluster (minimizes sum of distances to other points)

**Advantages over K-means**:
- More robust to outliers
- Works with any distance metric (not just Euclidean)
- Medoids are actual data points (interpretable)

**Disadvantages**:
- More computationally expensive
- Slower convergence

### PAM Algorithm (Partitioning Around Medoids)

**Algorithm**:
1. Initialize: Randomly select $k$ medoids
2. Assign each point to nearest medoid
3. For each medoid $m$ and non-medoid $o$:
   - Swap $m$ and $o$
   - Compute total cost change $\Delta$
   - If $\Delta < 0$, keep swap; otherwise revert
4. Repeat until no improvement

**Cost Function**:
$$C = \sum_{i=1}^n \min_{j \in M} d(\mathbf{x}_i, \mathbf{m}_j)$$

where $M$ is set of medoids and $d$ is distance metric.

### CLARA (Clustering Large Applications)

Scales PAM to large datasets:
- Sample data randomly
- Apply PAM to sample
- Assign all points to medoids from sample

**Tradeoff**: Speed vs. quality (may miss clusters in unsampled regions)

## Hierarchical Clustering

Hierarchical clustering creates a tree of clusters (dendrogram) showing relationships at all scales.

### Types

**Agglomerative (Bottom-Up)**:
- Start with each point as its own cluster
- Merge closest clusters iteratively
- Continue until one cluster remains

**Divisive (Top-Down)**:
- Start with all points in one cluster
- Split clusters iteratively
- Continue until each point is its own cluster

### Agglomerative Algorithm

**Algorithm**:
1. Initialize: Each point is a cluster
2. Compute distance matrix between all clusters
3. Merge two closest clusters
4. Update distance matrix
5. Repeat steps 3-4 until one cluster remains

**Complexity**: $O(n^3)$ naive, $O(n^2 \log n)$ with efficient data structures

### Linkage Criteria

**Single Linkage (Nearest Neighbor)**:
$$d(C_i, C_j) = \min_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

- Tends to create elongated clusters (chaining effect)
- Sensitive to noise

**Complete Linkage (Farthest Neighbor)**:
$$d(C_i, C_j) = \max_{\mathbf{x} \in C_i, \mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

- Creates compact clusters
- Sensitive to outliers

**Average Linkage**:
$$d(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{\mathbf{x} \in C_i} \sum_{\mathbf{y} \in C_j} d(\mathbf{x}, \mathbf{y})$$

- Balance between single and complete
- Less sensitive to outliers

**Ward's Method**:
Minimizes increase in within-cluster variance:
$$d(C_i, C_j) = \frac{|C_i||C_j|}{|C_i| + |C_j|} \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|^2$$

- Tends to create spherical clusters
- Similar to K-means objective

### Dendrogram

Tree diagram showing cluster merges:
- **Leaves**: Individual data points
- **Branches**: Cluster merges
- **Height**: Distance at which clusters merge
- **Cut**: Horizontal line determines number of clusters

### Advantages

- No need to specify $k$ (can choose from dendrogram)
- Visual representation of cluster structure
- Deterministic (given linkage criterion)

### Disadvantages

- Computationally expensive ($O(n^2)$ or worse)
- Sensitive to noise and outliers
- Once merge is made, cannot undo

## DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) finds clusters as dense regions separated by sparse regions.

### Key Concepts

**$\epsilon$-neighborhood**: Points within distance $\epsilon$ of a point

**Core Point**: Point with at least $minPts$ points in its $\epsilon$-neighborhood

**Border Point**: Point in $\epsilon$-neighborhood of core point but not core itself

**Noise Point**: Point that is neither core nor border

### Algorithm

1. Mark all points as unvisited
2. For each unvisited point $p$:
   - Mark $p$ as visited
   - If $p$ has $< minPts$ neighbors: mark as noise
   - Else: create new cluster $C$, add $p$ to $C$
     - For each point $p'$ in $\epsilon$-neighborhood of $p$:
       - If $p'$ is unvisited: mark as visited, add to $C$
       - If $p'$ is noise: change to border point, add to $C$
       - If $p'$ is core: add all points in $p'$'s neighborhood to $C$

### Parameters

**$\epsilon$**: Maximum distance for neighborhood
- Too small: Many small clusters, many noise points
- Too large: Few large clusters, few noise points

**$minPts$**: Minimum points to form core
- Typically $minPts = 2d$ where $d$ is dimensionality
- At least 3-4 for 2D data

### Advantages

- Finds clusters of arbitrary shape
- Handles noise and outliers naturally
- No need to specify number of clusters
- Robust to initialization

### Disadvantages

- Sensitive to parameters ($\epsilon$, $minPts$)
- Struggles with varying densities
- May merge close clusters
- Computationally expensive ($O(n^2)$ naive, $O(n \log n)$ with spatial indexing)

### Variants

**OPTICS**: Ordering points to identify clustering structure
- Addresses varying density problem
- Creates reachability plot instead of fixed $\epsilon$

**HDBSCAN**: Hierarchical DBSCAN
- Combines hierarchical and density-based clustering
- More robust to parameter selection

## Gaussian Mixture Models

Gaussian Mixture Models (GMM) assume data is generated from a mixture of Gaussian distributions.

### Model

Data is generated from $k$ Gaussian components:

$$p(\mathbf{x}) = \sum_{j=1}^k \pi_j \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_j, \Sigma_j)$$

where:
- $\pi_j$: Mixing coefficient (prior probability of component $j$)
- $\boldsymbol{\mu}_j$: Mean of component $j$
- $\Sigma_j$: Covariance matrix of component $j$

Constraints: $\sum_{j=1}^k \pi_j = 1$, $\pi_j \geq 0$

### Soft Assignment

Unlike K-means (hard assignment), GMM uses soft assignment:

$$\gamma_{ij} = P(z_i = j | \mathbf{x}_i) = \frac{\pi_j \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_j, \Sigma_j)}{\sum_{l=1}^k \pi_l \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_l, \Sigma_l)}$$

where $z_i$ is latent variable indicating component.

### Expectation-Maximization (EM) Algorithm

**E-step**: Compute responsibilities $\gamma_{ij}$

**M-step**: Update parameters:
$$\boldsymbol{\mu}_j = \frac{\sum_{i=1}^n \gamma_{ij} \mathbf{x}_i}{\sum_{i=1}^n \gamma_{ij}}$$

$$\Sigma_j = \frac{\sum_{i=1}^n \gamma_{ij} (\mathbf{x}_i - \boldsymbol{\mu}_j)(\mathbf{x}_i - \boldsymbol{\mu}_j)^T}{\sum_{i=1}^n \gamma_{ij}}$$

$$\pi_j = \frac{1}{n}\sum_{i=1}^n \gamma_{ij}$$

### Covariance Constraints

**Full Covariance**: $\Sigma_j$ unconstrained (flexible but many parameters)

**Diagonal Covariance**: $\Sigma_j$ diagonal (features independent within cluster)

**Spherical Covariance**: $\Sigma_j = \sigma_j^2 I$ (similar to K-means)

**Tied Covariance**: $\Sigma_j = \Sigma$ (all clusters share covariance)

### Advantages

- Probabilistic framework
- Soft assignments (handles uncertainty)
- Can model ellipsoidal clusters
- Handles overlapping clusters

### Disadvantages

- Assumes Gaussian distribution
- More parameters to estimate
- EM may converge to local optimum
- Slower than K-means

## Cluster Validation

Cluster validation assesses the quality of clustering results without ground truth.

### Internal Validation

Uses only the data and clustering results.

**Within-Cluster Sum of Squares (WCSS)**:
$$WCSS = \sum_{j=1}^k \sum_{i \in C_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$

Lower is better (but decreases with more clusters).

**Between-Cluster Sum of Squares (BCSS)**:
$$BCSS = \sum_{j=1}^k |C_j| \|\boldsymbol{\mu}_j - \bar{\mathbf{x}}\|^2$$

Higher is better.

**Silhouette Score**: Measures how similar a point is to its own cluster vs. other clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:
- $a(i)$: Average distance to points in same cluster
- $b(i)$: Average distance to points in nearest other cluster

Range: $[-1, 1]$, higher is better.

**Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances:

$$DB = \frac{1}{k}\sum_{j=1}^k \max_{l \neq j} \frac{\sigma_j + \sigma_l}{d(\boldsymbol{\mu}_j, \boldsymbol{\mu}_l)}$$

Lower is better.

### External Validation

Requires ground truth labels.

**Rand Index**: Fraction of pairs correctly clustered:

$$RI = \frac{TP + TN}{TP + TN + FP + FN}$$

where $TP$: pairs in same cluster and same class, etc.

**Adjusted Rand Index**: Corrected for chance:

$$ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$

Range: $[-1, 1]$, higher is better.

**Normalized Mutual Information**: Measures shared information:

$$NMI = \frac{I(C; K)}{\sqrt{H(C)H(K)}}$$

where $C$ are clusters, $K$ are classes.

## Choosing Number of Clusters

Selecting the optimal number of clusters $k$ is a fundamental challenge.

### Elbow Method

Plot WCSS vs. $k$:
- Look for "elbow" where decrease slows
- Subjective but commonly used

### Silhouette Analysis

Plot average silhouette score vs. $k$:
- Choose $k$ with highest silhouette score
- More objective than elbow method

### Gap Statistic

Compares WCSS to expected WCSS under null distribution:

$$\text{Gap}(k) = E[\log WCSS_k^*] - \log WCSS_k$$

Choose $k$ that maximizes gap statistic.

### Information Criteria

**Akaike Information Criterion (AIC)**:
$$AIC = 2p - 2\ln(L)$$

**Bayesian Information Criterion (BIC)**:
$$BIC = p\ln(n) - 2\ln(L)$$

where $p$ is number of parameters, $L$ is likelihood.

Lower is better.

### Cross-Validation

Use stability-based methods:
- Cluster on subsets of data
- Measure consistency of cluster assignments
- Choose $k$ with highest stability

## Comparison of Clustering Methods

| Method | Cluster Shape | Noise Handling | Scalability | Requires $k$ |
|--------|---------------|----------------|-------------|--------------|
| K-means | Spherical | Poor | Good | Yes |
| K-medoids | Spherical | Good | Moderate | Yes |
| Hierarchical | Arbitrary | Moderate | Poor | No |
| DBSCAN | Arbitrary | Excellent | Moderate | No |
| GMM | Ellipsoidal | Moderate | Moderate | Yes |

### When to Use

**K-means**: Spherical clusters, known $k$, fast computation needed

**K-medoids**: Need robustness to outliers, interpretable centers

**Hierarchical**: Unknown $k$, need dendrogram visualization

**DBSCAN**: Arbitrary shapes, noise present, unknown $k$

**GMM**: Probabilistic framework needed, overlapping clusters, soft assignments

## Key Takeaways

1. **Clustering** partitions data into groups based on similarity, with intra-cluster similarity and inter-cluster dissimilarity as key objectives.

2. **K-means** minimizes within-cluster sum of squares through alternating assignment and update steps, assuming spherical clusters and requiring specification of $k$.

3. **K-medoids** uses actual data points as centers (medoids), making it more robust to outliers and applicable to any distance metric, though computationally more expensive.

4. **Hierarchical Clustering** creates a dendrogram through agglomerative (bottom-up) or divisive (top-down) methods, with linkage criteria (single, complete, average, Ward's) determining merge strategy.

5. **DBSCAN** finds density-based clusters of arbitrary shape, naturally handling noise through core/border/noise point classification based on $\epsilon$ and $minPts$ parameters.

6. **Gaussian Mixture Models** assume data from mixture of Gaussians, using EM algorithm for soft assignments and allowing ellipsoidal clusters through covariance matrices.

7. **Cluster Validation** uses internal metrics (silhouette score, Davies-Bouldin) or external metrics (Rand Index, NMI) when ground truth is available.

8. **Choosing $k$** involves elbow method, silhouette analysis, gap statistic, information criteria, or cross-validation-based stability measures.

9. **Method Selection** depends on cluster shape assumptions, noise handling needs, scalability requirements, and whether $k$ is known, with K-means for speed, DBSCAN for arbitrary shapes, and GMM for probabilistic framework.

10. **Clustering Challenges** include no ground truth for evaluation, unknown number of clusters, assumptions about cluster shape, handling noise/outliers, and scalability for large datasets.
