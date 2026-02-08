# Instance Based Learning KNN

## Table of Contents

1. [Introduction to Instance-Based Learning](#introduction-to-instance-based-learning)
2. [K-Nearest Neighbors Algorithm](#k-nearest-neighbors-algorithm)
3. [Distance Metrics](#distance-metrics)
4. [Curse of Dimensionality](#curse-of-dimensionality)
5. [KD-Trees and Spatial Data Structures](#kd-trees-and-spatial-data-structures)
6. [Approximate Nearest Neighbor Search](#approximate-nearest-neighbor-search)
7. [Weighted KNN](#weighted-knn)
8. [KNN Variants and Extensions](#knn-variants-and-extensions)
9. [Practical Considerations](#practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Instance-Based Learning

Instance-based learning, also known as lazy learning or memory-based learning, stores training examples and makes predictions based on similarity to stored instances rather than learning an explicit model.

### What is Instance-Based Learning?

Unlike eager learning algorithms (e.g., decision trees, neural networks) that build a model during training, instance-based learning:

- **Defers Processing**: No explicit model is built during training
- **Stores Examples**: Keeps all training data in memory
- **Lazy Evaluation**: Computations occur at prediction time
- **Local Approximation**: Predictions based on nearby training examples

### Characteristics

**Advantages**:
- Adapts to new training data without retraining
- Can model complex decision boundaries
- No assumptions about data distribution
- Simple to understand and implement

**Disadvantages**:
- Slow prediction (must search through all examples)
- Memory intensive (stores all training data)
- Sensitive to irrelevant features
- No explicit model for interpretation

### Types of Instance-Based Learning

- **K-Nearest Neighbors (KNN)**: Uses $k$ closest examples
- **Radial Basis Functions**: Weighted by distance
- **Case-Based Reasoning**: Uses domain-specific similarity
- **Locally Weighted Regression**: Fits local models

## K-Nearest Neighbors Algorithm

K-Nearest Neighbors is the most popular instance-based learning algorithm, making predictions based on the $k$ closest training examples.

### Algorithm Description

**Training Phase**:
1. Store all training examples $\mathcal{D} = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$
2. No model is built

**Prediction Phase**:
1. Given query point $\mathbf{x}_q$
2. Find $k$ nearest neighbors: $N_k(\mathbf{x}_q) = \{\mathbf{x}_{i_1}, \mathbf{x}_{i_2}, \ldots, \mathbf{x}_{i_k}\}$
3. For classification: majority vote of labels
4. For regression: average of target values

### Classification

For classification with $k$ neighbors:

$$\hat{y}_q = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_k(\mathbf{x}_q)} \mathbb{1}(y_i = c)$$

where $\mathcal{C}$ is the set of classes and $\mathbb{1}(\cdot)$ is the indicator function.

**Tie-Breaking**: When classes are tied, common strategies:
- Use $k-1$ neighbors (reduce $k$ by 1)
- Prefer class with closest neighbor
- Random selection

### Regression

For regression:

$$\hat{y}_q = \frac{1}{k} \sum_{i \in N_k(\mathbf{x}_q)} y_i$$

This is the mean of the $k$ nearest neighbors' target values.

### Choosing K

The choice of $k$ significantly affects performance:

**Small $k$ (e.g., $k=1$)**:
- Low bias (can fit training data perfectly)
- High variance (sensitive to noise)
- Complex decision boundaries
- Risk of overfitting

**Large $k$ (e.g., $k=n$)**:
- High bias (always predicts majority class)
- Low variance (stable predictions)
- Smooth decision boundaries
- Risk of underfitting

**Optimal $k$**: Typically chosen via cross-validation, often $k = \sqrt{n}$ as a rule of thumb.

### Decision Boundaries

KNN creates piecewise linear decision boundaries that adapt to the data:

- **$k=1$**: Voronoi diagram (each point gets its own region)
- **Larger $k$**: Smoother boundaries, averaging over neighbors
- **Non-linear**: Can model complex, non-linear boundaries

## Distance Metrics

The choice of distance metric determines which examples are considered "nearest."

### Euclidean Distance

Most common distance metric:

$$d_E(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{l=1}^d (x_{il} - x_{jl})^2} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$$

**Properties**:
- Rotation invariant
- Sensitive to feature scales
- Assumes features are equally important

**Use Cases**: When features are on similar scales and relationships are isotropic.

### Manhattan Distance (L1)

Sum of absolute differences:

$$d_M(\mathbf{x}_i, \mathbf{x}_j) = \sum_{l=1}^d |x_{il} - x_{jl}| = \|\mathbf{x}_i - \mathbf{x}_j\|_1$$

**Properties**:
- Less sensitive to outliers than Euclidean
- Prefers axis-aligned paths
- Useful for high-dimensional sparse data

**Use Cases**: When features are independent or when outliers are problematic.

### Minkowski Distance

Generalization of Euclidean and Manhattan:

$$d_p(\mathbf{x}_i, \mathbf{x}_j) = \left(\sum_{l=1}^d |x_{il} - x_{jl}|^p\right)^{1/p}$$

**Special Cases**:
- $p=1$: Manhattan distance
- $p=2$: Euclidean distance
- $p \to \infty$: Chebyshev distance ($\max_l |x_{il} - x_{jl}|$)

### Cosine Similarity

Measures angle between vectors:

$$\cos(\theta) = \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|} = \frac{\sum_{l=1}^d x_{il} x_{jl}}{\sqrt{\sum_{l=1}^d x_{il}^2} \sqrt{\sum_{l=1}^d x_{jl}^2}}$$

**Distance**: $d_{\cos}(\mathbf{x}_i, \mathbf{x}_j) = 1 - \cos(\theta)$

**Properties**:
- Scale invariant (normalized)
- Measures direction, not magnitude
- Useful for text data (TF-IDF vectors)

**Use Cases**: When magnitude is less important than direction (e.g., document similarity).

### Hamming Distance

For categorical/binary data:

$$d_H(\mathbf{x}_i, \mathbf{x}_j) = \sum_{l=1}^d \mathbb{1}(x_{il} \neq x_{jl})$$

Counts the number of positions where values differ.

**Use Cases**: Binary features, categorical data, error-correcting codes.

### Mahalanobis Distance

Accounts for feature correlations:

$$d_{Mah}(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{(\mathbf{x}_i - \mathbf{x}_j)^T \Sigma^{-1} (\mathbf{x}_i - \mathbf{x}_j)}$$

where $\Sigma$ is the covariance matrix.

**Properties**:
- Scale invariant
- Accounts for feature correlations
- More appropriate when features are correlated

**Use Cases**: When features have different variances and are correlated.

### Weighted Distances

Weight features by importance:

$$d_w(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{l=1}^d w_l (x_{il} - x_{jl})^2}$$

where $w_l$ are feature weights, often learned during training.

## Curse of Dimensionality

The curse of dimensionality refers to phenomena that occur in high-dimensional spaces that make distance-based methods like KNN less effective.

### Volume Concentration

As dimensionality increases, the volume of a hypercube concentrates in its corners:

- Most points are far from the center
- Distances between points become similar
- Nearest and farthest neighbors become equidistant

### Distance Concentration

In high dimensions, distances become less discriminative:

$$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

All pairwise distances converge to the same value, making nearest neighbor search meaningless.

### Empty Space Phenomenon

High-dimensional spaces are mostly empty:

- Data occupies a tiny fraction of the space
- Need exponentially more data to fill the space
- Local neighborhoods become non-local

### Implications for KNN

**Problems**:
- Nearest neighbors may not be "near" in a meaningful sense
- All points become approximately equidistant
- Performance degrades with dimensionality
- Need more data to maintain performance

**Solutions**:
- Feature selection (reduce dimensionality)
- Dimensionality reduction (PCA, feature engineering)
- Weighted distances (emphasize relevant features)
- Use domain knowledge to select relevant features

### Sample Complexity

To maintain performance in $d$ dimensions, need $n \sim O(2^d)$ samples (exponential growth).

This motivates dimensionality reduction and feature selection.

## KD-Trees and Spatial Data Structures

Efficient data structures can speed up nearest neighbor search from $O(nd)$ to $O(\log n)$ average case.

### KD-Tree Construction

A KD-tree (k-dimensional tree) recursively partitions space:

**Algorithm**:
1. Choose dimension with largest variance (or cycle through dimensions)
2. Find median along that dimension
3. Split data at median
4. Recursively build left and right subtrees

**Properties**:
- Balanced binary tree
- Each node represents a hyperrectangle
- Leaf nodes contain data points

### KD-Tree Search

**Nearest Neighbor Search**:
1. Traverse tree to find leaf containing query point
2. Check distance to points in leaf
3. Backtrack, checking other branches if necessary
4. Maintain best distance found so far

**Pruning**: Skip branches whose hyperrectangle is farther than current best distance.

**Average Case**: $O(\log n)$ for balanced tree
**Worst Case**: $O(n)$ when all points are needed

### Limitations of KD-Trees

**High Dimensions**:
- Performance degrades in high dimensions ($d > 20$)
- Curse of dimensionality affects tree structure
- May degenerate to linear search

**Dynamic Data**:
- Insertions/deletions can unbalance tree
- May require rebuilding

**Alternative**: Use approximate methods or other data structures (ball trees, locality-sensitive hashing).

### Ball Trees

Similar to KD-trees but use hyperspheres (balls) instead of hyperrectangles:

- Each node represents a ball containing its points
- Can be more efficient in high dimensions
- Better for non-axis-aligned data

## Approximate Nearest Neighbor Search

For large-scale applications, approximate methods trade accuracy for speed.

### Locality-Sensitive Hashing (LSH)

LSH hashes similar points to the same bucket with high probability:

**Algorithm**:
1. Create multiple hash tables with different hash functions
2. Hash query point to buckets
3. Search points in those buckets
4. Return nearest among candidates

**Hash Functions**: 
- For Euclidean distance: random projections
- For cosine similarity: random hyperplanes
- For Hamming distance: random bit sampling

**Properties**:
- Sub-linear query time: $O(dn^\rho)$ where $\rho < 1$
- Probabilistic guarantees
- Tradeoff between accuracy and speed

### Product Quantization

Compress vectors into compact codes:

1. Split vector into subvectors
2. Quantize each subvector separately
3. Represent vector as tuple of quantizer indices
4. Use lookup tables for fast distance computation

**Use Cases**: Large-scale image retrieval, recommendation systems.

### Annoy (Approximate Nearest Neighbors Oh Yeah)

Uses random projection trees:

- Builds multiple random projection trees
- Searches all trees in parallel
- Combines results

**Properties**: Fast, memory-efficient, used in production systems.

## Weighted KNN

Weight neighbors by distance, giving closer neighbors more influence.

### Distance-Weighted KNN

Weight each neighbor inversely proportional to distance:

$$w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i)^p}$$

where $p$ is a parameter (typically $p=2$).

**Classification**:
$$\hat{y}_q = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_k(\mathbf{x}_q)} w_i \mathbb{1}(y_i = c)$$

**Regression**:
$$\hat{y}_q = \frac{\sum_{i \in N_k(\mathbf{x}_q)} w_i y_i}{\sum_{i \in N_k(\mathbf{x}_q)} w_i}$$

### Kernel Functions

Use kernel functions for weighting:

**Gaussian Kernel**:
$$w_i = \exp\left(-\frac{d(\mathbf{x}_q, \mathbf{x}_i)^2}{2\sigma^2}\right)$$

**Epanechnikov Kernel**:
$$w_i = \max(0, 1 - d(\mathbf{x}_q, \mathbf{x}_i)^2)$$

**Triangular Kernel**:
$$w_i = \max(0, 1 - |d(\mathbf{x}_q, \mathbf{x}_i)|)$$

### Advantages

- Smoother predictions
- Less sensitive to choice of $k$
- Can use all neighbors (not just $k$)
- Reduces impact of distant neighbors

## KNN Variants and Extensions

Various extensions address limitations of standard KNN.

### Edited Nearest Neighbor

Remove noisy examples from training set:

1. For each training example, find $k$ nearest neighbors
2. If predicted label differs from true label, remove example
3. Repeat until convergence

**Effect**: Reduces storage, improves accuracy, removes outliers.

### Condensed Nearest Neighbor

Find minimal subset that correctly classifies all training examples:

1. Start with empty set
2. Add examples that are misclassified by current set
3. Repeat until all examples are correctly classified

**Effect**: Dramatically reduces storage while maintaining accuracy.

### Learning Vector Quantization (LVQ)

Learn prototype vectors (codebook vectors) instead of using all training examples:

1. Initialize prototype vectors
2. For each training example:
   - Find nearest prototype
   - If same class: move prototype toward example
   - If different class: move prototype away
3. Repeat until convergence

**Advantages**: Much faster prediction, lower memory.

### Local Linear Embedding (LLE)

For dimensionality reduction, KNN is used to define local neighborhoods, then learns low-dimensional representation preserving local structure.

### KNN for Time Series

Adapt KNN for sequential data:
- Use dynamic time warping (DTW) as distance metric
- Consider subsequences as neighbors
- Account for temporal dependencies

## Practical Considerations

### Preprocessing

**Normalization**: Critical for KNN since it's distance-based:
- Standardize features to zero mean, unit variance
- Or use min-max scaling to [0,1]
- Prevents features with large scales from dominating

**Feature Selection**: Remove irrelevant features:
- Reduces dimensionality (mitigates curse of dimensionality)
- Improves accuracy
- Speeds up computation

### Computational Efficiency

**Brute Force**: $O(nd)$ per query (slow for large $n$)

**Optimizations**:
- KD-trees: $O(\log n)$ average case
- Ball trees: Better for high dimensions
- LSH: Sub-linear for approximate search
- Parallelization: Search can be parallelized

### Handling Missing Values

**Strategies**:
- Impute missing values before computing distances
- Use distance metrics that handle missing values
- Weight features by completeness

### Choosing Parameters

**$k$**: Use cross-validation to select optimal $k$

**Distance Metric**: 
- Euclidean: Default for continuous features
- Manhattan: More robust to outliers
- Cosine: For normalized/sparse data
- Domain-specific: Use domain knowledge

**Weights**: Learn feature weights or use equal weights

### Memory Considerations

KNN stores all training data:
- Memory: $O(nd)$ for $n$ examples with $d$ features
- For large datasets, consider:
  - Condensed/edited nearest neighbor
  - Prototype methods (LVQ)
  - Approximate methods

## Key Takeaways

1. **Instance-Based Learning** defers processing to prediction time, storing all training examples and making predictions based on similarity rather than explicit models.

2. **K-Nearest Neighbors** predicts by finding $k$ closest training examples, using majority vote (classification) or average (regression), with $k$ chosen via cross-validation.

3. **Distance Metrics** (Euclidean, Manhattan, Minkowski, cosine, Mahalanobis) determine neighbor selection, with choice depending on data characteristics and feature scales.

4. **Curse of Dimensionality** causes distances to become less discriminative in high dimensions, requiring feature selection, dimensionality reduction, or weighted distances.

5. **KD-Trees** provide efficient nearest neighbor search ($O(\log n)$ average) by recursively partitioning space, though performance degrades in high dimensions.

6. **Approximate Methods** (LSH, product quantization, Annoy) trade accuracy for speed, enabling scalable KNN for large datasets with sub-linear query time.

7. **Weighted KNN** assigns higher weights to closer neighbors using distance-based or kernel-based weighting, producing smoother predictions and reducing sensitivity to $k$.

8. **KNN Variants** include edited/condensed nearest neighbor (reduce storage), LVQ (learn prototypes), and extensions for time series and dimensionality reduction.

9. **Preprocessing** is critical: normalization ensures features contribute equally, feature selection reduces dimensionality, and missing value handling maintains distance validity.

10. **KNN** excels when data has local structure, interpretability matters, and adaptation to new data is needed, but requires careful handling of high dimensions, computational efficiency, and memory constraints.
