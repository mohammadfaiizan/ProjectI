# Anomaly Detection Methods

## Table of Contents

1. [Introduction to Anomaly Detection](#introduction-to-anomaly-detection)
2. [Statistical Methods](#statistical-methods)
3. [Isolation Forest](#isolation-forest)
4. [One-Class SVM](#one-class-svm)
5. [Autoencoders for Anomaly Detection](#autoencoders-for-anomaly-detection)
6. [Local Outlier Factor](#local-outlier-factor)
7. [Density-Based Methods](#density-based-methods)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Challenges and Considerations](#challenges-and-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Anomaly Detection

Anomaly detection identifies rare items, events, or observations that deviate significantly from the majority of data.

### What are Anomalies?

**Anomaly (Outlier)**: Data point that differs substantially from other observations.

**Types**:
- **Point Anomalies**: Individual anomalous instances
- **Contextual Anomalies**: Anomalous in specific context (e.g., time, location)
- **Collective Anomalies**: Collection of related instances anomalous together

### Applications

- **Fraud Detection**: Credit card fraud, insurance claims
- **Network Security**: Intrusion detection, malware
- **Healthcare**: Disease detection, medical errors
- **Manufacturing**: Defect detection, quality control
- **Finance**: Market manipulation, unusual trading

### Challenges

- **Unbalanced Data**: Anomalies are rare (often <1% of data)
- **No Labels**: Often no labeled anomalies for training
- **Definition**: What constitutes an anomaly is domain-specific
- **Evolving**: Normal behavior changes over time
- **Evaluation**: Difficult to evaluate without ground truth

### Approaches

- **Statistical**: Assume normal data follows distribution
- **Distance-Based**: Measure distance to normal instances
- **Density-Based**: Identify low-density regions
- **Isolation-Based**: Isolate anomalies using random partitions
- **Reconstruction-Based**: Model normal data, flag poor reconstructions

## Statistical Methods

Statistical methods assume normal data follows a known distribution.

### Z-Score Method

For univariate data, flag points with large z-scores:

$$z_i = \frac{x_i - \mu}{\sigma}$$

**Threshold**: $|z_i| > 3$ (or other threshold)

**Assumptions**: 
- Data is normally distributed
- Mean $\mu$ and std $\sigma$ known or estimated

**Limitations**: 
- Sensitive to outliers (mean/std affected)
- Assumes Gaussian distribution

### Modified Z-Score

Uses median and MAD (Median Absolute Deviation) for robustness:

$$M_i = \frac{0.6745(x_i - \tilde{x})}{\text{MAD}}$$

where $\tilde{x}$ is median and $\text{MAD} = \text{median}(|x_i - \tilde{x}|)$.

**Advantages**: Robust to outliers

### Interquartile Range (IQR) Method

Flag points outside:

$$[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$$

where $\text{IQR} = Q_3 - Q_1$ (interquartile range).

**Advantages**: Non-parametric, robust

### Grubbs' Test

Statistical test for detecting a single outlier:

$$G = \frac{\max|x_i - \bar{x}|}{s}$$

Critical value depends on sample size and significance level.

**Limitations**: Assumes normal distribution, tests one outlier at a time

### Multivariate Methods

**Mahalanobis Distance**:

$$d_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

where $\boldsymbol{\mu}$ is mean vector and $\Sigma$ is covariance matrix.

Flag points with $d_M(\mathbf{x}) > \tau$ (e.g., $\chi^2$ threshold).

**Advantages**: Accounts for correlations between features

**Limitations**: Sensitive to outliers (affects $\boldsymbol{\mu}$ and $\Sigma$)

### Robust Multivariate Methods

**Minimum Covariance Determinant (MCD)**: 
- Find subset of data minimizing covariance determinant
- More robust estimates of $\boldsymbol{\mu}$ and $\Sigma$

**Isolation**: Less affected by outliers

## Isolation Forest

Isolation Forest isolates anomalies using random tree partitions.

### Intuition

**Key Insight**: Anomalies are easier to isolate (require fewer partitions) than normal points.

**Example**: In a random partition, an anomaly (far from others) is likely isolated quickly, while normal points (clustered together) require more partitions.

### Algorithm

**Training**:
1. Build $t$ isolation trees
2. For each tree:
   - Randomly sample subset of data
   - Randomly select feature and split value
   - Recursively partition until:
     - Tree reaches max depth, or
     - Only one point remains, or
     - All points have same value
3. Average path lengths across trees

**Anomaly Score**:
$$s(\mathbf{x}) = 2^{-\frac{E[h(\mathbf{x})]}{c(n)}}$$

where:
- $E[h(\mathbf{x})]$: Average path length across trees
- $c(n)$: Normalization constant (average path length for $n$ points)

**Interpretation**:
- $s(\mathbf{x}) \approx 1$: Anomaly (short path length)
- $s(\mathbf{x}) \approx 0$: Normal (long path length)

### Properties

- **Linear Complexity**: $O(n)$ per tree, $O(t \cdot n)$ total
- **Subsampling**: Each tree uses subset, reducing computation
- **No Distance Metric**: Works with any data type
- **Handles High Dimensions**: Effective even in high-dimensional spaces

### Advantages

- Fast (linear time complexity)
- Handles high-dimensional data well
- No assumptions about data distribution
- Works with mixed data types

### Limitations

- May struggle with clustered anomalies
- Performance depends on contamination rate
- Less interpretable than some methods

## One-Class SVM

One-Class SVM learns a decision boundary around normal data.

### Problem Formulation

Find hyperplane that separates normal data from origin with maximum margin:

$$\min_{\mathbf{w}, \rho, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 - \rho + \frac{1}{\nu n}\sum_{i=1}^n \xi_i$$

subject to:
$$\mathbf{w}^T \phi(\mathbf{x}_i) \geq \rho - \xi_i, \quad \xi_i \geq 0$$

where:
- $\mathbf{w}$: Weight vector
- $\rho$: Offset (distance from origin)
- $\boldsymbol{\xi}$: Slack variables
- $\nu$: Upper bound on fraction of outliers

### Decision Function

$$f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \phi(\mathbf{x}) - \rho)$$

- $f(\mathbf{x}) = +1$: Normal (inside boundary)
- $f(\mathbf{x}) = -1$: Anomaly (outside boundary)

### Kernel Trick

Use kernel function for nonlinear boundaries:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i K(\mathbf{x}_i, \mathbf{x}) - \rho\right)$$

Common kernels: RBF, polynomial

### Parameter $\nu$

- **Upper bound** on fraction of outliers
- **Lower bound** on fraction of support vectors
- Typical range: $[0.01, 0.5]$

### Advantages

- Handles nonlinear boundaries via kernels
- Probabilistic interpretation possible
- Memory efficient (uses only support vectors)

### Limitations

- Sensitive to kernel and parameter selection
- May struggle with high-dimensional data
- Computationally expensive for large datasets

## Autoencoders for Anomaly Detection

Autoencoders learn to reconstruct normal data; anomalies have high reconstruction error.

### Architecture

**Encoder**: $h = f(\mathbf{x})$ maps input to latent representation

**Decoder**: $\hat{\mathbf{x}} = g(h)$ reconstructs input from representation

**Objective**: Minimize reconstruction error on normal data:
$$\min \sum_{i=1}^n \|\mathbf{x}_i - g(f(\mathbf{x}_i))\|^2$$

### Anomaly Detection

**Assumption**: Autoencoder learns to reconstruct normal data well but fails on anomalies.

**Anomaly Score**: Reconstruction error
$$s(\mathbf{x}) = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \|\mathbf{x} - g(f(\mathbf{x}))\|^2$$

**Threshold**: Flag points with $s(\mathbf{x}) > \tau$

### Variants

**Variational Autoencoders (VAE)**: 
- Probabilistic framework
- Anomaly score: Negative log-likelihood or reconstruction error

**Denoising Autoencoders**:
- Trained to reconstruct clean data from noisy input
- More robust to noise

**Sparse Autoencoders**:
- Encourages sparse representations
- May better capture normal patterns

### Advantages

- Learns complex, nonlinear patterns
- No assumptions about data distribution
- Can handle high-dimensional data
- End-to-end learning

### Limitations

- Requires training data (preferably only normal data)
- May overfit to training distribution
- Computationally expensive
- Black box (less interpretable)

## Local Outlier Factor

LOF measures local density deviation to identify outliers.

### Intuition

**Key Idea**: Compare local density of a point to densities of its neighbors.

**Anomaly**: Point with significantly lower local density than its neighbors.

### Definitions

**$k$-Distance**: Distance to $k$-th nearest neighbor

**Reachability Distance**:
$$rd_k(p, o) = \max(k\text{-distance}(o), d(p, o))$$

**Local Reachability Density**:
$$lrd_k(p) = \frac{1}{\frac{1}{k}\sum_{o \in N_k(p)} rd_k(p, o)}$$

where $N_k(p)$ are $k$ nearest neighbors.

**Local Outlier Factor**:
$$LOF_k(p) = \frac{\frac{1}{k}\sum_{o \in N_k(p)} lrd_k(o)}{lrd_k(p)}$$

### Interpretation

- $LOF \approx 1$: Similar density to neighbors (normal)
- $LOF > 1$: Lower density than neighbors (outlier)
- $LOF < 1$: Higher density than neighbors (inlier)

**Threshold**: Typically $LOF > 1.5$ or $2.0$ indicates anomaly

### Advantages

- Handles varying densities
- Relative measure (compares to neighbors)
- No assumptions about data distribution
- Works with arbitrary distance metrics

### Limitations

- Sensitive to parameter $k$
- Computationally expensive ($O(n^2)$ for distance computation)
- May struggle with high-dimensional data

## Density-Based Methods

Identify anomalies as points in low-density regions.

### DBSCAN for Anomaly Detection

DBSCAN naturally identifies noise points:
- **Core Points**: High local density
- **Border Points**: Near core points
- **Noise Points**: Low density (anomalies)

**Anomaly Score**: Binary (noise or not) or distance to nearest core point

### Kernel Density Estimation

Estimate density using KDE:

$$\hat{p}(\mathbf{x}) = \frac{1}{nh}\sum_{i=1}^n K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$

**Anomaly Score**: Low density $\hat{p}(\mathbf{x}) < \tau$

**Advantages**: Probabilistic, smooth density estimate

**Limitations**: Curse of dimensionality, bandwidth selection

### Gaussian Mixture Models

Model normal data as GMM:

$$p(\mathbf{x}) = \sum_{j=1}^k \pi_j \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_j, \Sigma_j)$$

**Anomaly Score**: Low probability $p(\mathbf{x}) < \tau$

**Advantages**: Probabilistic, handles multi-modal data

**Limitations**: Assumes Gaussian components, requires number of components

## Evaluation Metrics

Evaluating anomaly detection is challenging due to class imbalance.

### Confusion Matrix

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actual Normal** | TN | FP |
| **Actual Anomaly** | FN | TP |

### Metrics

**Precision**: $\frac{TP}{TP + FP}$ (of predicted anomalies, how many are true)

**Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ (of true anomalies, how many detected)

**F1-Score**: $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**Specificity**: $\frac{TN}{TN + FP}$ (of normal points, how many correctly identified)

### ROC Curve and AUC

Plot TPR (Recall) vs. FPR ($1 - \text{Specificity}$) for different thresholds.

**AUC**: Area under ROC curve
- $AUC = 1$: Perfect classifier
- $AUC = 0.5$: Random classifier

### Precision-Recall Curve

Plot Precision vs. Recall for different thresholds.

**AUC-PR**: Area under PR curve
- Better for imbalanced data than ROC-AUC
- Focuses on positive class (anomalies)

### Challenges

- **No Ground Truth**: Often no labeled anomalies
- **Class Imbalance**: Very few anomalies
- **Cost-Sensitive**: False positives and false negatives have different costs
- **Temporal**: Anomalies may appear over time

## Challenges and Considerations

### Class Imbalance

Anomalies are rare (<1% typical):
- Standard metrics (accuracy) misleading
- Use precision, recall, F1, AUC-PR
- Consider cost-sensitive evaluation

### No Labels

Often no labeled anomalies:
- Unsupervised methods required
- Evaluation difficult
- May need domain experts to validate

### Definition Ambiguity

What constitutes anomaly is domain-specific:
- Statistical outlier may not be interesting
- Context matters (e.g., time, location)
- Requires domain knowledge

### Concept Drift

Normal behavior changes over time:
- Need adaptive methods
- Online/streaming algorithms
- Periodic retraining

### High Dimensionality

Curse of dimensionality affects distance-based methods:
- Distances become less meaningful
- Need dimensionality reduction or specialized methods
- Isolation Forest handles this well

### Interpretability

Understanding why point is anomalous:
- Statistical methods: z-score, distance
- Isolation Forest: Path length
- LOF: Density comparison
- Autoencoders: Reconstruction error (less interpretable)

## Key Takeaways

1. **Anomaly Detection** identifies rare, deviant instances using statistical, distance-based, density-based, isolation-based, or reconstruction-based methods.

2. **Statistical Methods** (z-score, IQR, Mahalanobis distance) assume distributions, flagging points beyond thresholds, with robust variants (modified z-score, MCD) handling outliers better.

3. **Isolation Forest** isolates anomalies using random partitions, with anomaly score $s(\mathbf{x}) = 2^{-E[h(\mathbf{x})]/c(n)}$ based on average path length, effective in high dimensions with linear complexity.

4. **One-Class SVM** learns decision boundary around normal data via kernel trick, with parameter $\nu$ controlling outlier fraction, suitable for nonlinear boundaries but sensitive to hyperparameters.

5. **Autoencoders** learn to reconstruct normal data, using reconstruction error $\|\mathbf{x} - g(f(\mathbf{x}))\|^2$ as anomaly score, learning complex patterns but requiring training data and being less interpretable.

6. **Local Outlier Factor** compares local density to neighbors: $LOF_k(p) = \frac{\frac{1}{k}\sum_{o \in N_k(p)} lrd_k(o)}{lrd_k(p)}$, handling varying densities but sensitive to parameter $k$.

7. **Density-Based Methods** (DBSCAN noise points, KDE low density, GMM low probability) identify low-density regions, with probabilistic interpretations but suffering from curse of dimensionality.

8. **Evaluation Metrics** must handle class imbalance: use precision, recall, F1-score, AUC-PR (better than ROC-AUC for imbalanced data), with cost-sensitive evaluation when false positives/negatives have different costs.

9. **Challenges** include class imbalance (rare anomalies), no labels (unsupervised needed), definition ambiguity (domain-specific), concept drift (adaptive methods), high dimensionality (specialized methods), and interpretability needs.

10. **Method Selection** depends on data characteristics (dimensionality, distribution), availability of labels, interpretability needs, computational constraints, and domain requirements, with Isolation Forest for speed/high-dim, One-Class SVM for nonlinear boundaries, Autoencoders for complex patterns, and LOF for varying densities.
