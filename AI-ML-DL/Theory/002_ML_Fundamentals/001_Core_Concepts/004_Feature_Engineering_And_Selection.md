# Feature Engineering And Selection

## Table of Contents

1. [Introduction to Feature Engineering](#introduction-to-feature-engineering)
2. [Feature Extraction](#feature-extraction)
3. [Feature Transformation](#feature-transformation)
4. [Feature Selection Methods](#feature-selection-methods)
5. [Dimensionality Reduction](#dimensionality-reduction)
6. [Feature Importance and Interpretability](#feature-importance-and-interpretability)
7. [Domain-Specific Feature Engineering](#domain-specific-feature-engineering)
8. [Handling Categorical Features](#handling-categorical-features)
9. [Temporal and Sequential Features](#temporal-and-sequential-features)
10. [Key Takeaways](#key-takeaways)

## Introduction to Feature Engineering

Feature engineering is the process of creating, transforming, and selecting features that improve machine learning model performance. It is often considered one of the most important and time-consuming aspects of the machine learning pipeline.

### What are Features?

Features are measurable properties or characteristics of the data that algorithms use to make predictions. In mathematical terms, if we have an input $x \in \mathcal{X}$, features are functions $\phi_i: \mathcal{X} \rightarrow \mathbb{R}$ that map inputs to numerical values:

$$\phi(x) = [\phi_1(x), \phi_2(x), \ldots, \phi_d(x)]^T$$

The feature vector $\phi(x) \in \mathbb{R}^d$ represents the input in a $d$-dimensional feature space.

### Importance of Feature Engineering

Well-engineered features can:
- **Improve Model Performance**: Better features lead to better predictions
- **Reduce Data Requirements**: Good features allow simpler models to perform well
- **Increase Interpretability**: Meaningful features are easier to understand
- **Handle Domain Knowledge**: Incorporate expert knowledge into the model

### The Feature Engineering Pipeline

The typical feature engineering process involves:

1. **Feature Extraction**: Creating new features from raw data
2. **Feature Transformation**: Scaling, normalizing, or applying functions to features
3. **Feature Selection**: Choosing the most relevant features
4. **Feature Validation**: Ensuring features improve model performance

### Curse of Dimensionality

As the number of features increases, several problems arise:

- **Sparse Data**: High-dimensional spaces are mostly empty
- **Distance Metrics**: Distances become less meaningful in high dimensions
- **Overfitting**: More features increase model complexity and overfitting risk
- **Computational Cost**: More features require more computation

This motivates both feature selection and dimensionality reduction techniques.

## Feature Extraction

Feature extraction involves creating new features from existing data, often combining or transforming raw inputs to capture more informative patterns.

### Polynomial Features

Polynomial features capture interactions and non-linear relationships. For input $x = [x_1, x_2]$, degree-2 polynomial features include:

$$\phi(x) = [x_1, x_2, x_1^2, x_2^2, x_1 x_2]$$

This expands the feature space from 2 to 5 dimensions, allowing linear models to capture quadratic relationships.

**Use Cases**:
- Capturing non-linear relationships with linear models
- Modeling interactions between features
- Approximating complex functions

**Challenges**:
- Exponential growth in feature count: $O(d^k)$ for degree $k$ and $d$ original features
- Risk of overfitting with high degrees
- Computational cost increases rapidly

### Interaction Features

Interaction features capture relationships between variables. For features $x_i$ and $x_j$, interactions include:

- **Product**: $x_i \cdot x_2$
- **Ratio**: $x_i / x_j$ (when $x_j \neq 0$)
- **Difference**: $x_i - x_j$
- **Maximum/Minimum**: $\max(x_i, x_j)$, $\min(x_i, x_j)$

**Example**: In predicting house prices, an interaction between square footage and number of bedrooms might capture that larger houses benefit more from additional bedrooms.

### Binning and Discretization

Binning converts continuous features into categorical features by dividing the range into bins:

$$x_{\text{binned}} = \begin{cases}
1 & \text{if } x \in [a_1, a_2) \\
2 & \text{if } x \in [a_2, a_3) \\
\vdots \\
k & \text{if } x \in [a_k, a_{k+1}]
\end{cases}$$

**Methods**:
- **Equal-Width Binning**: Bins have equal size
- **Equal-Frequency Binning**: Each bin contains approximately equal number of samples
- **Quantile Binning**: Bins based on quantiles of the distribution

**Advantages**:
- Handles non-linear relationships
- Robust to outliers
- Can improve tree-based models

### Text Feature Extraction

For text data, common feature extraction methods include:

**Bag of Words**: Represent documents as vectors of word counts:
$$\phi_{\text{BoW}}(d) = [\text{count}(w_1, d), \text{count}(w_2, d), \ldots, \text{count}(w_n, d)]$$

**TF-IDF**: Term Frequency-Inverse Document Frequency weights words by importance:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

where $N$ is total documents and $\text{DF}(t)$ is document frequency of term $t$.

**N-grams**: Sequences of $n$ consecutive words capture phrase-level information.

### Image Feature Extraction

Traditional computer vision features include:

- **Histogram of Oriented Gradients (HOG)**: Captures edge and gradient information
- **Local Binary Patterns (LBP)**: Texture descriptors
- **Scale-Invariant Feature Transform (SIFT)**: Keypoint descriptors
- **Color Histograms**: Distribution of color values

Modern deep learning approaches learn features automatically through convolutional layers.

## Feature Transformation

Feature transformation modifies existing features to improve their utility for machine learning algorithms.

### Scaling and Normalization

Many algorithms are sensitive to feature scales. Common transformations include:

**Standardization (Z-score normalization)**:
$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation. Results in features with mean 0 and variance 1.

**Min-Max Scaling**:
$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Scales features to $[0, 1]$ range.

**Robust Scaling**:
$$x_{\text{robust}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$

Uses median and interquartile range, robust to outliers.

### Logarithmic Transformation

Logarithmic transformation is useful for:
- **Right-Skewed Distributions**: Makes distributions more symmetric
- **Multiplicative Relationships**: Converts $y = ax_1 x_2$ to $\log y = \log a + \log x_1 + \log x_2$
- **Heteroscedasticity**: Stabilizes variance

Common transformations:
- Natural log: $\log(x)$
- Log base 10: $\log_{10}(x)$
- Log(1+x): Handles zeros

### Power Transformations

Power transformations (Box-Cox family) can normalize distributions:

$$x^{(\lambda)} = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$

The parameter $\lambda$ is chosen to maximize normality. Special cases:
- $\lambda = 1$: No transformation
- $\lambda = 0.5$: Square root
- $\lambda = 0$: Logarithm
- $\lambda = -1$: Reciprocal

### Rank Transformation

Rank transformation converts values to their ranks, making the distribution uniform:

$$x_{\text{rank}} = \text{rank}(x)$$

**Advantages**:
- Robust to outliers
- Handles non-linear monotonic relationships
- Normalizes any distribution to uniform

**Disadvantages**:
- Loses information about magnitude differences
- Ties require special handling

## Feature Selection Methods

Feature selection identifies the most relevant features, reducing dimensionality and improving model performance.

### Filter Methods

Filter methods select features based on statistical properties, independent of the learning algorithm.

**Correlation-Based Selection**: Select features highly correlated with target but lowly correlated with each other.

**Mutual Information**: Measures dependence between features and target:
$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Features with high mutual information are selected.

**Chi-Square Test**: For categorical features, tests independence from target:
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ and $E_i$ are observed and expected frequencies.

**Variance Threshold**: Removes features with variance below a threshold (constant or near-constant features).

**Advantages**:
- Fast and scalable
- Independent of learning algorithm
- Good for initial feature screening

**Disadvantages**:
- Ignores feature interactions
- May miss features that are useful in combination

### Wrapper Methods

Wrapper methods use the learning algorithm itself to evaluate feature subsets.

**Forward Selection**: Start with empty set, iteratively add best feature:
1. Start with empty feature set $S = \emptyset$
2. For each feature $f \notin S$, evaluate $S \cup \{f\}$
3. Add feature that improves performance most
4. Repeat until no improvement

**Backward Elimination**: Start with all features, iteratively remove worst feature:
1. Start with all features $S = \mathcal{F}$
2. For each feature $f \in S$, evaluate $S \setminus \{f\}$
3. Remove feature that hurts performance least
4. Repeat until performance degrades significantly

**Recursive Feature Elimination (RFE)**: Uses model coefficients or feature importance to recursively remove features.

**Advantages**:
- Considers feature interactions
- Optimizes for specific learning algorithm
- Often finds better feature subsets

**Disadvantages**:
- Computationally expensive ($O(2^d)$ for exhaustive search)
- Risk of overfitting to validation set
- Requires careful cross-validation

### Embedded Methods

Embedded methods perform feature selection as part of the learning process.

**L1 Regularization (Lasso)**: Adds penalty proportional to sum of absolute coefficients:
$$\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|_1$$

The L1 penalty encourages sparsity, automatically performing feature selection.

**Elastic Net**: Combines L1 and L2 regularization:
$$\min_{\beta} \|y - X\beta\|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$$

**Tree-Based Feature Importance**: Decision trees and random forests provide feature importance scores based on:
- Information gain (entropy reduction)
- Gini impurity reduction
- Mean decrease in accuracy when feature is permuted

**Advantages**:
- More efficient than wrapper methods
- Considers feature interactions
- Integrated with learning process

**Disadvantages**:
- Tied to specific algorithms
- May require hyperparameter tuning

## Dimensionality Reduction

Dimensionality reduction reduces the number of features while preserving important information.

### Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance in the data. For data matrix $X \in \mathbb{R}^{n \times d}$, PCA finds:

$$\max_{w} \text{Var}(Xw) \quad \text{subject to} \quad \|w\| = 1$$

The solution is given by eigenvectors of the covariance matrix $C = \frac{1}{n}X^T X$ corresponding to largest eigenvalues.

**Properties**:
- Linear transformation
- Preserves maximum variance
- Components are uncorrelated
- Can be computed via SVD: $X = U \Sigma V^T$

**Limitations**:
- Assumes linear relationships
- Sensitive to feature scaling
- May not preserve local structure

### Independent Component Analysis (ICA)

ICA finds statistically independent components, useful for source separation:

$$X = AS$$

where $A$ is mixing matrix and $S$ contains independent sources.

**Applications**:
- Signal processing
- Feature extraction
- Blind source separation

### Non-Linear Dimensionality Reduction

**t-SNE**: t-Distributed Stochastic Neighbor Embedding preserves local neighborhoods:
- Computes similarity in high and low dimensions
- Uses t-distribution for low-dimensional similarities
- Excellent for visualization

**UMAP**: Uniform Manifold Approximation and Projection:
- Preserves both local and global structure
- More scalable than t-SNE
- Can be used for dimensionality reduction (not just visualization)

**Autoencoders**: Neural networks that learn compressed representations:
- Encoder: $h = f(x)$ maps input to latent representation
- Decoder: $\hat{x} = g(h)$ reconstructs input from representation
- Trained to minimize reconstruction error

## Feature Importance and Interpretability

Understanding which features contribute most to predictions is crucial for model interpretability.

### Permutation Importance

Permutation importance measures feature importance by:
1. Train model on original data
2. For each feature, permute its values
3. Measure decrease in model performance
4. Larger decrease indicates higher importance

**Advantages**:
- Model-agnostic
- Easy to interpret
- Accounts for feature interactions

### SHAP Values

SHAP (SHapley Additive exPlanations) values provide unified framework for feature attribution:

$$\phi_i(f, x) = \sum_{S \subseteq \mathcal{F} \setminus \{i\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} [f(S \cup \{i\}) - f(S)]$$

SHAP values satisfy:
- **Efficiency**: $\sum_i \phi_i = f(x) - f(\emptyset)$
- **Symmetry**: Equal features get equal values
- **Dummy**: Features with no effect get zero
- **Additivity**: For ensemble models

### Partial Dependence Plots

Partial dependence plots show the marginal effect of a feature on predictions:

$$\text{PD}_j(x_j) = \frac{1}{n} \sum_{i=1}^n f(x_j, x_{-j}^{(i)})$$

where $x_{-j}^{(i)}$ are other features from training data.

### Feature Interaction Analysis

Understanding feature interactions reveals how features work together:

- **H-statistic**: Measures interaction strength
- **Partial dependence**: Visualize interactions between pairs
- **Tree-based methods**: Naturally capture interactions

## Domain-Specific Feature Engineering

Different domains require specialized feature engineering techniques.

### Time Series Features

For temporal data:
- **Lag Features**: Previous values $x_{t-k}$
- **Rolling Statistics**: Mean, std, min, max over windows
- **Time-Based Features**: Hour of day, day of week, seasonality
- **Difference Features**: $x_t - x_{t-1}$
- **Fourier Transform**: Frequency domain features

### Geospatial Features

For location data:
- **Distance Features**: Distance to landmarks, city centers
- **Coordinate Features**: Latitude, longitude
- **Spatial Aggregations**: Average values in neighborhoods
- **Clustering**: Group nearby locations

### Image Features

Beyond raw pixels:
- **Color Statistics**: Mean, std of color channels
- **Texture Features**: Local Binary Patterns, Gabor filters
- **Shape Features**: Contours, moments
- **Deep Features**: Activations from pre-trained CNNs

### Graph Features

For network data:
- **Node Features**: Degree, centrality measures
- **Edge Features**: Weight, type
- **Subgraph Features**: Motif counts
- **Embeddings**: Node2Vec, Graph2Vec

## Handling Categorical Features

Categorical features require special treatment as most algorithms expect numerical inputs.

### One-Hot Encoding

Creates binary features for each category:

$$x_{\text{one-hot}} = [\mathbb{1}(x = c_1), \mathbb{1}(x = c_2), \ldots, \mathbb{1}(x = c_k)]$$

**Advantages**:
- No ordinality assumption
- Works with any algorithm

**Disadvantages**:
- High dimensionality for many categories
- Sparse representation
- Can cause overfitting

### Label Encoding

Maps categories to integers: $\{red, blue, green\} \rightarrow \{0, 1, 2\}$

**Use Cases**:
- Tree-based algorithms (can handle ordinality)
- When categories have natural ordering

**Limitations**:
- Introduces artificial ordering
- Not suitable for linear models

### Target Encoding

Encodes categories by their target statistics:

$$\text{encode}(c) = \mathbb{E}[y | x = c]$$

**Variants**:
- **Mean Encoding**: Average target per category
- **Bayesian Encoding**: Shrinkage toward global mean
- **Leave-One-Out**: Exclude current example when computing mean

**Advantages**:
- Captures target relationship
- Low dimensionality

**Disadvantages**:
- Risk of overfitting
- Requires careful cross-validation

### Embedding Methods

Learn dense representations:
- **Word Embeddings**: For text categories
- **Entity Embeddings**: For categorical variables
- **Autoencoders**: Learn compressed representations

## Temporal and Sequential Features

Time-dependent data requires features that capture temporal patterns.

### Lag Features

Previous values of the target or features:
- $y_{t-1}, y_{t-2}, \ldots, y_{t-k}$: Target lags
- $x_{t-1}, x_{t-2}, \ldots, x_{t-k}$: Feature lags

### Rolling Window Features

Statistics over sliding windows:
- **Rolling Mean**: $\bar{x}_t = \frac{1}{w}\sum_{i=t-w+1}^t x_i$
- **Rolling Std**: Standard deviation over window
- **Rolling Min/Max**: Extreme values

### Time-Based Features

Extract temporal patterns:
- **Cyclical Encoding**: $\sin(2\pi t/T)$, $\cos(2\pi t/T)$ for period $T$
- **Time Since**: Time since last event
- **Seasonality**: Day of week, month, quarter

### Sequence Features

For sequences:
- **N-grams**: Subsequences of length $n$
- **Skip-grams**: Gaps in sequences
- **Positional Encoding**: Position in sequence
- **Sequence Statistics**: Length, diversity, entropy

## Key Takeaways

1. **Feature Engineering** is crucial for model performance, involving extraction, transformation, and selection of informative features.

2. **Feature Extraction** creates new features through polynomial transformations, interactions, binning, and domain-specific methods like text/image processing.

3. **Feature Transformation** includes scaling (standardization, min-max), logarithmic/power transformations, and rank transformations to improve feature utility.

4. **Filter Methods** select features based on statistical properties (correlation, mutual information) independently of the learning algorithm.

5. **Wrapper Methods** use the learning algorithm to evaluate feature subsets (forward selection, backward elimination) but are computationally expensive.

6. **Embedded Methods** perform feature selection during learning (L1 regularization, tree importance) and are more efficient than wrappers.

7. **Dimensionality Reduction** (PCA, ICA, t-SNE, UMAP, autoencoders) reduces feature count while preserving important information.

8. **Feature Importance** can be measured through permutation importance, SHAP values, and partial dependence plots for model interpretability.

9. **Domain-Specific Features** are essential: time series (lags, rolling stats), geospatial (distances, aggregations), images (texture, deep features), graphs (centrality, embeddings).

10. **Categorical Features** require encoding (one-hot, label, target encoding) or embedding methods, with choice depending on algorithm and data characteristics.
