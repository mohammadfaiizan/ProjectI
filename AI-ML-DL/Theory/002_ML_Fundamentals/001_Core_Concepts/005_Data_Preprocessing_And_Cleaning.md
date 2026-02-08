# Data Preprocessing And Cleaning

## Table of Contents

1. [Introduction to Data Preprocessing](#introduction-to-data-preprocessing)
2. [Missing Data Handling](#missing-data-handling)
3. [Outlier Detection and Treatment](#outlier-detection-and-treatment)
4. [Normalization and Standardization](#normalization-and-standardization)
5. [Encoding Categorical Variables](#encoding-categorical-variables)
6. [Data Splitting Strategies](#data-splitting-strategies)
7. [Handling Imbalanced Data](#handling-imbalanced-data)
8. [Data Quality Assessment](#data-quality-assessment)
9. [Data Validation and Integrity](#data-validation-and-integrity)
10. [Key Takeaways](#key-takeaways)

## Introduction to Data Preprocessing

Data preprocessing is a critical step in the machine learning pipeline that transforms raw data into a format suitable for analysis and modeling. Real-world data is often incomplete, inconsistent, noisy, and contains errors that can significantly degrade model performance.

### The Importance of Data Preprocessing

The quality of data directly impacts model performance. As the saying goes: "Garbage in, garbage out." Well-preprocessed data can:
- Improve model accuracy and generalization
- Reduce training time and computational costs
- Enhance model interpretability
- Handle real-world data imperfections

### Common Data Issues

Real-world datasets commonly suffer from:

- **Missing Values**: Incomplete records with null or undefined values
- **Outliers**: Extreme values that deviate significantly from the norm
- **Inconsistencies**: Formatting differences, typos, duplicate entries
- **Scale Differences**: Features measured on vastly different scales
- **Categorical Encoding**: Non-numeric data requiring conversion
- **Imbalanced Classes**: Unequal representation of target classes
- **Noise**: Random errors or irrelevant information

### Preprocessing Pipeline

A typical preprocessing pipeline follows these steps:

1. **Data Collection**: Gather data from various sources
2. **Data Cleaning**: Handle missing values, outliers, inconsistencies
3. **Data Transformation**: Scale, normalize, encode variables
4. **Feature Engineering**: Create new features from existing ones
5. **Data Splitting**: Divide into training, validation, and test sets
6. **Data Validation**: Verify data quality and integrity

## Missing Data Handling

Missing data is one of the most common problems in real-world datasets and requires careful handling to avoid introducing bias.

### Types of Missingness

Understanding why data is missing is crucial for appropriate handling:

**Missing Completely at Random (MCAR)**: The probability of missingness is independent of both observed and unobserved data:
$$P(\text{missing} | X_{\text{obs}}, X_{\text{miss}}) = P(\text{missing})$$

**Missing at Random (MAR)**: The probability of missingness depends only on observed data:
$$P(\text{missing} | X_{\text{obs}}, X_{\text{miss}}) = P(\text{missing} | X_{\text{obs}})$$

**Missing Not at Random (MNAR)**: The probability of missingness depends on unobserved values:
$$P(\text{missing} | X_{\text{obs}}, X_{\text{miss}}) \neq P(\text{missing} | X_{\text{obs}})$$

### Deletion Methods

**Listwise Deletion (Complete Case Analysis)**: Remove rows with any missing values.

**Advantages**:
- Simple and fast
- No assumptions about missingness mechanism
- Preserves original data distribution (if MCAR)

**Disadvantages**:
- Loss of information
- Reduced sample size
- Potential bias if not MCAR
- May remove important cases

**Pairwise Deletion**: Use available data for each analysis, keeping different subsets for different variables.

**Use Cases**: When missingness patterns vary across variables and sample size is large.

### Imputation Methods

**Mean/Median/Mode Imputation**: Replace missing values with central tendency measures:
- **Mean**: For continuous variables: $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$
- **Median**: Robust to outliers: $\text{median}(x)$
- **Mode**: For categorical variables: most frequent value

**Advantages**: Simple, preserves sample size

**Disadvantages**: Underestimates variance, ignores relationships between variables

**Forward Fill / Backward Fill**: For time series, propagate last known value forward or next known value backward.

**Regression Imputation**: Predict missing values using other variables:
$$\hat{x}_i = f(x_{-i})$$

where $x_{-i}$ are other features. Can use linear regression, random forests, or other models.

**K-Nearest Neighbors Imputation**: Use values from $k$ most similar instances:
$$\hat{x}_i = \frac{1}{k} \sum_{j \in N_k(i)} x_j$$

where $N_k(i)$ are $k$ nearest neighbors.

**Multiple Imputation**: Create multiple imputed datasets, analyze each, and combine results:
1. Generate $m$ imputed datasets
2. Analyze each dataset
3. Combine results using Rubin's rules

**Advantages**: Accounts for uncertainty in imputation, provides valid statistical inference

**Disadvantages**: More complex, computationally intensive

### Advanced Imputation Methods

**Expectation-Maximization (EM) Algorithm**: Iteratively estimates parameters and imputes missing values:
1. **E-step**: Estimate missing values given current parameters
2. **M-step**: Update parameters given imputed values
3. Repeat until convergence

**MICE (Multiple Imputation by Chained Equations)**: Iteratively imputes each variable using others:
- Models each variable with missing values conditional on others
- Iterates until convergence
- Generates multiple imputed datasets

**Deep Learning Imputation**: Use autoencoders or generative models to learn data distribution and impute missing values.

### Handling Missing Data in Practice

**Guidelines**:
- Understand missingness mechanism (MCAR, MAR, MNAR)
- Explore patterns of missingness
- Consider domain knowledge
- Compare multiple imputation strategies
- Validate imputation quality
- Document imputation decisions

## Outlier Detection and Treatment

Outliers are observations that deviate significantly from the majority of the data and can adversely affect model performance.

### Causes of Outliers

- **Measurement Errors**: Instrument malfunctions, data entry mistakes
- **Sampling Errors**: Including observations from different populations
- **Natural Variation**: Legitimate extreme values
- **Data Processing Errors**: Incorrect transformations or aggregations

### Statistical Methods for Outlier Detection

**Z-Score Method**: Flag observations with $|z| > 3$:
$$z_i = \frac{x_i - \mu}{\sigma}$$

where $\mu$ and $\sigma$ are mean and standard deviation.

**Modified Z-Score**: Uses median and MAD (Median Absolute Deviation) for robustness:
$$M_i = \frac{0.6745(x_i - \tilde{x})}{\text{MAD}}$$

where $\tilde{x}$ is the median and $\text{MAD} = \text{median}(|x_i - \tilde{x}|)$.

**IQR Method**: Identify outliers beyond 1.5 IQR from quartiles:
- Lower bound: $Q_1 - 1.5 \times \text{IQR}$
- Upper bound: $Q_3 + 1.5 \times \text{IQR}$
- IQR = $Q_3 - Q_1$

**Grubbs' Test**: Statistical test for detecting a single outlier:
$$G = \frac{\max|x_i - \bar{x}|}{s}$$

Critical values depend on sample size and significance level.

### Machine Learning Methods

**Isolation Forest**: Randomly partitions data, outliers require fewer partitions to isolate:
- Builds random trees
- Measures average path length
- Shorter paths indicate outliers

**Local Outlier Factor (LOF)**: Measures local density deviation:
$$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$

where $\text{lrd}_k$ is local reachability density and $N_k(p)$ are $k$ nearest neighbors.

**One-Class SVM**: Learns a decision boundary around normal data, flags points outside as outliers.

**DBSCAN**: Density-based clustering that identifies outliers as noise points not belonging to any cluster.

### Multivariate Outlier Detection

**Mahalanobis Distance**: Measures distance accounting for covariance:
$$d_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1}(x - \mu)}$$

where $\mu$ is mean vector and $\Sigma$ is covariance matrix.

**Principal Component Analysis**: Project to lower dimensions and detect outliers in principal component space.

**Elliptic Envelope**: Fits an ellipse around data, points outside are outliers.

### Outlier Treatment Strategies

**Removal**: Delete outliers if they are errors or irrelevant.

**Capping/Winsorization**: Replace extreme values with threshold values:
$$x_{\text{capped}} = \begin{cases}
\text{lower\_bound} & \text{if } x < \text{lower\_bound} \\
x & \text{if } \text{lower\_bound} \leq x \leq \text{upper\_bound} \\
\text{upper\_bound} & \text{if } x > \text{upper\_bound}
\end{cases}$$

**Transformation**: Apply log, square root, or other transformations to reduce impact.

**Separate Modeling**: Model outliers separately if they represent a distinct population.

**Robust Methods**: Use algorithms robust to outliers (e.g., median instead of mean, robust regression).

## Normalization and Standardization

Different features often have different scales, which can bias algorithms sensitive to feature magnitudes.

### Standardization (Z-Score Normalization)

Transforms features to have zero mean and unit variance:

$$z_i = \frac{x_i - \mu}{\sigma}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation.

**Properties**:
- Mean = 0, Variance = 1
- Preserves shape of distribution
- Sensitive to outliers

**Use Cases**:
- Algorithms assuming features are on similar scales (SVM, neural networks, k-means)
- Principal Component Analysis
- When feature distributions are approximately normal

### Min-Max Normalization

Scales features to a fixed range, typically [0, 1]:

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Properties**:
- Range: [0, 1]
- Preserves relationships between values
- Sensitive to outliers (min/max values)

**Use Cases**:
- Neural networks (especially with sigmoid/tanh activations)
- Image pixel values
- When bounded range is required

### Robust Scaling

Uses median and interquartile range, robust to outliers:

$$x_{\text{robust}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$

where IQR = $Q_3 - Q_1$.

**Advantages**:
- Not affected by outliers
- Preserves median at zero
- IQR-based scaling

**Use Cases**:
- Data with outliers
- When median is more representative than mean

### Unit Vector Scaling (L2 Normalization)

Scales each sample to unit norm:

$$x_{\text{norm}} = \frac{x}{\|x\|_2}$$

where $\|x\|_2 = \sqrt{\sum_i x_i^2}$.

**Use Cases**:
- Text classification (TF-IDF vectors)
- Cosine similarity calculations
- When direction matters more than magnitude

### When to Normalize

**Algorithms Requiring Normalization**:
- Gradient descent-based methods (neural networks, logistic regression)
- Distance-based algorithms (k-NN, k-means, SVM with RBF kernel)
- Regularized models (Ridge, Lasso)
- Principal Component Analysis

**Algorithms Not Requiring Normalization**:
- Tree-based methods (decision trees, random forests)
- Naive Bayes (for count-based features)
- Algorithms using information gain or Gini impurity

## Encoding Categorical Variables

Most machine learning algorithms require numerical inputs, necessitating encoding of categorical variables.

### One-Hot Encoding

Creates binary features for each category:

For categories $\{A, B, C\}$:
- Category A: $[1, 0, 0]$
- Category B: $[0, 1, 0]$
- Category C: $[0, 0, 1]$

**Advantages**:
- No ordinality assumption
- Works with any algorithm
- Preserves category independence

**Disadvantages**:
- High dimensionality for many categories
- Sparse representation
- Can cause multicollinearity (one category can be inferred from others)

**Solution**: Use $k-1$ features for $k$ categories (drop one to avoid multicollinearity).

### Label Encoding

Maps categories to integers: $\{red, blue, green\} \rightarrow \{0, 1, 2\}$

**Use Cases**:
- Tree-based algorithms (can handle ordinality)
- When categories have natural ordering
- Ordinal variables (e.g., small, medium, large)

**Limitations**:
- Introduces artificial ordering for nominal variables
- Not suitable for linear models (assumes ordering)

### Ordinal Encoding

Similar to label encoding but preserves natural ordering:
- Small: 1
- Medium: 2
- Large: 3

### Target Encoding (Mean Encoding)

Encodes categories by their target statistics:

$$\text{encode}(c) = \mathbb{E}[y | x = c]$$

**Variants**:
- **Simple Mean**: Average target per category
- **Bayesian Encoding**: Shrinkage toward global mean:
  $$\text{encode}(c) = \frac{n_c \bar{y}_c + \alpha \bar{y}}{n_c + \alpha}$$
- **Leave-One-Out**: Exclude current example when computing mean
- **K-Fold Target Encoding**: Use out-of-fold means to prevent overfitting

**Advantages**:
- Captures target relationship
- Low dimensionality
- Can improve model performance

**Disadvantages**:
- Risk of overfitting (data leakage)
- Requires careful cross-validation
- Sensitive to small category sizes

### Binary Encoding

Represents categories as binary code, then splits into separate binary features:
- More compact than one-hot for many categories
- Reduces dimensionality compared to one-hot

### Hash Encoding

Uses hash function to map categories to fixed number of features:
- Constant dimensionality regardless of categories
- Handles new categories (out-of-vocabulary)
- May have collisions

## Data Splitting Strategies

Proper data splitting is crucial for unbiased model evaluation and selection.

### Simple Train-Test Split

Randomly divides data into training and test sets:
- Typical split: 70-80% training, 20-30% test
- Simple and fast
- May not be representative for small datasets

### Train-Validation-Test Split

Three-way split:
- **Training Set**: Used to learn model parameters
- **Validation Set**: Used to tune hyperparameters and select models
- **Test Set**: Used for final, unbiased performance assessment

Typical splits: 60% train, 20% validation, 20% test

**Critical Rule**: Test set should never be used during model development.

### Stratified Splitting

Maintains class distribution across splits:
- Important for imbalanced datasets
- Ensures each split has similar class proportions
- Prevents one split from missing a class entirely

### Time Series Splitting

For temporal data, maintain temporal order:
- **Forward Chaining**: Train on past, test on future
- **Rolling Window**: Fixed training window, sliding test window
- **Expanding Window**: Growing training set, fixed test window

**Example**: For time series data, never shuffle randomly as this breaks temporal dependencies.

### Cross-Validation

**K-Fold Cross-Validation**: Divides data into $k$ folds:
1. Train on $k-1$ folds
2. Validate on remaining fold
3. Repeat $k$ times
4. Average results

**Advantages**:
- Uses all data for training and validation
- Reduces variance in performance estimates
- More robust for small datasets

**Stratified K-Fold**: Maintains class distribution in each fold.

**Leave-One-Out Cross-Validation (LOOCV)**: $k = n$, trains on $n-1$ samples, validates on one.

**Nested Cross-Validation**: Outer loop for model evaluation, inner loop for hyperparameter tuning.

### Group-Based Splitting

When data has groups (e.g., patients, subjects):
- Keep all samples from same group in same split
- Prevents data leakage
- Important for clustered or hierarchical data

## Handling Imbalanced Data

Class imbalance occurs when target classes are unequally represented, causing models to favor majority classes.

### Resampling Methods

**Oversampling**: Increase minority class samples:
- **Random Oversampling**: Duplicate minority samples
- **SMOTE**: Synthetic Minority Oversampling Technique creates synthetic samples
- **ADASYN**: Adaptive Synthetic Sampling adjusts based on difficulty

**Undersampling**: Reduce majority class samples:
- **Random Undersampling**: Remove random majority samples
- **Tomek Links**: Remove borderline majority samples
- **Edited Nearest Neighbors**: Remove misclassified majority samples

**Combined Methods**: Combine oversampling and undersampling (e.g., SMOTE + Tomek Links).

### Algorithm-Level Methods

**Class Weights**: Assign higher weights to minority class:
$$\text{weight}_i = \frac{n_{\text{total}}}{n_{\text{classes}} \times n_{\text{class}_i}}$$

**Cost-Sensitive Learning**: Modify loss function to penalize minority class misclassification more.

**Threshold Tuning**: Adjust classification threshold (default 0.5) to optimize for precision, recall, or F1-score.

### Evaluation Metrics

For imbalanced data, accuracy is misleading. Use:
- **Precision-Recall Curve**: Better than ROC for imbalanced data
- **F1-Score**: Harmonic mean of precision and recall
- **Area Under PR Curve (AUC-PR)**: Summarizes precision-recall performance
- **Confusion Matrix**: Shows class-specific performance

## Data Quality Assessment

Assessing data quality helps identify issues before modeling.

### Completeness

Measure proportion of non-missing values:
$$\text{Completeness} = \frac{\text{non-missing values}}{\text{total values}}$$

### Consistency

Check for:
- Format consistency (dates, phone numbers)
- Value consistency (same entity, different representations)
- Referential integrity (foreign key constraints)

### Accuracy

Validate against:
- Domain knowledge
- External sources
- Business rules

### Timeliness

Assess data freshness and relevance for the task.

### Validity

Check if values conform to expected:
- Data types
- Ranges
- Formats
- Constraints

## Data Validation and Integrity

### Data Type Validation

Ensure correct data types:
- Numeric: integers, floats
- Categorical: strings, enums
- Temporal: dates, timestamps
- Boolean: true/false

### Range Validation

Check values are within expected ranges:
- Age: 0-150
- Percentages: 0-100
- Ratings: 1-5

### Format Validation

Verify formats match specifications:
- Email addresses
- Phone numbers
- Postal codes
- URLs

### Business Rule Validation

Enforce domain-specific rules:
- Start date < End date
- Sum of parts = Total
- Mutually exclusive categories

### Duplicate Detection

Identify and handle duplicates:
- Exact duplicates
- Near duplicates (fuzzy matching)
- Entity resolution (same entity, different records)

## Key Takeaways

1. **Data Preprocessing** is essential for transforming raw, imperfect data into a format suitable for machine learning algorithms.

2. **Missing Data** requires understanding the missingness mechanism (MCAR, MAR, MNAR) and choosing appropriate imputation methods (mean, regression, KNN, multiple imputation).

3. **Outlier Detection** uses statistical methods (Z-score, IQR) and ML methods (Isolation Forest, LOF) to identify and handle extreme values through removal, capping, or robust methods.

4. **Normalization and Standardization** (Z-score, min-max, robust scaling) ensure features are on similar scales, critical for distance-based and gradient-based algorithms.

5. **Categorical Encoding** converts non-numeric data using one-hot, label, ordinal, or target encoding, with choice depending on variable type and algorithm requirements.

6. **Data Splitting** (train-validation-test, k-fold CV, stratified splits) ensures unbiased evaluation, with temporal and group-based considerations for time series and clustered data.

7. **Imbalanced Data** requires resampling (SMOTE, undersampling), class weights, cost-sensitive learning, and appropriate metrics (PR curve, F1-score) beyond accuracy.

8. **Data Quality Assessment** evaluates completeness, consistency, accuracy, timeliness, and validity to identify issues before modeling.

9. **Data Validation** ensures data integrity through type, range, format, and business rule checks, preventing errors from propagating to models.

10. **Preprocessing Pipeline** should be documented, reproducible, and validated, with careful consideration of data leakage when using target-based methods like target encoding.
