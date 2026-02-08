# Cross Validation Resampling

## Table of Contents

1. [Introduction to Resampling Methods](#introduction-to-resampling-methods)
2. [Holdout Method](#holdout-method)
3. [K-Fold Cross-Validation](#k-fold-cross-validation)
4. [Stratified Cross-Validation](#stratified-cross-validation)
5. [Leave-One-Out Cross-Validation](#leave-one-out-cross-validation)
6. [Bootstrap Methods](#bootstrap-methods)
7. [Nested Cross-Validation](#nested-cross-validation)
8. [Time Series Cross-Validation](#time-series-cross-validation)
9. [Group-Based Cross-Validation](#group-based-cross-validation)
10. [Key Takeaways](#key-takeaways)

## Introduction to Resampling Methods

Resampling methods estimate model performance by repeatedly sampling from the available data.

### Why Resampling?

**Problem**: Need to estimate generalization error, but test set should not be used during development.

**Solution**: Resample training data to simulate train/test splits.

### Goals

- **Model Evaluation**: Estimate performance on unseen data
- **Model Selection**: Choose between different models
- **Hyperparameter Tuning**: Select optimal hyperparameters
- **Variance Estimation**: Assess stability of performance estimates

### Types of Resampling

- **Holdout**: Single train/test split
- **Cross-Validation**: Multiple train/test splits
- **Bootstrap**: Sample with replacement
- **Time Series CV**: Maintains temporal order

### Bias-Variance Tradeoff

- **More Data for Training**: Lower bias, higher variance in estimate
- **More Data for Validation**: Lower variance, higher bias
- **Multiple Splits**: Reduce variance through averaging

## Holdout Method

Simplest resampling method: single train/test split.

### Procedure

1. Randomly split data into:
   - **Training Set**: Used to train model (typically 60-80%)
   - **Test Set**: Used for final evaluation (typically 20-40%)
2. Train model on training set
3. Evaluate on test set

### Advantages

- Simple and fast
- Requires single model training
- Standard practice for final evaluation

### Disadvantages

- **High Variance**: Single split may not be representative
- **Data Waste**: Test set not used for training
- **Sensitive to Split**: Different splits yield different results
- **No Model Selection**: Cannot use test set for hyperparameter tuning

### When to Use

- Large datasets (variance less of concern)
- Final evaluation after model selection
- Quick baseline assessment

### Three-Way Split

For model development:
- **Training Set**: Train model (60%)
- **Validation Set**: Tune hyperparameters (20%)
- **Test Set**: Final evaluation (20%)

**Critical Rule**: Test set used only once, at the very end.

## K-Fold Cross-Validation

K-fold CV divides data into $k$ folds and uses each fold as validation set once.

### Algorithm

1. Randomly shuffle data
2. Divide into $k$ folds of approximately equal size
3. For $i = 1$ to $k$:
   - Use fold $i$ as validation set
   - Use remaining $k-1$ folds as training set
   - Train model and evaluate on fold $i$
4. Average performance across $k$ folds

### Performance Estimate

$$\text{CV Score} = \frac{1}{k}\sum_{i=1}^k \text{Score}_i$$

where $\text{Score}_i$ is performance on fold $i$.

### Common Choices of K

**$k = 5$**: 
- Good balance between bias and variance
- 5 models to train
- Common default

**$k = 10$**: 
- Lower bias (more data for training)
- More stable estimate
- Standard choice

**$k = 3$**: 
- Faster computation
- Higher variance
- Use when data is limited

### Advantages

- **Uses All Data**: Every sample used for both training and validation
- **Lower Variance**: Averaging reduces variance
- **More Reliable**: Less sensitive to particular split
- **Better Estimate**: More accurate than single holdout

### Disadvantages

- **Computational Cost**: Train $k$ models
- **Bias**: Slightly pessimistic (less data per fold than full training)
- **Assumes IID**: Requires independent, identically distributed data

### Stratification

For classification, maintain class distribution in each fold (stratified k-fold).

## Stratified Cross-Validation

Stratified CV ensures each fold has similar class distribution as full dataset.

### Why Stratify?

**Problem**: Random splits may create folds with imbalanced classes.

**Example**: Binary classification with 90% class A, 10% class B
- Random split may put all class B in one fold
- That fold would have 100% class B (not representative)

### Procedure

1. For each class, divide samples into $k$ folds
2. Combine corresponding folds from all classes
3. Each fold maintains original class distribution

### Advantages

- **Representative Folds**: Each fold reflects full data distribution
- **Better Estimates**: More reliable performance estimates
- **Handles Imbalance**: Works well with imbalanced data

### When to Use

- Classification problems
- Imbalanced datasets
- Small datasets (where random split may be unrepresentative)

## Leave-One-Out Cross-Validation

LOOCV uses $k = n$ (one sample per fold).

### Algorithm

1. For $i = 1$ to $n$:
   - Use sample $i$ as validation set
   - Use remaining $n-1$ samples as training set
   - Train and evaluate
2. Average performance

### Performance Estimate

$$\text{LOOCV Score} = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{y}_{-i})$$

where $\hat{y}_{-i}$ is prediction for sample $i$ using model trained on all samples except $i$.

### Advantages

- **Unbiased**: Uses $n-1$ samples for training (nearly full data)
- **Deterministic**: No randomness in fold assignment
- **Uses All Data**: Maximum data utilization

### Disadvantages

- **Computational Cost**: Train $n$ models ($O(n)$ times slower)
- **High Variance**: Single sample validation (high variance in estimate)
- **Expensive**: Prohibitive for large $n$

### When to Use

- Very small datasets
- Need unbiased estimate
- Computational cost acceptable

### Efficient Computation

For some models (e.g., linear regression), LOOCV can be computed efficiently without training $n$ models:

$$\text{LOOCV} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{1 - h_i}\right)^2$$

where $h_i$ is leverage of sample $i$.

## Bootstrap Methods

Bootstrap samples with replacement to estimate performance.

### Bootstrap Sampling

**Procedure**:
1. Sample $n$ points with replacement from dataset of size $n$
2. Some points appear multiple times, some not at all
3. Expected fraction of unique points: $1 - (1-1/n)^n \approx 0.632$

### Bootstrap Performance Estimate

**0.632 Bootstrap**:
1. For $b = 1$ to $B$:
   - Create bootstrap sample
   - Train on bootstrap sample
   - Evaluate on out-of-bag (OOB) samples
2. Average OOB performance

**Bootstrap Estimate**:
$$\hat{\text{Err}}^{(1)} = \frac{1}{B}\sum_{b=1}^B \frac{1}{|C^{-b}|}\sum_{i \in C^{-b}} L(y_i, \hat{f}_b(\mathbf{x}_i))$$

where $C^{-b}$ are OOB samples for bootstrap $b$.

### 0.632 Estimator

Combines training error and bootstrap error:

$$\hat{\text{Err}}^{(0.632)} = 0.368 \cdot \overline{\text{err}} + 0.632 \cdot \hat{\text{Err}}^{(1)}$$

where $\overline{\text{err}}$ is average training error.

### Advantages

- **Flexible**: Works with any performance metric
- **Variance Estimation**: Can estimate variance of performance
- **Confidence Intervals**: Provides confidence intervals

### Disadvantages

- **Optimistic Bias**: Bootstrap samples overlap with training data
- **Computational Cost**: Train $B$ models (typically $B = 100$ or more)
- **Less Common**: Not as widely used as k-fold CV

## Nested Cross-Validation

Nested CV uses outer loop for evaluation and inner loop for hyperparameter tuning.

### Problem

Using same CV for both hyperparameter tuning and evaluation causes **optimistic bias**:
- Hyperparameters selected to optimize CV score
- Same CV used for final evaluation
- Performance estimate is biased (too optimistic)

### Solution: Nested CV

**Outer Loop** (Evaluation):
- $k_1$ folds for final performance estimate
- Each fold: train on $k_1-1$ folds, test on held-out fold

**Inner Loop** (Hyperparameter Tuning):
- For each outer fold, use $k_2$-fold CV on training data
- Select hyperparameters optimizing inner CV
- Train final model with selected hyperparameters on outer training fold
- Evaluate on outer test fold

### Algorithm

For each outer fold $i = 1, \ldots, k_1$:
1. Outer training: $D_{\text{train}}^{(i)} = D \setminus D_{\text{test}}^{(i)}$
2. Inner CV on $D_{\text{train}}^{(i)}$:
   - For each hyperparameter configuration:
     - Perform $k_2$-fold CV
     - Average performance
   - Select best hyperparameters $\lambda^*$
3. Train model with $\lambda^*$ on $D_{\text{train}}^{(i)}$
4. Evaluate on $D_{\text{test}}^{(i)}$

Final performance: Average over $k_1$ outer folds.

### Advantages

- **Unbiased**: Separate data for tuning and evaluation
- **Reliable**: Honest performance estimate
- **Best Practice**: Recommended for model selection

### Disadvantages

- **Computational Cost**: $k_1 \times k_2 \times$ (number of hyperparameter configs) models
- **Expensive**: May be prohibitive for large hyperparameter grids

### When to Use

- Need unbiased performance estimate
- Hyperparameter tuning required
- Computational resources available
- Research/publication (best practice)

## Time Series Cross-Validation

Time series data requires special handling to maintain temporal order.

### Problem with Standard CV

Standard k-fold CV randomly shuffles data, breaking temporal dependencies:
- Future data used to predict past (data leakage)
- Violates temporal order
- Unrealistic evaluation

### Forward Chaining (Time Series Split)

**Procedure**:
1. Start with initial training window
2. For each time step:
   - Train on data up to time $t$
   - Validate on time $t+1$
   - Expand training window
3. Continue until end of data

**Example**: 
- Fold 1: Train on $[1, t_1]$, test on $[t_1+1, t_2]$
- Fold 2: Train on $[1, t_2]$, test on $[t_2+1, t_3]$
- Fold 3: Train on $[1, t_3]$, test on $[t_3+1, t_4]$

### Rolling Window

**Fixed Training Window**:
- Training window of fixed size $w$
- Slide window forward
- Each fold: train on $w$ points, test on next point(s)

**Example**:
- Fold 1: Train on $[1, w]$, test on $[w+1]$
- Fold 2: Train on $[2, w+1]$, test on $[w+2]$
- Fold 3: Train on $[3, w+2]$, test on $[w+3]$

### Expanding Window

**Growing Training Window**:
- Start with initial window
- Each fold adds more data to training set
- More data over time

**Example**:
- Fold 1: Train on $[1, w]$, test on $[w+1]$
- Fold 2: Train on $[1, w+1]$, test on $[w+2]$
- Fold 3: Train on $[1, w+2]$, test on $[w+3]$

### Advantages

- **Realistic**: Maintains temporal order
- **No Data Leakage**: Future not used to predict past
- **Practical**: Matches real-world deployment

### Considerations

- **Non-Stationarity**: Data distribution may change over time
- **Concept Drift**: Relationships may evolve
- **Window Size**: Balance between data availability and recency

## Group-Based Cross-Validation

When data has groups (e.g., patients, subjects), keep groups together.

### Problem

Standard CV may split groups:
- Some samples from same patient in train, others in test
- Model sees similar data in both sets
- Overly optimistic performance

### Group K-Fold

**Procedure**:
1. Identify groups (e.g., patient IDs)
2. Assign groups to folds (not individual samples)
3. All samples from same group in same fold

**Example**: Medical data with multiple measurements per patient
- Patient 1: samples $[1, 2, 3]$ → Fold 1
- Patient 2: samples $[4, 5]$ → Fold 2
- Patient 3: samples $[6, 7, 8]$ → Fold 1

### Leave-One-Group-Out

Similar to LOOCV but for groups:
- Each fold: one group as test, others as train
- Number of folds = number of groups

### Applications

- **Medical Data**: Multiple measurements per patient
- **Repeated Measures**: Longitudinal studies
- **Clustered Data**: Hierarchical structure
- **Recommendation Systems**: User-based groups

### Advantages

- **Realistic**: Tests generalization to new groups
- **No Leakage**: Prevents information leakage within groups
- **Appropriate**: Matches deployment scenario

## Key Takeaways

1. **Resampling Methods** estimate model performance by repeatedly sampling from data, used for evaluation, model selection, and hyperparameter tuning.

2. **Holdout Method** uses single train/test split (60-80% train, 20-40% test), simple but high variance, suitable for large datasets or final evaluation.

3. **K-Fold Cross-Validation** divides data into $k$ folds, uses each as validation once, averaging performance: $\text{CV} = \frac{1}{k}\sum_{i=1}^k \text{Score}_i$, with $k=10$ standard and $k=5$ common alternative.

4. **Stratified K-Fold** maintains class distribution in each fold, essential for classification and imbalanced data, ensuring representative folds.

5. **Leave-One-Out CV** uses $k=n$ (one sample per fold), nearly unbiased but computationally expensive ($O(n)$ models), suitable for very small datasets.

6. **Bootstrap Methods** sample with replacement, using out-of-bag samples for evaluation, providing variance estimates and confidence intervals but computationally expensive.

7. **Nested Cross-Validation** uses outer loop for evaluation and inner loop for hyperparameter tuning, preventing optimistic bias by separating tuning and evaluation data, best practice for model selection.

8. **Time Series CV** maintains temporal order through forward chaining (expanding/rolling windows), preventing data leakage and providing realistic evaluation for temporal data.

9. **Group-Based CV** keeps groups together (e.g., all samples from same patient), preventing information leakage within groups and testing generalization to new groups, essential for clustered/hierarchical data.

10. **Method Selection** depends on data size (holdout for large, LOOCV for small), data type (time series CV for temporal, group CV for clustered), purpose (nested CV for tuning+evaluation), and computational constraints, with k-fold CV being the default choice for most scenarios.
