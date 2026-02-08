# Ensemble Methods and Boosting

## Bagging

Bagging (Bootstrap Aggregating) reduces variance by averaging predictions from models trained on bootstrap samples.

### Algorithm

1. **Bootstrap:** Sample $n$ observations with replacement $B$ times
2. **Train:** Fit model on each bootstrap sample
3. **Predict:** Average predictions (regression) or majority vote (classification)

**Prediction:**
$$\hat{f}_{bag}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(\mathbf{x})$$

### Variance Reduction

**Bias:** Unchanged (same as base model)
**Variance:** Reduced by factor $\approx 1/B$ (if models independent)

**Why it works:** 
- High-variance models benefit most (trees, neural networks)
- Low-variance models benefit less (linear regression)

### Bootstrap Aggregation

**Bootstrap sample:** Random sample with replacement of size $n$ from original data.

**Expected overlap:** Each observation appears in ~63% of bootstrap samples on average.

**Out-of-bag (OOB):** Observations not in bootstrap sample can be used for validation.

## Random Forests

Random forests combine bagging with random feature selection.

### Algorithm

For each tree $b = 1, \ldots, B$:

1. **Bootstrap sample:** Sample $n$ observations with replacement
2. **Train tree:** 
   - At each split, randomly select $m$ features from $p$ total
   - Choose best split among $m$ features
   - Grow tree to maximum depth (or other stopping criterion)
3. **No pruning:** Trees grown to full depth

**Prediction:**
$$\hat{f}_{RF}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B}\hat{T}_b(\mathbf{x})$$

### Key Parameters

**$B$:** Number of trees
- More trees: Lower variance, but diminishing returns
- Typical: 100-500 trees

**$m$:** Number of features per split
- Classification: $m = \sqrt{p}$ (default)
- Regression: $m = p/3$ (default)
- Smaller $m$: More decorrelation, but may miss important features

**Max depth:** Control tree complexity
**Min samples per leaf:** Prevent overfitting

### Why Random Forests Work

**Variance reduction:** Averaging reduces variance

**Bias reduction:** More complex trees (grown to full depth) have lower bias

**Decorrelation:** Random feature selection reduces correlation between trees

**Error decomposition:**
$$\mathbb{E}[(Y - \hat{f}_{RF})^2] = \sigma^2 + \rho\sigma_T^2 + (1-\rho)\sigma_T^2/B$$

where:
- $\sigma^2$: Irreducible error
- $\sigma_T^2$: Variance of single tree
- $\rho$: Correlation between trees

Lower $\rho$ (more decorrelation) → lower error.

### Variable Importance

**Mean decrease in impurity:**
- Sum of Gini/entropy decreases over all splits using feature
- Average across all trees
- Normalize by number of trees

**Permutation importance:**
- Shuffle feature values
- Measure increase in OOB error
- Average across trees
- More robust to correlated features

### Out-of-Bag Error

For each observation, predict using trees that didn't include it.

**OOB error:** Misclassification rate on OOB predictions.

**Advantages:**
- Built-in validation
- No need for separate test set
- Efficient (no extra computation)

## Boosting

Boosting builds an ensemble sequentially, with each model correcting errors of previous models.

### AdaBoost

**Algorithm:**

1. **Initialize weights:** $w_i^{(1)} = 1/n$ for all $i$

2. **For $m = 1$ to $M$:**
   - Fit classifier $G_m(\mathbf{x})$ using weights $w_i^{(m)}$
   - Compute error: $\epsilon_m = \sum_{i=1}^{n}w_i^{(m)}\mathbf{1}(y_i \neq G_m(\mathbf{x}_i))$
   - Compute weight: $\alpha_m = \frac{1}{2}\ln((1-\epsilon_m)/\epsilon_m)$
   - Update weights: $w_i^{(m+1)} = w_i^{(m)}\exp(-\alpha_m y_i G_m(\mathbf{x}_i))/Z_m$
     where $Z_m$ normalizes weights

3. **Final classifier:**
   $$G(\mathbf{x}) = \text{sign}\left(\sum_{m=1}^{M}\alpha_m G_m(\mathbf{x})\right)$$

**Interpretation:**
- Misclassified observations get higher weights
- Subsequent models focus on hard examples
- $\alpha_m$ larger for better classifiers

### Gradient Boosting Framework

General framework for boosting any loss function.

**Algorithm:**

1. **Initialize:**
   $$F_0(\mathbf{x}) = \arg\min_\gamma \sum_{i=1}^{n}L(y_i, \gamma)$$

2. **For $m = 1$ to $M$:**
   - Compute negative gradient (pseudo-residuals):
     $$r_{im} = -\left[\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}\right]$$
   - Fit regression tree $h_m(\mathbf{x})$ to $r_{im}$
   - Find step size:
     $$\gamma_m = \arg\min_\gamma \sum_{i=1}^{n}L(y_i, F_{m-1}(\mathbf{x}_i) + \gamma h_m(\mathbf{x}_i))$$
   - Update:
     $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\gamma_m h_m(\mathbf{x})$$
     where $\nu$ is learning rate (shrinkage)

3. **Output:** $F_M(\mathbf{x})$

**Loss functions:**
- **Regression:** Squared error, absolute error, Huber
- **Classification:** Logistic loss, exponential loss

**Learning rate:** Smaller $\nu$ requires more iterations but often better generalization.

## XGBoost

XGBoost (Extreme Gradient Boosting) adds regularization and optimizations.

### Objective Function

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n}L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

where regularization term:
$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2$$

- $T$: Number of leaves
- $\mathbf{w}$: Leaf weights
- $\gamma$: Complexity penalty
- $\lambda$: L2 regularization

### Second-Order Approximation

Use second-order Taylor expansion:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n}[L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t^2(\mathbf{x}_i)] + \Omega(f_t)$$

where:
- $g_i = \partial L/\partial \hat{y}_i^{(t-1)}$ (first derivative)
- $h_i = \partial^2 L/\partial \hat{y}_i^{(t-1)}$ (second derivative)

**Advantage:** More accurate than first-order methods.

### Tree Construction

**Greedy algorithm:** At each split, choose feature and threshold maximizing gain:

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

where $G_L, G_R$ are sums of gradients in left/right child, $H_L, H_R$ are sums of hessians.

### Features

**Regularization:** Prevents overfitting

**Handles missing data:** Learns default direction for missing values

**Parallelization:** Parallel tree construction

**Approximate algorithm:** Uses histograms for speed

**Sparsity awareness:** Handles sparse features efficiently

**Cross-validation:** Built-in CV during training

### Hyperparameters

**Learning rate ($\nu$):** Typically 0.01-0.3, smaller = more trees needed

**Max depth:** Tree complexity, typically 3-10

**Subsample:** Fraction of data per tree (row sampling), prevents overfitting

**Colsample_bytree:** Fraction of features per tree (column sampling)

**Min child weight:** Minimum sum of hessians in leaf

**$\lambda$:** L2 regularization

**$\gamma$:** Minimum loss reduction for split

**Early stopping:** Stop if validation error doesn't improve for $k$ rounds

## LightGBM

LightGBM uses leaf-wise growth and histogram-based algorithm.

### Leaf-Wise Growth

**Level-wise (XGBoost):** Grow all leaves at same level
**Leaf-wise (LightGBM):** Grow leaf with largest loss reduction

**Advantages:**
- Lower training error for same number of leaves
- Faster training
- Better accuracy often

**Disadvantages:**
- May overfit with small data
- Need to limit max depth

### Histogram-Based Algorithm

**Discretize features:** Create histograms with bins

**Advantages:**
- Faster (fewer split candidates)
- Lower memory usage
- Handles large datasets

### Gradient-Based One-Side Sampling (GOSS)

**Idea:** Focus on observations with large gradients

**Algorithm:**
1. Keep top $a \times 100\%$ observations by gradient
2. Randomly sample $b \times 100\%$ from rest
3. Weight sampled observations by $(1-a)/b$

**Advantage:** Faster training while maintaining accuracy

### Exclusive Feature Bundling (EFB)

**Idea:** Bundle mutually exclusive sparse features

**Advantage:** Reduces number of features, faster training

### Hyperparameters

Similar to XGBoost, but:
- **Num leaves:** Instead of max depth (more intuitive)
- **Feature fraction:** Column sampling
- **Bagging fraction:** Row sampling
- **Min data in leaf:** Instead of min child weight

## Stacking and Blending

### Stacking

**Level 1:** Train multiple base models
**Level 2:** Train meta-model on base model predictions

**Algorithm:**

1. **Split data:** Training and hold-out
2. **Cross-validation:** For each base model, get out-of-fold predictions
3. **Train meta-model:** Use out-of-fold predictions as features
4. **Final prediction:** Base models predict on test, meta-model combines

**Meta-model:** Often simple (linear regression, logistic regression).

**Advantages:**
- Can combine different model types
- Learns optimal combination

**Disadvantages:**
- More complex
- Risk of overfitting

### Blending

Simpler version of stacking:
- Single hold-out set (not CV)
- Train base models on training set
- Predict on hold-out set
- Train meta-model on hold-out predictions

**Advantage:** Simpler, faster
**Disadvantage:** Less data for meta-model

## Hyperparameter Tuning

### Grid Search

**Exhaustive search:** Try all combinations in grid

**Advantages:**
- Systematic
- Guaranteed to find best in grid

**Disadvantages:**
- Computationally expensive
- May miss optimal values between grid points

### Random Search

**Random sampling:** Sample hyperparameters randomly

**Advantages:**
- Faster
- May find better values (not restricted to grid)

**Disadvantages:**
- May miss good regions

### Bayesian Optimization

**Gaussian process:** Model performance as function of hyperparameters

**Acquisition function:** Choose next hyperparameters to evaluate
- **Expected improvement:** Maximize expected improvement over current best
- **Upper confidence bound:** Balance exploration/exploitation

**Advantages:**
- Efficient (fewer evaluations needed)
- Adapts to function shape

**Disadvantages:**
- More complex
- Requires tuning of GP hyperparameters

### Cross-Validation

**k-fold CV:** Standard approach

**Time series:** Use forward chaining (walk-forward)

**Nested CV:** Outer CV for model selection, inner CV for hyperparameter tuning

## Avoiding Overfitting in Financial Prediction

### Challenges

**Low signal-to-noise:** Financial returns have low predictability

**Non-stationarity:** Relationships change over time

**Look-ahead bias:** Easy to accidentally use future information

**Survivorship bias:** Only surviving assets in dataset

### Strategies

**Regularization:** L1/L2 penalties, early stopping

**Feature selection:** Remove irrelevant features

**Simpler models:** Prefer interpretable models when possible

**Out-of-sample testing:** Always validate on unseen data

**Walk-forward analysis:** Retrain model periodically

**Ensemble diversity:** Use different model types, features, time periods

**Cross-validation:** Proper time-series CV (no future data)

### Evaluation

**Sharpe ratio:** Risk-adjusted returns

**Information ratio:** Active return per unit of tracking error

**Maximum drawdown:** Worst peak-to-trough decline

**Calmar ratio:** Return / maximum drawdown

**Out-of-sample:** Always test on future data

**Transaction costs:** Include in evaluation

**Slippage:** Account for execution costs

## Practical Considerations

### Model Selection

**Bias-variance trade-off:**
- Simple models: High bias, low variance
- Complex models: Low bias, high variance
- Ensemble: Can reduce both

**Computational cost:**
- Bagging: Parallelizable
- Boosting: Sequential (harder to parallelize)
- Consider training time vs accuracy

### Interpretability

**Feature importance:** Understand what drives predictions

**SHAP values:** Explain individual predictions

**Partial dependence:** Visualize feature effects

**Trade-off:** More complex models less interpretable

### Production

**Latency:** Fast prediction for real-time use

**Model monitoring:** Track performance over time

**Retraining:** Update model periodically

**Version control:** Track model versions

**A/B testing:** Compare model versions
