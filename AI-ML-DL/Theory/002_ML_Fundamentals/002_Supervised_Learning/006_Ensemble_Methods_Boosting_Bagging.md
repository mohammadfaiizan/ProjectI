# Ensemble Methods Boosting Bagging

## Table of Contents

1. [Introduction to Ensemble Methods](#introduction-to-ensemble-methods)
2. [Bias-Variance Decomposition](#bias-variance-decomposition)
3. [Bagging](#bagging)
4. [Random Forests](#random-forests)
5. [Boosting](#boosting)
6. [AdaBoost](#adaboost)
7. [Gradient Boosting](#gradient-boosting)
8. [Voting Classifiers](#voting-classifiers)
9. [Stacking](#stacking)
10. [Key Takeaways](#key-takeaways)

## Introduction to Ensemble Methods

Ensemble methods combine multiple models to achieve better performance than any single model.

### What are Ensembles?

An ensemble combines predictions from multiple base learners (models):

$$\hat{y}_{\text{ensemble}}(\mathbf{x}) = f(h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_M(\mathbf{x}))$$

where $h_i$ are base learners and $f$ is a combination function.

### Why Ensembles Work

**Wisdom of Crowds**: Combining independent opinions reduces error

**Error Reduction**: If base learners have error rate $\epsilon < 0.5$ and make independent errors:

$$P(\text{majority wrong}) = \sum_{k > M/2} \binom{M}{k} \epsilon^k (1-\epsilon)^{M-k}$$

This decreases exponentially with $M$ (number of learners).

### Types of Ensembles

- **Bagging**: Train models on bootstrap samples, average predictions
- **Boosting**: Train models sequentially, each correcting previous errors
- **Stacking**: Learn how to combine base learners
- **Voting**: Simple majority or weighted voting

### Key Principles

1. **Diversity**: Base learners should make different errors
2. **Accuracy**: Base learners should be better than random
3. **Independence**: Ideally, errors should be uncorrelated

## Bias-Variance Decomposition

Understanding bias and variance helps explain why ensembles work.

### Decomposition for Ensembles

For an ensemble of $M$ models with predictions $\hat{y}_i$:

$$\text{Bias} = \mathbb{E}[\bar{y}] - y^*$$

$$\text{Variance} = \mathbb{E}[(\bar{y} - \mathbb{E}[\bar{y}])^2] = \frac{1}{M^2}\sum_{i,j} \text{Cov}(\hat{y}_i, \hat{y}_j)$$

where $\bar{y} = \frac{1}{M}\sum_{i=1}^M \hat{y}_i$ is the ensemble prediction.

### Effect of Averaging

If models are uncorrelated:

$$\text{Var}(\bar{y}) = \frac{\sigma^2}{M}$$

Variance decreases linearly with $M$.

If models have correlation $\rho$:

$$\text{Var}(\bar{y}) = \frac{\sigma^2}{M}(1 + (M-1)\rho)$$

Variance reduction depends on correlation.

### Ensemble Goals

- **Reduce Variance**: Bagging, Random Forests
- **Reduce Bias**: Boosting
- **Reduce Both**: Well-designed ensembles

## Bagging

Bootstrap Aggregating (Bagging) reduces variance by training models on bootstrap samples.

### Algorithm

**Training**:
1. For $m = 1$ to $M$:
   - Draw bootstrap sample $\mathcal{D}_m$ (sample $n$ examples with replacement)
   - Train model $h_m$ on $\mathcal{D}_m$
2. Combine: $\hat{y}(\mathbf{x}) = \frac{1}{M}\sum_{m=1}^M h_m(\mathbf{x})$

**Bootstrap Sample**: Random sample with replacement of size $n$ from $n$ training examples.

**Properties**:
- Each example appears in ~63.2% of bootstrap samples (on average)
- ~36.8% of examples are out-of-bag (OOB) for each model

### Why Bagging Works

**Variance Reduction**: Averaging reduces variance

$$\text{Var}(\bar{X}) = \frac{\text{Var}(X)}{M}$$

**Bias**: Unchanged (models trained on same distribution)

**Independence**: Bootstrap sampling creates diversity

### Out-of-Bag Error

Each model has OOB samples not used in training:

**OOB Error**:
1. For each example, find models where it was OOB
2. Evaluate those models on the example
3. Average predictions
4. Compare to true label

**Advantages**:
- Free validation set
- Unbiased estimate of generalization error
- No need for separate validation set

### When Bagging Helps

- **High Variance Models**: Models that overfit benefit most
- **Unstable Algorithms**: Small data changes cause large prediction changes
- **Examples**: Decision trees, neural networks

**Less Effective For**:
- Stable algorithms (e.g., k-NN with large $k$)
- Already low-variance models

## Random Forests

Random Forests combine bagging with random feature selection for decision trees.

### Algorithm

**Training**:
1. For $m = 1$ to $M$:
   - Draw bootstrap sample $\mathcal{D}_m$
   - Grow tree $T_m$:
     - At each node, randomly select $m \leq d$ features
     - Choose best split from $m$ features (not all $d$)
     - Grow to maximum depth (no pruning)
2. Predict: $\hat{y}(\mathbf{x}) = \frac{1}{M}\sum_{m=1}^M T_m(\mathbf{x})$

**Key Parameter**: $m = \sqrt{d}$ (classification) or $m = d/3$ (regression)

### Why Random Forests Work

**Variance Reduction**: Averaging many trees

**Bias**: Individual trees have low bias (can fit data well)

**Independence**: Random feature selection decorrelates trees

**Error Decomposition**:
$$\text{Error} = \text{Bias}^2 + \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$$

where $\rho$ is average correlation between trees and $\sigma^2$ is average variance.

Random feature selection reduces $\rho$, improving ensemble performance.

### Feature Importance

**Mean Decrease in Impurity**:
- Sum Gini/entropy reductions across all splits on feature
- Average across trees

**Permutation Importance**:
- Measure OOB accuracy
- Permute feature values
- Measure decrease in accuracy
- Average across trees

### Advantages

- Handles high-dimensional data
- Provides feature importance
- Handles missing values
- No need for feature scaling
- Robust to outliers

### Hyperparameters

- **$M$**: Number of trees (more is better, but diminishing returns)
- **$m$**: Number of features per split (controls diversity)
- **Max Depth**: Tree depth (deeper = more complex, risk of overfitting)
- **Min Samples Split**: Minimum samples to split node

## Boosting

Boosting trains models sequentially, with each model focusing on examples misclassified by previous models.

### Intuition

1. Train first model on data
2. Identify mistakes
3. Train second model to correct those mistakes
4. Continue adding models
5. Combine predictions

### Key Differences from Bagging

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Data | Bootstrap samples | Weighted/focused samples |
| Focus | Reduce variance | Reduce bias |
| Base Learners | Can be weak | Should be weak |

### Weak Learners

A weak learner performs slightly better than random guessing:

$$\text{Error} < 0.5 - \gamma$$

for some $\gamma > 0$.

Boosting combines weak learners into a strong learner.

## AdaBoost

Adaptive Boosting (AdaBoost) is the first practical boosting algorithm.

### Algorithm

**Input**: Training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, weak learner, $M$ iterations

**Initialize**: $w_i^{(1)} = 1/n$ for all $i$

**For $m = 1$ to $M$**:
1. Train weak learner $h_m$ on weighted data
2. Compute weighted error: $\epsilon_m = \sum_{i: h_m(\mathbf{x}_i) \neq y_i} w_i^{(m)}$
3. Compute weight: $\alpha_m = \frac{1}{2}\log\frac{1-\epsilon_m}{\epsilon_m}$
4. Update weights: $w_i^{(m+1)} = \frac{w_i^{(m)}}{Z_m} \exp(-\alpha_m y_i h_m(\mathbf{x}_i))$
   where $Z_m$ normalizes weights
5. Normalize: $\sum_i w_i^{(m+1)} = 1$

**Output**: $H(\mathbf{x}) = \text{sign}\left(\sum_{m=1}^M \alpha_m h_m(\mathbf{x})\right)$

### Weight Update

Weights increase for misclassified examples:

$$w_i^{(m+1)} \propto w_i^{(m)} \exp(\alpha_m) \quad \text{if } h_m(\mathbf{x}_i) \neq y_i$$

$$w_i^{(m+1)} \propto w_i^{(m)} \exp(-\alpha_m) \quad \text{if } h_m(\mathbf{x}_i) = y_i$$

### Model Weight $\alpha_m$

$$\alpha_m = \frac{1}{2}\log\frac{1-\epsilon_m}{\epsilon_m}$$

- Large $\alpha_m$: Model has low error (high confidence)
- Small $\alpha_m$: Model has high error (low confidence)
- $\alpha_m = 0$: Model performs at random ($\epsilon_m = 0.5$)

### Training Error Bound

AdaBoost's training error is bounded by:

$$\frac{1}{n}\sum_{i=1}^n \mathbb{1}(H(\mathbf{x}_i) \neq y_i) \leq \prod_{m=1}^M \sqrt{1 - 4\gamma_m^2} \leq \exp(-2\sum_{m=1}^M \gamma_m^2)$$

where $\gamma_m = 0.5 - \epsilon_m$.

If each weak learner has $\gamma_m \geq \gamma > 0$, error decreases exponentially.

### Advantages

- Simple and effective
- No hyperparameter tuning needed (just $M$)
- Provable error bounds
- Works with any weak learner

### Limitations

- Sensitive to noisy data and outliers
- Requires weak learners (not too strong)
- Sequential training (cannot parallelize)

## Gradient Boosting

Gradient Boosting views boosting as gradient descent in function space.

### Function Space Optimization

Instead of optimizing parameters, optimize functions:

$$\min_{F} \sum_{i=1}^n L(y_i, F(\mathbf{x}_i))$$

where $F$ is a function (ensemble).

### Algorithm

**Initialize**: $F_0(\mathbf{x}) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$

**For $m = 1$ to $M$**:
1. Compute residuals: $r_{im} = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$
2. Fit tree $h_m$ to residuals: $h_m = \arg\min_h \sum_{i=1}^n (r_{im} - h(\mathbf{x}_i))^2$
3. Update: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})$

**Output**: $F_M(\mathbf{x})$

where $\nu$ is learning rate (shrinkage).

### Loss Functions

**Regression**: Squared error $L(y, \hat{y}) = (y - \hat{y})^2$

Residuals: $r_i = y_i - \hat{y}_i$

**Classification**: Logistic loss $L(y, \hat{y}) = \log(1 + e^{-y\hat{y}})$

Residuals: $r_i = \frac{y_i}{1 + e^{y_i \hat{y}_i}}$

### Shrinkage (Learning Rate)

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})$$

- Smaller $\nu$: More trees needed, better generalization
- Larger $\nu$: Fewer trees, risk of overfitting
- Typical: $\nu \in [0.01, 0.1]$

### Regularization

**Tree Constraints**:
- Maximum depth (typically 3-6)
- Minimum samples per leaf
- Maximum number of leaf nodes

**Subsampling**: Train each tree on random subset of data (stochastic gradient boosting)

**Early Stopping**: Stop when validation error stops improving

### XGBoost and LightGBM

Modern implementations with optimizations:
- **XGBoost**: Regularized objective, approximate split finding, parallelization
- **LightGBM**: Gradient-based sampling, exclusive feature bundling, leaf-wise growth

## Voting Classifiers

Simple ensemble method combining predictions from multiple models.

### Hard Voting

Majority vote:

$$\hat{y} = \arg\max_{y} \sum_{m=1}^M \mathbb{1}(h_m(\mathbf{x}) = y)$$

Each model gets one vote.

### Soft Voting

Average probabilities:

$$\hat{y} = \arg\max_{y} \frac{1}{M}\sum_{m=1}^M P_m(y | \mathbf{x})$$

Requires models to output probabilities.

### Weighted Voting

Weight models by performance:

$$\hat{y} = \arg\max_{y} \sum_{m=1}^M w_m \mathbb{1}(h_m(\mathbf{x}) = y)$$

where $w_m$ are weights (e.g., based on accuracy).

### When Voting Works

- Models make different types of errors
- Models have similar performance
- Diversity in model types (e.g., SVM + Random Forest + Neural Network)

## Stacking

Stacking learns how to combine base learners using a meta-learner.

### Algorithm

**Level 1 (Base Learners)**:
1. Train $M$ base learners $h_1, \ldots, h_M$
2. Use cross-validation to generate out-of-fold predictions
3. Create meta-features: $\mathbf{z}_i = [h_1(\mathbf{x}_i), \ldots, h_M(\mathbf{x}_i)]$

**Level 2 (Meta-Learner)**:
1. Train meta-learner $g$ on $(\mathbf{z}_i, y_i)$
2. Meta-learner learns how to combine base learners

**Prediction**:
$$\hat{y} = g([h_1(\mathbf{x}), \ldots, h_M(\mathbf{x})])$$

### Meta-Learner Choices

- **Linear Regression**: Simple, interpretable
- **Logistic Regression**: For classification
- **Neural Network**: Can learn complex combinations
- **Simple Average**: Falls back to voting if meta-learner is identity

### Advantages

- Can learn optimal combination
- Handles different model types
- Often better than simple voting

### Disadvantages

- More complex
- Risk of overfitting meta-learner
- Requires careful cross-validation

### Blending

Similar to stacking but uses hold-out validation set instead of cross-validation:
- Simpler
- Less data for meta-learner
- Faster

## Key Takeaways

1. **Ensemble Methods** combine multiple models to achieve better performance through variance reduction (bagging) or bias reduction (boosting).

2. **Bagging** trains models on bootstrap samples and averages predictions, reducing variance while maintaining bias, most effective for high-variance models.

3. **Random Forests** combine bagging with random feature selection for decision trees, decorrelating trees and reducing both bias and variance.

4. **Boosting** trains models sequentially, with each model focusing on examples misclassified by previous models, reducing bias through adaptive weighting.

5. **AdaBoost** adaptively reweights training examples, giving more weight to misclassified examples, with model weights $\alpha_m = \frac{1}{2}\log\frac{1-\epsilon_m}{\epsilon_m}$ based on error rate.

6. **Gradient Boosting** views boosting as gradient descent in function space, fitting trees to residuals $r_i = -\frac{\partial L}{\partial F}$ with shrinkage $\nu$ for regularization.

7. **Voting Classifiers** combine predictions via majority vote (hard) or probability averaging (soft), simple but effective when models are diverse.

8. **Stacking** learns optimal combination using a meta-learner trained on out-of-fold predictions from base learners, often outperforming simple voting.

9. **Key Principles** for effective ensembles include diversity (different errors), accuracy (better than random), and independence (uncorrelated errors) among base learners.

10. **Ensemble Selection** depends on problem: bagging for variance reduction, boosting for bias reduction, voting for simplicity, stacking for optimal combination, with Random Forests and Gradient Boosting being most popular in practice.
