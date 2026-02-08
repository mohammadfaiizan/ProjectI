# Tree Based Methods Decision Trees

## Table of Contents

1. [Introduction to Decision Trees](#introduction-to-decision-trees)
2. [Decision Tree Algorithms](#decision-tree-algorithms)
3. [Splitting Criteria](#splitting-criteria)
4. [Pruning and Regularization](#pruning-and-regularization)
5. [Random Forests](#random-forests)
6. [Gradient Boosting](#gradient-boosting)
7. [XGBoost and LightGBM](#xgboost-and-lightgbm)
8. [Feature Importance](#feature-importance)
9. [Advantages and Limitations](#advantages-and-limitations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Decision Trees

Decision trees are non-parametric supervised learning methods that partition the feature space into regions and make predictions based on simple if-then rules.

### What are Decision Trees?

A decision tree is a flowchart-like structure where:
- **Internal Nodes**: Represent tests on features
- **Branches**: Represent outcomes of tests
- **Leaf Nodes**: Represent class labels (classification) or values (regression)

Each path from root to leaf represents a decision rule.

### Tree Structure

For a tree with depth $D$, the decision process follows:

1. Start at root node
2. Test feature $x_j$ against threshold $t$
3. Move to left child if $x_j \leq t$, right child otherwise
4. Repeat until reaching a leaf node
5. Return prediction from leaf

### Advantages

- **Interpretability**: Easy to visualize and understand
- **No Assumptions**: No distributional assumptions
- **Handles Mixed Data**: Works with both numerical and categorical features
- **Feature Selection**: Automatically selects relevant features
- **Non-linear**: Can capture non-linear relationships

### Limitations

- **Overfitting**: Prone to overfitting without regularization
- **Instability**: Small data changes can drastically alter tree structure
- **Greedy**: Local optimization may miss global optimum
- **Axis-Aligned**: Creates axis-aligned boundaries (may not be optimal)

## Decision Tree Algorithms

Several algorithms exist for constructing decision trees, differing in splitting criteria and handling of continuous vs. categorical features.

### ID3 Algorithm

Iterative Dichotomiser 3 (ID3) uses information gain for splitting:

**Algorithm**:
1. Start with root node containing all training data
2. For each feature, calculate information gain
3. Split on feature with highest information gain
4. Recursively apply to child nodes
5. Stop when all samples in node belong to same class or no features remain

**Limitations**:
- Only handles categorical features
- No pruning (overfitting risk)
- Biased toward features with many values

### C4.5 Algorithm

C4.5 extends ID3 with improvements:

**Enhancements**:
- Handles continuous features via threshold-based splits
- Handles missing values
- Uses gain ratio instead of information gain (reduces bias)
- Includes pruning

**Gain Ratio**:
$$\text{GainRatio}(S, A) = \frac{\text{InformationGain}(S, A)}{\text{SplitInfo}(S, A)}$$

where SplitInfo penalizes features with many values.

### CART Algorithm

Classification and Regression Trees (CART) uses Gini impurity:

**Algorithm**:
1. For each feature and threshold, calculate impurity reduction
2. Choose split maximizing reduction
3. Recursively build left and right subtrees
4. Stop when stopping criteria met

**Features**:
- Handles both classification and regression
- Binary splits (simpler than multi-way)
- Uses Gini impurity for classification
- Uses MSE for regression

### Regression Trees

For regression, trees predict the mean value in each leaf:

**Splitting Criterion**: Minimize mean squared error:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2$$

where $\bar{y}$ is the mean target in the node.

**Prediction**: For a leaf node, predict the mean of training samples in that leaf.

## Splitting Criteria

The choice of splitting criterion determines how trees partition the feature space.

### Entropy and Information Gain

**Entropy** measures impurity:

$$H(S) = -\sum_{i=1}^c p_i \log_2 p_i$$

where $p_i$ is proportion of class $i$ in set $S$, and $c$ is number of classes.

**Information Gain** measures reduction in entropy:

$$\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

where $S_v$ is subset of $S$ with value $v$ for feature $A$.

**Properties**:
- Maximum when classes are equally distributed
- Zero when all samples belong to one class
- Favors splits that create pure nodes

### Gini Impurity

Gini impurity measures probability of misclassification:

$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

**Gini Gain**:
$$\text{GiniGain}(S, A) = Gini(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} Gini(S_v)$$

**Properties**:
- Similar to entropy but computationally faster
- Maximum when classes are equally distributed
- Zero when all samples belong to one class
- Used in CART algorithm

### Variance Reduction (Regression)

For regression trees, minimize variance:

$$\text{Var}(S) = \frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2$$

**Variance Reduction**:
$$\text{VarReduction}(S, A) = \text{Var}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Var}(S_v)$$

### Chi-Square Test

Chi-square test measures independence between feature and target:

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

where $O_{ij}$ is observed frequency and $E_{ij}$ is expected frequency.

Higher chi-square indicates stronger association, suggesting a good split.

### Handling Continuous Features

For continuous features, find optimal threshold:

1. Sort unique values of feature
2. Consider midpoints between consecutive values as thresholds
3. Evaluate each threshold using splitting criterion
4. Choose threshold with best score

**Example**: For feature values $[1, 3, 5, 7]$, consider thresholds $[2, 4, 6]$.

## Pruning and Regularization

Pruning prevents overfitting by removing branches that don't improve generalization.

### Pre-Pruning (Early Stopping)

Stop tree growth before perfect fit:

**Stopping Criteria**:
- Maximum depth reached
- Minimum samples per leaf
- Minimum samples to split
- Minimum impurity decrease
- Maximum number of leaf nodes

**Advantages**: Fast, simple

**Disadvantages**: May stop too early (underfitting)

### Post-Pruning

Grow full tree, then remove branches:

**Cost-Complexity Pruning**:
Minimize: $\text{Loss}(T) + \alpha |T|$

where:
- $T$ is tree
- $|T|$ is number of leaf nodes
- $\alpha$ is complexity parameter

**Algorithm**:
1. Grow full tree
2. For each $\alpha$, find subtree minimizing cost-complexity
3. Use cross-validation to select best $\alpha$
4. Return pruned tree

### Reduced Error Pruning

1. Grow tree on training set
2. For each node, evaluate error on validation set
3. Remove node if error doesn't increase
4. Greedily remove nodes bottom-up

### Minimum Description Length (MDL)

Balance tree complexity and fit quality using information theory:

$$\text{MDL} = \text{EncodingLength}(\text{Tree}) + \text{EncodingLength}(\text{Errors})$$

Prefer simpler trees that fit data well.

## Random Forests

Random Forests combine multiple decision trees through bagging and random feature selection.

### Bagging (Bootstrap Aggregating)

**Algorithm**:
1. Create $B$ bootstrap samples from training data
2. Train decision tree on each bootstrap sample
3. For prediction, average predictions (regression) or majority vote (classification)

**Bootstrap Sample**: Random sample with replacement of size $n$ from $n$ training samples.

**Variance Reduction**: Averaging reduces variance:
$$\text{Var}(\bar{X}) = \frac{\text{Var}(X)}{B}$$

### Random Forest Algorithm

Random Forest adds random feature selection to bagging:

**Algorithm**:
1. For $b = 1$ to $B$:
   - Draw bootstrap sample $\mathcal{D}_b$
   - Grow tree $T_b$:
     - At each node, randomly select $m \leq d$ features
     - Choose best split from $m$ features (not all $d$)
2. Output: $\hat{f}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B T_b(\mathbf{x})$

**Key Parameter**: $m = \sqrt{d}$ (classification) or $m = d/3$ (regression)

### Why Random Forests Work

**Variance Reduction**: Averaging many trees reduces variance

**Bias**: Individual trees have low bias (can fit data well)

**Independence**: Random feature selection decorrelates trees

**Error Decomposition**:
$$\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

Random Forests reduce variance while maintaining low bias.

### Out-of-Bag (OOB) Error

Each tree is trained on bootstrap sample, leaving ~37% samples out-of-bag.

**OOB Error**: Evaluate each tree on its OOB samples, average across trees.

**Advantages**:
- No need for separate validation set
- Unbiased estimate of generalization error
- Free cross-validation

## Gradient Boosting

Gradient Boosting builds an ensemble by sequentially adding trees that correct previous mistakes.

### Boosting Intuition

Instead of training independent models, train models sequentially:
1. Train first model on data
2. Train second model on residuals (errors) of first model
3. Continue adding models that correct previous errors
4. Combine predictions

### Gradient Boosting Algorithm

**Algorithm**:
1. Initialize: $F_0(\mathbf{x}) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$
2. For $m = 1$ to $M$:
   - Compute residuals: $r_{im} = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$
   - Fit tree $h_m$ to residuals: $h_m = \arg\min_h \sum_{i=1}^n (r_{im} - h(\mathbf{x}_i))^2$
   - Update: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})$
3. Output: $F_M(\mathbf{x})$

where $\nu$ is learning rate (shrinkage).

### Loss Functions

**Regression**: Squared error $L(y, \hat{y}) = (y - \hat{y})^2$

**Classification**: Logistic loss $L(y, \hat{y}) = \log(1 + e^{-y\hat{y}})$

### Shrinkage (Learning Rate)

Shrinkage parameter $\nu \in (0, 1]$ controls contribution of each tree:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})$$

**Effect**:
- Smaller $\nu$: More trees needed, but better generalization
- Larger $\nu$: Fewer trees, but risk of overfitting

### Regularization

**Tree Constraints**:
- Maximum depth
- Minimum samples per leaf
- Maximum number of leaf nodes

**Subsampling**: Train each tree on random subset of data (stochastic gradient boosting)

**Early Stopping**: Stop when validation error stops improving

## XGBoost and LightGBM

Modern gradient boosting implementations with optimizations.

### XGBoost (Extreme Gradient Boosting)

**Key Features**:
- **Regularized Objective**: Adds L1 and L2 regularization
- **Tree Construction**: Approximate algorithm for finding splits
- **Parallelization**: Parallel tree construction
- **Handles Missing Values**: Learns default direction for missing values
- **Cross-Validation**: Built-in CV

**Objective Function**:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$ regularizes tree complexity.

**Split Finding**:
- Approximate algorithm using quantiles
- Handles sparse data efficiently
- Cache-aware access patterns

### LightGBM

**Key Features**:
- **Gradient-Based One-Side Sampling (GOSS)**: Focus on samples with large gradients
- **Exclusive Feature Bundling (EFB)**: Combine sparse features
- **Leaf-Wise Growth**: Grows tree leaf-wise instead of level-wise
- **Faster Training**: Often 10x faster than XGBoost

**GOSS**: 
- Keep samples with large gradients
- Randomly sample from remaining samples
- Reduces computational cost while maintaining accuracy

**EFB**:
- Bundle mutually exclusive features
- Reduces number of features
- Speeds up training

### Comparison

| Feature | XGBoost | LightGBM |
|---------|---------|----------|
| Speed | Fast | Very Fast |
| Memory | Moderate | Low |
| Accuracy | High | High |
| Missing Values | Handled | Handled |
| Categorical | Needs encoding | Native support |

## Feature Importance

Understanding which features contribute most to predictions.

### Gini Importance

For each feature, sum Gini impurity reductions across all splits:

$$\text{Importance}_j = \frac{1}{B} \sum_{b=1}^B \sum_{t \in T_b: \text{split on } j} p(t) \Delta Gini(t)$$

where $p(t)$ is proportion of samples reaching node $t$.

### Permutation Importance

1. Train model on original data
2. For each feature $j$:
   - Permute values of feature $j$
   - Measure decrease in model performance
   - Larger decrease = higher importance

**Advantages**: Model-agnostic, accounts for interactions

### Mean Decrease in Accuracy

For Random Forests:
1. Evaluate model on OOB samples
2. For each feature, permute in OOB samples
3. Measure decrease in accuracy
4. Average across trees

### SHAP Values

SHAP (SHapley Additive exPlanations) provides unified feature attribution:

$$\phi_i = \sum_{S \subseteq \mathcal{F} \setminus \{i\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} [f(S \cup \{i\}) - f(S)]$$

Tree-specific algorithms (TreeSHAP) make this efficient for tree models.

## Advantages and Limitations

### Advantages

- **Interpretability**: Easy to visualize and understand
- **No Preprocessing**: Handles mixed data types, missing values
- **Non-linear**: Captures complex interactions
- **Feature Selection**: Automatically selects relevant features
- **Robust**: Random Forests handle noise well
- **Scalable**: Can handle large datasets

### Limitations

- **Overfitting**: Single trees prone to overfitting (mitigated by ensemble methods)
- **Instability**: Small data changes can alter tree structure
- **Extrapolation**: Poor at extrapolating beyond training range
- **Axis-Aligned**: Decision boundaries are axis-aligned
- **Memory**: Deep trees can use significant memory

### When to Use

**Use Decision Trees When**:
- Interpretability is important
- Mixed data types (numerical and categorical)
- Non-linear relationships expected
- Feature interactions are important

**Use Random Forests When**:
- Need robust predictions
- Have sufficient data
- Want feature importance
- Baseline for comparison

**Use Gradient Boosting When**:
- Maximum accuracy needed
- Can tune hyperparameters
- Have computational resources
- Competitions or production systems

## Key Takeaways

1. **Decision Trees** partition feature space using if-then rules, creating interpretable models that handle mixed data types without distributional assumptions.

2. **ID3** uses information gain, **C4.5** adds gain ratio and handles continuous features, and **CART** uses Gini impurity for binary splits.

3. **Splitting Criteria** include information gain (entropy reduction), Gini impurity, variance reduction (regression), and chi-square tests, with continuous features handled via threshold optimization.

4. **Pruning** prevents overfitting through pre-pruning (early stopping) or post-pruning (cost-complexity, reduced error), balancing model complexity and fit quality.

5. **Random Forests** combine bagging (bootstrap sampling) with random feature selection, reducing variance through averaging while maintaining low bias.

6. **Gradient Boosting** sequentially adds trees that correct previous errors, minimizing residuals through gradient descent in function space with shrinkage for regularization.

7. **XGBoost** adds L1/L2 regularization, approximate split finding, and parallelization, while **LightGBM** uses GOSS and EFB for faster training with similar accuracy.

8. **Feature Importance** can be measured via Gini importance (sum of impurity reductions), permutation importance (performance decrease), or SHAP values for unified attribution.

9. **Advantages** include interpretability, no preprocessing needs, non-linear capabilities, and automatic feature selection, while **limitations** include overfitting risk and axis-aligned boundaries.

10. **Tree-based methods** excel when interpretability matters, data is mixed-type, relationships are non-linear, and feature interactions are important, with Random Forests providing robust baselines and Gradient Boosting achieving high accuracy.
