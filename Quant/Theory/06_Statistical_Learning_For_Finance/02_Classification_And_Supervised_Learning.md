# Classification and Supervised Learning

## Logistic Regression

Logistic regression models the probability of a binary outcome.

### Model

$$P(Y = 1 | \mathbf{X}) = \frac{e^{\boldsymbol{\beta}^T\mathbf{X}}}{1 + e^{\boldsymbol{\beta}^T\mathbf{X}}} = \frac{1}{1 + e^{-\boldsymbol{\beta}^T\mathbf{X}}}$$

**Logit transformation:**
$$\text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = \boldsymbol{\beta}^T\mathbf{X}$$

**Interpretation:** Log-odds are linear in predictors.

### Estimation

**Maximum likelihood:**
$$L(\boldsymbol{\beta}) = \prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i}$$

$$\ln L(\boldsymbol{\beta}) = \sum_{i=1}^{n}[y_i\ln p_i + (1-y_i)\ln(1-p_i)]$$

**Optimization:** No closed form, use iterative methods (Newton-Raphson, gradient descent).

### Coefficients

**Odds ratio:**
$$OR_j = e^{\beta_j}$$

Interpretation: One-unit increase in $X_j$ multiplies odds by $e^{\beta_j}$.

**Marginal effect:**
$$\frac{\partial P(Y=1|\mathbf{X})}{\partial X_j} = \beta_j p(1-p)$$

Depends on current probability level.

### Applications: Default Prediction

**Model:** Predict probability of default:
$$P(\text{Default} = 1 | \mathbf{X}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \text{DebtRatio} + \beta_2 \text{Income} + \cdots)}}$$

**Features:**
- Financial ratios
- Credit history
- Macro variables

**Use:** Credit scoring, loan approval.

## Decision Trees

Decision trees partition the feature space using simple rules.

### Construction

**Algorithm (CART):**
1. Start with all data
2. For each feature and threshold, compute split criterion
3. Choose best split (maximize information gain)
4. Recursively split child nodes
5. Stop when stopping criterion met (max depth, min samples)

### Splitting Criteria

**Gini impurity:**
$$Gini = 1 - \sum_{k=1}^{K}p_k^2$$

where $p_k$ is proportion of class $k$ in node.

**Entropy:**
$$Entropy = -\sum_{k=1}^{K}p_k\ln p_k$$

**Information gain:**
$$IG = Entropy_{parent} - \sum_{child}\frac{n_{child}}{n_{parent}}Entropy_{child}$$

**Classification error:**
$$Error = 1 - \max_k p_k$$

### Pruning

**Pre-pruning:** Stop splitting early (max depth, min samples per leaf).

**Post-pruning:** Build full tree, then remove branches that don't improve validation performance.

**Cost-complexity pruning:** Minimize:
$$R_\alpha(T) = R(T) + \alpha|T|$$

where $R(T)$ is misclassification rate and $|T|$ is number of leaves.

### Advantages and Disadvantages

**Advantages:**
- Interpretable
- Handles non-linearity
- No distributional assumptions
- Handles mixed data types

**Disadvantages:**
- Unstable (small data changes → different tree)
- Prone to overfitting
- Greedy (may miss global optimum)

## Random Forests

Random forests combine many decision trees via bagging.

### Algorithm

1. **Bootstrap:** Sample $n$ observations with replacement
2. **Train tree:** On bootstrap sample, using random subset of features at each split
3. **Repeat:** Build $B$ trees
4. **Predict:** Average predictions (classification: majority vote)

### Key Features

**Bootstrap aggregation (bagging):**
- Reduces variance
- Trees trained on different data

**Random feature selection:**
- At each split, consider only $m$ randomly chosen features
- Typical: $m = \sqrt{p}$ for classification, $m = p/3$ for regression
- Reduces correlation between trees

### Out-of-Bag (OOB) Error

For each observation, predict using trees that didn't include it in bootstrap sample.

**OOB error:** Misclassification rate on OOB predictions.

**Advantage:** Built-in validation, no need for separate test set.

### Variable Importance

**Mean decrease in impurity:**
- Sum of Gini decreases over all splits using feature
- Average across trees

**Permutation importance:**
- Shuffle feature values
- Measure increase in error
- Larger increase → more important

### Hyperparameters

- **$B$:** Number of trees (more is better, but diminishing returns)
- **$m$:** Features per split (tune via cross-validation)
- **Max depth:** Control tree complexity
- **Min samples per leaf:** Prevent overfitting

## Gradient Boosting

Gradient boosting builds an ensemble by sequentially adding trees that correct previous errors.

### Algorithm

**Initialize:**
$$F_0(\mathbf{x}) = \arg\min_\gamma \sum_{i=1}^{n}L(y_i, \gamma)$$

**For $m = 1$ to $M$:**

1. **Compute residuals:**
   $$r_{im} = -\left[\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}\right]$$

2. **Fit tree to residuals:**
   $$h_m(\mathbf{x}) = \arg\min_h \sum_{i=1}^{n}(r_{im} - h(\mathbf{x}_i))^2$$

3. **Update:**
   $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \gamma_m h_m(\mathbf{x})$$

   where $\gamma_m$ minimizes loss and $\nu$ is learning rate.

**Output:** $F_M(\mathbf{x})$

### XGBoost

XGBoost (Extreme Gradient Boosting) adds regularization:

**Objective:**
$$\mathcal{L}^{(t)} = \sum_{i=1}^{n}L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2$ penalizes tree complexity.

**Features:**
- **Regularization:** Prevents overfitting
- **Handles missing data:** Learns default direction
- **Parallelization:** Faster training
- **Feature importance:** Built-in

**Hyperparameters:**
- **Learning rate ($\nu$):** Smaller = more trees needed, better generalization
- **Max depth:** Tree complexity
- **Subsample:** Fraction of data per tree
- **Colsample:** Fraction of features per tree
- **$\lambda$, $\gamma$:** Regularization parameters

### LightGBM

LightGBM uses leaf-wise growth instead of level-wise:

**Advantages:**
- Faster training
- Lower memory usage
- Often better accuracy

**Features:**
- **Histogram binning:** Discretize features for speed
- **Gradient-based one-side sampling:** Focus on large gradients
- **Exclusive feature bundling:** Combine sparse features

### Applications: Financial Prediction

**Return prediction:**
- Features: Past returns, volume, technical indicators
- Target: Next period return (binary: up/down)

**Default prediction:**
- Features: Financial ratios, credit history
- Target: Default indicator

**Signal generation:**
- Predict probability of favorable outcome
- Rank assets by predicted probability

## Support Vector Machines (SVM)

SVM finds the optimal separating hyperplane with maximum margin.

### Linear SVM

**Hard margin:** (linearly separable data)

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$

subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ for all $i$.

**Soft margin:** (allows misclassification)

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i$$

subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$.

**$C$:** Controls trade-off between margin and misclassification.

### Kernel Trick

For non-linear boundaries, map to higher dimensions:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$$

**Common kernels:**
- **Polynomial:** $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T\mathbf{x}_j + 1)^d$
- **RBF:** $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$
- **Sigmoid:** $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\kappa\mathbf{x}_i^T\mathbf{x}_j + c)$

**Advantage:** Compute in original space, not high-dimensional space.

### Applications: Classification

**Stock direction:** Predict up/down movement
**Sector classification:** Classify stocks into sectors
**Regime detection:** Identify market regimes

## Model Evaluation

### Confusion Matrix

For binary classification:

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

**Metrics:**
- **Accuracy:** $(TP + TN)/(TP + TN + FP + FN)$
- **Precision:** $TP/(TP + FP)$
- **Recall (Sensitivity):** $TP/(TP + FN)$
- **Specificity:** $TN/(TN + FP)$
- **F1-score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

### ROC Curve

Plot True Positive Rate (TPR) vs False Positive Rate (FPR) for different thresholds.

**AUC (Area Under Curve):**
- **AUC = 1:** Perfect classifier
- **AUC = 0.5:** Random classifier
- **AUC > 0.5:** Better than random

**Interpretation:** Probability that classifier ranks random positive higher than random negative.

### Precision-Recall Curve

Plot Precision vs Recall for different thresholds.

**Use:** When classes are imbalanced (more informative than ROC).

**AUC-PR:** Area under precision-recall curve.

### Brier Score

For probabilistic predictions:

$$BS = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$$

where $p_i$ is predicted probability and $y_i$ is actual outcome.

**Lower is better.** Measures calibration of probabilities.

### Cross-Validation

**k-fold CV:**
1. Split data into $k$ folds
2. Train on $k-1$ folds, validate on remaining fold
3. Repeat for each fold
4. Average performance

**Stratified k-fold:** Maintains class proportions in each fold.

**Time series:** Use forward chaining (train on past, test on future).

## Handling Class Imbalance

Financial data often has imbalanced classes (e.g., defaults are rare).

### Methods

**Oversampling:** Duplicate minority class examples (SMOTE: create synthetic examples)

**Undersampling:** Remove majority class examples

**Class weights:** Weight loss function by inverse class frequency

**Threshold tuning:** Adjust classification threshold (not always 0.5)

**Cost-sensitive learning:** Penalize misclassifying minority class more

### Evaluation

Use metrics robust to imbalance:
- **Precision-Recall curve** (not just ROC)
- **F1-score**
- **Matthews Correlation Coefficient (MCC)**

Avoid accuracy when classes are imbalanced.

## Practical Considerations

### Feature Engineering

- **Domain knowledge:** Use financial expertise
- **Interaction terms:** Capture non-linearities
- **Time-based features:** Lags, moving averages
- **Normalization:** Scale features appropriately

### Overfitting

- **Regularization:** L1/L2 penalties
- **Early stopping:** Stop training when validation error increases
- **Cross-validation:** Tune hyperparameters
- **Ensemble methods:** Reduce variance

### Interpretability

- **Feature importance:** Understand what drives predictions
- **SHAP values:** Explain individual predictions
- **Partial dependence plots:** Visualize feature effects
- **Model-agnostic methods:** LIME, SHAP

### Production Considerations

- **Latency:** Fast prediction for real-time use
- **Model monitoring:** Track performance over time
- **Retraining:** Update model periodically
- **A/B testing:** Compare model versions
