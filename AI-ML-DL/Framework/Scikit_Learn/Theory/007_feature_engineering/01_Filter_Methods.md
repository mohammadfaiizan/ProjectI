# Filter Methods for Feature Selection

---

## Table of Contents

- [Overview](#overview)
- [VarianceThreshold](#variancethreshold)
- [Univariate Selection](#univariate-selection)
- [Mutual Information](#mutual-information)
- [Chi-Squared Selection](#chi-squared-selection)
- [Comparison and When to Use](#comparison-and-when-to-use)

---

## Overview

**Filter methods** rank or filter features based on intrinsic properties of the data, independent of any learning algorithm. They are fast, scalable, and suitable as a first pass before model training. Scikit-learn provides several filter-based selectors.

| Method | Score Function | Use Case |
|--------|----------------|----------|
| VarianceThreshold | Variance | Remove constant/low-variance features |
| SelectKBest (f_classif) | F-statistic | Classification, numeric features |
| SelectKBest (f_regression) | F-statistic | Regression, numeric features |
| SelectKBest (chi2) | Chi-squared | Categorical/non-negative features |
| mutual_info_classif | Mutual information | Classification, any feature type |
| mutual_info_regression | Mutual information | Regression |

---

## VarianceThreshold

**VarianceThreshold** removes features whose variance does not exceed a threshold. Features with zero or near-zero variance provide no discriminative information and can cause numerical issues.

### Parameters

- **threshold**: Minimum variance to retain (default 0.0). Features with variance <= threshold are removed.

### Key Attributes

- **variances_**: Variance of each feature (computed during fit)
- **get_support()**: Boolean mask of selected features

### Usage

```python
from sklearn.feature_selection import VarianceThreshold

X = [[1, 2, 0], [1, 2, 0], [1, 2, 0]]
selector = VarianceThreshold(threshold=0.0)
X_selected = selector.fit_transform(X)
# Removes constant column (index 2)
```

### Important Notes

- **threshold=0.0** removes only constant features
- Higher thresholds remove features with little variation
- Does not consider relationship with target; use for preprocessing only

---

## Univariate Selection

**Univariate selection** evaluates each feature independently against the target using statistical tests. **SelectKBest** and **SelectPercentile** wrap score functions to select top features.

### SelectKBest

Selects the k highest-scoring features.

```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Classification: F-test (ANOVA)
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Regression: F-test (linear correlation)
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
```

### SelectPercentile

Selects the top percentile of features by score.

```python
from sklearn.feature_selection import SelectPercentile

selector = SelectPercentile(score_func=f_classif, percentile=25)
X_selected = selector.fit_transform(X, y)
```

### f_classif and f_regression

- **f_classif**: F-value from one-way ANOVA. Measures how well a feature separates classes. Assumes linear relationship.
- **f_regression**: F-value from correlation with target. For regression tasks.

### Key Attributes

- **scores_**: Score for each feature
- **pvalues_**: P-value for each feature (when available from score function)

### Important Notes

- Univariate methods ignore feature interactions
- **f_classif** and **f_regression** assume linear relationships
- Good for initial screening; may miss non-linear dependencies

---

## Mutual Information

**Mutual information** measures the dependency between each feature and the target, capturing both linear and non-linear relationships. It is model-free and works with mixed data types.

### mutual_info_classif

For classification targets.

```python
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from functools import partial

mi_scores = mutual_info_classif(X, y, random_state=42)
selector = SelectKBest(partial(mutual_info_classif, random_state=42), k=5)
X_selected = selector.fit_transform(X, y)
```

### mutual_info_regression

For regression targets.

```python
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X, y, random_state=42)
```

### Parameters

- **discrete_features**: Boolean mask or array of indices. Indicates which features are discrete (e.g., categorical).
- **n_neighbors**: Number of neighbors for entropy estimation (default 5). Higher values smooth estimates but increase computation.

### Important Notes

- **Mutual information** is non-negative; zero means independence
- Handles non-linear relationships better than F-tests
- **discrete_features** improves accuracy for categorical inputs
- More computationally expensive than F-tests

---

## Chi-Squared Selection

**Chi-squared (chi2)** tests the independence between each feature and the target. Suitable for non-negative features (e.g., counts, binned continuous data).

### Requirements

- Features must be **non-negative**
- Typically used with count data or discretized continuous features

### Usage

```python
from sklearn.feature_selection import chi2, SelectKBest

# X must be non-negative (e.g., binned or count data)
scores, pvalues = chi2(X_nonneg, y)
selector = SelectKBest(score_func=chi2, k=5)
X_selected = selector.fit_transform(X_nonneg, y)
```

### When to Use

- Categorical features (after encoding to counts or one-hot)
- Binned continuous features
- Text features (bag-of-words counts)

### Important Notes

- **chi2** assumes non-negative inputs; negative values cause errors
- For continuous data, bin first (e.g., **KBinsDiscretizer**) or use **MinMaxScaler** and scale to non-negative range
- Good for categorical vs categorical independence tests

---

## Comparison and When to Use

| Scenario | Recommended Method |
|----------|-------------------|
| Remove constant/redundant features | VarianceThreshold |
| Classification, numeric, linear relationship | SelectKBest + f_classif |
| Regression, numeric, linear relationship | SelectKBest + f_regression |
| Non-linear or mixed relationships | mutual_info_classif / mutual_info_regression |
| Categorical or count data | SelectKBest + chi2 |

### Workflow

1. Apply **VarianceThreshold** to remove constant features
2. Use **SelectKBest** with appropriate score function (f_classif, f_regression, chi2, or mutual_info) for initial selection
3. Consider **SelectPercentile** when you want to keep a proportion of features rather than a fixed count
