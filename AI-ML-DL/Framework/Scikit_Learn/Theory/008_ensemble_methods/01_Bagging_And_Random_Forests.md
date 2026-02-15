# Bagging and Random Forests

---

## Table of Contents

- [Overview](#overview)
- [Bootstrap Aggregating (Bagging)](#bootstrap-aggregating-bagging)
- [Random Forest Classifier](#random-forest-classifier)
- [Random Forest Regressor](#random-forest-regressor)
- [Extra Trees](#extra-trees)
- [Key Parameters](#key-parameters)
- [Out-of-Bag Score](#out-of-bag-score)
- [When to Use Each Method](#when-to-use-each-method)

---

## Overview

**Ensemble methods** combine multiple base learners to improve predictive performance. **Bagging** (Bootstrap Aggregating) trains many models on bootstrap samples and averages their predictions. **Random Forest** and **Extra Trees** are bagging variants that use decision trees with additional randomization.

---

## Bootstrap Aggregating (Bagging)

**Bagging** reduces variance by training multiple models on different bootstrap samples of the training data. Each bootstrap sample is drawn with replacement and typically has the same size as the original dataset.

### Algorithm

1. For each base learner: sample n examples with replacement from the training set
2. Train the base learner on the bootstrap sample
3. For prediction: average (regression) or majority vote (classification)

### BaggingClassifier and BaggingRegressor

**BaggingClassifier** and **BaggingRegressor** wrap any base estimator and apply bagging. The default base estimator is a decision tree.

| Parameter | Description |
|-----------|-------------|
| **estimator** | Base learner (default: DecisionTreeClassifier/Regressor) |
| **n_estimators** | Number of bootstrap samples and models |
| **max_samples** | Fraction or count of samples per bootstrap (default: 1.0) |
| **max_features** | Fraction or count of features per model (default: 1.0) |
| **bootstrap** | Sample with replacement (default: True) |
| **bootstrap_features** | Bootstrap features as well (default: False) |

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)
bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
```

### Why Bagging Works

- **Variance reduction**: Averaging uncorrelated models reduces variance
- **Bias unchanged**: Each model has similar bias; averaging does not change it
- **Overfitting mitigation**: Bootstrap sampling creates diversity; no single model sees all data

---

## Random Forest Classifier

**RandomForestClassifier** is a bagging ensemble of decision trees with two sources of randomness:

1. **Bootstrap sampling**: Each tree is trained on a bootstrap sample
2. **Feature subsampling**: At each split, only a random subset of features is considered

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_estimators** | Number of trees in the forest |
| **max_depth** | Maximum depth of each tree (None = fully grown) |
| **max_features** | Features to consider per split: `sqrt`, `log2`, int, or float |
| **min_samples_split** | Minimum samples required to split a node |
| **min_samples_leaf** | Minimum samples in each leaf |
| **bootstrap** | Use bootstrap samples (default: True) |

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)
print(rf.feature_importances_)
```

### feature_importances_

**RandomForestClassifier** provides **feature_importances_** based on the total reduction in impurity (Gini or entropy) achieved by splits on each feature, averaged over all trees. Values sum to 1.0.

---

## Random Forest Regressor

**RandomForestRegressor** applies the same bagging and feature subsampling logic to regression. Predictions are the mean of all tree outputs.

| Parameter | Description |
|-----------|-------------|
| **n_estimators** | Number of trees |
| **max_depth** | Maximum tree depth |
| **max_features** | Features per split |
| **oob_score** | Compute out-of-bag score (default: False) |

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB R2: {rf.oob_score_:.4f}")
```

---

## Extra Trees

**ExtraTreesClassifier** and **ExtraTreesRegressor** (Extremely Randomized Trees) add a third source of randomness:

- **Random split thresholds**: Instead of choosing the best split, a random threshold is chosen for each candidate feature

### Random Forest vs Extra Trees

| Aspect | Random Forest | Extra Trees |
|--------|---------------|-------------|
| **Split selection** | Best split among candidates | Random split threshold |
| **Variance** | Lower than single tree | Often lower than RF |
| **Bias** | Similar to RF | Slightly higher |
| **Speed** | Slower (optimization) | Faster (no optimization) |

```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

et_clf = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42)
et_reg = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)
```

---

## Key Parameters

### n_estimators

More trees generally improve performance but increase training time. Diminishing returns after 100–200 trees for many datasets.

### max_depth

Limiting depth reduces overfitting. Typical values: 5–20. Use cross-validation to tune.

### max_features

- **sqrt**: Use sqrt(n_features) per split (common for classification)
- **log2**: Use log2(n_features) per split
- **float**: Fraction of features (e.g., 0.5 = 50%)
- **int**: Exact number of features

Smaller values increase diversity but may increase bias.

### bootstrap

When **bootstrap=True** (default), each tree is trained on a bootstrap sample. When False, each tree sees the full dataset (like pasting).

---

## Out-of-Bag Score

When **bootstrap=True**, each sample is left out of approximately 37% of the bootstrap samples. These **out-of-bag (OOB)** samples can be used for validation without a separate holdout set.

Set **oob_score=True** to compute the OOB score. For **RandomForestRegressor**, this is R2; for **RandomForestClassifier**, it is accuracy.

```python
rf = RandomForestRegressor(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
print(rf.oob_score_)
```

**Note**: OOB requires **bootstrap=True**. **RandomForestClassifier** and **RandomForestRegressor** support **oob_score**. **BaggingClassifier** and **BaggingRegressor** also support **oob_score=True**.

---

## When to Use Each Method

| Method | Use Case |
|--------|----------|
| **BaggingClassifier/Regressor** | Custom base estimators; need flexibility |
| **RandomForest** | Default choice for tabular data; robust, interpretable |
| **ExtraTrees** | Faster training; slightly different bias-variance tradeoff |

---

## Summary

- **Bagging** reduces variance by averaging over bootstrap-trained models
- **Random Forest** = bagging + feature subsampling at each split
- **Extra Trees** = Random Forest + random split thresholds
- Use **max_features**, **max_depth**, and **n_estimators** to control bias-variance
- **oob_score** provides validation without extra data
- **feature_importances_** supports interpretability
