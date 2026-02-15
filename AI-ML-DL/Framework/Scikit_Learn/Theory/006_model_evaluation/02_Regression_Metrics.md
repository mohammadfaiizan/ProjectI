# Regression Metrics

---

## Table of Contents

- [Overview](#overview)
- [Mean Squared Error](#mean-squared-error)
- [Mean Absolute Error](#mean-absolute-error)
- [R2 Score](#r2-score)
- [Mean Absolute Percentage Error](#mean-absolute-percentage-error)
- [Explained Variance Score](#explained-variance-score)
- [Multi-output Metrics](#multi-output-metrics)

---

## Overview

Regression metrics measure how well a model predicts continuous targets. Different metrics emphasize different aspects: **MSE** penalizes large errors, **MAE** is scale-invariant to outliers, **R2** explains variance, and **MAPE** is relative.

---

## Mean Squared Error

### mean_squared_error

**MSE** is the average of squared differences between predictions and actual values.

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- Penalizes large errors more (squared term)
- Same units as squared target (e.g., dollars squared)
- Sensitive to outliers

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
```

### RMSE (Root Mean Squared Error)

\[
\text{RMSE} = \sqrt{\text{MSE}}
\]

Use `squared=False` to get RMSE directly:

```python
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

---

## Mean Absolute Error

### mean_absolute_error

**MAE** is the average of absolute differences.

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

- Same units as target
- Less sensitive to outliers than MSE
- Not differentiable at zero (for optimization)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

---

## R2 Score

### r2_score

**R2** (coefficient of determination) measures the proportion of variance in the target explained by the model.

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

- R2 = 1: perfect predictions
- R2 = 0: model predicts mean
- R2 < 0: worse than predicting mean

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

### multioutput Parameter

For multi-output regression:

| Value | Description |
|-------|-------------|
| **raw_values** | R2 per output |
| **variance_weighted** | Weight by variance of each output |
| **uniform_average** | Simple mean of per-output R2 |

```python
r2_score(y_true, y_pred, multioutput='variance_weighted')
```

---

## Mean Absolute Percentage Error

### mean_absolute_percentage_error

**MAPE** is the average of absolute percentage errors.

\[
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
\]

Scikit-learn returns MAPE as a decimal (multiply by 100 for percentage). Undefined when \(y_i = 0\); use with care for targets that can be zero.

```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred)
```

---

## Explained Variance Score

### explained_variance_score

**Explained variance** measures the proportion of variance explained, ignoring scale.

\[
\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
\]

- Similar to R2 but uses variance of errors
- Bounded by 1 (perfect), can be negative
- Less sensitive to bias than R2

```python
from sklearn.metrics import explained_variance_score

evs = explained_variance_score(y_true, y_pred)
```

---

## Multi-output Metrics

For targets with multiple outputs (e.g., predicting multiple variables), most metrics accept a 2D array and support **multioutput**:

```python
# y_true, y_pred shape: (n_samples, n_outputs)
mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
```

### Summary Table

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MSE** | \(\frac{1}{n}\sum(y - \hat{y})^2\) | Default, penalizes large errors |
| **RMSE** | \(\sqrt{\text{MSE}}\) | Same units as target |
| **MAE** | \(\frac{1}{n}\sum|y - \hat{y}|\) | Robust to outliers |
| **R2** | \(1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}\) | Variance explained |
| **MAPE** | \(\frac{100}{n}\sum|\frac{y - \hat{y}}{y}|\) | Relative error |
| **Explained Var** | \(1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}\) | Scale-invariant |

---
