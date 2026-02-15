# Linear and Regularized Regression

---

## Table of Contents

- [Overview](#overview)
- [Linear Regression](#linear-regression)
- [Ridge Regression](#ridge-regression)
- [Lasso Regression](#lasso-regression)
- [Elastic Net](#elastic-net)
- [SGD Regressor](#sgd-regressor)

---

## Overview

Linear models assume a linear relationship between features and target. **Regularization** penalizes model complexity to reduce overfitting.

---

## Linear Regression

**Ordinary Least Squares (OLS)** minimizes the sum of squared residuals.

### Key Concepts

- **fit**: Estimates coefficients via closed-form solution
- **coef_**: Slope for each feature
- **intercept_**: Bias term
- **score**: Returns R² (coefficient of determination)

### Formula

\[
\hat{y} = \beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p
\]

### Code Example

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.coef_, model.intercept_)
print(model.score(X_test, y_test))
```

---

## Ridge Regression

**Ridge** adds L2 penalty (sum of squared coefficients) to OLS.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **alpha** | Regularization strength (higher = more shrinkage) |
| **solver** | `cholesky`, `svd`, `lsqr`, `sag`, `saga` |

### RidgeCV

Uses cross-validation to select optimal **alpha**:

```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10], cv=5)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.alpha_)
```

---

## Lasso Regression

**Lasso** adds L1 penalty (sum of absolute coefficients), enabling **feature selection** by driving some coefficients to zero.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **alpha** | Regularization strength |

### Feature Selection via coef_

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
# Zero coefficients indicate excluded features
selected = np.abs(lasso.coef_) > 1e-5
```

### LassoCV

```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
```

---

## Elastic Net

**Elastic Net** combines L1 and L2 penalties.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **alpha** | Overall regularization strength |
| **l1_ratio** | L1/(L1+L2); 1=Lasso, 0=Ridge |

```python
from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=0.5, l1_ratio=0.5)
en.fit(X_train, y_train)
```

---

## SGD Regressor

**SGDRegressor** uses Stochastic Gradient Descent for scalable, incremental learning.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **loss** | `squared_error`, `huber`, `epsilon_insensitive` |
| **penalty** | `l2`, `l1`, `elasticnet` |
| **learning_rate** | `constant`, `invscaling`, `adaptive` |

### partial_fit for Incremental Learning

```python
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(max_iter=1)
for X_batch, y_batch in batches:
    sgd.partial_fit(X_batch, y_batch)
```
