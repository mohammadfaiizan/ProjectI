# Specialized Regression

---

## Table of Contents

- [Bayesian Regression](#bayesian-regression)
- [Robust Regressors](#robust-regressors)
- [Quantile Regression](#quantile-regression)
- [PLS Regression](#pls-regression)
- [Isotonic Regression](#isotonic-regression)

---

## Bayesian Regression

**BayesianRidge** and **ARDRegression** provide probabilistic estimates with automatic relevance determination.

### BayesianRidge

- Estimates **posterior** over coefficients
- **return_std=True** in predict() yields uncertainty

```python
from sklearn.linear_model import BayesianRidge

br = BayesianRidge()
br.fit(X_train, y_train)
y_pred, y_std = br.predict(X_test, return_std=True)
```

### ARDRegression

- **Automatic Relevance Determination**: per-feature precision
- Drives irrelevant features to zero

```python
from sklearn.linear_model import ARDRegression

ard = ARDRegression()
ard.fit(X_train, y_train)
print(ard.lambda_)  # Per-feature precision
```

---

## Robust Regressors

Models resistant to **outliers**:

| Model | Strategy |
|-------|----------|
| **HuberRegressor** | Quadratic loss near zero, linear far away |
| **RANSACRegressor** | Random sample consensus; fits on inliers |
| **TheilSenRegressor** | Median of slopes; high breakdown point |

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

huber = HuberRegressor(epsilon=1.35)
ransac = RANSACRegressor()
theil = TheilSenRegressor()
```

---

## Quantile Regression

**QuantileRegressor** predicts conditional quantiles (e.g., median, 90th percentile).

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **quantile** | Target quantile (0.5 = median) |
| **alpha** | L2 regularization |

### Prediction Intervals

```python
from sklearn.linear_model import QuantileRegressor

qr_low = QuantileRegressor(quantile=0.1, alpha=0.5)
qr_high = QuantileRegressor(quantile=0.9, alpha=0.5)
qr_low.fit(X_train, y_train)
qr_high.fit(X_train, y_train)
lower = qr_low.predict(X_test)
upper = qr_high.predict(X_test)
```

---

## PLS Regression

**Partial Least Squares** finds latent components that maximize covariance between X and y.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_components** | Number of latent components |

```python
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=3)
pls.fit(X_train, y_train)
X_latent = pls.transform(X_train)
```

---

## Isotonic Regression

**IsotonicRegression** fits a non-decreasing (or non-increasing) step function.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **increasing** | `True`, `False`, or `auto` |
| **out_of_bounds** | `nan`, `clip` |

```python
from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression(increasing=True)
iso.fit(X_train.ravel(), y_train)
y_pred = iso.predict(X_test.ravel())
```
