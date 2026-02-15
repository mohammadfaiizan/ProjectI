# Polynomial Regression and Model Comparison

---

## Table of Contents

- [Polynomial Regression](#polynomial-regression)
- [Model Comparison](#model-comparison)

---

## Polynomial Regression

**Polynomial regression** models non-linear relationships by adding polynomial features to a linear model.

### PolynomialFeatures

Expands features to degree d:

- degree=2: \(x_1, x_2\) → \(1, x_1, x_2, x_1^2, x_1 x_2, x_2^2\)

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **degree** | Polynomial degree |
| **include_bias** | Include intercept column |
| **interaction_only** | Only interaction terms, no \(x^2\) |

### Pipeline Example

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### Overfitting

Higher degree increases flexibility but risks overfitting. Use **cross-validation** to tune degree.

---

## Model Comparison

### Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root MSE |
| **R²** | Coefficient of determination |
| **MAE** | Mean Absolute Error |

### Comparison Workflow

```python
from sklearn.model_selection import cross_val_score

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'SVR': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=10),
    'DecisionTree': DecisionTreeRegressor(max_depth=10),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name}: CV MSE = {-scores.mean():.4f}")
```

### Considerations

| Factor | Notes |
|--------|-------|
| **Scalability** | SGD, linear models scale well; KNN does not |
| **Interpretability** | Linear, tree models more interpretable |
| **Outliers** | Use robust regressors |
| **Non-linearity** | SVR, KNN, trees, polynomial |
