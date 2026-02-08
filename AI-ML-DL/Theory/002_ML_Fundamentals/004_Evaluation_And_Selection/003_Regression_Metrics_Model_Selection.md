# Regression Metrics Model Selection

## Table of Contents

1. [Introduction to Regression Metrics](#introduction-to-regression-metrics)
2. [Mean Squared Error](#mean-squared-error)
3. [Mean Absolute Error](#mean-absolute-error)
4. [Root Mean Squared Error](#root-mean-squared-error)
5. [R-Squared and Adjusted R-Squared](#r-squared-and-adjusted-r-squared)
6. [Information Criteria](#information-criteria)
7. [Residual Analysis](#residual-analysis)
8. [Model Selection Strategies](#model-selection-strategies)
9. [Cross-Validation for Regression](#cross-validation-for-regression)
10. [Key Takeaways](#key-takeaways)

## Introduction to Regression Metrics

Regression metrics quantify how well a model predicts continuous target values, measuring the discrepancy between predictions and actual values.

### Why Metrics Matter

- **Model Evaluation**: Assess prediction accuracy
- **Model Comparison**: Compare different models
- **Hyperparameter Tuning**: Optimize model parameters
- **Feature Selection**: Identify important features
- **Business Impact**: Translate errors to business costs

### Types of Metrics

- **Scale-Dependent**: Depends on target scale (MSE, MAE, RMSE)
- **Scale-Independent**: Normalized metrics (R², MAPE)
- **Residual-Based**: Analyze prediction errors
- **Information-Theoretic**: Balance fit and complexity

### Characteristics of Good Metrics

- **Interpretable**: Easy to understand and explain
- **Robust**: Not overly sensitive to outliers
- **Scale-Appropriate**: Match problem requirements
- **Differentiable**: For optimization (if needed)

## Mean Squared Error

MSE is the most common regression metric, penalizing large errors more than small ones.

### Definition

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

where:
- $y_i$: Actual value
- $\hat{y}_i$: Predicted value
- $n$: Number of samples

### Properties

- **Non-Negative**: $\text{MSE} \geq 0$, equals 0 only for perfect predictions
- **Squared Units**: Units are squared (e.g., if target is in dollars, MSE is in dollars²)
- **Differentiable**: Smooth, suitable for optimization
- **Sensitive to Outliers**: Large errors contribute disproportionately

### Why Squared?

**Mathematical Convenience**:
- Differentiable everywhere
- Leads to closed-form solutions (e.g., OLS)
- Connects to maximum likelihood (Gaussian assumption)

**Penalizes Large Errors**: 
- Quadratic penalty emphasizes large deviations
- Encourages model to avoid large mistakes

### Limitations

- **Scale-Dependent**: Cannot compare across different scales
- **Outlier Sensitivity**: Single outlier can dominate MSE
- **Units**: Squared units less interpretable

### Use Cases

- Optimization objective (gradient descent)
- Maximum likelihood estimation
- When large errors are particularly costly
- Theoretical analysis

## Mean Absolute Error

MAE measures average absolute deviation, treating all errors equally.

### Definition

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

### Properties

- **Same Units**: Same units as target (interpretable)
- **Robust**: Less sensitive to outliers than MSE
- **Linear Penalty**: All errors weighted equally
- **Non-Differentiable**: Not differentiable at zero (but subgradient exists)

### Comparison with MSE

**MSE vs. MAE**:
- **MSE**: Emphasizes large errors (quadratic)
- **MAE**: Treats all errors equally (linear)

**Example**: Errors $[1, 1, 10]$
- MAE = $(1 + 1 + 10)/3 = 4$
- MSE = $(1 + 1 + 100)/3 = 34$
- MSE dominated by large error

### When to Use MAE

- Interpretability important (same units as target)
- Outliers present (robust)
- All errors equally costly
- Median regression (MAE minimizes median error)

### Median Absolute Error (MedAE)

$$\text{MedAE} = \text{median}(|y_i - \hat{y}_i|)$$

Even more robust to outliers than MAE.

## Root Mean Squared Error

RMSE is the square root of MSE, providing interpretable units.

### Definition

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

### Properties

- **Same Units**: Same units as target (like MAE)
- **Interpretable**: "Average error magnitude"
- **Sensitive to Outliers**: Inherits from MSE
- **Differentiable**: Square root preserves differentiability (except at 0)

### Interpretation

RMSE can be interpreted as:
- "On average, predictions are off by RMSE units"
- Standard deviation of residuals
- Typical prediction error magnitude

### Comparison

**RMSE vs. MAE**:
- **RMSE**: Penalizes large errors more (like MSE)
- **MAE**: Treats all errors equally

**RMSE ≥ MAE**: Always (by Jensen's inequality)

**Difference**: Larger difference indicates more variable errors (outliers)

### When to Use RMSE

- Need interpretable units
- Want to penalize large errors
- Standard metric in many domains
- Comparing models on same scale

## R-Squared and Adjusted R-Squared

R² measures proportion of variance explained by the model.

### R-Squared (Coefficient of Determination)

$$\text{R}^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

where:
- $\text{SS}_{\text{res}}$: Sum of squared residuals
- $\text{SS}_{\text{tot}}$: Total sum of squares
- $\bar{y}$: Mean of target values

### Interpretation

- **R² = 1**: Perfect predictions (all variance explained)
- **R² = 0**: Model performs as well as predicting mean
- **R² < 0**: Model worse than mean prediction

**Range**: $(-\infty, 1]$ (can be negative for poor models)

### Properties

- **Scale-Independent**: Normalized metric
- **Proportion**: Fraction of variance explained
- **Comparable**: Can compare across different problems (with caution)

### Limitations

- **Increases with Features**: Adding features always increases R² (even if irrelevant)
- **No Penalty for Complexity**: Doesn't account for overfitting
- **Can be Misleading**: High R² doesn't guarantee good predictions

### Adjusted R-Squared

Penalizes for number of features:

$$\text{Adjusted R}^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

where $p$ is number of features (excluding intercept).

### Properties

- **Penalizes Complexity**: Decreases when adding irrelevant features
- **Comparable**: Can compare models with different numbers of features
- **Always ≤ R²**: Adjusted R² ≤ R²

**Interpretation**: Proportion of variance explained, adjusted for model complexity

### When to Use

- **R²**: Quick assessment, comparing models with same features
- **Adjusted R²**: Comparing models with different numbers of features
- **Both**: Provide complementary information

## Information Criteria

Information criteria balance model fit and complexity.

### Akaike Information Criterion (AIC)

$$\text{AIC} = 2p - 2\ln(L)$$

where:
- $p$: Number of parameters
- $L$: Maximum likelihood value

**For Linear Regression** (assuming Gaussian errors):
$$\text{AIC} = n\ln(\text{MSE}) + 2p$$

### Interpretation

- **Lower is Better**: Penalizes both poor fit and high complexity
- **Tradeoff**: Balance between fit ($-\ln(L)$) and complexity ($2p$)
- **Relative**: Compare models (absolute value less meaningful)

### Bayesian Information Criterion (BIC)

$$\text{BIC} = p\ln(n) - 2\ln(L)$$

**For Linear Regression**:
$$\text{BIC} = n\ln(\text{MSE}) + p\ln(n)$$

### Comparison

**AIC vs. BIC**:
- **BIC**: Stronger penalty for complexity ($\ln(n)$ vs. $2$)
- **BIC**: More likely to select simpler models
- **AIC**: Asymptotically efficient (minimizes prediction error)
- **BIC**: Consistent (selects true model if it exists, as $n \to \infty$)

### When to Use

- **AIC**: Prediction-focused, larger sample sizes
- **BIC**: Model selection, smaller sample sizes, when true model exists
- **Both**: Provide different perspectives on model complexity

## Residual Analysis

Analyzing residuals (prediction errors) reveals model deficiencies.

### Residuals

$$\text{Residual}_i = e_i = y_i - \hat{y}_i$$

### Residual Plots

**Residuals vs. Predicted Values**:
- **Random Scatter**: Good (no patterns)
- **Funnel Shape**: Heteroscedasticity (non-constant variance)
- **Curved Pattern**: Non-linearity (model misspecification)
- **Trend**: Systematic bias

**Residuals vs. Features**:
- Identify which features cause problems
- Detect interactions or non-linearities

**Q-Q Plot** (Quantile-Quantile):
- Check normality assumption
- Points should lie on diagonal line

### Assumptions

**Linear Regression Assumptions**:
1. **Linearity**: $E[\epsilon] = 0$ (residuals centered at zero)
2. **Independence**: Residuals uncorrelated
3. **Homoscedasticity**: Constant variance $\text{Var}(\epsilon) = \sigma^2$
4. **Normality**: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ (for inference)

### Detecting Violations

**Non-Linearity**: 
- Curved pattern in residual plot
- Solution: Add polynomial features, transformations

**Heteroscedasticity**:
- Funnel shape in residual plot
- Solution: Transformations, weighted least squares

**Non-Normality**:
- Deviations from diagonal in Q-Q plot
- Solution: Transformations, robust methods

**Outliers**:
- Points far from others in residual plot
- Solution: Robust methods, investigate data quality

### Cook's Distance

Measures influence of each observation:

$$D_i = \frac{(e_i)^2}{p \cdot \text{MSE}} \cdot \frac{h_i}{(1-h_i)^2}$$

where $h_i$ is leverage.

**Threshold**: $D_i > \frac{4}{n}$ indicates influential observation

## Model Selection Strategies

Choosing the best model involves balancing fit and complexity.

### Overfitting vs. Underfitting

**Overfitting**: 
- Low training error, high validation error
- Model too complex
- Solution: Reduce complexity, add regularization

**Underfitting**:
- High training and validation error
- Model too simple
- Solution: Increase complexity, add features

### Bias-Variance Tradeoff

- **Simple Models**: High bias, low variance
- **Complex Models**: Low bias, high variance
- **Optimal**: Balance bias and variance

### Model Selection Methods

**Information Criteria**:
- AIC, BIC balance fit and complexity
- No need for validation set
- Fast computation

**Cross-Validation**:
- More reliable estimate of generalization
- Requires validation set
- Computationally expensive

**Regularization**:
- L1 (Lasso), L2 (Ridge) control complexity
- Built into training process
- Hyperparameter tuning needed

### Feature Selection

**Filter Methods**: 
- Select features before training
- Based on correlation, mutual information
- Fast but may miss interactions

**Wrapper Methods**:
- Use model performance to select features
- Forward selection, backward elimination
- Computationally expensive

**Embedded Methods**:
- Feature selection during training
- Lasso (L1 regularization)
- Efficient and effective

### Stepwise Regression

**Forward Selection**:
1. Start with no features
2. Add feature that improves model most
3. Repeat until no improvement

**Backward Elimination**:
1. Start with all features
2. Remove feature that hurts least
3. Repeat until performance degrades

**Limitations**: 
- Greedy (may miss optimal subset)
- Multiple testing issues
- Can overfit

## Cross-Validation for Regression

CV provides reliable performance estimates for regression models.

### K-Fold CV for Regression

Same procedure as classification:
1. Divide data into $k$ folds
2. For each fold:
   - Train on $k-1$ folds
   - Evaluate on held-out fold
3. Average performance across folds

### Metrics for CV

Common metrics averaged across folds:
- **MSE**: $\text{CV-MSE} = \frac{1}{k}\sum_{i=1}^k \text{MSE}_i$
- **MAE**: $\text{CV-MAE} = \frac{1}{k}\sum_{i=1}^k \text{MAE}_i$
- **R²**: $\text{CV-R}^2 = \frac{1}{k}\sum_{i=1}^k \text{R}^2_i$

### Leave-One-Out CV

Special case with $k = n$:
- Nearly unbiased estimate
- High variance
- Computationally expensive

### Time Series CV

For temporal data:
- Maintain temporal order
- Forward chaining
- Rolling or expanding windows

### Nested CV

For hyperparameter tuning:
- Outer loop: Model evaluation
- Inner loop: Hyperparameter selection
- Prevents optimistic bias

## Key Takeaways

1. **Regression Metrics** quantify prediction accuracy for continuous targets, with scale-dependent metrics (MSE, MAE, RMSE) and scale-independent metrics (R²) serving different purposes.

2. **Mean Squared Error** $\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ penalizes large errors quadratically, differentiable and suitable for optimization but sensitive to outliers and scale-dependent.

3. **Mean Absolute Error** $\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$ treats all errors equally, robust to outliers, interpretable (same units as target) but non-differentiable at zero.

4. **Root Mean Squared Error** $\text{RMSE} = \sqrt{\text{MSE}}$ provides interpretable units (same as target), penalizes large errors, always $\geq$ MAE with larger difference indicating more variable errors.

5. **R-Squared** $\text{R}^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ measures proportion of variance explained, scale-independent but increases with features; **Adjusted R²** penalizes complexity: $1 - \frac{(1-R^2)(n-1)}{n-p-1}$.

6. **Information Criteria** balance fit and complexity: **AIC** $= 2p - 2\ln(L)$ focuses on prediction, **BIC** $= p\ln(n) - 2\ln(L)$ stronger penalty, consistent model selection, with lower values indicating better models.

7. **Residual Analysis** examines $e_i = y_i - \hat{y}_i$ through plots (vs. predicted, vs. features, Q-Q) to detect non-linearity, heteroscedasticity, non-normality, and outliers, with Cook's distance identifying influential observations.

8. **Model Selection** balances bias-variance tradeoff, using information criteria (fast), cross-validation (reliable), regularization (built-in), and feature selection (filter/wrapper/embedded methods) to prevent overfitting.

9. **Cross-Validation for Regression** averages metrics (MSE, MAE, R²) across $k$ folds, with LOOCV for small datasets, time series CV for temporal data, and nested CV for hyperparameter tuning.

10. **Metric Selection** depends on problem requirements: MSE/RMSE for optimization and large error penalty, MAE for robustness and interpretability, R² for variance explanation, information criteria for complexity-aware selection, with residual analysis essential for diagnosing model issues.
