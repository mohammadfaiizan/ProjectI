# Regression and Factor Models

## Ordinary Least Squares (OLS)

OLS is the foundation of linear regression and factor modeling.

### Model Specification

$$Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \cdots + \beta_p X_{pi} + \epsilon_i$$

In matrix form:
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where:
- $\mathbf{y}$ is $n \times 1$ vector of dependent variable
- $\mathbf{X}$ is $n \times (p+1)$ matrix of regressors (including intercept)
- $\boldsymbol{\beta}$ is $(p+1) \times 1$ vector of coefficients
- $\boldsymbol{\epsilon}$ is $n \times 1$ vector of errors

### OLS Estimator

Minimize sum of squared residuals:
$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n}(Y_i - \mathbf{X}_i^T \boldsymbol{\beta})^2$$

**Solution:**
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Fitted values:**
$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y}$$

where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is the hat matrix.

**Residuals:**
$$\hat{\boldsymbol{\epsilon}} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$$

### Assumptions

**Classical linear regression assumptions:**

1. **Linearity:** $\mathbb{E}[\boldsymbol{\epsilon} | \mathbf{X}] = 0$
2. **Homoskedasticity:** $\text{Var}(\epsilon_i | \mathbf{X}) = \sigma^2$ (constant)
3. **No autocorrelation:** $\text{Cov}(\epsilon_i, \epsilon_j | \mathbf{X}) = 0$ for $i \neq j$
4. **Exogeneity:** $\mathbf{X}$ is fixed or $\mathbb{E}[\boldsymbol{\epsilon} | \mathbf{X}] = 0$
5. **No perfect multicollinearity:** $\mathbf{X}^T\mathbf{X}$ is invertible
6. **Normality (optional):** $\boldsymbol{\epsilon} | \mathbf{X} \sim N(0, \sigma^2\mathbf{I})$

### Properties (BLUE)

Under assumptions 1-5, OLS is **Best Linear Unbiased Estimator (BLUE)**:

- **Unbiased:** $\mathbb{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$
- **Efficient:** Minimum variance among linear unbiased estimators
- **Consistent:** $\hat{\boldsymbol{\beta}} \xrightarrow{p} \boldsymbol{\beta}$ as $n \to \infty$

**Variance:**
$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$$

**Estimated variance:**
$$\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}) = \hat{\sigma}^2(\mathbf{X}^T\mathbf{X})^{-1}$$

where $\hat{\sigma}^2 = \frac{1}{n-p-1}\sum_{i=1}^{n}\hat{\epsilon}_i^2$ is the residual variance estimator.

### Inference

**t-test for coefficient:**
$$t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)} \sim t_{n-p-1}$$

under $H_0: \beta_j = 0$.

**F-test for joint significance:**
$$F = \frac{(RSS_0 - RSS_1)/(p_1 - p_0)}{RSS_1/(n - p_1 - 1)} \sim F_{p_1-p_0, n-p_1-1}$$

where $RSS$ is residual sum of squares.

### Goodness of Fit

**R-squared:**
$$R^2 = 1 - \frac{SSR}{SST} = \frac{SSE}{SST}$$

where:
- $SSR = \sum \hat{\epsilon}_i^2$ (sum of squared residuals)
- $SSE = \sum (\hat{y}_i - \bar{y})^2$ (explained sum of squares)
- $SST = \sum (y_i - \bar{y})^2$ (total sum of squares)

**Adjusted R-squared:**
$$\bar{R}^2 = 1 - \frac{SSR/(n-p-1)}{SST/(n-1)}$$

Penalizes for number of parameters.

## Generalized Least Squares (GLS)

GLS handles heteroskedasticity and autocorrelation.

### Model

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where $\boldsymbol{\epsilon} \sim N(0, \boldsymbol{\Omega})$ and $\boldsymbol{\Omega}$ is known positive definite matrix.

### GLS Estimator

$$\hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^T\boldsymbol{\Omega}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\Omega}^{-1}\mathbf{y}$$

**Variance:**
$$\text{Var}(\hat{\boldsymbol{\beta}}_{GLS}) = (\mathbf{X}^T\boldsymbol{\Omega}^{-1}\mathbf{X})^{-1}$$

### Weighted Least Squares (WLS)

Special case when $\boldsymbol{\Omega}$ is diagonal:
$$\boldsymbol{\Omega} = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_n^2)$$

**Weighted estimator:**
$$\hat{\boldsymbol{\beta}}_{WLS} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y}$$

where $\mathbf{W} = \boldsymbol{\Omega}^{-1}$.

**Use:** Heteroskedastic errors with known weights.

### Feasible GLS (FGLS)

When $\boldsymbol{\Omega}$ is unknown, estimate it first:

1. Estimate $\hat{\boldsymbol{\beta}}$ via OLS
2. Estimate $\hat{\boldsymbol{\Omega}}$ from residuals
3. Apply GLS with $\hat{\boldsymbol{\Omega}}$

**Iterate until convergence.**

## Regularization: Ridge, Lasso, Elastic Net

Regularization prevents overfitting by penalizing large coefficients.

### Ridge Regression

Adds L2 penalty:

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_2^2$$

**Solution:**
$$\hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

**Properties:**
- Shrinks coefficients toward zero
- Does not set coefficients to exactly zero
- Useful for multicollinearity

**Tuning parameter:** $\lambda > 0$ controls shrinkage (larger $\lambda$ = more shrinkage).

### Lasso Regression

Adds L1 penalty:

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1$$

**Properties:**
- Sets some coefficients to exactly zero (variable selection)
- Sparse solutions
- No closed form (solved via coordinate descent)

**Use:** Feature selection when many predictors.

### Elastic Net

Combines L1 and L2 penalties:

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2$$

**Advantages:**
- Variable selection (from Lasso)
- Handles correlated predictors (from Ridge)
- More stable than Lasso alone

### Cross-Validation for $\lambda$

Choose $\lambda$ via cross-validation:
1. Split data into folds
2. For each $\lambda$, fit on training folds, evaluate on validation fold
3. Choose $\lambda$ with lowest validation error

## Fama-French Factor Models

Factor models explain asset returns using common risk factors.

### Fama-French 3-Factor Model

$$r_i - r_f = \alpha_i + \beta_i^{MKT}(r_M - r_f) + \beta_i^{SMB}SMB + \beta_i^{HML}HML + \epsilon_i$$

where:
- $r_i$: Return on asset $i$
- $r_f$: Risk-free rate
- $r_M$: Market return
- **MKT:** Market factor ($r_M - r_f$)
- **SMB:** Small Minus Big (size factor)
- **HML:** High Minus Low (value factor)

**Factors:**
- **SMB:** Return on small cap portfolio minus large cap
- **HML:** Return on high book-to-market minus low book-to-market

**Interpretation:**
- $\alpha_i$: Abnormal return (alpha)
- $\beta_i^{MKT}$: Market exposure
- $\beta_i^{SMB}$: Size exposure
- $\beta_i^{HML}$: Value exposure

### Fama-French 5-Factor Model

Adds profitability and investment factors:

$$r_i - r_f = \alpha_i + \beta_i^{MKT}(r_M - r_f) + \beta_i^{SMB}SMB + \beta_i^{HML}HML + \beta_i^{RMW}RMW + \beta_i^{CMA}CMA + \epsilon_i$$

where:
- **RMW:** Robust Minus Weak (profitability factor)
- **CMA:** Conservative Minus Aggressive (investment factor)

### Carhart 4-Factor Model

Adds momentum to Fama-French 3-factor:

$$r_i - r_f = \alpha_i + \beta_i^{MKT}(r_M - r_f) + \beta_i^{SMB}SMB + \beta_i^{HML}HML + \beta_i^{MOM}MOM + \epsilon_i$$

where **MOM** is the momentum factor (winners minus losers).

## PCA-Based Factor Models

Principal Component Analysis extracts statistical factors from return data.

### Principal Components

For return matrix $\mathbf{R}$ ($n \times T$, $n$ assets, $T$ periods):

1. **Standardize:** $\tilde{\mathbf{R}} = \mathbf{R} - \bar{\mathbf{R}}$
2. **Covariance:** $\boldsymbol{\Sigma} = \frac{1}{T-1}\tilde{\mathbf{R}}\tilde{\mathbf{R}}^T$
3. **Eigen decomposition:** $\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$
4. **Principal components:** $\mathbf{F} = \mathbf{V}^T\tilde{\mathbf{R}}$

**Properties:**
- First PC explains maximum variance
- PCs are orthogonal
- $\mathbf{R} = \mathbf{V}\mathbf{F} + \bar{\mathbf{R}}$ (reconstruction)

### Factor Model Representation

$$r_{it} = \alpha_i + \sum_{j=1}^{K}\beta_{ij} F_{jt} + \epsilon_{it}$$

where $F_{jt}$ are principal components (statistical factors).

**Estimation:**
- Factors: Principal components
- Loadings: Regression coefficients $\beta_{ij}$

### Variance Explained

**Scree plot:** Plot eigenvalues $\lambda_j$ vs component number.

**Variance explained by component $j$:**
$$\frac{\lambda_j}{\sum_{k=1}^{n}\lambda_k}$$

**Cumulative variance:**
$$\frac{\sum_{j=1}^{K}\lambda_j}{\sum_{k=1}^{n}\lambda_k}$$

**Rule of thumb:** Keep components explaining >80-90% of variance, or use elbow method.

### Statistical vs Economic Factors

**Statistical factors (PCA):**
- Data-driven
- No economic interpretation
- Maximize variance explained
- Orthogonal by construction

**Economic factors (Fama-French):**
- Theoretically motivated
- Interpretable (size, value, etc.)
- May be correlated
- Testable hypotheses

**Hybrid:** Use PCA to identify factors, then interpret economically.

## Cross-Sectional Regression

Cross-sectional regression regresses returns on characteristics at a point in time.

### Fama-MacBeth Procedure

**Step 1:** For each period $t$, run cross-sectional regression:
$$r_{it} = \gamma_{0t} + \gamma_{1t}X_{1i} + \gamma_{2t}X_{2i} + \cdots + \epsilon_{it}$$

**Step 2:** Average coefficients across time:
$$\hat{\gamma}_j = \frac{1}{T}\sum_{t=1}^{T}\hat{\gamma}_{jt}$$

**Step 3:** Standard errors:
$$SE(\hat{\gamma}_j) = \sqrt{\frac{1}{T(T-1)}\sum_{t=1}^{T}(\hat{\gamma}_{jt} - \hat{\gamma}_j)^2}$$

**Advantages:**
- Accounts for cross-sectional correlation
- Standard errors account for time-series variation
- Robust to heteroskedasticity

### Panel Regression

Alternative: Pool all data and use panel methods:

$$r_{it} = \alpha_i + \boldsymbol{\gamma}^T\mathbf{X}_{it} + \epsilon_{it}$$

**Fixed effects:** Include asset dummies $\alpha_i$
**Random effects:** Treat $\alpha_i$ as random

**Clustered standard errors:** Account for correlation within assets and time.

## Applications in Finance

### Alpha Generation

Factor models decompose returns:
- **Alpha ($\alpha$):** Abnormal return (skill)
- **Factor exposure:** Systematic risk
- **Idiosyncratic:** Asset-specific risk

**Use:** Identify skilled managers, construct portfolios.

### Risk Decomposition

Portfolio risk:
$$\sigma_P^2 = \sum_{i,j}w_i w_j \beta_i^T \boldsymbol{\Sigma}_F \beta_j + \sum_i w_i^2 \sigma_{\epsilon_i}^2$$

where $\boldsymbol{\Sigma}_F$ is factor covariance matrix.

**Decomposition:**
- Factor risk: First term
- Idiosyncratic risk: Second term

### Portfolio Construction

**Factor-neutral:** Set $\sum_i w_i \beta_{ij} = 0$ for all factors $j$ (hedge factor exposure).

**Factor tilting:** Target specific factor exposures.

**Risk budgeting:** Allocate risk across factors.

### Performance Attribution

Decompose portfolio return:
$$r_P = \alpha_P + \sum_j \beta_P^j F_j + \epsilon_P$$

**Attribution:**
- Alpha: Stock selection
- Factor exposure: Factor timing
- Idiosyncratic: Diversification

## Practical Considerations

### Factor Selection

- **Economic theory:** Should guide factor choice
- **Statistical significance:** Test if factors are priced
- **Stability:** Factors should be stable over time
- **Out-of-sample:** Validate on hold-out data

### Estimation Issues

- **Multicollinearity:** Factors may be correlated
- **Non-stationarity:** Factor loadings may change
- **Missing data:** Handle appropriately
- **Outliers:** Robust methods may be needed

### Model Validation

- **In-sample fit:** $R^2$, adjusted $R^2$
- **Out-of-sample:** Forecast evaluation
- **Stability:** Rolling window estimation
- **Economic significance:** Do coefficients make sense?
