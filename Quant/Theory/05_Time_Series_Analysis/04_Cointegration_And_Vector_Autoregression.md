# Cointegration and Vector Autoregression

## Spurious Regression

Spurious regression occurs when non-stationary series appear to be related but are actually independent.

### Granger-Newbold Phenomenon

Regressing two independent random walks:
$$Y_t = \alpha + \beta X_t + \epsilon_t$$

where $X_t$ and $Y_t$ are independent random walks.

**Problems:**
- High $R^2$ (misleading)
- Significant $t$-statistics (spurious)
- Low Durbin-Watson (residuals non-stationary)

**Solution:** Test for cointegration or difference the series.

## Cointegration

Cointegration occurs when a linear combination of non-stationary series is stationary.

### Definition

Two series $X_t$ and $Y_t$ are cointegrated if:
1. Both are $I(1)$ (integrated of order 1, i.e., non-stationary but first differences are stationary)
2. There exists $\beta$ such that $Y_t - \beta X_t$ is stationary

The vector $(1, -\beta)^T$ is the cointegrating vector.

**Economic interpretation:** Series move together in the long run, even if they drift apart in the short run.

### Example: Purchasing Power Parity

Exchange rate $E_t$ and price ratio $P_t/P_t^*$ should be cointegrated:
$$\ln E_t - \beta \ln(P_t/P_t^*) = u_t$$

where $u_t$ is stationary (deviation from PPP).

### Vector Error Correction Model (VECM)

If $X_t$ and $Y_t$ are cointegrated with cointegrating relationship $Y_t = \beta X_t + u_t$, the VECM is:

$$\Delta Y_t = \alpha_1 + \gamma_1 (Y_{t-1} - \beta X_{t-1}) + \sum_{i=1}^{p}\phi_{1i}\Delta Y_{t-i} + \sum_{i=1}^{p}\theta_{1i}\Delta X_{t-i} + \epsilon_{1t}$$

$$\Delta X_t = \alpha_2 + \gamma_2 (Y_{t-1} - \beta X_{t-1}) + \sum_{i=1}^{p}\phi_{2i}\Delta Y_{t-i} + \sum_{i=1}^{p}\theta_{2i}\Delta X_{t-i} + \epsilon_{2t}$$

The term $(Y_{t-1} - \beta X_{t-1})$ is the error correction term - it pulls the system back to equilibrium.

**Interpretation:**
- $\gamma_1 < 0$: $Y_t$ adjusts to restore equilibrium
- $\gamma_2 > 0$: $X_t$ adjusts to restore equilibrium
- Speed of adjustment: Larger $|\gamma|$ means faster adjustment

## Testing for Cointegration

### Engle-Granger Two-Step Procedure

**Step 1:** Estimate cointegrating regression:
$$Y_t = \alpha + \beta X_t + u_t$$

**Step 2:** Test residuals $\hat{u}_t$ for unit root using ADF test.

**Null:** $H_0$: No cointegration (residuals have unit root)
**Alternative:** $H_1$: Cointegration (residuals stationary)

**Critical values:** Non-standard (depend on number of variables). Use Engle-Granger critical values.

**Limitations:**
- Assumes one cointegrating relationship
- Sensitive to which variable is dependent
- Can't test for multiple cointegrating vectors

### Johansen Procedure

Johansen's method tests for multiple cointegrating relationships in a VAR framework.

### Vector Autoregression (VAR)

A VAR(p) model for $k$ variables:

$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^{p}\mathbf{\Phi}_i \mathbf{Y}_{t-i} + \boldsymbol{\epsilon}_t$$

where:
- $\mathbf{Y}_t$ is $k \times 1$ vector of variables
- $\mathbf{\Phi}_i$ are $k \times k$ coefficient matrices
- $\boldsymbol{\epsilon}_t \sim \text{WN}(0, \Sigma)$

**Example - VAR(1) with 2 variables:**
$$\begin{pmatrix} Y_t \\ X_t \end{pmatrix} = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix} + \begin{pmatrix} \phi_{11} & \phi_{12} \\ \phi_{21} & \phi_{22} \end{pmatrix}\begin{pmatrix} Y_{t-1} \\ X_{t-1} \end{pmatrix} + \begin{pmatrix} \epsilon_{1t} \\ \epsilon_{2t} \end{pmatrix}$$

### VAR in Levels vs Differences

**Levels:** If variables are stationary, estimate VAR in levels
**Differences:** If non-stationary and not cointegrated, difference first

**Cointegration:** If cointegrated, use VECM (VAR in differences with error correction term)

### Johansen Test

Write VAR(p) as VECM:

$$\Delta \mathbf{Y}_t = \mathbf{c} + \mathbf{\Pi} \mathbf{Y}_{t-1} + \sum_{i=1}^{p-1}\mathbf{\Gamma}_i \Delta \mathbf{Y}_{t-i} + \boldsymbol{\epsilon}_t$$

where $\mathbf{\Pi} = -\mathbf{I} + \sum_{i=1}^{p}\mathbf{\Phi}_i$.

**Key insight:** Rank of $\mathbf{\Pi}$ equals number of cointegrating relationships.

**Tests:**
1. **Trace test:** Tests $H_0$: at most $r$ cointegrating vectors
2. **Maximum eigenvalue test:** Tests $H_0$: exactly $r$ cointegrating vectors vs $r+1$

**Procedure:**
- Start with $r = 0$
- If reject, test $r = 1$, etc.
- Stop when cannot reject

### Estimation

**OLS:** Each equation can be estimated by OLS (efficient for VAR)
**MLE:** For VECM with cointegration restrictions

## Vector Autoregression (VAR)

VAR models multiple time series jointly, allowing for dynamic interactions.

### Specification

**VAR(p):**
$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^{p}\mathbf{\Phi}_i \mathbf{Y}_{t-i} + \boldsymbol{\epsilon}_t$$

**VAR(1) representation:** Write VAR(p) as VAR(1) using companion form:
$$\mathbf{Z}_t = \mathbf{A}\mathbf{Z}_{t-1} + \mathbf{u}_t$$

where $\mathbf{Z}_t = (\mathbf{Y}_t^T, \mathbf{Y}_{t-1}^T, \ldots, \mathbf{Y}_{t-p+1}^T)^T$.

### Stationarity

VAR is stationary if all eigenvalues of $\mathbf{A}$ lie inside the unit circle.

**Univariate AR:** Need roots outside unit circle
**VAR:** Need eigenvalues inside unit circle (opposite convention)

### Estimation

**OLS:** Estimate each equation separately by OLS. Efficient (same as SUR when regressors are same).

**MLE:** Under Gaussian errors, MLE = OLS.

**Model selection:** Use information criteria (AIC, BIC) to choose lag length $p$.

### Granger Causality

$X_t$ Granger causes $Y_t$ if past values of $X$ help predict $Y$ beyond past values of $Y$ alone.

**Test:** In VAR, test $H_0$: $\phi_{12,i} = 0$ for all $i$ in:
$$Y_t = c_1 + \sum_{i=1}^{p}\phi_{11,i}Y_{t-i} + \sum_{i=1}^{p}\phi_{12,i}X_{t-i} + \epsilon_{1t}$$

Use $F$-test or likelihood ratio test.

**Note:** Granger causality is about predictability, not true causality.

### Impulse Response Functions (IRF)

IRF shows how one variable responds to a shock in another variable.

**Definition:** Response of $Y_{t+h}$ to one-unit shock in $\epsilon_{1t}$:
$$\text{IRF}(h) = \frac{\partial Y_{t+h}}{\partial \epsilon_{1t}}$$

**Computation:** From VAR(1) representation:
$$\mathbf{Y}_{t+h} = \sum_{j=0}^{h}\mathbf{A}^j \boldsymbol{\epsilon}_{t+h-j}$$

IRF is given by elements of $\mathbf{A}^j$.

**Orthogonalized IRF:** Use Cholesky decomposition of $\Sigma$ to orthogonalize shocks (order matters).

**Generalized IRF:** Pesaran-Shin approach (order-independent).

### Forecast Error Variance Decomposition (FEVD)

FEVD decomposes forecast error variance into contributions from each shock.

**$h$-step ahead forecast error:**
$$\mathbf{Y}_{t+h} - \hat{\mathbf{Y}}_{t+h|t} = \sum_{j=0}^{h-1}\mathbf{A}^j \boldsymbol{\epsilon}_{t+h-j}$$

**Variance decomposition:** Compute contribution of each shock to variance of each variable.

**Use:** Understand sources of volatility.

## Applications

### Pairs Trading

Pairs trading exploits cointegration between two assets.

**Strategy:**
1. Identify cointegrated pair (e.g., two stocks in same sector)
2. When spread $Y_t - \beta X_t$ deviates from mean, trade:
   - If spread too high: Short $Y$, long $X$
   - If spread too low: Long $Y$, short $X$
3. Close when spread reverts

**Hedge ratio:** $\beta$ from cointegrating regression.

**Risk:** Cointegration may break down (structural break).

### Statistical Arbitrage

Extend pairs trading to multiple assets:
- Find cointegrating relationships
- Trade deviations from equilibrium
- Use VECM to forecast mean reversion

### Macro Factor Modeling

Use VAR/VECM to model macroeconomic variables:
- GDP, inflation, interest rates
- Understand dynamic relationships
- Forecast macro variables
- Use in asset pricing models

### Term Structure Modeling

Model yield curve using VAR:
- Different maturities as variables
- Understand yield curve dynamics
- Forecast interest rates

## VECM: Combining Cointegration and VAR

VECM combines short-run dynamics (VAR) with long-run relationships (cointegration).

### Specification

For $k$ variables with $r$ cointegrating relationships:

$$\Delta \mathbf{Y}_t = \mathbf{c} + \boldsymbol{\alpha}\boldsymbol{\beta}^T \mathbf{Y}_{t-1} + \sum_{i=1}^{p-1}\mathbf{\Gamma}_i \Delta \mathbf{Y}_{t-i} + \boldsymbol{\epsilon}_t$$

where:
- $\boldsymbol{\beta}$ is $k \times r$ matrix of cointegrating vectors
- $\boldsymbol{\alpha}$ is $k \times r$ matrix of adjustment speeds
- $\boldsymbol{\Gamma}_i$ capture short-run dynamics

### Identification

Cointegrating vectors are not unique: if $\boldsymbol{\beta}$ is cointegrating, so is $\boldsymbol{\beta}\mathbf{C}$ for any $r \times r$ matrix $\mathbf{C}$.

**Normalization:** Impose restrictions (e.g., set one element to 1 in each vector).

**Economic theory:** Should guide identification.

### Estimation

**Two-step:**
1. Estimate cointegrating vectors (Johansen)
2. Estimate VECM with fixed $\boldsymbol{\beta}$

**One-step:** Joint estimation via MLE.

### Forecasting

VECM forecasts combine:
- Long-run equilibrium (cointegration)
- Short-run adjustments (VAR dynamics)

**Advantage:** Captures both short-run and long-run relationships.

## Practical Considerations

### Testing Order

1. **Unit root tests:** Determine integration order
2. **Cointegration tests:** Test for cointegration
3. **VAR specification:** Choose lag length
4. **Estimation:** Estimate VECM

### Structural Breaks

Cointegration relationships may break:
- Test for breaks (Bai-Perron, CUSUM)
- Use rolling windows
- Regime-switching models

### Exogeneity

Some variables may be weakly exogenous (don't adjust to disequilibrium):
- Test using VECM
- Impose restrictions on $\boldsymbol{\alpha}$

### Overfitting

VAR/VECM can have many parameters:
- Use information criteria
- Economic theory to restrict
- Regularization (Bayesian VAR)

### Interpretation

- Cointegration: Long-run relationship
- Error correction: Speed of adjustment
- VAR coefficients: Short-run dynamics
- IRF: Dynamic responses
