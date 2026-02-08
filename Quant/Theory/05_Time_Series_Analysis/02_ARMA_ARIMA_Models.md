# ARMA and ARIMA Models

## Autoregressive (AR) Models

An AR(p) process depends on its past $p$ values plus a random shock.

### Definition

$$X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \epsilon_t$$

where $\epsilon_t \sim \text{WN}(0, \sigma^2)$.

Using the lag operator $L$ where $LX_t = X_{t-1}$:

$$\phi(L)X_t = \epsilon_t$$

where $\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p$ is the AR polynomial.

### Stationarity Condition

An AR(p) process is stationary if all roots of $\phi(z) = 0$ lie outside the unit circle ($|z| > 1$).

For AR(1): $X_t = \phi X_{t-1} + \epsilon_t$
- Stationary if $|\phi| < 1$

For AR(2): Need to check roots of $1 - \phi_1 z - \phi_2 z^2 = 0$

### Mean

For a stationary AR(p):
$$\mu = \mathbb{E}[X_t] = \frac{\phi_0}{1 - \phi_1 - \cdots - \phi_p}$$

Often $\phi_0 = 0$ (no constant), so $\mu = 0$.

### Yule-Walker Equations

The Yule-Walker equations relate AR coefficients to autocorrelations:

$$\rho(h) = \phi_1 \rho(h-1) + \phi_2 \rho(h-2) + \cdots + \phi_p \rho(h-p)$$

for $h \geq 1$.

In matrix form:
$$\begin{pmatrix}
\rho(1) \\
\rho(2) \\
\vdots \\
\rho(p)
\end{pmatrix}
=
\begin{pmatrix}
1 & \rho(1) & \cdots & \rho(p-1) \\
\rho(1) & 1 & \cdots & \rho(p-2) \\
\vdots & \vdots & \ddots & \vdots \\
\rho(p-1) & \rho(p-2) & \cdots & 1
\end{pmatrix}
\begin{pmatrix}
\phi_1 \\
\phi_2 \\
\vdots \\
\phi_p
\end{pmatrix}$$

This allows:
1. **Estimation:** Estimate $\rho(h)$ from data, solve for $\phi$
2. **Theoretical ACF:** Given $\phi$, compute $\rho(h)$

### ACF of AR(p)

The ACF of an AR(p) process decays exponentially (or as a mixture of exponentials). It does not cut off but decays smoothly.

**AR(1) example:**
$$\rho(h) = \phi^h$$

Decays geometrically.

### PACF of AR(p)

The PACF cuts off at lag $p$:
- $\phi_{hh} \neq 0$ for $h \leq p$
- $\phi_{hh} = 0$ for $h > p$

This property helps identify AR order.

## Moving Average (MA) Models

An MA(q) process depends on past $q$ shocks.

### Definition

$$X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}$$

Using lag operator:
$$X_t = \theta(L)\epsilon_t$$

where $\theta(L) = 1 + \theta_1 L + \theta_2 L^2 + \cdots + \theta_q L^q$.

### Invertibility

An MA(q) is invertible if all roots of $\theta(z) = 0$ lie outside the unit circle.

Invertibility allows representing the MA as an infinite AR:
$$X_t = \sum_{i=1}^{\infty}\pi_i X_{t-i} + \epsilon_t$$

**Why important:** Invertible MA processes have unique representations and are easier to estimate.

### Mean and Variance

$$\mathbb{E}[X_t] = 0$$
$$\text{Var}(X_t) = \sigma^2(1 + \theta_1^2 + \theta_2^2 + \cdots + \theta_q^2)$$

### ACF of MA(q)

The ACF cuts off at lag $q$:

$$\rho(h) = \begin{cases}
\frac{\sum_{j=0}^{q-h}\theta_j \theta_{j+h}}{\sum_{j=0}^{q}\theta_j^2} & \text{if } h \leq q \\
0 & \text{if } h > q
\end{cases}$$

where $\theta_0 = 1$.

**Key property:** $\rho(h) = 0$ for $h > q$ - helps identify MA order.

### PACF of MA(q)

The PACF decays (does not cut off), similar to how ACF behaves for AR processes.

## ARMA Models

ARMA(p,q) combines AR and MA components.

### Definition

$$X_t = \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}$$

Or:
$$\phi(L)X_t = \theta(L)\epsilon_t$$

### Stationarity and Invertibility

- **Stationary:** If AR part is stationary (roots of $\phi(z) = 0$ outside unit circle)
- **Invertible:** If MA part is invertible (roots of $\theta(z) = 0$ outside unit circle)

### ACF and PACF

Both ACF and PACF decay (neither cuts off), making identification more challenging than pure AR or MA.

### Parsimony

ARMA models are parsimonious - can capture complex dynamics with few parameters. For example, ARMA(1,1) can approximate higher-order AR or MA models.

## Model Identification

### Box-Jenkins Methodology

1. **Identification:** Use ACF/PACF to suggest $(p,q)$
2. **Estimation:** Estimate parameters
3. **Diagnostics:** Check residuals
4. **Forecasting:** Use model for predictions

### Identification Rules

**AR(p):**
- ACF: Decays
- PACF: Cuts off at lag $p$

**MA(q):**
- ACF: Cuts off at lag $q$
- PACF: Decays

**ARMA(p,q):**
- ACF: Decays
- PACF: Decays
- Need to try different $(p,q)$ combinations

### Information Criteria

**AIC (Akaike Information Criterion):**
$$AIC = -2\ln L + 2k$$

**BIC (Bayesian Information Criterion):**
$$BIC = -2\ln L + k\ln n$$

where $L$ is likelihood, $k$ is number of parameters, $n$ is sample size.

**Choose model with lowest AIC/BIC.**

BIC penalizes complexity more (prefers simpler models).

## Estimation

### Maximum Likelihood Estimation (MLE)

For Gaussian innovations, the likelihood is:

$$L(\phi, \theta, \sigma^2) = (2\pi\sigma^2)^{-n/2}|\Sigma|^{-1/2}\exp\left(-\frac{1}{2\sigma^2}\mathbf{X}^T\Sigma^{-1}\mathbf{X}\right)$$

where $\Sigma$ is the covariance matrix of $\mathbf{X} = (X_1, \ldots, X_n)^T$.

**Optimization:** Maximize log-likelihood numerically.

**Initial values:** Use method of moments or conditional likelihood.

### Conditional Sum of Squares

For ARMA models, use conditional likelihood:

$$L_{cond} = \prod_{t=m+1}^{n}f(X_t | X_{t-1}, \ldots, X_1)$$

where $m = \max(p,q)$.

Set initial values $\epsilon_0 = \epsilon_{-1} = \cdots = 0$ and $X_0 = X_{-1} = \cdots = 0$.

**Advantages:** Simpler, faster
**Disadvantages:** Less efficient than exact MLE

### Yule-Walker Estimation

For AR(p), solve Yule-Walker equations using sample ACF:

$$\hat{\phi} = \hat{R}^{-1}\hat{\rho}$$

where $\hat{R}$ is the sample autocorrelation matrix and $\hat{\rho}$ is the vector of sample autocorrelations.

**Properties:** Consistent but less efficient than MLE.

## ARIMA Models

ARIMA(p,d,q) extends ARMA to non-stationary series by differencing.

### Definition

If $\Delta^d X_t$ is ARMA(p,q), then $X_t$ is ARIMA(p,d,q):

$$\phi(L)(1-L)^d X_t = \theta(L)\epsilon_t$$

where $(1-L)^d$ is the differencing operator applied $d$ times.

### Differencing

**First difference:** $\Delta X_t = X_t - X_{t-1}$

**Second difference:** $\Delta^2 X_t = \Delta(\Delta X_t) = X_t - 2X_{t-1} + X_{t-2}$

**Seasonal difference:** $\Delta_s X_t = X_t - X_{t-s}$ (for period $s$)

### Choosing d

- **ADF test:** Test for unit root, difference until stationary
- **Visual:** Plot series and differences
- **ACF:** If ACF decays very slowly, likely needs differencing

**Typical:** $d = 0, 1, 2$ (rarely more)

### Example: Random Walk

Random walk: $X_t = X_{t-1} + \epsilon_t$

First difference: $\Delta X_t = \epsilon_t$ (white noise)

So random walk is ARIMA(0,1,0).

## SARIMA Models

Seasonal ARIMA includes seasonal components.

### Definition

SARIMA(p,d,q)(P,D,Q)$_s$:

$$\phi(L)\Phi(L^s)(1-L)^d(1-L^s)^D X_t = \theta(L)\Theta(L^s)\epsilon_t$$

where:
- $(p,d,q)$: Non-seasonal orders
- $(P,D,Q)$: Seasonal orders
- $s$: Seasonal period (e.g., 12 for monthly, 4 for quarterly)

**Example:** SARIMA(1,1,1)(1,1,1)$_{12}$ for monthly data with yearly seasonality.

### Identification

1. Remove seasonality: Apply $(1-L^s)^D$
2. Remove trend: Apply $(1-L)^d$
3. Identify ARMA orders for differenced series

## Model Selection

### Information Criteria

Compare models using AIC or BIC:
- Lower is better
- BIC penalizes complexity more
- Trade-off: Fit vs parsimony

### Cross-Validation

1. Split data into train/test
2. Fit on training set
3. Evaluate on test set (e.g., MSE)
4. Choose model with best out-of-sample performance

### Residual Diagnostics

**Check residuals $\hat{\epsilon}_t$:**

1. **ACF/PACF:** Should be white noise (no significant autocorrelation)
2. **Ljung-Box test:** Test for remaining autocorrelation
3. **Normality:** Q-Q plots, Jarque-Bera test
4. **Heteroskedasticity:** Check for ARCH effects

## Forecasting

### Point Forecasts

For ARMA(p,q), the $h$-step ahead forecast is:

$$\hat{X}_{t+h|t} = \mathbb{E}[X_{t+h} | X_t, X_{t-1}, \ldots]$$

**AR(1) example:**
$$\hat{X}_{t+h|t} = \phi^h X_t$$

**MA(1) example:**
$$\hat{X}_{t+1|t} = \theta_1 \epsilon_t$$
$$\hat{X}_{t+h|t} = 0 \quad \text{for } h > 1$$

### Forecast Function

The forecast function shows how forecasts evolve:
- **AR:** Forecasts converge to mean
- **MA:** Forecasts converge to mean after $q$ steps
- **ARMA:** Combination of both

### Prediction Intervals

For Gaussian innovations, the $h$-step forecast error is:

$$e_{t+h} = X_{t+h} - \hat{X}_{t+h|t}$$

Variance:
$$\text{Var}(e_{t+h}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2$$

where $\psi_j$ are MA coefficients of the ARMA representation.

**95% prediction interval:**
$$\hat{X}_{t+h|t} \pm 1.96\sqrt{\text{Var}(e_{t+h})}$$

### Updating Forecasts

As new data arrives, update forecasts:
$$\hat{X}_{t+h|t+1} = \hat{X}_{t+h|t} + \psi_{h-1}\epsilon_{t+1}$$

where $\epsilon_{t+1} = X_{t+1} - \hat{X}_{t+1|t}$ is the one-step forecast error.

## Applications in Finance

### Stock Returns

- Often close to white noise (ARIMA(0,0,0))
- Weak autocorrelation possible
- GARCH for volatility (squared returns show ARMA structure)

### Interest Rates

- May show mean reversion (AR component)
- Often non-stationary (need differencing)
- ARIMA models common

### Exchange Rates

- Often close to random walk (ARIMA(0,1,0))
- Hard to forecast
- Focus on volatility modeling

## Practical Considerations

### Overfitting

- Too many parameters: Fits noise, poor out-of-sample
- Use information criteria and cross-validation
- Prefer simpler models

### Structural Breaks

- Parameters may change over time
- Test for breaks (Chow test, CUSUM)
- Use rolling windows or regime-switching models

### Non-linearity

- ARMA assumes linearity
- Financial data may be nonlinear
- Consider threshold models, neural networks

### Computational Issues

- Optimization may not converge
- Multiple local maxima
- Try different starting values
- Check stationarity/invertibility constraints
