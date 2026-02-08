# Stationarity and Autocorrelation

## Stationarity

Stationarity is a fundamental concept in time series analysis. A stationary process has statistical properties that do not change over time.

### Strict Stationarity

A process $\{X_t\}$ is strictly stationary if the joint distribution of $(X_{t_1}, \ldots, X_{t_k})$ is the same as $(X_{t_1+h}, \ldots, X_{t_k+h})$ for all $k$ and $h$.

This means all moments and joint distributions are time-invariant.

### Weak Stationarity (Covariance Stationarity)

A process $\{X_t\}$ is weakly stationary if:

1. **Constant mean:** $\mathbb{E}[X_t] = \mu$ for all $t$
2. **Constant variance:** $\text{Var}(X_t) = \sigma^2$ for all $t$
3. **Covariance depends only on lag:** $\text{Cov}(X_t, X_{t+h}) = \gamma(h)$ for all $t$

Weak stationarity only requires the first two moments to be constant, making it easier to verify than strict stationarity.

**Note:** Strict stationarity implies weak stationarity (if second moments exist), but not vice versa. For Gaussian processes, they are equivalent.

### Why Stationarity Matters

Most time series methods assume stationarity:
- Autocorrelation functions
- ARMA models
- Spectral analysis
- Forecasting

Non-stationary series must be transformed (e.g., differencing) before analysis.

## Autocorrelation Function (ACF)

The autocorrelation function measures the correlation between a series and its lagged values.

### Definition

For a stationary process, the autocovariance function is:
$$\gamma(h) = \text{Cov}(X_t, X_{t+h}) = \mathbb{E}[(X_t - \mu)(X_{t+h} - \mu)]$$

The autocorrelation function is:
$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{\gamma(h)}{\sigma^2}$$

**Properties:**
- $\rho(0) = 1$
- $\rho(h) = \rho(-h)$ (symmetric)
- $|\rho(h)| \leq 1$

### Sample ACF

For observed data $x_1, \ldots, x_n$, the sample autocovariance is:
$$\hat{\gamma}(h) = \frac{1}{n}\sum_{t=1}^{n-h}(x_t - \bar{x})(x_{t+h} - \bar{x})$$

The sample ACF is:
$$\hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}$$

**Note:** Some authors use $1/(n-h)$ instead of $1/n$ in the denominator. The $1/n$ version ensures the sample covariance matrix is positive semi-definite.

### Confidence Bands

Under the null hypothesis that $\rho(h) = 0$ for $h > 0$ (white noise), the sample ACF is approximately:
$$\hat{\rho}(h) \sim N\left(0, \frac{1}{n}\right)$$

Approximate 95% confidence bands are:
$$\pm \frac{1.96}{\sqrt{n}}$$

Values outside these bands suggest significant autocorrelation.

**Bartlett's formula:** For more general processes, the variance of $\hat{\rho}(h)$ is:
$$\text{Var}(\hat{\rho}(h)) \approx \frac{1}{n}\sum_{j=-\infty}^{\infty}\rho(j)^2$$

## Partial Autocorrelation Function (PACF)

The partial autocorrelation function measures the correlation between $X_t$ and $X_{t+h}$ after removing the effect of intermediate lags.

### Definition

The PACF $\phi_{hh}$ is the correlation between $X_t$ and $X_{t+h}$ after removing the linear dependence on $X_{t+1}, \ldots, X_{t+h-1}$.

Equivalently, it's the last coefficient in the regression:
$$X_t = \phi_{h1}X_{t-1} + \phi_{h2}X_{t-2} + \cdots + \phi_{hh}X_{t-h} + \epsilon_t$$

### Computation

PACF can be computed using the Durbin-Levinson algorithm or by solving the Yule-Walker equations recursively.

**Durbin-Levinson:**
$$\phi_{11} = \rho(1)$$
$$\phi_{h+1,h+1} = \frac{\rho(h+1) - \sum_{j=1}^{h}\phi_{hj}\rho(h+1-j)}{1 - \sum_{j=1}^{h}\phi_{hj}\rho(j)}$$

### Sample PACF

Sample PACF is computed from sample ACF using the same recursive formulas.

**Confidence bands:** Under the null of an AR(p) process, $\hat{\phi}_{hh} \sim N(0, 1/n)$ for $h > p$.

## White Noise

White noise is the simplest stationary process.

### Definition

A process $\{\epsilon_t\}$ is white noise if:

1. $\mathbb{E}[\epsilon_t] = 0$ for all $t$
2. $\text{Var}(\epsilon_t) = \sigma^2$ for all $t$
3. $\text{Cov}(\epsilon_t, \epsilon_s) = 0$ for $t \neq s$

**Notation:** $\epsilon_t \sim \text{WN}(0, \sigma^2)$

### Properties

- Stationary (both strict and weak)
- ACF: $\rho(0) = 1$, $\rho(h) = 0$ for $h \neq 0$
- PACF: $\phi_{hh} = 0$ for all $h > 0$

### Gaussian White Noise

If $\epsilon_t \sim N(0, \sigma^2)$ and independent, it's Gaussian white noise.

## Random Walk

A random walk is a non-stationary process.

### Definition

$$X_t = X_{t-1} + \epsilon_t = X_0 + \sum_{i=1}^{t}\epsilon_i$$

where $\epsilon_t$ is white noise.

### Properties

**Mean:** $\mathbb{E}[X_t] = X_0$ (constant if $X_0$ is fixed)

**Variance:** $\text{Var}(X_t) = t\sigma^2$ (grows with $t$ - non-stationary!)

**Autocovariance:** For $h > 0$:
$$\text{Cov}(X_t, X_{t+h}) = \text{Var}(X_t) = t\sigma^2$$

**ACF:** For large $t$:
$$\rho(h) \approx \sqrt{\frac{t}{t+h}} \approx 1$$

The ACF decays very slowly, a signature of non-stationarity.

### Random Walk with Drift

$$X_t = \mu + X_{t-1} + \epsilon_t = X_0 + \mu t + \sum_{i=1}^{t}\epsilon_i$$

Has a deterministic trend component $\mu t$.

## Unit Root Tests

Unit root tests determine if a series has a unit root (is non-stationary).

### Augmented Dickey-Fuller (ADF) Test

Tests the null hypothesis $H_0$: unit root (non-stationary) against $H_1$: stationary.

**Model:**
$$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p}\delta_i \Delta X_{t-i} + \epsilon_t$$

where $\Delta X_t = X_t - X_{t-1}$.

**Test statistic:**
$$ADF = \frac{\hat{\gamma}}{SE(\hat{\gamma})}$$

**Null:** $\gamma = 0$ (unit root)
**Alternative:** $\gamma < 0$ (stationary)

**Critical values:** Non-standard (depend on specification). Reject if $ADF < \text{critical value}$.

**Specifications:**
- **No constant, no trend:** $\alpha = \beta = 0$
- **Constant:** $\beta = 0$
- **Constant and trend:** Both included

### Phillips-Perron (PP) Test

Similar to ADF but uses non-parametric correction for serial correlation:

$$PP = \frac{\hat{\gamma}}{SE(\hat{\gamma})} - \frac{1}{2}\frac{\hat{\sigma}^2 - \hat{\sigma}_{\epsilon}^2}{SE(\hat{\gamma})}\frac{T}{SE(\hat{\gamma})}$$

where $\hat{\sigma}^2$ and $\hat{\sigma}_{\epsilon}^2$ are long-run and short-run variance estimates.

**Advantage:** Robust to heteroskedasticity and serial correlation.

### KPSS Test

KPSS tests the null of stationarity (opposite of ADF):

**Null:** $H_0$: Stationary
**Alternative:** $H_1$: Unit root

**Test statistic:**
$$KPSS = \frac{1}{T^2\hat{\sigma}^2}\sum_{t=1}^{T}S_t^2$$

where $S_t = \sum_{i=1}^{t}(X_i - \bar{X})$ is the partial sum.

**Reject if:** $KPSS > \text{critical value}$ (suggests non-stationary)

**Use:** Often used together with ADF - if both reject their nulls, suggests near-unit-root behavior.

## Trend Stationarity vs Difference Stationarity

### Trend Stationary

A process is trend stationary if:
$$X_t = \mu + \beta t + \epsilon_t$$

where $\epsilon_t$ is stationary. Removing the trend yields stationarity.

**Treatment:** Detrend (regress on time), then analyze residuals.

### Difference Stationary

A process is difference stationary if $\Delta X_t$ is stationary but $X_t$ is not (e.g., random walk).

**Treatment:** Difference the series, then analyze $\Delta X_t$.

### Which to Use?

- **ADF test:** Helps identify difference stationarity
- **Visual inspection:** Plot series and differences
- **Economic theory:** Should guide choice

**Warning:** Wrong treatment leads to spurious results.

## Ljung-Box Test

Tests for remaining autocorrelation in residuals.

### Test Statistic

$$Q = n(n+2)\sum_{h=1}^{m}\frac{\hat{\rho}(h)^2}{n-h}$$

where $m$ is the maximum lag tested.

**Null:** $H_0$: No autocorrelation (white noise)
**Alternative:** $H_1$: Autocorrelation exists

**Distribution:** Under $H_0$, $Q \sim \chi^2(m)$ asymptotically.

**Reject if:** $Q > \chi^2_{\alpha}(m)$

### Use

- Check model residuals for remaining structure
- Validate that model captures autocorrelation

## Durbin-Watson Test

Tests for first-order autocorrelation in residuals (common in regression).

### Test Statistic

$$DW = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n}e_t^2}$$

where $e_t$ are residuals.

**Interpretation:**
- $DW \approx 2$: No autocorrelation
- $DW < 2$: Positive autocorrelation
- $DW > 2$: Negative autocorrelation

**Critical values:** Depend on sample size and number of regressors. Tables provide $d_L$ and $d_U$:
- If $DW < d_L$: Reject $H_0$ (positive autocorrelation)
- If $DW > 4 - d_L$: Reject $H_0$ (negative autocorrelation)
- If $d_L < DW < d_U$ or $4-d_U < DW < 4-d_L$: Inconclusive

**Limitations:** Only tests first-order autocorrelation. Not valid with lagged dependent variables.

## Applications: Testing Stock Returns

### Returns vs Prices

**Stock prices:** Typically non-stationary (random walk)
**Stock returns:** $r_t = \ln(S_t/S_{t-1})$ typically stationary

### Testing Returns

1. **Visual inspection:** Plot returns - should look stationary
2. **ADF test:** Should reject unit root (confirm stationary)
3. **ACF/PACF:** Check for autocorrelation structure
4. **Ljung-Box:** Test for significant autocorrelation

### Typical Findings

- **Returns:** Often show weak autocorrelation (close to white noise)
- **Squared returns:** Show strong autocorrelation (volatility clustering)
- **Absolute returns:** Show autocorrelation (volatility clustering)

This motivates GARCH models for volatility.

## Practical Considerations

### Sample Size

- Small samples: Tests have low power, wide confidence bands
- Large samples: More reliable, but may detect trivial autocorrelation

### Multiple Testing

Testing many lags increases type I error. Consider:
- Bonferroni correction
- False discovery rate control
- Focus on economically meaningful lags

### Model Selection

Use ACF/PACF to guide model selection:
- **AR(p):** PACF cuts off at lag $p$, ACF decays
- **MA(q):** ACF cuts off at lag $q$, PACF decays
- **ARMA(p,q):** Both decay

### Financial Data

Financial returns often show:
- **Weak autocorrelation** in levels (efficient markets)
- **Strong autocorrelation** in squares/absolute values (volatility)
- **Non-stationarity** in prices, stationarity in returns
