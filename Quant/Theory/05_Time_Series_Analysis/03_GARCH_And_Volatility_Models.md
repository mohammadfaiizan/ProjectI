# GARCH and Volatility Models

## ARCH Models

Autoregressive Conditional Heteroskedasticity (ARCH) models capture volatility clustering - periods of high volatility followed by high volatility, and low followed by low.

### Engle's ARCH(q) Model

The ARCH(q) model specifies:

$$X_t = \sigma_t \epsilon_t$$

$$\sigma_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2 + \alpha_2 X_{t-2}^2 + \cdots + \alpha_q X_{t-q}^2$$

where $\epsilon_t \sim \text{IID}(0,1)$ and $\alpha_0 > 0$, $\alpha_i \geq 0$ for $i = 1, \ldots, q$.

**Interpretation:** Variance depends on past squared returns.

### Conditional Heteroskedasticity

The key feature is that variance is **conditional** on past information:

$$\text{Var}(X_t | \mathcal{F}_{t-1}) = \sigma_t^2$$

where $\mathcal{F}_{t-1}$ is the information set at time $t-1$.

This differs from unconditional variance, which may be constant.

### Unconditional Variance

For ARCH(1):
$$\sigma_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2$$

Taking expectations:
$$\mathbb{E}[\sigma_t^2] = \alpha_0 + \alpha_1 \mathbb{E}[X_{t-1}^2] = \alpha_0 + \alpha_1 \mathbb{E}[\sigma_{t-1}^2]$$

If stationary:
$$\sigma^2 = \frac{\alpha_0}{1 - \alpha_1}$$

**Stationarity condition:** $\alpha_1 < 1$ (and $\alpha_0 > 0$).

For ARCH(q), need $\sum_{i=1}^{q}\alpha_i < 1$.

## GARCH Models

GARCH (Generalized ARCH) extends ARCH by including lagged variances.

### Bollerslev's GARCH(p,q) Model

$$X_t = \sigma_t \epsilon_t$$

$$\sigma_t^2 = \alpha_0 + \sum_{i=1}^{q}\alpha_i X_{t-i}^2 + \sum_{j=1}^{p}\beta_j \sigma_{t-j}^2$$

where $\alpha_0 > 0$, $\alpha_i \geq 0$, $\beta_j \geq 0$.

**GARCH(1,1):** Most common specification:
$$\sigma_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

### Persistence

The persistence parameter is:
$$\alpha_1 + \beta_1$$

For GARCH(1,1):
- **Stationarity:** $\alpha_1 + \beta_1 < 1$
- **IGARCH:** $\alpha_1 + \beta_1 = 1$ (integrated, non-stationary variance)
- **High persistence:** Close to 1 means shocks have long-lasting effects

### Unconditional Variance

For GARCH(1,1):
$$\sigma^2 = \frac{\alpha_0}{1 - \alpha_1 - \beta_1}$$

**Interpretation:** Long-run average variance.

### Half-Life

The half-life measures how long it takes for a shock to decay by half:

$$\text{Half-life} = \frac{\ln(0.5)}{\ln(\alpha_1 + \beta_1)}$$

For $\alpha_1 + \beta_1 = 0.95$, half-life $\approx 13.5$ periods.

## EGARCH Model

Exponential GARCH allows asymmetric effects (leverage effect).

### Specification

$$\ln(\sigma_t^2) = \alpha_0 + \sum_{i=1}^{q}\alpha_i g(\epsilon_{t-i}) + \sum_{j=1}^{p}\beta_j \ln(\sigma_{t-j}^2)$$

where:
$$g(\epsilon_t) = \theta \epsilon_t + \gamma(|\epsilon_t| - \mathbb{E}[|\epsilon_t|])$$

**Leverage effect:** $\theta < 0$ means negative returns increase volatility more than positive returns.

**Advantages:**
- No positivity constraints (log ensures $\sigma_t^2 > 0$)
- Captures asymmetry
- More flexible

## GJR-GARCH Model

GJR-GARCH (Glosten-Jagannathan-Runkle) also captures asymmetry:

$$\sigma_t^2 = \alpha_0 + \sum_{i=1}^{q}(\alpha_i + \gamma_i I_{t-i})X_{t-i}^2 + \sum_{j=1}^{p}\beta_j \sigma_{t-j}^2$$

where $I_{t-i} = 1$ if $X_{t-i} < 0$, else $0$.

**Leverage effect:** $\gamma_i > 0$ means negative returns have larger impact.

**Stationarity:** Need $\sum_{i=1}^{q}(\alpha_i + \gamma_i/2) + \sum_{j=1}^{p}\beta_j < 1$.

## GARCH-M Model

GARCH-in-Mean includes volatility in the mean equation:

$$X_t = \mu + \lambda \sigma_t^2 + \sigma_t \epsilon_t$$

$$\sigma_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

**Interpretation:** $\lambda$ is the risk premium - higher volatility requires higher expected return.

**Use:** Model risk-return tradeoff.

## Multivariate GARCH

Multivariate GARCH models volatility and correlation of multiple assets.

### DCC-GARCH

Dynamic Conditional Correlation GARCH:

**Step 1:** Estimate univariate GARCH for each asset:
$$\sigma_{i,t}^2 = \alpha_{i,0} + \alpha_{i,1}X_{i,t-1}^2 + \beta_{i,1}\sigma_{i,t-1}^2$$

**Step 2:** Standardize returns:
$$z_{i,t} = \frac{X_{i,t}}{\sigma_{i,t}}$$

**Step 3:** Model correlation:
$$Q_t = (1 - a - b)\bar{Q} + a z_{t-1}z_{t-1}^T + b Q_{t-1}$$

$$R_t = \text{diag}(Q_t)^{-1/2} Q_t \text{diag}(Q_t)^{-1/2}$$

where $R_t$ is the correlation matrix.

**Advantages:**
- Parsimonious
- Ensures positive definite correlation matrix
- Allows time-varying correlations

### BEKK Model

BEKK ensures positive definiteness:

$$\Sigma_t = CC^T + A X_{t-1}X_{t-1}^T A^T + B \Sigma_{t-1} B^T$$

where $C$, $A$, $B$ are matrices.

**Properties:**
- Always positive definite
- More parameters than DCC
- Harder to estimate

### CCC Model

Constant Conditional Correlation assumes:

$$\Sigma_t = D_t R D_t$$

where $D_t$ is diagonal with GARCH variances, and $R$ is constant correlation matrix.

**Advantages:** Simpler
**Disadvantages:** Unrealistic (correlations do vary)

## Realized Volatility

Realized volatility uses high-frequency data to estimate daily volatility.

### Definition

For intraday returns $r_{i,t}$ on day $t$:
$$\text{RV}_t = \sum_{i=1}^{n}r_{i,t}^2$$

where $n$ is the number of intraday periods.

**Asymptotic theory:** As $n \to \infty$, RV converges to integrated variance:
$$\text{RV}_t \to \int_{t-1}^{t}\sigma^2(s)ds$$

### Range-Based Estimators

Using high-low prices:

**Parkinson estimator:**
$$\hat{\sigma}^2 = \frac{1}{4\ln 2}(\ln H_t - \ln L_t)^2$$

**Garman-Klass:**
$$\hat{\sigma}^2 = 0.5(\ln H_t - \ln L_t)^2 - (2\ln 2 - 1)(\ln C_t - \ln O_t)^2$$

where $H$, $L$, $O$, $C$ are high, low, open, close.

**Advantages:** Use only daily data
**Disadvantages:** Less efficient than RV

### HAR-RV Model

Heterogeneous Autoregressive model for Realized Volatility:

$$\text{RV}_t = \beta_0 + \beta_1 \text{RV}_{t-1} + \beta_5 \text{RV}_{t-5:t-1} + \beta_{22} \text{RV}_{t-22:t-1} + \epsilon_t$$

where $\text{RV}_{t-h:t-1} = \frac{1}{h}\sum_{i=1}^{h}\text{RV}_{t-i}$ is the average over $h$ days.

**Interpretation:** Short-term, medium-term, long-term volatility components.

## Estimation

### Quasi-Maximum Likelihood (QML)

Assume $\epsilon_t \sim N(0,1)$ even if not Gaussian:

$$L = \prod_{t=1}^{T}\frac{1}{\sqrt{2\pi\sigma_t^2}}\exp\left(-\frac{X_t^2}{2\sigma_t^2}\right)$$

$$\ln L = -\frac{T}{2}\ln(2\pi) - \frac{1}{2}\sum_{t=1}^{T}\left(\ln\sigma_t^2 + \frac{X_t^2}{\sigma_t^2}\right)$$

**Properties:**
- Consistent under regularity conditions
- Robust to non-normality
- Standard errors need adjustment (Bollerslev-Wooldridge)

### Initial Values

Need initial values for $\sigma_0^2, \sigma_{-1}^2, \ldots$.

**Options:**
- Use unconditional variance
- Use sample variance
- Estimate as parameters

### Constraints

Ensure positivity and stationarity:
- $\alpha_0 > 0$, $\alpha_i \geq 0$, $\beta_j \geq 0$
- $\sum \alpha_i + \sum \beta_j < 1$

Use constrained optimization or reparameterize.

## Forecasting Volatility

### In-Sample vs Out-of-Sample

**In-sample:** Fit model to all data, evaluate on same data
**Out-of-sample:** Fit on training set, evaluate on test set

Out-of-sample is more realistic but requires data splitting.

### One-Step Forecast

For GARCH(1,1):
$$\hat{\sigma}_{t+1|t}^2 = \alpha_0 + \alpha_1 X_t^2 + \beta_1 \sigma_t^2$$

### Multi-Step Forecast

For GARCH(1,1), $h$-step ahead:
$$\hat{\sigma}_{t+h|t}^2 = \sigma^2 + (\alpha_1 + \beta_1)^{h-1}(\sigma_{t+1|t}^2 - \sigma^2)$$

Forecasts converge to unconditional variance as $h \to \infty$.

### Evaluation Metrics

**MSE:** Mean squared error
$$\text{MSE} = \frac{1}{T}\sum_{t=1}^{T}(\sigma_t^2 - \hat{\sigma}_t^2)^2$$

**MAE:** Mean absolute error
$$\text{MAE} = \frac{1}{T}\sum_{t=1}^{T}|\sigma_t^2 - \hat{\sigma}_t^2|$$

**QLIKE:** Quasi-likelihood
$$\text{QLIKE} = \frac{1}{T}\sum_{t=1}^{T}\left(\ln\hat{\sigma}_t^2 + \frac{\sigma_t^2}{\hat{\sigma}_t^2}\right)$$

**Problem:** True volatility $\sigma_t^2$ is unobserved. Use proxies:
- Squared returns (noisy)
- Realized volatility (if available)
- Range-based estimators

### Mincer-Zarnowitz Regression

Regress realized volatility on forecast:

$$\sigma_t^2 = \alpha + \beta \hat{\sigma}_t^2 + \epsilon_t$$

Test $H_0: \alpha = 0, \beta = 1$ (unbiased forecast).

## Applications

### Value at Risk (VaR)

VaR using GARCH:

$$\text{VaR}_\alpha = -\hat{\sigma}_{t+1|t} \times z_\alpha$$

where $z_\alpha$ is the $\alpha$-quantile of the return distribution.

**Advantages:** Adapts to changing volatility
**Disadvantages:** Assumes distribution shape

### Option Pricing

Use GARCH volatility forecasts in option pricing models.

**GARCH option pricing:** Extend Black-Scholes with GARCH volatility.

### Portfolio Optimization

Use GARCH to forecast covariance matrix for mean-variance optimization.

### Risk Management

- Dynamic hedging (adjust for volatility)
- Position sizing based on volatility forecasts
- Stress testing with volatility scenarios

## Extensions

### Long Memory GARCH

FIGARCH (Fractionally Integrated GARCH) allows long memory in volatility:
$$(1 - \beta L)(1-L)^d \sigma_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2$$

where $0 < d < 1$ is the fractional integration parameter.

### Regime-Switching GARCH

Allow parameters to switch between regimes:
$$\sigma_t^2 = \alpha_{0,s_t} + \alpha_{1,s_t} X_{t-1}^2 + \beta_{1,s_t} \sigma_{t-1}^2$$

where $s_t$ is the regime state.

### Component GARCH

Separate short-term and long-term volatility components.

### Stochastic Volatility

Alternative to GARCH - volatility is unobserved stochastic process (not function of past returns).
