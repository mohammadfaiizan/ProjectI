# Modern Portfolio Theory and CAPM

## Introduction

Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952, revolutionized finance by providing a mathematical framework for portfolio construction. The theory demonstrates that investors can achieve superior risk-return trade-offs through diversification. The Capital Asset Pricing Model (CAPM), developed by Sharpe, Lintner, and Mossin, extends MPT to provide a theoretical foundation for asset pricing and expected returns.

## Markowitz Mean-Variance Optimization

### The Optimization Problem

Given $n$ assets with expected returns $\boldsymbol{\mu} = (\mu_1, \mu_2, \ldots, \mu_n)^T$ and covariance matrix $\boldsymbol{\Sigma}$, the portfolio optimization problem seeks to find weights $\boldsymbol{w} = (w_1, w_2, \ldots, w_n)^T$ that minimize portfolio variance for a given expected return, or maximize expected return for a given variance.

The portfolio expected return is:
$$\mu_p = \boldsymbol{w}^T \boldsymbol{\mu} = \sum_{i=1}^n w_i \mu_i$$

The portfolio variance is:
$$\sigma_p^2 = \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}$$

where $\sigma_{ij}$ is the covariance between assets $i$ and $j$.

### The Efficient Frontier

The efficient frontier is the set of portfolios that offer the highest expected return for a given level of risk, or equivalently, the lowest risk for a given expected return. Mathematically, for a target return $\mu_0$, we solve:

$$\min_{\boldsymbol{w}} \frac{1}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}$$

subject to:
$$\boldsymbol{w}^T \boldsymbol{\mu} = \mu_0$$
$$\boldsymbol{w}^T \boldsymbol{1} = 1$$

where $\boldsymbol{1}$ is a vector of ones (the budget constraint).

Using Lagrange multipliers, the solution is:
$$\boldsymbol{w}^* = \frac{1}{D} \left[ B \boldsymbol{\Sigma}^{-1} \boldsymbol{1} - A \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} \right] + \frac{\mu_0}{D} \left[ C \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} - A \boldsymbol{\Sigma}^{-1} \boldsymbol{1} \right]$$

where:
- $A = \boldsymbol{1}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$
- $B = \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}$
- $C = \boldsymbol{1}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{1}$
- $D = BC - A^2$

The efficient frontier in mean-variance space is a hyperbola:
$$\sigma_p^2 = \frac{C \mu_p^2 - 2A \mu_p + B}{D}$$

### Tangency Portfolio

When a risk-free asset with return $r_f$ is available, the efficient frontier becomes a straight line. The tangency portfolio is the portfolio of risky assets that maximizes the Sharpe ratio:

$$\max_{\boldsymbol{w}} \frac{\boldsymbol{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}}}$$

The solution is:
$$\boldsymbol{w}_{tan} = \frac{\boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f \boldsymbol{1})}{\boldsymbol{1}^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f \boldsymbol{1})}$$

The tangency portfolio has the property that all investors, regardless of risk aversion, hold a combination of the risk-free asset and this portfolio.

## Two-Fund and One-Fund Theorems

### Two-Fund Separation Theorem

The two-fund separation theorem states that any efficient portfolio can be constructed as a combination of two efficient portfolios. If $\boldsymbol{w}_1$ and $\boldsymbol{w}_2$ are two efficient portfolios with different expected returns, then any efficient portfolio $\boldsymbol{w}$ can be written as:

$$\boldsymbol{w} = \alpha \boldsymbol{w}_1 + (1-\alpha) \boldsymbol{w}_2$$

for some $\alpha \in \mathbb{R}$.

This implies that all investors need only consider two mutual funds, regardless of their risk preferences.

### One-Fund Theorem

When a risk-free asset exists, the one-fund theorem states that all efficient portfolios are combinations of the risk-free asset and the tangency portfolio. Any efficient portfolio $\boldsymbol{w}_e$ can be written as:

$$\boldsymbol{w}_e = w_f \cdot 0 + (1-w_f) \boldsymbol{w}_{tan}$$

where $w_f$ is the weight in the risk-free asset.

## Capital Market Line

The Capital Market Line (CML) is the line connecting the risk-free rate to the tangency portfolio in mean-standard deviation space:

$$E[R_p] = r_f + \frac{E[R_{tan}] - r_f}{\sigma_{tan}} \sigma_p$$

where $R_{tan}$ and $\sigma_{tan}$ are the return and standard deviation of the tangency portfolio.

The slope of the CML is the Sharpe ratio of the tangency portfolio:
$$SR_{tan} = \frac{E[R_{tan}] - r_f}{\sigma_{tan}}$$

All portfolios on the CML are efficient, and any portfolio below the CML is dominated.

## Security Market Line

The Security Market Line (SML) relates the expected return of an asset to its beta:

$$E[R_i] = r_f + \beta_i (E[R_m] - r_f)$$

where:
- $\beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)} = \frac{\sigma_{im}}{\sigma_m^2}$
- $R_m$ is the market portfolio return

The SML represents the relationship between risk (beta) and expected return for all assets, whether efficient or not.

## Capital Asset Pricing Model

### Derivation

CAPM assumes:
1. Investors are mean-variance optimizers
2. All investors have homogeneous expectations
3. All investors can borrow and lend at the risk-free rate
4. No taxes or transaction costs
5. All assets are perfectly divisible
6. Markets are efficient

Under these assumptions, the market portfolio is the tangency portfolio. For any asset $i$, consider a portfolio $p$ consisting of asset $i$ and the market portfolio $m$:

$$R_p = w_i R_i + (1-w_i) R_m$$

The portfolio variance is:
$$\sigma_p^2 = w_i^2 \sigma_i^2 + (1-w_i)^2 \sigma_m^2 + 2w_i(1-w_i)\sigma_{im}$$

At the optimum, the derivative with respect to $w_i$ must satisfy:
$$\frac{\partial \sigma_p^2}{\partial w_i} \bigg|_{w_i=0} = -2\sigma_m^2 + 2\sigma_{im} = 0$$

This gives $\sigma_{im} = \sigma_m^2$, which implies:
$$\beta_i = \frac{\sigma_{im}}{\sigma_m^2} = 1$$

For the market portfolio, $\beta_m = 1$. The expected return-beta relationship follows:
$$E[R_i] = r_f + \beta_i (E[R_m] - r_f)$$

### Beta

Beta measures the sensitivity of an asset's returns to market movements:

$$\beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$$

Beta can be estimated via regression:
$$R_{i,t} - r_{f,t} = \alpha_i + \beta_i (R_{m,t} - r_{f,t}) + \epsilon_{i,t}$$

Properties:
- $\beta = 1$: asset moves with the market
- $\beta > 1$: asset is more volatile than the market
- $\beta < 1$: asset is less volatile than the market
- $\beta < 0$: asset moves opposite to the market

### Alpha

Alpha measures the excess return of an asset beyond what is predicted by CAPM:

$$\alpha_i = E[R_i] - [r_f + \beta_i (E[R_m] - r_f)]$$

In the regression framework:
$$R_{i,t} - r_{f,t} = \alpha_i + \beta_i (R_{m,t} - r_{f,t}) + \epsilon_{i,t}$$

A positive alpha indicates outperformance relative to the market, while a negative alpha indicates underperformance.

### CAPM Testing

Testing CAPM involves several approaches:

**Time-series regression:**
$$R_{i,t} - r_{f,t} = \alpha_i + \beta_i (R_{m,t} - r_{f,t}) + \epsilon_{i,t}$$

CAPM predicts $\alpha_i = 0$ for all assets.

**Cross-sectional regression:**
$$\bar{R}_i - \bar{r}_f = \gamma_0 + \gamma_1 \hat{\beta}_i + u_i$$

CAPM predicts $\gamma_0 = 0$ and $\gamma_1 = \bar{R}_m - \bar{r}_f$.

**Fama-MacBeth procedure:**
1. Estimate betas using rolling windows
2. Form portfolios sorted by beta
3. Run cross-sectional regressions each period
4. Test if average $\gamma_0 = 0$ and $\gamma_1 = \bar{R}_m - \bar{r}_f$

Empirical evidence shows:
- Beta explains only a small portion of cross-sectional return variation
- Other factors (size, value, momentum) appear significant
- Alpha estimates are often non-zero

## Risk-Adjusted Performance Metrics

### Sharpe Ratio

The Sharpe ratio measures excess return per unit of risk:

$$SR = \frac{E[R_p] - r_f}{\sigma_p}$$

where $\sigma_p$ is the standard deviation of portfolio returns.

Interpretation:
- Higher Sharpe ratio indicates better risk-adjusted performance
- Typically annualized: $SR_{annual} = SR_{period} \times \sqrt{252}$ (for daily returns)

### Information Ratio

The Information Ratio measures active return per unit of active risk:

$$IR = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)} = \frac{\alpha}{\sigma(\epsilon)}$$

where $R_b$ is the benchmark return and $\sigma(\epsilon)$ is tracking error.

The IR is related to the t-statistic of alpha:
$$t(\alpha) = IR \times \sqrt{T}$$

where $T$ is the number of periods.

### Sortino Ratio

The Sortino ratio uses downside deviation instead of total volatility:

$$Sortino = \frac{E[R_p] - r_f}{\sigma_{down}}$$

where:
$$\sigma_{down} = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(R_{p,t} - r_f, 0)^2}$$

This metric penalizes only negative deviations from the risk-free rate.

## Limitations of Mean-Variance Optimization

### Estimation Error

Mean-variance optimization is highly sensitive to estimation errors in expected returns and covariances. Small changes in inputs can lead to dramatically different optimal portfolios.

**The problem:**
- Expected returns are notoriously difficult to estimate
- Sample covariance matrices require $T > n$ observations
- Out-of-sample performance often deteriorates significantly

**Solutions:**
- Shrinkage estimators for covariance matrices (Ledoit-Wolf)
- Black-Litterman model for expected returns
- Robust optimization techniques
- Resampling methods

### Non-Normal Returns

Mean-variance optimization assumes returns are normally distributed, but empirical evidence shows:
- Fat tails and skewness
- Time-varying volatility (GARCH effects)
- Regime changes

**Impact:**
- Variance may not fully capture risk
- Higher moments (skewness, kurtosis) matter
- Downside risk measures may be more appropriate

### Transaction Costs

The basic model ignores transaction costs, which can significantly impact realized returns, especially for high-turnover strategies.

### Single-Period Framework

The model is static and doesn't account for:
- Multi-period rebalancing
- Dynamic strategies
- Time-varying risk preferences

## Example: Portfolio Optimization

Consider three assets with:
- Expected returns: $\mu = [0.10, 0.12, 0.08]^T$
- Covariance matrix:
$$\boldsymbol{\Sigma} = \begin{bmatrix}
0.04 & 0.01 & 0.005 \\
0.01 & 0.05 & 0.01 \\
0.005 & 0.01 & 0.03
\end{bmatrix}$$
- Risk-free rate: $r_f = 0.03$

The tangency portfolio weights are:
$$\boldsymbol{w}_{tan} = \frac{\boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f \boldsymbol{1})}{\boldsymbol{1}^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - r_f \boldsymbol{1})}$$

Computing:
$$\boldsymbol{\mu} - r_f \boldsymbol{1} = [0.07, 0.09, 0.05]^T$$

After matrix inversion and multiplication:
$$\boldsymbol{w}_{tan} \approx [0.35, 0.50, 0.15]^T$$

The tangency portfolio has:
- Expected return: $\mu_{tan} \approx 0.111$
- Standard deviation: $\sigma_{tan} \approx 0.198$
- Sharpe ratio: $SR \approx 0.409$

Any efficient portfolio can be constructed as:
$$R_p = w_f \cdot 0.03 + (1-w_f) R_{tan}$$

where $w_f$ determines the risk-return trade-off.
