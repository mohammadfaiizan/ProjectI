# Risk Measures: VaR, CVaR, and Stress Testing

## Introduction

Risk measurement is fundamental to portfolio management, regulatory compliance, and risk management. Value at Risk (VaR) and Expected Shortfall (CVaR) are cornerstone risk metrics. This document covers their theoretical foundations, estimation methods, properties, and applications in quantitative finance.

## Value at Risk

### Definition

Value at Risk (VaR) is the maximum loss that will not be exceeded with a given probability over a specified time horizon. Formally, for a portfolio with value $V_t$ and return $R$:

$$VaR_\alpha = -\inf\{x : P(R \leq x) \geq \alpha\}$$

or equivalently:

$$P(R \leq -VaR_\alpha) = \alpha$$

where $\alpha$ is the confidence level (typically 0.01 or 0.05 for 1% or 5% VaR).

For a portfolio value $V_0$, absolute VaR is:
$$VaR_\alpha = V_0 \times VaR_\alpha^{return}$$

### Parametric VaR (Normal Distribution)

If returns are normally distributed: $R \sim N(\mu, \sigma^2)$, then:

$$VaR_\alpha = -(\mu + \sigma \Phi^{-1}(\alpha))$$

where $\Phi^{-1}(\alpha)$ is the $\alpha$-quantile of the standard normal distribution.

For daily returns with $\mu \approx 0$:
$$VaR_\alpha \approx -\sigma \Phi^{-1}(\alpha)$$

Common values:
- $\Phi^{-1}(0.01) = -2.326$ (1% VaR)
- $\Phi^{-1}(0.05) = -1.645$ (5% VaR)

**Example:** Portfolio with $\sigma = 0.02$ (2% daily volatility):
- 1% VaR = $-0.02 \times (-2.326) = 0.0465$ or 4.65%
- 5% VaR = $-0.02 \times (-1.645) = 0.0329$ or 3.29%

### Parametric VaR (t-Distribution)

For fat-tailed returns, the t-distribution may be more appropriate:

$$R \sim t_\nu(\mu, \sigma^2)$$

where $\nu$ is degrees of freedom.

$$VaR_\alpha = -(\mu + \sigma t_\nu^{-1}(\alpha))$$

where $t_\nu^{-1}(\alpha)$ is the $\alpha$-quantile of the t-distribution with $\nu$ degrees of freedom.

For $\nu = 4$:
- $t_4^{-1}(0.01) \approx -3.747$ (vs -2.326 for normal)
- Captures fat tails better

### Historical Simulation VaR

Historical simulation uses empirical quantiles:

1. Collect historical returns: $\{R_1, R_2, \ldots, R_T\}$
2. Sort: $R_{(1)} \leq R_{(2)} \leq \ldots \leq R_{(T)}$
3. VaR is the $\lceil \alpha T \rceil$-th smallest return:

$$VaR_\alpha = -R_{(\lceil \alpha T \rceil)}$$

**Advantages:**
- No distributional assumptions
- Captures fat tails and skewness
- Simple to implement

**Disadvantages:**
- Requires long history
- Assumes past is representative
- Equal weight to all observations

### Monte Carlo VaR

Monte Carlo simulation generates scenarios:

1. Specify return distribution: $R \sim F(\theta)$
2. Generate $N$ scenarios: $\{R^{(1)}, R^{(2)}, \ldots, R^{(N)}\}$
3. Sort scenarios
4. VaR is the $\lceil \alpha N \rceil$-th smallest return

**Advantages:**
- Flexible: any distribution
- Can incorporate dependencies
- Can simulate complex portfolios

**Disadvantages:**
- Computationally intensive
- Model risk: depends on distribution choice

### Portfolio VaR

For a portfolio with weights $\boldsymbol{w}$ and covariance matrix $\boldsymbol{\Sigma}$:

$$\sigma_p^2 = \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}$$

Assuming normal returns:
$$VaR_\alpha = -\sqrt{\boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}} \Phi^{-1}(\alpha)$$

**Diversification effect:** Portfolio VaR is typically less than the sum of individual VaRs due to correlation.

### VaR Scaling

To convert VaR across time horizons:

$$VaR_\alpha(T) = VaR_\alpha(1) \times \sqrt{T}$$

This assumes:
- Returns are i.i.d.
- Normal distribution
- No autocorrelation

For non-normal distributions, scaling is more complex.

## Expected Shortfall (CVaR)

### Definition

Conditional Value at Risk (CVaR), also called Expected Shortfall (ES), is the expected loss conditional on exceeding VaR:

$$CVaR_\alpha = -E[R | R \leq -VaR_\alpha]$$

or equivalently:

$$CVaR_\alpha = -\frac{1}{\alpha} \int_0^\alpha VaR_u \, du$$

### Properties

**For normal distribution:**
If $R \sim N(\mu, \sigma^2)$:
$$CVaR_\alpha = -\mu + \sigma \frac{\phi(\Phi^{-1}(\alpha))}{\alpha}$$

where $\phi$ is the standard normal PDF.

For $\mu = 0$:
$$CVaR_\alpha = \sigma \frac{\phi(\Phi^{-1}(\alpha))}{\alpha}$$

**Example:** For $\alpha = 0.05$:
- $\Phi^{-1}(0.05) = -1.645$
- $\phi(-1.645) = 0.103$
- $CVaR_{0.05} = \sigma \times \frac{0.103}{0.05} = 2.06\sigma$

Compare to $VaR_{0.05} = 1.645\sigma$. CVaR is larger, reflecting tail risk.

### Estimation

**Historical simulation:**
$$CVaR_\alpha = -\frac{1}{\lfloor \alpha T \rfloor} \sum_{i=1}^{\lfloor \alpha T \rfloor} R_{(i)}$$

Average of the worst $\alpha T$ returns.

**Monte Carlo:**
$$CVaR_\alpha = -\frac{1}{\lfloor \alpha N \rfloor} \sum_{i=1}^{\lfloor \alpha N \rfloor} R_{(i)}$$

## Coherent Risk Measures

### Axioms

A risk measure $\rho$ is coherent if it satisfies:

1. **Monotonicity:** If $X \leq Y$ (almost surely), then $\rho(X) \geq \rho(Y)$
2. **Subadditivity:** $\rho(X + Y) \leq \rho(X) + \rho(Y)$
3. **Positive homogeneity:** $\rho(\lambda X) = \lambda \rho(X)$ for $\lambda \geq 0$
4. **Translation invariance:** $\rho(X + c) = \rho(X) - c$ for constant $c$

### VaR and Coherence

VaR violates subadditivity in general. Consider two independent assets with:
- $P(X = 0) = 0.96$, $P(X = -100) = 0.04$
- Same distribution for $Y$

For 5% VaR:
- $VaR_{0.05}(X) = VaR_{0.05}(Y) = 0$ (loss occurs with 4% probability)
- $VaR_{0.05}(X + Y) > 0$ (loss occurs with probability $1 - 0.96^2 = 0.0784 > 0.05$)

Thus: $VaR_{0.05}(X + Y) > VaR_{0.05}(X) + VaR_{0.05}(Y)$

### CVaR and Coherence

CVaR satisfies all four axioms, making it a coherent risk measure.

**Proof sketch:**
- Monotonicity: Clear from definition
- Subadditivity: Follows from convexity of the tail expectation
- Positive homogeneity: $CVaR_\alpha(\lambda X) = \lambda CVaR_\alpha(X)$
- Translation invariance: $CVaR_\alpha(X + c) = CVaR_\alpha(X) - c$

### Spectral Risk Measures

Spectral risk measures generalize CVaR:

$$\rho_\phi(X) = \int_0^1 \phi(u) VaR_u(X) \, du$$

where $\phi$ is a weighting function (risk spectrum) satisfying:
- $\phi(u) \geq 0$
- $\int_0^1 \phi(u) \, du = 1$
- $\phi(u_1) \geq \phi(u_2)$ for $u_1 \leq u_2$ (monotonicity)

CVaR is a special case with:
$$\phi(u) = \begin{cases}
1/\alpha & \text{if } u \leq \alpha \\
0 & \text{if } u > \alpha
\end{cases}$$

## Stress Testing and Scenario Analysis

### Stress Testing

Stress testing evaluates portfolio performance under extreme scenarios:

**Types:**
1. **Historical scenarios:** Past crises (2008, COVID-19)
2. **Hypothetical scenarios:** Tail events not yet observed
3. **Factor shocks:** Extreme moves in risk factors
4. **Reverse stress testing:** Find scenarios causing target loss

### Scenario Construction

**Factor-based scenarios:**
For factor model: $R = \alpha + \sum_{j=1}^k \beta_j F_j + \epsilon$

Stress scenario: $F_j^{stress} = F_j^{normal} + \Delta_j$

Portfolio return: $R^{stress} = \alpha + \sum_{j=1}^k \beta_j F_j^{stress} + \epsilon$

**Correlation breakdown:**
During crises, correlations increase. Adjust correlation matrix:
$$\boldsymbol{\Sigma}^{stress} = (1-\rho) \boldsymbol{\Sigma}^{normal} + \rho \boldsymbol{J}$$

where $\boldsymbol{J}$ is matrix of ones and $\rho$ is stress correlation.

### Scenario Analysis Framework

1. **Identify risk factors:** Market, credit, liquidity, operational
2. **Define scenarios:** Severity, probability, duration
3. **Model impact:** P&L, VaR, capital requirements
4. **Assess adequacy:** Compare to capital, limits
5. **Mitigation:** Hedging, position reduction

## Regulatory Frameworks

### Basel III/IV

Basel III introduced stricter capital requirements:

**Market risk capital:**
- VaR-based: $K = \max(VaR_{t-1}, k \times \frac{1}{60}\sum_{i=1}^{60} VaR_{t-i}) + SR$
- $k$: multiplier (typically 3)
- $SR$: Stressed VaR

**Stressed VaR:**
- VaR calculated using parameters from stressed period
- Captures tail risk better than standard VaR

**Basel IV (FRTB):**
- Expected Shortfall replaces VaR
- Confidence level: 97.5%
- Stressed calibration
- Non-modellable risk factors (NMRF)

### Margin Requirements

**Initial margin:** Collateral required to open position
**Variation margin:** Daily mark-to-market adjustments

**SPAN margin:** Standard Portfolio Analysis of Risk
- Scenarios across price and volatility moves
- Maximum loss across scenarios

**VaR-based margin:**
$$Margin = VaR_\alpha \times Multiplier$$

## Backtesting VaR

### Kupiec Test

Tests if VaR violations occur with correct frequency.

**Null hypothesis:** $H_0: p = \alpha$ where $p$ is true violation rate.

**Test statistic:**
$$LR = -2 \ln \left( \frac{(1-\alpha)^{T-x} \alpha^x}{(1-\hat{p})^{T-x} \hat{p}^x} \right)$$

where:
- $T$: number of observations
- $x$: number of violations
- $\hat{p} = x/T$: sample violation rate

Under $H_0$, $LR \sim \chi^2(1)$.

**Decision rule:** Reject $H_0$ if $LR > \chi^2_{1-\gamma}(1)$.

### Christoffersen Test

Tests independence of violations (no clustering).

**Hypotheses:**
- $H_0^1$: Unconditional coverage (Kupiec test)
- $H_0^2$: Independence
- $H_0^3$: Conditional coverage (both)

**Transition matrix:**
$$\Pi = \begin{bmatrix}
\pi_{00} & \pi_{01} \\
\pi_{10} & \pi_{11}
\end{bmatrix}$$

where $\pi_{ij} = P(I_t = j | I_{t-1} = i)$ and $I_t$ is violation indicator.

**Test statistic:**
$$LR_{ind} = -2 \ln \left( \frac{(1-\hat{\pi})^n \hat{\pi}^m}{(1-\hat{\pi}_0)^{n_0} \hat{\pi}_0^{m_0} (1-\hat{\pi}_1)^{n_1} \hat{\pi}_1^{m_1}} \right)$$

where:
- $n_0, m_0$: transitions from non-violation
- $n_1, m_1$: transitions from violation
- $\hat{\pi} = (m_0 + m_1)/(n_0 + n_1 + m_0 + m_1)$

Under independence, $LR_{ind} \sim \chi^2(1)$.

### Traffic Light Approach

Basel framework uses traffic light zones:

- **Green zone:** 0-4 violations (out of 250 days) - multiplier = 3
- **Yellow zone:** 5-9 violations - multiplier increases
- **Red zone:** 10+ violations - multiplier = 4, model review required

## Example: VaR Calculation

Consider a portfolio with:
- Value: $V_0 = \$1,000,000$
- Daily return: $R \sim N(0.0005, 0.02^2)$ (mean 0.05%, std 2%)

**1% VaR (1-day):**
$$VaR_{0.01} = -(\mu + \sigma \Phi^{-1}(0.01)) = -(0.0005 + 0.02 \times (-2.326)) = 0.04702$$

Absolute VaR: $\$1,000,000 \times 0.04702 = \$47,020$

**5% VaR (1-day):**
$$VaR_{0.05} = -(0.0005 + 0.02 \times (-1.645)) = 0.0334$$

Absolute VaR: $\$33,400$

**10-day VaR (scaled):**
$$VaR_{0.01}(10) = VaR_{0.01}(1) \times \sqrt{10} = 0.04702 \times 3.162 = 0.1487$$

Absolute VaR: $\$148,700$

**CVaR (1%, 1-day):**
$$CVaR_{0.01} = -0.0005 + 0.02 \times \frac{\phi(-2.326)}{0.01} = -0.0005 + 0.02 \times \frac{0.02665}{0.01} = 0.0528$$

Absolute CVaR: $\$52,800$

CVaR exceeds VaR, reflecting tail risk beyond the VaR threshold.
