# Monte Carlo Simulation

## Introduction

Monte Carlo simulation is a fundamental numerical method in quantitative finance, used for option pricing, risk measurement, and complex derivative valuation. This document covers random number generation, quasi-random sequences, Monte Carlo integration, path generation, and applications.

## Random Number Generation

### Uniform Random Numbers

Uniform random numbers $U \sim \text{Uniform}(0,1)$ are the foundation.

**Linear congruential generator (LCG):**
$$X_{n+1} = (a X_n + c) \bmod m$$

where $a$, $c$, $m$ are parameters.

**Properties:**
- Period: Maximum $m$
- Quality depends on parameters
- Simple but limited

**Modern generators:**
- Mersenne Twister: Period $2^{19937}-1$
- Xorshift: Fast, good quality
- Cryptographic generators: For security

### Normal Random Numbers

#### Box-Muller Transform

Generate two independent standard normals from two uniforms:

$$Z_0 = \sqrt{-2 \ln U_1} \cos(2\pi U_2)$$
$$Z_1 = \sqrt{-2 \ln U_1} \sin(2\pi U_2)$$

where $U_1, U_2 \sim \text{Uniform}(0,1)$ are independent.

**Properties:**
- Exact (no approximation)
- Produces pairs
- Requires trigonometric functions

#### Ziggurat Algorithm

More efficient rejection sampling method:

1. Divide standard normal PDF into rectangles
2. Generate uniform random point
3. Accept/reject based on position
4. Handle tail region separately

**Advantages:**
- Faster than Box-Muller
- Avoids trigonometric functions
- Widely used in practice

### Correlated Normal Random Numbers

For vector $\boldsymbol{Z} \sim N(\boldsymbol{0}, \boldsymbol{\Sigma})$:

**Cholesky decomposition:**
$$\boldsymbol{\Sigma} = \boldsymbol{L} \boldsymbol{L}^T$$

where $\boldsymbol{L}$ is lower triangular.

**Generation:**
$$\boldsymbol{Z} = \boldsymbol{L} \boldsymbol{X}$$

where $\boldsymbol{X} \sim N(\boldsymbol{0}, \boldsymbol{I})$ (independent standard normals).

**Example:** For 2D with correlation $\rho$:
$$\boldsymbol{L} = \begin{bmatrix}
1 & 0 \\
\rho & \sqrt{1-\rho^2}
\end{bmatrix}$$

Then:
$$Z_1 = X_1$$
$$Z_2 = \rho X_1 + \sqrt{1-\rho^2} X_2$$

## Quasi-Random Sequences

### Low-Discrepancy Sequences

Quasi-random sequences fill space more uniformly than random sequences.

**Discrepancy:**
$$D_N^* = \sup_{\boldsymbol{x} \in [0,1]^d} \left| \frac{\#\{\boldsymbol{u}_i \in [0,\boldsymbol{x}]\}}{N} - \prod_{j=1}^d x_j \right|$$

Lower discrepancy → better coverage.

### Sobol Sequences

Sobol sequences are a popular low-discrepancy sequence.

**Properties:**
- Discrepancy: $O((\log N)^d / N)$
- Deterministic (reproducible)
- Good for up to moderate dimensions

**Construction:**
- Based on primitive polynomials
- Direction numbers determine sequence
- Requires initialization

### Halton Sequences

Halton sequences use different bases for each dimension:

**Construction:**
For dimension $j$ with base $p_j$ (prime):
$$u_n^{(j)} = \sum_{k=0}^{\infty} a_k^{(j)} p_j^{-(k+1)}$$

where $n = \sum_k a_k^{(j)} p_j^k$ is base-$p_j$ expansion.

**Properties:**
- Simple construction
- Good for low dimensions
- Degrades in high dimensions

### Faure Sequences

Faure sequences use same base for all dimensions:

**Properties:**
- Better than Halton in higher dimensions
- More complex construction
- Good theoretical properties

## Monte Carlo Integration

### Basic Method

Estimate integral:
$$I = \int_{\Omega} f(\boldsymbol{x}) d\boldsymbol{x}$$

**Monte Carlo estimator:**
$$\hat{I}_N = \frac{1}{N} \sum_{i=1}^N f(\boldsymbol{X}_i)$$

where $\boldsymbol{X}_i$ are random samples from uniform distribution over $\Omega$.

**For general domain:**
$$\hat{I}_N = \frac{|\Omega|}{N} \sum_{i=1}^N f(\boldsymbol{X}_i)$$

where $|\Omega|$ is volume of domain.

### Convergence Rate

**Standard MC:**
$$E[(\hat{I}_N - I)^2] = \frac{\sigma^2}{N}$$

where $\sigma^2 = \text{Var}(f(\boldsymbol{X}))$.

**Convergence:** $O(1/\sqrt{N})$

**Confidence interval:**
$$\hat{I}_N \pm z_{\alpha/2} \frac{\sigma}{\sqrt{N}}$$

where $z_{\alpha/2}$ is standard normal quantile.

### Quasi-Monte Carlo

**QMC estimator:**
$$\hat{I}_N^{QMC} = \frac{1}{N} \sum_{i=1}^N f(\boldsymbol{u}_i)$$

where $\boldsymbol{u}_i$ are quasi-random points.

**Convergence:** $O((\log N)^d / N)$ for Sobol

**Advantages:**
- Faster convergence (for smooth $f$)
- Deterministic (reproducible)
- Better error bounds

**Disadvantages:**
- Requires smooth integrand
- Error bounds depend on dimension
- May not work well for discontinuous functions

### Effective Dimension

**Notion:** Function may effectively depend on fewer dimensions.

**Truncation dimension:** First $s$ dimensions explain most variance.

**Superposition dimension:** Sum of low-dimensional functions.

**QMC benefits:** Most when effective dimension is low.

## Path Generation

### Geometric Brownian Motion

For stock price following:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

**Exact solution:**
$$S_t = S_0 \exp\left((\mu - \frac{\sigma^2}{2})t + \sigma W_t\right)$$

**Discretization:**
$$S_{t+\Delta t} = S_t \exp\left((\mu - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z\right)$$

where $Z \sim N(0,1)$.

**Path generation:**
```python
for i in range(N_steps):
    Z = normal_random()
    S[i+1] = S[i] * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
```

### Correlated Assets

For $n$ assets with correlation matrix $\boldsymbol{\rho}$:

**Cholesky:**
$$\boldsymbol{\rho} = \boldsymbol{L} \boldsymbol{L}^T$$

**Path generation:**
$$\boldsymbol{S}_{t+\Delta t} = \boldsymbol{S}_t \odot \exp\left((\boldsymbol{\mu} - \frac{1}{2}\boldsymbol{\sigma}^2)\Delta t + \sqrt{\Delta t} \boldsymbol{L} \boldsymbol{Z}\right)$$

where $\odot$ is element-wise multiplication and $\boldsymbol{Z} \sim N(\boldsymbol{0}, \boldsymbol{I})$.

### Stochastic Volatility Models

**Heston model:**
$$dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t^S$$
$$dV_t = \kappa(\theta - V_t)dt + \sigma_V \sqrt{V_t} dW_t^V$$

with correlation: $dW_t^S dW_t^V = \rho dt$.

**Discretization (Euler):**
$$V_{t+\Delta t} = V_t + \kappa(\theta - V_t)\Delta t + \sigma_V \sqrt{V_t} \sqrt{\Delta t} Z_V$$

**Issue:** $V_t$ may become negative.

**Fix:** Use full truncation or reflection:
$$V_{t+\Delta t} = \max(V_t + \kappa(\theta - V_t)\Delta t + \sigma_V \sqrt{V_t} \sqrt{\Delta t} Z_V, 0)$$

### Jump Processes

**Merton jump-diffusion:**
$$dS_t = \mu S_t dt + \sigma S_t dW_t + S_t dJ_t$$

where $J_t$ is compound Poisson process.

**Simulation:**
1. Generate Poisson jumps: $N_t \sim \text{Poisson}(\lambda t)$
2. Generate jump sizes: $Y_i \sim N(\mu_J, \sigma_J^2)$
3. Combine: $S_t = S_0 \exp((\mu - \sigma^2/2)t + \sigma W_t + \sum_{i=1}^{N_t} Y_i)$

## Applications

### Option Pricing

**European call:**
$$C = e^{-rT} E[\max(S_T - K, 0)]$$

**Monte Carlo:**
1. Generate $N$ paths: $S_T^{(i)}$
2. Compute payoffs: $P^{(i)} = \max(S_T^{(i)} - K, 0)$
3. Estimate: $\hat{C} = e^{-rT} \frac{1}{N} \sum_{i=1}^N P^{(i)}$

**Standard error:**
$$SE = e^{-rT} \frac{\sigma_P}{\sqrt{N}}$$

where $\sigma_P$ is standard deviation of payoffs.

**American options:** Require dynamic programming (see other methods).

### Value at Risk

**Portfolio VaR:**
1. Generate scenarios: $\boldsymbol{R}^{(i)} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
2. Compute portfolio returns: $R_p^{(i)} = \boldsymbol{w}^T \boldsymbol{R}^{(i)}$
3. Sort: $R_p^{(1)} \leq R_p^{(2)} \leq \ldots \leq R_p^{(N)}$
4. VaR: $-R_p^{(\lceil \alpha N \rceil)}$

**With non-normal returns:** Use appropriate distribution or historical simulation.

### Credit Risk

**Portfolio credit loss:**
$$L = \sum_{i=1}^n LGD_i \times \mathbb{1}(\text{default}_i)$$

where $LGD_i$ is loss given default.

**Simulation:**
1. Generate default indicators (correlated)
2. Compute losses
3. Estimate distribution

**Correlated defaults:** Use copula models (Gaussian, t-copula).

## Greek Estimation via Monte Carlo

### Bump-and-Revalue

**Delta:**
$$\Delta \approx \frac{V(S_0 + \epsilon) - V(S_0 - \epsilon)}{2\epsilon}$$

**Vega:**
$$\text{Vega} \approx \frac{V(\sigma_0 + \epsilon) - V(\sigma_0 - \epsilon)}{2\epsilon}$$

**Method:**
1. Price with original parameters
2. Bump parameter
3. Re-price
4. Compute finite difference

**Disadvantages:**
- Requires multiple MC runs
- Expensive
- Numerical error from finite differences

### Pathwise Method

**Delta for European call:**
$$\Delta = e^{-rT} E\left[\mathbb{1}(S_T > K) \frac{S_T}{S_0}\right]$$

**Derivation:**
$$\frac{\partial}{\partial S_0} E[\max(S_T - K, 0)] = E\left[\mathbb{1}(S_T > K) \frac{\partial S_T}{\partial S_0}\right]$$

Since $S_T = S_0 e^{(r-\sigma^2/2)T + \sigma W_T}$:
$$\frac{\partial S_T}{\partial S_0} = \frac{S_T}{S_0}$$

**Advantages:**
- Single MC run
- Lower variance than bump-and-revalue
- Exact (no finite difference error)

**Limitations:**
- Requires smooth payoff
- May not work for path-dependent options

### Likelihood Ratio Method

**Delta:**
$$\Delta = E\left[\max(S_T - K, 0) \frac{\partial \ln f(S_T)}{\partial S_0}\right]$$

where $f$ is density of $S_T$.

**For GBM:**
$$\frac{\partial \ln f(S_T)}{\partial S_0} = \frac{W_T}{S_0 \sigma T}$$

**Advantages:**
- Works for discontinuous payoffs
- Single MC run

**Disadvantages:**
- Higher variance
- Requires density

### Combined Methods

**Vega (pathwise + LR):**
For Heston model, combine methods:
- Pathwise for $S_T$ dependence
- LR for volatility process

## Example: European Option Pricing

Price European call with:
- $S_0 = 100$
- $K = 105$
- $r = 0.05$
- $\sigma = 0.2$
- $T = 1$

**Monte Carlo (10,000 paths):**

```python
N = 10000
payoffs = []
for i in range(N):
    Z = normal_random()
    S_T = 100 * exp((0.05 - 0.5*0.2^2)*1 + 0.2*sqrt(1)*Z)
    payoff = max(S_T - 105, 0)
    payoffs.append(payoff)

price = exp(-0.05*1) * mean(payoffs)
std_error = exp(-0.05*1) * std(payoffs) / sqrt(N)
```

**Result:**
- Price: $\$8.02$
- Standard error: $\$0.08$
- 95% CI: $[7.86, 8.18]$

**Black-Scholes price:** $\$8.02$ (matches!)

**With quasi-random (Sobol):**
- Price: $\$8.02$
- Standard error: $\$0.03$ (lower!)

Quasi-random reduces variance, providing more accurate estimates with fewer samples.
