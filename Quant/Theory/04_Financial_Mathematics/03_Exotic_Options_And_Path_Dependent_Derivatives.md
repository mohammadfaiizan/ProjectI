# Exotic Options and Path-Dependent Derivatives

## Introduction

Exotic options have payoffs that depend on the path of the underlying asset price, not just its terminal value. This path dependence makes pricing more complex and often requires numerical methods.

## Barrier Options

Barrier options are activated or deactivated when the underlying crosses a predetermined barrier level $H$.

### Types of Barrier Options

**Out options** (knock-out):
- **Up-and-out call:** Cancelled if $S(t) \geq H$ for any $t \in [0,T]$
- **Down-and-out put:** Cancelled if $S(t) \leq H$ for any $t \in [0,T]$

**In options** (knock-in):
- **Up-and-in call:** Activated only if $S(t) \geq H$ for some $t \in [0,T]$
- **Down-and-in put:** Activated only if $S(t) \leq H$ for some $t \in [0,T]$

### Parity Relationships

Barrier options satisfy parity relationships:
$$\text{Vanilla} = \text{Knock-out} + \text{Knock-in}$$

For example:
$$C_{vanilla} = C_{up-and-out} + C_{up-and-in}$$

This follows from the fact that either the barrier is hit (activating the in-option) or not (keeping the out-option alive).

### Pricing via Reflection Principle

The reflection principle uses symmetry properties of Brownian motion. For a down-and-out call with barrier $H < S_0$:

The probability that the barrier is not hit and $S(T) > K$ involves:
1. Direct path probability: $P(S(T) > K, \min_{0 \leq t \leq T} S(t) > H)$
2. Reflected path probability: Paths hitting the barrier are reflected

Using the reflection principle, the price is:
$$C_{DO} = C_{BS}(S_0, K) - \left(\frac{H}{S_0}\right)^{2\lambda} C_{BS}\left(\frac{H^2}{S_0}, K\right)$$

where $\lambda = \frac{r - \sigma^2/2}{\sigma^2}$ and $C_{BS}$ is the Black-Scholes call price.

### Rebates

Barrier options often include rebates paid if the barrier is hit:
- **Rebate at hit:** Paid immediately when barrier is crossed
- **Rebate at expiry:** Paid at maturity if barrier was hit

The rebate value is:
$$R_{hit} = R \cdot e^{-r\tau_H}$$

where $\tau_H$ is the first hitting time of the barrier.

### Continuity Correction

For discrete barrier monitoring (e.g., daily closes), a continuity correction adjusts the barrier:
$$H_{adjusted} = H \cdot e^{\pm \beta \sigma \sqrt{\Delta t}}$$

where $\beta \approx 0.5826$ (Siegert's constant) and the sign depends on whether monitoring is from above or below.

## Asian Options

Asian options have payoffs based on the average price of the underlying over a period.

### Types of Asian Options

**Arithmetic average:**
$$A_T = \frac{1}{n}\sum_{i=1}^{n} S(t_i)$$

**Geometric average:**
$$G_T = \left(\prod_{i=1}^{n} S(t_i)\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \ln S(t_i)\right)$$

**Fixed strike:**
- Call: $\max(A_T - K, 0)$
- Put: $\max(K - A_T, 0)$

**Floating strike:**
- Call: $\max(S_T - A_T, 0)$
- Put: $\max(A_T - S_T, 0)$

### Pricing Geometric Average Options

Geometric averages are easier to price because $\ln G_T$ is normally distributed.

For a geometric average call:
$$G_T = S_0 \exp\left(\frac{1}{n}\sum_{i=1}^{n} \left((r - \frac{\sigma^2}{2})t_i + \sigma W(t_i)\right)\right)$$

The variance of $\ln G_T$ is:
$$\sigma_G^2 = \sigma^2 \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n} \min(t_i, t_j)$$

For continuous averaging over $[0,T]$:
$$\sigma_G^2 = \frac{\sigma^2 T}{3}$$

The geometric average option can be priced using Black-Scholes with adjusted volatility:
$$C_{geometric} = S_0 e^{(r_G - r)T}N(d_1) - Ke^{-rT}N(d_2)$$

where $r_G = r - \frac{\sigma^2}{2} + \frac{\sigma_G^2}{2}$.

### Pricing Arithmetic Average Options

Arithmetic averages have no closed-form solution because the sum of log-normal variables is not log-normal.

**Approximation methods:**

1. **Moment matching:** Match first two moments to a log-normal distribution
2. **Edgeworth expansion:** Expand around the log-normal approximation
3. **Monte Carlo:** Most accurate but computationally expensive

**Levy's approximation:**
$$C_{arithmetic} \approx S_0 e^{(r_A - r)T}N(d_1) - Ke^{-rT}N(d_2)$$

where $r_A$ and $\sigma_A$ are chosen to match moments of $A_T$.

### Control Variate Method

Use geometric average as control variate:
$$C_{arithmetic} = C_{geometric} + \mathbb{E}[C_{arithmetic} - C_{geometric}]$$

The expectation is estimated via Monte Carlo, reducing variance.

## Lookback Options

Lookback options have payoffs based on the maximum or minimum price over the option's life.

### Floating Strike Lookback

**Call:** $\max(S_T - m_T, 0) = S_T - m_T$ (always in-the-money)
where $m_T = \min_{0 \leq t \leq T} S(t)$

**Put:** $\max(M_T - S_T, 0) = M_T - S_T$ (always in-the-money)
where $M_T = \max_{0 \leq t \leq T} S(t)$

### Fixed Strike Lookback

**Call:** $\max(M_T - K, 0)$
**Put:** $\max(K - m_T, 0)$

### Pricing Floating Strike Lookback

For a floating strike lookback call, the price is:
$$C_{floating} = S_0 N(a_1) - S_0 e^{-rT}N(a_2) + S_0 \frac{\sigma^2}{2r}\left[\left(\frac{S_0}{m_0}\right)^{-2r/\sigma^2}N(-a_3) - e^{-rT}N(-a_1)\right]$$

where $m_0$ is the current minimum and $a_1, a_2, a_3$ are functions of $S_0, m_0, T, r, \sigma$.

### Partial Lookback Options

Partial lookback options use the maximum/minimum over a subperiod $[T_1, T]$:
$$M_{T_1,T} = \max_{T_1 \leq t \leq T} S(t)$$

Pricing requires conditioning on $S(T_1)$ and $M_{T_1,T}$.

## Digital/Binary Options

Digital options pay a fixed amount if a condition is met.

### Cash-or-Nothing

**Call:** Pays $Q$ if $S(T) > K$, else $0$
**Put:** Pays $Q$ if $S(T) < K$, else $0$

Price under Black-Scholes:
$$C_{cash} = Qe^{-rT}N(d_2)$$

where $d_2 = \frac{\ln(S/K) + (r - \sigma^2/2)T}{\sigma\sqrt{T}}$.

### Asset-or-Nothing

**Call:** Pays $S(T)$ if $S(T) > K$, else $0$
**Put:** Pays $S(T)$ if $S(T) < K$, else $0$

Price:
$$C_{asset} = S_0 N(d_1)$$

Note: $C_{vanilla} = C_{asset} - K \cdot C_{cash}/Q$ (decomposition of vanilla option).

### One-Touch Options

One-touch options pay if the barrier is hit at any time before expiration.

For an up-and-in one-touch:
$$P_{one-touch} = \left(\frac{H}{S_0}\right)^{2\lambda} N\left(\frac{\ln(H^2/(S_0 K)) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}\right) + e^{-rT}N\left(\frac{\ln(S_0/H) - (r + \sigma^2/2)T}{\sigma\sqrt{T}}\right)$$

## Basket Options

Basket options have payoffs based on a weighted average of multiple underlying assets.

### Payoff

For a basket of $n$ assets:
$$B_T = \sum_{i=1}^{n} w_i S_i(T)$$

Call payoff: $\max(B_T - K, 0)$

### Pricing Challenges

1. **Correlation:** Assets are correlated, affecting basket volatility
2. **No closed form:** Sum of log-normals is not log-normal
3. **High dimensionality:** $n$-dimensional problem

### Approximation Methods

**Moment matching:**
- Compute mean and variance of $B_T$
- Approximate as log-normal
- Use Black-Scholes with adjusted parameters

**Basket volatility:**
$$\sigma_B^2 = \sum_{i=1}^{n}\sum_{j=1}^{n} w_i w_j \sigma_i \sigma_j \rho_{ij}$$

where $\rho_{ij}$ is the correlation between assets $i$ and $j$.

**Monte Carlo:**
Simulate correlated paths:
$$S_i(T) = S_i(0)\exp\left((r_i - \frac{\sigma_i^2}{2})T + \sigma_i \sqrt{T} Z_i\right)$$

where $(Z_1, \ldots, Z_n) \sim N(0, \Sigma)$ with correlation matrix $\Sigma$.

## Rainbow Options

Rainbow options involve multiple assets with payoffs based on their relative performance.

### Best-of Options

**Call on maximum:**
$$\max(\max(S_1(T), S_2(T), \ldots, S_n(T)) - K, 0)$$

**Put on minimum:**
$$\max(K - \min(S_1(T), S_2(T), \ldots, S_n(T)), 0)$$

### Worst-of Options

**Call on minimum:**
$$\max(\min(S_1(T), S_2(T), \ldots, S_n(T)) - K, 0)$$

### Spread Options

Payoff based on the difference:
$$\max(S_1(T) - S_2(T) - K, 0)$$

Pricing requires modeling the joint distribution of $S_1(T)$ and $S_2(T)$.

## Cliquet Options

Cliquet (ratchet) options lock in gains periodically.

### Structure

Over periods $[t_0, t_1], [t_1, t_2], \ldots, [t_{n-1}, t_n]$:
- Compute return: $R_i = \frac{S(t_i) - S(t_{i-1})}{S(t_{i-1})}$
- Lock in: $\max(R_i, 0)$ or $\max(R_i, \text{floor})$

Total payoff:
$$\sum_{i=1}^{n} \max(R_i, 0)$$

### Pricing

Each period is independent, so:
$$V = \sum_{i=1}^{n} e^{-rt_i} \mathbb{E}[\max(R_i, 0)]$$

Each term is a forward-starting option, priced using Black-Scholes with appropriate forward volatility.

## Autocallable Structures

Autocallable products automatically redeem if certain conditions are met.

### Structure

- **Autocall trigger:** If $S(t_i) \geq H$ at observation date $t_i$, product redeems
- **Coupon:** Fixed coupon paid if not called
- **Final payoff:** If not called, may have additional features (put, call, etc.)

### Pricing Components

1. **Autocall probability:** $P(S(t_i) \geq H)$ for each observation date
2. **Coupon leg:** Present value of coupons conditional on survival
3. **Final payoff:** Expected value of terminal payoff conditional on no autocall

### Example: Autocallable Note

- Principal: \$100
- Maturity: 3 years
- Observation dates: Annual
- Autocall level: 120% of initial
- Coupon: 8% per year if not called
- Final payoff: If not called, $\max(S_T/S_0, 0.8)$ (80% capital protection)

Pricing requires:
- Probability of autocall at each date
- Expected coupon payments
- Expected final payoff

## Numerical Methods for Exotics

Most exotic options require numerical methods:

1. **Monte Carlo:** Universal but slow for early exercise
2. **Finite differences:** PDE approach, handles early exercise
3. **Tree methods:** Discrete approximation
4. **Fourier methods:** For certain payoffs with known transforms

### Monte Carlo for Path-Dependent Options

For a path-dependent payoff $H(S(t_1), \ldots, S(t_n))$:

1. Simulate paths: $S(t_1), \ldots, S(t_n)$
2. Compute payoff: $H(S(t_1), \ldots, S(t_n))$
3. Average: $\hat{V} = \frac{1}{M}\sum_{j=1}^{M} e^{-rT}H^{(j)}$

Variance reduction techniques (antithetic variates, control variates) are crucial.

### Early Exercise for American-Style Exotics

American-style exotics (e.g., American barrier options) require:
- **Longstaff-Schwartz:** Regression-based Monte Carlo
- **Finite differences:** PDE with free boundary
- **Tree methods:** Backward induction with exercise decision

The exercise boundary depends on both $S(t)$ and the path-dependent variable (e.g., running maximum).
