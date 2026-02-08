# Numerical Methods for Derivatives

## Introduction

Most derivatives cannot be priced analytically and require numerical methods. The main approaches are Monte Carlo simulation, finite difference methods, and Fourier transform methods.

## Monte Carlo Methods

Monte Carlo simulation estimates option prices by simulating many possible paths of the underlying asset and averaging payoffs.

### Basic Monte Carlo

For a European option with payoff $H(S(T))$:

1. **Simulate paths:** Generate $M$ independent paths $S^{(j)}(T)$ for $j = 1, \ldots, M$
2. **Compute payoffs:** $H^{(j)} = H(S^{(j)}(T))$
3. **Estimate price:** $\hat{V} = e^{-rT}\frac{1}{M}\sum_{j=1}^{M}H^{(j)}$

**Convergence:** By the Law of Large Numbers, $\hat{V} \to V$ as $M \to \infty$.

**Error:** Standard error is $\sigma_H / \sqrt{M}$ where $\sigma_H^2 = \text{Var}(H)$.

### Path Simulation

For geometric Brownian motion:
$$S(T) = S(0)\exp\left((r - \frac{\sigma^2}{2})T + \sigma\sqrt{T}Z\right)$$

where $Z \sim N(0,1)$.

For time steps:
$$S(t_{i+1}) = S(t_i)\exp\left((r - \frac{\sigma^2}{2})\Delta t + \sigma\sqrt{\Delta t}Z_i\right)$$

### Path-Dependent Options

For path-dependent payoffs (e.g., Asian, lookback):

1. Simulate full path: $S(t_1), \ldots, S(t_n)$
2. Compute path-dependent variable (e.g., average, maximum)
3. Compute payoff from path-dependent variable

**Example - Asian call:**
$$A = \frac{1}{n}\sum_{i=1}^{n}S(t_i)$$
$$H = \max(A - K, 0)$$

### Multi-Asset Options

For basket options with $d$ assets:

1. Generate correlated normal random variables:
   $$(Z_1, \ldots, Z_d) \sim N(0, \Sigma)$$
   
   where $\Sigma$ is the correlation matrix.

2. Simulate each asset:
   $$S_i(T) = S_i(0)\exp\left((r_i - \frac{\sigma_i^2}{2})T + \sigma_i\sqrt{T}Z_i\right)$$

3. Compute basket and payoff.

**Cholesky decomposition:** To generate correlated normals, decompose $\Sigma = LL^T$ and set $Z = LX$ where $X \sim N(0,I)$.

## Variance Reduction Techniques

Variance reduction reduces the Monte Carlo error without increasing the number of simulations.

### Antithetic Variates

For each path $S^{(j)}(T)$, also simulate $S^{(-j)}(T)$ using $-Z$ instead of $Z$.

**Estimator:**
$$\hat{V} = e^{-rT}\frac{1}{M}\sum_{j=1}^{M}\frac{H^{(j)} + H^{(-j)}}{2}$$

**Variance reduction:** Works when payoff is monotonic in the random variable. Can reduce variance by factor of 2 or more.

### Control Variates

Use a related variable $Y$ with known expectation $\mathbb{E}[Y]$:

$$\hat{V}_{CV} = \hat{V} - \beta(\bar{Y} - \mathbb{E}[Y])$$

where $\bar{Y} = \frac{1}{M}\sum_{j=1}^{M}Y^{(j)}$ and $\beta$ is chosen to minimize variance.

**Optimal $\beta$:**
$$\beta^* = \frac{\text{Cov}(H, Y)}{\text{Var}(Y)}$$

**Common control variates:**
- **Stock price:** $Y = S(T)$, $\mathbb{E}[S(T)] = S(0)e^{rT}$
- **Geometric average:** For arithmetic Asian options
- **Lower bound:** For exotic options

### Importance Sampling

Change the probability measure to sample more from important regions:

$$\hat{V}_{IS} = \frac{1}{M}\sum_{j=1}^{M}H^{(j)}\frac{f(X^{(j)})}{g(X^{(j)})}$$

where $f$ is the original density and $g$ is the importance sampling density.

**Application:** For deep out-of-the-money options, sample more from the tail where payoffs are non-zero.

### Stratified Sampling

Divide the sample space into strata and sample from each:

$$\hat{V}_{strat} = \sum_{i=1}^{k}p_i \hat{V}_i$$

where $p_i$ is the probability of stratum $i$ and $\hat{V}_i$ is the estimate from stratum $i$.

**Latin Hypercube Sampling:** Special case ensuring each dimension is stratified.

## Finite Difference Methods

Finite difference methods solve the Black-Scholes PDE numerically by discretizing time and space.

### Grid Setup

Create a grid:
- **Time:** $t_0 = 0 < t_1 < \cdots < t_N = T$ with $\Delta t = T/N$
- **Stock:** $S_{\min} = S_0 < S_1 < \cdots < S_M = S_{\max}$ with $\Delta S = (S_{\max} - S_{\min})/M$

Option values: $V_{i,j} = V(S_i, t_j)$

### Explicit Method

Approximate derivatives:

$$\frac{\partial V}{\partial t} \approx \frac{V_{i,j+1} - V_{i,j}}{\Delta t}$$

$$\frac{\partial V}{\partial S} \approx \frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S}$$

$$\frac{\partial^2 V}{\partial S^2} \approx \frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{(\Delta S)^2}$$

Substitute into Black-Scholes PDE and solve for $V_{i,j}$:

$$V_{i,j} = a_i V_{i-1,j+1} + b_i V_{i,j+1} + c_i V_{i+1,j+1}$$

where coefficients $a_i, b_i, c_i$ depend on $S_i, r, \sigma, \Delta t, \Delta S$.

**Stability:** Requires $\Delta t \leq \frac{(\Delta S)^2}{\sigma^2 S_{\max}^2}$ (restrictive).

**Advantages:** Simple, fast per step
**Disadvantages:** Stability constraint, many time steps needed

### Implicit Method

Use future time step for spatial derivatives:

$$\frac{\partial V}{\partial t} \approx \frac{V_{i,j+1} - V_{i,j}}{\Delta t}$$

$$\frac{\partial V}{\partial S} \approx \frac{V_{i+1,j+1} - V_{i-1,j+1}}{2\Delta S}$$

$$\frac{\partial^2 V}{\partial S^2} \approx \frac{V_{i+1,j+1} - 2V_{i,j+1} + V_{i-1,j+1}}{(\Delta S)^2}$$

This gives a system of equations:

$$V_{i,j} = a_i V_{i-1,j+1} + b_i V_{i,j+1} + c_i V_{i+1,j+1}$$

Solve using tridiagonal matrix algorithm (Thomas algorithm).

**Stability:** Unconditionally stable (no time step restriction).

**Advantages:** Stable, fewer time steps
**Disadvantages:** Slower per step (solving linear system)

### Crank-Nicolson Method

Average explicit and implicit:

$$\frac{V_{i,j+1} - V_{i,j}}{\Delta t} = \frac{1}{2}\left[\mathcal{L}V_{i,j} + \mathcal{L}V_{i,j+1}\right]$$

where $\mathcal{L}$ is the spatial differential operator.

**Properties:**
- Unconditionally stable
- Second-order accurate in time and space
- Most commonly used method

### Boundary Conditions

**At $S = 0$:**
- Call: $V(0,t) = 0$
- Put: $V(0,t) = Ke^{-r(T-t)}$

**At $S = S_{\max}$:**
- Call: $V(S_{\max},t) = S_{\max} - Ke^{-r(T-t)}$ (for large $S$)
- Put: $V(S_{\max},t) = 0$

**At maturity $t = T$:**
- Call: $V(S,T) = \max(S - K, 0)$
- Put: $V(S,T) = \max(K - S, 0)$

### Grid Design

**Stock grid:**
- Use log-space: $x = \ln S$, then $V(x,t)$
- Or use non-uniform grid (finer near strike)
- Ensure strike $K$ is on grid

**Time grid:**
- Uniform is typical
- May refine near expiration

**Convergence:** Check that solution converges as $\Delta S \to 0$ and $\Delta t \to 0$.

## American Options

American options allow early exercise, requiring free boundary conditions.

### Longstaff-Schwartz (LSM)

LSM uses Monte Carlo with regression to estimate continuation value.

**Algorithm:**
1. Simulate $M$ paths forward to maturity
2. At each exercise date $t_i$, working backwards:
   - Identify in-the-money paths
   - Regress continuation value on basis functions of $S(t_i)$
   - Compare continuation value to exercise value
   - Exercise if exercise value > continuation value
3. Average payoffs along optimal exercise paths

**Basis functions:** Polynomials, Laguerre, Hermite, or other functions of $S$.

**Continuation value:**
$$C(S,t_i) = e^{-r\Delta t}\mathbb{E}[V(S(t_{i+1}), t_{i+1}) | S(t_i) = S]$$

Estimated via regression:
$$C(S,t_i) \approx \sum_{k=1}^{K}\beta_k \phi_k(S)$$

**Advantages:**
- Handles high dimensions
- Works with path-dependent features
- Flexible

**Disadvantages:**
- Low bias (suboptimal exercise)
- Requires many paths
- Basis function choice matters

### Projected Successive Over-Relaxation (PSOR)

PSOR extends the implicit finite difference method for American options.

**Free boundary:** Exercise boundary $S^*(t)$ where $V(S^*(t),t) = S^*(t) - K$ (for call).

**Algorithm:**
At each time step, solve:
$$\max(V_{i,j+1}, \text{exercise value}_i)$$

using successive over-relaxation with projection.

**Advantages:**
- Accurate
- Fast
- Handles early exercise

**Disadvantages:**
- One-dimensional only
- Requires PDE formulation

### Tree Methods

Binomial/trinomial trees naturally handle early exercise:

$$V_{i,j} = \max\left(e^{-r\Delta t}[pV_{i+1,j+1} + (1-p)V_{i-1,j+1}], \text{exercise value}_i\right)$$

## Fourier Methods

Fourier methods use characteristic functions to price options efficiently.

### Characteristic Function

The characteristic function of $\ln S(T)$ is:

$$\phi(u) = \mathbb{E}[e^{iu\ln S(T)}]$$

For geometric Brownian motion:
$$\phi(u) = \exp\left(iu\ln S(0) + iu(r - \frac{\sigma^2}{2})T - \frac{u^2\sigma^2 T}{2}\right)$$

### Carr-Madan Method

Carr-Madan expresses the option price as a Fourier integral.

For a call option:
$$C(K) = \frac{e^{-\alpha\ln K}}{\pi}\int_0^{\infty}e^{-iv\ln K}\psi(v)dv$$

where:
$$\psi(v) = \frac{e^{-rT}\phi(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}$$

and $\alpha > 0$ is a damping parameter.

**Numerical integration:** Use FFT (Fast Fourier Transform) to compute for many strikes simultaneously.

**Advantages:**
- Fast (FFT is $O(N\log N)$)
- Computes many strikes at once
- Works for any model with known characteristic function

**Disadvantages:**
- Requires characteristic function
- Damping parameter choice matters
- May have numerical issues

### COS Method

The COS method expands the density in a cosine series:

$$f(x) \approx \sum_{k=0}^{N-1}A_k\cos\left(k\pi\frac{x-a}{b-a}\right)$$

where $[a,b]$ is the domain and:
$$A_k = \frac{2}{b-a}\text{Re}\left[\phi\left(\frac{k\pi}{b-a}\right)e^{-ik\pi a/(b-a)}\right]$$

Option prices are computed from the cosine coefficients.

**Advantages:**
- Very fast convergence
- Accurate
- Works for many models

**Disadvantages:**
- Requires domain truncation
- Characteristic function needed

## Comparison of Methods

### Monte Carlo

**Best for:**
- High-dimensional problems
- Path-dependent options
- Complex payoffs
- When Greeks are needed (bump-and-revalue)

**Not ideal for:**
- American options (though LSM works)
- When speed is critical
- When high accuracy needed

### Finite Differences

**Best for:**
- Low-dimensional PDEs
- American options
- When entire price surface needed
- When Greeks needed (from grid)

**Not ideal for:**
- High dimensions
- Path-dependent (unless PDE reformulated)
- Complex boundaries

### Fourier Methods

**Best for:**
- European options
- Models with known characteristic function
- Multiple strikes needed
- Fast pricing

**Not ideal for:**
- American options
- Path-dependent
- Models without characteristic function

## Practical Considerations

### Computational Cost

- **Monte Carlo:** $O(M)$ where $M$ is number of paths
- **Finite differences:** $O(M \times N)$ where $M$ is space steps, $N$ is time steps
- **Fourier:** $O(N\log N)$ for FFT

### Accuracy

- **Monte Carlo:** Error $\sim 1/\sqrt{M}$
- **Finite differences:** Error $\sim (\Delta S)^2 + (\Delta t)^2$ (for Crank-Nicolson)
- **Fourier:** Error depends on truncation and discretization

### Greeks

**Monte Carlo:**
- Bump-and-revalue: Run twice with perturbed parameters
- Pathwise: Differentiate payoff (when possible)
- Likelihood ratio: Use density derivatives

**Finite differences:**
- Compute from grid: $\Delta \approx \frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S}$

**Fourier:**
- Differentiate Fourier representation
- Or use finite differences on Fourier prices
