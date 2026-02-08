# Portfolio Optimization and Constraints

## Introduction

Portfolio optimization extends beyond basic mean-variance optimization to incorporate realistic constraints, transaction costs, and robust methods. This document covers advanced optimization techniques used in quantitative portfolio management.

## Quadratic Programming for Portfolio Optimization

### Standard Form

The mean-variance optimization problem is a quadratic program:

$$\min_{\boldsymbol{w}} \frac{1}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} - \lambda \boldsymbol{w}^T \boldsymbol{\mu}$$

subject to:
$$\boldsymbol{A}\boldsymbol{w} = \boldsymbol{b}$$
$$\boldsymbol{C}\boldsymbol{w} \leq \boldsymbol{d}$$

where:
- $\lambda$: risk aversion parameter
- $\boldsymbol{A}, \boldsymbol{b}$: equality constraints
- $\boldsymbol{C}, \boldsymbol{d}$: inequality constraints

### Solving via Lagrange Multipliers

The Lagrangian is:
$$L = \frac{1}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} - \lambda \boldsymbol{w}^T \boldsymbol{\mu} + \boldsymbol{\nu}^T (\boldsymbol{A}\boldsymbol{w} - \boldsymbol{b}) + \boldsymbol{\mu}^T (\boldsymbol{C}\boldsymbol{w} - \boldsymbol{d})$$

First-order conditions:
$$\frac{\partial L}{\partial \boldsymbol{w}} = \boldsymbol{\Sigma}\boldsymbol{w} - \lambda \boldsymbol{\mu} + \boldsymbol{A}^T \boldsymbol{\nu} + \boldsymbol{C}^T \boldsymbol{\mu} = \boldsymbol{0}$$

This leads to a system of linear equations (KKT conditions).

### Interior Point Methods

For large-scale problems, interior point methods are efficient:

1. Convert to barrier problem:
$$\min_{\boldsymbol{w}} \frac{1}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} - \lambda \boldsymbol{w}^T \boldsymbol{\mu} - \tau \sum_i \ln(s_i)$$

where $s_i$ are slack variables for inequalities.

2. Solve sequence of problems with $\tau \to 0$
3. Use Newton's method at each step

**Complexity:** $O(n^3)$ per iteration, typically converges in 10-50 iterations.

## Black-Litterman Model

### Motivation

Mean-variance optimization is sensitive to expected return estimates. Black-Litterman combines:
- Market equilibrium returns (prior)
- Investor views (likelihood)
- Bayesian updating (posterior)

### Equilibrium Returns

From CAPM, equilibrium expected returns are:
$$\boldsymbol{\Pi} = \delta \boldsymbol{\Sigma} \boldsymbol{w}_{mkt}$$

where:
- $\boldsymbol{\Pi}$: equilibrium expected returns
- $\delta$: risk aversion parameter
- $\boldsymbol{w}_{mkt}$: market capitalization weights

Risk aversion can be estimated:
$$\delta = \frac{E[R_m] - r_f}{\sigma_m^2}$$

### Investor Views

Views are expressed as:
$$\boldsymbol{P} \boldsymbol{\mu} = \boldsymbol{Q} + \boldsymbol{\epsilon}$$

where:
- $\boldsymbol{P}$: $k \times n$ pick matrix (which assets in each view)
- $\boldsymbol{Q}$: $k \times 1$ expected returns from views
- $\boldsymbol{\epsilon} \sim N(\boldsymbol{0}, \boldsymbol{\Omega})$: uncertainty in views

**Example:** View that stock 1 will outperform stock 2 by 3%:
$$\boldsymbol{P} = [1, -1, 0, \ldots, 0], \quad \boldsymbol{Q} = 0.03$$

### Posterior Distribution

Using Bayes' theorem, the posterior expected returns are:
$$\boldsymbol{\mu}_{BL} = [(\tau \boldsymbol{\Sigma})^{-1} + \boldsymbol{P}^T \boldsymbol{\Omega}^{-1} \boldsymbol{P}]^{-1} [(\tau \boldsymbol{\Sigma})^{-1} \boldsymbol{\Pi} + \boldsymbol{P}^T \boldsymbol{\Omega}^{-1} \boldsymbol{Q}]$$

Posterior covariance:
$$\boldsymbol{M} = [(\tau \boldsymbol{\Sigma})^{-1} + \boldsymbol{P}^T \boldsymbol{\Omega}^{-1} \boldsymbol{P}]^{-1}$$

**Interpretation:**
- $\tau$: confidence in prior (typically 0.05-0.1)
- $\boldsymbol{\Omega}$: confidence in views (diagonal matrix)
- Higher $\tau$: more weight to equilibrium
- Lower $\boldsymbol{\Omega}$: more weight to views

### Optimal Portfolio

With Black-Litterman expected returns:
$$\boldsymbol{w}^* = \frac{1}{\delta} \boldsymbol{M}^{-1} \boldsymbol{\mu}_{BL}$$

or equivalently:
$$\boldsymbol{w}^* = \boldsymbol{w}_{mkt} + \frac{1}{\delta} \boldsymbol{M}^{-1} \boldsymbol{P}^T \boldsymbol{\Omega}^{-1} (\boldsymbol{Q} - \boldsymbol{P} \boldsymbol{\mu}_{BL})$$

The optimal portfolio is the market portfolio plus a tilt based on views.

## Robust Optimization

### Uncertainty Sets

Robust optimization accounts for parameter uncertainty by optimizing over worst-case scenarios.

**Uncertainty set for expected returns:**
$$\mathcal{U}_\mu = \{\boldsymbol{\mu} : (\boldsymbol{\mu} - \hat{\boldsymbol{\mu}})^T \boldsymbol{\Sigma}_\mu^{-1} (\boldsymbol{\mu} - \hat{\boldsymbol{\mu}}) \leq \kappa^2\}$$

**Uncertainty set for covariance:**
$$\mathcal{U}_\Sigma = \{\boldsymbol{\Sigma} : \boldsymbol{\Sigma} = \hat{\boldsymbol{\Sigma}} + \boldsymbol{\Delta}, \|\boldsymbol{\Delta}\| \leq \rho\}$$

### Worst-Case Optimization

The robust optimization problem:
$$\max_{\boldsymbol{w}} \min_{\boldsymbol{\mu} \in \mathcal{U}_\mu, \boldsymbol{\Sigma} \in \mathcal{U}_\Sigma} \boldsymbol{w}^T \boldsymbol{\mu} - \frac{\lambda}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w}$$

**Ellipsoidal uncertainty (expected returns):**
The inner minimization gives:
$$\min_{\boldsymbol{\mu} \in \mathcal{U}_\mu} \boldsymbol{w}^T \boldsymbol{\mu} = \boldsymbol{w}^T \hat{\boldsymbol{\mu}} - \kappa \sqrt{\boldsymbol{w}^T \boldsymbol{\Sigma}_\mu \boldsymbol{w}}$$

The robust problem becomes:
$$\max_{\boldsymbol{w}} \boldsymbol{w}^T \hat{\boldsymbol{\mu}} - \kappa \sqrt{\boldsymbol{w}^T \boldsymbol{\Sigma}_\mu \boldsymbol{w}} - \frac{\lambda}{2} \boldsymbol{w}^T \hat{\boldsymbol{\Sigma}} \boldsymbol{w}$$

This penalizes portfolios sensitive to estimation error.

### Distributionally Robust Optimization

Instead of point estimates, consider ambiguity sets of distributions:

$$\max_{\boldsymbol{w}} \min_{P \in \mathcal{P}} E_P[\boldsymbol{w}^T \boldsymbol{R}] - \frac{\lambda}{2} \text{Var}_P(\boldsymbol{w}^T \boldsymbol{R})$$

where $\mathcal{P}$ is a set of plausible distributions (e.g., all distributions with given first two moments).

## Risk Parity and Risk Budgeting

### Risk Parity

Risk parity equalizes risk contributions across assets. The risk contribution of asset $i$ is:
$$RC_i = w_i \frac{\partial \sigma_p}{\partial w_i} = w_i \frac{(\boldsymbol{\Sigma}\boldsymbol{w})_i}{\sigma_p}$$

Risk parity requires:
$$RC_i = RC_j \quad \forall i,j$$

This leads to:
$$w_i (\boldsymbol{\Sigma}\boldsymbol{w})_i = w_j (\boldsymbol{\Sigma}\boldsymbol{w})_j$$

**Solution:** For diagonal covariance (uncorrelated assets):
$$w_i \propto \frac{1}{\sigma_i}$$

For general covariance, solve numerically:
$$\min_{\boldsymbol{w}} \sum_{i=1}^n \sum_{j=1}^n (RC_i - RC_j)^2$$

subject to $\boldsymbol{w}^T \boldsymbol{1} = 1$, $\boldsymbol{w} \geq \boldsymbol{0}$.

### Risk Budgeting

Risk budgeting allocates risk according to a target:
$$RC_i = b_i \sigma_p$$

where $b_i$ is the risk budget for asset $i$ and $\sum_i b_i = 1$.

**Solution:**
$$w_i = \frac{b_i \sigma_p}{(\boldsymbol{\Sigma}\boldsymbol{w})_i}$$

This is a fixed-point equation solved iteratively.

## Constraints

### Long-Only Constraint

$$\boldsymbol{w} \geq \boldsymbol{0}$$

Prevents short selling. Makes optimization easier (convex feasible region) but limits diversification.

### Turnover Constraint

Limits trading from current portfolio $\boldsymbol{w}_0$:
$$\|\boldsymbol{w} - \boldsymbol{w}_0\|_1 \leq \tau$$

or:
$$\sum_{i=1}^n |w_i - w_{0,i}| \leq \tau$$

This can be linearized using auxiliary variables:
$$w_i - w_{0,i} = w_i^+ - w_i^-$$
$$w_i^+, w_i^- \geq 0$$
$$\sum_i (w_i^+ + w_i^-) \leq \tau$$

### Sector Constraints

Limit exposure to sectors:
$$\boldsymbol{S}^T \boldsymbol{w} \leq \boldsymbol{s}_{max}$$

where $\boldsymbol{S}$ is a sector indicator matrix and $\boldsymbol{s}_{max}$ are maximum sector weights.

### Tracking Error Constraint

Limit deviation from benchmark $\boldsymbol{w}_b$:
$$\sqrt{(\boldsymbol{w} - \boldsymbol{w}_b)^T \boldsymbol{\Sigma} (\boldsymbol{w} - \boldsymbol{w}_b)} \leq TE_{max}$$

This is a quadratic constraint, making the problem more complex.

### Cardinality Constraint

Limit number of positions:
$$\sum_{i=1}^n \mathbb{1}(w_i \neq 0) \leq K$$

This makes the problem mixed-integer, requiring specialized solvers.

## Transaction Costs in Optimization

### Linear Transaction Costs

Simple model:
$$TC = \sum_{i=1}^n c_i |w_i - w_{0,i}| V_0$$

where $c_i$ is the transaction cost rate for asset $i$.

**Net return:**
$$\mu_{net} = \boldsymbol{w}^T \boldsymbol{\mu} - \frac{TC}{V_0}$$

### Quadratic Transaction Costs

Model market impact:
$$TC = \sum_{i=1}^n \lambda_i (w_i - w_{0,i})^2 V_0$$

where $\lambda_i$ captures temporary market impact.

**Optimization problem:**
$$\max_{\boldsymbol{w}} \boldsymbol{w}^T \boldsymbol{\mu} - \frac{\lambda}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} - \sum_{i=1}^n \lambda_i (w_i - w_{0,i})^2$$

This remains a quadratic program.

### Optimal Trading with Costs

Almgren-Chriss framework (see Market Microstructure section) optimizes execution considering:
- Temporary impact: $h(\boldsymbol{v})$
- Permanent impact: $g(\boldsymbol{v})$
- Risk: variance of execution cost

## Resampling and Bootstrapped Efficient Frontiers

### The Problem

Sample estimates $\hat{\boldsymbol{\mu}}$ and $\hat{\boldsymbol{\Sigma}}$ contain estimation error, leading to poor out-of-sample performance.

### Resampling Method

1. Generate $B$ bootstrap samples from historical returns
2. For each sample $b$, estimate $\hat{\boldsymbol{\mu}}^{(b)}$ and $\hat{\boldsymbol{\Sigma}}^{(b)}$
3. Solve optimization for each sample: $\boldsymbol{w}^{(b)}$
4. Average: $\bar{\boldsymbol{w}} = \frac{1}{B} \sum_{b=1}^B \boldsymbol{w}^{(b)}$

**Rationale:** Averaging reduces impact of estimation error.

### Bootstrapped Efficient Frontier

1. For each target return $\mu_0$:
   - Resample and optimize $B$ times
   - Average optimal portfolios: $\bar{\boldsymbol{w}}(\mu_0)$
   - Calculate out-of-sample return and risk
2. Plot bootstrapped frontier

This frontier is typically more stable than the sample frontier.

### Limitations

- Assumes returns are i.i.d. (may not hold)
- Doesn't address model uncertainty
- Can be computationally intensive

## Hierarchical Risk Parity (HRP)

### Motivation

HRP addresses correlation structure without requiring full covariance matrix inversion.

### Algorithm

1. **Tree clustering:** Use correlation to form hierarchical clusters
   - Distance: $d_{ij} = \sqrt{2(1 - \rho_{ij})}$
   - Linkage: single, complete, or average

2. **Quasi-diagonalization:** Reorder assets to group correlated assets

3. **Recursive bisection:** 
   - Start with all assets
   - Split into two clusters
   - Allocate risk budget: $b_1, b_2$ (e.g., inverse variance)
   - Recursively allocate within each cluster

**Risk allocation:**
$$b_i = \frac{1/\sigma_i^2}{\sum_{j \in cluster} 1/\sigma_j^2}$$

**Portfolio weights:**
$$w_i = b_i \times w_{cluster}$$

### Advantages

- No matrix inversion required
- Handles singular/near-singular covariance matrices
- Captures correlation structure
- More stable than mean-variance

### Example

Consider 4 assets with correlation matrix:
$$\boldsymbol{\rho} = \begin{bmatrix}
1 & 0.9 & 0.1 & 0.1 \\
0.9 & 1 & 0.1 & 0.1 \\
0.1 & 0.1 & 1 & 0.8 \\
0.1 & 0.1 & 0.8 & 1
\end{bmatrix}$$

Clustering: Assets 1-2 form cluster A, assets 3-4 form cluster B.

Risk allocation:
- Cluster A: $b_A \propto 1/\sigma_A^2$
- Cluster B: $b_B \propto 1/\sigma_B^2$

Within clusters:
- Assets 1-2: allocate based on inverse variance
- Assets 3-4: allocate based on inverse variance

Final weights reflect both within-cluster and between-cluster diversification.

## Example: Constrained Optimization

Consider optimizing a portfolio with:
- 3 assets: $\boldsymbol{\mu} = [0.10, 0.12, 0.08]^T$
- Covariance:
$$\boldsymbol{\Sigma} = \begin{bmatrix}
0.04 & 0.01 & 0.005 \\
0.01 & 0.05 & 0.01 \\
0.005 & 0.01 & 0.03
\end{bmatrix}$$
- Current portfolio: $\boldsymbol{w}_0 = [0.4, 0.4, 0.2]^T$
- Risk aversion: $\lambda = 2$
- Constraints:
  - Long-only: $\boldsymbol{w} \geq \boldsymbol{0}$
  - Budget: $\boldsymbol{w}^T \boldsymbol{1} = 1$
  - Turnover: $\|\boldsymbol{w} - \boldsymbol{w}_0\|_1 \leq 0.3$

**Unconstrained solution:**
$$\boldsymbol{w}^* = \frac{1}{\lambda} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} \approx [0.45, 0.55, -0.10]^T$$

Violates long-only constraint.

**Constrained solution (with turnover):**
Using quadratic programming:
$$\boldsymbol{w}^* \approx [0.35, 0.50, 0.15]^T$$

This satisfies all constraints. The turnover is:
$$\|\boldsymbol{w}^* - \boldsymbol{w}_0\|_1 = |0.35-0.4| + |0.50-0.4| + |0.15-0.2| = 0.15 < 0.3$$

The constrained portfolio trades off optimality for feasibility.
