# Numerical Optimization

## Introduction

Optimization is central to quantitative finance: portfolio optimization, model calibration, parameter estimation. This document covers gradient-based methods, constrained optimization, derivative-free methods, and global optimization techniques.

## Gradient Descent

### Basic Algorithm

Minimize $f(\boldsymbol{x})$:

**Update:**
$$\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \alpha_k \nabla f(\boldsymbol{x}_k)$$

where $\alpha_k$ is step size.

### Step Size Selection

**Fixed step:** $\alpha_k = \alpha$ (constant)

**Line search:** Choose $\alpha_k$ to minimize:
$$\phi(\alpha) = f(\boldsymbol{x}_k - \alpha \nabla f(\boldsymbol{x}_k))$$

**Armijo condition:**
$$f(\boldsymbol{x}_k - \alpha \nabla f(\boldsymbol{x}_k)) \leq f(\boldsymbol{x}_k) - c \alpha \|\nabla f(\boldsymbol{x}_k)\|^2$$

where $c \in (0,1)$ (typically 0.1).

**Backtracking:** Start with $\alpha = 1$, reduce until Armijo satisfied.

### Convergence

**For convex $f$:**
- Convergence rate: $O(1/k)$
- With strong convexity: $O(\rho^k)$ (linear convergence)

**Stopping criterion:**
$$\|\nabla f(\boldsymbol{x}_k)\| < \epsilon$$

### Limitations

- Slow convergence (many iterations)
- Sensitive to conditioning
- May oscillate in narrow valleys

## Newton's Method

### Algorithm

**Update:**
$$\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \boldsymbol{H}_k^{-1} \nabla f(\boldsymbol{x}_k)$$

where $\boldsymbol{H}_k = \nabla^2 f(\boldsymbol{x}_k)$ is the Hessian.

### Derivation

**Quadratic approximation:**
$$f(\boldsymbol{x}) \approx f(\boldsymbol{x}_k) + \nabla f(\boldsymbol{x}_k)^T (\boldsymbol{x} - \boldsymbol{x}_k) + \frac{1}{2}(\boldsymbol{x} - \boldsymbol{x}_k)^T \boldsymbol{H}_k (\boldsymbol{x} - \boldsymbol{x}_k)$$

**Minimize approximation:**
$$\nabla f(\boldsymbol{x}_k) + \boldsymbol{H}_k (\boldsymbol{x} - \boldsymbol{x}_k) = \boldsymbol{0}$$

Solving gives Newton update.

### Convergence

**Quadratic convergence:** If $\boldsymbol{x}^*$ is optimal and $\boldsymbol{H}^*$ is positive definite:
$$\|\boldsymbol{x}_{k+1} - \boldsymbol{x}^*\| \leq C \|\boldsymbol{x}_k - \boldsymbol{x}^*\|^2$$

**Much faster than gradient descent** (when it works).

### Modifications

**Damped Newton:** Use step size:
$$\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \alpha_k \boldsymbol{H}_k^{-1} \nabla f(\boldsymbol{x}_k)$$

**Levenberg-Marquardt:** Regularize Hessian:
$$(\boldsymbol{H}_k + \lambda \boldsymbol{I})^{-1} \nabla f(\boldsymbol{x}_k)$$

### Hessian Computation

**Analytical:** Compute $\frac{\partial^2 f}{\partial x_i \partial x_j}$ (may be complex).

**Finite differences:**
$$\frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{f(\boldsymbol{x} + \boldsymbol{e}_i h + \boldsymbol{e}_j h) - f(\boldsymbol{x} + \boldsymbol{e}_i h) - f(\boldsymbol{x} + \boldsymbol{e}_j h) + f(\boldsymbol{x})}{h^2}$$

**Automatic differentiation:** Compute exactly, efficiently.

## Quasi-Newton Methods

### BFGS

**Broyden-Fletcher-Goldfarb-Shanno:**

Approximate Hessian inverse $\boldsymbol{B}_k \approx \boldsymbol{H}_k^{-1}$.

**Update:**
$$\boldsymbol{B}_{k+1} = \boldsymbol{B}_k + \frac{\boldsymbol{y}_k \boldsymbol{y}_k^T}{\boldsymbol{y}_k^T \boldsymbol{s}_k} - \frac{\boldsymbol{B}_k \boldsymbol{s}_k \boldsymbol{s}_k^T \boldsymbol{B}_k}{\boldsymbol{s}_k^T \boldsymbol{B}_k \boldsymbol{s}_k}$$

where:
- $\boldsymbol{s}_k = \boldsymbol{x}_{k+1} - \boldsymbol{x}_k$
- $\boldsymbol{y}_k = \nabla f(\boldsymbol{x}_{k+1}) - \nabla f(\boldsymbol{x}_k)$

**Update equation:**
$$\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \alpha_k \boldsymbol{B}_k \nabla f(\boldsymbol{x}_k)$$

**Properties:**
- Superlinear convergence
- No Hessian computation needed
- Maintains positive definiteness

### L-BFGS

**Limited-memory BFGS:**

Store only last $m$ pairs $(\boldsymbol{s}_i, \boldsymbol{y}_i)$.

**Memory:** $O(mn)$ instead of $O(n^2)$.

**Update:** Recursive formula using stored pairs.

**Advantages:**
- Efficient for large problems
- Good convergence
- Widely used in practice

## Constrained Optimization

### Lagrangian

**Problem:**
$$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.} \quad g_i(\boldsymbol{x}) = 0, \quad h_j(\boldsymbol{x}) \leq 0$$

**Lagrangian:**
$$L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\boldsymbol{x}) + \sum_i \lambda_i g_i(\boldsymbol{x}) + \sum_j \mu_j h_j(\boldsymbol{x})$$

**KKT conditions:**
$$\nabla_{\boldsymbol{x}} L = \boldsymbol{0}$$
$$g_i(\boldsymbol{x}) = 0$$
$$h_j(\boldsymbol{x}) \leq 0$$
$$\mu_j \geq 0$$
$$\mu_j h_j(\boldsymbol{x}) = 0$$

### Augmented Lagrangian

**Augmented Lagrangian:**
$$L_A(\boldsymbol{x}, \boldsymbol{\lambda}, \rho) = f(\boldsymbol{x}) + \sum_i \lambda_i g_i(\boldsymbol{x}) + \frac{\rho}{2} \sum_i g_i(\boldsymbol{x})^2$$

**Method:**
1. Minimize $L_A$ w.r.t. $\boldsymbol{x}$
2. Update multipliers: $\lambda_i \leftarrow \lambda_i + \rho g_i(\boldsymbol{x})$
3. Increase $\rho$ if needed
4. Repeat

**Advantages:**
- Handles equality constraints
- More stable than penalty methods

### Interior Point Methods

**Barrier function:**
$$B(\boldsymbol{x}, \mu) = f(\boldsymbol{x}) - \mu \sum_j \ln(-h_j(\boldsymbol{x}))$$

**Method:**
1. Minimize $B(\boldsymbol{x}, \mu)$
2. Decrease $\mu \to 0$
3. Solution approaches boundary

**Path-following:** Track central path as $\mu \to 0$.

**Advantages:**
- Handles inequality constraints
- Polynomial complexity (for convex problems)

### Sequential Quadratic Programming (SQP)

**Approach:** Solve sequence of quadratic programs.

**At iteration $k$:**
$$\min_{\boldsymbol{d}} \frac{1}{2} \boldsymbol{d}^T \boldsymbol{H}_k \boldsymbol{d} + \nabla f(\boldsymbol{x}_k)^T \boldsymbol{d}$$

subject to:
$$g_i(\boldsymbol{x}_k) + \nabla g_i(\boldsymbol{x}_k)^T \boldsymbol{d} = 0$$
$$h_j(\boldsymbol{x}_k) + \nabla h_j(\boldsymbol{x}_k)^T \boldsymbol{d} \leq 0$$

**Update:** $\boldsymbol{x}_{k+1} = \boldsymbol{x}_k + \boldsymbol{d}_k$.

**Convergence:** Superlinear under conditions.

## Derivative-Free Methods

### Nelder-Mead

**Simplex method:**

Maintain simplex of $n+1$ points.

**Operations:**
1. **Reflect:** Worst point through centroid
2. **Expand:** If reflection good, extend further
3. **Contract:** If reflection bad, shrink toward centroid
4. **Shrink:** Reduce all points toward best

**Advantages:**
- No derivatives needed
- Robust
- Simple

**Disadvantages:**
- Slow convergence
- May get stuck
- No convergence guarantees

### Genetic Algorithms

**Population-based:**

1. **Initialization:** Random population
2. **Selection:** Choose parents (fitness-based)
3. **Crossover:** Combine parents
4. **Mutation:** Random changes
5. **Replacement:** New generation
6. Repeat

**Operators:**
- **Crossover:** Blend or swap components
- **Mutation:** Small random perturbations
- **Selection:** Tournament, roulette wheel

**Advantages:**
- Global search
- Handles non-smooth functions
- Parallelizable

**Disadvantages:**
- Slow convergence
- Many parameters to tune
- No guarantees

### Differential Evolution

**Similar to genetic algorithms:**

**Mutation:**
$$\boldsymbol{v}_i = \boldsymbol{x}_{r1} + F(\boldsymbol{x}_{r2} - \boldsymbol{x}_{r3})$$

**Crossover:**
$$u_{i,j} = \begin{cases}
v_{i,j} & \text{if } U(0,1) < CR \text{ or } j = j_{rand} \\
x_{i,j} & \text{otherwise}
\end{cases}$$

**Selection:**
$$\boldsymbol{x}_i^{new} = \begin{cases}
\boldsymbol{u}_i & \text{if } f(\boldsymbol{u}_i) < f(\boldsymbol{x}_i) \\
\boldsymbol{x}_i & \text{otherwise}
\end{cases}$$

**Parameters:**
- $F$: Differential weight (typically 0.5-1.0)
- $CR$: Crossover probability (typically 0.7-0.9)

## Global Optimization

### Simulated Annealing

**Metropolis algorithm:**

1. Start at $\boldsymbol{x}_0$
2. Propose move: $\boldsymbol{x}_{new} = \boldsymbol{x}_k + \boldsymbol{\Delta}$
3. Accept if:
   - $f(\boldsymbol{x}_{new}) < f(\boldsymbol{x}_k)$, or
   - $U(0,1) < \exp(-(f(\boldsymbol{x}_{new}) - f(\boldsymbol{x}_k))/T)$
4. Decrease temperature $T$
5. Repeat

**Cooling schedule:**
$$T_k = T_0 \times \alpha^k$$

where $\alpha \in (0,1)$ (e.g., 0.95).

**Advantages:**
- Escapes local minima
- Probabilistic acceptance
- Asymptotically finds global optimum

**Disadvantages:**
- Slow
- Requires tuning
- No guarantees in finite time

### Basin Hopping

**Two-phase:**

1. **Local minimization:** From current point
2. **Random perturbation:** Jump to new region
3. **Local minimization:** From new point
4. Accept if better
5. Repeat

**Advantages:**
- Combines global and local search
- More efficient than pure random search

### Multi-Start

**Simple approach:**

1. Generate random starting points
2. Run local optimizer from each
3. Return best solution

**Advantages:**
- Simple
- Parallelizable
- Probabilistic guarantee with enough starts

**Disadvantages:**
- May waste computation
- No guidance on number of starts

## Applications

### Model Calibration

**Problem:** Find parameters $\boldsymbol{\theta}$ to match market prices.

**Objective:**
$$\min_{\boldsymbol{\theta}} \sum_i w_i (P_i^{model}(\boldsymbol{\theta}) - P_i^{market})^2$$

**Challenges:**
- Non-convex
- Multiple local minima
- Expensive function evaluations

**Methods:**
- Local: Levenberg-Marquardt, BFGS
- Global: Simulated annealing, differential evolution

### Portfolio Optimization

**Mean-variance:**
$$\min_{\boldsymbol{w}} \frac{1}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} - \lambda \boldsymbol{w}^T \boldsymbol{\mu}$$

subject to:
$$\boldsymbol{w}^T \boldsymbol{1} = 1$$
$$\boldsymbol{w} \geq \boldsymbol{0}$$

**Methods:**
- Quadratic programming (specialized)
- Interior point methods
- Projected gradient descent

### Maximum Likelihood Estimation

**Problem:**
$$\max_{\boldsymbol{\theta}} \sum_{i=1}^n \ln f(x_i | \boldsymbol{\theta})$$

**Methods:**
- Newton-Raphson
- BFGS
- EM algorithm (for latent variables)

## Example: Portfolio Optimization

Optimize portfolio with:
- Expected returns: $\boldsymbol{\mu} = [0.10, 0.12, 0.08]^T$
- Covariance:
$$\boldsymbol{\Sigma} = \begin{bmatrix}
0.04 & 0.01 & 0.005 \\
0.01 & 0.05 & 0.01 \\
0.005 & 0.01 & 0.03
\end{bmatrix}$$
- Risk aversion: $\lambda = 2$

**Problem:**
$$\min_{\boldsymbol{w}} \frac{1}{2} \boldsymbol{w}^T \boldsymbol{\Sigma} \boldsymbol{w} - 2 \boldsymbol{w}^T \boldsymbol{\mu}$$

subject to: $\boldsymbol{w}^T \boldsymbol{1} = 1$, $\boldsymbol{w} \geq \boldsymbol{0}$

**Using projected gradient descent:**

**Gradient:**
$$\nabla f = \boldsymbol{\Sigma} \boldsymbol{w} - 2 \boldsymbol{\mu}$$

**Update:**
$$\boldsymbol{w}_{k+1} = \text{Proj}_{\Delta} (\boldsymbol{w}_k - \alpha_k \nabla f)$$

where $\text{Proj}_{\Delta}$ projects onto simplex.

**Result:**
- $\boldsymbol{w}^* = [0.35, 0.50, 0.15]^T$
- Expected return: 11.1%
- Standard deviation: 19.8%
- Sharpe ratio: 0.409

**Convergence:** Typically 10-50 iterations depending on step size and tolerance.
