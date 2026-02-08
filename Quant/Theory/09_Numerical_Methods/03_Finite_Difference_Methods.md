# Finite Difference Methods

## Introduction

Finite difference methods solve partial differential equations (PDEs) numerically by discretizing time and space. In finance, they're used for option pricing, particularly for American options and exotic derivatives. This document covers PDE formulation, discretization schemes, stability, and convergence.

## PDE Formulation

### Black-Scholes PDE

For option price $V(S,t)$:

$$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0$$

with boundary conditions:
- Terminal: $V(S,T) = \max(S-K, 0)$ (for call)
- Boundary: $V(0,t) = 0$, $V(S \to \infty, t) \sim S - Ke^{-r(T-t)}$

### Heat Equation Transformation

Transform to heat equation via:
$$x = \ln(S/K)$$
$$\tau = T - t$$
$$u(x,\tau) = e^{r\tau} V(S,t)$$

**Resulting PDE:**
$$\frac{\partial u}{\partial \tau} = \frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2} + \left(r - \frac{\sigma^2}{2}\right)\frac{\partial u}{\partial x}$$

Further transformation gives standard heat equation:
$$\frac{\partial u}{\partial \tau} = \frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2}$$

**Advantages:**
- Simpler form
- Constant coefficients
- Better numerical properties

## Grid Design

### Uniform Grid

**Time:** $t_i = i\Delta t$, $i = 0,1,\ldots,M$ where $\Delta t = T/M$

**Space:** $S_j = j\Delta S$, $j = 0,1,\ldots,N$ where $\Delta S = S_{max}/N$

**Notation:** $V_{i,j} = V(S_j, t_i)$

### Non-Uniform Grid

**Log-spacing:**
$$S_j = S_{min} e^{j\Delta x}$$

where $\Delta x = \frac{\ln(S_{max}/S_{min})}{N}$.

**Advantages:**
- More points near current stock price
- Better resolution where needed
- Natural for log-transformed PDE

### Boundary Conditions

**Lower boundary ($S = 0$):**
- Call: $V(0,t) = 0$
- Put: $V(0,t) = Ke^{-r(T-t)}$

**Upper boundary ($S \to \infty$):**
- Call: $V(S,t) \sim S - Ke^{-r(T-t)}$
- Put: $V(S,t) \sim 0$

**Implementation:** Set $S_{max}$ large enough that boundary condition applies.

## Explicit Scheme

### Discretization

**Time derivative:**
$$\frac{\partial V}{\partial t} \approx \frac{V_{i+1,j} - V_{i,j}}{\Delta t}$$

**Space derivatives:**
$$\frac{\partial V}{\partial S} \approx \frac{V_{i,j+1} - V_{i,j-1}}{2\Delta S}$$

$$\frac{\partial^2 V}{\partial S^2} \approx \frac{V_{i,j+1} - 2V_{i,j} + V_{i,j-1}}{(\Delta S)^2}$$

### Update Equation

**Black-Scholes PDE becomes:**
$$V_{i,j} = V_{i+1,j} + \Delta t \left[rS_j \frac{V_{i+1,j+1} - V_{i+1,j-1}}{2\Delta S} + \frac{1}{2}\sigma^2 S_j^2 \frac{V_{i+1,j+1} - 2V_{i+1,j} + V_{i+1,j-1}}{(\Delta S)^2} - rV_{i+1,j}\right]$$

**Matrix form:**
$$\boldsymbol{V}_i = \boldsymbol{V}_{i+1} + \Delta t \boldsymbol{A} \boldsymbol{V}_{i+1}$$

where $\boldsymbol{A}$ is the discretization matrix.

### Stability: CFL Condition

**Courant-Friedrichs-Lewy (CFL) condition:**
$$\Delta t \leq \frac{(\Delta S)^2}{\sigma^2 S_{max}^2}$$

**Intuition:** Information propagation speed must not exceed grid resolution.

**Violation:** Solution becomes unstable (oscillations, blow-up).

**Accuracy:** $O(\Delta t) + O((\Delta S)^2)$

### Limitations

- Stability requires small $\Delta t$
- Many time steps needed
- Computationally expensive

## Implicit Scheme

### Discretization

Evaluate space derivatives at current time:

**Update equation:**
$$V_{i,j} = V_{i+1,j} + \Delta t \left[rS_j \frac{V_{i,j+1} - V_{i,j-1}}{2\Delta S} + \frac{1}{2}\sigma^2 S_j^2 \frac{V_{i,j+1} - 2V_{i,j} + V_{i,j-1}}{(\Delta S)^2} - rV_{i,j}\right]$$

**Matrix form:**
$$(\boldsymbol{I} - \Delta t \boldsymbol{A}) \boldsymbol{V}_i = \boldsymbol{V}_{i+1}$$

### Solution

Solve tridiagonal system:
$$\boldsymbol{B} \boldsymbol{V}_i = \boldsymbol{V}_{i+1}$$

where $\boldsymbol{B} = \boldsymbol{I} - \Delta t \boldsymbol{A}$.

**Thomas algorithm:** $O(N)$ for tridiagonal systems.

### Stability

**Unconditionally stable:** Works for any $\Delta t$.

**Advantages:**
- Larger time steps possible
- Fewer time steps
- More efficient

**Disadvantages:**
- Requires solving linear system
- Less accurate (first-order in time)

## Crank-Nicolson Scheme

### Discretization

Average explicit and implicit schemes:

**Update equation:**
$$V_{i,j} = V_{i+1,j} + \frac{\Delta t}{2} \left[\boldsymbol{A} \boldsymbol{V}_{i+1} + \boldsymbol{A} \boldsymbol{V}_i\right]$$

**Matrix form:**
$$\left(\boldsymbol{I} - \frac{\Delta t}{2}\boldsymbol{A}\right) \boldsymbol{V}_i = \left(\boldsymbol{I} + \frac{\Delta t}{2}\boldsymbol{A}\right) \boldsymbol{V}_{i+1}$$

### Properties

**Stability:** Unconditionally stable.

**Accuracy:** $O((\Delta t)^2) + O((\Delta S)^2)$ (second-order in time and space).

**Best of both:** Stability of implicit, accuracy of explicit.

**Most commonly used** for option pricing.

## American Options

### Free Boundary Problem

American options can be exercised early:
$$V(S,t) \geq \max(S-K, 0)$$

**Optimal exercise boundary:** $S^*(t)$ where exercise is optimal.

**PDE:** Same as European, but with early exercise constraint.

### Discretization

**At each time step:**
1. Solve PDE (as if European)
2. Apply early exercise:
$$V_{i,j} = \max(V_{i,j}^{European}, S_j - K)$$

### PSOR Algorithm

**Projected Successive Over-Relaxation:**

For each time step:
1. Solve: $(\boldsymbol{I} - \frac{\Delta t}{2}\boldsymbol{A}) \boldsymbol{V}_i = \boldsymbol{b}$
2. Project: $V_{i,j} = \max(V_{i,j}, S_j - K)$
3. Iterate until convergence

**Convergence:** Typically 5-10 iterations per time step.

### Linear Complementarity Problem

**Formulation:**
$$\begin{cases}
\frac{\partial V}{\partial t} + \mathcal{L}V \geq 0 \\
V(S,t) \geq \max(S-K, 0) \\
\left(\frac{\partial V}{\partial t} + \mathcal{L}V\right)(V - \max(S-K, 0)) = 0
\end{cases}$$

where $\mathcal{L}$ is the Black-Scholes operator.

**Discretization:** Leads to complementarity problem solved by PSOR.

## Multi-Dimensional PDE

### Two Assets

**PDE:**
$$\frac{\partial V}{\partial t} + rS_1\frac{\partial V}{\partial S_1} + rS_2\frac{\partial V}{\partial S_2} + \frac{1}{2}\sigma_1^2 S_1^2 \frac{\partial^2 V}{\partial S_1^2} + \frac{1}{2}\sigma_2^2 S_2^2 \frac{\partial^2 V}{\partial S_2^2} + \rho\sigma_1\sigma_2 S_1 S_2 \frac{\partial^2 V}{\partial S_1 \partial S_2} - rV = 0$$

**Grid:** $(S_1)_i \times (S_2)_j$ → $N^2$ points (curse of dimensionality).

### ADI Method

**Alternating Direction Implicit:**

Split time step:
1. Implicit in $S_1$, explicit in $S_2$
2. Implicit in $S_2$, explicit in $S_1$

**Advantages:**
- Tridiagonal systems (efficient)
- Unconditionally stable
- Second-order accurate

**Update:**
$$\left(\boldsymbol{I} - \frac{\Delta t}{2}\boldsymbol{A}_1\right) \boldsymbol{V}^{n+1/2} = \left(\boldsymbol{I} + \frac{\Delta t}{2}\boldsymbol{A}_2\right) \boldsymbol{V}^n$$

$$\left(\boldsymbol{I} - \frac{\Delta t}{2}\boldsymbol{A}_2\right) \boldsymbol{V}^{n+1} = \left(\boldsymbol{I} + \frac{\Delta t}{2}\boldsymbol{A}_1\right) \boldsymbol{V}^{n+1/2}$$

where $\boldsymbol{A}_1$ and $\boldsymbol{A}_2$ are operators in each direction.

## Convergence

### Consistency

**Definition:** Discretization error → 0 as $\Delta t, \Delta S \to 0$.

**Truncation error:** $O((\Delta t)^p) + O((\Delta S)^q)$

**Consistent if:** $p, q > 0$

### Stability

**Definition:** Errors don't grow unbounded.

**Von Neumann analysis:** Check eigenvalues of amplification matrix.

**CFL condition:** For explicit schemes.

**Lax-Richtmyer:** Stability + consistency → convergence.

### Lax Equivalence Theorem

**Statement:** For linear PDEs:
$$\text{Consistency} + \text{Stability} \Leftrightarrow \text{Convergence}$$

**Implication:** Check consistency and stability separately.

### Convergence Rate

**Error:** $E = |V_{exact} - V_{numerical}|$

**Order:** If $E = O((\Delta t)^p)$, scheme is $p$-th order.

**Crank-Nicolson:** $O((\Delta t)^2)$ → second-order.

**Richardson extrapolation:** Combine solutions at different resolutions to improve accuracy.

## Example: European Call Pricing

Price European call:
- $S_0 = 100$, $K = 100$, $r = 0.05$, $\sigma = 0.2$, $T = 1$

**Grid:**
- $S \in [0, 200]$, $N = 100$ points
- $t \in [0, 1]$, $M = 100$ time steps
- $\Delta S = 2$, $\Delta t = 0.01$

**Crank-Nicolson:**

**Terminal condition:**
$$V_{M,j} = \max(S_j - 100, 0)$$

**Boundary conditions:**
$$V_{i,0} = 0 \quad \text{(lower)}$$
$$V_{i,N} = S_N - Ke^{-r(T-t_i)} \quad \text{(upper)}$$

**Update:**
$$\left(\boldsymbol{I} - \frac{\Delta t}{2}\boldsymbol{A}\right) \boldsymbol{V}_i = \left(\boldsymbol{I} + \frac{\Delta t}{2}\boldsymbol{A}\right) \boldsymbol{V}_{i+1}$$

**Result:**
- $V_{0,50} = \$10.45$ (at $S = 100$)
- Black-Scholes: $\$10.45$
- Error: $< 0.01\%$

**Convergence:** As $N, M \to \infty$, error → 0 at rate $O((\Delta t)^2 + (\Delta S)^2)$.
