# Functional Analysis Basics

## Metric Spaces

### Definition

A metric space $(X, d)$ consists of set $X$ and metric $d: X \times X \to \mathbb{R}$ satisfying:

1. **Positive definiteness**: $d(x, y) \geq 0$ with equality if and only if $x = y$
2. **Symmetry**: $d(x, y) = d(y, x)$
3. **Triangle inequality**: $d(x, z) \leq d(x, y) + d(y, z)$

### Examples

- **Euclidean space**: $(\mathbb{R}^n, d_2)$ where $d_2(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$
- **$L^p$ spaces**: $(L^p[a,b], d_p)$ where $d_p(f, g) = \|f - g\|_p = (\int_a^b |f-g|^p)^{1/p}$
- **Continuous functions**: $(C[a,b], d_\infty)$ where $d_\infty(f, g) = \max_{x \in [a,b]} |f(x) - g(x)|$

### Convergence

Sequence $\{x_n\}$ converges to $x$ if:

$$\lim_{n \to \infty} d(x_n, x) = 0$$

**Cauchy sequence**: For every $\epsilon > 0$, there exists $N$ such that $d(x_n, x_m) < \epsilon$ for all $n, m \geq N$.

### Completeness

Metric space is complete if every Cauchy sequence converges.

**Examples**:
- $\mathbb{R}^n$: complete
- $C[a,b]$: complete
- $L^p[a,b]$: complete (Riesz-Fischer theorem)

## Banach Spaces

### Definition

A Banach space is a complete normed vector space $(X, \|\cdot\|)$.

**Norm**: Function $\|\cdot\|: X \to \mathbb{R}$ satisfying:
1. $\|x\| \geq 0$ with equality if and only if $x = 0$
2. $\|\alpha x\| = |\alpha|\|x\|$
3. $\|x + y\| \leq \|x\| + \|y\|$

**Completeness**: Every Cauchy sequence converges.

### Examples

- **$\mathbb{R}^n$**: With any $p$-norm, Banach space
- **$L^p(\Omega)$**: Space of $p$-integrable functions with norm $\|f\|_p = (\int_\Omega |f|^p)^{1/p}$
- **$C[a,b]$**: Continuous functions with $\|f\|_\infty = \max_{x \in [a,b]} |f(x)|$
- **$\ell^p$**: Sequences with $\|x\|_p = (\sum_{i=1}^{\infty} |x_i|^p)^{1/p} < \infty$

### Dual Space

Dual space $X^*$ consists of continuous linear functionals $f: X \to \mathbb{R}$.

**Norm**: $\|f\|_{X^*} = \sup_{\|x\| \leq 1} |f(x)|$

**Examples**:
- $(\mathbb{R}^n)^* \cong \mathbb{R}^n$ (Riesz representation)
- $(L^p)^* \cong L^q$ where $1/p + 1/q = 1$ (Riesz representation)

## Hilbert Spaces

### Definition

A Hilbert space is a complete inner product space $(H, \langle \cdot, \cdot \rangle)$.

**Inner product**: Function $\langle \cdot, \cdot \rangle: H \times H \to \mathbb{C}$ satisfying:
1. $\langle x, x \rangle \geq 0$ with equality if and only if $x = 0$
2. $\langle x, y \rangle = \overline{\langle y, x \rangle}$
3. $\langle \alpha x + \beta y, z \rangle = \alpha\langle x, z \rangle + \beta\langle y, z \rangle$

**Norm induced**: $\|x\| = \sqrt{\langle x, x \rangle}$

### Examples

- **$\mathbb{R}^n$**: With standard inner product $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T\mathbf{y}$
- **$L^2(\Omega)$**: With $\langle f, g \rangle = \int_\Omega f\overline{g}$
- **$\ell^2$**: Square-summable sequences with $\langle x, y \rangle = \sum_{i=1}^{\infty} x_i \overline{y_i}$

### Orthogonality

Vectors $x, y \in H$ are orthogonal if $\langle x, y \rangle = 0$.

**Orthogonal complement**: For subspace $M \subseteq H$:

$$M^\perp = \{x \in H : \langle x, m \rangle = 0 \text{ for all } m \in M\}$$

**Projection theorem**: For closed subspace $M$ and $x \in H$, there exists unique $m \in M$ minimizing $\|x - m\|$:

$$m = \text{proj}_M(x)$$

and $x - m \in M^\perp$.

### Orthonormal Bases

Set $\{e_i\}$ is orthonormal if $\langle e_i, e_j \rangle = \delta_{ij}$.

**Complete orthonormal system**: Every $x \in H$ can be written as:

$$x = \sum_{i=1}^{\infty} \langle x, e_i \rangle e_i$$

**Parseval's identity**: $\|x\|^2 = \sum_{i=1}^{\infty} |\langle x, e_i \rangle|^2$

### Riesz Representation Theorem

For Hilbert space $H$ and continuous linear functional $f \in H^*$, there exists unique $y \in H$ such that:

$$f(x) = \langle x, y \rangle$$

for all $x \in H$, and $\|f\|_{H^*} = \|y\|$.

**Consequence**: $H^* \cong H$ (isometric isomorphism).

## Operators

### Bounded Linear Operators

Operator $T: X \to Y$ between normed spaces is bounded if:

$$\|T\| = \sup_{\|x\| \leq 1} \|Tx\| < \infty$$

**Equivalence**: $T$ bounded if and only if $T$ continuous.

**Space of operators**: $B(X, Y)$ is Banach space with operator norm.

### Compact Operators

Operator $T: X \to Y$ is compact if image of bounded set is relatively compact (closure is compact).

**Properties**:
- Limit of compact operators is compact
- Composition with bounded operator is compact
- Finite-rank operators are compact

**Spectral theory**: Compact operators on Hilbert space have discrete spectrum (eigenvalues with finite multiplicity, accumulating only at 0).

### Self-Adjoint Operators

For Hilbert space $H$, operator $T: H \to H$ is self-adjoint if:

$$\langle Tx, y \rangle = \langle x, Ty \rangle$$

for all $x, y \in H$.

**Spectral theorem**: Self-adjoint compact operator has orthonormal eigenbasis:

$$Tx = \sum_{i=1}^{\infty} \lambda_i \langle x, e_i \rangle e_i$$

where $\lambda_i$ are eigenvalues and $e_i$ are eigenvectors.

## Fixed Point Theorems

### Banach Fixed Point Theorem

For complete metric space $(X, d)$ and contraction $T: X \to X$:

$$d(Tx, Ty) \leq \alpha d(x, y)$$

for some $\alpha < 1$, there exists unique fixed point $x^*$ such that $Tx^* = x^*$.

**Iteration**: $x_{n+1} = Tx_n$ converges to $x^*$ for any initial $x_0$.

**Rate**: $d(x_n, x^*) \leq \frac{\alpha^n}{1-\alpha} d(x_1, x_0)$

**Applications**:
- Existence and uniqueness of solutions to ODEs
- Iterative methods for solving equations
- Bellman equations in dynamic programming

### Brouwer Fixed Point Theorem

For continuous function $f: B^n \to B^n$ on closed unit ball $B^n \subset \mathbb{R}^n$, there exists fixed point $f(x^*) = x^*$.

**Generalization**: Works for any compact convex set in $\mathbb{R}^n$.

**Applications**:
- Existence of Nash equilibria
- Existence of solutions to nonlinear equations
- Economic equilibrium theory

### Schauder Fixed Point Theorem

For Banach space $X$, compact convex set $K \subseteq X$, and continuous $f: K \to K$, there exists fixed point.

**Applications**: 
- Existence of solutions to PDEs
- Nonlinear analysis

## Applications in Quantitative Finance

### Pricing Kernel in Complete Markets

**State space**: $(\Omega, \mathcal{F}, \mathbb{P})$ with states $\omega \in \Omega$.

**Payoff space**: $L^2(\Omega, \mathcal{F}, \mathbb{P})$ (square-integrable payoffs).

**Pricing functional**: Linear functional $\pi: L^2 \to \mathbb{R}$ assigning prices to payoffs.

**Riesz representation**: There exists unique $M \in L^2$ (pricing kernel) such that:

$$\pi(X) = E[MX]$$

for all payoffs $X$.

**Properties**:
- $M > 0$ (no arbitrage)
- $E[M] = e^{-rT}$ (normalization)
- $M$ encodes risk-neutral probabilities: $dQ/dP = M e^{rT}$

### Reproducing Kernel Hilbert Spaces (RKHS)

**Kernel**: Function $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ such that matrix $[K(x_i, x_j)]$ is positive semidefinite.

**RKHS**: Hilbert space $H$ of functions $f: \mathcal{X} \to \mathbb{R}$ with reproducing property:

$$f(x) = \langle f, K(\cdot, x) \rangle_H$$

**Examples**:
- Gaussian RBF kernel: $K(x, y) = \exp(-\gamma\|x-y\|^2)$
- Polynomial kernel: $K(x, y) = (1 + x^Ty)^d$

**Applications**:
- **Kernel methods**: Support vector machines, Gaussian processes
- **Nonparametric regression**: Estimate conditional expectations
- **Option pricing**: Model pricing functionals

### Functional Regression

**Model**: $Y = \int_0^T \beta(t)X(t)dt + \epsilon$

where $X(t)$ is functional predictor and $\beta(t)$ is functional coefficient.

**Estimation**: Minimize:

$$\sum_{i=1}^{n} \left(Y_i - \int_0^T \beta(t)X_i(t)dt\right)^2 + \lambda \|\beta\|^2_H$$

where $H$ is RKHS.

**Applications**:
- **Yield curve modeling**: Model interest rates as functions of time
- **Volatility surface**: Model implied volatility as function of strike and maturity
- **Term structure**: Model forward rates

### Optimal Control

**Hamilton-Jacobi-Bellman**: For value function $V(t, x)$:

$$0 = \frac{\partial V}{\partial t} + \max_u \left[f(t, x, u) + \mathcal{L}V(t, x)\right]$$

where $\mathcal{L}$ is infinitesimal generator.

**Existence**: Use fixed point theorems to show existence of solution.

**Applications**:
- **Portfolio optimization**: Merton problem
- **Optimal stopping**: American options
- **Consumption-investment**: Lifecycle models

### Measure-Valued Processes

**Space of measures**: $M(\mathbb{R})$ with weak topology.

**Convergence**: $\mu_n \to \mu$ if $\int f d\mu_n \to \int f d\mu$ for all continuous bounded $f$.

**Applications**:
- **Large portfolio limits**: As $n \to \infty$, portfolio distribution converges
- **Mean field games**: Limit of $n$-player games
- **Systemic risk**: Distribution of bank sizes

### Spectral Methods

**Eigenfunction expansion**: For operator $\mathcal{L}$, expand solution:

$$u(t, x) = \sum_{i=1}^{\infty} c_i(t) \phi_i(x)$$

where $\phi_i$ are eigenfunctions of $\mathcal{L}$.

**Applications**:
- **Option pricing**: Expand option price in eigenfunctions of generator
- **Volatility modeling**: Karhunen-Loève expansion of volatility process
- **Factor models**: Principal components as eigenfunctions

### Stochastic Analysis

**Itô calculus**: For $dX_t = \mu_t dt + \sigma_t dW_t$:

$$df(X_t) = f'(X_t)dX_t + \frac{1}{2}f''(X_t)\sigma_t^2 dt$$

**Functional Itô formula**: For functional $F(t, X_{[0,t]})$:

$$dF = \frac{\partial F}{\partial t} dt + \int_0^t \frac{\delta F}{\delta X_s} dX_s + \frac{1}{2}\int_0^t \frac{\delta^2 F}{\delta X_s^2} d\langle X \rangle_s$$

where $\delta F/\delta X_s$ is functional derivative.

**Applications**:
- **Path-dependent options**: Asian, lookback options
- **Volatility derivatives**: VIX, variance swaps
- **Rough volatility**: Fractional Brownian motion models
