# Itô Calculus and Stochastic Integrals

## Construction of Stochastic Integrals

### Motivation

For deterministic function $f$, Riemann integral:

$$\int_0^T f(t) dW(t) = \lim_{|\Pi| \to 0} \sum_{i=1}^{n} f(\xi_i)(W(t_i) - W(t_{i-1}))$$

**Problem**: Limit depends on choice of evaluation point $\xi_i$ due to infinite variation of $W(t)$.

**Solution**: Fix evaluation point (e.g., left endpoint) to get well-defined limit.

### Itô Integral for Simple Processes

**Simple process**: $H(t) = \sum_{i=1}^{n} H_{i-1} \mathbf{1}_{[t_{i-1}, t_i)}(t)$ where $H_{i-1}$ is $\mathcal{F}_{t_{i-1}}$-measurable.

**Itô integral**:

$$I_T(H) = \int_0^T H(t) dW(t) = \sum_{i=1}^{n} H_{i-1}(W(t_i) - W(t_{i-1}))$$

**Key property**: $H_{i-1}$ is evaluated at left endpoint (adapted to filtration).

### Extension to General Processes

**Class of integrands**: Processes $H(t)$ such that:
1. $H(t)$ is adapted to filtration $\{\mathcal{F}_t\}$
2. $E[\int_0^T H(t)^2 dt] < \infty$

**Construction**: 
1. Approximate $H$ by simple processes $H_n$
2. Show $I_T(H_n)$ converges in $L^2$
3. Define $I_T(H) = \lim_{n \to \infty} I_T(H_n)$

**Itô isometry**: 

$$E\left[\left(\int_0^T H(t) dW(t)\right)^2\right] = E\left[\int_0^T H(t)^2 dt\right]$$

### Properties

1. **Linearity**: $\int_0^T (aH(t) + bG(t)) dW(t) = a\int_0^T H(t) dW(t) + b\int_0^T G(t) dW(t)$

2. **Martingale property**: $M(t) = \int_0^t H(s) dW(s)$ is martingale

3. **Quadratic variation**: $\langle M \rangle_t = \int_0^t H(s)^2 ds$

4. **Continuity**: Sample paths are continuous almost surely

## Itô's Lemma

### One-Dimensional Case

**Itô's lemma**: For $f \in C^2$ and Itô process $X(t)$:

$$df(X(t)) = f'(X(t)) dX(t) + \frac{1}{2}f''(X(t)) d\langle X \rangle_t$$

where $d\langle X \rangle_t$ is quadratic variation.

**For $dX(t) = \mu(t) dt + \sigma(t) dW(t)$**:

$$df(X(t)) = \left(f'(X(t))\mu(t) + \frac{1}{2}f''(X(t))\sigma(t)^2\right) dt + f'(X(t))\sigma(t) dW(t)$$

**Key difference from chain rule**: Extra term $\frac{1}{2}f''(X(t))\sigma(t)^2 dt$ due to quadratic variation.

### Multidimensional Itô's Lemma

For $f(t, X_1(t), \ldots, X_n(t))$ where $dX_i(t) = \mu_i(t) dt + \sum_{j=1}^{m} \sigma_{ij}(t) dW_j(t)$:

$$df = \frac{\partial f}{\partial t} dt + \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} dX_i + \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \frac{\partial^2 f}{\partial x_i \partial x_j} d\langle X_i, X_j \rangle_t$$

where $d\langle X_i, X_j \rangle_t = \sum_{k=1}^{m} \sigma_{ik}(t)\sigma_{jk}(t) dt$.

**Matrix form**: 

$$df = \frac{\partial f}{\partial t} dt + \nabla f^T d\mathbf{X} + \frac{1}{2}\text{tr}(\mathbf{H}_f \boldsymbol{\Sigma}) dt$$

where $\boldsymbol{\Sigma}_{ij} = \sum_k \sigma_{ik}\sigma_{jk}$.

### Examples

**Geometric Brownian motion**: For $f(x) = \ln x$ and $dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$:

$$d\ln S(t) = \frac{1}{S(t)} dS(t) - \frac{1}{2S(t)^2} d\langle S \rangle_t$$

$$= \frac{1}{S(t)}(\mu S(t) dt + \sigma S(t) dW(t)) - \frac{1}{2S(t)^2}\sigma^2 S(t)^2 dt$$

$$= (\mu - \frac{\sigma^2}{2}) dt + \sigma dW(t)$$

**Solution**: $S(t) = S(0) \exp((\mu - \sigma^2/2)t + \sigma W(t))$.

**Power function**: For $f(x) = x^n$:

$$d(S(t)^n) = nS(t)^{n-1} dS(t) + \frac{1}{2}n(n-1)S(t)^{n-2} d\langle S \rangle_t$$

$$= nS(t)^n\left(\mu + \frac{(n-1)\sigma^2}{2}\right) dt + n\sigma S(t)^n dW(t)$$

## Product Rule for Itô Processes

### Itô Product Rule

For Itô processes $X(t)$ and $Y(t)$:

$$d(X(t)Y(t)) = X(t) dY(t) + Y(t) dX(t) + d\langle X, Y \rangle_t$$

**Proof**: Apply Itô's lemma to $f(x, y) = xy$:

$$df = y dx + x dy + \frac{1}{2}(0 \cdot dx^2 + 2 \cdot dxdy + 0 \cdot dy^2)$$

$$= y dx + x dy + dxdy = y dx + x dy + d\langle X, Y \rangle_t$$

**Comparison**: Extra term $d\langle X, Y \rangle_t$ compared to deterministic product rule.

### Integration by Parts

**Formula**:

$$\int_0^T X(t) dY(t) = X(T)Y(T) - X(0)Y(0) - \int_0^T Y(t) dX(t) - \langle X, Y \rangle_T$$

**Derivation**: From product rule:

$$d(XY) = X dY + Y dX + d\langle X, Y \rangle$$

Integrate and rearrange.

## Stochastic Chain Rule

### General Form

For $f(t, X(t))$ where $X(t)$ is Itô process:

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX + \frac{1}{2}\frac{\partial^2 f}{\partial x^2} d\langle X \rangle$$

**Time-dependent function**: If $f(t, x) = g(t)h(x)$:

$$df = h(X(t))g'(t) dt + g(t)h'(X(t)) dX + \frac{1}{2}g(t)h''(X(t)) d\langle X \rangle$$

### Composition

For $f(X(t), Y(t))$:

$$df = \frac{\partial f}{\partial x} dX + \frac{\partial f}{\partial y} dY + \frac{1}{2}\left(\frac{\partial^2 f}{\partial x^2} d\langle X \rangle + 2\frac{\partial^2 f}{\partial x \partial y} d\langle X, Y \rangle + \frac{\partial^2 f}{\partial y^2} d\langle Y \rangle\right)$$

## Key Examples

### Geometric Brownian Motion

**SDE**: $dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$

**Solution**: $S(t) = S(0) \exp((\mu - \sigma^2/2)t + \sigma W(t))$

**Verification**: Apply Itô's lemma to $f(x) = \ln x$:

$$d\ln S = \frac{1}{S} dS - \frac{1}{2S^2} d\langle S \rangle = (\mu - \sigma^2/2) dt + \sigma dW$$

Integrate to get solution.

### Ornstein-Uhlenbeck Process

**SDE**: $dX(t) = -\theta X(t) dt + \sigma dW(t)$

**Solution**: Use integrating factor $e^{\theta t}$:

$$d(e^{\theta t}X(t)) = e^{\theta t} dX(t) + \theta e^{\theta t} X(t) dt = \sigma e^{\theta t} dW(t)$$

Integrate:

$$X(t) = X(0)e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)} dW(s)$$

**Properties**:
- Mean-reverting: $E[X(t)] = X(0)e^{-\theta t} \to 0$
- Stationary distribution: $\mathcal{N}(0, \sigma^2/(2\theta))$

**Applications**: 
- Interest rate modeling (Vasicek model)
- Volatility modeling

### CIR Process

**SDE**: $dX(t) = \kappa(\theta - X(t)) dt + \sigma\sqrt{X(t)} dW(t)$

**Properties**:
- Mean-reverting: $E[X(t)] \to \theta$ as $t \to \infty$
- Square-root process: Volatility proportional to $\sqrt{X(t)}$
- Non-negative: If $2\kappa\theta \geq \sigma^2$, process stays non-negative

**Applications**:
- Interest rate modeling (CIR model)
- Stochastic volatility (Heston model)

**Feller condition**: $2\kappa\theta \geq \sigma^2$ ensures $X(t) > 0$ almost surely.

### Heston Model

**Stock price**: $dS(t) = \mu S(t) dt + \sqrt{V(t)} S(t) dW_1(t)$

**Variance**: $dV(t) = \kappa(\theta - V(t)) dt + \sigma\sqrt{V(t)} dW_2(t)$

**Correlation**: $d\langle W_1, W_2 \rangle_t = \rho dt$

**Properties**:
- Stochastic volatility
- Mean-reverting variance
- Correlation between price and volatility

**Option pricing**: Characteristic function available for Fourier methods.

## Comparison with Stratonovich Calculus

### Stratonovich Integral

**Definition**: 

$$\int_0^T H(t) \circ dW(t) = \lim_{|\Pi| \to 0} \sum_{i=1}^{n} H\left(\frac{t_i + t_{i-1}}{2}\right)(W(t_i) - W(t_{i-1}))$$

**Evaluation at midpoint** instead of left endpoint.

### Chain Rule

**Stratonovich chain rule**: 

$$df(X(t)) = f'(X(t)) \circ dX(t)$$

**No correction term** (like deterministic calculus).

### Conversion

**Relationship**:

$$\int_0^T H(t) \circ dW(t) = \int_0^T H(t) dW(t) + \frac{1}{2}\int_0^T H'(t) dt$$

if $H$ is differentiable.

**SDE conversion**: For $dX = \mu dt + \sigma \circ dW$:

$$dX = \left(\mu + \frac{1}{2}\sigma'\sigma\right) dt + \sigma dW$$

(Itô form).

### When to Use

**Itô**: 
- Finance (martingale property important)
- Mathematical finance theory
- Risk-neutral pricing

**Stratonovich**:
- Physics (preserves chain rule)
- Geometric structures
- When conversion is natural

## Applications in Finance

### Option Pricing

**Black-Scholes**: For $dS = rS dt + \sigma S dW$ under risk-neutral measure:

**Call option**: $C(t, S) = e^{-r(T-t)} E^Q[\max(S_T - K, 0) | \mathcal{F}_t]$

**Itô's lemma**: 

$$dC = \frac{\partial C}{\partial t} dt + \frac{\partial C}{\partial S} dS + \frac{1}{2}\frac{\partial^2 C}{\partial S^2} d\langle S \rangle$$

**Black-Scholes PDE**: 

$$\frac{\partial C}{\partial t} + rS\frac{\partial C}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 C}{\partial S^2} = rC$$

### Greeks

**Delta**: $\Delta = \frac{\partial C}{\partial S}$

**Gamma**: $\Gamma = \frac{\partial^2 C}{\partial S^2}$

**Itô's lemma for portfolio**: 

$$d\Pi = \Delta dS + \frac{1}{2}\Gamma d\langle S \rangle + \cdots$$

**Delta hedging**: Set $\Delta = 0$ to eliminate $dS$ term.

### Stochastic Volatility

**Heston model**: Use Itô's lemma for $C(t, S, V)$:

$$dC = \frac{\partial C}{\partial t} dt + \frac{\partial C}{\partial S} dS + \frac{\partial C}{\partial V} dV + \frac{1}{2}\frac{\partial^2 C}{\partial S^2} d\langle S \rangle + \frac{\partial^2 C}{\partial S \partial V} d\langle S, V \rangle + \frac{1}{2}\frac{\partial^2 C}{\partial V^2} d\langle V \rangle$$

**PDE**: 

$$\frac{\partial C}{\partial t} + rS\frac{\partial C}{\partial S} + \kappa(\theta - V)\frac{\partial C}{\partial V} + \frac{1}{2}VS^2\frac{\partial^2 C}{\partial S^2} + \rho\sigma VS\frac{\partial^2 C}{\partial S \partial V} + \frac{1}{2}\sigma^2 V\frac{\partial^2 C}{\partial V^2} = rC$$

### Interest Rate Models

**Vasicek**: $dr(t) = \kappa(\theta - r(t)) dt + \sigma dW(t)$

**Bond price**: $P(t, T) = E^Q[\exp(-\int_t^T r(s) ds) | \mathcal{F}_t]$

**Itô's lemma**: Derive PDE for bond price.

**CIR**: $dr(t) = \kappa(\theta - r(t)) dt + \sigma\sqrt{r(t)} dW(t)$

**Affine structure**: Bond prices exponential-affine in $r(t)$.

### Risk-Neutral Pricing

**Change of numeraire**: For numeraire $N(t)$:

$$\frac{S(t)}{N(t)} = E^Q\left[\frac{S(T)}{N(T)}\Big|\mathcal{F}_t\right]$$

**Itô's lemma**: Derive dynamics of $S(t)/N(t)$.

**Applications**: 
- Foreign exchange options
- Inflation-linked derivatives
- Quanto options
