# Brownian Motion and Random Walks

## Random Walks

### Simple Random Walk

One-dimensional simple random walk:

$$S_n = \sum_{i=1}^{n} X_i$$

where $X_i$ are i.i.d. with $\mathbb{P}(X_i = 1) = p$ and $\mathbb{P}(X_i = -1) = 1-p$.

**Symmetric random walk**: $p = 1/2$.

**Properties**:
- $E[S_n] = n(2p - 1)$
- $\text{Var}(S_n) = 4np(1-p)$
- For symmetric case: $E[S_n] = 0$, $\text{Var}(S_n) = n$

### Scaled Random Walk

**Time scaling**: For time step $\Delta t = 1/n$:

$$W_n(t) = \frac{1}{\sqrt{n}} S_{\lfloor nt \rfloor}$$

**Properties**:
- $E[W_n(t)] = 0$
- $\text{Var}(W_n(t)) = t$ (for symmetric case)
- Central limit theorem: $W_n(t) \xrightarrow{d} \mathcal{N}(0, t)$

### Donsker's Theorem

**Functional central limit theorem**: For symmetric random walk:

$$\{W_n(t) : t \in [0,1]\} \xrightarrow{d} \{W(t) : t \in [0,1]\}$$

where $W(t)$ is Brownian motion, and convergence is in distribution on space of continuous functions.

**Implications**:
- Scaled random walk converges to Brownian motion
- Many properties of random walks extend to Brownian motion
- Provides discrete approximation to continuous processes

## Brownian Motion

### Definition

A standard Brownian motion (Wiener process) $\{W(t) : t \geq 0\}$ is a stochastic process satisfying:

1. **$W(0) = 0$** (starts at origin)
2. **Independent increments**: For $0 \leq s < t$, $W(t) - W(s)$ is independent of $\{W(u) : u \leq s\}$
3. **Stationary increments**: $W(t) - W(s) \sim \mathcal{N}(0, t-s)$
4. **Continuous paths**: $W(t)$ is continuous almost surely

**Alternative definition**: Gaussian process with:
- $E[W(t)] = 0$
- $\text{Cov}(W(s), W(t)) = \min(s, t)$

### Properties

**Finite-dimensional distributions**: For $0 \leq t_1 < \cdots < t_n$:

$$(W(t_1), \ldots, W(t_n)) \sim \mathcal{N}_n(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\Sigma_{ij} = \min(t_i, t_j)$.

**Scaling**: For $c > 0$, $\{c^{-1/2}W(ct) : t \geq 0\}$ is Brownian motion.

**Time reversal**: $\{W(T) - W(T-t) : t \in [0,T]\}$ is Brownian motion.

**Symmetry**: $\{-W(t) : t \geq 0\}$ is Brownian motion.

### Non-Differentiability

**Theorem**: Brownian motion is nowhere differentiable almost surely.

**Proof idea**: For any $t$, consider:

$$\lim_{h \to 0} \frac{W(t+h) - W(t)}{h}$$

Since $W(t+h) - W(t) \sim \mathcal{N}(0, h)$, the limit doesn't exist.

**Implications**: 
- Classical calculus doesn't apply
- Need stochastic calculus (Itô calculus)
- Quadratic variation is finite (unlike differentiable functions)

### Quadratic Variation

**Definition**: For partition $\Pi = \{0 = t_0 < t_1 < \cdots < t_n = T\}$:

$$Q_\Pi = \sum_{i=1}^{n} (W(t_i) - W(t_{i-1}))^2$$

**Theorem**: As mesh $|\Pi| = \max_i(t_i - t_{i-1}) \to 0$:

$$Q_\Pi \xrightarrow{L^2} T$$

**Almost sure convergence**: For nested partitions, $Q_\Pi \to T$ almost surely.

**Notation**: $dW(t)^2 = dt$ (formal multiplication rule).

**Comparison**: For differentiable function $f$:

$$\sum (f(t_i) - f(t_{i-1}))^2 \leq \max |f'|^2 \sum (t_i - t_{i-1})^2 \to 0$$

Brownian motion has infinite total variation but finite quadratic variation.

## Geometric Brownian Motion

### Definition

Geometric Brownian motion:

$$S(t) = S(0) \exp\left((\mu - \frac{\sigma^2}{2})t + \sigma W(t)\right)$$

**SDE form**: $dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$

**Properties**:
- $E[S(t)] = S(0)e^{\mu t}$
- $\text{Var}(S(t)) = S(0)^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$
- Log-normal distribution: $\ln S(t) \sim \mathcal{N}(\ln S(0) + (\mu - \sigma^2/2)t, \sigma^2 t)$

**Applications**: 
- Stock price model (Black-Scholes)
- Asset prices in continuous-time finance

## Reflection Principle

### Statement

For Brownian motion $W(t)$ and level $a > 0$, define:

$$\tau_a = \inf\{t \geq 0 : W(t) = a\}$$

**Reflection principle**: For $t > 0$:

$$\mathbb{P}(\tau_a \leq t, W(t) \leq b) = \mathbb{P}(W(t) \geq 2a - b)$$

for $b \leq a$.

**Geometric interpretation**: After hitting $a$, reflect path about level $a$.

### Maximum Distribution

**Maximum**: $M(t) = \max_{0 \leq s \leq t} W(s)$

**Distribution**: For $a > 0$:

$$\mathbb{P}(M(t) \geq a) = 2\mathbb{P}(W(t) \geq a) = 2(1 - \Phi(a/\sqrt{t}))$$

**PDF**: 

$$f_{M(t)}(a) = \frac{2}{\sqrt{2\pi t}} e^{-a^2/(2t)}, \quad a \geq 0$$

**Joint distribution**: For $a > 0$ and $b \leq a$:

$$\mathbb{P}(M(t) \geq a, W(t) \leq b) = \mathbb{P}(W(t) \geq 2a - b)$$

### Minimum Distribution

**Minimum**: $m(t) = \min_{0 \leq s \leq t} W(s)$

**Distribution**: For $a < 0$:

$$\mathbb{P}(m(t) \leq a) = 2\mathbb{P}(W(t) \leq a) = 2\Phi(a/\sqrt{t})$$

**Symmetry**: $m(t) \sim -M(t)$.

## First Passage Times

### Hitting Time Distribution

**Hitting time**: $\tau_a = \inf\{t \geq 0 : W(t) = a\}$ for $a \neq 0$.

**Distribution**: For $a > 0$:

$$f_{\tau_a}(t) = \frac{a}{\sqrt{2\pi t^3}} e^{-a^2/(2t)}, \quad t > 0$$

**CDF**: 

$$F_{\tau_a}(t) = 2(1 - \Phi(a/\sqrt{t})) = 2\Phi(-a/\sqrt{t})$$

**Moments**: 
- $E[\tau_a] = \infty$ (infinite expected hitting time)
- But $\mathbb{P}(\tau_a < \infty) = 1$ (hits level almost surely)

### Hitting Probabilities

**Two-sided barrier**: For $a < 0 < b$:

$$\mathbb{P}(\tau_a < \tau_b) = \frac{b}{b - a}$$

$$\mathbb{P}(\tau_b < \tau_a) = \frac{-a}{b - a}$$

**Interpretation**: Probability of hitting lower barrier first is proportional to distance to upper barrier.

**Expected time**: 

$$E[\min(\tau_a, \tau_b)] = -ab$$

**Derivation**: Use optional stopping theorem with martingale $W(t)^2 - t$.

### Drifted Brownian Motion

**Process**: $X(t) = \mu t + \sigma W(t)$

**Hitting time**: $\tau_a = \inf\{t : X(t) = a\}$

**Laplace transform**: For $\theta > 0$:

$$E[e^{-\theta \tau_a}] = \exp\left(\frac{a\mu}{\sigma^2} - \frac{a}{\sigma^2}\sqrt{\mu^2 + 2\theta\sigma^2}\right)$$

**Probability of hitting**: For $\mu < 0$ and $a > 0$:

$$\mathbb{P}(\tau_a < \infty) = e^{2\mu a/\sigma^2}$$

**Applications**: 
- Default modeling (first passage to default barrier)
- Option pricing (barrier options)

## Applications in Finance

### Stock Price Modeling

**Geometric Brownian motion**: 

$$S(t) = S(0) \exp((\mu - \sigma^2/2)t + \sigma W(t))$$

**Properties**:
- Log-normal distribution
- Constant volatility $\sigma$
- Drift $\mu$ (expected return)

**Limitations**:
- Constant volatility (volatility smile not captured)
- Normal log returns (fat tails not captured)
- No jumps

### Barrier Options

**Down-and-out call**: Option expires if stock hits barrier $B < S(0)$.

**Pricing**: Use reflection principle:

$$C_{\text{DO}} = C_{\text{BS}} - \left(\frac{B}{S(0)}\right)^{2r/\sigma^2 - 1} C_{\text{BS}}(B^2/S(0))$$

where $C_{\text{BS}}$ is Black-Scholes price.

**First passage**: Probability of hitting barrier:

$$\mathbb{P}(\min_{0 \leq s \leq T} S(s) \leq B) = \Phi\left(\frac{\ln(B/S(0)) - (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}\right) + \left(\frac{B}{S(0)}\right)^{2(\mu - \sigma^2/2)/\sigma^2} \Phi\left(\frac{\ln(B/S(0)) + (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}\right)$$

### Default Modeling

**First passage model**: Default occurs when firm value hits barrier:

$$\tau = \inf\{t : V(t) \leq D\}$$

where $V(t)$ follows geometric Brownian motion and $D$ is default barrier.

**Default probability**:

$$\mathbb{P}(\tau \leq T) = \Phi\left(\frac{\ln(D/V(0)) - (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}\right) + \left(\frac{D}{V(0)}\right)^{2(\mu - \sigma^2/2)/\sigma^2} \Phi\left(\frac{\ln(D/V(0)) + (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}\right)$$

### Drawdown Analysis

**Maximum drawdown**: 

$$\text{MDD} = \max_{0 \leq t \leq T} \left(\max_{0 \leq s \leq t} S(s) - S(t)\right)$$

**Distribution**: For geometric Brownian motion, use reflection principle.

**Expected maximum drawdown**: 

$$E[\text{MDD}] \approx \frac{\sigma^2}{2\mu} \left(1 - e^{-2\mu T/\sigma^2}\right)$$

for small $\mu$.

### Risk Metrics

**Value at Risk**: For log returns $r_t = \ln(S_t/S_{t-1}) \sim \mathcal{N}(\mu, \sigma^2)$:

$$\text{VaR}_{\alpha} = -(\mu + \sigma \Phi^{-1}(\alpha))$$

**Expected shortfall**: 

$$\text{ES}_{\alpha} = -\mu + \sigma \frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha}$$

where $\phi$ is standard normal PDF.

### Monte Carlo Simulation

**Discretization**: Euler-Maruyama scheme:

$$S_{t+\Delta t} = S_t \exp\left((\mu - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t} Z\right)$$

where $Z \sim \mathcal{N}(0,1)$.

**Convergence**: As $\Delta t \to 0$, converges to continuous process.

**Variance reduction**: Use antithetic variates, control variates.
