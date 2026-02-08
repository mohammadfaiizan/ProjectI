# Martingales and Stopping Times

## Martingales

### Definition

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space with a filtration $\{\mathcal{F}_t\}_{t \geq 0}$. A stochastic process $\{M_t\}_{t \geq 0}$ is a martingale with respect to $\{\mathcal{F}_t\}$ if:

1. $M_t$ is $\mathcal{F}_t$-measurable (adapted) for all $t$
2. $E[|M_t|] < \infty$ for all $t$ (integrable)
3. $E[M_t | \mathcal{F}_s] = M_s$ almost surely for all $s \leq t$ (martingale property)

**Interpretation**: A martingale represents a "fair game" where, given all information up to time $s$, the expected future value equals the current value. No strategy can yield a positive expected gain.

### Discrete-Time Martingales

For a discrete-time process $\{M_n\}_{n=0,1,2,\ldots}$ with filtration $\{\mathcal{F}_n\}$, the martingale property is:

$$E[M_{n+1} | \mathcal{F}_n] = M_n$$

**Example**: Symmetric random walk. Let $X_1, X_2, \ldots$ be independent with $\mathbb{P}(X_i = 1) = \mathbb{P}(X_i = -1) = 1/2$. Then:

$$M_n = \sum_{i=1}^{n} X_i$$

is a martingale with respect to $\mathcal{F}_n = \sigma(X_1, \ldots, X_n)$.

**Example**: Product martingale. If $Y_1, Y_2, \ldots$ are independent with $E[Y_i] = 1$, then:

$$M_n = \prod_{i=1}^{n} Y_i$$

is a martingale.

### Continuous-Time Martingales

**Example**: Brownian motion. A standard Brownian motion $\{W_t\}$ is a martingale:

$$E[W_t | \mathcal{F}_s] = W_s$$

since $W_t - W_s$ is independent of $\mathcal{F}_s$ and $E[W_t - W_s] = 0$.

**Example**: Geometric Brownian motion (discounted). If $S_t = S_0 e^{(r-\sigma^2/2)t + \sigma W_t}$ and $B_t = e^{rt}$, then the discounted process:

$$M_t = \frac{S_t}{B_t} = S_0 e^{-\sigma^2 t/2 + \sigma W_t}$$

is a martingale under the physical measure.

### Submartingales and Supermartingales

A process $\{X_t\}$ is a **submartingale** if:

$$E[X_t | \mathcal{F}_s] \geq X_s \quad \text{for } s \leq t$$

A process $\{X_t\}$ is a **supermartingale** if:

$$E[X_t | \mathcal{F}_s] \leq X_s \quad \text{for } s \leq t$$

**Interpretation**:
- Submartingale: favorable game (expected value increases)
- Supermartingale: unfavorable game (expected value decreases)
- Martingale: fair game (expected value constant)

**Example**: If $\{M_t\}$ is a martingale and $\varphi$ is a convex function with $E[|\varphi(M_t)|] < \infty$, then $\{\varphi(M_t)\}$ is a submartingale (by Jensen's inequality).

**Example**: In risk-neutral pricing, if $S_t$ is an asset price and $r$ is the risk-free rate, then $\{e^{-rt} S_t\}$ is a martingale under the risk-neutral measure. Under the physical measure, it may be a submartingale if the asset has positive risk premium.

## Martingale Convergence Theorem

### Doob's Martingale Convergence Theorem

If $\{M_n\}$ is a submartingale with $\sup_n E[M_n^+] < \infty$ (where $M_n^+ = \max(M_n, 0)$), then:

$$M_n \to M_\infty \quad \text{almost surely}$$

as $n \to \infty$, where $M_\infty$ is an integrable random variable.

**Corollary**: If $\{M_n\}$ is a non-negative supermartingale, then $M_n \to M_\infty$ almost surely.

**Application**: In optimal stopping problems, the value function converges as the horizon extends to infinity.

### $L^p$ Convergence

If $\{M_n\}$ is a martingale with $\sup_n E[|M_n|^p] < \infty$ for $p > 1$, then $M_n \to M_\infty$ in $L^p$:

$$\lim_{n \to \infty} E[|M_n - M_\infty|^p] = 0$$

For $p = 2$, this is equivalent to:

$$E[(M_n - M_\infty)^2] \to 0$$

### Uniform Integrability and Convergence

If $\{M_n\}$ is a uniformly integrable martingale, then $M_n \to M_\infty$ in $L^1$:

$$E[|M_n - M_\infty|] \to 0$$

and:

$$M_n = E[M_\infty | \mathcal{F}_n]$$

This is the martingale representation property.

## Stopping Times

### Definition

A stopping time (or optional time) $\tau$ with respect to a filtration $\{\mathcal{F}_t\}$ is a random variable $\tau: \Omega \to [0, \infty]$ such that:

$$\{\tau \leq t\} \in \mathcal{F}_t \quad \forall t \geq 0$$

**Interpretation**: At time $t$, we can determine whether $\tau$ has occurred ($\tau \leq t$) based on information available up to time $t$.

**Equivalent definition**: $\{\tau < t\} \in \mathcal{F}_t$ for all $t$ (for right-continuous filtrations).

### Examples

**Example**: First hitting time. For a process $\{X_t\}$ and a set $A$:

$$\tau_A = \inf\{t \geq 0 : X_t \in A\}$$

is a stopping time (provided the filtration is right-continuous).

**Example**: Constant time. For any constant $c \geq 0$, $\tau = c$ is a stopping time.

**Example**: First time a process exceeds a level. For a process $\{X_t\}$:

$$\tau = \inf\{t \geq 0 : X_t > a\}$$

is a stopping time.

**Non-example**: Last time a process exceeds a level is generally not a stopping time (requires future information).

### Sigma-Algebra of a Stopping Time

For a stopping time $\tau$, the sigma-algebra $\mathcal{F}_\tau$ is:

$$\mathcal{F}_\tau = \{A \in \mathcal{F} : A \cap \{\tau \leq t\} \in \mathcal{F}_t \text{ for all } t\}$$

**Interpretation**: $\mathcal{F}_\tau$ contains events that are "known" at time $\tau$.

**Properties**:
- If $\tau = t$ (constant), then $\mathcal{F}_\tau = \mathcal{F}_t$
- If $\sigma \leq \tau$, then $\mathcal{F}_\sigma \subseteq \mathcal{F}_\tau$
- If $\{M_t\}$ is adapted and $\tau$ is a stopping time, then $M_\tau$ is $\mathcal{F}_\tau$-measurable

## Optional Stopping Theorem

### Discrete-Time Optional Stopping Theorem

Let $\{M_n\}$ be a martingale and $\tau$ be a stopping time. If one of the following conditions holds:

1. $\tau$ is bounded (there exists $N$ such that $\tau \leq N$ almost surely)
2. $\tau$ is almost surely finite and $\{M_{n \wedge \tau}\}$ is uniformly integrable
3. $E[\tau] < \infty$ and $|M_{n+1} - M_n| \leq K$ for some constant $K$

then:

$$E[M_\tau] = E[M_0]$$

**Warning**: The theorem does not hold in general without additional conditions. For example, for a symmetric random walk $M_n$ starting at 0 and $\tau = \inf\{n : M_n = 1\}$, we have $E[M_\tau] = 1 \neq 0 = E[M_0]$, but $E[\tau] = \infty$.

**Example**: Gambler's ruin. A gambler starts with $a$ dollars and plays until reaching either $0$ or $N$ dollars. Each bet is fair (martingale). If $\tau$ is the stopping time, then:

$$E[M_\tau] = a = 0 \cdot \mathbb{P}(\text{ruin}) + N \cdot \mathbb{P}(\text{win})$$

so $\mathbb{P}(\text{win}) = a/N$.

### Continuous-Time Optional Stopping Theorem

Let $\{M_t\}$ be a right-continuous martingale and $\tau$ be a stopping time. If one of the following holds:

1. $\tau$ is bounded
2. $\{M_{t \wedge \tau}\}$ is uniformly integrable
3. $E[\tau] < \infty$ and $|M_t - M_s| \leq K|t-s|$ for some $K$

then:

$$E[M_\tau] = E[M_0]$$

**Application**: In American option pricing, the optimal exercise time $\tau^*$ maximizes $E[e^{-r\tau} (S_\tau - K)^+]$. The optional stopping theorem helps characterize $\tau^*$.

### Wald's Identity

For a random walk $S_n = \sum_{i=1}^{n} X_i$ where $X_i$ are i.i.d. with $E[X_i] = \mu$ and $\tau$ is a stopping time with $E[\tau] < \infty$, then:

$$E[S_\tau] = \mu E[\tau]$$

**Proof**: Use the fact that $M_n = S_n - n\mu$ is a martingale and apply optional stopping.

**Example**: For a symmetric random walk ($\mu = 0$) and a stopping time with $E[\tau] < \infty$:

$$E[S_\tau] = 0$$

## Doob's Maximal Inequality

### Discrete-Time Maximal Inequality

For a submartingale $\{M_n\}$ and $\lambda > 0$:

$$\lambda \mathbb{P}\left(\max_{0 \leq k \leq n} M_k \geq \lambda\right) \leq E[M_n^+]$$

where $M_n^+ = \max(M_n, 0)$.

**Corollary**: For a non-negative submartingale:

$$\mathbb{P}\left(\max_{0 \leq k \leq n} M_k \geq \lambda\right) \leq \frac{E[M_n]}{\lambda}$$

### Continuous-Time Maximal Inequality

For a right-continuous submartingale $\{M_t\}$:

$$\lambda \mathbb{P}\left(\sup_{0 \leq s \leq t} M_s \geq \lambda\right) \leq E[M_t^+]$$

### $L^p$ Maximal Inequality

For a martingale $\{M_n\}$ and $p > 1$:

$$E\left[\left(\max_{0 \leq k \leq n} |M_k|\right)^p\right] \leq \left(\frac{p}{p-1}\right)^p E[|M_n|^p]$$

This is Doob's $L^p$ inequality.

**Application**: Bounds on maximum drawdown in portfolio value.

## Doob Decomposition

### Discrete-Time Doob Decomposition

Any adapted integrable process $\{X_n\}$ can be uniquely decomposed as:

$$X_n = M_n + A_n$$

where:
- $\{M_n\}$ is a martingale
- $\{A_n\}$ is a predictable process (i.e., $A_n$ is $\mathcal{F}_{n-1}$-measurable) with $A_0 = 0$

The decomposition is:

$$A_n = \sum_{k=1}^{n} E[X_k - X_{k-1} | \mathcal{F}_{k-1}]$$

$$M_n = X_n - A_n$$

**Interpretation**: $A_n$ is the "drift" (predictable component) and $M_n$ is the "noise" (martingale component).

**Example**: If $\{X_n\}$ is a submartingale, then $A_n$ is non-decreasing.

### Continuous-Time Doob-Meyer Decomposition

For a right-continuous submartingale $\{X_t\}$ of class D (uniformly integrable family $\{X_\tau : \tau \text{ bounded stopping time}\}$), there exists a unique decomposition:

$$X_t = M_t + A_t$$

where:
- $\{M_t\}$ is a right-continuous martingale
- $\{A_t\}$ is an increasing predictable process with $A_0 = 0$

This is the Doob-Meyer decomposition.

**Application**: In semimartingale theory, any semimartingale can be decomposed into a local martingale and a finite-variation process.

## Martingale Representation Theorem

### Discrete-Time Representation

In a complete market, any square-integrable martingale $\{M_n\}$ adapted to the filtration generated by independent increments can be represented as:

$$M_n = M_0 + \sum_{k=1}^{n} H_k (X_k - X_{k-1})$$

for some predictable process $\{H_k\}$, where $\{X_k\}$ is the underlying process.

### Continuous-Time Representation (Brownian Motion)

Let $\{W_t\}$ be a Brownian motion and $\{\mathcal{F}_t\}$ be its natural filtration. If $\{M_t\}$ is a square-integrable martingale with respect to $\{\mathcal{F}_t\}$, then there exists a predictable process $\{H_t\}$ with $E[\int_0^T H_t^2 dt] < \infty$ such that:

$$M_t = M_0 + \int_0^t H_s dW_s$$

**Uniqueness**: The representation is unique in the sense that if:

$$M_t = M_0 + \int_0^t H_s dW_s = M_0 + \int_0^t \tilde{H}_s dW_s$$

then $H_t = \tilde{H}_t$ almost everywhere.

**Application**: In complete markets, any derivative payoff can be perfectly hedged using the underlying asset. The hedging strategy is given by $H_t$.

### Clark-Ocone Formula

For a functional $F$ of the Brownian path, the martingale representation is:

$$E[F | \mathcal{F}_t] = E[F] + \int_0^t E[D_s F | \mathcal{F}_s] dW_s$$

where $D_s F$ is the Malliavin derivative.

## Applications in Finance

### Fair Pricing

In a risk-neutral world, the price of a derivative is:

$$V_t = E^Q\left[e^{-r(T-t)} V_T \Big| \mathcal{F}_t\right]$$

The discounted price process $\{e^{-rt} V_t\}$ is a $Q$-martingale. This ensures no arbitrage.

**Example**: European call option. Under Black-Scholes:

$$C_t = S_t \Phi(d_1) - Ke^{-r(T-t)} \Phi(d_2)$$

where $d_1$ and $d_2$ depend on $S_t$, $K$, $r$, $\sigma$, and $T-t$. The process $\{e^{-rt} C_t\}$ is a martingale.

### Optimal Exercise Boundaries

For American options, the optimal exercise time $\tau^*$ maximizes:

$$E[e^{-r\tau} (S_\tau - K)^+]$$

over all stopping times $\tau \leq T$.

The continuation region is $\{S_t < b_t\}$ and the exercise region is $\{S_t \geq b_t\}$, where $b_t$ is the optimal exercise boundary. In the continuation region, the option value satisfies:

$$V_t = E[e^{-r(\tau^* - t)} (S_{\tau^*} - K)^+ | \mathcal{F}_t]$$

and $\{e^{-rt} V_t\}$ is a supermartingale (strictly before exercise, a martingale after optimal exercise).

### Hedging Strategies

The martingale representation theorem gives the hedging strategy. For a derivative with payoff $V_T$:

$$V_t = E^Q[e^{-r(T-t)} V_T | \mathcal{F}_t] = V_0 + \int_0^t \Delta_s dS_s$$

where $\Delta_t = \frac{\partial V_t}{\partial S_t}$ is the delta hedge ratio.

**Self-financing**: The portfolio value $\Pi_t = V_t - \Delta_t S_t$ invested in the risk-free asset satisfies:

$$d\Pi_t = r \Pi_t dt$$

ensuring the strategy is self-financing.

### Risk-Neutral Valuation

The fundamental theorem of asset pricing states that absence of arbitrage is equivalent to the existence of a risk-neutral measure $Q$ under which discounted asset prices are martingales:

$$E^Q\left[\frac{S_T}{B_T} \Big| \mathcal{F}_t\right] = \frac{S_t}{B_t}$$

This martingale property is the cornerstone of derivative pricing.

### Optimal Stopping and American Options

The value of an American option is:

$$V_t = \sup_{\tau \in \mathcal{T}_{t,T}} E^Q[e^{-r(\tau-t)} (S_\tau - K)^+ | \mathcal{F}_t]$$

where $\mathcal{T}_{t,T}$ is the set of stopping times in $[t,T]$. The optimal stopping time is:

$$\tau^* = \inf\{s \geq t : S_s \geq b_s\}$$

where $b_s$ is the exercise boundary. The process $\{e^{-rt} V_t\}$ is a supermartingale, and equality holds in the continuation region.

### Snell Envelope

For an American option, the Snell envelope is:

$$U_t = \operatorname{ess}\sup_{\tau \in \mathcal{T}_{t,T}} E[Z_\tau | \mathcal{F}_t]$$

where $Z_t$ is the exercise value. The Snell envelope is the smallest supermartingale dominating the exercise value and equals the option value.

Martingales and stopping times provide the mathematical framework for understanding fair games, optimal stopping, and derivative pricing in financial markets.
