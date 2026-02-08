# Jump Diffusion and Lévy Processes

## Poisson Process

### Definition

A Poisson process $\{N(t) : t \geq 0\}$ with intensity $\lambda > 0$ is counting process satisfying:

1. **$N(0) = 0$**
2. **Independent increments**: $N(t) - N(s)$ independent of $\{N(u) : u \leq s\}$
3. **Stationary increments**: $N(t) - N(s) \sim \text{Poisson}(\lambda(t-s))$
4. **Jump size**: $N(t)$ increases by 1 at jump times

**Distribution**: 

$$\mathbb{P}(N(t) = n) = \frac{(\lambda t)^n e^{-\lambda t}}{n!}$$

**Properties**:
- $E[N(t)] = \lambda t$
- $\text{Var}(N(t)) = \lambda t$
- Inter-arrival times: $T_i - T_{i-1} \sim \text{Exponential}(\lambda)$

### Compound Poisson Process

**Definition**: 

$$X(t) = \sum_{i=1}^{N(t)} Y_i$$

where $\{Y_i\}$ are i.i.d. jump sizes and $N(t)$ is Poisson process.

**Properties**:
- $E[X(t)] = \lambda t E[Y_1]$
- $\text{Var}(X(t)) = \lambda t E[Y_1^2]$
- Characteristic function: $\phi_X(u, t) = \exp(\lambda t(\phi_Y(u) - 1))$

## Merton's Jump-Diffusion Model

### Stock Price Dynamics

**SDE**: 

$$dS(t) = \mu S(t-) dt + \sigma S(t-) dW(t) + S(t-)(e^J - 1) dN(t)$$

where:
- $W(t)$: Brownian motion
- $N(t)$: Poisson process with intensity $\lambda$
- $J$: Jump size (random variable)

**Notation**: $S(t-)$ denotes left limit (value before jump).

**Solution**: 

$$S(t) = S(0) \exp\left((\mu - \frac{\sigma^2}{2})t + \sigma W(t) + \sum_{i=1}^{N(t)} J_i\right)$$

### Jump Size Distribution

**Normal jumps**: $J \sim \mathcal{N}(\mu_J, \sigma_J^2)$

**Log-normal**: $e^J$ is log-normal.

**Double-exponential**: Kou's model uses asymmetric exponential distribution.

### Option Pricing

**Risk-neutral measure**: Under $Q$:

$$dS(t) = rS(t-) dt + \sigma S(t-) d\tilde{W}(t) + S(t-)(e^J - 1) d\tilde{N}(t)$$

where jump intensity and distribution may change.

**Characteristic function**: Available in closed form for normal jumps.

**Fourier methods**: Use Carr-Madan method to price options.

**Merton's formula**: Series expansion:

$$C = \sum_{n=0}^{\infty} \frac{e^{-\lambda'T}(\lambda'T)^n}{n!} C_{\text{BS}}(S, K, T, r, \sigma_n, q_n)$$

where $\lambda'$ is risk-neutral intensity and $\sigma_n$, $q_n$ adjusted for $n$ jumps.

## Kou's Double-Exponential Jump Model

### Jump Size Distribution

**PDF**: 

$$f_J(x) = p\lambda_+ e^{-\lambda_+ x}\mathbf{1}_{x \geq 0} + (1-p)\lambda_- e^{\lambda_- |x|}\mathbf{1}_{x < 0}$$

where:
- $p$: Probability of positive jump
- $\lambda_+$: Rate of positive jumps
- $\lambda_-$: Rate of negative jumps

**Properties**:
- Asymmetric (captures negative skewness)
- Exponential tails
- Finite moments

### Stock Price

**SDE**: Same as Merton but with double-exponential jumps.

**Advantages**:
- Captures volatility smile better
- Asymmetric distribution matches market
- Analytical option pricing formulas available

## Lévy Processes

### Definition

A Lévy process $\{X(t) : t \geq 0\}$ satisfies:

1. **$X(0) = 0$**
2. **Independent increments**: $X(t) - X(s)$ independent of $\{X(u) : u \leq s\}$
3. **Stationary increments**: $X(t) - X(s) \sim X(t-s)$
4. **Stochastic continuity**: $\lim_{h \to 0} \mathbb{P}(|X(t+h) - X(t)| > \epsilon) = 0$

**Examples**:
- Brownian motion: $X(t) = \mu t + \sigma W(t)$
- Poisson process: $X(t) = N(t)$
- Compound Poisson: $X(t) = \sum_{i=1}^{N(t)} Y_i$
- Jump-diffusion: Combination of Brownian motion and compound Poisson

### Lévy-Khintchine Representation

**Characteristic function**: 

$$\phi_X(u, t) = E[e^{iuX(t)}] = \exp(t\psi(u))$$

where Lévy exponent:

$$\psi(u) = i\mu u - \frac{\sigma^2 u^2}{2} + \int_{\mathbb{R}} (e^{iux} - 1 - iux\mathbf{1}_{|x| \leq 1}) \nu(dx)$$

**Components**:
- $\mu$: Drift
- $\sigma^2$: Brownian motion variance
- $\nu$: Lévy measure (jump intensity and size distribution)

**Lévy measure**: $\nu(A)$ gives expected number of jumps with size in $A$ per unit time.

### Properties

**Infinite divisibility**: For any $n$, $X(t) = \sum_{i=1}^{n} X_i$ where $X_i$ are i.i.d.

**Self-similarity**: Some Lévy processes are self-similar (e.g., stable processes).

**Path properties**: 
- Continuous: Brownian motion only
- Finite activity: Compound Poisson (finite jumps)
- Infinite activity: Infinite jumps in finite time (e.g., variance gamma)

## Variance Gamma Process

### Definition

**Variance Gamma (VG)**: Time-changed Brownian motion:

$$X(t) = \theta G(t) + \sigma W(G(t))$$

where $G(t) \sim \text{Gamma}(t/\nu, \nu)$ is gamma subordinator.

**Subordinator**: Increasing Lévy process (time change).

**Properties**:
- Infinite activity (infinitely many small jumps)
- Finite variation
- Three parameters: $\sigma$ (volatility), $\theta$ (skewness), $\nu$ (kurtosis)

### Characteristic Function

**VG characteristic function**:

$$\phi_{VG}(u, t) = \left(1 - i\theta\nu u + \frac{\sigma^2\nu u^2}{2}\right)^{-t/\nu}$$

**Moments**:
- $E[X(t)] = \theta t$
- $\text{Var}(X(t)) = (\sigma^2 + \theta^2\nu)t$
- Skewness: $\frac{\theta\nu(3\sigma^2 + 2\theta^2\nu)}{(\sigma^2 + \theta^2\nu)^{3/2}\sqrt{t}}$
- Excess kurtosis: $\frac{3\nu(\sigma^4 + 4\theta^2\sigma^2\nu + 2\theta^4\nu^2)}{(\sigma^2 + \theta^2\nu)^2 t}$

### Applications

**Stock returns**: Model log returns as VG process.

**Option pricing**: Characteristic function available for Fourier methods.

**Calibration**: Fit to market option prices.

## Itô's Lemma for Jump-Diffusion

### General Form

For process:

$$dX(t) = \mu(t, X(t-)) dt + \sigma(t, X(t-)) dW(t) + \gamma(t, X(t-), J) dN(t)$$

and function $f(t, x)$:

$$df(t, X(t)) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX^c(t) + \frac{1}{2}\frac{\partial^2 f}{\partial x^2} d\langle X^c \rangle_t + [f(t, X(t)) - f(t, X(t-))]$$

where $X^c(t)$ is continuous part.

**Decomposition**: 

$$df = df^{\text{continuous}} + df^{\text{jump}}$$

**Continuous part**: Standard Itô's lemma.

**Jump part**: $f(t, X(t)) - f(t, X(t-))$ (difference at jump times).

### Example: Geometric Jump-Diffusion

**SDE**: $dS(t) = \mu S(t-) dt + \sigma S(t-) dW(t) + S(t-)(e^J - 1) dN(t)$

**Apply Itô's lemma** to $f(x) = \ln x$:

$$d\ln S(t) = \frac{1}{S(t-)} dS^c(t) - \frac{1}{2S(t-)^2} d\langle S^c \rangle_t + [\ln S(t) - \ln S(t-)]$$

$$= (\mu - \frac{\sigma^2}{2}) dt + \sigma dW(t) + J dN(t)$$

**Solution**: 

$$S(t) = S(0) \exp\left((\mu - \frac{\sigma^2}{2})t + \sigma W(t) + \sum_{i=1}^{N(t)} J_i\right)$$

## Option Pricing with Jumps

### Risk-Neutral Measure

**Change of measure**: Under risk-neutral measure $Q$:

$$dS(t) = rS(t-) dt + \sigma S(t-) d\tilde{W}(t) + S(t-)(e^J - 1) d\tilde{N}(t)$$

**Jump intensity**: May change: $\lambda^Q \neq \lambda^P$

**Jump distribution**: May change: $J^Q \neq J^P$ (risk premium for jump risk).

### Characteristic Function Method

**Stock price**: $S(T) = S(0) \exp(X(T))$ where $X(T)$ is Lévy process.

**Call option**: 

$$C = e^{-rT} E^Q[\max(S(T) - K, 0)]$$

**Fourier inversion**: 

$$C = e^{-rT} \int_K^{\infty} (S - K) f^Q(S) dS$$

**Characteristic function**: 

$$\phi_X(u) = E^Q[e^{iuX(T)}]$$

**Carr-Madan method**: Express option price as Fourier integral:

$$C = \frac{e^{-\alpha \ln K}}{\pi} \int_0^{\infty} e^{-iv\ln K} \frac{\phi_X(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v} dv$$

where $\alpha > 0$ is damping parameter.

### Merton's Series Solution

**Expansion**: For Merton model with normal jumps:

$$C = \sum_{n=0}^{\infty} \frac{e^{-\lambda'T}(\lambda'T)^n}{n!} C_{\text{BS}}\left(S_n, K, T, r_n, \sigma_n\right)$$

where:
- $S_n = S(0) e^{n\mu_J + n\sigma_J^2/2}$ (adjusted stock price)
- $\sigma_n^2 = \sigma^2 + n\sigma_J^2/T$ (adjusted volatility)
- $r_n = r - \lambda'(e^{\mu_J + \sigma_J^2/2} - 1) + n(\mu_J + \sigma_J^2/2)/T$ (adjusted rate)

**Interpretation**: Weighted average of Black-Scholes prices conditional on $n$ jumps.

## Applications in Finance

### Volatility Smile

**Problem**: Black-Scholes assumes constant volatility, but market shows volatility smile.

**Solution**: Jump-diffusion models generate volatility smile:
- Negative jumps: Downward skew
- Positive jumps: Upward skew
- Jump risk: Higher implied volatility for out-of-money options

**Calibration**: Fit jump parameters to match market volatility smile.

### Crash Risk

**Rare events**: Large negative jumps model market crashes.

**Tail risk**: Jump models capture tail risk better than diffusion-only models.

**Risk management**: VaR and stress testing using jump models.

### Credit Risk

**Default jumps**: Model default as jump to zero.

**Reduced-form models**: Default intensity with jumps:

$$d\lambda(t) = \kappa(\theta - \lambda(t)) dt + \sigma\sqrt{\lambda(t)} dW(t) + J dN(t)$$

**Jump-to-default**: $J$ represents sudden increase in default intensity.

### High-Frequency Trading

**Microstructure**: Small jumps capture microstructure effects.

**Tick data**: Model price changes as compound Poisson.

**Optimal execution**: Account for jump risk in execution strategies.

### Energy Markets

**Price spikes**: Electricity and gas prices exhibit jumps.

**Mean-reverting jump-diffusion**: 

$$dS(t) = \kappa(\theta - S(t)) dt + \sigma dW(t) + J dN(t)$$

**Spike modeling**: Large positive jumps model price spikes.

### Insurance

**Claim arrivals**: Poisson process models claim arrivals.

**Claim sizes**: Compound Poisson models aggregate claims.

**Ruin theory**: Study probability of ruin using Lévy processes.

### Exotic Options

**Barrier options**: Jumps can cause barrier to be crossed between monitoring dates.

**Lookback options**: Maximum/minimum affected by jumps.

**Asian options**: Average price includes jump contributions.

**Pricing**: Use Monte Carlo or Fourier methods.
