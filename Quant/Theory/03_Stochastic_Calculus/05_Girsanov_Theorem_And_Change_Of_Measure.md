# Girsanov Theorem and Change of Measure

## Radon-Nikodym Derivative

### Definition

For probability measures $P$ and $Q$ on $(\Omega, \mathcal{F})$, if $Q \ll P$ (absolute continuity), there exists Radon-Nikodym derivative:

$$\frac{dQ}{dP}(\omega) = \lim_{n \to \infty} \frac{Q(A_n)}{P(A_n)}$$

for appropriate sequence of sets $A_n \to \{\omega\}$.

**Properties**:
- $E_Q[X] = E_P\left[X \frac{dQ}{dP}\right]$
- $\frac{dQ}{dP} \geq 0$ almost surely
- $E_P\left[\frac{dQ}{dP}\right] = 1$

### Conditional Expectation

**Bayes' rule**: For sub-sigma-algebra $\mathcal{G}$:

$$E_Q[X|\mathcal{G}] = \frac{E_P[X \frac{dQ}{dP}|\mathcal{G}]}{E_P[\frac{dQ}{dP}|\mathcal{G}]}$$

**Process version**: For filtration $\{\mathcal{F}_t\}$:

$$E_Q[X|\mathcal{F}_t] = \frac{E_P[X \frac{dQ}{dP}|\mathcal{F}_t]}{E_P[\frac{dQ}{dP}|\mathcal{F}_t]} = \frac{E_P[X Z_T|\mathcal{F}_t]}{Z_t}$$

where $Z_t = E_P[\frac{dQ}{dP}|\mathcal{F}_t]$ is density process.

## Equivalent Measures

### Definition

Measures $P$ and $Q$ are equivalent (denoted $P \sim Q$) if:

$$P \ll Q \quad \text{and} \quad Q \ll P$$

**Equivalently**: $P$ and $Q$ have same null sets.

**Radon-Nikodym derivative**: $\frac{dQ}{dP} > 0$ almost surely.

### Financial Interpretation

**Physical measure $P$**: Actual probability distribution of market.

**Risk-neutral measure $Q$**: Equivalent measure under which discounted prices are martingales.

**No arbitrage**: Equivalent to existence of equivalent martingale measure.

## Girsanov Theorem

### Statement

**Girsanov theorem**: Let $\{W(t)\}$ be Brownian motion under $P$ and $\{\theta(t)\}$ be adapted process with:

$$E_P\left[\exp\left(\frac{1}{2}\int_0^T \theta(t)^2 dt\right)\right] < \infty$$

Define:

$$Z(t) = \exp\left(-\int_0^t \theta(s) dW(s) - \frac{1}{2}\int_0^t \theta(s)^2 ds\right)$$

and measure $Q$ by:

$$\frac{dQ}{dP}\Big|_{\mathcal{F}_t} = Z(t)$$

Then:

$$\tilde{W}(t) = W(t) + \int_0^t \theta(s) ds$$

is Brownian motion under $Q$.

### Novikov's Condition

**Sufficient condition**: If:

$$E_P\left[\exp\left(\frac{1}{2}\int_0^T \theta(t)^2 dt\right)\right] < \infty$$

then $Z(t)$ is martingale (Novikov's condition).

**Weaker condition**: Kazamaki's condition (weaker than Novikov).

### Proof Outline

**Step 1**: Show $Z(t)$ is martingale under $P$.

**Step 2**: Define $Q$ by $dQ = Z(T) dP$.

**Step 3**: Show $\tilde{W}(t)$ has independent increments under $Q$.

**Step 4**: Show increments are Gaussian with correct variance.

**Step 5**: Use Lévy's characterization: continuous martingale with quadratic variation $t$ is Brownian motion.

### Multidimensional Case

For $d$-dimensional Brownian motion $\mathbf{W}(t)$ and vector process $\boldsymbol{\theta}(t)$:

$$Z(t) = \exp\left(-\int_0^t \boldsymbol{\theta}(s)^T d\mathbf{W}(s) - \frac{1}{2}\int_0^t \|\boldsymbol{\theta}(s)\|^2 ds\right)$$

Then:

$$\tilde{\mathbf{W}}(t) = \mathbf{W}(t) + \int_0^t \boldsymbol{\theta}(s) ds$$

is $d$-dimensional Brownian motion under $Q$.

## Risk-Neutral Pricing Framework

### From Physical to Risk-Neutral Measure

**Stock price under $P$**: 

$$dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$$

**Market price of risk**: 

$$\theta(t) = \frac{\mu - r}{\sigma}$$

**Girsanov transformation**: 

$$\tilde{W}(t) = W(t) + \int_0^t \frac{\mu - r}{\sigma} ds = W(t) + \frac{\mu - r}{\sigma}t$$

**Stock price under $Q$**: 

$$dS(t) = rS(t) dt + \sigma S(t) d\tilde{W}(t)$$

**Drift change**: Expected return changes from $\mu$ to $r$ (risk-free rate).

### Martingale Property

**Discounted stock price**: Under $Q$:

$$d(e^{-rt}S(t)) = e^{-rt}S(t)\sigma d\tilde{W}(t)$$

**Martingale**: $e^{-rt}S(t)$ is $Q$-martingale:

$$e^{-rt}S(t) = E^Q[e^{-rT}S(T)|\mathcal{F}_t]$$

**Pricing formula**: 

$$S(t) = e^{-r(T-t)} E^Q[S(T)|\mathcal{F}_t]$$

### Option Pricing

**Call option**: 

$$C(t, S(t)) = e^{-r(T-t)} E^Q[\max(S(T) - K, 0)|\mathcal{F}_t]$$

**Under $Q$**: $S(T) = S(t) \exp((r - \sigma^2/2)(T-t) + \sigma(\tilde{W}(T) - \tilde{W}(t)))$

**Black-Scholes formula**: 

$$C(t, S) = S\Phi(d_1) - Ke^{-r(T-t)}\Phi(d_2)$$

where:

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$$

$$d_2 = d_1 - \sigma\sqrt{T-t}$$

## Cameron-Martin Theorem

### Statement

**Cameron-Martin theorem**: For deterministic function $h(t)$ with $\int_0^T h(t)^2 dt < \infty$, the process:

$$\tilde{W}(t) = W(t) + \int_0^t h(s) ds$$

has same law as $W(t)$ under measure $Q$ defined by:

$$\frac{dQ}{dP} = \exp\left(-\int_0^T h(s) dW(s) - \frac{1}{2}\int_0^T h(s)^2 ds\right)$$

**Special case**: Girsanov with deterministic $\theta(t) = h(t)$.

### Applications

**Drift change**: Change drift of Brownian motion by deterministic function.

**Likelihood ratio**: Radon-Nikodym derivative gives likelihood ratio for hypothesis testing.

## Numeraire Change

### General Framework

**Numeraire**: Positive price process $N(t)$ used as unit of account.

**Change of numeraire**: Switch from numeraire $N_1(t)$ to $N_2(t)$.

**New measure**: 

$$\frac{dQ_2}{dQ_1}\Big|_{\mathcal{F}_t} = \frac{N_1(0)}{N_2(0)} \frac{N_2(t)}{N_1(t)}$$

**Martingale property**: Under $Q_2$, $\frac{S(t)}{N_2(t)}$ is martingale.

### T-Forward Measure

**Definition**: $Q^T$ is $T$-forward measure with numeraire $P(t, T)$ (zero-coupon bond).

**Density**: 

$$\frac{dQ^T}{dQ}\Big|_{\mathcal{F}_t} = \frac{P(t, T)}{P(0, T) B(t)}$$

where $B(t) = e^{rt}$ is bank account.

**Forward price**: 

$$F(t, T) = \frac{S(t)}{P(t, T)} = E^{Q^T}[S(T)|\mathcal{F}_t]$$

**Martingale**: Forward price is martingale under $T$-forward measure.

### Applications

**Bond options**: Price using $T$-forward measure:

$$C(t) = P(t, T) E^{Q^T}[\max(P(T, S) - K, 0)|\mathcal{F}_t]$$

**Swaptions**: Use swap measure (annuity as numeraire).

**Caps/Floors**: Use forward measure for each caplet.

## Applications

### Black-Scholes Derivation via Measure Change

**Physical measure $P$**: 

$$dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$$

**Market price of risk**: $\theta = \frac{\mu - r}{\sigma}$

**Girsanov**: 

$$\frac{dQ}{dP} = \exp\left(-\frac{\mu - r}{\sigma} W(T) - \frac{1}{2}\left(\frac{\mu - r}{\sigma}\right)^2 T\right)$$

**Under $Q$**: 

$$dS(t) = rS(t) dt + \sigma S(t) d\tilde{W}(t)$$

**Option price**: 

$$C(0) = e^{-rT} E^Q[\max(S(T) - K, 0)]$$

**Evaluation**: Use log-normal distribution of $S(T)$ under $Q$.

### Foreign Exchange Options

**Domestic/foreign rates**: $r_d$, $r_f$

**FX rate**: $X(t)$ (units of domestic currency per unit of foreign)

**Under domestic risk-neutral measure $Q_d$**: 

$$dX(t) = (r_d - r_f)X(t) dt + \sigma X(t) dW_d(t)$$

**Under foreign risk-neutral measure $Q_f$**: 

$$dX(t) = (r_d - r_f)X(t) dt + \sigma X(t) dW_f(t)$$

**Change of measure**: 

$$\frac{dQ_f}{dQ_d} = \exp\left(-\int_0^T \sigma dW_d(t) - \frac{1}{2}\sigma^2 T\right)$$

**Quanto options**: Options with payoff in different currency.

### Stochastic Interest Rates

**Hull-White model**: 

$$dr(t) = (\theta(t) - \kappa r(t)) dt + \sigma dW(t)$$

**Change to $T$-forward measure**: 

$$\frac{dQ^T}{dQ} = \frac{P(t, T)}{P(0, T) e^{rt}}$$

**Drift change**: 

$$dr(t) = (\theta(t) - \kappa r(t) - \sigma^2 B(t, T)) dt + \sigma d\tilde{W}(t)$$

where $B(t, T) = \frac{1 - e^{-\kappa(T-t)}}{\kappa}$.

**Bond options**: Price using $T$-forward measure.

### Incomplete Markets

**Multiple risk-neutral measures**: If market incomplete, multiple equivalent martingale measures exist.

**Minimal martingale measure**: Choose measure closest to physical measure.

**Variance-optimal measure**: Minimize variance of Radon-Nikodym derivative.

**Utility indifference**: Choose measure based on utility maximization.

### Calibration

**Market prices**: Observed option prices determine risk-neutral measure.

**Implied volatility**: Extract volatility from market prices.

**Density**: Risk-neutral density $f^Q(S_T)$ from option prices:

$$f^Q(S_T) = e^{rT} \frac{\partial^2 C}{\partial K^2}\Big|_{K = S_T}$$

**Change of measure**: Convert between physical and risk-neutral densities.

### Credit Risk

**Default intensity**: Under physical measure $P$:

$$d\lambda(t) = \kappa(\theta - \lambda(t)) dt + \sigma\sqrt{\lambda(t)} dW(t)$$

**Risk-neutral intensity**: Under risk-neutral measure $Q$:

$$d\lambda(t) = \kappa^Q(\theta^Q - \lambda(t)) dt + \sigma\sqrt{\lambda(t)} d\tilde{W}(t)$$

**Market price of risk**: $\theta^Q \neq \theta$ (risk premium for default risk).

**Credit spreads**: Reflect both physical default probability and risk premium.

### Term Structure Models

**Heath-Jarrow-Morton**: Forward rates under risk-neutral measure:

$$df(t, T) = \alpha(t, T) dt + \sigma(t, T) dW(t)$$

**No arbitrage**: Drift determined by volatility:

$$\alpha(t, T) = \sigma(t, T) \int_t^T \sigma(t, s) ds$$

**Change of measure**: Can change to forward measure for pricing.
