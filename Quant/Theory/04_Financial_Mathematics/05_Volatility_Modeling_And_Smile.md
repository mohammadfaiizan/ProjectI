# Volatility Modeling and Smile

## Implied Volatility

Implied volatility is the volatility parameter that, when inserted into the Black-Scholes formula, reproduces the market price of an option.

### Definition

For a market-observed call price $C_{market}$, implied volatility $\sigma_{imp}$ satisfies:

$$C_{market} = C_{BS}(S, K, T, r, \sigma_{imp})$$

where $C_{BS}$ is the Black-Scholes formula.

### Properties

- **Uniqueness:** Black-Scholes is monotonic in volatility, so $\sigma_{imp}$ is unique
- **Moneyness dependence:** Implied volatility varies with strike $K$ and maturity $T$
- **Volatility surface:** $\sigma_{imp}(K, T)$ forms a two-dimensional surface

### Inversion

Implied volatility is found by solving:
$$f(\sigma) = C_{BS}(\sigma) - C_{market} = 0$$

Using Newton-Raphson or bisection methods. The vega provides the derivative:
$$\frac{\partial f}{\partial \sigma} = \nu = S n(d_1)\sqrt{T}$$

## Volatility Surface

The volatility surface $\sigma_{imp}(K, T)$ shows how implied volatility varies across strikes and maturities.

### Moneyness Measures

Common measures:
- **Strike:** $K$
- **Moneyness:** $K/S$ or $S/K$
- **Log-moneyness:** $\ln(K/S)$
- **Delta:** Convert strike to delta for normalization

### Volatility Smile

The volatility smile (or skew) is the pattern of implied volatility across strikes for a given maturity.

**Smile:** U-shaped curve, higher vol for OTM calls and puts
**Skew:** Downward sloping, higher vol for OTM puts (lower strikes)
**Smirk:** Asymmetric smile, common in equity markets

### Term Structure

The term structure shows how implied volatility varies with maturity:
- **Upward sloping:** Longer-dated options have higher vol
- **Downward sloping:** Shorter-dated options have higher vol
- **Hump:** Volatility peaks at intermediate maturities

## Empirical Patterns

### Equity Markets

- **Volatility skew:** OTM puts trade at higher implied vol than OTM calls
- **Explanation:** Demand for downside protection, crash fears
- **Term structure:** Often upward sloping, reflecting uncertainty

### FX Markets

- **Volatility smile:** More symmetric than equities
- **Explanation:** Two currencies, no clear direction bias
- **Term structure:** Relatively flat

### Interest Rate Markets

- **Volatility skew:** Varies with market conditions
- **Swaption vol:** Different patterns for different tenors

## Explanations for Volatility Smile

### Stylized Explanations

1. **Leverage effect:** As stock falls, leverage increases, volatility rises
2. **Volatility clustering:** High volatility periods cluster
3. **Jump risk:** Sudden large moves not captured by Black-Scholes
4. **Stochastic volatility:** Volatility itself is random
5. **Supply and demand:** Imbalances in option supply/demand

### Model Implications

Different models produce different smile shapes:
- **Constant vol (BS):** Flat smile (inconsistent with markets)
- **Stochastic vol:** Can produce smiles
- **Jump-diffusion:** Can produce smiles and skews
- **Local volatility:** Can fit any smile exactly

## Local Volatility Model

The local volatility model assumes volatility is a deterministic function of time and stock price: $\sigma(S,t)$.

### Dupire's Equation

Dupire showed that if call prices $C(K,T)$ are known for all strikes and maturities, the local volatility function is:

$$\sigma_{loc}^2(K,T) = \frac{\frac{\partial C}{\partial T} + rK\frac{\partial C}{\partial K}}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}$$

This is derived from the forward PDE for call prices.

### Implementation

1. **Interpolate/extrapolate** market prices to create a smooth surface $C(K,T)$
2. **Compute derivatives** numerically
3. **Apply Dupire's formula** to get $\sigma_{loc}(S,t)$

### Properties

- **Fits market exactly:** Can reproduce all market prices
- **Forward-looking:** Uses today's prices to infer future vol
- **No-arbitrage:** Consistent with market prices
- **Limitations:** Assumes deterministic volatility, may not be realistic

### Forward PDE

The call price satisfies:
$$\frac{\partial C}{\partial T} = \frac{1}{2}\sigma_{loc}^2(K,T)K^2\frac{\partial^2 C}{\partial K^2} - rK\frac{\partial C}{\partial K}$$

This forward PDE allows pricing with local volatility.

## Stochastic Volatility Models

Stochastic volatility models assume volatility itself is random.

### Heston Model

The Heston model specifies:

$$dS(t) = rS(t)dt + \sqrt{V(t)}S(t)dW_1(t)$$

$$dV(t) = \kappa(\theta - V(t))dt + \sigma_v\sqrt{V(t)}dW_2(t)$$

where:
- $V(t)$ is the variance (volatility squared)
- $\kappa$: speed of mean reversion
- $\theta$: long-term variance
- $\sigma_v$: volatility of volatility
- $dW_1 dW_2 = \rho dt$: correlation

**Properties:**
- Mean-reverting variance
- Closed-form option prices via characteristic functions
- Can produce volatility smiles

### Characteristic Function

The Heston model has a known characteristic function:

$$\phi(u) = \mathbb{E}[e^{iu\ln S(T)}] = e^{C(u,T) + D(u,T)V(0) + iu\ln S(0)}$$

where $C$ and $D$ are known functions. Option prices are computed via Fourier inversion (Carr-Madan method).

### SABR Model

The Stochastic Alpha Beta Rho (SABR) model is popular for interest rate options:

$$dF(t) = \sigma(t)F(t)^\beta dW_1(t)$$

$$d\sigma(t) = \alpha\sigma(t)dW_2(t)$$

where:
- $F(t)$: forward rate
- $\beta$: skew parameter ($0 \leq \beta \leq 1$)
- $\alpha$: volatility of volatility
- $dW_1 dW_2 = \rho dt$

**Approximation:** Hagan et al. derived an approximate implied volatility formula, making SABR easy to use.

### Volatility of Volatility

The vol-of-vol parameter controls smile curvature:
- **High vol-of-vol:** Steeper smile
- **Low vol-of-vol:** Flatter smile

## Rough Volatility

Rough volatility models use fractional Brownian motion to capture long memory in volatility.

### Fractional Brownian Motion

Fractional Brownian motion $B^H(t)$ has:
$$\mathbb{E}[(B^H(t) - B^H(s))^2] = |t-s|^{2H}$$

where $H$ is the Hurst exponent. For $H < 1/2$, the process is rough (anti-persistent).

### Rough Heston

The rough Heston model uses:

$$V(t) = V(0) + \frac{1}{\Gamma(\alpha)}\int_0^t (t-s)^{\alpha-1}\kappa(\theta - V(s))ds + \frac{1}{\Gamma(\alpha)}\int_0^t (t-s)^{\alpha-1}\sigma_v\sqrt{V(s)}dW(s)$$

where $\alpha = H + 1/2$ and $H < 1/2$ gives rough paths.

**Properties:**
- Better fit to short-dated option prices
- Captures volatility clustering
- More complex pricing (requires numerical methods)

## Volatility Trading

### Variance Swaps

A variance swap pays:
$$\text{Payoff} = N_{var}(\sigma_{realized}^2 - \sigma_{strike}^2)$$

where:
$$\sigma_{realized}^2 = \frac{252}{T}\sum_{i=1}^{n}\left(\ln\frac{S_i}{S_{i-1}}\right)^2$$

**Replication:** Under certain conditions, a variance swap can be replicated using a portfolio of options:

$$V_{var} \approx \frac{2}{T}\int_0^{S_0}\frac{P(K)}{K^2}dK + \frac{2}{T}\int_{S_0}^{\infty}\frac{C(K)}{K^2}dK$$

This is the **variance swap replication formula**.

### VIX Index

The VIX is the implied volatility of 30-day S&P 500 options, computed using:

$$\text{VIX}^2 = \frac{2}{T}\sum_i\frac{\Delta K_i}{K_i^2}e^{rT}Q(K_i) - \frac{1}{T}\left(\frac{F}{K_0} - 1\right)^2$$

where $Q(K_i)$ are mid-quotes for out-of-the-money options.

### Volatility of Volatility

Vol-of-vol measures how much implied volatility itself moves. It's a key parameter in stochastic volatility models and affects:
- Smile curvature
- Vega hedging
- Volatility trading strategies

## Calibration Methods

### Objective Function

Calibrate model parameters $\theta$ to minimize:

$$\min_{\theta} \sum_{i=1}^{n}w_i(C_{model}(K_i, T_i; \theta) - C_{market}(K_i, T_i))^2$$

where $w_i$ are weights (often inverse of bid-ask spread).

### Challenges

1. **Ill-posedness:** Multiple parameter sets may fit equally well
2. **Stability:** Parameters may change dramatically with small price changes
3. **Overfitting:** Model fits noise, not signal
4. **Computational cost:** Pricing many options is expensive

### Regularization

Add penalty terms:
$$\min_{\theta} \sum_{i}w_i(C_{model} - C_{market})^2 + \lambda R(\theta)$$

where $R(\theta)$ penalizes large or unstable parameters.

### Methods

1. **Local optimization:** Gradient descent, Levenberg-Marquardt
2. **Global optimization:** Simulated annealing, genetic algorithms
3. **Bayesian:** Prior distributions on parameters
4. **Machine learning:** Neural networks to learn volatility surfaces

## Model Comparison

### Local Volatility

**Pros:**
- Fits market exactly
- Fast pricing (PDE)
- No-arbitrage

**Cons:**
- Deterministic vol (unrealistic)
- Wrong dynamics for hedging
- Forward smile may be unrealistic

### Stochastic Volatility (Heston)

**Pros:**
- Realistic dynamics
- Closed-form prices
- Can produce smiles

**Cons:**
- May not fit market exactly
- Parameters may be unstable
- Limited flexibility

### Rough Volatility

**Pros:**
- Better fit to short-dated options
- Captures long memory
- More realistic

**Cons:**
- Complex pricing
- Newer, less tested
- Computational challenges

## Practical Considerations

### Interpolation and Extrapolation

Market data is sparse. Need to:
- **Interpolate** between strikes and maturities
- **Extrapolate** beyond market data
- Ensure **no-arbitrage** conditions

### Time Decay

Implied volatility surfaces change over time:
- **Sticky strike:** Vol stays constant for fixed strike
- **Sticky delta:** Vol stays constant for fixed delta
- **Sticky moneyness:** Vol stays constant for fixed $K/S$

### Model Risk

Different models give different prices and Greeks:
- **Hedging:** Model choice affects hedge ratios
- **P&L attribution:** Model affects risk decomposition
- **Capital:** Model affects risk capital calculations
