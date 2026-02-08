# Greeks and Hedging

## Introduction to Greeks

Greeks measure the sensitivity of option prices to various parameters. They are essential for risk management and hedging strategies. The main Greeks are partial derivatives of the option price with respect to underlying variables.

## Delta

Delta ($\Delta$) measures the sensitivity of the option price to changes in the underlying asset price:

$$\Delta = \frac{\partial V}{\partial S}$$

### Delta for European Options

For a European call option under Black-Scholes:
$$\Delta_C = N(d_1)$$

where $d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$.

For a European put option:
$$\Delta_P = N(d_1) - 1 = -N(-d_1)$$

### Delta Properties

- **Call delta:** $0 \leq \Delta_C \leq 1$
  - Deep out-of-the-money: $\Delta \approx 0$
  - At-the-money: $\Delta \approx 0.5$
  - Deep in-the-money: $\Delta \approx 1$

- **Put delta:** $-1 \leq \Delta_P \leq 0$
  - Deep out-of-the-money: $\Delta \approx 0$
  - At-the-money: $\Delta \approx -0.5$
  - Deep in-the-money: $\Delta \approx -1$

- **Delta as probability:** Under the risk-neutral measure, $\Delta_C$ approximates the probability of finishing in-the-money.

### Delta Hedging

A delta-neutral portfolio has zero net delta:
$$\Delta_{portfolio} = \sum_i n_i \Delta_i = 0$$

where $n_i$ is the quantity of option $i$ and $\Delta_i$ is its delta.

To hedge a long call position:
- Sell $\Delta$ shares of stock (or buy puts with delta $-\Delta$)

The hedged portfolio value change is approximately:
$$\Delta V \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma (\Delta S)^2 + \Theta \Delta t + \cdots$$

## Gamma

Gamma ($\Gamma$) measures the rate of change of delta with respect to the underlying price:

$$\Gamma = \frac{\partial \Delta}{\partial S} = \frac{\partial^2 V}{\partial S^2}$$

### Gamma Formula

For European options under Black-Scholes:
$$\Gamma = \frac{n(d_1)}{S\sigma\sqrt{T-t}}$$

where $n(d_1) = \frac{1}{\sqrt{2\pi}}e^{-d_1^2/2}$ is the standard normal probability density function.

Note: Gamma is the same for calls and puts.

### Gamma Properties

- **Maximum gamma:** Occurs when the option is at-the-money
- **Time decay:** Gamma increases as expiration approaches (for ATM options)
- **Volatility effect:** Lower volatility increases gamma (steeper delta curve)
- **Sign:** $\Gamma > 0$ for long options (both calls and puts)

### Gamma Scalping

Gamma scalping exploits the convexity of option positions. A long gamma position profits from large moves in either direction.

**Strategy:**
1. Start delta-neutral with long gamma
2. If stock moves up, delta becomes positive → sell shares
3. If stock moves down, delta becomes negative → buy shares
4. Profit from buying low and selling high

The P&L from gamma scalping over a small move $\Delta S$ is approximately:
$$P\&L \approx \frac{1}{2}\Gamma (\Delta S)^2$$

This requires frequent rebalancing, making it costly in the presence of transaction costs.

## Theta

Theta ($\Theta$) measures the time decay of the option price:

$$\Theta = -\frac{\partial V}{\partial t}$$

The negative sign ensures theta is typically negative (options lose value over time).

### Theta Formula

For a European call:
$$\Theta_C = -\frac{S n(d_1) \sigma}{2\sqrt{T-t}} - rKe^{-r(T-t)}N(d_2)$$

For a European put:
$$\Theta_P = -\frac{S n(d_1) \sigma}{2\sqrt{T-t}} + rKe^{-r(T-t)}N(-d_2)$$

### Theta Properties

- **Always negative for long options** (time decay)
- **Maximum decay:** For at-the-money options near expiration
- **Time effect:** Theta increases (becomes more negative) as expiration approaches
- **Interest rate effect:** Higher $r$ increases call theta (more negative), decreases put theta

### Theta Trading

Theta traders sell options to collect premium, betting that time decay will exceed adverse price moves. This requires careful delta hedging and gamma management.

## Vega

Vega ($\nu$) measures sensitivity to volatility:

$$\nu = \frac{\partial V}{\partial \sigma}$$

Note: Vega is not a Greek letter; it's denoted $\nu$ or $\mathcal{V}$.

### Vega Formula

For European options:
$$\nu = S n(d_1) \sqrt{T-t}$$

Vega is the same for calls and puts.

### Vega Properties

- **Always positive** for long options
- **Maximum vega:** For at-the-money options with moderate time to expiration
- **Time decay:** Vega decreases as expiration approaches
- **Volatility smile:** Implied volatility varies with strike, complicating vega hedging

### Vega Hedging

To hedge vega risk:
- Use other options (not the underlying)
- Create a vega-neutral portfolio: $\sum_i n_i \nu_i = 0$

Vega hedging is crucial for volatility trading strategies.

## Rho

Rho ($\rho$) measures sensitivity to the risk-free interest rate:

$$\rho = \frac{\partial V}{\partial r}$$

### Rho Formula

For a European call:
$$\rho_C = K(T-t)e^{-r(T-t)}N(d_2)$$

For a European put:
$$\rho_P = -K(T-t)e^{-r(T-t)}N(-d_2)$$

### Rho Properties

- **Call rho:** Positive (higher rates increase call value)
- **Put rho:** Negative (higher rates decrease put value)
- **Time effect:** Rho increases with time to expiration
- **Magnitude:** Typically small relative to other Greeks for short-dated options

## Higher-Order Greeks

### Vanna

Vanna measures the sensitivity of delta to volatility changes:

$$\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial \nu}{\partial S}$$

For European options:
$$\text{Vanna} = -n(d_1)\frac{d_2}{\sigma}$$

Vanna is important when hedging delta in the presence of volatility skew.

### Volga (Vomma)

Volga measures the sensitivity of vega to volatility changes:

$$\text{Volga} = \frac{\partial \nu}{\partial \sigma} = \frac{\partial^2 V}{\partial \sigma^2}$$

For European options:
$$\text{Volga} = S n(d_1) \sqrt{T-t} \frac{d_1 d_2}{\sigma}$$

Volga is important for vega hedging when volatility is stochastic.

### Charm

Charm measures the sensitivity of delta to time:

$$\text{Charm} = \frac{\partial \Delta}{\partial t}$$

For a European call:
$$\text{Charm} = -n(d_1)\left(\frac{r}{\sigma\sqrt{T-t}} - \frac{d_2}{2(T-t)}\right)$$

Charm is useful for understanding how delta changes over time, important for delta hedging strategies.

### Speed

Speed measures the rate of change of gamma:

$$\text{Speed} = \frac{\partial \Gamma}{\partial S} = \frac{\partial^3 V}{\partial S^3}$$

For European options:
$$\text{Speed} = -\frac{\Gamma}{S}\left(1 + \frac{d_1}{\sigma\sqrt{T-t}}\right)$$

Speed helps anticipate gamma changes and is important for gamma scalping strategies.

## Taylor Expansion of Option Portfolio

The value change of an option portfolio can be approximated using a Taylor expansion:

$$\Delta V \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma (\Delta S)^2 + \Theta \Delta t + \nu \Delta \sigma + \rho \Delta r + \cdots$$

Higher-order terms include:
- **Cross terms:** $\frac{1}{2}\frac{\partial^2 V}{\partial S \partial \sigma}\Delta S \Delta \sigma$ (vanna effect)
- **Third-order terms:** Speed, color (time derivative of gamma)

### P&L Attribution

For a delta-hedged portfolio:
$$P\&L \approx \frac{1}{2}\Gamma (\Delta S)^2 + \Theta \Delta t + \nu \Delta \sigma + \text{higher order terms}$$

This decomposition helps understand the sources of P&L:
- **Gamma P&L:** From realized volatility vs implied volatility
- **Theta P&L:** Time decay (cost of carry)
- **Vega P&L:** Changes in implied volatility

## Dynamic Hedging

Dynamic hedging involves continuously rebalancing the hedge to maintain delta neutrality. In practice, rebalancing occurs at discrete intervals.

### Hedging P&L

The hedging error over a time interval $[t, t+\Delta t]$ is:

$$\text{Hedging Error} = V(t+\Delta t) - V(t) - \Delta(t)[S(t+\Delta t) - S(t)] - r[V(t) - \Delta(t)S(t)]\Delta t$$

The first term is the option value change, the second is the hedge P&L, and the third is the financing cost.

### Discrete Hedging Error

For a delta-hedged portfolio, the hedging error is approximately:

$$\text{Error} \approx \frac{1}{2}\Gamma [(\Delta S)^2 - \sigma^2 S^2 \Delta t]$$

This shows that hedging error comes from:
- **Realized volatility** differing from implied volatility
- **Discrete rebalancing** (jump risk)

### Optimal Hedging Frequency

The trade-off:
- **More frequent rebalancing:** Lower hedging error but higher transaction costs
- **Less frequent rebalancing:** Higher hedging error but lower costs

Optimal frequency balances these factors, depending on gamma, volatility, and transaction costs.

## Greek Profiles

### Across Strike Price

**Delta:**
- Increases monotonically from 0 to 1 for calls
- Decreases monotonically from 0 to -1 for puts
- Steepest at-the-money

**Gamma:**
- Peaks at-the-money
- Decreases rapidly away from ATM
- Higher for lower strikes (put skew effect)

**Vega:**
- Maximum at-the-money
- Decreases symmetrically away from ATM
- Shape similar to gamma

### Across Maturity

**Delta:**
- More stable for longer-dated options
- Becomes more binary near expiration

**Gamma:**
- Increases as expiration approaches (for ATM options)
- "Gamma explosion" near expiration

**Theta:**
- More negative near expiration
- Time decay accelerates

**Vega:**
- Increases with time to expiration (up to a point)
- Then decreases for very long-dated options

### Across Volatility

**Delta:**
- Lower volatility: steeper delta curve (higher gamma)
- Higher volatility: flatter delta curve

**Gamma:**
- Inversely related to volatility
- Lower vol → higher gamma (more sensitivity)

**Vega:**
- Increases with volatility level
- Higher vol → higher vega sensitivity

## Portfolio Greeks

For a portfolio of options:
$$\Delta_{portfolio} = \sum_i n_i \Delta_i$$
$$\Gamma_{portfolio} = \sum_i n_i \Gamma_i$$
$$\Theta_{portfolio} = \sum_i n_i \Theta_i$$
$$\nu_{portfolio} = \sum_i n_i \nu_i$$

where $n_i$ is the quantity (positive for long, negative for short).

### Risk Limits

Trading desks often impose limits on portfolio Greeks:
- **Delta limit:** Maximum net delta exposure
- **Gamma limit:** Maximum gamma (to control hedging costs)
- **Vega limit:** Maximum vega exposure
- **Theta limit:** Maximum daily theta (time decay budget)

## Practical Considerations

### Transaction Costs

Greeks assume frictionless trading. In practice:
- Bid-ask spreads reduce hedging effectiveness
- Commissions add to hedging costs
- Market impact affects large positions

### Model Risk

Greeks depend on the pricing model:
- **Black-Scholes:** Assumes constant volatility
- **Local volatility:** Greeks differ due to vol smile
- **Stochastic vol:** Additional vega terms

### Smile Effects

When volatility varies with strike:
- **Sticky strike:** Greeks computed at fixed implied vol
- **Sticky delta:** Greeks adjusted for vol changes
- **Sticky moneyness:** Vol moves with underlying

The choice affects hedging performance and P&L attribution.
