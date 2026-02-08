# Options Pricing and Black-Scholes

## No-Arbitrage Principle

The fundamental principle of derivatives pricing is the absence of arbitrage opportunities. An arbitrage is a risk-free profit opportunity that requires no initial investment. In efficient markets, arbitrage opportunities are quickly eliminated by market participants.

### Replication Principle

The replication principle states that if two portfolios have identical payoffs in all future states, they must have the same price today. Otherwise, an arbitrage opportunity exists.

Consider a European call option with strike $K$ and maturity $T$. We can replicate its payoff using:
- A long position in $\Delta$ shares of the underlying stock
- A short position in a risk-free bond

The replicating portfolio value at time $t$ is:
$$V(t) = \Delta S(t) - B(t)$$

where $B(t)$ is the value of the bond position. At maturity, we require:
$$V(T) = \max(S(T) - K, 0)$$

### Risk-Neutral Pricing

Under the risk-neutral measure $\mathbb{Q}$, all assets earn the risk-free rate $r$. The price of a derivative with payoff $H(T)$ at maturity $T$ is:

$$V(0) = e^{-rT} \mathbb{E}^{\mathbb{Q}}[H(T)]$$

This expectation is taken under the risk-neutral probability measure, not the real-world measure $\mathbb{P}$. The risk-neutral measure adjusts probabilities to account for risk aversion, ensuring that discounted asset prices are martingales.

## Black-Scholes Model

### Assumptions

The Black-Scholes model makes the following assumptions:

1. **Stock price follows geometric Brownian motion:**
   $$dS(t) = \mu S(t)dt + \sigma S(t)dW(t)$$
   where $\mu$ is the drift, $\sigma$ is the volatility, and $W(t)$ is a standard Brownian motion.

2. **Risk-free rate $r$ is constant** and known.

3. **No dividends** are paid during the option's life.

4. **No transaction costs** or taxes.

5. **Frictionless markets** - continuous trading is possible.

6. **Volatility $\sigma$ is constant** and known.

7. **Short selling** is allowed with full use of proceeds.

### PDE Derivation

Consider a portfolio $\Pi$ consisting of:
- One option $V(S,t)$
- $-\Delta$ shares of stock

The portfolio value is:
$$\Pi = V - \Delta S$$

Using Itô's lemma, the change in option value is:
$$dV = \frac{\partial V}{\partial t}dt + \frac{\partial V}{\partial S}dS + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}dt$$

The change in portfolio value is:
$$d\Pi = dV - \Delta dS$$

Substituting and choosing $\Delta = \frac{\partial V}{\partial S}$ to eliminate the stochastic term:
$$d\Pi = \left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right)dt$$

Since the portfolio is risk-free, it must earn the risk-free rate:
$$d\Pi = r\Pi dt = r\left(V - \frac{\partial V}{\partial S}S\right)dt$$

Equating the two expressions for $d\Pi$:
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

This is the **Black-Scholes partial differential equation**.

### Boundary Conditions

For a European call option:
- **Final condition:** $V(S,T) = \max(S - K, 0)$
- **Boundary at $S = 0$:** $V(0,t) = 0$ (option worthless if stock hits zero)
- **Boundary as $S \to \infty$:** $V(S,t) \sim S - Ke^{-r(T-t)}$ (option behaves like stock minus discounted strike)

For a European put option:
- **Final condition:** $V(S,T) = \max(K - S, 0)$
- **Boundary at $S = 0$:** $V(0,t) = Ke^{-r(T-t)}$ (option worth discounted strike)
- **Boundary as $S \to \infty$:** $V(S,t) = 0$ (option worthless)

### Black-Scholes Formula

The solution to the Black-Scholes PDE for a European call option is:

$$C(S,t) = S_0 N(d_1) - Ke^{-r(T-t)}N(d_2)$$

where:
$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$$

$$d_2 = d_1 - \sigma\sqrt{T-t} = \frac{\ln(S/K) + (r - \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$$

and $N(\cdot)$ is the cumulative distribution function of the standard normal distribution.

For a European put option:
$$P(S,t) = Ke^{-r(T-t)}N(-d_2) - S_0 N(-d_1)$$

### Black-Scholes via Risk-Neutral Expectation

Under the risk-neutral measure, the stock price follows:
$$dS(t) = rS(t)dt + \sigma S(t)dW^{\mathbb{Q}}(t)$$

The solution is:
$$S(T) = S(0)\exp\left((r - \frac{\sigma^2}{2})T + \sigma W^{\mathbb{Q}}(T)\right)$$

The call option price is:
$$C(0) = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\max(S(T) - K, 0)]$$

Computing this expectation:
$$C(0) = e^{-rT}\int_{K}^{\infty}(S(T) - K)f(S(T))dS(T)$$

where $f(S(T))$ is the log-normal density. This integral yields the Black-Scholes formula.

## Put-Call Parity

Put-call parity is a fundamental relationship between European call and put options with the same strike and maturity:

$$C - P = S - Ke^{-rT}$$

This relationship holds by no-arbitrage arguments. Consider two portfolios:

**Portfolio A:** One call option plus $Ke^{-rT}$ in cash
**Portfolio B:** One put option plus one share of stock

Both portfolios have payoff $\max(S(T), K)$ at maturity, so they must have the same value today.

### Proof

At maturity $T$:
- Portfolio A: $\max(S(T) - K, 0) + K = \max(S(T), K)$
- Portfolio B: $\max(K - S(T), 0) + S(T) = \max(S(T), K)$

Since the payoffs are identical, by no-arbitrage:
$$C + Ke^{-rT} = P + S$$

Rearranging gives put-call parity.

## Early Exercise and American Options

American options can be exercised at any time before maturity, making them more valuable than European options. For a non-dividend paying stock:

- **American call:** Never optimal to exercise early, so $C_{AM} = C_{EU}$
- **American put:** May be optimal to exercise early, so $P_{AM} \geq P_{EU}$

### Early Exercise Decision

For an American call on a non-dividend paying stock, early exercise is never optimal because:
1. The option provides insurance against downside moves
2. The time value is positive: $C > S - K$ for in-the-money options
3. Holding the option is better than exercising and holding stock

For an American put, early exercise may be optimal when:
- The put is deep in-the-money
- Interest rates are high (time value of receiving $K$ early)
- Volatility is low (less value in waiting)

The early exercise boundary $S^*(t)$ satisfies:
$$P(S^*(t), t) = K - S^*(t)$$

## Black-76 Model

The Black-76 model prices options on futures contracts. The key difference is that the underlying follows:

$$dF(t) = \sigma F(t)dW(t)$$

Note the absence of drift under the risk-neutral measure (futures are martingales).

The Black-76 formula for a call option on a future is:

$$C(F,t) = e^{-r(T-t)}[F(t)N(d_1) - KN(d_2)]$$

where:
$$d_1 = \frac{\ln(F/K) + \sigma^2(T-t)/2}{\sigma\sqrt{T-t}}$$

$$d_2 = d_1 - \sigma\sqrt{T-t}$$

For a put option:
$$P(F,t) = e^{-r(T-t)}[KN(-d_2) - F(t)N(-d_1)]$$

## Binomial Tree Model

The binomial tree model provides a discrete-time approximation to continuous-time option pricing. It assumes the stock price can move to one of two states in each period.

### Cox-Ross-Rubinstein (CRR) Model

In the CRR model, over a time step $\Delta t$:
- Stock moves up: $S \to Su$ with probability $p$
- Stock moves down: $S \to Sd$ with probability $1-p$

The parameters are chosen as:
$$u = e^{\sigma\sqrt{\Delta t}}$$
$$d = e^{-\sigma\sqrt{\Delta t}} = \frac{1}{u}$$
$$p = \frac{e^{r\Delta t} - d}{u - d}$$

The risk-neutral probability $p$ ensures the expected return equals the risk-free rate.

### Option Pricing on Tree

Working backwards from maturity:
$$V(i,j) = e^{-r\Delta t}[pV(i+1,j+1) + (1-p)V(i+1,j)]$$

where $V(i,j)$ is the option value at node $(i,j)$ (time step $i$, stock level $j$).

At maturity:
$$V(N,j) = \max(S_0 u^j d^{N-j} - K, 0) \quad \text{(for call)}$$

### Convergence to Black-Scholes

As $\Delta t \to 0$ and the number of steps $N \to \infty$, the binomial tree converges to the Black-Scholes formula. This follows from the Central Limit Theorem: the binomial distribution converges to the log-normal distribution.

The convergence is $O(\Delta t)$ for the CRR model. More sophisticated models (e.g., Jarrow-Rudd) achieve faster convergence.

### Example: Two-Period Binomial Tree

Consider a call option with $S_0 = 100$, $K = 100$, $r = 0.05$, $\sigma = 0.2$, $T = 1$ year, $N = 2$ steps.

$\Delta t = 0.5$, $u = e^{0.2\sqrt{0.5}} \approx 1.152$, $d \approx 0.868$, $p = \frac{e^{0.05 \cdot 0.5} - 0.868}{1.152 - 0.868} \approx 0.550$

Stock tree:
- $t=0$: $S_0 = 100$
- $t=0.5$: $Su = 115.2$ or $Sd = 86.8$
- $t=1$: $Su^2 = 132.7$, $Sud = 100$, $Sd^2 = 75.4$

Option payoffs at $t=1$:
- $C_{uu} = 32.7$, $C_{ud} = 0$, $C_{dd} = 0$

Working backwards:
- $C_u = e^{-0.05 \cdot 0.5}[0.550 \cdot 32.7 + 0.450 \cdot 0] \approx 17.5$
- $C_d = 0$
- $C_0 = e^{-0.05 \cdot 0.5}[0.550 \cdot 17.5 + 0.450 \cdot 0] \approx 9.4$

The Black-Scholes value is approximately $10.45$, showing convergence improves with more steps.
