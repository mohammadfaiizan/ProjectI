# Market Impact and Optimal Execution

## Introduction

Market impact—the price movement caused by trading—is a critical cost. Optimal execution algorithms minimize total cost including impact, risk, and opportunity cost. This document covers impact models, optimal execution frameworks, and execution quality metrics.

## Price Impact Models

### Linear Impact Model

Simplest model:
$$\Delta p = \lambda Q$$

where:
- $\Delta p$: Price change
- $\lambda$: Impact coefficient
- $Q$: Trade size

**Estimation:**
$$\lambda = \frac{\text{Cov}(\Delta p, Q)}{\text{Var}(Q)}$$

**Limitations:**
- Doesn't capture size effects
- No time dependence
- Too simple for large orders

### Square-Root Law

Empirical observation:
$$Impact \propto \sigma \sqrt{\frac{Q}{ADV}}$$

**Formulation:**
$$Impact = \alpha \sigma \sqrt{\frac{Q}{ADV}}$$

where:
- $\alpha$: Constant (typically 0.5-1.0)
- $\sigma$: Volatility
- $Q$: Trade size
- $ADV$: Average daily volume

**Properties:**
- Concave in trade size
- Captures empirical regularity
- Widely used in practice

### Transient Impact Model

Impact decays over time:

$$Impact(t) = h(v_t) + \int_0^t g(v_s) e^{-\rho(t-s)} ds$$

where:
- $h(v_t)$: Temporary impact (current trading rate)
- $g(v_s)$: Permanent impact (past trading)
- $\rho$: Decay rate

**Common forms:**
- Temporary: $h(v) = \eta v^\gamma$ ($\gamma = 1/2$ or $1$)
- Permanent: $g(v) = \kappa v$

### Permanent vs Temporary

**Permanent impact:** Reflects information content
- Persists indefinitely
- Reflects new equilibrium
- Model: $g(v) = \kappa v$

**Temporary impact:** Due to order flow imbalance
- Decays over time
- Recovers as liquidity returns
- Model: $h(v) = \eta v^\gamma e^{-\rho t}$

**Total impact:**
$$Impact_{total} = Impact_{permanent} + Impact_{temporary}$$

## Obizhaeva-Wang Model

### Setup

Obizhaeva-Wang (2013) models block trading with transient impact:

**Price dynamics:**
$$dS_t = \sigma dW_t + f(v_t) dt$$

where $f(v_t)$ is impact function.

**Impact function:**
$$f(v) = \kappa v + \eta \frac{dv}{dt}$$

**Inventory:**
$$\frac{dx_t}{dt} = -v_t$$

with $x_0 = X$, $x_T = 0$.

### Optimal Strategy

**For linear impact:** $f(v) = \kappa v$

Optimal trading rate:
$$v_t^* = \frac{X}{T} \left[1 + \frac{\sinh(\alpha(T-t))}{\sinh(\alpha T)}\right]$$

where $\alpha = \sqrt{\lambda \sigma^2 / \kappa}$.

**Properties:**
- Starts fast, slows down
- More risk-averse → faster
- Higher volatility → faster

**For transient impact:** More complex, involves exponential decay terms.

## Almgren-Chriss in Detail

### Problem Formulation

Execute $X$ shares over $[0,T]$ to minimize:
$$E[Cost] + \lambda \text{Var}(Cost)$$

**Price:**
$$S_t = S_0 + \sigma W_t + \int_0^t g(v_s) ds$$

**Inventory:**
$$x_t = X - \int_0^t v_s ds$$

**Execution cost:**
$$Cost = \int_0^T v_t S_t dt + \int_0^T h(v_t) v_t dt$$

### Linear Impact Case

**Temporary:** $h(v) = \eta v$
**Permanent:** $g(v) = \kappa v$

**Optimal trading rate:**
$$v_t^* = \frac{X}{T} \frac{\cosh(\alpha(T-t))}{\cosh(\alpha T)}$$

where:
$$\alpha = \sqrt{\frac{\lambda \sigma^2}{\eta + \kappa/2}}$$

**Expected cost:**
$$E[Cost] = X S_0 + \frac{\eta X^2}{T} + \frac{\kappa X^2}{2}$$

**Variance:**
$$\text{Var}(Cost) = \frac{\sigma^2 X^2 T}{3}$$

### Risk-Aversion Parameter

$\lambda$ balances:
- **Low $\lambda$:** Focus on expected cost → slower trading
- **High $\lambda$:** Focus on risk → faster trading

**Interpretation:**
- $\lambda = 0$: Risk-neutral (minimize expected cost only)
- $\lambda > 0$: Risk-averse (trade off cost and risk)

**Typical values:**
- Conservative: $\lambda = 10^{-6}$
- Moderate: $\lambda = 10^{-5}$
- Aggressive: $\lambda = 10^{-4}$

### Time Decay

As $t \to T$, inventory adjustment increases:
- Must close position
- Less time to manage risk
- Quotes skew more aggressively

## Multi-Asset Execution

### Portfolio Execution

For portfolio with $n$ assets:

**State:**
- Inventory vector: $\boldsymbol{x}_t \in \mathbb{R}^n$
- Trading rates: $\boldsymbol{v}_t \in \mathbb{R}^n$

**Price:**
$$d\boldsymbol{S}_t = \boldsymbol{\Sigma}^{1/2} d\boldsymbol{W}_t + \boldsymbol{G}(\boldsymbol{v}_t) dt$$

where $\boldsymbol{\Sigma}$ is covariance matrix.

**Cost:**
$$Cost = \int_0^T \boldsymbol{v}_t^T \boldsymbol{S}_t dt + \int_0^T \boldsymbol{v}_t^T \boldsymbol{H}(\boldsymbol{v}_t) dt$$

### Cross-Impact

Assets impact each other:

**Impact matrix:**
$$\boldsymbol{G}(\boldsymbol{v}) = \boldsymbol{\kappa} \boldsymbol{v}$$

where $\boldsymbol{\kappa}$ is cross-impact matrix.

**Optimal strategy:** More complex, involves matrix equations.

**Key insight:** Correlated assets should be traded together to reduce risk.

### Portfolio Transitions

**Problem:** Transition from portfolio $\boldsymbol{w}_0$ to $\boldsymbol{w}_T$.

**Constraints:**
- Start: $\boldsymbol{x}_0 = V_0 \boldsymbol{w}_0$
- End: $\boldsymbol{x}_T = V_T \boldsymbol{w}_T$
- Value changes: $V_t$ evolves with prices

**Solution:** Extend Almgren-Chriss to handle:
- Changing target (due to price movements)
- Rebalancing constraints
- Cross-asset impact

## Adaptive Execution

### Real-Time Adjustment

Adapt execution based on:
1. **Market conditions:** Volatility, liquidity
2. **Price movements:** Favorable/unfavorable
3. **Execution progress:** Ahead/behind schedule

### Adaptive Almgren-Chriss

**Dynamic risk aversion:**
$$\lambda_t = \lambda_0 \times \frac{\sigma_t^2}{\sigma_0^2}$$

Adjusts to current volatility.

**Dynamic time horizon:**
If ahead of schedule, can slow down:
$$T_{remaining} = T - t + \text{buffer}$$

**Price adjustment:**
If price moves favorably, can:
- Slow down (let price work for you)
- Speed up (lock in gains)

### Reinforcement Learning

**State:** $(x_t, S_t, t, \text{market state})$
**Action:** Trading rate $v_t$
**Reward:** $-(\text{cost} + \lambda \times \text{risk})$

**Learn:** Optimal policy $\pi^*(s)$

**Advantages:**
- Adapts to market regimes
- Learns from experience
- Handles complex dynamics

**Challenges:**
- Requires data
- Training time
- Overfitting risk

## Execution Quality Metrics

### Implementation Shortfall

$$IS = \frac{1}{X} \sum_{i=1}^n p_i Q_i - p_0$$

where:
- $p_i$: Execution price
- $Q_i$: Quantity
- $p_0$: Arrival price

**Decomposition:**
$$IS = \text{Delay} + \text{Impact} + \text{Opportunity}$$

### VWAP Slippage

$$Slippage_{VWAP} = \bar{p} - VWAP$$

where $\bar{p}$ is average execution price.

**Interpretation:**
- Negative: Better than VWAP
- Positive: Worse than VWAP

### Reversion

**Temporary impact reversion:**
$$Reversion = p_{execution} - p_{post-execution}$$

Measures how much temporary impact reverts.

**Permanent impact:**
$$Permanent = p_{post-execution} - p_{arrival}$$

Measures persistent price change.

### Participation-Weighted Price (PWP)

Weight execution prices by participation rate:

$$PWP = \frac{\sum_i p_i \times \text{participation}_i}{\sum_i \text{participation}_i}$$

**Benchmark:** Compare to market PWP.

### Effective Spread

$$s_{eff} = 2|p - m|$$

where $m$ is mid-price at execution time.

**Average:**
$$\bar{s}_{eff} = \frac{1}{n} \sum_{i=1}^n 2|p_i - m_i|$$

## Machine Learning for Execution

### Impact Prediction

**Features:**
- Order size
- Volatility
- Liquidity (spread, depth)
- Time of day
- Recent order flow

**Target:** Market impact

**Models:**
- Linear regression
- Random forests
- Neural networks

### Optimal Execution

**State:** Current inventory, market conditions
**Action:** Trading rate
**Reward:** Negative cost

**Methods:**
- Q-learning
- Policy gradient
- Actor-critic

### Order Flow Prediction

Predict future order flow to:
- Anticipate liquidity
- Avoid adverse periods
- Time executions

**Features:**
- Historical order flow
- Market indicators
- Cross-asset signals

## Example: Optimal Execution

Execute 100,000 shares over 1 day:

**Parameters:**
- Current price: $\$100$
- Volatility: $\sigma = 0.02$ (2% daily)
- Temporary impact: $\eta = 10^{-6}$ per share
- Permanent impact: $\kappa = 5 \times 10^{-7}$ per share
- Risk aversion: $\lambda = 10^{-5}$
- Time: $T = 1$ day (252 trading minutes)

**Almgren-Chriss solution:**

$$\alpha = \sqrt{\frac{10^{-5} \times 0.0004}{10^{-6} + 2.5 \times 10^{-7}}} = \sqrt{\frac{4 \times 10^{-9}}{1.25 \times 10^{-6}}} = \sqrt{3.2 \times 10^{-3}} = 0.0566$$

**Optimal trading rate:**
$$v_t = \frac{100,000}{252} \frac{\cosh(0.0566 \times (252-t))}{\cosh(0.0566 \times 252)}$$

**At start ($t=0$):**
$$v_0 = \frac{100,000}{252} \times 1.0 = 397 \text{ shares/minute}$$

**At end ($t=252$):**
$$v_{252} = \frac{100,000}{252} \times \frac{1}{\cosh(14.3)} \approx 0$$

**Expected cost:**
$$E[Cost] = 100,000 \times 100 + \frac{10^{-6} \times 10^{10}}{252} + \frac{5 \times 10^{-7} \times 10^{10}}{2}$$
$$= 10,000,000 + 39,683 + 2,500 = \$10,042,183$$

**Cost per share:** $\$0.042$ (4.2 cents)

**Components:**
- Fundamental: $\$100.00$ per share
- Temporary impact: $\$0.040$ per share
- Permanent impact: $\$0.002$ per share

**Variance:**
$$\text{Var}(Cost) = \frac{0.0004 \times 10^{10} \times 252}{3} = 336,000,000$$

**Standard deviation:** $\$18,330$ (0.183% of trade value)

The optimal strategy trades faster initially to reduce risk, then slows down as deadline approaches.
