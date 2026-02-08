# Transaction Costs and Execution Algorithms

## Introduction

Transaction costs significantly impact portfolio performance, especially for high-turnover strategies. Understanding and managing these costs through optimal execution algorithms is crucial. This document covers transaction cost components, execution algorithms, and transaction cost analysis.

## Transaction Costs

### Explicit Costs

**Commissions:** Fees paid to brokers
$$C_{comm} = \sum_i c_i \times Q_i$$

where $c_i$ is commission rate for trade $i$ and $Q_i$ is quantity.

**Exchange fees:** Trading fees, clearing fees
$$C_{exchange} = \sum_i f_i \times Q_i$$

**Taxes:** Securities transaction taxes (varies by jurisdiction)

### Implicit Costs

Implicit costs are often larger than explicit costs:

1. **Bid-ask spread:** Paying the spread
2. **Market impact:** Price movement due to trading
3. **Timing cost:** Price movement while waiting to trade

## Market Impact

### Temporary vs Permanent Impact

**Temporary impact:** Price movement that reverts
- Due to order flow imbalance
- Recovers quickly (seconds to minutes)

**Permanent impact:** Price movement that persists
- Due to information content
- Reflects new equilibrium price

### Square-Root Law

Empirical observation: Market impact scales as square root of trade size:

$$Impact \propto \sigma \sqrt{\frac{Q}{ADV}}$$

where:
- $\sigma$: Volatility
- $Q$: Trade size
- $ADV$: Average daily volume

**Formulation:**
$$Impact = \alpha \sigma \sqrt{\frac{Q}{ADV}}$$

where $\alpha$ is a constant (typically 0.5-1.0).

### Linear Impact Model

Simpler model:
$$Impact = \lambda Q$$

where $\lambda$ is the price impact coefficient.

**Estimation:**
$$\lambda = \frac{\text{Cov}(\Delta p, Q)}{\text{Var}(Q)}$$

### Transient Impact Model

Almgren-Chriss model:
$$Impact(t) = h(v_t) + \int_0^t g(v_s) e^{-\rho(t-s)} ds$$

where:
- $h(v_t)$: Temporary impact (depends on current trading rate $v_t$)
- $g(v_s)$: Permanent impact (depends on past trading)
- $\rho$: Decay rate

**Common forms:**
- Temporary: $h(v) = \eta v^\gamma$ (often $\gamma = 1/2$ or $1$)
- Permanent: $g(v) = \kappa v$

## Almgren-Chriss Optimal Execution Framework

### Setup

Goal: Execute $X$ shares over time $[0,T]$ to minimize:
$$E[Cost] + \lambda \text{Var}(Cost)$$

where $\lambda$ is risk aversion.

**State variables:**
- $x_t$: Remaining shares to trade
- $S_t$: Asset price
- $v_t$: Trading rate

**Price dynamics:**
$$dS_t = \sigma dW_t + g(v_t) dt$$

where:
- $\sigma dW_t$: Random walk
- $g(v_t)$: Permanent impact

**Inventory evolution:**
$$\frac{dx_t}{dt} = -v_t$$

with boundary conditions: $x_0 = X$, $x_T = 0$.

### Cost Function

**Execution cost:**
$$Cost = \int_0^T v_t S_t dt + \int_0^T h(v_t) v_t dt$$

where:
- First term: Cost of trading at current price
- Second term: Temporary impact cost

**With permanent impact:**
$$Cost = \int_0^T v_t [S_0 + \sigma W_t + \int_0^t g(v_s) ds] dt + \int_0^T h(v_t) v_t dt$$

### Optimal Strategy

For linear temporary impact $h(v) = \eta v$ and permanent impact $g(v) = \kappa v$:

**Optimal trading rate:**
$$v_t^* = \frac{X}{T} \left(1 + \frac{\sinh(\alpha(T-t))}{\sinh(\alpha T)}\right)$$

where:
$$\alpha = \sqrt{\frac{\lambda \sigma^2}{\eta + \kappa/2}}$$

**Properties:**
- Starts fast (high $v_0$)
- Slows down over time
- More risk-averse ($\lambda$) → faster trading
- Higher volatility ($\sigma$) → faster trading

### Expected Cost

$$E[Cost] = X S_0 + \frac{\eta X^2}{T} + \frac{\kappa X^2}{2}$$

**Components:**
1. $X S_0$: Fundamental cost
2. $\frac{\eta X^2}{T}$: Temporary impact (decreases with $T$)
3. $\frac{\kappa X^2}{2}$: Permanent impact (independent of $T$)

### Risk

$$\text{Var}(Cost) = \frac{\sigma^2 X^2 T}{3}$$

Risk increases with execution time $T$.

## Execution Algorithms

### VWAP (Volume-Weighted Average Price)

**Goal:** Execute at average price weighted by volume.

**Strategy:** Trade in proportion to expected volume:
$$v_t = \frac{V_t^{expected}}{V_{total}^{expected}} \times X$$

where $V_t^{expected}$ is expected volume in period $t$.

**Advantages:**
- Simple
- Natural benchmark
- Low market impact (spreads trades)

**Disadvantages:**
- Doesn't adapt to market conditions
- May miss opportunities
- Requires volume forecasts

### TWAP (Time-Weighted Average Price)

**Goal:** Execute evenly over time.

**Strategy:**
$$v_t = \frac{X}{T}$$

Constant trading rate.

**Advantages:**
- Very simple
- Predictable
- Low implementation cost

**Disadvantages:**
- Ignores volume patterns
- May trade during illiquid periods
- Doesn't adapt to opportunities

### POV (Percentage of Volume)

**Goal:** Trade at constant percentage of market volume.

**Strategy:**
$$v_t = \alpha \times V_t^{observed}$$

where $\alpha$ is participation rate (e.g., 10%).

**Advantages:**
- Adapts to liquidity
- Natural risk control
- Market-aware

**Disadvantages:**
- Requires real-time volume data
- May take longer than planned
- Volume uncertainty

### Implementation Shortfall (IS)

**Goal:** Minimize difference between execution price and arrival price.

**Definition:**
$$IS = \frac{1}{X} \sum_{i=1}^n p_i Q_i - p_0$$

where:
- $p_i$: Execution price for trade $i$
- $Q_i$: Quantity for trade $i$
- $p_0$: Arrival price (price when decision made)

**Decomposition:**
$$IS = \text{Delay Cost} + \text{Market Impact} + \text{Opportunity Cost}$$

**Minimization:** Use Almgren-Chriss or similar framework.

## Arrival Price Benchmarks

### Arrival Price

Price at time decision is made (when order arrives).

**Use:** Common benchmark for execution quality.

### Previous Close

Closing price from previous day.

**Use:** Standard benchmark, but may be stale.

### Opening Price

Opening auction price.

**Use:** For orders placed before market open.

### Mid-Price

Average of best bid and ask: $m = \frac{a+b}{2}$.

**Use:** Theoretical execution price for small orders.

### Decision Price

Price when investment decision was made (may differ from arrival price).

**Use:** Most relevant for investment performance.

## Pre-Trade Analytics

### Impact Estimation

Estimate expected market impact before trading:

**Square-root model:**
$$E[Impact] = \alpha \sigma \sqrt{\frac{Q}{ADV}}$$

**Factors:**
- Trade size relative to ADV
- Volatility
- Liquidity (spread, depth)
- Market conditions

### Cost Prediction

Predict total execution cost:

$$E[Total Cost] = \text{Spread} + E[Impact] + \text{Delay Cost}$$

**Components:**
- Spread: $\frac{1}{2}(a-b)$
- Impact: From impact model
- Delay: Price drift while waiting

### Optimal Participation Rate

For POV strategy, choose $\alpha$ to balance:
- Execution time (lower $\alpha$ → longer)
- Market impact (lower $\alpha$ → less impact)

**Trade-off:**
$$\min_\alpha E[Cost(\alpha)] + \lambda \text{Var}(Cost(\alpha))$$

## Post-Trade Analysis: TCA

### Transaction Cost Analysis (TCA)

TCA measures actual execution quality:

**Metrics:**
1. **Execution shortfall:** $IS = \bar{p} - p_0$
2. **VWAP slippage:** $\bar{p} - VWAP$
3. **TWAP slippage:** $\bar{p} - TWAP$
4. **Market impact:** Temporary and permanent
5. **Timing cost:** Price movement during execution

### Decomposition

**Total cost:**
$$Total Cost = \text{Explicit} + \text{Implicit}$$

**Implicit cost:**
$$Implicit = \text{Spread} + \text{Impact} + \text{Timing}$$

**Impact:**
$$Impact = \text{Permanent} + \text{Temporary}$$

### Benchmark Comparison

Compare execution to benchmarks:

**Arrival price:**
$$IS_{arrival} = \bar{p} - p_{arrival}$$

**VWAP:**
$$IS_{VWAP} = \bar{p} - VWAP$$

**TWAP:**
$$IS_{TWAP} = \bar{p} - TWAP$$

**Decision price:**
$$IS_{decision} = \bar{p} - p_{decision}$$

### Performance Attribution

Decompose execution performance:

1. **Market movement:** $p_{close} - p_{arrival}$ (uncontrollable)
2. **Execution quality:** $IS - (p_{close} - p_{arrival})$ (controllable)

**Good execution:** Minimizes controllable component.

### Reporting

TCA reports typically include:
- Summary statistics (mean, median IS)
- Distribution of costs
- Breakdown by:
  - Order size
  - Volatility regime
  - Time of day
  - Market conditions
- Comparison to benchmarks
- Trend analysis

## Example: Execution Cost Calculation

Consider executing 10,000 shares:

**Market data:**
- Arrival price: $\$100.00$
- Best bid: $\$99.98$
- Best ask: $\$100.02$
- Spread: $\$0.04$ (0.04%)

**Execution:**
- Trade 1: 3,000 shares at $\$100.01$
- Trade 2: 4,000 shares at $\$100.03$
- Trade 3: 3,000 shares at $\$100.02$
- Average execution price: $\$100.02$

**Costs:**

**Explicit:**
- Commission: $\$0.001$ per share = $\$10$
- Exchange fee: $\$0.0001$ per share = $\$1$
- Total explicit: $\$11$

**Implicit:**

**Spread cost:**
- Half-spread: $\$0.02$ per share
- Total: $\$0.02 \times 10,000 = \$200$

**Market impact:**
- Temporary: Price moved from $\$100.00$ to $\$100.02$ during execution
- Impact: $\$0.02$ per share = $\$200$
- After execution, price reverts to $\$100.01$
- Temporary component: $\$100.02 - \$100.01 = \$0.01$ per share = $\$100$
- Permanent component: $\$100.01 - \$100.00 = \$0.01$ per share = $\$100$

**Timing cost:**
- Price drift: $\$100.01 - \$100.00 = \$0.01$ per share = $\$100$

**Total implicit:** $\$200 + \$200 + \$100 = \$500$

**Total cost:** $\$11 + \$500 = \$511$ (0.511% of trade value)

**Implementation shortfall:**
$$IS = \$100.02 - \$100.00 = \$0.02 \text{ per share} = 0.02\%$$

**VWAP comparison:**
If VWAP during execution period was $\$100.015$:
$$IS_{VWAP} = \$100.02 - \$100.015 = \$0.005 \text{ per share}$$

Execution slightly worse than VWAP.
