# Market Making and Liquidity

## Introduction

Market makers provide liquidity by continuously quoting bid and ask prices, earning the spread. Successful market making requires managing inventory risk, adverse selection, and understanding liquidity dynamics. This document covers market making strategies, liquidity measurement, and liquidity risk.

## Market Maker Role

### Function

Market makers:
1. **Provide liquidity:** Stand ready to buy and sell
2. **Quote prices:** Continuously post bid and ask quotes
3. **Earn spread:** Profit from bid-ask spread
4. **Manage inventory:** Control position size and risk

### Revenue Sources

**Spread revenue:**
$$Revenue = \sum_t (a_t - b_t) \times Q_t$$

where:
- $a_t, b_t$: Ask and bid prices at time $t$
- $Q_t$: Quantity traded

**Rebates:** Some exchanges pay rebates for providing liquidity (maker-taker model).

### Costs

1. **Adverse selection:** Trading with informed traders
2. **Inventory risk:** Holding positions
3. **Order processing:** Technology, operations
4. **Capital requirements:** Margin, regulatory capital

## Avellaneda-Stoikov Model

### Setup

Avellaneda-Stoikov (2008) models optimal market making with:
- Inventory $q_t$ (can be positive or negative)
- Bid and ask quotes: $S_t^b, S_t^a$
- Arrival rates: $\lambda^b(S_t^b), \lambda^a(S_t^a)$
- Asset price follows: $dS_t = \sigma dW_t$

### Objective

Maximize expected utility:
$$\max E[-e^{-\gamma(X_T + q_T S_T)}]$$

where:
- $X_T$: Cash at time $T$
- $q_T$: Inventory at time $T$
- $\gamma$: Risk aversion

### Optimal Quotes

**Bid quote:**
$$S_t^b = S_t - \frac{1}{2}\gamma \sigma^2 (T-t) q_t - \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\lambda^b}\right)$$

**Ask quote:**
$$S_t^a = S_t + \frac{1}{2}\gamma \sigma^2 (T-t) q_t + \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\lambda^a}\right)$$

**Components:**
1. **Mid-price:** $S_t$ (current asset price)
2. **Inventory adjustment:** $\frac{1}{2}\gamma \sigma^2 (T-t) q_t$
   - Positive inventory: Lower bid, raise ask (want to sell)
   - Negative inventory: Raise bid, lower ask (want to buy)
3. **Spread component:** $\frac{1}{\gamma} \ln(1 + \gamma/\lambda)$
   - Wider spread when arrival rates are low
   - Narrower spread when arrival rates are high

### Key Insights

1. **Inventory management:** Quotes adjust to control inventory
2. **Risk aversion:** Higher $\gamma$ → wider spreads, more inventory control
3. **Time decay:** As $t \to T$, inventory adjustment increases (must close position)
4. **Arrival rates:** Higher arrival rates allow tighter spreads

## Bid-Ask Spread Determinants

### Components

The bid-ask spread has three components:

$$s = s_{adverse} + s_{inventory} + s_{processing}$$

### Adverse Selection Component

Compensates for trading with informed traders:

$$s_{adverse} = \mu (v_H - v_L)$$

from Glosten-Milgrom model, where:
- $\mu$: Probability of informed trading
- $v_H - v_L$: Information asymmetry

**Factors:**
- Higher information asymmetry → wider spread
- More informed trading → wider spread
- Earnings announcements, news events → wider spreads

### Inventory Risk Component

Compensates for holding inventory:

$$s_{inventory} = \gamma \sigma^2 q \times \text{holding period}$$

where:
- $\gamma$: Risk aversion
- $\sigma^2$: Return variance
- $q$: Inventory level

**Factors:**
- Higher volatility → wider spread
- Larger inventory → wider spread
- Longer holding period → wider spread

### Order Processing Component

Covers costs of processing orders:

$$s_{processing} = \text{fixed cost per trade}$$

**Factors:**
- Technology costs
- Exchange fees
- Operational costs

### Empirical Decomposition

**Roll (1984) model:**
If prices follow: $p_t = p_{t-1} + \epsilon_t$ with $\epsilon_t = \pm s/2$

Then: $\text{Cov}(\Delta p_t, \Delta p_{t-1}) = -s^2/4$

So: $s = 2\sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}$

**Huang-Stoll (1997) decomposition:**
$$s = 2[\theta \lambda + (1-\theta) \gamma \sigma^2 q]$$

where $\theta$ is the proportion due to adverse selection.

## Liquidity Measures

### Bid-Ask Spread

**Quoted spread:** $s = a - b$

**Effective spread:** Actual execution cost:
$$s_{eff} = 2|p - m|$$

where $p$ is execution price and $m$ is mid-price.

**Realized spread:** Spread after price reversal:
$$s_{realized} = 2E[p_{t+1} - p_t | \text{trade at } t]$$

Measures temporary vs permanent impact.

### Depth

**Quoted depth:** Total quantity at best bid and ask:
$$Depth = q_1^b + q_1^a$$

**Market depth:** Quantity needed to move price by X%:
$$Depth(X\%) = \min\{Q : |p(Q) - p_0|/p_0 \geq X\%\}$$

### Resilience

Resilience measures how quickly liquidity recovers after a trade.

**Definition:** Time for spread to return to normal after a large trade.

**Measurement:**
$$Resilience = \frac{1}{T_{recovery}}$$

where $T_{recovery}$ is time to recover.

### Amihud Illiquidity Measure

Amihud (2002) measures price impact per unit volume:

$$Illiq = \frac{1}{T} \sum_{t=1}^T \frac{|R_t|}{V_t}$$

where:
- $R_t$: Return on day $t$
- $V_t$: Dollar volume on day $t$

**Interpretation:**
- Higher value: More illiquid (large price moves per unit volume)
- Lower value: More liquid

**Annualized:**
$$Illiq_{annual} = Illiq \times 252$$

### Roll Measure

From Roll (1984), using negative autocovariance:

$$s_{Roll} = 2\sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}$$

**Advantages:**
- Uses only price data
- No need for bid-ask quotes

**Disadvantages:**
- Requires negative autocovariance
- Sensitive to estimation

### Kyle's Lambda

Price impact coefficient from Kyle's model:

$$\lambda = \frac{\text{Cov}(\Delta p, Q)}{\text{Var}(Q)}$$

where $Q$ is signed order flow.

**Interpretation:**
- Higher $\lambda$: Less liquid (large price impact)
- Lower $\lambda$: More liquid

## Liquidity Risk

### Definition

Liquidity risk is the risk that an asset cannot be traded quickly enough to prevent a loss or meet an obligation.

### Types

1. **Asset liquidity risk:** Asset becomes illiquid
2. **Funding liquidity risk:** Cannot obtain funding
3. **Market liquidity risk:** Market-wide illiquidity

### Liquidity Spirals

Liquidity spirals occur when illiquidity begets more illiquidity:

**Mechanism:**
1. Initial shock reduces liquidity
2. Prices fall due to forced selling
3. Margin calls force more selling
4. Liquidity further deteriorates
5. Spiral continues

**Mathematical model:**
$$L_t = L_0 e^{-\alpha t} - \beta \int_0^t e^{-\alpha(t-s)} P_s ds$$

where:
- $L_t$: Liquidity at time $t$
- $P_t$: Price pressure
- $\alpha$: Natural recovery rate
- $\beta$: Spiral strength

### Fire Sales

Fire sales occur when forced selling depresses prices below fundamental value.

**Causes:**
- Margin calls
- Redemptions
- Regulatory requirements
- Distress

**Impact:**
- Prices fall below fair value
- Contagion to other assets
- Systemic risk

### Measurement

**Liquidity at Risk (LaR):**
Similar to VaR, but for liquidity:

$$LaR_\alpha = \text{Maximum liquidity cost with probability } \alpha$$

**Components:**
- Bid-ask spread widening
- Market impact
- Execution delay

## Dark Pools

### Definition

Dark pools are private trading venues where orders are not displayed publicly.

### Types

1. **Broker-dealer pools:** Operated by brokers
2. **Exchange-operated:** Run by exchanges
3. **Independent:** Third-party operators

### Advantages

1. **Reduced market impact:** Hidden orders
2. **Price improvement:** Often trade at mid-price
3. **Anonymity:** Hide trading intentions

### Disadvantages

1. **Price discovery:** Less transparent
2. **Information leakage:** May reveal intentions to pool operators
3. **Fragmentation:** Liquidity split across venues

## Internalization

### Definition

Internalization occurs when brokers match client orders internally rather than routing to exchanges.

### Mechanism

1. Client submits order
2. Broker checks internal inventory
3. If match, execute internally
4. Otherwise, route to exchange

### Benefits

1. **Price improvement:** Can offer better prices
2. **Speed:** Faster execution
3. **Cost savings:** Avoid exchange fees

### Concerns

1. **Best execution:** May not get best price
2. **Price discovery:** Less price information
3. **Conflicts of interest:** Broker vs client

## Payment for Order Flow

### Definition

Payment for order flow (PFOF) is compensation brokers receive for routing orders to market makers.

### Mechanism

1. Broker routes order to market maker
2. Market maker pays broker
3. Market maker executes order (often at better price)

### Economics

**Broker:** Receives payment
**Market maker:** Pays but earns spread
**Client:** May get price improvement

### Controversy

**Arguments against:**
- Potential conflict of interest
- May not get best execution
- Opaque pricing

**Arguments for:**
- Price improvement for clients
- Lower costs for brokers
- Efficient market making

## Example: Market Making Strategy

Consider a market maker with:
- Current inventory: $q = 100$ shares
- Asset price: $S = \$100$
- Volatility: $\sigma = 0.02$ (2% daily)
- Risk aversion: $\gamma = 0.1$
- Time to close: $T-t = 1$ day
- Arrival rates: $\lambda^b = \lambda^a = 10$ orders/hour

**Optimal quotes (Avellaneda-Stoikov):**

Spread component:
$$\frac{1}{\gamma} \ln(1 + \gamma/\lambda) = \frac{1}{0.1} \ln(1 + 0.1/10) = 10 \times 0.00995 = 0.0995$$

Inventory adjustment:
$$\frac{1}{2}\gamma \sigma^2 (T-t) q = \frac{1}{2} \times 0.1 \times 0.0004 \times 1 \times 100 = 0.002$$

**Bid quote:**
$$S^b = 100 - 0.002 - 0.0995 = \$99.8985$$

**Ask quote:**
$$S^a = 100 + 0.002 + 0.0995 = \$100.1015$$

**Spread:** $100.1015 - 99.8985 = \$0.203$ (0.203%)

The market maker quotes:
- Slightly lower bid (wants to reduce inventory)
- Standard spread to earn revenue
- Inventory adjustment is small (low risk, short horizon)

If inventory increases to $q = 500$:
- Inventory adjustment: $0.01$
- Bid: $\$99.8895$ (lower, wants to sell more)
- Ask: $\$100.1105$ (higher, less willing to buy)

The quotes skew more aggressively to manage inventory risk.
