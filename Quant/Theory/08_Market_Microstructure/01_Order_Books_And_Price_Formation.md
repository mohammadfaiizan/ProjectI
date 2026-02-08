# Order Books and Price Formation

## Introduction

Market microstructure studies how prices are formed through the interaction of buyers and sellers. The limit order book is the central mechanism in modern electronic markets. Understanding order book dynamics, price discovery, and the behavior of informed versus uninformed traders is essential for quantitative trading.

## Limit Order Book Structure

### Basic Components

A limit order book consists of:

1. **Bid side:** Buy orders, sorted by price (highest first)
2. **Ask side:** Sell orders, sorted by price (lowest first)
3. **Orders:** Each order specifies price, quantity, timestamp

**Notation:**
- $b_i$: Bid price at level $i$ (with $b_1 > b_2 > \ldots$)
- $a_i$: Ask price at level $i$ (with $a_1 < a_2 < \ldots$)
- $q_i^b$: Quantity at bid level $i$
- $q_i^a$: Quantity at ask level $i$

### Bid-Ask Spread

The bid-ask spread is:
$$s = a_1 - b_1$$

**Percentage spread:**
$$s\% = \frac{a_1 - b_1}{m} \times 100\%$$

where $m = \frac{a_1 + b_1}{2}$ is the mid-price.

**Tight spread:** Low transaction costs, high liquidity
**Wide spread:** High transaction costs, low liquidity

### Depth

Depth measures liquidity available at each price level:

**Cumulative depth at $k$ ticks:**
$$D_k^b = \sum_{i=1}^k q_i^b$$
$$D_k^a = \sum_{i=1}^k q_i^a$$

**Total depth:** Sum of all quantities on both sides.

### Order Book Levels

Modern markets display multiple price levels (typically 5-20 levels). The shape of the order book provides information about:
- Liquidity distribution
- Support/resistance levels
- Market maker activity

## Order Types

### Market Orders

Market orders execute immediately at the best available price.

**Execution:**
- Buy market order: Executes against ask side, starting at $a_1$
- Sell market order: Executes against bid side, starting at $b_1$

**Cost:** Pay the spread (buy at ask, sell at bid).

### Limit Orders

Limit orders specify maximum (buy) or minimum (sell) execution price.

**Execution priority:**
1. **Price:** Better prices execute first
2. **Time:** Among same price, earlier orders execute first (FIFO)

**Advantages:**
- Price protection
- Can provide liquidity (earn spread)

**Disadvantages:**
- Execution uncertainty
- May not fill if price moves away

### Stop Orders

Stop orders become market orders when price reaches a trigger level.

**Stop-loss:** Sell when price falls to $X$ (limits losses)
**Stop-buy:** Buy when price rises to $X$ (captures breakouts)

### Iceberg Orders

Iceberg orders show only a portion of total quantity, hiding large orders.

**Mechanism:**
- Display quantity: $q_{display}$
- Hidden quantity: $q_{hidden}$
- As displayed quantity executes, hidden quantity replenishes

**Purpose:** Reduce market impact of large orders.

### Immediate-or-Cancel (IOC)

IOC orders execute immediately or are cancelled.

**Use case:** Quick execution without leaving orders in book.

### Fill-or-Kill (FOK)

FOK orders must execute completely or are cancelled.

**Use case:** All-or-nothing execution requirement.

## Price Discovery Mechanisms

### Continuous Double Auction

In continuous trading, orders execute as they arrive if they can match:

**Matching rules:**
- Buy order with price $\geq a_1$: Executes immediately
- Sell order with price $\leq b_1$: Executes immediately
- Otherwise: Order enters book as limit order

**Price-time priority:**
- Better price executes first
- Same price: earlier order executes first

### Call Auction

In call auctions, orders accumulate and execute at a single price:

**Mechanism:**
1. Orders accumulate during call period
2. At call time, find price $p^*$ maximizing volume:
$$p^* = \arg\max_p \min(D^b(p), D^a(p))$$

where $D^b(p)$ and $D^a(p)$ are cumulative demand and supply at price $p$.

**Use cases:**
- Market open/close
- Less liquid securities
- Reducing volatility

### Tick Size

Tick size is the minimum price increment (e.g., \$0.01 for stocks).

**Impact:**
- Larger tick: Higher spreads, more depth
- Smaller tick: Tighter spreads, less depth
- Affects market maker profitability

### Lot Size

Lot size is the minimum tradeable quantity (e.g., 100 shares for round lots).

**Odd lots:** Quantities not multiples of lot size
- May have different execution rules
- Often trade at worse prices

### Priority Rules

**Price-time priority:**
- Orders at better price execute first
- Among same price, earlier orders execute first

**Pro-rata allocation:**
- At same price, allocate proportionally to quantities
- Used in some futures markets

## Order Book Imbalance

### Definition

Order book imbalance measures the asymmetry between buy and sell pressure:

$$Imbalance = \frac{D^b - D^a}{D^b + D^a}$$

where $D^b$ and $D^a$ are total bid and ask depth.

**Range:** $[-1, 1]$
- $+1$: All buy orders (extreme buying pressure)
- $-1$: All sell orders (extreme selling pressure)
- $0$: Balanced

### As a Signal

Order book imbalance predicts short-term price movements:

**Intuition:**
- High imbalance: More buyers than sellers → price likely to rise
- Low imbalance: More sellers than buyers → price likely to fall

**Empirical evidence:**
- Imbalance predicts next-tick returns
- Effect decays quickly (seconds to minutes)
- Stronger in less liquid stocks

### Volume Imbalance

$$VI = \frac{V^b - V^a}{V^b + V^a}$$

where $V^b$ and $V^a$ are recent buy and sell volumes.

### Price Impact

Price impact relates order flow to price changes:

$$\Delta p = \lambda \times Q$$

where:
- $\Delta p$: price change
- $\lambda$: price impact coefficient (Kyle's lambda)
- $Q$: net order flow (buy volume - sell volume)

## Kyle's Model

### Setup

Kyle (1985) models price formation with:
- One informed trader (knows true value $v$)
- Multiple noise traders
- One market maker (sets prices)

### Timeline

1. Informed trader observes $v \sim N(p_0, \Sigma_0)$
2. Informed trader chooses order $x$
3. Noise traders submit orders $u \sim N(0, \sigma_u^2)$
4. Market maker observes total order flow $y = x + u$
5. Market maker sets price $p = E[v | y]$

### Equilibrium

**Market maker's pricing rule:**
$$p(y) = p_0 + \lambda y$$

where:
$$\lambda = \frac{\Sigma_0}{\Sigma_0 + \sigma_u^2} \times \frac{1}{2\sigma_u}$$

**Informed trader's demand:**
$$x(v) = \beta(v - p_0)$$

where:
$$\beta = \frac{\sigma_u}{\sqrt{\Sigma_0}}$$

**Interpretation:**
- $\lambda$: Price impact (how much price moves per unit order flow)
- Higher $\lambda$: Less liquid market
- $\beta$: Trading intensity (how aggressively informed trader trades)

### Key Results

1. **Price impact:** $\lambda$ increases with information asymmetry ($\Sigma_0$) and decreases with noise trading ($\sigma_u$)

2. **Market depth:** $\frac{1}{\lambda}$ measures market depth (orders per unit price change)

3. **Information efficiency:** Price reveals information: $p = p_0 + \lambda y$ incorporates order flow information

## Glosten-Milgrom Model

### Setup

Glosten-Milgrom (1985) models bid-ask spread with:
- Sequential trades
- Market maker sets bid and ask quotes
- Trades reveal information

### Information Structure

- True value: $v \in \{v_L, v_H\}$ (low or high)
- Informed trader knows $v$
- Uninformed traders trade for liquidity reasons
- Probability of informed trading: $\mu$

### Market Maker's Problem

Market maker sets:
- Bid: $b = E[v | \text{sell}]$
- Ask: $a = E[v | \text{buy}]$

**Bayesian updating:**
- If buy order arrives:
  - With probability $\mu$: Informed trader (knows $v = v_H$)
  - With probability $1-\mu$: Uninformed trader (buy or sell with equal probability)

**Ask price:**
$$a = E[v | \text{buy}] = \mu v_H + (1-\mu) \frac{v_L + v_H}{2}$$

**Bid price:**
$$b = E[v | \text{sell}] = \mu v_L + (1-\mu) \frac{v_L + v_H}{2}$$

### Spread

$$s = a - b = \mu(v_H - v_L)$$

**Components:**
- Spread increases with probability of informed trading ($\mu$)
- Spread increases with information asymmetry ($v_H - v_L$)

**Interpretation:**
- Spread compensates market maker for adverse selection
- No spread from order processing costs (assumed zero)
- No inventory costs (assumed risk-neutral market maker)

## Informed vs Uninformed Traders

### Informed Traders

**Characteristics:**
- Possess private information about asset value
- Trade to profit from information
- Cause adverse selection for market makers

**Trading patterns:**
- Buy when undervalued, sell when overvalued
- Larger positions when information is stronger
- Trade quickly before information becomes public

### Uninformed Traders

**Characteristics:**
- No private information
- Trade for liquidity reasons (rebalancing, consumption)
- Provide liquidity to informed traders

**Types:**
1. **Liquidity traders:** Need to trade for exogenous reasons
2. **Noise traders:** Trade based on sentiment, not fundamentals

### Adverse Selection

Adverse selection occurs when market makers trade with better-informed counterparties.

**Cost:**
- Market makers lose money to informed traders
- Compensated through spread
- Spread = expected loss from informed trading

**Mitigation:**
- Wider spreads
- Order flow analysis
- Limit order usage (let informed traders provide liquidity)

## Example: Order Book Analysis

Consider an order book:

**Bid side:**
- Level 1: \$100.00, 500 shares
- Level 2: \$99.99, 300 shares
- Level 3: \$99.98, 200 shares

**Ask side:**
- Level 1: \$100.05, 400 shares
- Level 2: \$100.06, 350 shares
- Level 3: \$100.07, 250 shares

**Metrics:**
- Mid-price: $m = \frac{100.00 + 100.05}{2} = \$100.025$
- Spread: $s = 100.05 - 100.00 = \$0.05$
- Percentage spread: $s\% = \frac{0.05}{100.025} \times 100 = 0.05\%$

**Depth:**
- Bid depth (3 levels): $500 + 300 + 200 = 1,000$ shares
- Ask depth (3 levels): $400 + 350 + 250 = 1,000$ shares
- Total depth: 2,000 shares

**Imbalance:**
$$Imbalance = \frac{1000 - 1000}{1000 + 1000} = 0$$

The book is balanced.

**Price impact of 1,000 share buy order:**
- Consumes: 400 shares at \$100.05, 350 at \$100.06, 250 at \$100.07
- Average execution price: $\frac{400 \times 100.05 + 350 \times 100.06 + 250 \times 100.07}{1000} = \$100.0595$
- Price impact: $100.0595 - 100.025 = \$0.0345$ (3.45 cents)

The order moves the mid-price up by 3.45 cents, demonstrating temporary market impact.
