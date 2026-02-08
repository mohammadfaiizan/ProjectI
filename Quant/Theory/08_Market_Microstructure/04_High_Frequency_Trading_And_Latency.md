# High Frequency Trading and Latency

## Introduction

High-frequency trading (HFT) uses sophisticated technology to execute trades in milliseconds or microseconds. Latency—the time delay in trading systems—is critical. This document covers HFT strategies, latency optimization, tick data analysis, and regulatory considerations.

## HFT Strategies

### Market Making

HFT market makers provide liquidity by continuously quoting bid and ask prices.

**Strategy:**
1. Quote tight spreads
2. Update quotes rapidly as market moves
3. Manage inventory through dynamic hedging
4. Profit from spread and rebates

**Requirements:**
- Ultra-low latency
- Real-time risk management
- Co-location

**Risks:**
- Adverse selection
- Inventory risk
- Technology failures

### Statistical Arbitrage

HFT stat arb exploits short-term mispricings:

**Types:**
1. **Pairs trading:** Long-short correlated assets
2. **Index arbitrage:** ETF vs underlying basket
3. **Cross-market arbitrage:** Same asset on different venues

**Example - ETF arbitrage:**
- ETF price: $\$100.10$
- Net asset value (NAV): $\$100.05$
- Arbitrage: Buy NAV components, sell ETF
- Profit: $\$0.05$ per share (before costs)

**Challenges:**
- Execution must be simultaneous
- Costs must be less than mispricing
- Opportunity window is seconds

### Latency Arbitrage

Exploiting speed advantages to trade on information before others:

**Mechanism:**
1. Detect price change on one venue
2. Quickly trade on another venue
3. Profit from temporary mispricing

**Example:**
- Stock price rises on Exchange A
- HFT detects this in 1ms
- Buys on Exchange B (price hasn't updated yet)
- Sells when Exchange B price updates
- Profit from speed advantage

**Controversy:**
- May harm other market participants
- Raises questions about fairness
- Regulatory scrutiny

### Momentum and Order Flow

Trading on short-term momentum signals:

**Signals:**
- Order flow imbalance
- Price trends (microseconds to seconds)
- Volume spikes

**Strategy:**
- Detect momentum early
- Enter position quickly
- Exit before reversal

## Co-Location

### Definition

Co-location places trading servers physically close to exchange matching engines to minimize latency.

### Latency Components

Total latency = Network latency + Processing latency

**Network latency:**
- Propagation delay: $d/c$ where $d$ is distance, $c$ is speed of light
- For 100km: $\approx 0.5$ ms
- For 1km: $\approx 5$ microseconds

**Processing latency:**
- Exchange matching: 10-100 microseconds
- Order processing: 1-10 microseconds
- Network switches: 1-5 microseconds

### Co-Location Benefits

**Typical latencies:**
- Co-located: 10-50 microseconds
- Non co-located: 1-10 milliseconds

**Impact:**
- 100x speed advantage
- Critical for latency-sensitive strategies
- Significant cost (co-location fees)

## Network Latency

### Factors

1. **Distance:** Physical distance to exchange
2. **Network path:** Number of hops, routing
3. **Technology:** Fiber vs copper, network equipment
4. **Protocol:** TCP vs UDP, custom protocols

### Optimization

**Microwave networks:**
- Faster than fiber (speed of light in air > in glass)
- Used for very short distances
- Weather sensitive

**Fiber optimization:**
- Straight-line paths
- Low-latency switches
- Custom protocols

**Network topology:**
- Minimize hops
- Direct connections
- Redundant paths

## Hardware Optimization

### CPUs

**Requirements:**
- High clock speed
- Low latency memory access
- Multiple cores for parallel processing

**Optimizations:**
- CPU pinning (affinity)
- Disable power management
- Custom kernels

### Memory

**Latency hierarchy:**
- L1 cache: ~1 nanosecond
- L2 cache: ~3 nanoseconds
- RAM: ~100 nanoseconds
- Disk: ~10 milliseconds

**Optimization:**
- Keep hot data in cache
- Prefetch data
- Avoid memory allocations in hot path

### Network Cards

**Specialized hardware:**
- FPGA-based network cards
- On-card processing
- Bypass operating system

**Benefits:**
- Sub-microsecond processing
- Deterministic latency
- High throughput

## Tick Data Analysis

### Structure

Tick data contains:
- Timestamp (nanosecond precision)
- Price
- Volume
- Order type (buy/sell)
- Order ID (for order book reconstruction)

### Timestamps

**Precision:**
- Nanoseconds: Modern exchanges
- Microseconds: Common
- Milliseconds: Older systems

**Challenges:**
- Clock synchronization (NTP, PTP)
- Multiple data sources
- Time zone handling

### Order Flow

**Message types:**
1. **New order:** Order enters book
2. **Cancel:** Order cancelled
3. **Trade:** Execution
4. **Modify:** Order quantity/price changed

**Reconstruction:**
- Start with initial order book snapshot
- Apply messages chronologically
- Reconstruct order book at any time

### Message Rates

**Typical rates:**
- Liquid stocks: 10,000-100,000 messages/second
- Less liquid: 100-1,000 messages/second
- Market data: Millions of messages/second (all symbols)

**Challenges:**
- Storage: Terabytes per day
- Processing: Real-time filtering
- Analysis: Efficient algorithms

## Microstructure Noise

### Definition

Microstructure noise is the deviation of observed prices from fundamental value due to:
- Bid-ask bounce
- Discrete prices
- Order flow effects

### Impact on Estimation

**Realized variance:**
Using high-frequency returns:
$$RV = \sum_{i=1}^n r_i^2$$

where $r_i = p_i - p_{i-1}$.

**Problem:** Microstructure noise biases $RV$ upward.

**True variance:** $\sigma^2$
**Observed:** $RV \approx \sigma^2 + 2n \times \text{noise variance}$

### Two-Scale Estimator

Zhang-Mykland-Aït-Sahalia (2005) two-scale estimator:

1. **Averaged estimator:**
$$RV^{(K)} = \frac{1}{K} \sum_{k=1}^K RV_k$$

where $RV_k$ uses every $k$-th observation.

2. **Bias correction:**
$$RV_{TS} = RV^{(K)} - \frac{\bar{n}}{n} RV^{(1)}$$

where $\bar{n} = (n-K+1)/K$.

**Optimal K:** $K^* \propto n^{2/3}$

### Kernel Estimator

Barndorff-Nielsen-Hansen-Lunde-Shephard (2008) kernel estimator:

$$RV_{kernel} = \sum_{i=1}^n r_i^2 + 2 \sum_{h=1}^H w_h \sum_{i=1}^{n-h} r_i r_{i+h}$$

where $w_h$ are kernel weights (e.g., Parzen kernel).

**Properties:**
- Consistent under noise
- Optimal bandwidth: $H \propto n^{1/2}$

## Realized Variance at High Frequency

### Realized Variance

$$RV = \sum_{i=1}^n r_i^2$$

**Convergence:** As sampling frequency increases:
$$RV \xrightarrow{p} \int_0^T \sigma_t^2 dt$$

**Rate:** $O_p(n^{-1/2})$ under no noise

### With Microstructure Noise

**Bias:** $E[RV] = \int_0^T \sigma_t^2 dt + 2n \omega^2$

where $\omega^2$ is noise variance.

**Solution:** Use sparse sampling or noise-robust estimators.

### Pre-Averaging

Podolskij-Vetter (2009) pre-averaging:

1. Pre-average returns:
$$\bar{r}_i = \frac{1}{K} \sum_{j=0}^{K-1} r_{i+j}$$

2. Compute variance:
$$RV_{pre} = \frac{1}{\psi_2} \sum_{i=1}^{n-K+1} \bar{r}_i^2$$

where $\psi_2$ corrects for bias.

**Optimal K:** $K \propto n^{1/2}$

## Regulations

### Reg NMS (US)

**National Market System Regulation:**
- Order protection rule
- Access rule
- Sub-penny rule

**Impact on HFT:**
- Protected quotes
- Access fees
- Decimalization

### MiFID II (Europe)

**Markets in Financial Instruments Directive II:**
- Best execution requirements
- Transaction reporting
- Algorithm testing

**Impact:**
- More transparency
- Algorithm governance
- Cost disclosure

### Circuit Breakers

**Purpose:** Halt trading during extreme volatility.

**Types:**
1. **Price limits:** Trading halts if price moves X%
2. **Volatility halts:** Based on recent volatility
3. **Market-wide:** All markets halt

**Impact on HFT:**
- Can't trade during halts
- Must manage positions
- Liquidity dries up

### Market Maker Obligations

Some exchanges require market makers to:
- Quote continuously
- Maintain maximum spreads
- Provide minimum depth

**Benefits:**
- Ensures liquidity
- Reduces volatility

**Costs:**
- Obligations during stress
- Capital requirements

## Flash Crashes and Market Stability

### Flash Crash (May 6, 2010)

**Events:**
1. Large sell order executed via algorithm
2. HFTs withdrew liquidity
3. Prices fell rapidly
4. Circuit breakers triggered
5. Prices recovered quickly

**Lessons:**
- HFT can amplify shocks
- Liquidity can disappear quickly
- Need better safeguards

### Mechanisms

**Liquidity withdrawal:**
- HFTs detect unusual activity
- Withdraw quotes to avoid losses
- Liquidity disappears
- Prices gap

**Feedback loops:**
1. Price falls
2. HFTs reduce positions
3. Less liquidity
4. Prices fall more
5. Loop continues

### Mitigation

**Circuit breakers:**
- Limit up/down
- Trading halts
- Volatility pauses

**Market maker obligations:**
- Continuous quoting
- Maximum spreads

**Order types:**
- Stop-loss limits
- Price limits

**Monitoring:**
- Real-time surveillance
- Anomaly detection
- Intervention protocols

## Example: Latency Calculation

Consider an HFT system:

**Components:**
- Signal detection: 5 microseconds
- Strategy logic: 10 microseconds
- Risk checks: 5 microseconds
- Order generation: 2 microseconds
- Network (co-located): 15 microseconds
- Exchange processing: 20 microseconds

**Total latency:** 5 + 10 + 5 + 2 + 15 + 20 = 57 microseconds

**For non co-located:**
- Network: 2 milliseconds (2,000 microseconds)
- Total: ~2,057 microseconds

**Speed advantage:** 2,057 / 57 ≈ 36x faster

**Impact on arbitrage:**
- Mispricing: $\$0.01$ per share
- Opportunity duration: 100 microseconds
- Co-located: Can capture (57 < 100)
- Non co-located: Cannot capture (2,057 > 100)

Co-location is essential for capturing short-lived opportunities.
