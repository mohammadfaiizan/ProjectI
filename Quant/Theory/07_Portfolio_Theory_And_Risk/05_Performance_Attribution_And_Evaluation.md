# Performance Attribution and Evaluation

## Introduction

Performance attribution decomposes portfolio returns into sources of excess return, enabling understanding of what drove performance. Performance evaluation assesses whether returns represent skill or luck. This document covers attribution methodologies, risk-adjusted metrics, and statistical evaluation techniques.

## Returns-Based Analysis

### CAPM Alpha

The single-factor model:
$$R_p - r_f = \alpha_p + \beta_p (R_m - r_f) + \epsilon_p$$

where:
- $\alpha_p$: CAPM alpha (excess return not explained by market exposure)
- $\beta_p$: market beta
- $\epsilon_p$: residual return

**Estimation:** OLS regression:
$$\hat{\alpha}_p = \bar{R}_p - r_f - \hat{\beta}_p (\bar{R}_m - r_f)$$

**Standard error:**
$$SE(\hat{\alpha}_p) = \frac{\sigma(\epsilon_p)}{\sqrt{T}} \sqrt{1 + \frac{\bar{R}_m^2}{\sigma^2(R_m)}}$$

**t-statistic:**
$$t(\alpha_p) = \frac{\hat{\alpha}_p}{SE(\hat{\alpha}_p)}$$

For significance at 5% level, need $|t| > 1.96$ (or 2.0 approximately).

### Multi-Factor Alpha

Using Fama-French three-factor model:
$$R_p - r_f = \alpha_p + \beta_p (R_m - r_f) + s_p SMB + h_p HML + \epsilon_p$$

Alpha now represents excess return after controlling for market, size, and value exposures.

**Interpretation:**
- Positive alpha: outperformance after risk adjustment
- Negative alpha: underperformance
- Alpha near zero: returns explained by factor exposures

### Information Ratio

The Information Ratio measures risk-adjusted active return:
$$IR = \frac{\alpha_p}{\sigma(\epsilon_p)} = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)}$$

where $R_b$ is the benchmark return.

**Relationship to t-statistic:**
$$t(\alpha_p) = IR \times \sqrt{T}$$

For $T = 60$ months, $IR = 0.5$ gives $t = 0.5 \times \sqrt{60} = 3.87$ (highly significant).

**Interpretation:**
- $IR > 0.5$: Good
- $IR > 0.75$: Very good
- $IR > 1.0$: Excellent

## Brinson-Fachler Attribution

### Framework

Brinson-Fachler decomposes excess return into:
1. **Allocation effect:** Sector/asset class selection
2. **Selection effect:** Security selection within sectors
3. **Interaction effect:** Combined effect

### Mathematical Decomposition

For portfolio $p$ and benchmark $b$:

**Portfolio return:**
$$R_p = \sum_{i=1}^n w_{p,i} R_{p,i}$$

**Benchmark return:**
$$R_b = \sum_{i=1}^n w_{b,i} R_{b,i}$$

**Excess return:**
$$R_p - R_b = \sum_{i=1}^n (w_{p,i} R_{p,i} - w_{b,i} R_{b,i})$$

**Decomposition:**
$$R_p - R_b = \sum_{i=1}^n (w_{p,i} - w_{b,i}) R_{b,i} + \sum_{i=1}^n w_{b,i} (R_{p,i} - R_{b,i}) + \sum_{i=1}^n (w_{p,i} - w_{b,i}) (R_{p,i} - R_{b,i})$$

Terms:
1. **Allocation:** $\sum_i (w_{p,i} - w_{b,i}) R_{b,i}$
   - Over/underweighting sectors that outperform/underperform
2. **Selection:** $\sum_i w_{b,i} (R_{p,i} - R_{b,i})$
   - Security selection within sectors
3. **Interaction:** $\sum_i (w_{p,i} - w_{b,i}) (R_{p,i} - R_{b,i})$
   - Combined effect of allocation and selection

### Example

Consider two sectors with:

| Sector | Portfolio Weight | Benchmark Weight | Portfolio Return | Benchmark Return |
|--------|-----------------|------------------|------------------|------------------|
| Tech   | 0.60            | 0.40             | 0.15             | 0.12             |
| Finance| 0.40            | 0.60             | 0.08             | 0.10             |

Portfolio return: $R_p = 0.60 \times 0.15 + 0.40 \times 0.08 = 0.122$
Benchmark return: $R_b = 0.40 \times 0.12 + 0.60 \times 0.10 = 0.108$
Excess return: $0.122 - 0.108 = 0.014$ (1.4%)

**Decomposition:**
- Allocation: $(0.60-0.40) \times 0.12 + (0.40-0.60) \times 0.10 = 0.004$
- Selection: $0.40 \times (0.15-0.12) + 0.60 \times (0.08-0.10) = 0.000$
- Interaction: $(0.60-0.40) \times (0.15-0.12) + (0.40-0.60) \times (0.08-0.10) = 0.010$

Total: $0.004 + 0.000 + 0.010 = 0.014$ ✓

The excess return came primarily from:
1. Overweighting Tech (allocation: +0.4%)
2. Interaction of overweight and outperformance (+1.0%)

## Fixed-Income Attribution

### Yield Curve Effect

Decompose bond return:
$$R = R_{yield} + R_{spread} + R_{currency}$$

**Yield curve effect:**
$$R_{yield} = -D \times \Delta y + \frac{1}{2} C \times (\Delta y)^2$$

where:
- $D$: duration
- $C$: convexity
- $\Delta y$: yield change

**Decomposition:**
- **Level:** Parallel shift
- **Slope:** Steepening/flattening
- **Curvature:** Change in curvature

### Spread Effect

$$R_{spread} = -D_{spread} \times \Delta s$$

where:
- $D_{spread}$: spread duration
- $\Delta s$: spread change

**Components:**
- Credit spread: Default risk premium
- Liquidity spread: Illiquidity premium
- Option-adjusted spread: Embedded option value

### Currency Effect

For international bonds:
$$R_{currency} = \frac{S_t - S_{t-1}}{S_{t-1}}$$

where $S_t$ is the exchange rate (foreign currency per domestic).

**Hedged return:**
$$R_{hedged} = R_{local} + R_{currency} - R_{hedge\ cost}$$

## Risk-Adjusted Metrics

### Sharpe Ratio

$$SR = \frac{E[R_p] - r_f}{\sigma(R_p)}$$

**Annualization:** For monthly returns:
$$SR_{annual} = SR_{monthly} \times \sqrt{12}$$

**Interpretation:**
- $SR < 0.5$: Poor
- $0.5 \leq SR < 1.0$: Acceptable
- $1.0 \leq SR < 2.0$: Good
- $SR \geq 2.0$: Excellent

### Information Ratio

$$IR = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)}$$

Measures active return per unit of tracking error.

### Calmar Ratio

$$Calmar = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}$$

Focuses on downside risk. Higher is better.

### Sterling Ratio

$$Sterling = \frac{\text{Annualized Return}}{\text{Average Maximum Drawdown}}$$

Uses average of worst drawdowns over rolling periods.

### Omega Ratio

Omega measures the probability-weighted ratio of gains to losses:

$$\Omega(K) = \frac{\int_K^\infty (1-F(x))dx}{\int_{-\infty}^K F(x)dx}$$

where $F$ is the return distribution and $K$ is a threshold (often 0).

**Discrete version:**
$$\Omega(K) = \frac{\sum_{R_i > K} (R_i - K)}{\sum_{R_i < K} (K - R_i)}$$

Omega > 1 indicates more probability mass above threshold than below.

## Drawdown Analysis

### Maximum Drawdown

Maximum drawdown is the largest peak-to-trough decline:

$$MDD = \max_{t \in [0,T]} \left[ \max_{s \in [0,t]} V_s - V_t \right]$$

where $V_t$ is portfolio value at time $t$.

**Percentage:**
$$MDD\% = \frac{MDD}{\max_{s \in [0,t]} V_s}$$

### Drawdown Duration

Time from peak to recovery:
$$DD_{duration} = \inf\{t > t_{peak} : V_t \geq V_{peak}\}$$

### Recovery Time

Time to recover from maximum drawdown.

### Average Drawdown

$$ADD = \frac{1}{N} \sum_{i=1}^N DD_i$$

where $DD_i$ are individual drawdowns.

### Ulcer Index

$$UI = \sqrt{\frac{1}{T} \sum_{t=1}^T D_t^2}$$

where $D_t$ is drawdown at time $t$ (distance from recent peak).

## Strategy Capacity Estimation

### Definition

Capacity is the maximum assets under management (AUM) before returns degrade significantly.

### Factors Affecting Capacity

1. **Market liquidity:** Average daily volume
2. **Position sizing:** Maximum position as % of volume
3. **Market impact:** Cost of trading
4. **Strategy type:** 
   - Market making: High capacity
   - Statistical arbitrage: Medium capacity
   - Event-driven: Low capacity

### Estimation Methods

**Trading volume approach:**
$$Capacity = \sum_{i=1}^n \min(w_i \times AUM, \alpha \times ADV_i)$$

where:
- $w_i$: optimal weight in asset $i$
- $ADV_i$: average daily volume
- $\alpha$: maximum % of volume (typically 5-20%)

**Market impact approach:**
Estimate capacity where market impact equals expected alpha:
$$Impact(Capacity) = \alpha$$

Using square-root law:
$$Impact = \sigma \sqrt{\frac{Q}{ADV}}$$

where $Q$ is trade size.

**Empirical approach:**
- Run strategy with increasing AUM
- Measure degradation in Sharpe ratio or alpha
- Capacity = AUM where Sharpe drops by X%

## Statistical Significance of Performance

### Bootstrap Test

Test if observed performance could occur by chance:

1. **Null hypothesis:** $H_0: \alpha = 0$ (no skill)

2. **Bootstrap procedure:**
   - Generate $B$ bootstrap samples by resampling returns
   - For each sample $b$, calculate $\alpha^{(b)}$
   - Distribution of $\{\alpha^{(b)}\}$ under $H_0$

3. **p-value:**
$$p = \frac{1}{B} \sum_{b=1}^B \mathbb{1}(|\alpha^{(b)}| \geq |\hat{\alpha}|)$$

4. **Decision:** Reject $H_0$ if $p < 0.05$

### Luck vs Skill

**False discovery rate (FDR):**
In a universe of $N$ strategies, some will have positive alpha by luck.

**Benjamini-Hochberg procedure:**
1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(N)}$
2. Find largest $k$ such that: $p_{(k)} \leq \frac{k}{N} \alpha$
3. Reject hypotheses 1 through $k$

Controls FDR at level $\alpha$.

### Multiple Testing Correction

**Bonferroni correction:**
Reject $H_0$ only if $p < \frac{\alpha}{N}$

Very conservative, controls family-wise error rate.

**FDR control:**
Less conservative, controls expected proportion of false discoveries.

### Persistence Testing

Test if performance persists over time:

**Split-sample test:**
- Period 1: Estimate alpha
- Period 2: Test if alpha persists

**Regression:**
$$\alpha_{t+1} = \beta_0 + \beta_1 \alpha_t + \epsilon_t$$

Test $H_0: \beta_1 = 0$ (no persistence).

**Rank correlation:**
Spearman correlation of alphas across periods.

### Survivorship Bias

Survivorship bias occurs when only surviving funds are analyzed.

**Impact:**
- Overestimates average returns
- Underestimates risk
- Biases performance metrics upward

**Adjustment:**
- Include dead funds in sample
- Use databases with full history
- Account for fund closures

## Example: Performance Evaluation

Consider a fund with 60 months of returns:

**Summary statistics:**
- Mean monthly return: 1.2%
- Standard deviation: 3.5%
- Risk-free rate: 0.3% (monthly)
- Benchmark return: 0.9% (monthly)
- Benchmark volatility: 2.8%

**Metrics:**
- Excess return: $1.2\% - 0.9\% = 0.3\%$ monthly
- Tracking error: $\sigma(R_p - R_b) = 2.1\%$
- Information Ratio: $IR = 0.3\% / 2.1\% = 0.143$
- Sharpe ratio: $SR = (1.2\% - 0.3\%) / 3.5\% = 0.257$

**CAPM regression:**
$$R_p - r_f = \alpha + \beta (R_m - r_f) + \epsilon$$

Results:
- $\hat{\alpha} = 0.15\%$ monthly
- $\hat{\beta} = 1.12$
- $SE(\hat{\alpha}) = 0.27\%$
- $t(\alpha) = 0.15 / 0.27 = 0.56$ (not significant)

**Interpretation:**
- Positive alpha but not statistically significant
- $IR = 0.143$ is low (need > 0.5 for good performance)
- Performance could be due to luck
- Need longer history or higher alpha to conclude skill

**Bootstrap test:**
- Generate 10,000 bootstrap samples
- Calculate alpha for each
- p-value = proportion with $|\alpha| \geq 0.15\%$
- Result: $p = 0.62$ (not significant)

Conclusion: Cannot reject null hypothesis of no skill. Performance is consistent with luck.
