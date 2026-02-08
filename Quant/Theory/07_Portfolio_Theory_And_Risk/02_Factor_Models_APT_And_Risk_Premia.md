# Factor Models, APT, and Risk Premia

## Introduction

Factor models provide a framework for understanding asset returns through exposure to common risk factors. The Arbitrage Pricing Theory (APT) offers an alternative to CAPM, allowing multiple risk factors. Factor investing has become central to quantitative finance, with systematic strategies targeting specific risk premia.

## Arbitrage Pricing Theory

### Derivation

APT assumes that asset returns follow a factor structure:

$$R_i = E[R_i] + \sum_{j=1}^k \beta_{ij} F_j + \epsilon_i$$

where:
- $F_j$ are common risk factors with $E[F_j] = 0$
- $\beta_{ij}$ is the sensitivity of asset $i$ to factor $j$
- $\epsilon_i$ is asset-specific noise with $E[\epsilon_i] = 0$ and $\text{Cov}(\epsilon_i, F_j) = 0$

The key assumption is that $\epsilon_i$ are uncorrelated across assets: $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$.

### No-Arbitrage Condition

Consider a portfolio with weights $\boldsymbol{w}$ such that:
- Zero investment: $\sum_{i=1}^n w_i = 0$
- Zero factor exposure: $\sum_{i=1}^n w_i \beta_{ij} = 0$ for all $j$

The portfolio return is:
$$R_p = \sum_{i=1}^n w_i R_i = \sum_{i=1}^n w_i E[R_i] + \sum_{i=1}^n w_i \epsilon_i$$

As $n \to \infty$, diversification eliminates idiosyncratic risk:
$$\text{Var}(R_p) = \sum_{i=1}^n w_i^2 \sigma_{\epsilon_i}^2 \to 0$$

For no arbitrage, this riskless portfolio must have zero expected return:
$$\sum_{i=1}^n w_i E[R_i] = 0$$

Using Farkas' lemma, this implies:
$$E[R_i] = \lambda_0 + \sum_{j=1}^k \beta_{ij} \lambda_j$$

where:
- $\lambda_0$ is the risk-free rate (or zero-beta return)
- $\lambda_j$ is the risk premium for factor $j$

### Factor Structure

The APT pricing equation is:
$$E[R_i] = r_f + \sum_{j=1}^k \beta_{ij} \lambda_j$$

This is a multi-factor extension of CAPM. If the market portfolio is the only factor, APT reduces to CAPM.

### Identification of Factors

Factors can be:
1. **Macroeconomic factors**: GDP growth, inflation, interest rates
2. **Statistical factors**: Principal components of returns
3. **Fundamental factors**: Size, value, momentum, profitability

The choice depends on the application and data availability.

## Fama-French Factor Models

### Three-Factor Model

Fama and French (1993) proposed a three-factor model:

$$R_i - r_f = \alpha_i + \beta_i (R_m - r_f) + s_i SMB + h_i HML + \epsilon_i$$

where:
- $R_m - r_f$: market excess return
- $SMB$: Small Minus Big (size factor)
- $HML$: High Minus Low (value factor)

**SMB Construction:**
1. Sort stocks by market cap into Small (S) and Big (B)
2. Sort by book-to-market into Low (L), Medium (M), High (H)
3. Form six portfolios: SL, SM, SH, BL, BM, BH
4. $SMB = \frac{1}{3}(SL + SM + SH) - \frac{1}{3}(BL + BM + BH)$

**HML Construction:**
1. Use the same six portfolios
2. $HML = \frac{1}{2}(SH + BH) - \frac{1}{2}(SL + BL)$

### Five-Factor Model

Fama and French (2015) extended to five factors:

$$R_i - r_f = \alpha_i + \beta_i (R_m - r_f) + s_i SMB + h_i HML + r_i RMW + c_i CMA + \epsilon_i$$

where:
- $RMW$: Robust Minus Weak (profitability factor)
- $CMA$: Conservative Minus Aggressive (investment factor)

**RMW Construction:**
1. Sort by operating profitability (OP)
2. $RMW = \frac{1}{2}(High OP portfolios) - \frac{1}{2}(Low OP portfolios)$

**CMA Construction:**
1. Sort by asset growth (investment)
2. $CMA = \frac{1}{2}(Conservative portfolios) - \frac{1}{2}(Aggressive portfolios)$

### Momentum Factor

Carhart (1997) added momentum:

$$R_i - r_f = \alpha_i + \beta_i (R_m - r_f) + s_i SMB + h_i HML + m_i MOM + \epsilon_i$$

**MOM Construction:**
1. Sort stocks by past 12-month return (skipping the most recent month)
2. $MOM = \frac{1}{2}(Winners) - \frac{1}{2}(Losers)$

Momentum is typically calculated with a 1-month skip to avoid microstructure effects.

## Factor Investing

### Factor Construction

Factors are constructed through:

1. **Univariate sorts**: Rank by a single characteristic
2. **Bivariate sorts**: Control for one factor while sorting by another
3. **Multivariate sorts**: Multiple characteristics simultaneously

**Example: Value factor with size control:**
- Sort into size terciles
- Within each tercile, sort by book-to-market
- Form value portfolios within each size group
- Combine across size groups

### Rebalancing

Factor portfolios require periodic rebalancing:

**Frequency considerations:**
- Monthly: Common for academic factors
- Quarterly: Reduces turnover
- Annual: Lower transaction costs but slower factor capture

**Turnover management:**
- Buffer zones to reduce unnecessary trades
- Transaction cost models in optimization
- Gradual transitions rather than discrete jumps

### Factor Timing

Factor timing attempts to predict when factors will outperform:

**Approaches:**
1. **Macroeconomic indicators**: Economic cycles, interest rates
2. **Valuation metrics**: Factor spreads, relative valuations
3. **Momentum signals**: Recent factor performance
4. **Regime models**: Hidden Markov models, regime-switching

**Challenges:**
- Factor timing is notoriously difficult
- Transaction costs can erode benefits
- Overfitting risk

### Factor Decay

Factor decay refers to the diminishing returns of factor strategies over time:

**Causes:**
1. **Arbitrage**: As factors become known, they get traded away
2. **Crowding**: Too many investors pursuing the same strategy
3. **Structural changes**: Market evolution, regulation
4. **Data mining**: Factors may have been spurious

**Evidence:**
- Many academic factors show declining alphas post-publication
- Some factors persist (value, momentum) while others fade
- Implementation matters: gross vs net returns

### Factor Crowding

Crowding occurs when too many investors hold similar positions:

**Signs:**
- High correlation between factor portfolios
- Declining Sharpe ratios
- Increased volatility during stress periods
- Difficulty executing trades

**Measurement:**
- Active share: deviation from benchmark
- Crowding score: similarity to other funds
- Liquidity metrics: bid-ask spreads, market impact

## Risk Premia

### Equity Risk Premium

The equity risk premium (ERP) is the excess return of stocks over risk-free bonds:

$$ERP = E[R_m] - r_f$$

**Historical estimates:**
- US: 6-7% annually (long-term)
- Varies by country and time period
- Subject to survivorship bias

**Forward-looking estimates:**
- Dividend discount model: $ERP = \frac{D_1}{P_0} + g - r_f$
- Earnings yield: $ERP = E/P - r_f$
- Survey-based: Analyst expectations

### Term Premium

The term premium is the excess return of long-term bonds over short-term bonds:

$$TP = E[R_{long}] - r_f$$

**Theories:**
- Liquidity preference: Investors demand compensation for illiquidity
- Expectations hypothesis: $TP = 0$ (contested)
- Risk premium: Compensation for interest rate risk

**Measurement:**
- Yield curve slope: $10Y - 2Y$ spread
- Forward rates vs realized rates
- Affine term structure models

### Credit Premium

The credit premium compensates for default risk:

$$CP = E[R_{credit}] - E[R_{govt}]$$

where $R_{credit}$ and $R_{govt}$ are returns on credit and government bonds of similar maturity.

**Components:**
- Expected loss: $EL = PD \times LGD$
- Risk premium: Compensation for uncertainty
- Liquidity premium: Illiquidity of credit markets

**Models:**
- Merton model: Structural approach
- Reduced-form models: Intensity-based
- Credit spreads: Market-implied default probabilities

### Volatility Premium

The volatility premium is the difference between implied and realized volatility:

$$VP = \sigma_{implied} - \sigma_{realized}$$

**Sources:**
- Demand for insurance: Investors pay for downside protection
- Volatility risk premium: Compensation for volatility risk
- Supply constraints: Limited ability to sell volatility

**Trading:**
- Selling options: Collect premium, bear volatility risk
- Variance swaps: Direct volatility exposure
- VIX strategies: Long/short volatility

### Liquidity Premium

The liquidity premium compensates for illiquidity:

$$LP = E[R_{illiquid}] - E[R_{liquid}]$$

**Components:**
- Transaction costs: Bid-ask spreads, market impact
- Holding period risk: Difficulty exiting positions
- Funding costs: Margin requirements, capital constraints

**Measurement:**
- Bid-ask spreads
- Amihud illiquidity measure: $\frac{|R|}{Volume}$
- Roll measure: Covariance of returns
- Price impact: Kyle's lambda

## Smart Beta vs Alpha

### Smart Beta

Smart beta strategies systematically capture factor exposures:

**Characteristics:**
- Rules-based, transparent
- Low cost relative to active management
- Factor tilts: value, momentum, quality, low volatility
- Rebalancing discipline

**Examples:**
- Equal-weight indices
- Fundamental indexing (by sales, earnings, etc.)
- Minimum volatility portfolios
- Factor ETFs

### Alpha

Alpha represents skill-based excess returns:

**Definition:**
$$\alpha = E[R_p] - E[R_p^{benchmark}]$$

After controlling for factor exposures:
$$\alpha = E[R_p] - \left[r_f + \sum_{j=1}^k \beta_j \lambda_j\right]$$

**Sources:**
- Security selection: Picking winners within factors
- Factor timing: Dynamic factor allocation
- Alternative data: Non-traditional information
- Market microstructure: Exploiting inefficiencies

### The Blurring Line

The distinction between smart beta and alpha is increasingly blurred:

- Many "alpha" strategies are factor exposures
- Smart beta requires implementation skill
- Factor timing can be systematic or discretionary
- Data and technology advantages create alpha

## Example: Factor Model Estimation

Consider estimating a three-factor model for a stock:

$$R_i - r_f = \alpha_i + \beta_i (R_m - r_f) + s_i SMB + h_i HML + \epsilon_i$$

Using 60 months of data, we estimate:

| Coefficient | Estimate | Std Error | t-stat |
|-------------|----------|-----------|--------|
| $\alpha_i$  | 0.002    | 0.003     | 0.67   |
| $\beta_i$   | 1.15     | 0.05      | 23.0   |
| $s_i$       | 0.30     | 0.08      | 3.75   |
| $h_i$       | 0.45     | 0.09      | 5.0    |

Interpretation:
- The stock has market beta of 1.15 (15% more volatile than market)
- Positive size loading (0.30): tilts toward small caps
- Positive value loading (0.45): tilts toward value stocks
- Alpha is not statistically significant (t = 0.67)

The expected excess return is:
$$E[R_i - r_f] = 0.002 + 1.15 \times ERP + 0.30 \times E[SMB] + 0.45 \times E[HML]$$

If $ERP = 0.06$, $E[SMB] = 0.02$, $E[HML] = 0.04$:
$$E[R_i - r_f] = 0.002 + 1.15 \times 0.06 + 0.30 \times 0.02 + 0.45 \times 0.04 = 0.083$$

The stock is expected to outperform the risk-free rate by 8.3% annually, with 6.9% coming from factor exposures and 0.2% from alpha.
