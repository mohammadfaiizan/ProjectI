# Mental Math and Market Sense

## Problem 1: Quick Multiplication Tricks

**Problem 1a:** Multiply 47 × 43

**Technique:** $(a+b)(a-b) = a^2 - b^2$ where $a=45$, $b=2$:
$$47 \times 43 = 45^2 - 2^2 = 2025 - 4 = 2021$$

**Problem 1b:** Multiply 25 × 24

**Technique:** $25 \times 24 = 25 \times (25-1) = 625 - 25 = 600$

Or: $25 \times 24 = 100 \times 6 = 600$ (since $24 = 4 \times 6$)

**Problem 1c:** Square numbers ending in 5

**Technique:** $75^2 = (7 \times 8)$ followed by $25 = 5625$

General: $(10n+5)^2 = 100n(n+1) + 25$

## Problem 2: Percentage Calculations

**Problem 2a:** 15% of 240

**Technique:** $15\% = 10\% + 5\%$
- $10\%$ of 240 = 24
- $5\%$ of 240 = 12
- Total: 36

**Problem 2b:** 23% of 450

**Technique:** $23\% = 20\% + 3\%$
- $20\%$ of 450 = 90
- $3\%$ of 450 = 13.5
- Total: 103.5

**Problem 2c:** What percentage is 18 of 72?

**Technique:** $18/72 = 1/4 = 25\%$

Or: $18/72 = 9/36 = 1/4 = 25\%$

## Problem 3: Ratio Calculations

**Problem 3a:** If A:B = 3:5 and B:C = 4:7, find A:B:C

**Technique:** Make B common:
- A:B = 3:5 = 12:20
- B:C = 4:7 = 20:35
- A:B:C = 12:20:35

**Problem 3b:** Divide 120 in ratio 2:3:5

**Technique:** Total parts = 2+3+5 = 10
- Part 1: $(2/10) \times 120 = 24$
- Part 2: $(3/10) \times 120 = 36$
- Part 3: $(5/10) \times 120 = 60$

## Problem 4: Compound Interest Mental Math

**Problem 4a:** \$1000 at 6% annually for 3 years

**Approximation:** Rule of 72: Doubles in 72/6 = 12 years
- After 3 years: ~1.18× (using $(1.06)^3 \approx 1.19$)
- Value: ~\$1,190

**Exact:** $(1.06)^3 = 1.191016$ → \$1,191.02

**Problem 4b:** \$5000 at 8% compounded quarterly for 2 years

**Quarterly rate:** 8%/4 = 2%
**Periods:** 2 × 4 = 8
**Approximation:** $(1.02)^8 \approx 1.17$ (using $(1+x)^n \approx 1+nx$ for small $x$)
**Value:** ~\$5,850

**More accurate:** $(1.02)^8 = 1.17166$ → \$5,858.30

## Problem 5: Probability Estimation Shortcuts

**Problem 5a:** Probability of at least one head in 3 coin flips

**Technique:** Complement: $1 - P(\text{all tails}) = 1 - (1/2)^3 = 1 - 1/8 = 7/8$

**Problem 5b:** Probability of rolling sum of 7 with two dice

**Technique:** Count favorable: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 ways
Total: 36 ways
Probability: 6/36 = 1/6

## Problem 6: Greeks Intuition

**Problem 6a:** How does delta change as stock price increases for ATM call?

**Answer:** Delta increases (approaches 1 as ITM).

**Intuition:** ATM call has delta ~0.5. As stock rises, becomes more ITM, delta → 1.

**Problem 6b:** How does gamma change as expiration approaches?

**Answer:** Gamma increases (peaks at expiration for ATM options).

**Intuition:** Time value decays, making delta more sensitive to price moves.

**Problem 6c:** If volatility increases, what happens to vega?

**Answer:** Vega typically decreases (diminishing marginal effect).

**Intuition:** Higher vol → option already has high time value → additional vol has less impact.

## Problem 7: Market Scenario Analysis

**Problem 7a:** If Fed raises rates 50bp, what happens to:
- Bond prices: Decrease (inverse relationship)
- Stock prices: Typically decrease (higher discount rate, borrowing costs)
- Dollar: Strengthens (higher yields attract capital)
- Gold: Typically decreases (opportunity cost increases)

**Problem 7b:** If oil prices spike 20%, what happens to:
- Airlines: Negative (fuel costs)
- Energy stocks: Positive (revenue)
- Consumer spending: Negative (discretionary income)
- Inflation: Increases (transportation costs)

## Problem 8: P&L Estimation for Option Positions

**Problem 8a:** Long 100 calls, delta = 0.5. Stock moves +\$2. Approximate P&L?

**Answer:** $\Delta P = \text{delta} \times \text{shares} \times \Delta S = 0.5 \times 100 \times 2 = \$100$

**Note:** Ignores gamma (non-linear), but good first approximation.

**Problem 8b:** Long straddle (call + put), both ATM. Stock moves ±\$5. P&L?

**Answer:** 
- ATM options: delta ≈ 0.5 for call, -0.5 for put
- Net delta ≈ 0 (delta-neutral)
- P&L from gamma: $\frac{1}{2} \times \Gamma \times (\Delta S)^2$
- For ATM: gamma high, so significant gain from move

**Approximation:** If gamma = 0.1 per share, move of \$5:
- P&L ≈ $0.5 \times 0.1 \times 5^2 \times 100 = \$125$ per contract

## Problem 9: Yield Curve Movement Implications

**Problem 9a:** Yield curve steepens (long rates rise more than short). Impact on:
- Banks: Positive (wider net interest margin)
- Long bonds: Negative (price falls)
- Mortgage REITs: Negative (borrowing costs rise)

**Problem 9b:** Yield curve inverts (short > long). What does it signal?

**Answer:** Often signals recession expectation (market expects rates to fall).

**Impact:**
- Financials: Negative (narrow margins)
- Growth stocks: May benefit (lower long-term discount rates)

## Problem 10: Quick Valuation Estimates

**Problem 10a:** Company with P/E = 20, earnings growth = 10%. PEG ratio?

**Answer:** PEG = P/E / Growth = 20 / 10 = 2.0

**Interpretation:** Fairly valued (PEG = 1 is typical benchmark).

**Problem 10b:** Stock trading at \$50, EPS = \$2.50. What's P/E?

**Answer:** P/E = 50 / 2.50 = 20

## Problem 11: Mental Division Tricks

**Problem 11a:** Divide 144 by 12

**Technique:** $144 = 12 \times 12$, so $144 / 12 = 12$

**Problem 11b:** Divide 1,890 by 9

**Technique:** Sum of digits: 1+8+9+0 = 18, divisible by 9
- $1890 = 9 \times 210$
- Answer: 210

**Rule:** Number divisible by 9 if sum of digits divisible by 9.

## Problem 12: Square Root Approximation

**Problem 12a:** Estimate $\sqrt{50}$

**Technique:** $\sqrt{50} = \sqrt{25 \times 2} = 5\sqrt{2} \approx 5 \times 1.414 = 7.07$

**Problem 12b:** Estimate $\sqrt{120}$

**Technique:** $\sqrt{120} = \sqrt{100 \times 1.2} = 10\sqrt{1.2} \approx 10 \times 1.095 = 10.95$

Or: Linear approximation: $\sqrt{121} = 11$, so $\sqrt{120} \approx 10.95$

## Problem 13: Market Cap Quick Calculations

**Problem 13a:** Company with 100M shares, price \$25. Market cap?

**Answer:** 100M × \$25 = \$2.5B

**Problem 13b:** Market cap \$50B, 2B shares. Price per share?

**Answer:** \$50B / 2B = \$25

## Problem 14: Volatility Intuition

**Problem 14a:** Stock with 20% annual vol. Daily vol approximation?

**Answer:** $\sigma_{daily} \approx \sigma_{annual} / \sqrt{252} \approx 20\% / 16 \approx 1.25\%$

**Rule of thumb:** Divide annual vol by 16 for daily.

**Problem 14b:** If daily returns have std dev 2%, annual vol?

**Answer:** $\sigma_{annual} \approx \sigma_{daily} \times \sqrt{252} \approx 2\% \times 16 \approx 32\%$

## Problem 15: Option Pricing Intuition

**Problem 15a:** ATM call, 30 days to expiration, vol = 20%. Rough price?

**Approximation:** Time value ≈ $\sigma \sqrt{T} \times 0.4 \times S$
- $\sigma \sqrt{T} = 0.20 \times \sqrt{30/365} \approx 0.20 \times 0.29 \approx 0.058$
- For \$100 stock: Price ≈ \$5.80

**Black-Scholes:** More accurate, but this gives ballpark.

**Problem 15b:** How does call price change if time to expiration doubles?

**Answer:** Increases (more time value), but not linearly (square root of time).

**Approximation:** If time doubles, price increases by factor $\sqrt{2} \approx 1.41$ (for ATM, all else equal).

## Problem 16: Bond Math Mental Calculations

**Problem 16a:** Bond with 5% coupon, trading at 95. Current yield?

**Answer:** Current yield = Annual coupon / Price = 5 / 95 ≈ 5.26%

**Problem 16b:** Yield to maturity approximation for bond at discount?

**Answer:** YTM > Current yield > Coupon rate (for discount bond).

**Approximation:** YTM ≈ (Coupon + (100-Price)/Years) / ((100+Price)/2)

## Problem 17: Portfolio Return Calculations

**Problem 17a:** Portfolio: 60% stocks (+10%), 40% bonds (+3%). Portfolio return?

**Answer:** $0.6 \times 10\% + 0.4 \times 3\% = 6\% + 1.2\% = 7.2\%$

**Problem 17b:** If portfolio return is 8% and stocks returned 12%, what did bonds return? (60/40 split)

**Answer:** $8\% = 0.6 \times 12\% + 0.4 \times r_b$
$8\% = 7.2\% + 0.4r_b$
$r_b = 2\%$

## Problem 18: Correlation Intuition

**Problem 18a:** If correlation between stocks and bonds is -0.3, and stocks fall 10%, what happens to bonds on average?

**Answer:** Bonds rise approximately $0.3 \times 10\% = 3\%$ (negative correlation).

**Note:** Correlation doesn't imply causation, but gives expected relationship.

## Problem 19: Risk Metrics Quick Estimates

**Problem 19a:** Portfolio with 15% annual return, 20% volatility. Sharpe ratio? (risk-free = 3%)

**Answer:** Sharpe = (15% - 3%) / 20% = 12% / 20% = 0.6

**Problem 19b:** If Information Ratio = 0.5 and tracking error = 4%, what's alpha?

**Answer:** IR = Alpha / Tracking Error
Alpha = 0.5 × 4% = 2%

## Problem 20: Market Regime Changes

**Problem 20a:** If VIX spikes from 15 to 30, what typically happens to:
- Option prices: Increase (higher implied vol)
- Stock prices: Decrease (fear, selling pressure)
- Correlation: Increases (everything moves together)

**Problem 20b:** During market stress, correlations typically:
- Increase (diversification breaks down)
- All assets move together
- Safe havens (bonds, gold) may decouple

**Key insight:** Market regimes matter - relationships change in stress.
