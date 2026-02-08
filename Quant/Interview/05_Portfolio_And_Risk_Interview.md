# Portfolio And Risk Interview

## Q1: Explain Markowitz portfolio optimization and its main problems.

**A1:** Markowitz optimization finds the portfolio weights w that maximize expected return μ'w subject to a target risk level (variance w'Σw) or minimize risk subject to a target return. The solution is w* = (μ'Σ⁻¹μ)⁻¹Σ⁻¹μ for the tangency portfolio, or w* = Σ⁻¹1/(1'Σ⁻¹1) for the minimum variance portfolio. Problems include: (1) Estimation error: means and covariances are estimated from historical data and are noisy, especially means, leading to extreme weights and poor out-of-sample performance. (2) Non-stationarity: parameters change over time, so historical estimates may not predict future. (3) Instability: small changes in inputs cause large changes in optimal weights. (4) Concentration: optimal portfolios often concentrate in a few assets. (5) Transaction costs: frequent rebalancing is expensive. Solutions include: shrinkage estimators, robust optimization, Black-Litterman model, factor models, and regularization (e.g., constraints on weights or turnover).

---

## Q2: Derive the CAPM and explain how to test it empirically.

**A2:** CAPM assumes all investors hold the market portfolio and derives from Markowitz optimization. In equilibrium, the market portfolio is mean-variance efficient. For any asset i, E[R_i] = R_f + β_i(E[R_m] - R_f), where β_i = Cov(R_i,R_m)/Var(R_m). Derivation: if the market portfolio is efficient, its Sharpe ratio is maximized. For any asset, adding a small amount doesn't improve the Sharpe ratio, so d/dw[Sharpe] = 0 at w=0. This gives the security market line. Empirical testing uses the regression: R_i - R_f = α_i + β_i(R_m - R_f) + ε_i. CAPM predicts α_i = 0 for all assets. Tests include: (1) Time series: estimate α and test H₀: α = 0. (2) Cross-sectional: regress average returns on betas, test if intercept equals R_f and slope equals market premium. (3) Fama-MacBeth: run cross-sectional regressions each period, test average α. Evidence shows: small stocks and value stocks have positive alphas (anomalies), suggesting CAPM is incomplete. This led to factor models like Fama-French.

---

## Q3: What are the limitations of the Sharpe ratio as a performance measure?

**A3:** The Sharpe ratio SR = (μ - r_f)/σ measures excess return per unit of risk. Limitations: (1) Assumes normal returns: it only uses mean and variance, ignoring skewness and kurtosis. Strategies with occasional large losses (negative skew) may have good Sharpe but high tail risk. (2) Not appropriate for non-normal distributions: options strategies, credit portfolios, or tail-risk strategies have fat tails that Sharpe doesn't capture. (3) Time period dependent: Sharpe varies with measurement period and can be misleading for strategies with time-varying risk. (4) Doesn't account for correlation: a strategy may have good standalone Sharpe but poor diversification benefits. (5) Can be manipulated: by smoothing returns or hiding tail risk. (6) Not a coherent risk measure: doesn't satisfy subadditivity in all cases. Alternatives include: Sortino ratio (uses downside deviation), Calmar ratio (return/max drawdown), or risk-adjusted measures based on VaR/CVaR. For non-normal returns, consider higher moments or full distribution analysis.

---

## Q4: Compare different methods for calculating Value at Risk (VaR).

**A4:** VaR(α) is the loss that will not be exceeded with probability (1-α), i.e., P(Loss ≤ VaR) = 1-α. Methods: (1) Parametric (variance-covariance): assumes returns are multivariate normal. VaR = -μ + z_ασ, where z_α is the α-quantile. For portfolios: VaR = √(w'Σw) * z_α. Fast but assumes normality, missing fat tails. (2) Historical simulation: orders historical returns, takes the (1-α) quantile. No distributional assumptions, captures fat tails, but assumes history repeats and needs sufficient data. (3) Monte Carlo: simulates returns from a model (e.g., multivariate normal, GARCH, copula), computes portfolio returns, takes quantile. Flexible, can incorporate complex dependencies, but model-dependent and computationally intensive. (4) Cornish-Fisher: adjusts normal VaR for skewness and kurtosis using expansion. Hybrid approaches combine methods. Each has trade-offs: parametric is fast but may underestimate tail risk; historical is simple but backward-looking; Monte Carlo is flexible but requires good models.

---

## Q5: What is Conditional Value at Risk (CVaR) and how does it differ from VaR?

**A5:** CVaR (also called Expected Shortfall or Tail VaR) is the expected loss given that the loss exceeds VaR: CVaR(α) = E[Loss | Loss ≥ VaR(α)]. While VaR tells you the threshold, CVaR tells you the average loss beyond that threshold. For continuous distributions, CVaR = (1/α)∫₀ᵅ VaR(u)du. Key differences: (1) Coherence: CVaR is a coherent risk measure (satisfies subadditivity, monotonicity, etc.), while VaR is not coherent (violates subadditivity in some cases). (2) Tail information: CVaR captures the severity of tail losses, not just the threshold. (3) Optimization: CVaR optimization is convex and tractable, while VaR optimization is non-convex. (4) Regulatory: Basel III uses CVaR (Expected Shortfall) for market risk. (5) Interpretation: CVaR answers "how bad can it get on average in the worst α% of cases?" while VaR answers "what's the worst loss in (1-α)% of cases?" For normal distributions, CVaR ≈ VaR * (1 + z_α * φ(z_α)/α), where φ is the normal PDF. CVaR is generally preferred for risk management.

---

## Q6: Why is VaR not a coherent risk measure?

**A6:** A coherent risk measure satisfies: (1) Monotonicity: if X ≤ Y, then ρ(X) ≥ ρ(Y). (2) Translation invariance: ρ(X + c) = ρ(X) - c. (3) Positive homogeneity: ρ(λX) = λρ(X) for λ ≥ 0. (4) Subadditivity: ρ(X + Y) ≤ ρ(X) + ρ(Y). VaR satisfies the first three but violates subadditivity in some cases. Example: two independent assets, each with VaR(0.05) = 100. The portfolio may have VaR > 200 if the joint distribution has tail dependence. This means diversification can appear to increase risk under VaR, which is counterintuitive. The violation occurs because VaR only looks at a quantile and ignores the tail beyond it. For elliptical distributions (like multivariate normal), VaR is subadditive, but for general distributions it's not. This is why regulators moved to Expected Shortfall (CVaR), which is coherent. The lack of subadditivity also makes VaR optimization difficult (non-convex) and can lead to perverse incentives in risk management.

---

## Q7: Explain the intuition behind risk parity portfolios.

**A7:** Risk parity equalizes risk contributions from each asset rather than equalizing dollar allocations. In a traditional 60/40 stock/bond portfolio, stocks contribute ~90% of risk due to higher volatility. Risk parity allocates so each asset contributes equally: RC_i = w_i * (∂σ_p/∂w_i) = w_i * (Σw)_i/σ_p. Setting RC_i = RC_j for all i,j gives w_i ∝ 1/(marginal risk contribution). For uncorrelated assets with equal Sharpe ratios, this reduces to w_i ∝ 1/σ_i (inverse volatility weighting). Intuition: by equalizing risk contributions, the portfolio is more balanced and less dependent on any single asset. Benefits: better diversification, more stable risk profile, often better risk-adjusted returns than market cap weighting. Challenges: requires leverage (low-risk assets like bonds need higher weights to contribute equal risk), sensitive to correlation estimates, and may underperform in strong equity bull markets. Popularized by Bridgewater's All Weather fund, risk parity has become a common institutional strategy.

---

## Q8: Describe the Black-Litterman model and its advantages.

**A8:** Black-Litterman combines market equilibrium (implied from market cap weights) with investor views to form expected returns. Starting point: reverse-engineer implied returns from market portfolio using μ_implied = λΣw_market, where λ is risk aversion. Then incorporate views: if you have a view that portfolio P has return Q with confidence Ω, the Black-Litterman expected returns are: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹μ_implied + P'Ω⁻¹Q]. Here, τ controls confidence in the prior (typically small, e.g., 0.05). Advantages: (1) Stable: starts from equilibrium, so outputs are reasonable even with extreme views. (2) Handles estimation error: shrinks toward market, reducing impact of noisy estimates. (3) Flexible: can incorporate any views (absolute or relative). (4) Intuitive: separates market information from active views. (5) Produces well-diversified portfolios without extreme weights. The model is widely used in practice for asset allocation, especially when combining quantitative models with fundamental views. It addresses the main problem with Markowitz: unstable mean estimates.

---

## Q9: How do you construct a factor model like Fama-French?

**A9:** Factor models decompose returns as: R_i = α_i + β₁ᵢF₁ + β₂ᵢF₂ + ... + βₖᵢFₖ + ε_i, where Fⱼ are factor returns and ε_i is idiosyncratic risk. Fama-French uses: R_i - R_f = α_i + β_mkt(R_m - R_f) + β_smb*SMB + β_hml*HML + ε_i. SMB (Small Minus Big) is the return of small-cap minus large-cap stocks. HML (High Minus Low) is the return of high book-to-market minus low book-to-market stocks. Construction: (1) Form portfolios: sort stocks by size and book-to-market into 2x3 = 6 portfolios. (2) Compute factor returns: SMB = average of small portfolios minus average of large portfolios. HML = average of high B/M minus average of low B/M. (3) Estimate betas: regress stock returns on factors using time series regression. (4) Test: check if α = 0 (factors explain returns) and if factors are priced (cross-sectional regression of average returns on betas). The three-factor model explains much of the cross-section of returns, with value and size effects beyond market beta. Extended versions add momentum, profitability, and investment factors.

---

## Q10: What is the difference between alpha and beta in portfolio management?

**A10:** Beta (β) measures systematic risk: sensitivity to market movements. β = Cov(R_i, R_m)/Var(R_m). A stock with β = 1.5 moves 1.5% for every 1% market move. Beta is not diversifiable and is priced by the market (higher beta → higher expected return via CAPM). Alpha (α) measures risk-adjusted excess return: α = E[R_i] - R_f - β(E[R_m] - R_f). Alpha represents skill: returns beyond what's explained by market exposure. In factor models: R_i = α + β₁F₁ + β₂F₂ + ... + ε, alpha is the intercept. Positive alpha means outperformance after accounting for factor exposures. In practice: beta is typically stable and can be hedged (go short market), while alpha is what active managers seek. Beta is "cheap" (can get via index funds), alpha is "expensive" (requires skill). Decomposing returns: Total Return = Risk-free + Beta Return + Alpha. For hedge funds, alpha generation is the value proposition, though true alpha is rare and may be due to hidden factor exposures or luck.

---

## Q11: Explain tracking error and information ratio.

**A11:** Tracking error (TE) measures deviation from a benchmark: TE = std(R_portfolio - R_benchmark). It's the volatility of active returns. A portfolio with TE = 5% means the difference between portfolio and benchmark returns has 5% annualized volatility. Tracking error comes from active bets: different sector weights, stock selection, or factor tilts. Information ratio (IR) measures risk-adjusted active return: IR = (R_p - R_b)/TE = Active Return / Tracking Error. It's the Sharpe ratio of active returns. A good IR is typically 0.5-1.0 for equity managers. IR = IC * √BR, where IC is information coefficient (correlation between forecasts and returns) and BR is breadth (number of independent bets). To improve IR: increase skill (IC) or increase diversification (BR). Tracking error is managed through constraints: sector neutrality, beta targeting, or factor exposure limits. High tracking error means high active risk, which can lead to significant underperformance. Low tracking error means the portfolio closely follows the benchmark, limiting upside but also downside.

---

## Q12: What is maximum drawdown and why is it important?

**A12:** Maximum drawdown (MDD) is the largest peak-to-trough decline: MDD = max_t [max_{s≤t} R_s - R_t], where R_t is cumulative return. It measures the worst loss from a previous high. For example, if a portfolio goes from $100 to $150, then to $120, the drawdown is (150-120)/150 = 20%. Maximum drawdown is the largest such decline over the entire period. It's important because: (1) Psychological: large drawdowns can cause investor redemptions, even if the strategy eventually recovers. (2) Risk management: it captures tail risk better than volatility for strategies with skewness. (3) Survival: large drawdowns can force liquidation or margin calls. (4) Performance: recovery time matters—a 50% loss requires a 100% gain to break even. (5) Risk-adjusted returns: Calmar ratio = Annual Return / MDD measures return per unit of worst-case risk. Drawdown-based risk management includes: stop-losses, position sizing based on current drawdown, and reducing leverage during drawdowns. Many hedge funds have drawdown limits (e.g., 20%) that trigger risk reduction.

---

## Q13: Describe different approaches to stress testing portfolios.

**A13:** Stress testing evaluates portfolio performance under extreme but plausible scenarios. Approaches: (1) Historical scenarios: apply past crises (2008 financial crisis, dot-com crash, COVID-19) to current portfolio. Revalue all positions using historical price moves. (2) Hypothetical scenarios: design scenarios based on expert judgment (e.g., 30% equity drop, 200bp rate rise, 50% credit spread widening). (3) Factor stress: shock factor exposures (e.g., all equity betas, credit spreads, FX rates) by specified amounts. (4) Monte Carlo stress: simulate extreme tail events using fat-tailed distributions or copulas. (5) Reverse stress testing: find scenarios that cause specific losses (e.g., "what causes a 20% loss?"). (6) Sensitivity analysis: vary one risk factor at a time to see isolated impact. (7) Correlation breakdown: stress correlations toward 1 (diversification fails in crises). Key considerations: ensure scenarios are plausible, cover all risk factors, account for non-linearities (options), include second-order effects (gamma, cross-greeks), and consider liquidity constraints. Stress tests inform risk limits, capital allocation, and hedging strategies.

---

## Q14: Why do correlations break down during market crises?

**A14:** During crises, correlations increase dramatically (often toward 1), causing diversification to fail. Reasons: (1) Common shocks: systemic events (liquidity crises, policy responses) affect all assets simultaneously. (2) Flight to quality: investors sell risky assets and buy safe assets, creating correlated flows. (3) Leverage unwinding: forced deleveraging causes correlated selling across asset classes. (4) Information cascades: panic spreads, making markets move together. (5) Liquidity spirals: illiquidity in one market spills over, causing correlated price moves. (6) Regime changes: crises represent regime shifts where historical correlations don't apply. This is why stress testing uses stressed correlations (e.g., 0.8-0.9 instead of 0.3). The breakdown is asymmetric: correlations spike in downturns but remain low in upturns. This creates tail dependence: extreme losses occur together. Models that assume constant correlations underestimate tail risk. Solutions: use regime-switching models, stress correlations in risk models, hold tail hedges (options, safe assets), and avoid over-reliance on historical diversification benefits.

---

## Q15: What are common portfolio rebalancing strategies?

**A15:** Rebalancing maintains target allocations as prices change. Strategies: (1) Calendar-based: rebalance monthly, quarterly, or annually regardless of drift. Simple but may rebalance unnecessarily or miss large drifts. (2) Threshold-based: rebalance when weights drift beyond bands (e.g., ±5% from target). More efficient, reduces trading costs, but requires monitoring. (3) Risk-based: rebalance when portfolio risk (volatility, VaR) exceeds target. Focuses on risk rather than weights. (4) Optimal control: minimize expected cost (transaction costs + tracking error) using dynamic programming. Theoretically optimal but complex. (5) No rebalancing: let winners run (momentum) or mean-revert naturally. (6) Volatility targeting: adjust leverage to maintain constant portfolio volatility. Considerations: transaction costs (bid-ask spreads, commissions) favor less frequent rebalancing; taxes (capital gains) favor less rebalancing in taxable accounts; momentum vs mean reversion affects optimal frequency; and liquidity constraints may prevent rebalancing. In practice, threshold-based rebalancing with transaction cost considerations is common. The rebalancing frequency affects returns: too frequent increases costs, too infrequent increases tracking error.

---

## Q16: Explain position sizing methods and their trade-offs.

**A16:** Position sizing determines how much capital to allocate to each trade. Methods: (1) Equal weighting: allocate 1/N to each of N positions. Simple but ignores risk differences. (2) Risk parity: size so each position contributes equal risk, w_i ∝ 1/σ_i. Equalizes risk contributions. (3) Volatility targeting: size to achieve target volatility, w_i = (target_vol / σ_i) * (1/portfolio_vol). Maintains constant portfolio risk. (4) Kelly criterion: w* = (p*b - q)/b, where p is win probability, q = 1-p, b is win/loss ratio. Maximizes long-term growth but can be aggressive. Fractional Kelly (e.g., ½ Kelly) is often used. (5) Risk budgeting: allocate risk budget (VaR, CVaR) across positions based on expected risk-adjusted returns. (6) Mean-variance: use Markowitz optimization with constraints. Trade-offs: equal weighting is simple but suboptimal; risk parity is robust but may miss opportunities; Kelly is optimal for growth but risky; volatility targeting stabilizes returns but may limit upside. Considerations include: correlation (diversified positions can be larger), estimation error (shrink aggressive sizes), and leverage constraints. Most practitioners use risk-based sizing with constraints.

---

## Q17: Derive the Kelly criterion and explain its use in portfolio management.

**A17:** Kelly criterion maximizes long-term geometric mean return (or log wealth). For a bet with win probability p, win amount b (on a $1 bet), loss amount 1: if you bet fraction f of wealth, expected log wealth is E[log(W)] = p*log(1 + fb) + (1-p)*log(1 - f). Maximizing: d/df = pb/(1+fb) - (1-p)/(1-f) = 0 gives f* = (pb - (1-p))/b = (p*b - q)/b, where q = 1-p. For continuous returns with mean μ and variance σ², f* = μ/σ² (Sharpe ratio / volatility). Kelly properties: maximizes long-term growth, minimizes time to reach wealth target, but can lead to large drawdowns. In practice, use fractional Kelly (e.g., f*/2 or f*/4) to reduce volatility while capturing most growth benefits. For multiple uncorrelated bets, Kelly allocates f*ᵢ to each. For correlated bets, the multivariate Kelly is f* = Σ⁻¹μ, where Σ is covariance and μ is expected returns. Challenges: requires accurate return estimates (sensitive to estimation error), assumes infinite horizon and no constraints, and can be too aggressive for risk-averse investors. Many hedge funds use Kelly-inspired sizing with risk limits.

---

## Q18: What metrics are used to evaluate hedge fund performance?

**A18:** Key metrics: (1) Return metrics: absolute return, annualized return, excess return vs benchmark. (2) Risk metrics: volatility, maximum drawdown, VaR, CVaR. (3) Risk-adjusted: Sharpe ratio, Sortino ratio (uses downside deviation), Calmar ratio (return/MDD), information ratio. (4) Alpha: regression alpha from factor models (market, size, value, momentum, etc.). True alpha is rare—most "alpha" is exposure to alternative factors. (5) Consistency: percentage of positive months, win rate, average win vs average loss. (6) Tail risk: skewness, kurtosis, tail ratio (average of best 5% vs worst 5% returns). (7) Correlation: correlation to equity markets (low is good for diversification). (8) Capacity: assets under management, whether strategy scales. (9) Fees: management fee (typically 1-2%) and performance fee (typically 20% of profits). Net returns matter. (10) Lock-ups and gates: liquidity terms affect effective returns. No single metric is sufficient—evaluate holistically. Be wary of survivorship bias, backfill bias, and selection bias in databases. Out-of-sample performance and consistency across market regimes are crucial.

---

## Q19: Explain Basel capital requirements for market risk.

**A19:** Basel regulations require banks to hold capital against market risk. Basel I used simple rules (e.g., 8% of risk-weighted assets). Basel II introduced internal models: banks could use VaR models subject to backtesting. Capital = max(VaR_t-1, k * avg(VaR_t-60:t-1), where k ≥ 3 based on backtesting quality. Basel 2.5 added stressed VaR: capital = VaR + Stressed VaR (using crisis period parameters). Basel III (Fundamental Review of the Trading Book, FRTB) uses Expected Shortfall (ES) instead of VaR, computed at 97.5% confidence. Capital = ES_R,S + ES_R,C, where ES_R,S is ES for modellable risks (using internal models) and ES_R,C is for non-modellable risks (using standardized approach). ES must be computed under stressed conditions. The standardized approach provides a floor and is used when internal models aren't approved. Capital requirements ensure banks can absorb losses and maintain solvency. Higher capital reduces leverage and systemic risk but may reduce profitability. Banks must balance regulatory compliance with business objectives.

---

## Q20: How do you measure and manage model risk in quantitative finance?

**A20:** Model risk is the risk of loss from using incorrect or misapplied models. Sources: (1) Model error: wrong assumptions (e.g., normal returns, constant volatility). (2) Parameter error: estimated parameters differ from true values. (3) Implementation error: bugs, numerical issues, data errors. (4) Misapplication: using a model outside its domain (e.g., Black-Scholes for very long maturities). Measurement: (1) Backtesting: compare model predictions to realized outcomes (e.g., VaR exceptions). (2) Stress testing: test models under extreme scenarios. (3) Sensitivity analysis: vary inputs to see output changes. (4) Benchmarking: compare to alternative models or market prices. (5) Out-of-sample testing: validate on data not used for calibration. Management: (1) Model governance: approval processes, documentation, version control. (2) Model validation: independent review, ongoing monitoring. (3) Model diversity: use multiple models, don't rely on one. (4) Conservative assumptions: use stressed parameters, add margins. (5) Limits: cap model-driven positions, require human oversight. (6) Regular updates: recalibrate, retest, retire outdated models. Model risk is particularly high for complex derivatives, new products, or during regime changes. The cost of model errors can be enormous, as seen in the 2008 crisis.

---
