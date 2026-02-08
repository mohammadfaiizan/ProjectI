# Financial Mathematics Interview

## Q1: Derive the Black-Scholes formula step by step.

**A1:** Start with the Black-Scholes-Merton assumptions: geometric Brownian motion dS = μS dt + σS dW, constant risk-free rate r, no dividends, no arbitrage. Apply Ito's lemma to the option value V(S,t) to get dV = (∂V/∂t + μS∂V/∂S + ½σ²S²∂²V/∂S²)dt + σS∂V/∂S dW. Construct a risk-free portfolio: long one option, short Δ = ∂V/∂S shares. The portfolio value is Π = V - ΔS, and dΠ = dV - ΔdS. Substituting and choosing Δ eliminates the dW term, making the portfolio risk-free. By no-arbitrage, dΠ = rΠ dt. This yields the Black-Scholes PDE: ∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² = rV. Solving with boundary condition V(S,T) = max(S-K,0) for a call gives C = S₀N(d₁) - Ke^(-rT)N(d₂), where d₁ = [ln(S₀/K) + (r+σ²/2)T]/(σ√T) and d₂ = d₁ - σ√T.

---

## Q2: Prove put-call parity from first principles.

**A2:** Put-call parity states C - P = S - Ke^(-rT) for European options on non-dividend-paying stocks. Consider two portfolios: Portfolio A holds a call option and Ke^(-rT) cash, Portfolio B holds a put option and one share of stock. At expiration, if S_T > K, Portfolio A is worth (S_T - K) + K = S_T, and Portfolio B is worth 0 + S_T = S_T. If S_T ≤ K, Portfolio A is worth 0 + K = K, and Portfolio B is worth (K - S_T) + S_T = K. Since both portfolios have identical payoffs at expiration, by no-arbitrage they must have the same value today: C + Ke^(-rT) = P + S. Rearranging gives C - P = S - Ke^(-rT). This relationship holds regardless of the underlying price process and is model-independent, making it a fundamental arbitrage relationship used to price puts from calls or vice versa.

---

## Q3: Why is Black-Scholes implied volatility not constant across strikes and maturities?

**A3:** The Black-Scholes model assumes constant volatility, but market prices exhibit volatility smiles and skews: implied volatility varies with strike (moneyness) and maturity. This occurs because real markets violate BS assumptions. Stock returns show fat tails and skewness, not log-normal distributions. Volatility is stochastic, not constant, and jumps occur in prices. The leverage effect causes volatility to increase when prices fall, creating negative skew. Supply and demand imbalances for different strikes create local volatility variations. Market makers charge different premiums for tail risk, reflected in higher implied vol for out-of-the-money puts. The term structure shows volatility clustering: high vol periods are followed by high vol periods. These phenomena are captured by models like local volatility, stochastic volatility (Heston), or jump-diffusion, but the simple BS model cannot reproduce them with a single constant volatility parameter.

---

## Q4: Explain the intuition and formulas for the main Greeks: delta, gamma, theta, vega, and rho.

**A4:** Delta (Δ = ∂V/∂S) measures price sensitivity to underlying moves: for a call, Δ ∈ [0,1], increasing as S increases. It represents the hedge ratio: how many shares to hold to hedge the option. Gamma (Γ = ∂²V/∂S²) measures delta sensitivity, highest for at-the-money options near expiration. It's the "convexity" of the option, crucial for delta hedging P&L. Theta (Θ = -∂V/∂T) measures time decay, typically negative (option loses value over time). Vega (ν = ∂V/∂σ) measures sensitivity to volatility changes, highest for at-the-money options. Rho (ρ = ∂V/∂r) measures interest rate sensitivity, positive for calls, negative for puts. For a call: Δ = N(d₁), Γ = n(d₁)/(Sσ√T), Θ = -Sn(d₁)σ/(2√T) - rKe^(-rT)N(d₂), ν = Sn(d₁)√T, ρ = KTe^(-rT)N(d₂). These Greeks guide hedging strategies and risk management.

---

## Q5: Derive the P&L from delta hedging an option over a short time period.

**A5:** Consider delta hedging a short call option. You sell the call for premium C and buy Δ shares. Over time dt, the option value changes by dC = (∂C/∂t)dt + (∂C/∂S)dS + ½(∂²C/∂S²)(dS)². The hedge portfolio P&L is dC - ΔdS. Substituting and using Δ = ∂C/∂S, the dS terms cancel, leaving dP&L = (∂C/∂t)dt + ½Γ(dS)². The first term is theta decay (negative), and the second is gamma P&L (positive when |dS| is large). Since (dS)² = σ²S²dt in the continuous limit, dP&L = (Θ + ½Γσ²S²)dt. This shows that delta hedging P&L comes from theta (time decay) and gamma (convexity benefit). The Black-Scholes PDE ensures this equals r(C - ΔS)dt = r(hedge portfolio value)dt, so the hedge is risk-free. In practice, discrete hedging introduces tracking error because (dS)² ≠ σ²S²dt exactly.

---

## Q6: What is gamma scalping and how does it work?

**A6:** Gamma scalping exploits the convexity (gamma) of options through dynamic delta hedging. When you're long gamma (e.g., long options), delta increases as the underlying rises and decreases as it falls. If the stock moves up, your delta becomes more positive, so you sell shares at higher prices. If it moves down, delta becomes less positive (or negative), so you buy shares at lower prices. This "buy low, sell high" pattern generates profit from volatility, even if the stock ends where it started. The profit comes from the difference between realized volatility and implied volatility: if realized vol exceeds implied vol, gamma scalping is profitable. The strategy requires frequent rebalancing and incurs transaction costs, so it works best with high volatility and low trading costs. Market makers who are short gamma do the opposite: they lose money from volatility and must charge higher premiums to compensate.

---

## Q7: How do you price barrier options and what makes them challenging?

**A7:** Barrier options pay off only if the underlying price crosses (knock-in) or doesn't cross (knock-out) a barrier level H before expiration. For example, a down-and-out call expires worthless if S_t < H for any t ≤ T. Pricing uses the reflection principle: the probability that a Brownian motion crosses a barrier can be computed using the method of images or solving the Black-Scholes PDE with boundary conditions V(H,t) = 0 for knock-out options. For a down-and-out call with H < K, the price is C_do = C_bs - (H/S)^(2r/σ²-1) * C_bs(H²/S, K), where C_bs is the standard Black-Scholes call price. Challenges include: barrier monitoring (discrete vs continuous), which affects pricing significantly; volatility smile effects are amplified near barriers; Greeks are discontinuous at barriers, making hedging difficult; and path dependency requires Monte Carlo or PDE methods for complex barriers.

---

## Q8: What are the main challenges in pricing Asian options?

**A8:** Asian options have payoffs depending on the average price over the option's life, e.g., max(A_T - K, 0) where A_T = (1/T)∫₀ᵀ S_t dt. The challenge is that the average of log-normal variables is not log-normal, so closed-form solutions don't exist in general. The distribution of the arithmetic average is complex, requiring numerical methods like Monte Carlo simulation, PDE approaches, or moment-matching approximations. Geometric average options are easier because the geometric average of log-normal variables is log-normal, allowing closed-form solutions. For arithmetic averages, common approaches include: Monte Carlo with variance reduction, PDE methods in higher dimensions (time and running average), moment-matching to approximate the distribution, or using control variates. The discrete averaging case (e.g., daily closes) is more common in practice and requires careful handling of the averaging dates and potential early exercise features.

---

## Q9: Compare local volatility models versus stochastic volatility models.

**A9:** Local volatility models assume volatility is a deterministic function of time and spot price: σ(S,t). The Dupire formula extracts local volatility from market prices: σ_loc²(K,T) = 2[∂C/∂T + rK∂C/∂K] / [K²∂²C/∂K²]. This perfectly matches market prices but has limitations: it predicts future smiles incorrectly (smile dynamics are wrong), and volatility tends to decrease when spot rises, which can be unrealistic. Stochastic volatility models assume volatility follows its own stochastic process, e.g., Heston: dV = κ(θ-V)dt + σ_v√V dW_v, with correlation ρ between price and vol shocks. These models capture realistic volatility dynamics and can match current smiles, but calibration is more complex and they may not perfectly fit all market prices. Local vol is often used for exotics pricing when you need exact calibration, while stochastic vol is preferred for understanding vol dynamics and hedging.

---

## Q10: Explain the Heston stochastic volatility model and its key parameters.

**A10:** The Heston model assumes: dS = rS dt + √V S dW₁, dV = κ(θ-V)dt + σ_v√V dW₂, with dW₁dW₂ = ρ dt. Here, V is the variance (volatility squared), κ is the mean reversion speed (how fast V returns to θ), θ is the long-term variance, σ_v is the volatility of volatility (vol-of-vol), and ρ is the correlation between price and volatility shocks. Negative ρ captures the leverage effect (volatility increases when prices fall). The model allows for closed-form option pricing via characteristic functions and Fourier inversion. Key features: variance is always positive (Feller condition: 2κθ ≥ σ_v²), mean reversion prevents variance from exploding, and correlation creates skew. Calibration involves fitting these five parameters to market prices, often using least squares on implied volatilities. The model reproduces volatility smiles and can be extended with jumps.

---

## Q11: When would you use the SABR model and what are its advantages?

**A11:** The SABR (Stochastic Alpha Beta Rho) model is dF = αF^β dW₁, dα = να dW₂, with dW₁dW₂ = ρ dt, where F is the forward rate, α is the volatility-like parameter, β controls the distribution (β=1 is log-normal, β=0 is normal), ν is vol-of-vol, and ρ is correlation. SABR is widely used for interest rate derivatives, especially swaptions and caps/floors, because it models the forward rate directly (natural for rates) and provides an analytic approximation for implied volatility: σ_implied(K) ≈ (α/(F^(1-β))) * [1 + ...]. This makes it fast to calibrate and price. Advantages: simple structure with few parameters, good fit to market smiles, analytic approximations enable fast pricing, and it's become an industry standard for rates. Limitations: the approximation can break down for very low strikes or long maturities, and it doesn't capture all market features. It's less common for equity options where local vol or Heston are preferred.

---

## Q12: Describe different approaches to calibrating volatility models to market data.

**A12:** Calibration finds model parameters that best match market prices (or implied volatilities). Common approaches: (1) Least squares: minimize Σᵢ (σ_model(Kᵢ,Tᵢ) - σ_market(Kᵢ,Tᵢ))² over strikes and maturities. (2) Weighted least squares: weight by liquidity or vega to emphasize liquid options. (3) Regularization: add penalty terms to prevent overfitting, e.g., smoothness constraints on local volatility surfaces. (4) Time series: estimate parameters from historical returns (e.g., GARCH for Heston), though this may not match current market prices. (5) Sequential calibration: fit short maturities first, then longer ones, ensuring consistency. (6) Cross-validation: hold out some strikes/maturities to test out-of-sample fit. Challenges include: many local minima, parameters may be unstable over time, perfect fit doesn't guarantee good pricing for exotics, and there's a trade-off between fit quality and model complexity. Good calibration balances fit with realistic parameter values.

---

## Q13: Explain the volatility smile and why it exists.

**A13:** The volatility smile (or skew) shows that implied volatility varies with strike price, typically higher for out-of-the-money puts (low strikes) and lower for out-of-the-money calls (high strikes). This creates a downward-sloping curve (skew) for equities. Causes include: (1) Leverage effect: when stocks fall, leverage increases, making the firm riskier, so volatility rises. This makes OTM puts more valuable, requiring higher implied vol. (2) Supply and demand: investors buy puts for protection, driving up prices and implied vol. (3) Fat tails: real returns have more extreme moves than log-normal, especially crashes, so tail options need higher vol. (4) Stochastic volatility: actual volatility changes over time, and the market prices this. (5) Jumps: sudden price moves, especially downward, increase the value of OTM puts. The smile is most pronounced for short maturities and fades for longer ones. It's asymmetric for equities (skew) but more symmetric for FX (smile). Understanding the smile is crucial for pricing exotics and managing volatility risk.

---

## Q14: How do you price a bond and what is duration?

**A14:** A bond price is the present value of future cash flows: P = Σᵢ Cᵢe^(-rᵢtᵢ) + Fe^(-rₙT), where Cᵢ are coupon payments, F is face value, and rᵢ are discount rates (often from the yield curve). For a flat yield curve, P = C[1 - e^(-yT)]/y + Fe^(-yT), where y is yield-to-maturity. Duration measures price sensitivity to yield changes. Macaulay duration is D_Mac = (1/P)Σᵢ tᵢCᵢe^(-ytᵢ), a weighted average of cash flow times. Modified duration is D_mod = -dP/dy / P = D_Mac/(1+y), giving the percentage price change for a small yield change: ΔP/P ≈ -D_mod Δy. For zero-coupon bonds, duration equals maturity. Duration increases with maturity and decreases with coupon rate and yield. Convexity (C = d²P/dy² / P) captures second-order effects: ΔP/P ≈ -D_mod Δy + ½C(Δy)². Duration is used for immunization and interest rate risk management.

---

## Q15: Compare the Vasicek and CIR interest rate models.

**A15:** Both are mean-reverting models: Vasicek: dr = κ(θ-r)dt + σ dW, CIR: dr = κ(θ-r)dt + σ√r dW. Key difference: Vasicek has constant volatility σ, while CIR has volatility proportional to √r. This makes CIR rates always positive (if 2κθ ≥ σ², the Feller condition), while Vasicek can go negative, which is unrealistic but sometimes acceptable. Vasicek has closed-form bond prices: P(t,T) = A(t,T)e^(-B(t,T)r), where A and B are known functions. CIR also has closed-form solutions but more complex. Both allow analytic pricing of bond options. Vasicek is simpler and sometimes used for negative rate environments, but CIR is more realistic for positive rates. Calibration: Vasicek has three parameters (κ, θ, σ), CIR has the same three. Both can be extended to multi-factor models. In practice, CIR is more common for positive rate modeling, while Hull-White (time-dependent Vasicek) and other models are also popular.

---

## Q16: What is the difference between forward rates and spot rates?

**A16:** Spot rate r(t,T) is the yield on a zero-coupon bond maturing at T, observed at time t. It's the rate for borrowing/lending from t to T. Forward rate f(t,T₁,T₂) is the rate agreed today (t) for borrowing/lending from T₁ to T₂ in the future. They're related by: (1+r(t,T₂))^(T₂-t) = (1+r(t,T₁))^(T₁-t) * (1+f(t,T₁,T₂))^(T₂-T₁), or in continuous time: f(t,T₁,T₂) = [r(t,T₂)(T₂-t) - r(t,T₁)(T₁-t)]/(T₂-T₁). The instantaneous forward rate is f(t,T) = -∂lnP(t,T)/∂T = r(t,T) + (T-t)∂r(t,T)/∂T. Forward rates are implied by the yield curve and represent market expectations of future rates (plus term premia). They're used to price forward rate agreements (FRAs) and are fundamental in interest rate modeling. The forward curve can differ from expected future spot rates due to risk premia and market expectations.

---

## Q17: How do you price an interest rate swap?

**A17:** An interest rate swap exchanges fixed rate payments for floating rate payments (typically LIBOR or SOFR). To price, value both legs and take the difference. The fixed leg pays C at times tᵢ: V_fixed = CΣᵢ P(0,tᵢ)τᵢ, where P(0,tᵢ) are discount factors and τᵢ are day count fractions. The floating leg pays floating rates: V_float = Σᵢ F(0,tᵢ₋₁,tᵢ)P(0,tᵢ)τᵢ, where F are forward rates. At initiation, the swap rate (par rate) makes V_fixed = V_float, so C = Σᵢ F(0,tᵢ₋₁,tᵢ)P(0,tᵢ)τᵢ / Σⱼ P(0,tⱼ)τⱼ. Alternatively, the floating leg can be valued as: notional * [1 - P(0,T)] using the fact that a floating rate bond trades at par. After initiation, the swap value is the difference between fixed and floating leg values. Swaps are typically quoted as spreads over the risk-free curve, and credit adjustments may be needed for non-collateralized swaps.

---

## Q18: Explain the basics of credit default swaps (CDS).

**A18:** A CDS is insurance against default: the protection buyer pays periodic premiums (spread) to the protection seller, who pays the loss given default if a credit event occurs. The premium leg pays s * N * τ at payment dates, where s is the CDS spread, N is notional, τ is day count. The protection leg pays (1-R)N if default occurs, where R is recovery rate. At initiation, the CDS spread is set so both legs have equal value. Pricing: s = (1-R) * λ / (1 + r + λ), approximately, where λ is the hazard rate (default intensity). More precisely, s = (1-R)∫₀ᵀ P(0,t)λ(t)e^(-∫₀ᵗ λ(s)ds)dt / Σᵢ P(0,tᵢ)τᵢ. CDS spreads reflect market-implied default probabilities and recovery assumptions. They're used for hedging credit risk, speculation, and as indicators of credit quality. Basis risk exists between CDS and bond spreads due to different liquidity, funding costs, and contract terms.

---

## Q19: What is the difference between risk-neutral and real-world probability measures?

**A19:** Risk-neutral measure Q is a probability measure where all assets earn the risk-free rate (discounted prices are martingales). It's used for pricing derivatives via expectation: V = E^Q[e^(-rT)Payoff]. Real-world measure P is the actual probability distribution of asset prices, incorporating risk premia. Under P, risky assets earn expected returns μ > r. The measures are related by the Radon-Nikodym derivative or Girsanov theorem, which changes the drift: dW^Q = dW^P + (μ-r)/σ dt. Risk-neutral pricing works because we can hedge derivatives, creating risk-free portfolios, so risk preferences don't matter. For pricing, use Q. For risk management (VaR, stress testing), use P to model actual losses. For forecasting expected returns, use P. The risk-neutral measure is unique if markets are complete (all payoffs can be replicated), but may not be unique in incomplete markets. Calibration uses Q to match market prices, while historical estimation uses P.

---

## Q20: Derive the Black-Scholes PDE using the hedging argument.

**A20:** Start with a portfolio: long one option V(S,t), short Δ shares. Portfolio value: Π = V - ΔS. Using Ito's lemma: dV = (∂V/∂t + μS∂V/∂S + ½σ²S²∂²V/∂S²)dt + σS∂V/∂S dW. The stock follows dS = μS dt + σS dW. Portfolio change: dΠ = dV - ΔdS = [∂V/∂t + μS∂V/∂S + ½σ²S²∂²V/∂S² - ΔμS]dt + [σS∂V/∂S - ΔσS]dW. Choose Δ = ∂V/∂S to eliminate the dW term (delta hedging), making the portfolio risk-free. Then dΠ = [∂V/∂t + ½σ²S²∂²V/∂S²]dt. By no-arbitrage, a risk-free portfolio must earn the risk-free rate: dΠ = rΠ dt = r(V - ΔS)dt = r(V - S∂V/∂S)dt. Equating: ∂V/∂t + ½σ²S²∂²V/∂S² = r(V - S∂V/∂S). Rearranging gives the Black-Scholes PDE: ∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² = rV. This PDE, with boundary conditions, determines option prices without needing the real-world drift μ, which is why risk-neutral pricing works.

---
