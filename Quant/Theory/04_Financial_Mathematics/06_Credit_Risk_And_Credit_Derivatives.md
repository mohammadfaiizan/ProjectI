# Credit Risk and Credit Derivatives

## Credit Risk Fundamentals

### Probability of Default (PD)

The probability of default $PD(t,T)$ is the probability that a borrower defaults between times $t$ and $T$.

**Discrete time:**
$$PD(t,T) = P(\tau \leq T | \tau > t)$$

where $\tau$ is the default time.

**Survival probability:**
$$SP(t,T) = 1 - PD(t,T) = P(\tau > T | \tau > t)$$

### Loss Given Default (LGD)

Loss given default is the percentage of exposure lost in default:

$$LGD = 1 - \text{Recovery Rate}$$

Typically $LGD \approx 0.4$ to $0.6$ (recovery rate 40-60%).

### Exposure at Default (EAD)

Exposure at default is the amount at risk at the time of default. For loans, this is the outstanding principal. For derivatives, it's more complex (depends on mark-to-market).

### Expected Loss

Expected loss is:

$$EL = PD \times LGD \times EAD$$

**Present value:**
$$EL_{PV} = PD \times LGD \times EAD \times D(0,T)$$

where $D(0,T)$ is the discount factor.

### Unexpected Loss

Unexpected loss measures the volatility of losses:

$$UL = EAD \times LGD \times \sqrt{PD(1-PD)}$$

This assumes independence. With correlation, portfolio UL is more complex.

## Structural Models

Structural models derive default from the firm's balance sheet structure.

### Merton Model

The Merton model treats equity as a call option on firm assets.

**Assumptions:**
- Firm has assets $A(t)$ and debt $D$ (zero-coupon bond)
- Default occurs if $A(T) < D$ at maturity $T$
- Asset value follows geometric Brownian motion:
  $$dA(t) = \mu_A A(t)dt + \sigma_A A(t)dW(t)$$

**Default probability:**
$$PD = P(A(T) < D) = N(-d_2)$$

where:
$$d_2 = \frac{\ln(A(0)/D) + (\mu_A - \sigma_A^2/2)T}{\sigma_A\sqrt{T}}$$

**Distance to default:**
$$DD = \frac{\ln(A(0)/D)}{\sigma_A\sqrt{T}}$$

Higher distance to default implies lower default probability.

**Equity as call option:**
$$E(0) = A(0)N(d_1) - De^{-rT}N(d_2)$$

where $E(0)$ is equity value and $d_1 = d_2 + \sigma_A\sqrt{T}$.

**Implied asset volatility:** Can be backed out from equity volatility using:
$$\sigma_E = \frac{A(0)}{E(0)}N(d_1)\sigma_A$$

### KMV Model

KMV (now Moody's KMV) extends Merton:
- Uses market data (equity prices, volatility)
- Estimates asset value and volatility
- Computes distance to default
- Maps to default probability via historical database

**EDF (Expected Default Frequency):** KMV's proprietary default probability measure.

### First-Passage Models

Default occurs when assets hit a barrier $B$ (not just at maturity):

$$\tau = \inf\{t \geq 0 : A(t) \leq B\}$$

This gives higher default probabilities than Merton for short horizons.

## Reduced-Form Models

Reduced-form models model default as an exogenous event with a hazard rate.

### Hazard Rate

The hazard rate $\lambda(t)$ is the instantaneous default probability:

$$\lambda(t) = \lim_{\Delta t \to 0}\frac{P(t < \tau \leq t + \Delta t | \tau > t)}{\Delta t}$$

### Survival Probability

Given hazard rate $\lambda(t)$, survival probability is:

$$SP(0,T) = \exp\left(-\int_0^T \lambda(s)ds\right)$$

For constant hazard rate $\lambda$:
$$SP(0,T) = e^{-\lambda T}$$

### Default Probability

$$PD(0,T) = 1 - \exp\left(-\int_0^T \lambda(s)ds\right)$$

### Default Time Distribution

The default time has density:
$$f(\tau) = \lambda(\tau)\exp\left(-\int_0^{\tau}\lambda(s)ds\right)$$

## Credit Default Swaps (CDS)

A CDS provides insurance against default. The protection buyer pays periodic premiums and receives a payment if default occurs.

### CDS Structure

**Premium leg:** Protection buyer pays $s$ (spread) times notional $N$ periodically
**Protection leg:** Protection seller pays $N \times LGD$ if default occurs

### CDS Pricing

Under risk-neutral measure, value both legs:

**Premium leg PV:**
$$PV_{premium} = sN\sum_{i=1}^{n}\tau_i D(0,t_i)SP(0,t_i)$$

**Protection leg PV:**
$$PV_{protection} = NLGD\int_0^T D(0,t)\lambda(t)SP(0,t)dt$$

For constant hazard rate:
$$PV_{protection} = NLGD\lambda\int_0^T e^{-(r+\lambda)t}dt = NLGD\frac{\lambda}{r+\lambda}(1 - e^{-(r+\lambda)T})$$

**Fair spread:** Set $PV_{premium} = PV_{protection}$ and solve for $s$:

$$s = \frac{LGD\int_0^T D(0,t)\lambda(t)SP(0,t)dt}{\sum_{i=1}^{n}\tau_i D(0,t_i)SP(0,t_i)}$$

For constant hazard rate:
$$s \approx \lambda \times LGD$$

### CDS Basis

CDS basis is the difference between CDS spread and bond spread:
$$\text{Basis} = s_{CDS} - s_{bond}$$

In theory, these should be equal (no-arbitrage). In practice, basis exists due to:
- Liquidity differences
- Funding costs
- Market segmentation

### CDS Curve

CDS spreads vary by maturity, forming a term structure. Can extract hazard rates from CDS curve via bootstrapping.

## Collateralized Debt Obligations (CDO)

CDOs pool credit exposures and tranche them by seniority.

### Structure

**Reference portfolio:** Pool of bonds, loans, or CDS
**Tranches:**
- **Equity (first loss):** Absorbs first losses (e.g., 0-3%)
- **Mezzanine:** Next losses (e.g., 3-7%)
- **Senior:** Last losses (e.g., 7-100%)

### Loss Distribution

Portfolio loss:
$$L_{portfolio} = \sum_{i=1}^{n}LGD_i \times \mathbf{1}_{\tau_i \leq T}$$

Tranche loss:
$$L_{tranche}[a,b] = \min(\max(L_{portfolio} - a, 0), b - a)$$

### Correlation

Default correlation is crucial for CDO pricing:

$$\rho_{ij} = \frac{P(\tau_i \leq T, \tau_j \leq T) - PD_i PD_j}{\sqrt{PD_i(1-PD_i)PD_j(1-PD_j)}}$$

**Gaussian copula model:**
- Correlate asset values (not defaults directly)
- Default when asset value falls below threshold
- Correlation parameter $\rho$ drives tranche values

**Base correlation:** Correlation implied from market tranche prices. Different for each tranche (correlation smile).

### Pricing

Tranche value = Premium leg - Protection leg

Both legs depend on expected tranche loss:
$$\mathbb{E}[L_{tranche}(T)]$$

Computed via:
- Monte Carlo simulation
- Analytical approximations (large portfolio limit)
- Factor models

## Counterparty Risk

Counterparty risk is the risk that a counterparty defaults before fulfilling obligations.

### Credit Valuation Adjustment (CVA)

CVA is the adjustment to derivative value for counterparty credit risk:

$$CVA = -\int_0^T \mathbb{E}[LGD \times \max(V(t), 0) \times \lambda(t)SP(0,t)]D(0,t)dt$$

where $V(t)$ is the derivative value at time $t$ (exposure).

**Simplified (no wrong-way risk):**
$$CVA \approx LGD \times \sum_{i}EE(t_i) \times PD(t_{i-1}, t_i) \times D(0,t_i)$$

where $EE(t_i)$ is expected exposure.

### Debit Valuation Adjustment (DVA)

DVA accounts for one's own credit risk:

$$DVA = \int_0^T \mathbb{E}[LGD_{own} \times \max(-V(t), 0) \times \lambda_{own}(t)SP_{own}(0,t)]D(0,t)dt$$

**Bilateral CVA:**
$$BCVA = CVA - DVA$$

### Expected Exposure

Expected exposure (EE) is:
$$EE(t) = \mathbb{E}[\max(V(t), 0)]$$

**Potential Future Exposure (PFE):** High percentile (e.g., 95%) of exposure distribution.

### Wrong-Way Risk

Wrong-way risk occurs when exposure and counterparty credit quality are negatively correlated (e.g., selling protection on a counterparty).

**Right-way risk:** Positive correlation (e.g., buying protection from a highly rated counterparty).

## Credit Risk Models

### CreditMetrics

- Portfolio model
- Uses asset correlations
- Simulates joint defaults
- Computes portfolio loss distribution

### CreditRisk+

- Actuarial approach
- Models number of defaults (Poisson)
- Severity distribution
- Analytical (no simulation needed)

### KMV Portfolio Manager

- Extends KMV to portfolios
- Uses asset correlations
- Computes portfolio risk metrics

## Regulatory Capital

### Basel Framework

Basel III requires capital for credit risk:

**Standardized approach:** Risk weights based on ratings
**Internal ratings-based (IRB):** Bank estimates PD, LGD, EAD

**Capital requirement:**
$$K = LGD \times N\left(\frac{N^{-1}(PD) + \sqrt{\rho}N^{-1}(0.999)}{\sqrt{1-\rho}}\right) - PD \times LGD$$

where $\rho$ is asset correlation (function of PD).

### Stress Testing

Banks must stress test credit portfolios:
- Adverse scenarios
- Economic downturns
- Sector-specific shocks

## Applications

### Credit Portfolio Management

- Diversification
- Concentration limits
- Risk-return optimization
- Capital allocation

### Trading Strategies

- Relative value (CDS vs bonds)
- Curve trades (steepening/flattening)
- Correlation trades (CDO tranches)
- Capital structure arbitrage

### Risk Management

- Credit limits
- Exposure monitoring
- CVA management
- Collateral management
