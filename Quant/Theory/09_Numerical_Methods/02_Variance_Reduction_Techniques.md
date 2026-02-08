# Variance Reduction Techniques

## Introduction

Monte Carlo simulation can have high variance, requiring many samples for accurate estimates. Variance reduction techniques reduce the number of samples needed by exploiting problem structure, correlations, and analytical results. This document covers major variance reduction methods.

## Antithetic Variates

### Principle

For symmetric payoffs, use both $X$ and $-X$ (or $U$ and $1-U$).

**Estimator:**
$$\hat{\mu}_{AV} = \frac{1}{2N} \sum_{i=1}^N [f(X_i) + f(-X_i)]$$

where $X_i \sim N(0,1)$.

### Variance Reduction

**Variance:**
$$\text{Var}(\hat{\mu}_{AV}) = \frac{1}{4N}[\text{Var}(f(X)) + \text{Var}(f(-X)) + 2\text{Cov}(f(X), f(-X))]$$

If $f$ is monotonic:
$$\text{Cov}(f(X), f(-X)) < 0$$

So variance is reduced.

**Variance reduction ratio:**
$$VRR = \frac{\text{Var}(\hat{\mu}_{MC})}{\text{Var}(\hat{\mu}_{AV})} = \frac{2}{1 + \rho}$$

where $\rho$ is correlation between $f(X)$ and $f(-X)$.

**Best case:** $\rho = -1$ → $VRR = \infty$ (perfect negative correlation)
**Worst case:** $\rho = 1$ → $VRR = 1$ (no reduction)

### Application: Option Pricing

For European call:
$$C = e^{-rT} E[\max(S_T - K, 0)]$$

**With antithetic:**
1. Generate $Z_i \sim N(0,1)$
2. Compute $S_T^{(i)} = S_0 e^{(r-\sigma^2/2)T + \sigma\sqrt{T}Z_i}$
3. Compute $S_T^{(-i)} = S_0 e^{(r-\sigma^2/2)T - \sigma\sqrt{T}Z_i}$
4. Average: $\hat{C} = \frac{e^{-rT}}{2N} \sum_i [\max(S_T^{(i)} - K, 0) + \max(S_T^{(-i)} - K, 0)]$

**Effectiveness:** Works well for monotonic payoffs (calls, puts).

## Control Variates

### Principle

Use a correlated random variable with known expectation to reduce variance.

**Estimator:**
$$\hat{\mu}_{CV} = \frac{1}{N} \sum_{i=1}^N f(X_i) - c\left(\frac{1}{N} \sum_{i=1}^N g(X_i) - E[g(X)]\right)$$

where:
- $g(X)$ is control variate
- $E[g(X)]$ is known
- $c$ is control coefficient

### Optimal Coefficient

**Variance:**
$$\text{Var}(\hat{\mu}_{CV}) = \frac{1}{N}[\sigma_f^2 - 2c\sigma_{fg} + c^2\sigma_g^2]$$

**Optimal $c$:**
$$c^* = \frac{\text{Cov}(f(X), g(X))}{\text{Var}(g(X))} = \frac{\sigma_{fg}}{\sigma_g^2}$$

**Minimum variance:**
$$\text{Var}(\hat{\mu}_{CV}^*) = \frac{\sigma_f^2}{N}(1 - \rho^2)$$

where $\rho$ is correlation between $f$ and $g$.

**Variance reduction:**
$$VRR = \frac{1}{1 - \rho^2}$$

**Best case:** $\rho = \pm 1$ → $VRR = \infty$

### Estimation of $c^*$

**Sample estimate:**
$$\hat{c} = \frac{\sum_i (f_i - \bar{f})(g_i - \bar{g})}{\sum_i (g_i - \bar{g})^2}$$

**Bias:** Using estimated $c$ introduces small bias, but variance reduction usually dominates.

### Multiple Control Variates

**Extension:**
$$\hat{\mu}_{MCV} = \bar{f} - \sum_{j=1}^k c_j (\bar{g}_j - E[g_j])$$

**Optimal coefficients:** Solve:
$$\boldsymbol{c}^* = \boldsymbol{\Sigma}_g^{-1} \boldsymbol{\sigma}_{fg}$$

where:
- $\boldsymbol{\Sigma}_g$: Covariance matrix of controls
- $\boldsymbol{\sigma}_{fg}$: Covariance vector between $f$ and controls

### Common Control Variates

**Option pricing:**
1. **Stock price:** $S_T$ (known: $E[S_T] = S_0 e^{rT}$)
2. **Discounted stock:** $e^{-rT} S_T$ (known: $E[e^{-rT} S_T] = S_0$)
3. **Geometric average:** For Asian options

**Example - European call:**
$$C = e^{-rT} E[\max(S_T - K, 0)]$$

Use $S_T$ as control:
$$\hat{C}_{CV} = e^{-rT}\bar{P} - c(\bar{S}_T - S_0 e^{rT})$$

where $P_i = \max(S_T^{(i)} - K, 0)$.

**Optimal $c$:**
$$c^* = \frac{\text{Cov}(P, S_T)}{\text{Var}(S_T)}$$

## Importance Sampling

### Principle

Change the sampling distribution to focus on important regions.

**Original:**
$$\mu = \int f(x) p(x) dx = E_p[f(X)]$$

**Importance sampling:**
$$\mu = \int f(x) \frac{p(x)}{q(x)} q(x) dx = E_q\left[f(X)\frac{p(X)}{q(X)}\right]$$

where $q$ is the importance distribution.

**Estimator:**
$$\hat{\mu}_{IS} = \frac{1}{N} \sum_{i=1}^N f(X_i) \frac{p(X_i)}{q(X_i)}$$

where $X_i \sim q$.

### Optimal Importance Distribution

**Variance:**
$$\text{Var}_q(\hat{\mu}_{IS}) = \frac{1}{N} E_q\left[\left(f(X)\frac{p(X)}{q(X)}\right)^2\right] - \frac{\mu^2}{N}$$

**Optimal $q$:**
$$q^*(x) = \frac{|f(x)| p(x)}{\int |f(x)| p(x) dx}$$

**Problem:** Requires knowing $\mu$ (what we're estimating).

**Practical:** Choose $q$ that approximates $q^*$.

### Exponential Tilting

For normal distribution, shift mean:

**Original:** $X \sim N(\mu, \sigma^2)$
**Tilted:** $Y \sim N(\mu + \theta, \sigma^2)$

**Likelihood ratio:**
$$\frac{p(X)}{q(Y)} = \exp\left(-\frac{\theta}{\sigma^2}(Y - \mu) + \frac{\theta^2}{2\sigma^2}\right)$$

**Optimal tilt:** Choose $\theta$ to minimize variance.

**For option pricing:** Tilt toward in-the-money region.

### Application: Rare Events

**Problem:** Estimate $P(X > a)$ where $a$ is large.

**Standard MC:** Most samples give $X < a$ → high variance.

**Importance sampling:** Sample from distribution shifted toward $a$.

**Example:** $X \sim N(0,1)$, estimate $P(X > 5)$.

**Standard:** $P(X > 5) \approx 2.87 \times 10^{-7}$ (rare!)

**IS:** Use $Y \sim N(5, 1)$:
$$P(X > 5) = E_Y\left[\mathbb{1}(Y > 5) \frac{\phi(Y)}{\phi(Y-5)}\right]$$

where $\phi$ is standard normal PDF.

**Variance reduction:** Dramatic for rare events.

## Stratified Sampling

### Principle

Partition sample space into strata, sample from each.

**Stratum $j$:** Region $A_j$ with probability $p_j$.

**Estimator:**
$$\hat{\mu}_{SS} = \sum_{j=1}^k p_j \bar{f}_j$$

where $\bar{f}_j$ is average of $f$ over samples from stratum $j$.

### Variance

**Variance:**
$$\text{Var}(\hat{\mu}_{SS}) = \sum_{j=1}^k \frac{p_j^2 \sigma_j^2}{n_j}$$

where:
- $\sigma_j^2$: Variance within stratum $j$
- $n_j$: Samples from stratum $j$

**Optimal allocation:**
$$n_j^* = n \frac{p_j \sigma_j}{\sum_{i=1}^k p_i \sigma_i}$$

**Proportional allocation:**
$$n_j = n p_j$$

### Stratification Variables

**Good stratification:**
- Highly correlated with $f$
- Easy to sample from conditional distribution
- Natural partition

**Example - Option pricing:**
Stratify by $S_T$:
1. Partition $S_T$ into intervals
2. Sample $S_T$ uniformly from each interval
3. Compute option payoff

**Variance reduction:** Significant if payoff varies within strata.

## Moment Matching

### Principle

Adjust samples to match theoretical moments.

**Standard MC:** $\{X_i\}$ with sample mean $\bar{X}$.

**Moment matching:** Transform to $\{Y_i\}$ with:
- $E[Y] = \mu$ (theoretical mean)
- $\text{Var}(Y) = \sigma^2$ (theoretical variance)

**Transformation:**
$$Y_i = \mu + \frac{\sigma}{\hat{\sigma}}(\bar{X} - \mu) + \frac{\sigma}{\hat{\sigma}}(X_i - \bar{X})$$

where $\hat{\sigma}$ is sample standard deviation.

### Properties

**Advantages:**
- Ensures correct first two moments
- Can reduce variance

**Disadvantages:**
- Introduces bias (samples not independent)
- May not work well for all problems

## Conditional Monte Carlo

### Principle

Use conditional expectation to reduce variance.

**Tower property:**
$$E[f(X,Y)] = E[E[f(X,Y) | Y]]$$

**If $E[f(X,Y) | Y]$ can be computed analytically:**
$$\hat{\mu}_{CMC} = \frac{1}{N} \sum_{i=1}^N E[f(X, Y_i) | Y_i]$$

**Variance:**
$$\text{Var}(\hat{\mu}_{CMC}) = \text{Var}(E[f(X,Y) | Y]) \leq \text{Var}(f(X,Y))$$

**Variance reduction:** Always reduces variance (Jensen's inequality).

### Application: Asian Options

**Payoff:** $\max(\bar{S} - K, 0)$ where $\bar{S} = \frac{1}{n}\sum_{i=1}^n S_{t_i}$.

**Conditional on path:** Can compute $E[\max(\bar{S} - K, 0) | \{S_{t_i}\}]$ analytically in some cases.

**Alternative:** Condition on geometric average (easier to compute).

## Combining Techniques

### Antithetic + Control Variates

**Estimator:**
$$\hat{\mu} = \frac{1}{2N}\sum_{i=1}^N [f(X_i) + f(-X_i)] - c(\bar{g} - E[g])$$

Combine benefits of both methods.

### Importance Sampling + Stratification

**Strategy:**
1. Use importance sampling to focus on important region
2. Stratify within that region

**Example:** For deep out-of-the-money options:
- Tilt toward strike
- Stratify by moneyness

### Multiple Methods

**General framework:**
Combine all applicable methods:
- Antithetic variates
- Control variates
- Importance sampling
- Stratification

**Guidelines:**
- Use methods that complement each other
- Avoid methods that conflict
- Test combinations empirically

## Variance Reduction Ratios

### Definition

$$VRR = \frac{\text{Var}(\hat{\mu}_{MC})}{\text{Var}(\hat{\mu}_{VR})}$$

where $\hat{\mu}_{VR}$ uses variance reduction.

**Interpretation:**
- $VRR = 10$: Need 10x fewer samples for same accuracy
- $VRR = 100$: Need 100x fewer samples

### Estimating VRR

**From samples:**
$$VRR \approx \frac{\hat{\sigma}_{MC}^2}{\hat{\sigma}_{VR}^2}$$

**Bootstrap:** Resample to estimate variance of variance estimates.

## Example: Option Pricing with Variance Reduction

Price European call:
- $S_0 = 100$, $K = 110$, $r = 0.05$, $\sigma = 0.2$, $T = 1$

**Standard MC (10,000 paths):**
- Price: $\$3.25$
- Standard error: $\$0.15$

**With antithetic variates:**
- Price: $\$3.24$
- Standard error: $\$0.10$
- VRR: $(0.15/0.10)^2 = 2.25$

**With control variate ($S_T$):**
- Price: $\$3.24$
- Standard error: $\$0.05$
- VRR: $(0.15/0.05)^2 = 9$

**With both:**
- Price: $\$3.24$
- Standard error: $\$0.04$
- VRR: $(0.15/0.04)^2 = 14.1$

**Combined methods provide substantial variance reduction, requiring fewer samples for the same accuracy.**
