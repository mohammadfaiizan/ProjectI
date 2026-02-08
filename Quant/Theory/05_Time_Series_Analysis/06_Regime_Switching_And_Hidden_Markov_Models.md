# Regime Switching and Hidden Markov Models

## Markov Chains

A Markov chain is a stochastic process where the future depends only on the current state, not the past.

### Definition

A process $\{S_t\}$ is a Markov chain if:
$$P(S_t = j | S_{t-1} = i, S_{t-2} = k, \ldots) = P(S_t = j | S_{t-1} = i)$$

**Markov property:** "Memoryless" - only current state matters.

### Transition Matrix

For a finite-state Markov chain with states $\{1, 2, \ldots, K\}$:

$$\mathbf{P} = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1K} \\
p_{21} & p_{22} & \cdots & p_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
p_{K1} & p_{K2} & \cdots & p_{KK}
\end{pmatrix}$$

where $p_{ij} = P(S_t = j | S_{t-1} = i)$.

**Properties:**
- $p_{ij} \geq 0$ for all $i, j$
- $\sum_{j=1}^{K}p_{ij} = 1$ for all $i$ (rows sum to 1)

### Multi-Step Transitions

$n$-step transition probabilities:
$$p_{ij}^{(n)} = P(S_{t+n} = j | S_t = i)$$

Computed as:
$$\mathbf{P}^{(n)} = \mathbf{P}^n$$

### Stationary Distribution

A stationary distribution $\boldsymbol{\pi}$ satisfies:
$$\boldsymbol{\pi}^T = \boldsymbol{\pi}^T \mathbf{P}$$

or equivalently:
$$\pi_j = \sum_{i=1}^{K}\pi_i p_{ij}$$

**Interpretation:** Long-run probability of being in each state.

**Computation:** Solve eigenvalue problem - $\boldsymbol{\pi}$ is left eigenvector of $\mathbf{P}$ with eigenvalue 1.

### Ergodicity

A Markov chain is ergodic if:
- **Irreducible:** All states communicate (can reach any state from any state)
- **Aperiodic:** No cycles preventing convergence

**Ergodic theorem:** For ergodic chains, the stationary distribution exists, is unique, and:
$$\lim_{n \to \infty}p_{ij}^{(n)} = \pi_j$$

regardless of initial state.

## Hidden Markov Models (HMM)

In HMMs, the state $S_t$ is unobserved (hidden), but we observe $Y_t$ which depends on $S_t$.

### Model Structure

**State process:** $\{S_t\}$ is a Markov chain with transition matrix $\mathbf{P}$

**Observation process:** Given $S_t = j$, $Y_t$ has distribution:
$$Y_t | S_t = j \sim f_j(\cdot)$$

**Parameters:**
- Transition probabilities: $p_{ij}$
- Observation distributions: $f_j$ (e.g., $N(\mu_j, \sigma_j^2)$)

### Example: Regime-Dependent Returns

$$r_t | S_t = j \sim N(\mu_j, \sigma_j^2)$$

where $S_t \in \{1, 2\}$ (bull/bear market).

### Forward-Backward Algorithm

Computes filtered and smoothed probabilities.

**Filtered probability:** $\xi_{jt} = P(S_t = j | Y_1, \ldots, Y_t)$

**Smoothed probability:** $\xi_{jt|T} = P(S_t = j | Y_1, \ldots, Y_T)$

### Forward Algorithm

**Initialization:**
$$\xi_{j0} = \pi_j \quad \text{(stationary distribution)}$$

**Recursion:**
$$\xi_{jt} = \frac{f_j(Y_t) \sum_{i=1}^{K}\xi_{i,t-1} p_{ij}}{\sum_{j=1}^{K}f_j(Y_t) \sum_{i=1}^{K}\xi_{i,t-1} p_{ij}}$$

**Interpretation:** Update prior belief about state using new observation.

### Backward Algorithm

**Initialization:**
$$\beta_{jT} = 1$$

**Recursion:**
$$\beta_{jt} = \sum_{i=1}^{K}p_{ji} f_i(Y_{t+1}) \beta_{i,t+1}$$

**Smoothed probability:**
$$\xi_{jt|T} = \frac{\xi_{jt} \beta_{jt}}{\sum_{i=1}^{K}\xi_{it} \beta_{it}}$$

### Viterbi Algorithm

Finds the most likely sequence of states (not probabilities).

**Algorithm:** Dynamic programming - find:
$$\arg\max_{S_1, \ldots, S_T} P(S_1, \ldots, S_T | Y_1, \ldots, Y_T)$$

**Recursion:**
$$\delta_{jt} = \max_{i}[\delta_{i,t-1} p_{ij}] f_j(Y_t)$$

Track backpointers to recover optimal path.

**Use:** Identify regime switches, dating business cycles.

### Baum-Welch Algorithm (EM)

Estimates HMM parameters when states are unobserved.

**E-step:** Compute expected sufficient statistics using forward-backward
**M-step:** Update parameters to maximize expected log-likelihood

**Iterate until convergence.**

**Parameters updated:**
- Transition probabilities: $p_{ij} = \frac{\sum_{t=2}^{T}\xi_{ijt}}{\sum_{t=2}^{T}\xi_{it}}$
- Observation parameters: Weighted MLE using smoothed probabilities

## Markov-Switching Models

Markov-switching models allow parameters to depend on an unobserved regime.

### Hamilton's Model

Hamilton's model for business cycles:

$$y_t = \mu_{S_t} + \sum_{i=1}^{p}\phi_i(y_{t-i} - \mu_{S_{t-i}}) + \epsilon_t$$

where $S_t \in \{1, 2\}$ (expansion/recession) and $\epsilon_t \sim N(0, \sigma^2)$.

**Features:**
- Mean $\mu_{S_t}$ switches with regime
- AR coefficients may also switch
- Regime follows Markov chain

### Estimation

1. **State space form:** Write as state space model
2. **Kalman filter:** Use filter for linear case, or
3. **Hamilton filter:** Specialized filter for Markov-switching
4. **MLE:** Maximize likelihood (nonlinear optimization)

**Likelihood:** Computed via forward algorithm.

### Regime-Dependent Volatility

Allow volatility to switch:

$$\sigma_t^2 = \sigma_{S_t}^2$$

or GARCH with switching:

$$\sigma_t^2 = \alpha_{0,S_t} + \alpha_{1,S_t} r_{t-1}^2 + \beta_{1,S_t} \sigma_{t-1}^2$$

**Use:** Model volatility regimes (high vol vs low vol periods).

### Regime-Dependent Mean

Returns with switching mean:

$$r_t = \mu_{S_t} + \epsilon_t$$

where $\epsilon_t \sim N(0, \sigma_{S_t}^2)$.

**Interpretation:** Different expected returns in different regimes.

## Applications

### Bull/Bear Market Detection

**Model:**
$$r_t | S_t = j \sim N(\mu_j, \sigma_j^2)$$

where $S_t \in \{1, 2\}$ (bull/bear).

**Inference:**
- Filtered probabilities: $P(S_t = \text{bull} | r_1, \ldots, r_t)$
- Smoothed probabilities: $P(S_t = \text{bull} | r_1, \ldots, r_T)$
- Viterbi: Most likely regime sequence

**Use:**
- Identify market phases
- Adjust strategy by regime
- Risk management (higher risk in bear)

### Asset Allocation Under Regimes

**Strategy:**
- Estimate regime probabilities
- Optimize portfolio conditional on regime:
  $$\max_{\mathbf{w}} \mathbb{E}[U(W_{t+1}) | S_t = j]$$

- Or use regime probabilities as weights:
  $$\mathbf{w}_t = \sum_{j=1}^{K}\xi_{jt} \mathbf{w}_j^*$$

where $\mathbf{w}_j^*$ is optimal portfolio in regime $j$.

**Dynamic allocation:** Rebalance as regime probabilities change.

### Volatility Regimes

Model volatility clustering as regime switching:

$$\sigma_t^2 = \sigma_{S_t}^2$$

**Advantages over GARCH:**
- Discrete regimes (clear high/low vol periods)
- Can model sudden switches
- May fit better than continuous GARCH

**Disadvantages:**
- Fewer regimes than GARCH flexibility
- Harder to estimate

### Term Structure Regimes

Model yield curve with switching factors:

$$\mathbf{y}_t = \mathbf{A} + \mathbf{B} \mathbf{f}_{S_t} + \boldsymbol{\epsilon}_t$$

where $\mathbf{f}_{S_t}$ are regime-dependent factors.

**Use:** Understand yield curve dynamics across regimes.

## Structural Breaks

Structural breaks are permanent changes in model parameters.

### CUSUM Test

Cumulative sum test for parameter stability:

$$S_t = \sum_{i=1}^{t}\hat{\epsilon}_i$$

where $\hat{\epsilon}_i$ are recursive residuals.

**Test statistic:**
$$CUSUM_t = \frac{S_t}{\hat{\sigma}\sqrt{T}}$$

**Critical values:** Non-standard. Reject if exceeds bounds.

### Bai-Perron Test

Tests for multiple structural breaks at unknown dates.

**Procedure:**
1. Test $H_0$: No breaks vs $H_1$: $m$ breaks
2. If reject, estimate break dates
3. Test for additional breaks

**Advantages:**
- Allows multiple breaks
- Estimates break dates
- Handles serial correlation

### Chow Test

Tests for structural break at known date $t^*$.

**Model:**
$$y_t = \begin{cases}
\mathbf{x}_t^T \boldsymbol{\beta}_1 + \epsilon_t & \text{if } t \leq t^* \\
\mathbf{x}_t^T \boldsymbol{\beta}_2 + \epsilon_t & \text{if } t > t^*
\end{cases}$$

**Test:** $H_0: \boldsymbol{\beta}_1 = \boldsymbol{\beta}_2$

**F-statistic:**
$$F = \frac{(RSS_{pooled} - RSS_1 - RSS_2)/k}{(RSS_1 + RSS_2)/(T-2k)}$$

where $RSS$ are residual sum of squares.

**Limitation:** Requires known break date.

## Practical Considerations

### Number of Regimes

- **Too few:** Miss important structure
- **Too many:** Overfitting, hard to interpret
- **Information criteria:** AIC, BIC to choose
- **Economic theory:** Should guide choice

### Identification

Regime-switching models may have identification issues:
- **Label switching:** Regimes can be relabeled
- **Scale restrictions:** May need to impose (e.g., $\mu_1 < \mu_2$)
- **Initial values:** Sensitive to starting values

### Estimation Challenges

- **Local maxima:** Multiple solutions
- **Slow convergence:** EM algorithm can be slow
- **Parameter constraints:** Ensure valid probabilities, positive variances

### Forecasting

**One-step ahead:**
$$\hat{y}_{t+1|t} = \sum_{j=1}^{K}\xi_{jt} \mathbb{E}[y_{t+1} | S_{t+1} = j]$$

**Multi-step:** Account for regime uncertainty:
- Regime may switch
- Average over possible future regimes

### Model Comparison

- **Likelihood ratio test:** Nested models
- **Information criteria:** AIC, BIC
- **Out-of-sample:** Forecast performance
- **Economic significance:** Do regimes matter?

### Regime Duration

Expected duration in regime $j$:
$$\mathbb{E}[D_j] = \frac{1}{1 - p_{jj}}$$

**Interpretation:** How long regimes typically last.

### Persistence

High $p_{jj}$ means persistent regimes (hard to switch out).

Low $p_{jj}$ means transient regimes (frequent switches).
