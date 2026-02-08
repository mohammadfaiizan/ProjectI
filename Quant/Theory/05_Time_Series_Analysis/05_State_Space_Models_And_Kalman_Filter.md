# State Space Models and Kalman Filter

## State Space Representation

State space models separate observed variables from unobserved (latent) state variables.

### General Form

**Observation equation:**
$$\mathbf{Y}_t = \mathbf{H}_t \boldsymbol{\theta}_t + \mathbf{d}_t + \boldsymbol{\epsilon}_t$$

**State equation:**
$$\boldsymbol{\theta}_t = \mathbf{F}_t \boldsymbol{\theta}_{t-1} + \mathbf{c}_t + \boldsymbol{\eta}_t$$

where:
- $\mathbf{Y}_t$: $n \times 1$ vector of observations
- $\boldsymbol{\theta}_t$: $m \times 1$ vector of unobserved states
- $\mathbf{H}_t$: $n \times m$ observation matrix
- $\mathbf{F}_t$: $m \times m$ transition matrix
- $\boldsymbol{\epsilon}_t \sim N(0, \mathbf{R}_t)$: observation noise
- $\boldsymbol{\eta}_t \sim N(0, \mathbf{Q}_t)$: state noise

**Assumptions:**
- $\boldsymbol{\epsilon}_t$ and $\boldsymbol{\eta}_s$ are independent for all $t, s$
- $\boldsymbol{\epsilon}_t$ and $\boldsymbol{\eta}_t$ may be correlated: $\text{Cov}(\boldsymbol{\epsilon}_t, \boldsymbol{\eta}_t) = \mathbf{S}_t$

### Time-Invariant Case

If matrices are constant:
$$\mathbf{Y}_t = \mathbf{H}\boldsymbol{\theta}_t + \boldsymbol{\epsilon}_t$$
$$\boldsymbol{\theta}_t = \mathbf{F}\boldsymbol{\theta}_{t-1} + \boldsymbol{\eta}_t$$

### Initial State

$$\boldsymbol{\theta}_0 \sim N(\boldsymbol{\mu}_0, \mathbf{P}_0)$$

## Kalman Filter

The Kalman filter recursively computes optimal estimates of the state given observations up to time $t$.

### Prediction Step

**Predicted state:**
$$\hat{\boldsymbol{\theta}}_{t|t-1} = \mathbf{F}_t \hat{\boldsymbol{\theta}}_{t-1|t-1} + \mathbf{c}_t$$

**Predicted covariance:**
$$\mathbf{P}_{t|t-1} = \mathbf{F}_t \mathbf{P}_{t-1|t-1} \mathbf{F}_t^T + \mathbf{Q}_t$$

**Predicted observation:**
$$\hat{\mathbf{Y}}_{t|t-1} = \mathbf{H}_t \hat{\boldsymbol{\theta}}_{t|t-1} + \mathbf{d}_t$$

**Innovation (forecast error):**
$$\boldsymbol{\nu}_t = \mathbf{Y}_t - \hat{\mathbf{Y}}_{t|t-1}$$

**Innovation covariance:**
$$\mathbf{S}_t = \mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^T + \mathbf{R}_t$$

### Update Step

**Kalman gain:**
$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^T \mathbf{S}_t^{-1}$$

**Updated state:**
$$\hat{\boldsymbol{\theta}}_{t|t} = \hat{\boldsymbol{\theta}}_{t|t-1} + \mathbf{K}_t \boldsymbol{\nu}_t$$

**Updated covariance:**
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1}$$

### Interpretation

- **Prediction:** Forecast state and observation using past information
- **Update:** Adjust forecast using new observation
- **Kalman gain:** Weight given to new observation (larger if observation is more informative relative to prediction uncertainty)

### Likelihood

The log-likelihood is:
$$\ln L = -\frac{1}{2}\sum_{t=1}^{T}\left[n\ln(2\pi) + \ln|\mathbf{S}_t| + \boldsymbol{\nu}_t^T \mathbf{S}_t^{-1} \boldsymbol{\nu}_t\right]$$

This allows maximum likelihood estimation of unknown parameters.

## Kalman Smoother

The smoother computes optimal estimates using all observations (past and future).

### Rauch-Tung-Striebel Smoother

**Backward recursion:**

**Smoothing gain:**
$$\mathbf{J}_t = \mathbf{P}_{t|t} \mathbf{F}_{t+1}^T \mathbf{P}_{t+1|t}^{-1}$$

**Smoothed state:**
$$\hat{\boldsymbol{\theta}}_{t|T} = \hat{\boldsymbol{\theta}}_{t|t} + \mathbf{J}_t(\hat{\boldsymbol{\theta}}_{t+1|T} - \hat{\boldsymbol{\theta}}_{t+1|t})$$

**Smoothed covariance:**
$$\mathbf{P}_{t|T} = \mathbf{P}_{t|t} + \mathbf{J}_t(\mathbf{P}_{t+1|T} - \mathbf{P}_{t+1|t})\mathbf{J}_t^T$$

**Use:** Better estimates for historical analysis, missing data imputation.

## Maximum Likelihood Estimation

Unknown parameters (elements of $\mathbf{F}$, $\mathbf{H}$, $\mathbf{Q}$, $\mathbf{R}$, etc.) are estimated via MLE.

### Procedure

1. **Initialize:** Guess parameter values
2. **Filter:** Run Kalman filter, compute likelihood
3. **Optimize:** Update parameters to maximize likelihood
4. **Iterate:** Repeat until convergence

### Numerical Optimization

- **Gradient-based:** Quasi-Newton methods (BFGS)
- **Derivative-free:** Nelder-Mead, simulated annealing
- **EM algorithm:** Alternative approach (treats states as missing data)

### Identifiability

Some parameters may not be identifiable:
- Scale restrictions needed
- Normalize certain parameters
- Check identification conditions

## Extended Kalman Filter (EKF)

EKF handles nonlinear state space models.

### Nonlinear Model

**Observation:** $\mathbf{Y}_t = h(\boldsymbol{\theta}_t) + \boldsymbol{\epsilon}_t$
**State:** $\boldsymbol{\theta}_t = f(\boldsymbol{\theta}_{t-1}) + \boldsymbol{\eta}_t$

### Linearization

Linearize around current estimate:

$$\mathbf{H}_t = \frac{\partial h}{\partial \boldsymbol{\theta}}\bigg|_{\hat{\boldsymbol{\theta}}_{t|t-1}}$$

$$\mathbf{F}_t = \frac{\partial f}{\partial \boldsymbol{\theta}}\bigg|_{\hat{\boldsymbol{\theta}}_{t-1|t-1}}$$

Then apply standard Kalman filter with these time-varying matrices.

### Limitations

- **Approximation:** Only exact for linear models
- **Bias:** Can be biased for strong nonlinearities
- **Divergence:** May diverge if linearization poor

## Unscented Kalman Filter (UKF)

UKF uses unscented transform instead of linearization.

### Unscented Transform

For nonlinear function $y = g(x)$ where $x \sim N(\mu, \Sigma)$:

1. **Select sigma points:** $2m+1$ points chosen deterministically
2. **Transform:** Apply $g$ to each sigma point
3. **Moment matching:** Compute mean and covariance of transformed points

**Advantages:**
- Captures mean and covariance to second order (vs first order for EKF)
- No need for derivatives
- Often more accurate than EKF

### UKF Algorithm

Same as Kalman filter but:
- Use unscented transform for prediction and update
- No linearization needed

## Particle Filters (Sequential Monte Carlo)

Particle filters use Monte Carlo simulation for nonlinear/non-Gaussian models.

### Basic Particle Filter

**Algorithm:**

1. **Initialization:** Sample $N$ particles $\boldsymbol{\theta}_0^{(i)} \sim p(\boldsymbol{\theta}_0)$

2. **Prediction:** For each particle, sample:
   $$\boldsymbol{\theta}_t^{(i)} \sim p(\boldsymbol{\theta}_t | \boldsymbol{\theta}_{t-1}^{(i)})$$

3. **Weighting:** Compute weights:
   $$w_t^{(i)} \propto p(\mathbf{Y}_t | \boldsymbol{\theta}_t^{(i)})$$

4. **Resampling:** Resample particles according to weights (with replacement)

5. **Estimate:** 
   $$\hat{\boldsymbol{\theta}}_{t|t} = \sum_{i=1}^{N}w_t^{(i)}\boldsymbol{\theta}_t^{(i)}$$

### Advantages

- Handles nonlinearities and non-Gaussian distributions
- Flexible
- No approximation (asymptotically exact)

### Disadvantages

- Computationally expensive
- Sample degeneracy (few particles have high weight)
- Requires tuning (number of particles, resampling scheme)

### Resampling Schemes

- **Multinomial:** Sample with replacement according to weights
- **Systematic:** More efficient
- **Stratified:** Reduces variance

## Applications

### Estimating Hidden Factors

**Example:** Factor model where factors are unobserved:
$$r_t = \boldsymbol{\beta}^T \mathbf{f}_t + \epsilon_t$$
$$\mathbf{f}_t = \mathbf{F} \mathbf{f}_{t-1} + \boldsymbol{\eta}_t$$

Use Kalman filter to estimate $\mathbf{f}_t$ from returns $\mathbf{r}_t$.

### Tracking Alpha Decay

Alpha (excess return) may decay over time:
$$\alpha_t = \phi \alpha_{t-1} + \eta_t$$

Use Kalman filter to track $\alpha_t$ from realized returns.

### Regime Detection

Hidden Markov models can be written in state space form:
$$s_t \in \{1, 2, \ldots, K\}$$
$$P(s_t = j | s_{t-1} = i) = p_{ij}$$

Use filter to estimate regime probabilities $P(s_t = j | \mathbf{Y}_t)$.

### Term Structure Models

Many term structure models (e.g., affine models) are state space models:
- Latent factors drive yield curve
- Yields are observed
- Kalman filter estimates factors

### Stochastic Volatility

Stochastic volatility models:
$$\ln \sigma_t^2 = \alpha + \phi \ln \sigma_{t-1}^2 + \eta_t$$

Use particle filter (nonlinear, non-Gaussian) or approximate with EKF/UKF.

### Missing Data

Kalman filter naturally handles missing observations:
- Set corresponding rows of $\mathbf{H}_t$ to zero
- Or use large observation noise for missing data

### Nowcasting

Nowcasting uses high-frequency data to estimate current state of low-frequency variable:
- GDP (quarterly) estimated using monthly/weekly indicators
- Kalman filter combines information efficiently

## Practical Considerations

### Initialization

- **Diffuse prior:** $\mathbf{P}_0$ very large (uninformative)
- **Stationary distribution:** If state is stationary, use unconditional distribution
- **Data-based:** Use first few observations

### Numerical Stability

- **Square-root filter:** Use Cholesky decomposition to ensure positive definiteness
- **Joseph form:** More stable covariance update
- **Regularization:** Add small positive definite matrix if needed

### Model Specification

- **Observability:** Can states be identified from observations?
- **Controllability:** Can inputs affect all states?
- **Stability:** Are state dynamics stable?

### Diagnostics

- **Innovations:** Should be white noise (test with Ljung-Box)
- **Normality:** Check if innovations are Gaussian
- **Outliers:** Detect using innovation magnitude

### Computational Efficiency

- **Sparse matrices:** Exploit structure
- **Parallelization:** Filter steps can be parallelized
- **Reduced models:** Approximate if full model too complex
