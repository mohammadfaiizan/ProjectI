# Bayesian Statistics

## Bayesian Framework

### Prior, Likelihood, and Posterior

Bayesian inference updates beliefs about parameters $\theta$ using data $X$:

**Prior distribution**: $\pi(\theta)$ - beliefs about $\theta$ before observing data

**Likelihood**: $L(\theta; X) = f(X|\theta)$ - probability of data given $\theta$

**Posterior distribution**: $\pi(\theta|X)$ - updated beliefs after observing data

**Bayes' theorem**:

$$\pi(\theta|X) = \frac{L(\theta; X)\pi(\theta)}{\int L(\theta; X)\pi(\theta) d\theta} = \frac{L(\theta; X)\pi(\theta)}{m(X)}$$

where $m(X) = \int L(\theta; X)\pi(\theta) d\theta$ is the marginal likelihood (evidence).

**Proportional form**:

$$\pi(\theta|X) \propto L(\theta; X)\pi(\theta)$$

### Conjugate Priors

A prior $\pi(\theta)$ is conjugate to likelihood $f(X|\theta)$ if the posterior belongs to the same family as the prior.

**Examples**:

1. **Normal-Normal**: $X|\mu \sim \mathcal{N}(\mu, \sigma^2)$ (known $\sigma^2$), $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$:
   $$\mu|X \sim \mathcal{N}\left(\frac{\tau_0^2\bar{X} + \sigma^2\mu_0/n}{\tau_0^2 + \sigma^2/n}, \frac{\tau_0^2\sigma^2/n}{\tau_0^2 + \sigma^2/n}\right)$$

2. **Beta-Binomial**: $X|\theta \sim \text{Binomial}(n, \theta)$, $\theta \sim \text{Beta}(\alpha, \beta)$:
   $$\theta|X \sim \text{Beta}(\alpha + X, \beta + n - X)$$

3. **Gamma-Poisson**: $X|\lambda \sim \text{Poisson}(\lambda)$, $\lambda \sim \text{Gamma}(\alpha, \beta)$:
   $$\lambda|X \sim \text{Gamma}\left(\alpha + \sum X_i, \beta + n\right)$$

4. **Inverse Gamma-Normal**: $X|\sigma^2 \sim \mathcal{N}(\mu, \sigma^2)$ (known $\mu$), $\sigma^2 \sim \text{InvGamma}(\alpha, \beta)$:
   $$\sigma^2|X \sim \text{InvGamma}\left(\alpha + n/2, \beta + \frac{1}{2}\sum(X_i - \mu)^2\right)$$

### Non-Informative Priors

**Jeffreys prior**: Invariant under reparameterization:

$$\pi(\theta) \propto \sqrt{I(\theta)}$$

where $I(\theta)$ is Fisher information:

$$I(\theta) = E\left[\left(\frac{\partial \ln f(X|\theta)}{\partial \theta}\right)^2\right]$$

**Examples**:
- Location parameter: $\pi(\mu) \propto 1$ (improper uniform)
- Scale parameter: $\pi(\sigma) \propto 1/\sigma$
- Binomial proportion: $\pi(\theta) \propto \theta^{-1/2}(1-\theta)^{-1/2}$ (Beta$(1/2, 1/2)$)

## MAP Estimation vs Posterior Mean

### Maximum A Posteriori (MAP)

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \pi(\theta|X) = \arg\max_\theta [\ln L(\theta; X) + \ln \pi(\theta)]$$

**Properties**:
- Mode of posterior distribution
- Penalized maximum likelihood (regularization)
- Not invariant under reparameterization

### Posterior Mean

$$\hat{\theta}_{\text{PM}} = E[\theta|X] = \int \theta \pi(\theta|X) d\theta$$

**Properties**:
- Minimizes posterior expected squared error: $E[(\theta - \hat{\theta})^2|X]$
- Invariant under linear transformations
- May not exist for improper priors

### Comparison

- **Posterior mean**: Optimal under squared error loss
- **MAP**: Optimal under 0-1 loss (for discrete parameters)
- **Posterior median**: Optimal under absolute error loss

## Bayesian Updating

### Sequential Learning

Bayesian updating is sequential: posterior becomes prior for next observation.

**Two-step update**:

$$\pi(\theta|X_1, X_2) \propto f(X_2|\theta)\pi(\theta|X_1) \propto f(X_2|\theta)f(X_1|\theta)\pi(\theta)$$

**Online learning**: Update beliefs as new data arrives:

$$\pi(\theta|X_{1:n+1}) \propto f(X_{n+1}|\theta)\pi(\theta|X_{1:n})$$

### Predictive Distribution

**Prior predictive**: Distribution of future observation before seeing data:

$$m(X) = \int f(X|\theta)\pi(\theta) d\theta$$

**Posterior predictive**: Distribution of future observation after seeing data:

$$f(X_{\text{new}}|X) = \int f(X_{\text{new}}|\theta)\pi(\theta|X) d\theta$$

**Example**: For $X|\mu \sim \mathcal{N}(\mu, \sigma^2)$ and $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$:

$$X_{\text{new}}|X \sim \mathcal{N}\left(E[\mu|X], \sigma^2 + \text{Var}(\mu|X)\right)$$

## Markov Chain Monte Carlo

### Metropolis-Hastings Algorithm

Generate samples from posterior $\pi(\theta|X)$:

1. **Initialize**: $\theta^{(0)}$
2. **For** $t = 1, 2, \ldots$:
   - **Propose**: $\theta^* \sim q(\theta^*|\theta^{(t-1)})$ (proposal distribution)
   - **Accept with probability**:
     $$\alpha = \min\left(1, \frac{\pi(\theta^*|X)q(\theta^{(t-1)}|\theta^*)}{\pi(\theta^{(t-1)}|X)q(\theta^*|\theta^{(t-1)})}\right)$$
   - **Set**: $\theta^{(t)} = \theta^*$ if accepted, else $\theta^{(t)} = \theta^{(t-1)}$

**Special cases**:
- **Random walk**: $q(\theta^*|\theta) = q(|\theta^* - \theta|)$ (symmetric)
- **Independence**: $q(\theta^*|\theta) = q(\theta^*)$ (independent of current state)
- **Metropolis**: Symmetric proposal $q(\theta^*|\theta) = q(\theta|\theta^*)$

### Gibbs Sampling

For multivariate $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_d)$:

1. **Initialize**: $\boldsymbol{\theta}^{(0)}$
2. **For** $t = 1, 2, \ldots$:
   - Sample $\theta_1^{(t)} \sim \pi(\theta_1|\theta_2^{(t-1)}, \ldots, \theta_d^{(t-1)}, X)$
   - Sample $\theta_2^{(t)} \sim \pi(\theta_2|\theta_1^{(t)}, \theta_3^{(t-1)}, \ldots, \theta_d^{(t-1)}, X)$
   - $\ldots$
   - Sample $\theta_d^{(t)} \sim \pi(\theta_d|\theta_1^{(t)}, \ldots, \theta_{d-1}^{(t)}, X)$

**Requirements**: Full conditionals must be available in closed form.

**Advantages**:
- No rejection (acceptance rate = 1)
- Often more efficient than Metropolis-Hastings
- Natural for hierarchical models

### Convergence Diagnostics

**Gelman-Rubin statistic**: Compare within-chain and between-chain variance:

$$\hat{R} = \sqrt{\frac{n-1}{n} + \frac{1}{n}\frac{B}{W}}$$

where $B$ is between-chain variance and $W$ is within-chain variance. $\hat{R} \approx 1$ indicates convergence.

**Effective sample size**: Accounts for autocorrelation:

$$\text{ESS} = \frac{M}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

where $M$ is number of samples and $\rho_k$ is lag-$k$ autocorrelation.

**Trace plots**: Visual inspection of chains.

**Autocorrelation function**: Should decay quickly.

## Bayesian Regression

### Linear Regression

**Model**: $Y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

**Likelihood**:

$$L(\boldsymbol{\beta}, \sigma^2; \mathbf{y}, \mathbf{X}) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})\right)$$

**Conjugate prior**: Normal-Inverse Gamma:

$$\boldsymbol{\beta}|\sigma^2 \sim \mathcal{N}(\boldsymbol{\beta}_0, \sigma^2\mathbf{V}_0)$$
$$\sigma^2 \sim \text{InvGamma}(\alpha_0, \beta_0)$$

**Posterior**:

$$\boldsymbol{\beta}|\sigma^2, \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_n, \sigma^2\mathbf{V}_n)$$

where:

$$\mathbf{V}_n = (\mathbf{V}_0^{-1} + \mathbf{X}^T\mathbf{X})^{-1}$$
$$\boldsymbol{\mu}_n = \mathbf{V}_n(\mathbf{V}_0^{-1}\boldsymbol{\beta}_0 + \mathbf{X}^T\mathbf{y})$$

**Marginal posterior for $\sigma^2$**:

$$\sigma^2|\mathbf{y} \sim \text{InvGamma}\left(\alpha_0 + n/2, \beta_0 + \frac{1}{2}(\mathbf{y}^T\mathbf{y} + \boldsymbol{\beta}_0^T\mathbf{V}_0^{-1}\boldsymbol{\beta}_0 - \boldsymbol{\mu}_n^T\mathbf{V}_n^{-1}\boldsymbol{\mu}_n)\right)$$

### Ridge Regression (Bayesian Lasso)

**Prior**: $\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$

This corresponds to $L_2$ penalty (ridge regression).

**Posterior mode**: $\hat{\boldsymbol{\beta}}_{\text{MAP}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

### Bayesian Variable Selection

**Spike-and-slab prior**:

$$\beta_j|\gamma_j \sim \begin{cases}
\delta_0 & \text{if } \gamma_j = 0 \\
\mathcal{N}(0, \tau^2) & \text{if } \gamma_j = 1
\end{cases}$$

where $\gamma_j \sim \text{Bernoulli}(\pi)$ indicates inclusion.

**Stochastic search**: Use MCMC to sample $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$.

## Hierarchical Models

### Structure

**Level 1** (data): $Y_{ij}|\theta_i \sim f(Y_{ij}|\theta_i)$

**Level 2** (parameters): $\theta_i|\boldsymbol{\phi} \sim \pi(\theta_i|\boldsymbol{\phi})$

**Level 3** (hyperparameters): $\boldsymbol{\phi} \sim \pi(\boldsymbol{\phi})$

**Example**: Random effects model

$$Y_{ij} = \mu + \alpha_i + \epsilon_{ij}$$

where $\alpha_i \sim \mathcal{N}(0, \sigma_\alpha^2)$ and $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$.

### Advantages

- **Borrowing strength**: Information from all groups informs each group
- **Shrinkage**: Estimates shrink toward overall mean
- **Uncertainty quantification**: Accounts for uncertainty in hyperparameters

## Applications

### Bayesian Portfolio Optimization

**Prior**: Beliefs about expected returns $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$

**Black-Litterman model**: Combines market equilibrium with views:

$$\boldsymbol{\mu}_{\text{BL}} = [(\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}^T\boldsymbol{\Omega}^{-1}\mathbf{P}]^{-1}[(\tau\boldsymbol{\Sigma})^{-1}\boldsymbol{\mu}_{\text{equil}} + \mathbf{P}^T\boldsymbol{\Omega}^{-1}\mathbf{Q}]$$

where:
- $\boldsymbol{\mu}_{\text{equil}}$: equilibrium returns
- $\mathbf{Q}$: investor views
- $\mathbf{P}$: pick matrix
- $\boldsymbol{\Omega}$: uncertainty in views
- $\tau$: confidence in prior

**Posterior optimal portfolio**:

$$\mathbf{w}^* = \frac{1}{\lambda}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_{\text{BL}}$$

### Signal Detection

**Model**: $X|\theta \sim \mathcal{N}(\theta, \sigma^2)$ where $\theta$ is signal strength

**Prior**: $\theta \sim \mathcal{N}(\mu_0, \tau_0^2)$

**Posterior**: $\theta|X \sim \mathcal{N}(\mu_n, \tau_n^2)$

**Detection rule**: Reject $H_0: \theta = 0$ if posterior probability $\mathbb{P}(\theta > 0|X) > \alpha$

**Bayes factor**: 

$$BF = \frac{\mathbb{P}(X|H_1)}{\mathbb{P}(X|H_0)} = \frac{\int f(X|\theta)\pi(\theta) d\theta}{f(X|0)}$$

### Model Averaging

**Bayesian model averaging**: Weight predictions by posterior model probabilities:

$$E[Y|X] = \sum_{k=1}^{K} \mathbb{P}(M_k|X) E[Y|X, M_k]$$

where $M_k$ are candidate models.

**Posterior model probability**:

$$\mathbb{P}(M_k|X) = \frac{m_k(X)\pi(M_k)}{\sum_{j=1}^{K} m_j(X)\pi(M_j)}$$

where $m_k(X)$ is marginal likelihood for model $k$.

### Risk Management

**Bayesian VaR**: Use posterior predictive distribution:

$$\text{VaR}_{\alpha} = -F^{-1}_{L|X}(1-\alpha)$$

where $F_{L|X}$ is posterior predictive CDF of loss.

**Parameter uncertainty**: Accounts for uncertainty in model parameters, leading to wider confidence intervals than frequentist methods.

### Calibration

**Bayesian calibration**: Update model parameters using observed prices:

$$\pi(\boldsymbol{\theta}|\mathbf{P}_{\text{obs}}) \propto L(\mathbf{P}_{\text{obs}}|\boldsymbol{\theta})\pi(\boldsymbol{\theta})$$

where $\mathbf{P}_{\text{obs}}$ are observed option prices and $\boldsymbol{\theta}$ are model parameters (e.g., volatility, mean reversion).

**Predictive pricing**: Price new options using:

$$P_{\text{new}} = \int P(\boldsymbol{\theta})\pi(\boldsymbol{\theta}|\mathbf{P}_{\text{obs}}) d\boldsymbol{\theta}$$
