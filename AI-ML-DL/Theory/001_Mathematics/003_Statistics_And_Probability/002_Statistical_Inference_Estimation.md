# Statistical Inference and Estimation

## Table of Contents

1. [Introduction](#introduction)
2. [Point Estimation](#point-estimation)
3. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
4. [Maximum A Posteriori Estimation](#maximum-a-posteriori-estimation)
5. [Bayesian Inference](#bayesian-inference)
6. [Properties of Estimators](#properties-of-estimators)
7. [Hypothesis Testing](#hypothesis-testing)
8. [Confidence Intervals](#confidence-intervals)
9. [Bootstrap Methods](#bootstrap-methods)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Statistical inference aims to draw conclusions about populations from samples, quantify uncertainty, and make predictions. In machine learning, inference methods are used for parameter estimation, model selection, uncertainty quantification, and hypothesis testing. This document covers maximum likelihood estimation (MLE), maximum a posteriori (MAP) estimation, Bayesian inference, hypothesis testing, and confidence intervals, with emphasis on their applications in ML.

## Point Estimation

### Problem Formulation

Given data $\mathcal{D} = \{x_1, \ldots, x_n\}$ assumed to be drawn from distribution $p(x | \boldsymbol{\theta})$ with unknown parameters $\boldsymbol{\theta}$, estimate $\boldsymbol{\theta}$.

**Point estimator**: Function $\hat{\boldsymbol{\theta}} = T(\mathcal{D})$ mapping data to parameter estimate.

**Example**: Sample mean $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ estimates population mean $\mu$.

### Estimator Properties

**Bias**: $\text{Bias}(\hat{\boldsymbol{\theta}}) = \mathbb{E}[\hat{\boldsymbol{\theta}}] - \boldsymbol{\theta}$

**Unbiased estimator**: $\text{Bias}(\hat{\boldsymbol{\theta}}) = \mathbf{0}$ (on average, estimate equals true value)

**Variance**: $\text{Var}(\hat{\boldsymbol{\theta}}) = \mathbb{E}[(\hat{\boldsymbol{\theta}} - \mathbb{E}[\hat{\boldsymbol{\theta}}])^2]$

**Mean squared error**: $\text{MSE}(\hat{\boldsymbol{\theta}}) = \mathbb{E}[(\hat{\boldsymbol{\theta}} - \boldsymbol{\theta})^2] = \text{Var}(\hat{\boldsymbol{\theta}}) + \text{Bias}(\hat{\boldsymbol{\theta}})^2$

**Consistency**: $\hat{\boldsymbol{\theta}}_n \xrightarrow{p} \boldsymbol{\theta}$ as $n \to \infty$ (converges in probability to true value)

**Efficiency**: Among unbiased estimators, prefer one with smallest variance.

### Method of Moments

**Method of moments**: Equate sample moments to population moments:

$$\frac{1}{n}\sum_{i=1}^n x_i^k = \mathbb{E}[X^k] \quad \text{for } k = 1, 2, \ldots$$

Solve for parameters.

**Example**: For $\mathcal{N}(\mu, \sigma^2)$:
- $\hat{\mu} = \bar{x}$ (first moment)
- $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$ (second moment)

## Maximum Likelihood Estimation

### Likelihood Function

**Likelihood**: For i.i.d. data $\mathcal{D} = \{x_1, \ldots, x_n\}$:

$$L(\boldsymbol{\theta}) = \prod_{i=1}^n p(x_i | \boldsymbol{\theta})$$

**Log-likelihood**:

$$\ell(\boldsymbol{\theta}) = \log L(\boldsymbol{\theta}) = \sum_{i=1}^n \log p(x_i | \boldsymbol{\theta})$$

**Interpretation**: Probability of observing data $\mathcal{D}$ given parameters $\boldsymbol{\theta}$ (as function of $\boldsymbol{\theta}$).

### Maximum Likelihood Estimator

**MLE**: $\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) = \arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta})$

**Computation**: Solve $\nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}) = \mathbf{0}$ (score equations).

### Examples

**Gaussian mean**: For $X_i \sim \mathcal{N}(\mu, \sigma^2)$ with known $\sigma^2$:

$$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

Score: $\frac{d\ell}{d\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$

Solution: $\hat{\mu}_{\text{MLE}} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$

**Bernoulli parameter**: For $X_i \sim \text{Bernoulli}(p)$:

$$\ell(p) = \sum_{i=1}^n [x_i \log p + (1-x_i)\log(1-p)]$$

Score: $\frac{d\ell}{dp} = \frac{\sum x_i}{p} - \frac{n - \sum x_i}{1-p} = 0$

Solution: $\hat{p}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i$ (sample proportion)

**Exponential rate**: For $X_i \sim \text{Exp}(\lambda)$:

$$\ell(\lambda) = n\log\lambda - \lambda\sum_{i=1}^n x_i$$

Score: $\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$

Solution: $\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}$

### Properties of MLE

**Invariance**: If $\hat{\theta}$ is MLE of $\theta$, then $g(\hat{\theta})$ is MLE of $g(\theta)$ for one-to-one $g$.

**Asymptotic normality**: Under regularity conditions:

$$\sqrt{n}(\hat{\boldsymbol{\theta}}_{\text{MLE}} - \boldsymbol{\theta}^*) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathbf{I}^{-1}(\boldsymbol{\theta}^*))$$

where $\mathbf{I}(\boldsymbol{\theta})$ is Fisher information matrix.

**Asymptotic efficiency**: MLE achieves Cramér-Rao lower bound asymptotically (minimum variance among consistent estimators).

**Consistency**: $\hat{\boldsymbol{\theta}}_{\text{MLE}} \xrightarrow{p} \boldsymbol{\theta}^*$ under regularity conditions.

### Fisher Information

**Fisher information matrix**:

$$\mathbf{I}(\boldsymbol{\theta})_{ij} = \mathbb{E}\left[\frac{\partial \ell}{\partial \theta_i} \frac{\partial \ell}{\partial \theta_j}\right] = -\mathbb{E}\left[\frac{\partial^2 \ell}{\partial \theta_i \partial \theta_j}\right]$$

**Cramér-Rao bound**: For unbiased estimator $\hat{\boldsymbol{\theta}}$:

$$\text{Cov}(\hat{\boldsymbol{\theta}}) \succeq \mathbf{I}^{-1}(\boldsymbol{\theta})$$

MLE asymptotically achieves this bound.

## Maximum A Posteriori Estimation

### Bayesian Framework

**Prior**: $p(\boldsymbol{\theta})$ (belief about parameters before seeing data)

**Posterior**: $p(\boldsymbol{\theta} | \mathcal{D}) = \frac{p(\mathcal{D} | \boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\mathcal{D})} \propto p(\mathcal{D} | \boldsymbol{\theta}) p(\boldsymbol{\theta})$

**MAP estimator**: $\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} p(\boldsymbol{\theta} | \mathcal{D}) = \arg\max_{\boldsymbol{\theta}} [\ell(\boldsymbol{\theta}) + \log p(\boldsymbol{\theta})]$

**Interpretation**: Mode of posterior distribution.

### MAP vs MLE

**MLE**: $\hat{\boldsymbol{\theta}} = \arg\max \ell(\boldsymbol{\theta})$ (ignores prior)

**MAP**: $\hat{\boldsymbol{\theta}} = \arg\max [\ell(\boldsymbol{\theta}) + \log p(\boldsymbol{\theta})]$ (incorporates prior)

**Connection**: MAP = MLE when prior is uniform (non-informative).

**Regularization**: MAP with Gaussian prior $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$ gives L2 regularization:

$$\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max [\ell(\boldsymbol{\theta}) - \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2]$$

### Conjugate Priors

**Conjugate prior**: Prior $p(\boldsymbol{\theta})$ is conjugate to likelihood $p(\mathbf{x} | \boldsymbol{\theta})$ if posterior $p(\boldsymbol{\theta} | \mathcal{D})$ has same form as prior.

**Examples**:
- **Gaussian-Gaussian**: Prior $\mathcal{N}(\mu_0, \sigma_0^2)$, likelihood $\mathcal{N}(\mu, \sigma^2)$, posterior $\mathcal{N}(\mu_n, \sigma_n^2)$
- **Beta-Bernoulli**: Prior $\text{Beta}(\alpha, \beta)$, likelihood $\text{Bernoulli}(p)$, posterior $\text{Beta}(\alpha + \sum x_i, \beta + n - \sum x_i)$
- **Gamma-Poisson**: Prior $\text{Gamma}(\alpha, \beta)$, likelihood $\text{Poisson}(\lambda)$, posterior $\text{Gamma}(\alpha + \sum x_i, \beta + n)$

## Bayesian Inference

### Posterior Distribution

**Full Bayesian inference**: Use entire posterior $p(\boldsymbol{\theta} | \mathcal{D})$, not just mode.

**Posterior mean**: $\mathbb{E}[\boldsymbol{\theta} | \mathcal{D}] = \int \boldsymbol{\theta} p(\boldsymbol{\theta} | \mathcal{D}) d\boldsymbol{\theta}$

**Posterior variance**: $\text{Var}(\boldsymbol{\theta} | \mathcal{D})$ quantifies uncertainty.

### Predictive Distribution

**Posterior predictive**: For new observation $\mathbf{x}_{\text{new}}$:

$$p(\mathbf{x}_{\text{new}} | \mathcal{D}) = \int p(\mathbf{x}_{\text{new}} | \boldsymbol{\theta}) p(\boldsymbol{\theta} | \mathcal{D}) d\boldsymbol{\theta}$$

**Interpretation**: Average predictions weighted by posterior probability.

**Example**: For Gaussian with conjugate prior:

$$p(x_{\text{new}} | \mathcal{D}) = \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2)$$

(posterior mean, data variance + parameter uncertainty)

### Bayesian Updating

**Sequential updating**: With new data $\mathcal{D}_{\text{new}}$:

$$p(\boldsymbol{\theta} | \mathcal{D}, \mathcal{D}_{\text{new}}) \propto p(\mathcal{D}_{\text{new}} | \boldsymbol{\theta}) p(\boldsymbol{\theta} | \mathcal{D})$$

Previous posterior becomes new prior.

### Computational Methods

**Analytical**: For conjugate priors, posterior has closed form.

**Markov Chain Monte Carlo (MCMC)**: Sample from posterior:
- Metropolis-Hastings
- Gibbs sampling
- Hamiltonian Monte Carlo

**Variational inference**: Approximate posterior $q(\boldsymbol{\theta}) \approx p(\boldsymbol{\theta} | \mathcal{D})$ by minimizing KL divergence.

## Properties of Estimators

### Unbiasedness

**Unbiased**: $\mathbb{E}[\hat{\boldsymbol{\theta}}] = \boldsymbol{\theta}$

**Example**: Sample mean $\bar{X} = \frac{1}{n}\sum X_i$ is unbiased for population mean: $\mathbb{E}[\bar{X}] = \mu$.

**Biased example**: Sample variance $S^2 = \frac{1}{n}\sum (X_i - \bar{X})^2$ has bias:

$$\mathbb{E}[S^2] = \frac{n-1}{n}\sigma^2$$

Unbiased version: $\hat{\sigma}^2 = \frac{1}{n-1}\sum (X_i - \bar{X})^2$.

### Consistency

**Consistency**: $\hat{\boldsymbol{\theta}}_n \xrightarrow{p} \boldsymbol{\theta}$ as $n \to \infty$

**Sufficient conditions**:
- Unbiasedness: $\mathbb{E}[\hat{\boldsymbol{\theta}}_n] = \boldsymbol{\theta}$
- Variance goes to zero: $\lim_{n \to \infty} \text{Var}(\hat{\boldsymbol{\theta}}_n) = 0$

**MLE**: Consistent under regularity conditions.

### Efficiency

**Efficiency**: Ratio of Cramér-Rao bound to actual variance:

$$\text{eff}(\hat{\boldsymbol{\theta}}) = \frac{\mathbf{I}^{-1}(\boldsymbol{\theta})}{\text{Var}(\hat{\boldsymbol{\theta}})}$$

**Efficient estimator**: Achieves efficiency = 1 (attains Cramér-Rao bound).

**MLE**: Asymptotically efficient.

### Sufficiency

**Sufficient statistic**: $T(\mathcal{D})$ is sufficient for $\boldsymbol{\theta}$ if $p(\mathcal{D} | T(\mathcal{D}), \boldsymbol{\theta}) = p(\mathcal{D} | T(\mathcal{D}))$ (doesn't depend on $\boldsymbol{\theta}$).

**Factorization theorem**: $T$ is sufficient if and only if:

$$p(\mathcal{D} | \boldsymbol{\theta}) = h(\mathcal{D}) g(T(\mathcal{D}), \boldsymbol{\theta})$$

**Example**: For exponential family, $T(\mathcal{D}) = \sum_{i=1}^n t(X_i)$ is sufficient.

## Hypothesis Testing

### Null and Alternative Hypotheses

**Null hypothesis**: $H_0: \boldsymbol{\theta} \in \Theta_0$

**Alternative hypothesis**: $H_1: \boldsymbol{\theta} \in \Theta_1$ (often $\Theta_1 = \Theta_0^c$)

**Example**: $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$

### Test Statistics

**Test statistic**: $T(\mathcal{D})$ used to decide between $H_0$ and $H_1$.

**Rejection region**: $R = \{T(\mathcal{D}) : \text{reject } H_0\}$

### Type I and Type II Errors

**Type I error**: Reject $H_0$ when it's true. Probability: $\alpha = P(\text{reject } H_0 | H_0 \text{ true})$

**Type II error**: Accept $H_0$ when it's false. Probability: $\beta = P(\text{accept } H_0 | H_1 \text{ true})$

**Power**: $1 - \beta = P(\text{reject } H_0 | H_1 \text{ true})$

### p-Values

**p-value**: Probability of observing test statistic at least as extreme as observed, assuming $H_0$ is true:

$$p = P(T(\mathcal{D}) \geq t_{\text{obs}} | H_0)$$

**Decision rule**: Reject $H_0$ if $p < \alpha$ (significance level).

**Interpretation**: Small $p$ suggests data is inconsistent with $H_0$.

### Common Tests

**t-test**: For testing mean $\mu = \mu_0$ with unknown variance:

$$t = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t_{n-1}$$

**Chi-square test**: For testing variance or goodness-of-fit.

**F-test**: For comparing variances or in ANOVA.

## Confidence Intervals

### Definition

**Confidence interval**: Random interval $[L(\mathcal{D}), U(\mathcal{D})]$ such that:

$$P(\boldsymbol{\theta} \in [L, U]) = 1 - \alpha$$

**Interpretation**: In repeated sampling, $100(1-\alpha)\%$ of intervals contain true $\boldsymbol{\theta}$.

**Note**: $\boldsymbol{\theta}$ is fixed, interval is random.

### Construction

**Pivotal quantity**: Function $Q(\mathcal{D}, \boldsymbol{\theta})$ with known distribution (doesn't depend on $\boldsymbol{\theta}$).

**Example**: For $\mathcal{N}(\mu, \sigma^2)$ with known $\sigma^2$:

$$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0, 1)$$

**Confidence interval**: $P(-z_{\alpha/2} \leq Z \leq z_{\alpha/2}) = 1 - \alpha$ gives:

$$\bar{X} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$$

### Asymptotic Confidence Intervals

**Wald interval**: Using asymptotic normality of MLE:

$$\hat{\boldsymbol{\theta}} \pm z_{\alpha/2} \sqrt{\mathbf{I}^{-1}(\hat{\boldsymbol{\theta}})/n}$$

**Likelihood ratio**: Based on likelihood ratio test statistic.

## Bootstrap Methods

### Bootstrap Principle

**Bootstrap**: Resample from empirical distribution to estimate sampling distribution of statistic.

**Algorithm**:
1. Draw bootstrap sample $\mathcal{D}^*_b = \{x^*_{b1}, \ldots, x^*_{bn}\}$ by sampling with replacement from $\mathcal{D}$
2. Compute statistic $\hat{\boldsymbol{\theta}}^*_b = T(\mathcal{D}^*_b)$
3. Repeat $B$ times
4. Use $\{\hat{\boldsymbol{\theta}}^*_1, \ldots, \hat{\boldsymbol{\theta}}^*_B\}$ to estimate distribution

### Bootstrap Confidence Intervals

**Percentile method**: $[\hat{\boldsymbol{\theta}}^*_{(\alpha/2)}, \hat{\boldsymbol{\theta}}^*_{(1-\alpha/2)}]$ (percentiles of bootstrap distribution)

**Bias-corrected**: Adjust for bias in bootstrap distribution.

**Advantages**: No distributional assumptions, works for complex statistics.

## Machine Learning Applications

### Parameter Estimation in Models

**Linear regression**: MLE gives normal equations:

$$\hat{\boldsymbol{\theta}}_{\text{MLE}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Logistic regression**: MLE via iterative optimization (Newton's method):

$$\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max \sum_{i=1}^n [y_i \log\sigma(\boldsymbol{\theta}^T\mathbf{x}_i) + (1-y_i)\log(1-\sigma(\boldsymbol{\theta}^T\mathbf{x}_i))]$$

### Regularization as MAP

**Ridge regression**: MAP with Gaussian prior:

$$\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|^2$$

**Lasso**: MAP with Laplace prior:

$$\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|_1$$

### Bayesian Neural Networks

**Posterior over weights**: $p(\mathbf{W} | \mathcal{D})$ captures uncertainty.

**Predictive distribution**: $p(y | \mathbf{x}, \mathcal{D}) = \int p(y | \mathbf{x}, \mathbf{W}) p(\mathbf{W} | \mathcal{D}) d\mathbf{W}$

**Approximation**: Variational inference or MCMC to approximate $p(\mathbf{W} | \mathcal{D})$.

### Model Selection

**Bayesian model comparison**: Compare models via marginal likelihood:

$$p(\mathcal{D} | \mathcal{M}_i) = \int p(\mathcal{D} | \boldsymbol{\theta}_i, \mathcal{M}_i) p(\boldsymbol{\theta}_i | \mathcal{M}_i) d\boldsymbol{\theta}_i$$

**Bayes factor**: $B_{12} = \frac{p(\mathcal{D} | \mathcal{M}_1)}{p(\mathcal{D} | \mathcal{M}_2)}$

### Uncertainty Quantification

**Prediction intervals**: From posterior predictive distribution:

$$P(y \in [L(\mathbf{x}), U(\mathbf{x})] | \mathbf{x}, \mathcal{D}) = 1 - \alpha$$

**Calibration**: Ensure predicted probabilities match empirical frequencies.

### A/B Testing

**Hypothesis test**: $H_0$: No difference between groups A and B

**Test statistic**: Difference in means, conversion rates, etc.

**Decision**: Reject $H_0$ if $p < 0.05$ (or chosen significance level).

## Key Takeaways

1. **MLE** maximizes likelihood function, providing point estimates that are consistent and asymptotically efficient.

2. **MAP** incorporates prior beliefs via Bayesian framework, equivalent to regularization in many cases.

3. **Bayesian inference** uses full posterior distribution, providing uncertainty quantification beyond point estimates.

4. **Unbiasedness** ensures estimator is correct on average, but biased estimators may have lower MSE.

5. **Consistency** ensures estimator converges to true value as sample size increases.

6. **Hypothesis testing** provides framework for making decisions under uncertainty with controlled error rates.

7. **Confidence intervals** quantify uncertainty in parameter estimates, more informative than point estimates alone.

8. **Bootstrap** provides distribution-free method for estimating sampling distributions and confidence intervals.

9. **Regularization** (L2, L1) corresponds to MAP estimation with appropriate priors (Gaussian, Laplace).

10. **Statistical inference** provides principled framework for learning from data, quantifying uncertainty, and making predictions in ML.
