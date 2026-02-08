# Hypothesis Testing and Confidence Intervals

## Neyman-Pearson Framework

### Components

A hypothesis test consists of:

1. **Null hypothesis** $H_0$: hypothesis to be tested
2. **Alternative hypothesis** $H_1$ (or $H_a$): alternative to null
3. **Test statistic** $T(X_1, \ldots, X_n)$: function of data
4. **Rejection region** $R$: set of values of $T$ leading to rejection of $H_0$
5. **Significance level** $\alpha$: maximum probability of Type I error

### Types of Hypotheses

- **Simple vs composite**: $H_0: \theta = \theta_0$ (simple) vs $H_0: \theta \leq \theta_0$ (composite)
- **One-sided vs two-sided**: $H_1: \theta > \theta_0$ vs $H_1: \theta \neq \theta_0$

### Decision Rules

- **Reject** $H_0$ if $T \in R$
- **Fail to reject** (or accept) $H_0$ if $T \notin R$

## Type I and Type II Errors

### Type I Error

Rejecting $H_0$ when it is true:

$$\alpha = \mathbb{P}(\text{reject } H_0 | H_0 \text{ true}) = \mathbb{P}(T \in R | H_0)$$

### Type II Error

Failing to reject $H_0$ when $H_1$ is true:

$$\beta = \mathbb{P}(\text{fail to reject } H_0 | H_1 \text{ true}) = \mathbb{P}(T \notin R | H_1)$$

### Power

The power of a test is:

$$\text{Power} = 1 - \beta = \mathbb{P}(\text{reject } H_0 | H_1 \text{ true})$$

**Desirable properties**:
- High power for alternatives of interest
- Power function $\pi(\theta) = \mathbb{P}_\theta(\text{reject } H_0)$ should be large for $\theta \in H_1$

## Likelihood Ratio Tests

### Definition

For testing $H_0: \theta \in \Theta_0$ vs $H_1: \theta \in \Theta_1$, the likelihood ratio statistic is:

$$\Lambda(X) = \frac{\sup_{\theta \in \Theta_0} L(\theta; X)}{\sup_{\theta \in \Theta} L(\theta; X)} = \frac{L(\hat{\theta}_0; X)}{L(\hat{\theta}; X)}$$

where $\hat{\theta}_0$ is MLE under $H_0$ and $\hat{\theta}$ is unrestricted MLE.

**Rejection region**: Reject $H_0$ if $\Lambda(X) \leq c$ for some critical value $c$.

### Neyman-Pearson Lemma

For simple hypotheses $H_0: \theta = \theta_0$ vs $H_1: \theta = \theta_1$, the most powerful test at level $\alpha$ rejects $H_0$ when:

$$\frac{L(\theta_1; X)}{L(\theta_0; X)} > k$$

where $k$ is chosen so that $\mathbb{P}_{\theta_0}(\text{reject}) = \alpha$.

### Wilks' Theorem

Under regularity conditions, if $H_0$ is true:

$$-2\ln \Lambda(X) \xrightarrow{d} \chi^2_r$$

where $r = \dim(\Theta) - \dim(\Theta_0)$ is the number of restrictions.

## p-Values

### Definition

The p-value is the smallest significance level at which $H_0$ would be rejected:

$$p = \inf\{\alpha : T \in R_\alpha\}$$

Equivalently, the probability under $H_0$ of observing a test statistic at least as extreme as the observed value:

$$p = \mathbb{P}_{H_0}(T \geq t_{\text{obs}})$$

### Interpretation

- **Small p-value** ($p < \alpha$): evidence against $H_0$
- **Large p-value** ($p \geq \alpha$): insufficient evidence to reject $H_0$

**Common misinterpretations**:
- p-value is NOT the probability that $H_0$ is true
- p-value does NOT measure effect size
- p-value depends on sample size

## Significance Levels

Common choices: $\alpha = 0.05$, $0.01$, $0.10$.

**Bonferroni correction**: For $m$ tests, use $\alpha/m$ for each test to maintain family-wise error rate $\alpha$.

**False Discovery Rate (FDR)**: For $m$ tests, control expected proportion of false discoveries:

$$\text{FDR} = E\left[\frac{V}{R}\right]$$

where $V$ is number of false rejections and $R$ is total rejections.

**Benjamini-Hochberg procedure**: Order p-values $p_{(1)} \leq \cdots \leq p_{(m)}$ and reject hypotheses with:

$$p_{(i)} \leq \frac{i\alpha}{m}$$

## Common Tests

### One-Sample t-Test

**Hypothesis**: $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$

**Test statistic** (when $\sigma$ unknown):

$$t = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t(n-1)$$

under $H_0$ if $X_i \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d.

**Rejection region**: $|t| > t_{\alpha/2, n-1}$

### Two-Sample t-Test

**Hypothesis**: $H_0: \mu_1 = \mu_2$ vs $H_1: \mu_1 \neq \mu_2$

**Equal variances** (pooled):

$$t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{1/n_1 + 1/n_2}} \sim t(n_1 + n_2 - 2)$$

where $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1 + n_2 - 2}$

**Unequal variances** (Welch's test):

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}} \sim t(\nu)$$

where $\nu = \frac{(S_1^2/n_1 + S_2^2/n_2)^2}{(S_1^2/n_1)^2/(n_1-1) + (S_2^2/n_2)^2/(n_2-1)}$

### Chi-Squared Test

**Goodness of fit**: Test if data follows distribution with probabilities $p_1, \ldots, p_k$:

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} \sim \chi^2(k-1-r)$$

where $O_i$ are observed counts, $E_i = np_i$ are expected counts, and $r$ is number of estimated parameters.

**Test of independence**: For contingency table:

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \sim \chi^2((r-1)(c-1))$$

where $E_{ij} = \frac{(\text{row } i \text{ total})(\text{col } j \text{ total})}{n}$

### F-Test

**Hypothesis**: $H_0: \sigma_1^2 = \sigma_2^2$ vs $H_1: \sigma_1^2 \neq \sigma_2^2$

**Test statistic**:

$$F = \frac{S_1^2}{S_2^2} \sim F(n_1-1, n_2-1)$$

under $H_0$ if $X_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$.

**ANOVA**: Test equality of means across $k$ groups:

$$F = \frac{\text{MSB}}{\text{MSE}} = \frac{\sum_{i=1}^{k} n_i(\bar{X}_i - \bar{X})^2/(k-1)}{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(X_{ij} - \bar{X}_i)^2/(n-k)} \sim F(k-1, n-k)$$

### Non-Parametric Tests

**Kolmogorov-Smirnov test**: Test if sample comes from distribution $F_0$:

$$D_n = \sup_x |F_n(x) - F_0(x)|$$

where $F_n$ is empirical CDF. Reject if $D_n > D_{\alpha, n}$.

**Mann-Whitney U test** (Wilcoxon rank-sum): Test if two samples have same distribution:

$$U = \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} \mathbf{1}_{X_i < Y_j}$$

Under $H_0$ (equal distributions), $U$ has known distribution.

**Kruskal-Wallis test**: Non-parametric ANOVA, tests equality of medians.

## Confidence Intervals

### Definition

A $(1-\alpha)$ confidence interval for parameter $\theta$ is an interval $[L(X), U(X)]$ such that:

$$\mathbb{P}_\theta(L(X) \leq \theta \leq U(X)) \geq 1 - \alpha$$

for all $\theta$.

**Interpretation**: In repeated sampling, $(1-\alpha)$ proportion of intervals contain true $\theta$.

### Construction Methods

**Pivotal quantity method**: Find function $Q(X, \theta)$ whose distribution doesn't depend on $\theta$. Then:

$$\mathbb{P}(a \leq Q(X, \theta) \leq b) = 1 - \alpha$$

Solve for $\theta$ to get confidence interval.

**Example**: For normal mean with known variance:

$$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0, 1)$$

Then:

$$\mathbb{P}\left(-z_{\alpha/2} \leq \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \leq z_{\alpha/2}\right) = 1 - \alpha$$

Solving: $\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$

**Inversion of hypothesis tests**: If test of $H_0: \theta = \theta_0$ at level $\alpha$ accepts when $T \in A(\theta_0)$, then:

$$C(X) = \{\theta : T(X) \in A(\theta)\}$$

is a $(1-\alpha)$ confidence set.

### Common Confidence Intervals

**Normal mean** (known variance):

$$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

**Normal mean** (unknown variance):

$$\bar{X} \pm t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}$$

**Normal variance**:

$$\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}, \frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}\right]$$

**Difference of means** (equal variances):

$$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, n_1+n_2-2} S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$

**Proportion** (large sample):

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

### Bootstrap Confidence Intervals

**Percentile method**: 
1. Generate bootstrap samples $X_1^*, \ldots, X_B^*$
2. Compute $\hat{\theta}_b^*$ for each sample
3. Confidence interval: $[\hat{\theta}_{(\alpha/2)}^*, \hat{\theta}_{(1-\alpha/2)}^*]$

**Bias-corrected and accelerated (BCa)**: Adjusts for bias and skewness.

**Bootstrap-t**: Uses bootstrap to estimate distribution of t-statistic.

## Power Analysis

### Sample Size Determination

For test $H_0: \mu = \mu_0$ vs $H_1: \mu = \mu_1$ with power $1-\beta$:

$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{(\mu_1 - \mu_0)^2}$$

**Two-sample case**:

$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{(\mu_1 - \mu_2)^2}$$

### Effect Size

Cohen's $d$:

$$d = \frac{\mu_1 - \mu_0}{\sigma}$$

**Interpretation**:
- Small: $d = 0.2$
- Medium: $d = 0.5$
- Large: $d = 0.8$

## Multiple Testing Corrections

### Family-Wise Error Rate (FWER)

Control probability of at least one Type I error:

$$\text{FWER} = \mathbb{P}(\text{at least one false rejection})$$

**Bonferroni correction**: For $m$ tests, use $\alpha/m$ for each test:

$$\text{FWER} \leq m \cdot \frac{\alpha}{m} = \alpha$$

**Holm-Bonferroni**: Step-down procedure:
1. Order p-values: $p_{(1)} \leq \cdots \leq p_{(m)}$
2. Reject $H_{(i)}$ if $p_{(i)} \leq \alpha/(m-i+1)$

### False Discovery Rate

**Benjamini-Hochberg**: Control expected proportion of false discoveries:

1. Order p-values: $p_{(1)} \leq \cdots \leq p_{(m)}$
2. Find largest $k$ such that $p_{(k)} \leq k\alpha/m$
3. Reject $H_{(1)}, \ldots, H_{(k)}$

**Benjamini-Yekutieli**: More conservative, works under arbitrary dependence.

## Applications in Finance

### Testing Strategy Significance

**Sharpe ratio test**: Test if Sharpe ratio $\text{SR} = \frac{\mu - r_f}{\sigma}$ exceeds threshold:

$$t = \frac{\hat{\text{SR}} - \text{SR}_0}{\hat{\text{SE}}(\hat{\text{SR}})}$$

where $\hat{\text{SE}}(\hat{\text{SR}}) = \sqrt{\frac{1 + \hat{\text{SR}}^2/2}{n}}$

**Information ratio test**: Test if IR exceeds zero:

$$t = \frac{\bar{R}_p - \bar{R}_b}{S_d/\sqrt{n}}$$

where $S_d$ is standard deviation of tracking error.

### Backtesting Statistical Tests

**Kupiec test**: Test if VaR violations occur at expected rate:

$$LR = -2\ln\left((1-p)^{T-x}p^x\right) + 2\ln\left((1-x/T)^{T-x}(x/T)^x\right) \sim \chi^2(1)$$

where $x$ is number of violations, $T$ is total periods, $p$ is expected violation rate.

**Christoffersen test**: Tests independence of violations:

$$LR_{\text{ind}} = -2\ln\left(\frac{(1-\pi_{01})^{n_{00}}\pi_{01}^{n_{01}}(1-\pi_{11})^{n_{10}}\pi_{11}^{n_{11}}}{(1-\pi)^n\pi^x}\right) \sim \chi^2(1)$$

where $\pi_{ij}$ are transition probabilities and $\pi$ is unconditional violation rate.

### Factor Model Testing

**Fama-MacBeth**: Two-step regression:
1. Time-series: $R_{it} = \alpha_i + \boldsymbol{\beta}_i^T \mathbf{F}_t + \epsilon_{it}$
2. Cross-section: $R_i = \lambda_0 + \boldsymbol{\lambda}^T \boldsymbol{\beta}_i + u_i$

Test $H_0: \lambda_j = 0$ for factor $j$:

$$t = \frac{\bar{\lambda}_j}{\text{SE}(\bar{\lambda}_j)}$$

where $\bar{\lambda}_j = \frac{1}{T}\sum_{t=1}^{T} \lambda_{jt}$.

### Cointegration Tests

**Engle-Granger**: Test if residuals from cointegrating regression are stationary:

$$ADF: \Delta \hat{\epsilon}_t = \alpha + \rho \hat{\epsilon}_{t-1} + \sum_{i=1}^{p} \gamma_i \Delta \hat{\epsilon}_{t-i} + u_t$$

Test $H_0: \rho = 0$ (no cointegration).

**Johansen test**: Tests number of cointegrating relationships using eigenvalues of matrix.

### Structural Break Tests

**Chow test**: Test for structural break at known time $t_0$:

$$F = \frac{(SSR_{\text{pooled}} - SSR_1 - SSR_2)/k}{(SSR_1 + SSR_2)/(n-2k)} \sim F(k, n-2k)$$

where $SSR$ are sum of squared residuals.

**CUSUM test**: Tests for unknown break points using cumulative sums of residuals.
