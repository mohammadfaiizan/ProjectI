# Limit Theorems and Convergence

## Modes of Convergence

### Almost Sure Convergence

A sequence $\{X_n\}$ converges almost surely to $X$ (denoted $X_n \xrightarrow{a.s.} X$) if:

$$\mathbb{P}\left(\lim_{n \to \infty} X_n = X\right) = 1$$

Equivalently:

$$\mathbb{P}\left(\{\omega : \lim_{n \to \infty} X_n(\omega) = X(\omega)\}\right) = 1$$

**Properties**:
- Strongest form of convergence
- Implies convergence in probability
- Preserved under continuous functions

### Convergence in Probability

$X_n \xrightarrow{p} X$ if for every $\epsilon > 0$:

$$\lim_{n \to \infty} \mathbb{P}(|X_n - X| > \epsilon) = 0$$

**Properties**:
- Weaker than almost sure convergence
- Implies convergence in distribution
- If $X_n \xrightarrow{p} c$ (constant), then $X_n \xrightarrow{d} c$

### Convergence in Distribution

$X_n \xrightarrow{d} X$ (or $X_n \xrightarrow{\mathcal{L}} X$) if:

$$\lim_{n \to \infty} F_{X_n}(x) = F_X(x)$$

for all continuity points $x$ of $F_X$.

**Equivalent conditions**:
- Pointwise convergence of CDFs at continuity points
- Convergence of characteristic functions: $\phi_{X_n}(t) \to \phi_X(t)$ for all $t$
- Convergence of MGFs (if they exist in a neighborhood of 0)

**Properties**:
- Weakest form of convergence
- Continuous mapping theorem: If $g$ is continuous, then $X_n \xrightarrow{d} X$ implies $g(X_n) \xrightarrow{d} g(X)$
- Slutsky's theorem: If $X_n \xrightarrow{d} X$ and $Y_n \xrightarrow{p} c$, then $X_n + Y_n \xrightarrow{d} X + c$ and $X_n Y_n \xrightarrow{d} cX$

### Convergence in $L^p$

$X_n \xrightarrow{L^p} X$ if:

$$\lim_{n \to \infty} E[|X_n - X|^p] = 0$$

**Special cases**:
- $L^2$ convergence (mean square convergence): $E[(X_n - X)^2] \to 0$
- $L^1$ convergence (convergence in mean): $E[|X_n - X|] \to 0$

**Properties**:
- Implies convergence in probability
- For $p \geq 1$, $L^p$ convergence implies $L^q$ convergence for $q < p$ (on bounded spaces)

### Relationships

$$\text{a.s.} \Rightarrow \text{in probability} \Rightarrow \text{in distribution}$$

$$L^p \Rightarrow \text{in probability} \Rightarrow \text{in distribution}$$

None of these implications are reversible in general.

## Weak Law of Large Numbers

### Statement

Let $X_1, X_2, \ldots$ be i.i.d. random variables with $E[|X_1|] < \infty$ and $E[X_1] = \mu$. Then:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{p} \mu$$

**Proof sketch**: Using Chebyshev's inequality:

$$\mathbb{P}(|\bar{X}_n - \mu| > \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} \to 0$$

as $n \to \infty$.

### Conditions

- **Finite variance**: If $\text{Var}(X_1) < \infty$, WLLN holds
- **Khintchine's WLLN**: If $E[|X_1|] < \infty$, WLLN holds (no variance assumption needed)

## Strong Law of Large Numbers

### Statement

Let $X_1, X_2, \ldots$ be i.i.d. random variables with $E[|X_1|] < \infty$ and $E[X_1] = \mu$. Then:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{a.s.} \mu$$

**Kolmogorov's SLLN**: If $E[|X_1|] < \infty$, then SLLN holds.

**Kolmogorov's sufficient condition**: If $\sum_{n=1}^{\infty} \frac{\text{Var}(X_n)}{n^2} < \infty$, then SLLN holds (even without identical distribution).

### Applications

- **Monte Carlo integration**: $\frac{1}{n}\sum_{i=1}^{n} f(X_i) \xrightarrow{a.s.} E[f(X)]$
- **Sample moments**: $\frac{1}{n}\sum_{i=1}^{n} X_i^k \xrightarrow{a.s.} E[X^k]$
- **Empirical distribution**: $\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}_{X_i \leq x} \xrightarrow{a.s.} F(x)$

## Central Limit Theorem

### Lindeberg-Lévy CLT

Let $X_1, X_2, \ldots$ be i.i.d. with $E[X_1] = \mu$ and $0 < \text{Var}(X_1) = \sigma^2 < \infty$. Then:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} = \frac{\sum_{i=1}^{n}(X_i - \mu)}{\sigma\sqrt{n}} \xrightarrow{d} Z \sim \mathcal{N}(0, 1)$$

Equivalently:

$$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

**Proof outline**: Using characteristic functions:

$$\phi_{\bar{X}_n}(t) = \left(\phi_{X_1}\left(\frac{t}{n}\right)\right)^n = \left(1 + i\mu\frac{t}{n} - \frac{\sigma^2 t^2}{2n} + o\left(\frac{1}{n}\right)\right)^n \to e^{i\mu t - \sigma^2 t^2/2}$$

### Lyapunov CLT

Let $X_1, X_2, \ldots$ be independent (not necessarily identically distributed) with $E[X_i] = \mu_i$, $\text{Var}(X_i) = \sigma_i^2$, and $s_n^2 = \sum_{i=1}^{n} \sigma_i^2$. If Lyapunov's condition holds:

$$\lim_{n \to \infty} \frac{1}{s_n^{2+\delta}} \sum_{i=1}^{n} E[|X_i - \mu_i|^{2+\delta}] = 0$$

for some $\delta > 0$, then:

$$\frac{\sum_{i=1}^{n}(X_i - \mu_i)}{s_n} \xrightarrow{d} \mathcal{N}(0, 1)$$

### Lindeberg-Feller CLT

Let $X_1, X_2, \ldots$ be independent with $E[X_i] = \mu_i$, $\text{Var}(X_i) = \sigma_i^2$, and $s_n^2 = \sum_{i=1}^{n} \sigma_i^2$. If Lindeberg's condition holds:

$$\lim_{n \to \infty} \frac{1}{s_n^2} \sum_{i=1}^{n} E[(X_i - \mu_i)^2 \mathbf{1}_{|X_i - \mu_i| > \epsilon s_n}] = 0$$

for all $\epsilon > 0$, then:

$$\frac{\sum_{i=1}^{n}(X_i - \mu_i)}{s_n} \xrightarrow{d} \mathcal{N}(0, 1)$$

and Feller's condition:

$$\lim_{n \to \infty} \max_{1 \leq i \leq n} \frac{\sigma_i^2}{s_n^2} = 0$$

### Multivariate CLT

Let $\mathbf{X}_1, \mathbf{X}_2, \ldots$ be i.i.d. $d$-dimensional random vectors with $E[\mathbf{X}_1] = \boldsymbol{\mu}$ and $\text{Cov}(\mathbf{X}_1) = \boldsymbol{\Sigma}$. Then:

$$\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu}) \xrightarrow{d} \mathcal{N}_d(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\bar{\mathbf{X}}_n = \frac{1}{n}\sum_{i=1}^{n} \mathbf{X}_i$.

### Delta Method

If $\sqrt{n}(X_n - \theta) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$ and $g$ is differentiable at $\theta$ with $g'(\theta) \neq 0$, then:

$$\sqrt{n}(g(X_n) - g(\theta)) \xrightarrow{d} \mathcal{N}(0, [g'(\theta)]^2 \sigma^2)$$

**Multivariate delta method**: If $\sqrt{n}(\mathbf{X}_n - \boldsymbol{\theta}) \xrightarrow{d} \mathcal{N}_d(\mathbf{0}, \boldsymbol{\Sigma})$ and $g: \mathbb{R}^d \to \mathbb{R}$ is differentiable at $\boldsymbol{\theta}$ with gradient $\nabla g(\boldsymbol{\theta})$, then:

$$\sqrt{n}(g(\mathbf{X}_n) - g(\boldsymbol{\theta})) \xrightarrow{d} \mathcal{N}(0, [\nabla g(\boldsymbol{\theta})]^T \boldsymbol{\Sigma} [\nabla g(\boldsymbol{\theta})])$$

**Example**: For sample variance $S_n^2$:

$$\sqrt{n}(S_n^2 - \sigma^2) \xrightarrow{d} \mathcal{N}(0, \mu_4 - \sigma^4)$$

where $\mu_4 = E[(X - \mu)^4]$.

## Berry-Esseen Theorem

### Statement

Let $X_1, X_2, \ldots$ be i.i.d. with $E[X_1] = 0$, $\text{Var}(X_1) = \sigma^2$, and $E[|X_1|^3] < \infty$. Then:

$$\sup_{x \in \mathbb{R}} \left|F_{\bar{X}_n}(x) - \Phi\left(\frac{x\sqrt{n}}{\sigma}\right)\right| \leq \frac{C E[|X_1|^3]}{\sigma^3 \sqrt{n}}$$

where $C$ is a universal constant ($C \approx 0.4748$).

**Interpretation**: Provides uniform bound on convergence rate in CLT. The error decreases as $O(1/\sqrt{n})$.

### Applications

- **Confidence intervals**: Berry-Esseen provides finite-sample error bounds
- **Sample size determination**: Ensures CLT approximation is sufficiently accurate

## Convergence Rates

### Rate of Convergence

A sequence $\{X_n\}$ converges to $X$ at rate $r_n$ if:

$$r_n(X_n - X) \xrightarrow{d} Y$$

for some non-degenerate $Y$ and rate sequence $r_n \to \infty$.

**Examples**:
- CLT: $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$ (rate $\sqrt{n}$)
- Sample variance: $\sqrt{n}(S_n^2 - \sigma^2) \xrightarrow{d} \mathcal{N}(0, \mu_4 - \sigma^4)$ (rate $\sqrt{n}$)

### Asymptotic Efficiency

An estimator $\hat{\theta}_n$ is asymptotically efficient if:

$$\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})$$

where $I(\theta)$ is the Fisher information. The Cramér-Rao bound shows this is the best possible rate.

## Slutsky's Theorem

If $X_n \xrightarrow{d} X$ and $Y_n \xrightarrow{p} c$ (constant), then:

1. $X_n + Y_n \xrightarrow{d} X + c$
2. $X_n Y_n \xrightarrow{d} cX$
3. $X_n / Y_n \xrightarrow{d} X / c$ (if $c \neq 0$)

**Applications**:
- **t-statistic**: If $\bar{X}_n \xrightarrow{d} \mathcal{N}(\mu, \sigma^2/n)$ and $S_n \xrightarrow{p} \sigma$, then:
  $$t_n = \frac{\bar{X}_n - \mu}{S_n/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

- **Sample correlation**: Asymptotic distribution of sample correlation coefficient

## Continuous Mapping Theorem

If $X_n \xrightarrow{d} X$ (or $X_n \xrightarrow{p} X$ or $X_n \xrightarrow{a.s.} X$) and $g$ is continuous, then:

$$g(X_n) \xrightarrow{d} g(X)$$

**Extensions**:
- **Continuous at a.s. points**: If $g$ is continuous at all points in a set $A$ with $\mathbb{P}(X \in A) = 1$, then the result holds
- **Vector case**: If $\mathbf{X}_n \xrightarrow{d} \mathbf{X}$ and $g: \mathbb{R}^d \to \mathbb{R}^k$ is continuous, then $g(\mathbf{X}_n) \xrightarrow{d} g(\mathbf{X})$

## Applications in Finance

### Portfolio Diversification

For portfolio return $R_p = \sum_{i=1}^{n} w_i R_i$ with i.i.d. returns $R_i$:

$$\text{Var}(R_p) = \sum_{i=1}^{n} w_i^2 \sigma^2 = \sigma^2 \sum_{i=1}^{n} w_i^2$$

For equal weights $w_i = 1/n$:

$$\text{Var}(R_p) = \frac{\sigma^2}{n} \to 0$$

as $n \to \infty$ (diversification benefit). However, if returns are correlated:

$$\text{Var}(R_p) = \frac{\sigma^2}{n} + \frac{n-1}{n} \rho \sigma^2 \to \rho \sigma^2$$

as $n \to \infty$, showing correlation limits diversification.

### Risk Aggregation

For portfolio loss $L = \sum_{i=1}^{n} L_i$ with independent losses:

$$\frac{L - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

This enables normal approximation for large portfolios:

$$\mathbb{P}(L > \ell) \approx 1 - \Phi\left(\frac{\ell - n\mu}{\sigma\sqrt{n}}\right)$$

### Option Pricing

In Black-Scholes, stock price $S_T = S_0 e^{(r-\sigma^2/2)T + \sigma\sqrt{T}Z}$ where $Z \sim \mathcal{N}(0,1)$. The CLT justifies the normal approximation for log returns over many small time steps.

### Statistical Inference

**Confidence intervals**: Using CLT:

$$\bar{X}_n \pm z_{\alpha/2} \frac{S_n}{\sqrt{n}}$$

is an approximate $(1-\alpha)$ confidence interval for $\mu$.

**Hypothesis testing**: Test statistic:

$$T_n = \frac{\bar{X}_n - \mu_0}{S_n/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

under $H_0: \mu = \mu_0$.

### Monte Carlo Methods

**Monte Carlo integration**: For $I = \int_a^b f(x) dx$:

$$\hat{I}_n = \frac{b-a}{n}\sum_{i=1}^{n} f(X_i)$$

where $X_i \sim \text{Uniform}(a,b)$. By SLLN:

$$\hat{I}_n \xrightarrow{a.s.} I$$

and by CLT:

$$\sqrt{n}(\hat{I}_n - I) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$$

where $\sigma_f^2 = \text{Var}(f(X))$.

**Variance reduction**: Importance sampling uses:

$$\hat{I}_n = \frac{1}{n}\sum_{i=1}^{n} \frac{f(X_i)}{g(X_i)}$$

where $X_i \sim g$ (importance distribution). CLT provides asymptotic confidence intervals.

### Empirical Process Theory

The empirical CDF:

$$F_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}_{X_i \leq x}$$

satisfies:

$$\sqrt{n}(F_n(x) - F(x)) \xrightarrow{d} \mathcal{N}(0, F(x)(1-F(x)))$$

The empirical process $\sqrt{n}(F_n - F)$ converges to a Gaussian process (Donsker's theorem).
