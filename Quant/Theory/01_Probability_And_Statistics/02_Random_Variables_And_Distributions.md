# Random Variables and Distributions

## Random Variables

A random variable $X$ is a measurable function from a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ to $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$. The distribution function (CDF) of $X$ is:

$$F_X(x) = \mathbb{P}(X \leq x) = \mathbb{P}(\{\omega \in \Omega : X(\omega) \leq x\})$$

### Discrete Random Variables

A random variable is discrete if it takes values in a countable set $\{x_1, x_2, \ldots\}$. The probability mass function (PMF) is:

$$p_X(x) = \mathbb{P}(X = x)$$

The CDF is:

$$F_X(x) = \sum_{x_i \leq x} p_X(x_i)$$

### Continuous Random Variables

A random variable is continuous if its CDF is absolutely continuous. The probability density function (PDF) satisfies:

$$F_X(x) = \int_{-\infty}^{x} f_X(t) dt$$

and:

$$f_X(x) = \frac{d}{dx} F_X(x)$$

almost everywhere.

## Discrete Distributions

### Bernoulli Distribution

$X \sim \text{Bernoulli}(p)$ with $p \in [0,1]$:

$$p_X(x) = \begin{cases}
p & \text{if } x = 1 \\
1-p & \text{if } x = 0
\end{cases}$$

- $E[X] = p$
- $\text{Var}(X) = p(1-p)$
- MGF: $M_X(t) = 1-p + pe^t$

### Binomial Distribution

$X \sim \text{Binomial}(n, p)$: sum of $n$ independent Bernoulli$(p)$ trials:

$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

- $E[X] = np$
- $\text{Var}(X) = np(1-p)$
- MGF: $M_X(t) = (1-p + pe^t)^n$

**Applications**: Number of successful trades in $n$ periods, default counts in a portfolio.

### Poisson Distribution

$X \sim \text{Poisson}(\lambda)$ with $\lambda > 0$:

$$p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

- $E[X] = \lambda$
- $\text{Var}(X) = \lambda$
- MGF: $M_X(t) = e^{\lambda(e^t - 1)}$

The Poisson distribution arises as the limit of Binomial$(n, p)$ as $n \to \infty$ and $np \to \lambda$ (Poisson limit theorem).

**Applications**: Arrival times in high-frequency trading, jump counts in jump-diffusion models.

### Geometric Distribution

$X \sim \text{Geometric}(p)$: number of failures before first success:

$$p_X(k) = (1-p)^k p, \quad k = 0, 1, 2, \ldots$$

- $E[X] = \frac{1-p}{p}$
- $\text{Var}(X) = \frac{1-p}{p^2}$
- MGF: $M_X(t) = \frac{p}{1-(1-p)e^t}$ for $t < -\ln(1-p)$

**Memoryless property**: $\mathbb{P}(X > m+n | X > m) = \mathbb{P}(X > n)$

### Negative Binomial Distribution

$X \sim \text{NegativeBinomial}(r, p)$: number of failures before $r$ successes:

$$p_X(k) = \binom{k+r-1}{r-1} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots$$

- $E[X] = \frac{r(1-p)}{p}$
- $\text{Var}(X) = \frac{r(1-p)}{p^2}$
- MGF: $M_X(t) = \left(\frac{p}{1-(1-p)e^t}\right)^r$

Note: Geometric$(p)$ = NegativeBinomial$(1, p)$.

### Hypergeometric Distribution

$X \sim \text{Hypergeometric}(N, K, n)$: number of successes in $n$ draws without replacement from population of size $N$ with $K$ successes:

$$p_X(k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}, \quad k = \max(0, n-N+K), \ldots, \min(n, K)$$

- $E[X] = n\frac{K}{N}$
- $\text{Var}(X) = n\frac{K}{N}\frac{N-K}{N}\frac{N-n}{N-1}$

As $N \to \infty$ with $K/N \to p$, Hypergeometric$(N, K, n) \to \text{Binomial}(n, p)$.

## Continuous Distributions

### Uniform Distribution

$X \sim \text{Uniform}(a, b)$:

$$f_X(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

- $E[X] = \frac{a+b}{2}$
- $\text{Var}(X) = \frac{(b-a)^2}{12}$
- MGF: $M_X(t) = \frac{e^{tb} - e^{ta}}{t(b-a)}$ for $t \neq 0$

### Normal Distribution

$X \sim \mathcal{N}(\mu, \sigma^2)$:

$$f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- $E[X] = \mu$
- $\text{Var}(X) = \sigma^2$
- MGF: $M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$

The standard normal $Z \sim \mathcal{N}(0,1)$ has PDF:

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

and CDF $\Phi(z)$.

**Properties**:
- Linear transformation: $aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$
- Sum of independent normals: $X + Y \sim \mathcal{N}(\mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2)$
- Central limit theorem: sample means converge to normal

### Exponential Distribution

$X \sim \text{Exponential}(\lambda)$ with $\lambda > 0$:

$$f_X(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

- $E[X] = \frac{1}{\lambda}$
- $\text{Var}(X) = \frac{1}{\lambda^2}$
- MGF: $M_X(t) = \frac{\lambda}{\lambda-t}$ for $t < \lambda$

**Memoryless property**: $\mathbb{P}(X > s+t | X > s) = \mathbb{P}(X > t)$

**Applications**: Inter-arrival times in Poisson processes, time to default in reduced-form models.

### Gamma Distribution

$X \sim \text{Gamma}(\alpha, \beta)$ with shape $\alpha > 0$ and rate $\beta > 0$:

$$f_X(x) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

where $\Gamma(\alpha) = \int_0^{\infty} t^{\alpha-1} e^{-t} dt$ is the gamma function.

- $E[X] = \frac{\alpha}{\beta}$
- $\text{Var}(X) = \frac{\alpha}{\beta^2}$
- MGF: $M_X(t) = \left(\frac{\beta}{\beta-t}\right)^{\alpha}$ for $t < \beta$

**Properties**:
- Exponential$(\lambda)$ = Gamma$(1, \lambda)$
- Chi-squared$(k)$ = Gamma$(k/2, 1/2)$
- Sum: If $X_i \sim \text{Gamma}(\alpha_i, \beta)$ are independent, then $\sum X_i \sim \text{Gamma}(\sum \alpha_i, \beta)$

### Beta Distribution

$X \sim \text{Beta}(\alpha, \beta)$ with $\alpha, \beta > 0$:

$$f_X(x) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}, \quad x \in [0,1]$$

- $E[X] = \frac{\alpha}{\alpha+\beta}$
- $\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

**Applications**: Prior distributions in Bayesian analysis, modeling probabilities.

### Chi-Squared Distribution

$X \sim \chi^2(k)$ with $k$ degrees of freedom:

$$X = \sum_{i=1}^{k} Z_i^2$$

where $Z_i \sim \mathcal{N}(0,1)$ are independent.

- $E[X] = k$
- $\text{Var}(X) = 2k$
- MGF: $M_X(t) = (1-2t)^{-k/2}$ for $t < 1/2$

**Properties**:
- $\chi^2(k) = \text{Gamma}(k/2, 1/2)$
- Sum: If $X_i \sim \chi^2(k_i)$ are independent, then $\sum X_i \sim \chi^2(\sum k_i)$

### Student-t Distribution

$X \sim t(\nu)$ with $\nu$ degrees of freedom:

$$X = \frac{Z}{\sqrt{S/\nu}}$$

where $Z \sim \mathcal{N}(0,1)$ and $S \sim \chi^2(\nu)$ are independent.

PDF:

$$f_X(x) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\Gamma(\nu/2)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}$$

- $E[X] = 0$ (if $\nu > 1$)
- $\text{Var}(X) = \frac{\nu}{\nu-2}$ (if $\nu > 2$)

As $\nu \to \infty$, $t(\nu) \to \mathcal{N}(0,1)$.

**Applications**: Small sample inference, robust regression, modeling heavy-tailed returns.

### F-Distribution

$X \sim F(d_1, d_2)$:

$$X = \frac{S_1/d_1}{S_2/d_2}$$

where $S_1 \sim \chi^2(d_1)$ and $S_2 \sim \chi^2(d_2)$ are independent.

- $E[X] = \frac{d_2}{d_2-2}$ (if $d_2 > 2$)
- $\text{Var}(X) = \frac{2d_2^2(d_1+d_2-2)}{d_1(d_2-2)^2(d_2-4)}$ (if $d_2 > 4$)

**Applications**: Testing equality of variances, ANOVA.

### Lognormal Distribution

$X \sim \text{Lognormal}(\mu, \sigma^2)$ if $\ln X \sim \mathcal{N}(\mu, \sigma^2)$:

$$f_X(x) = \frac{1}{x\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

- $E[X] = e^{\mu + \sigma^2/2}$
- $\text{Var}(X) = e^{2\mu + \sigma^2}(e^{\sigma^2} - 1)$

**Applications**: Stock prices (Black-Scholes), income distributions, option pricing.

### Pareto Distribution

$X \sim \text{Pareto}(x_m, \alpha)$ with scale $x_m > 0$ and shape $\alpha > 0$:

$$f_X(x) = \frac{\alpha x_m^{\alpha}}{x^{\alpha+1}}, \quad x \geq x_m$$

- $E[X] = \frac{\alpha x_m}{\alpha-1}$ (if $\alpha > 1$)
- $\text{Var}(X) = \frac{\alpha x_m^2}{(\alpha-1)^2(\alpha-2)}$ (if $\alpha > 2$)

**Power law tail**: $\mathbb{P}(X > x) \sim x^{-\alpha}$ for large $x$.

**Applications**: Tail risk modeling, wealth distributions, extreme losses.

### Weibull Distribution

$X \sim \text{Weibull}(\lambda, k)$ with scale $\lambda > 0$ and shape $k > 0$:

$$f_X(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, \quad x \geq 0$$

- $E[X] = \lambda \Gamma(1 + 1/k)$
- $\text{Var}(X) = \lambda^2 \left[\Gamma(1 + 2/k) - \Gamma^2(1 + 1/k)\right]$

**Properties**:
- Exponential$(\lambda)$ = Weibull$(\lambda, 1)$
- Hazard function: $h(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}$

**Applications**: Survival analysis, reliability modeling, failure times.

## Moment Generating Functions

The moment generating function (MGF) of $X$ is:

$$M_X(t) = E[e^{tX}] = \begin{cases}
\sum_x e^{tx} p_X(x) & \text{(discrete)} \\
\int_{-\infty}^{\infty} e^{tx} f_X(x) dx & \text{(continuous)}
\end{cases}$$

**Properties**:
- $E[X^n] = M_X^{(n)}(0)$ (if MGF exists)
- If $X$ and $Y$ are independent, $M_{X+Y}(t) = M_X(t) M_Y(t)$
- Uniqueness: MGF uniquely determines distribution (if it exists in a neighborhood of 0)

## Characteristic Functions

The characteristic function is:

$$\phi_X(t) = E[e^{itX}] = M_X(it)$$

**Advantages over MGF**:
- Always exists (bounded by 1)
- Inversion formula available
- Continuity theorem for convergence in distribution

**Inversion formula**: For continuous $X$:

$$f_X(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{-itx} \phi_X(t) dt$$

## Transformations of Random Variables

### One-to-One Transformations

If $Y = g(X)$ where $g$ is strictly monotone and differentiable, then:

$$f_Y(y) = f_X(g^{-1}(y)) \left|\frac{d}{dy} g^{-1}(y)\right| = \frac{f_X(x)}{|g'(x)|}\Big|_{x=g^{-1}(y)}$$

### Multivariate Case (Jacobian Method)

For transformation $(Y_1, Y_2) = (g_1(X_1, X_2), g_2(X_1, X_2))$:

$$f_{Y_1, Y_2}(y_1, y_2) = f_{X_1, X_2}(x_1, x_2) |J|^{-1}$$

where the Jacobian determinant is:

$$J = \det\begin{pmatrix}
\frac{\partial g_1}{\partial x_1} & \frac{\partial g_1}{\partial x_2} \\
\frac{\partial g_2}{\partial x_1} & \frac{\partial g_2}{\partial x_2}
\end{pmatrix}$$

**Example**: Box-Muller transformation generates independent standard normals from uniform random variables.

## Relationships Between Distributions

### Limiting Relationships

- Binomial$(n, p) \to \text{Poisson}(\lambda)$ as $n \to \infty$, $np \to \lambda$
- Binomial$(n, p) \to \mathcal{N}(np, np(1-p))$ as $n \to \infty$ (CLT)
- Poisson$(\lambda) \to \mathcal{N}(\lambda, \lambda)$ as $\lambda \to \infty$
- $t(\nu) \to \mathcal{N}(0,1)$ as $\nu \to \infty$
- Hypergeometric$(N, K, n) \to \text{Binomial}(n, K/N)$ as $N \to \infty$

### Special Cases

- Bernoulli$(p)$ = Binomial$(1, p)$
- Geometric$(p)$ = NegativeBinomial$(1, p)$
- Exponential$(\lambda)$ = Gamma$(1, \lambda)$ = Weibull$(\lambda, 1)$
- Chi-squared$(k)$ = Gamma$(k/2, 1/2)$
- Cauchy = $t(1)$

## Applications in Quantitative Finance

### Return Distributions

Log returns $r_t = \ln(S_t/S_{t-1})$ are often modeled as:
- Normal: $r_t \sim \mathcal{N}(\mu, \sigma^2)$ (Black-Scholes)
- Student-t: $r_t \sim t(\nu)$ (fat tails)
- Mixture: $r_t \sim \pi \mathcal{N}(\mu_1, \sigma_1^2) + (1-\pi)\mathcal{N}(\mu_2, \sigma_2^2)$ (regime switching)

### Default Modeling

Time to default $\tau$:
- Exponential$(\lambda)$: constant hazard rate
- Weibull$(\lambda, k)$: time-varying hazard
- Lognormal$(\mu, \sigma^2)$: log-normal default times

### Tail Risk

Extreme losses follow:
- Pareto distribution for power-law tails
- Generalized Pareto Distribution (GPD) for exceedances over threshold
- Lognormal for moderate tail behavior
