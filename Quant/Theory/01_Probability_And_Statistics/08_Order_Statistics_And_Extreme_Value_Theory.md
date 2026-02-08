# Order Statistics and Extreme Value Theory

## Order Statistics

Given a sample $X_1, X_2, \ldots, X_n$ of independent and identically distributed (i.i.d.) random variables with common distribution function $F$ and density $f$, the order statistics are the sorted values:

$$X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$$

where $X_{(k)}$ is the $k$-th smallest value.

### Distribution of Order Statistics

The cumulative distribution function of the $k$-th order statistic $X_{(k)}$ is:

$$F_{X_{(k)}}(x) = \mathbb{P}(X_{(k)} \leq x) = \sum_{j=k}^{n} \binom{n}{j} [F(x)]^j [1-F(x)]^{n-j}$$

This follows from the fact that $X_{(k)} \leq x$ if and only if at least $k$ of the $X_i$ are less than or equal to $x$.

The probability density function of $X_{(k)}$ is:

$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} [F(x)]^{k-1} [1-F(x)]^{n-k} f(x)$$

**Derivation**: The event that $X_{(k)} \in [x, x+dx]$ requires:
- Exactly $k-1$ observations less than $x$
- Exactly $1$ observation in $[x, x+dx]$
- Exactly $n-k$ observations greater than $x+dx$

The multinomial probability is:

$$f_{X_{(k)}}(x) dx = \frac{n!}{(k-1)!1!(n-k)!} [F(x)]^{k-1} [f(x) dx] [1-F(x)]^{n-k}$$

Dividing by $dx$ gives the result.

### Joint Distribution of Order Statistics

The joint density of all order statistics $(X_{(1)}, \ldots, X_{(n)})$ is:

$$f_{X_{(1)}, \ldots, X_{(n)}}(x_1, \ldots, x_n) = \begin{cases}
n! \prod_{i=1}^{n} f(x_i) & \text{if } x_1 \leq x_2 \leq \cdots \leq x_n \\
0 & \text{otherwise}
\end{cases}$$

The joint density of $X_{(i)}$ and $X_{(j)}$ for $i < j$ is:

$$f_{X_{(i)}, X_{(j)}}(x, y) = \frac{n!}{(i-1)!(j-i-1)!(n-j)!} [F(x)]^{i-1} [F(y) - F(x)]^{j-i-1} [1-F(y)]^{n-j} f(x) f(y)$$

for $x \leq y$, and $0$ otherwise.

### Spacing

The spacing between consecutive order statistics is:

$$D_k = X_{(k+1)} - X_{(k)}$$

For a uniform distribution on $[0,1]$, the spacings have a Dirichlet distribution. In particular, for $U_{(1)}, \ldots, U_{(n)}$ from $\text{Uniform}(0,1)$:

$$(D_1, D_2, \ldots, D_n) \sim \text{Dirichlet}(1, 1, \ldots, 1)$$

where $D_n = 1 - U_{(n)}$.

### Maximum and Minimum

The maximum $M_n = X_{(n)}$ has distribution:

$$F_{M_n}(x) = [F(x)]^n$$

$$f_{M_n}(x) = n [F(x)]^{n-1} f(x)$$

The minimum $m_n = X_{(1)}$ has distribution:

$$F_{m_n}(x) = 1 - [1-F(x)]^n$$

$$f_{m_n}(x) = n [1-F(x)]^{n-1} f(x)$$

**Example**: For $X_i \sim \text{Exponential}(\lambda)$:

$$F_{M_n}(x) = (1 - e^{-\lambda x})^n$$

$$E[M_n] = \frac{1}{\lambda} \sum_{k=1}^{n} \frac{1}{k} \approx \frac{1}{\lambda} (\ln n + \gamma)$$

where $\gamma$ is Euler's constant.

## Extreme Value Theory

Extreme Value Theory (EVT) studies the asymptotic behavior of extreme order statistics, particularly the maximum and minimum, as the sample size grows.

### Maximum Domain of Attraction

A distribution $F$ is said to be in the maximum domain of attraction of a distribution $G$ if there exist sequences $a_n > 0$ and $b_n$ such that:

$$\lim_{n \to \infty} \mathbb{P}\left(\frac{M_n - b_n}{a_n} \leq x\right) = \lim_{n \to \infty} [F(a_n x + b_n)]^n = G(x)$$

The Fisher-Tippett-Gnedenko theorem states that $G$ must be one of three types (up to location and scale):

1. **Gumbel (Type I)**: $G(x) = \exp(-e^{-x})$ for $x \in \mathbb{R}$
2. **Fréchet (Type II)**: $G(x) = \begin{cases} 0 & x \leq 0 \\ \exp(-x^{-\alpha}) & x > 0 \end{cases}$ for $\alpha > 0$
3. **Weibull (Type III)**: $G(x) = \begin{cases} \exp(-(-x)^{\alpha}) & x < 0 \\ 1 & x \geq 0 \end{cases}$ for $\alpha > 0$

### Generalized Extreme Value Distribution

The three types can be unified into the Generalized Extreme Value (GEV) distribution:

$$G_{\xi}(x) = \begin{cases}
\exp\left(-(1+\xi x)^{-1/\xi}\right) & \text{if } \xi \neq 0, 1+\xi x > 0 \\
\exp(-e^{-x}) & \text{if } \xi = 0
\end{cases}$$

where $\xi$ is the shape parameter:
- $\xi = 0$: Gumbel (light tails)
- $\xi > 0$: Fréchet (heavy tails)
- $\xi < 0$: Weibull (bounded support)

The location-scale family is:

$$G_{\xi, \mu, \sigma}(x) = G_{\xi}\left(\frac{x - \mu}{\sigma}\right)$$

with location parameter $\mu$ and scale parameter $\sigma > 0$.

### Block Maxima Method

The block maxima method divides data into blocks and models the maximum of each block using the GEV distribution.

**Procedure**:
1. Divide observations into $m$ blocks of size $n$ (e.g., yearly maxima)
2. Extract block maxima: $M_1, M_2, \ldots, M_m$
3. Fit GEV distribution to block maxima
4. Estimate tail probabilities and quantiles

**Example**: For daily returns, use monthly or yearly maxima. If $M$ follows GEV$(\xi, \mu, \sigma)$, then:

$$\mathbb{P}(M > x) = 1 - G_{\xi, \mu, \sigma}(x)$$

The $p$-quantile (Value at Risk) is:

$$\text{VaR}_p = \begin{cases}
\mu - \frac{\sigma}{\xi}[1 - (-\ln p)^{-\xi}] & \text{if } \xi \neq 0 \\
\mu - \sigma \ln(-\ln p) & \text{if } \xi = 0
\end{cases}$$

### Peaks Over Threshold (POT)

The POT method models exceedances over a high threshold $u$ rather than block maxima.

**Pickands-Balkema-de Haan Theorem**: For a distribution $F$ in the maximum domain of attraction, the conditional distribution of exceedances:

$$F_u(y) = \mathbb{P}(X - u \leq y | X > u) = \frac{F(u+y) - F(u)}{1-F(u)}$$

converges to the Generalized Pareto Distribution (GPD) as $u \to \infty$:

$$H_{\xi, \sigma}(y) = \begin{cases}
1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi} & \text{if } \xi \neq 0, y > 0, 1 + \frac{\xi y}{\sigma} > 0 \\
1 - e^{-y/\sigma} & \text{if } \xi = 0, y > 0
\end{cases}$$

**Procedure**:
1. Choose threshold $u$ (e.g., 95th percentile)
2. Extract exceedances: $Y_i = X_i - u$ for $X_i > u$
3. Fit GPD to exceedances
4. Estimate tail probabilities:

$$\mathbb{P}(X > x) = \mathbb{P}(X > u) \cdot \mathbb{P}(X > x | X > u)$$

For $x > u$:

$$\mathbb{P}(X > x) = \left(1 + \frac{\xi(x-u)}{\sigma}\right)^{-1/\xi} \cdot \frac{N_u}{n}$$

where $N_u$ is the number of exceedances.

### Parameter Estimation

**Maximum Likelihood Estimation**:

For GEV with observations $M_1, \ldots, M_m$:

$$L(\xi, \mu, \sigma) = \prod_{i=1}^{m} g_{\xi, \mu, \sigma}(M_i)$$

where $g$ is the GEV density:

$$g_{\xi, \mu, \sigma}(x) = \frac{1}{\sigma} \left(1 + \xi \frac{x-\mu}{\sigma}\right)^{-1/\xi - 1} \exp\left(-\left(1 + \xi \frac{x-\mu}{\sigma}\right)^{-1/\xi}\right)$$

For GPD with exceedances $Y_1, \ldots, Y_{N_u}$:

$$L(\xi, \sigma) = \prod_{i=1}^{N_u} h_{\xi, \sigma}(Y_i)$$

where $h$ is the GPD density:

$$h_{\xi, \sigma}(y) = \frac{1}{\sigma} \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi - 1}$$

**Method of Moments**:

For GPD, method of moments estimators:

$$\hat{\sigma} = \frac{\bar{Y} E[Y]}{E[Y] - \bar{Y}}$$

$$\hat{\xi} = \frac{1}{2}\left(1 - \frac{(\bar{Y})^2}{E[Y^2]}\right)$$

**Hill Estimator** (for Fréchet domain, $\xi > 0$):

For order statistics $X_{(1)} \geq X_{(2)} \geq \cdots \geq X_{(n)}$:

$$\hat{\xi}_k = \frac{1}{k} \sum_{i=1}^{k} \ln X_{(i)} - \ln X_{(k+1)}$$

where $k$ is the number of upper order statistics used.

### Threshold Selection

Choosing the threshold $u$ involves a bias-variance tradeoff:
- Too low: bias (distribution not in tail regime)
- Too high: variance (few exceedances)

**Methods**:
1. **Mean Excess Plot**: Plot $e(u) = E[X - u | X > u]$ vs $u$. For GPD, $e(u) = \frac{\sigma + \xi u}{1-\xi}$ (linear for $\xi < 1$). Choose $u$ where plot becomes linear.

2. **Stability**: Choose $u$ where parameter estimates stabilize.

3. **Fixed percentile**: Use 90th, 95th, or 99th percentile.

## Applications in Finance

### Value at Risk (VaR)

VaR at confidence level $\alpha$ is:

$$\text{VaR}_\alpha = \inf\{x : \mathbb{P}(L > x) \leq 1-\alpha\}$$

where $L$ is the loss. Using POT method:

$$\text{VaR}_\alpha = u + \frac{\sigma}{\xi}\left(\left(\frac{n}{N_u}(1-\alpha)\right)^{-\xi} - 1\right)$$

**Example**: For daily portfolio returns, if $u = 2\%$ (95th percentile), $N_u = 50$ exceedances out of $n = 1000$ observations, and fitted GPD has $\xi = 0.3$, $\sigma = 0.5\%$:

$$\text{VaR}_{0.99} = 0.02 + \frac{0.005}{0.3}\left(\left(\frac{1000}{50} \cdot 0.01\right)^{-0.3} - 1\right) \approx 0.045 = 4.5\%$$

### Expected Shortfall (Conditional VaR)

Expected Shortfall at level $\alpha$ is:

$$\text{ES}_\alpha = E[L | L > \text{VaR}_\alpha]$$

For GPD:

$$\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1-\xi} + \frac{\sigma - \xi u}{1-\xi}$$

provided $\xi < 1$.

### Tail Risk Measurement

EVT provides tools for:
- **Stress testing**: Estimate probabilities of extreme losses
- **Regulatory capital**: Basel III uses EVT for market risk
- **Portfolio optimization**: Incorporate tail risk constraints
- **Catastrophic loss modeling**: Insurance and operational risk

### Return Distribution Modeling

Financial returns often exhibit:
- Heavy tails ($\xi > 0$)
- Asymmetry (different tail behavior for gains vs losses)
- Time-varying volatility

**Two-tailed approach**: Model upper and lower tails separately:
- Upper tail: exceedances of positive threshold
- Lower tail: exceedances (in absolute value) of negative threshold

### Extreme Correlation

During crises, correlations increase. EVT can model:
- **Tail dependence**: Probability of joint extremes
- **Systemic risk**: Coexceedances across assets
- **Contagion**: Extreme events propagating

**Example**: For bivariate extremes, use copula methods with GEV margins to capture tail dependence structure.

### Operational Risk

EVT is used for:
- **Loss distribution approach**: Model operational losses exceeding threshold
- **Capital allocation**: Estimate capital for operational risk
- **Scenario analysis**: Generate extreme loss scenarios

## Practical Considerations

### Stationarity

EVT assumes stationarity. For financial data:
- Use rolling windows or time-varying parameters
- Account for volatility clustering (GARCH-EVT models)
- Consider regime-switching models

### Dependence

Classical EVT assumes independence. For dependent data:
- Use declustering (identify clusters of extremes)
- Adjust threshold or block size
- Use extremal index to account for clustering

The extremal index $\theta \in [0,1]$ measures clustering:
- $\theta = 1$: no clustering (asymptotically independent)
- $\theta < 1$: clustering (dependent extremes)

### Model Validation

- **QQ plots**: Compare empirical and theoretical quantiles
- **Return level plots**: Compare observed and predicted return levels
- **Goodness-of-fit tests**: Kolmogorov-Smirnov, Anderson-Darling
- **Backtesting**: Compare predicted VaR with realized losses

### Limitations

- **Small sample**: Few extreme observations
- **Model uncertainty**: Parameter estimates sensitive to threshold
- **Non-stationarity**: Structural breaks, regime changes
- **Dependence**: Clustering of extremes
- **Multivariate**: Complexity increases with dimension

Despite limitations, EVT remains the most rigorous framework for modeling extreme events in finance.
