# Multivariate Distributions and Joint Probability

## Joint Distributions

For random variables $X_1, X_2, \ldots, X_n$, the joint cumulative distribution function is:

$$F_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = \mathbb{P}(X_1 \leq x_1, \ldots, X_n \leq x_n)$$

### Discrete Case

The joint probability mass function is:

$$p_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = \mathbb{P}(X_1 = x_1, \ldots, X_n = x_n)$$

Properties:
- Non-negativity: $p_{X_1, \ldots, X_n}(x_1, \ldots, x_n) \geq 0$
- Normalization: $\sum_{x_1} \cdots \sum_{x_n} p_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = 1$

### Continuous Case

The joint probability density function satisfies:

$$F_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_n} f_{X_1, \ldots, X_n}(t_1, \ldots, t_n) dt_n \cdots dt_1$$

and:

$$f_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = \frac{\partial^n}{\partial x_1 \cdots \partial x_n} F_{X_1, \ldots, X_n}(x_1, \ldots, x_n)$$

Properties:
- Non-negativity: $f_{X_1, \ldots, X_n}(x_1, \ldots, x_n) \geq 0$
- Normalization: $\int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f_{X_1, \ldots, X_n}(x_1, \ldots, x_n) dx_n \cdots dx_1 = 1$

## Marginal Distributions

### Discrete Case

The marginal PMF of $X_1$:

$$p_{X_1}(x_1) = \sum_{x_2} \cdots \sum_{x_n} p_{X_1, \ldots, X_n}(x_1, \ldots, x_n)$$

### Continuous Case

The marginal PDF of $X_1$:

$$f_{X_1}(x_1) = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f_{X_1, \ldots, X_n}(x_1, \ldots, x_n) dx_2 \cdots dx_n$$

For bivariate case $(X, Y)$:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) dy$$

$$f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) dx$$

## Conditional Distributions

### Discrete Case

The conditional PMF of $X$ given $Y = y$:

$$p_{X|Y}(x|y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}$$

provided $p_Y(y) > 0$.

### Continuous Case

The conditional PDF of $X$ given $Y = y$:

$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}$$

provided $f_Y(y) > 0$.

**Chain rule**:

$$f_{X,Y}(x, y) = f_{X|Y}(x|y) f_Y(y) = f_{Y|X}(y|x) f_X(x)$$

**Bayes' rule for densities**:

$$f_{X|Y}(x|y) = \frac{f_{Y|X}(y|x) f_X(x)}{f_Y(y)} = \frac{f_{Y|X}(y|x) f_X(x)}{\int f_{Y|X}(y|t) f_X(t) dt}$$

## Multivariate Normal Distribution

### Definition

A random vector $\mathbf{X} = (X_1, \ldots, X_n)^T$ follows a multivariate normal distribution $\mathbf{X} \sim \mathcal{N}_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ if its joint PDF is:

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

where:
- $\boldsymbol{\mu} = E[\mathbf{X}]$ is the mean vector
- $\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X})$ is the covariance matrix (symmetric, positive definite)

### Properties

1. **Linear transformations**: If $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ where $\mathbf{A}$ is $m \times n$ and $\mathbf{b}$ is $m \times 1$, then:
   $$\mathbf{Y} \sim \mathcal{N}_m(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

2. **Marginal distributions**: If $\mathbf{X} = (\mathbf{X}_1^T, \mathbf{X}_2^T)^T$ with:
   $$\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$$
   then $\mathbf{X}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})$.

3. **Independence**: Components $X_i$ and $X_j$ are independent if and only if $\Sigma_{ij} = 0$.

4. **Conditional distributions**: 
   $$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$
   where:
   $$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$
   $$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

5. **Characteristic function**:
   $$\phi_{\mathbf{X}}(\mathbf{t}) = \exp\left(i\mathbf{t}^T\boldsymbol{\mu} - \frac{1}{2}\mathbf{t}^T\boldsymbol{\Sigma}\mathbf{t}\right)$$

### Bivariate Normal

For $(X, Y) \sim \mathcal{N}_2(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with:

$$\boldsymbol{\mu} = \begin{pmatrix} \mu_X \\ \mu_Y \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \sigma_X^2 & \rho\sigma_X\sigma_Y \\ \rho\sigma_X\sigma_Y & \sigma_Y^2 \end{pmatrix}$$

The joint PDF is:

$$f_{X,Y}(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)$$

The conditional distribution:

$$X | Y = y \sim \mathcal{N}\left(\mu_X + \rho\frac{\sigma_X}{\sigma_Y}(y - \mu_Y), \sigma_X^2(1-\rho^2)\right)$$

## Copulas

A copula is a multivariate distribution function with uniform marginals on $[0,1]$. Sklar's theorem states that any joint CDF $F_{X_1, \ldots, X_n}$ can be written as:

$$F_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = C(F_1(x_1), \ldots, F_n(x_n))$$

where $C$ is a copula and $F_i$ are marginal CDFs.

### Gaussian Copula

For a correlation matrix $\mathbf{R}$, the Gaussian copula is:

$$C_{\mathbf{R}}^{\text{Gauss}}(u_1, \ldots, u_n) = \Phi_{\mathbf{R}}(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_n))$$

where $\Phi_{\mathbf{R}}$ is the multivariate normal CDF with correlation $\mathbf{R}$ and $\Phi$ is the standard normal CDF.

**Properties**:
- Symmetric
- No tail dependence: $\lambda_U = \lambda_L = 0$
- Fully determined by correlation matrix

### t-Copula

The t-copula with $\nu$ degrees of freedom and correlation $\mathbf{R}$:

$$C_{\nu, \mathbf{R}}^{\text{t}}(u_1, \ldots, u_n) = t_{\nu, \mathbf{R}}(t_{\nu}^{-1}(u_1), \ldots, t_{\nu}^{-1}(u_n))$$

where $t_{\nu, \mathbf{R}}$ is the multivariate t-distribution CDF.

**Properties**:
- Symmetric tail dependence
- Upper tail dependence: $\lambda_U = 2t_{\nu+1}\left(-\sqrt{\nu+1}\sqrt{\frac{1-\rho}{1+\rho}}\right)$
- More flexible than Gaussian for tail modeling

### Archimedean Copulas

Archimedean copulas have the form:

$$C(u_1, \ldots, u_n) = \psi^{-1}(\psi(u_1) + \cdots + \psi(u_n))$$

where $\psi: [0,1] \to [0,\infty]$ is the generator function (decreasing, convex, $\psi(1) = 0$).

**Gumbel copula** (upper tail dependence):
$$\psi(t) = (-\ln t)^{\theta}, \quad \theta \geq 1$$
$$C(u_1, u_2) = \exp(-[(-\ln u_1)^{\theta} + (-\ln u_2)^{\theta}]^{1/\theta})$$

**Clayton copula** (lower tail dependence):
$$\psi(t) = \frac{1}{\theta}(t^{-\theta} - 1), \quad \theta > 0$$
$$C(u_1, u_2) = \max([u_1^{-\theta} + u_2^{-\theta} - 1]^{-1/\theta}, 0)$$

**Frank copula** (no tail dependence):
$$\psi(t) = -\ln\left(\frac{e^{-\theta t} - 1}{e^{-\theta} - 1}\right), \quad \theta \neq 0$$
$$C(u_1, u_2) = -\frac{1}{\theta}\ln\left(1 + \frac{(e^{-\theta u_1} - 1)(e^{-\theta u_2} - 1)}{e^{-\theta} - 1}\right)$$

### Tail Dependence

Upper tail dependence coefficient:

$$\lambda_U = \lim_{u \to 1^-} \mathbb{P}(X_2 > F_2^{-1}(u) | X_1 > F_1^{-1}(u)) = \lim_{u \to 1^-} \frac{1 - 2u + C(u, u)}{1 - u}$$

Lower tail dependence coefficient:

$$\lambda_L = \lim_{u \to 0^+} \mathbb{P}(X_2 \leq F_2^{-1}(u) | X_1 \leq F_1^{-1}(u)) = \lim_{u \to 0^+} \frac{C(u, u)}{u}$$

## Correlation and Dependence

### Pearson Correlation

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{E[(X - \mu_X)(Y - \mu_Y)]}{\sqrt{E[(X - \mu_X)^2]E[(Y - \mu_Y)^2]}}$$

**Properties**:
- $-1 \leq \rho_{XY} \leq 1$
- $\rho_{XY} = 1$ if $Y = aX + b$ with $a > 0$
- $\rho_{XY} = -1$ if $Y = aX + b$ with $a < 0$
- Measures linear dependence only
- Not invariant under monotone transformations

**Limitations**:
- Zero correlation does not imply independence (except for normal)
- Sensitive to outliers
- Only captures linear relationships

### Spearman's Rank Correlation

$$\rho_S = \text{Corr}(F_X(X), F_Y(Y)) = \text{Corr}(\text{rank}(X), \text{rank}(Y))$$

For sample $(x_i, y_i)_{i=1}^n$:

$$\hat{\rho}_S = 1 - \frac{6\sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the difference in ranks.

**Properties**:
- Invariant under monotone transformations
- Measures monotonic relationships
- Robust to outliers

### Kendall's Tau

$$\tau = \mathbb{P}((X_1 - X_2)(Y_1 - Y_2) > 0) - \mathbb{P}((X_1 - X_2)(Y_1 - Y_2) < 0)$$

For sample:

$$\hat{\tau} = \frac{2}{n(n-1)} \sum_{i<j} \text{sign}((x_i - x_j)(y_i - y_j))$$

**Properties**:
- Invariant under monotone transformations
- Measures concordance
- Related to copula: $\tau = 4\int_0^1 \int_0^1 C(u, v) dC(u, v) - 1$

## Covariance Matrix

For random vector $\mathbf{X} = (X_1, \ldots, X_n)^T$:

$$\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

with elements:

$$\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$$

**Properties**:
- Symmetric: $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$
- Positive semidefinite: $\mathbf{v}^T\boldsymbol{\Sigma}\mathbf{v} \geq 0$ for all $\mathbf{v}$
- Diagonal elements: $\Sigma_{ii} = \text{Var}(X_i)$

### Sample Covariance Matrix

For data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ (n observations, p variables):

$$\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}(\mathbf{X} - \bar{\mathbf{X}})^T(\mathbf{X} - \bar{\mathbf{X}})$$

where $\bar{\mathbf{X}}$ is the matrix of column means.

## Precision Matrix

The precision matrix (inverse covariance matrix) is:

$$\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$$

**Properties**:
- In multivariate normal, $\Omega_{ij} = 0$ if and only if $X_i$ and $X_j$ are conditionally independent given all other variables
- Useful in graphical models (Gaussian Markov random fields)
- Sparse precision matrices correspond to sparse conditional independence graphs

### Partial Correlation

The partial correlation between $X_i$ and $X_j$ given all other variables:

$$\rho_{ij|\text{rest}} = -\frac{\Omega_{ij}}{\sqrt{\Omega_{ii}\Omega_{jj}}}$$

This measures correlation after removing linear effects of other variables.

## Applications in Quantitative Finance

### Portfolio Theory

For portfolio return $R_p = \sum_{i=1}^{n} w_i R_i$:

$$\text{Var}(R_p) = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \Sigma_{ij}$$

where $\mathbf{w}$ is the weight vector and $\boldsymbol{\Sigma}$ is the covariance matrix of asset returns.

### Factor Models

In factor models $R_i = \alpha_i + \sum_{j=1}^{k} \beta_{ij} F_j + \epsilon_i$:

$$\boldsymbol{\Sigma} = \boldsymbol{\beta}\boldsymbol{\Sigma}_F\boldsymbol{\beta}^T + \boldsymbol{\Sigma}_{\epsilon}$$

where $\boldsymbol{\Sigma}_F$ is factor covariance and $\boldsymbol{\Sigma}_{\epsilon}$ is idiosyncratic covariance.

### Risk Aggregation

For portfolio loss $L = \sum_{i=1}^{n} L_i$:

- Under independence: $\text{Var}(L) = \sum_{i=1}^{n} \text{Var}(L_i)$
- Under dependence: $\text{Var}(L) = \sum_{i=1}^{n} \sum_{j=1}^{n} \text{Cov}(L_i, L_j)$

Copulas allow modeling non-linear dependence structures for tail risk.

### Credit Risk

Joint default probabilities:

$$\mathbb{P}(\text{default}_1, \text{default}_2) = C(F_1(\tau_1), F_2(\tau_2))$$

where $\tau_i$ are default times and $C$ is a copula (often Gaussian or t-copula).

### Correlation Trading

Pairs trading strategies exploit mean reversion in correlation. If $\rho_{XY}$ is high but temporarily low, one can:
- Long the pair when correlation is below historical mean
- Short when correlation exceeds historical mean

The correlation structure determines optimal hedge ratios.
