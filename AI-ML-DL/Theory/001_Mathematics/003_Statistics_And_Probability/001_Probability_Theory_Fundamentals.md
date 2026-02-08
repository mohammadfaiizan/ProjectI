# Probability Theory Fundamentals

## Table of Contents

1. [Introduction](#introduction)
2. [Sample Spaces and Events](#sample-spaces-and-events)
3. [Probability Measures](#probability-measures)
4. [Conditional Probability](#conditional-probability)
5. [Bayes' Theorem](#bayes-theorem)
6. [Random Variables](#random-variables)
7. [Probability Distributions](#probability-distributions)
8. [Expectation and Variance](#expectation-and-variance)
9. [Common Distributions](#common-distributions)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Probability theory provides the mathematical foundation for uncertainty quantification, statistical inference, and probabilistic modeling in machine learning. From representing data uncertainty to designing loss functions and understanding generalization, probability concepts permeate ML. This document covers sample spaces, probability measures, random variables, distributions, expectation, and variance, with emphasis on their role in ML applications.

## Sample Spaces and Events

### Sample Space

The **sample space** $\Omega$ is the set of all possible outcomes of a random experiment.

**Examples**:
- Coin flip: $\Omega = \{H, T\}$
- Die roll: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Continuous: $\Omega = \mathbb{R}$ (real numbers)

### Events

An **event** $E$ is a subset of the sample space: $E \subseteq \Omega$.

**Examples**:
- "Heads": $E = \{H\}$
- "Even number": $E = \{2, 4, 6\}$
- "Positive number": $E = (0, \infty)$

### Event Operations

**Union**: $E \cup F$ (either $E$ or $F$ occurs)

**Intersection**: $E \cap F$ (both $E$ and $F$ occur)

**Complement**: $E^c = \Omega \setminus E$ ($E$ does not occur)

**Mutually exclusive**: $E \cap F = \emptyset$ (cannot both occur)

### Sigma-Algebra

A **$\sigma$-algebra** $\mathcal{F}$ on $\Omega$ is a collection of events satisfying:
1. $\Omega \in \mathcal{F}$
2. If $E \in \mathcal{F}$, then $E^c \in \mathcal{F}$
3. If $E_1, E_2, \ldots \in \mathcal{F}$, then $\bigcup_{i=1}^\infty E_i \in \mathcal{F}$

This ensures we can assign probabilities to events consistently.

## Probability Measures

### Axioms of Probability

A **probability measure** $P: \mathcal{F} \to [0, 1]$ satisfies:

1. **Non-negativity**: $P(E) \geq 0$ for all $E \in \mathcal{F}$
2. **Normalization**: $P(\Omega) = 1$
3. **Countable additivity**: For disjoint events $E_1, E_2, \ldots$:
   $$P\left(\bigcup_{i=1}^\infty E_i\right) = \sum_{i=1}^\infty P(E_i)$$

### Properties

**Complement**: $P(E^c) = 1 - P(E)$

**Monotonicity**: If $E \subseteq F$, then $P(E) \leq P(F)$

**Union bound**: $P(E \cup F) = P(E) + P(F) - P(E \cap F)$

**Inclusion-exclusion**: For events $E_1, \ldots, E_n$:
$$P\left(\bigcup_{i=1}^n E_i\right) = \sum_{i} P(E_i) - \sum_{i<j} P(E_i \cap E_j) + \cdots + (-1)^{n+1} P(E_1 \cap \cdots \cap E_n)$$

### Conditional Probability

The **conditional probability** of $E$ given $F$ (with $P(F) > 0$) is:

$$P(E | F) = \frac{P(E \cap F)}{P(F)}$$

**Interpretation**: Probability of $E$ occurring given that $F$ has occurred.

**Properties**:
- $P(E | F) \geq 0$
- $P(\Omega | F) = 1$
- For disjoint $E_1, E_2, \ldots$: $P(\bigcup_i E_i | F) = \sum_i P(E_i | F)$

### Chain Rule

For events $E_1, \ldots, E_n$:

$$P(E_1 \cap \cdots \cap E_n) = P(E_1) P(E_2 | E_1) P(E_3 | E_1 \cap E_2) \cdots P(E_n | E_1 \cap \cdots \cap E_{n-1})$$

### Independence

Events $E$ and $F$ are **independent** if:

$$P(E \cap F) = P(E) P(F)$$

**Equivalent**: $P(E | F) = P(E)$ (knowing $F$ doesn't change probability of $E$).

**Mutual independence**: Events $E_1, \ldots, E_n$ are mutually independent if for any subset:

$$P\left(\bigcap_{i \in I} E_i\right) = \prod_{i \in I} P(E_i)$$

## Bayes' Theorem

### Statement

**Bayes' Theorem**:

$$P(E | F) = \frac{P(F | E) P(E)}{P(F)}$$

**Derivation**: From definition of conditional probability:
$$P(E | F) = \frac{P(E \cap F)}{P(F)} = \frac{P(F | E) P(E)}{P(F)}$$

### Law of Total Probability

If $E_1, \ldots, E_n$ partition $\Omega$ (disjoint and $\bigcup_i E_i = \Omega$), then:

$$P(F) = \sum_{i=1}^n P(F | E_i) P(E_i)$$

**Bayes' with total probability**:

$$P(E_i | F) = \frac{P(F | E_i) P(E_i)}{\sum_{j=1}^n P(F | E_j) P(E_j)}$$

### Interpretation

- **Prior**: $P(E)$ (belief before observing data)
- **Likelihood**: $P(F | E)$ (probability of data given hypothesis)
- **Posterior**: $P(E | F)$ (updated belief after observing data)
- **Evidence**: $P(F)$ (normalizing constant)

## Random Variables

### Definition

A **random variable** $X$ is a function $X: \Omega \to \mathbb{R}$ that assigns a real number to each outcome.

**Example**: For coin flip, $X(H) = 1$, $X(T) = 0$.

### Discrete Random Variables

$X$ is **discrete** if it takes countable values $\{x_1, x_2, \ldots\}$.

**Probability mass function (PMF)**: $p_X(x) = P(X = x)$

**Properties**:
- $p_X(x) \geq 0$ for all $x$
- $\sum_x p_X(x) = 1$

### Continuous Random Variables

$X$ is **continuous** if it takes uncountably many values.

**Probability density function (PDF)**: $f_X(x)$ such that:

$$P(a \leq X \leq b) = \int_a^b f_X(x) dx$$

**Properties**:
- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$

**Note**: $P(X = x) = 0$ for continuous $X$ (probability of exact value is zero).

### Cumulative Distribution Function

**CDF**: $F_X(x) = P(X \leq x)$

**Properties**:
- Non-decreasing: $F_X(x) \leq F_X(y)$ if $x \leq y$
- Right-continuous
- $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$

**Relationship**:
- Discrete: $F_X(x) = \sum_{y \leq x} p_X(y)$
- Continuous: $F_X(x) = \int_{-\infty}^x f_X(t) dt$, $f_X(x) = \frac{d}{dx} F_X(x)$

## Probability Distributions

### Joint Distribution

For random variables $X$ and $Y$:

**Discrete**: $p_{X,Y}(x, y) = P(X = x, Y = y)$

**Continuous**: $f_{X,Y}(x, y)$ such that:

$$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) dx dy$$

### Marginal Distribution

**Discrete**: $p_X(x) = \sum_y p_{X,Y}(x, y)$

**Continuous**: $f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) dy$

### Conditional Distribution

**Discrete**: $p_{X|Y}(x | y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}$ (if $p_Y(y) > 0$)

**Continuous**: $f_{X|Y}(x | y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}$ (if $f_Y(y) > 0$)

### Independence

Random variables $X$ and $Y$ are **independent** if:

$$p_{X,Y}(x, y) = p_X(x) p_Y(y) \quad \text{(discrete)}$$
$$f_{X,Y}(x, y) = f_X(x) f_Y(y) \quad \text{(continuous)}$$

**Equivalent**: $P(X \in A, Y \in B) = P(X \in A) P(Y \in B)$ for all sets $A, B$.

## Expectation and Variance

### Expectation (Mean)

**Discrete**: $\mathbb{E}[X] = \sum_x x p_X(x)$

**Continuous**: $\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) dx$

**Interpretation**: Center of mass, average value.

### Properties of Expectation

**Linearity**: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$

**Function of RV**: $\mathbb{E}[g(X)] = \sum_x g(x) p_X(x)$ (discrete) or $\int g(x) f_X(x) dx$ (continuous)

**Independence**: If $X$ and $Y$ are independent, then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

### Variance

**Variance**: $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$

**Standard deviation**: $\sigma_X = \sqrt{\text{Var}(X)}$

**Properties**:
- $\text{Var}(aX + b) = a^2\text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$
- If $X$ and $Y$ are independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### Covariance

**Covariance**: $\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$

**Correlation**: $\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$ (normalized covariance, $|\rho| \leq 1$)

**Independence**: If $X$ and $Y$ are independent, then $\text{Cov}(X, Y) = 0$ (but converse is not always true).

## Common Distributions

### Bernoulli Distribution

**PMF**: $p_X(x) = p^x(1-p)^{1-x}$ for $x \in \{0, 1\}$

**Parameters**: $p \in [0, 1]$ (success probability)

**Mean**: $\mathbb{E}[X] = p$

**Variance**: $\text{Var}(X) = p(1-p)$

**Application**: Binary classification, coin flips.

### Binomial Distribution

**PMF**: $p_X(k) = \binom{n}{k} p^k(1-p)^{n-k}$ for $k = 0, 1, \ldots, n$

**Parameters**: $n$ (number of trials), $p$ (success probability)

**Mean**: $\mathbb{E}[X] = np$

**Variance**: $\text{Var}(X) = np(1-p)$

**Interpretation**: Sum of $n$ independent Bernoulli($p$) random variables.

### Poisson Distribution

**PMF**: $p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}$ for $k = 0, 1, 2, \ldots$

**Parameters**: $\lambda > 0$ (rate)

**Mean**: $\mathbb{E}[X] = \lambda$

**Variance**: $\text{Var}(X) = \lambda$

**Application**: Counts of rare events, arrival processes.

### Gaussian (Normal) Distribution

**PDF**: $f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

**Parameters**: $\mu \in \mathbb{R}$ (mean), $\sigma > 0$ (standard deviation)

**Notation**: $X \sim \mathcal{N}(\mu, \sigma^2)$

**Mean**: $\mathbb{E}[X] = \mu$

**Variance**: $\text{Var}(X) = \sigma^2$

**Properties**:
- Symmetric about $\mu$
- 68-95-99.7 rule: $P(|X-\mu| \leq \sigma) \approx 0.68$, $P(|X-\mu| \leq 2\sigma) \approx 0.95$, $P(|X-\mu| \leq 3\sigma) \approx 0.997$
- Central limit theorem: Sum of many independent RVs is approximately Gaussian

**Standard normal**: $\mathcal{N}(0, 1)$ with PDF $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$.

### Exponential Distribution

**PDF**: $f_X(x) = \lambda e^{-\lambda x}$ for $x \geq 0$

**Parameters**: $\lambda > 0$ (rate)

**Mean**: $\mathbb{E}[X] = 1/\lambda$

**Variance**: $\text{Var}(X) = 1/\lambda^2$

**Memoryless property**: $P(X > s+t | X > s) = P(X > t)$

**Application**: Waiting times, inter-arrival times.

### Uniform Distribution

**Continuous uniform on $[a, b]$**:
$$f_X(x) = \begin{cases} \frac{1}{b-a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}$$

**Mean**: $\mathbb{E}[X] = (a+b)/2$

**Variance**: $\text{Var}(X) = (b-a)^2/12$

## Machine Learning Applications

### Probabilistic Models

**Generative models**: Model joint distribution $p(\mathbf{x}, y)$:
- Naive Bayes: $p(\mathbf{x}, y) = p(y) \prod_i p(x_i | y)$
- Gaussian Mixture Models: $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$

**Discriminative models**: Model conditional distribution $p(y | \mathbf{x})$:
- Logistic regression: $p(y=1 | \mathbf{x}) = \sigma(\boldsymbol{\theta}^T\mathbf{x})$
- Neural networks with softmax: $p(y=k | \mathbf{x}) = \text{softmax}_k(f(\mathbf{x}; \boldsymbol{\theta}))$

### Loss Functions

**Cross-entropy loss**: For classification with true distribution $p^*(y | \mathbf{x})$ and predicted $p(y | \mathbf{x})$:

$$L = -\mathbb{E}_{p^*}[\log p(y | \mathbf{x})] = -\sum_y p^*(y | \mathbf{x}) \log p(y | \mathbf{x})$$

**Mean squared error**: For regression:

$$L = \mathbb{E}[(y - \hat{y})^2] = \text{Var}(y - \hat{y}) + (\mathbb{E}[y - \hat{y}])^2$$

### Uncertainty Quantification

**Aleatoric uncertainty**: Inherent randomness in data (modeled by probability distributions)

**Epistemic uncertainty**: Uncertainty about model parameters (handled via Bayesian inference)

**Prediction intervals**: $P(y \in [a, b] | \mathbf{x}) = \alpha$ gives $100\alpha\%$ prediction interval.

### Bayesian Inference

**Prior**: $p(\boldsymbol{\theta})$ (belief about parameters)

**Likelihood**: $p(\mathcal{D} | \boldsymbol{\theta})$ (probability of data given parameters)

**Posterior**: $p(\boldsymbol{\theta} | \mathcal{D}) = \frac{p(\mathcal{D} | \boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\mathcal{D})}$ (updated belief)

**Predictive distribution**: $p(y | \mathbf{x}, \mathcal{D}) = \int p(y | \mathbf{x}, \boldsymbol{\theta}) p(\boldsymbol{\theta} | \mathcal{D}) d\boldsymbol{\theta}$

### Maximum Likelihood Estimation

**MLE**: $\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} p(\mathcal{D} | \boldsymbol{\theta})$

For i.i.d. data: $\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n p(\mathbf{x}_i | \boldsymbol{\theta})$

**Log-likelihood**: $\ell(\boldsymbol{\theta}) = \sum_{i=1}^n \log p(\mathbf{x}_i | \boldsymbol{\theta})$

### Sampling Methods

**Monte Carlo**: Estimate expectations via samples:

$$\mathbb{E}[g(X)] \approx \frac{1}{n}\sum_{i=1}^n g(x_i)$$

where $x_i \sim p_X$.

**Importance sampling**: For $p$ hard to sample, use proposal $q$:

$$\mathbb{E}_{p}[g(X)] = \mathbb{E}_{q}\left[g(X)\frac{p(X)}{q(X)}\right] \approx \frac{1}{n}\sum_{i=1}^n g(x_i)\frac{p(x_i)}{q(x_i)}$$

### Probabilistic Graphical Models

**Bayesian networks**: Represent conditional independence via directed graph

**Markov random fields**: Represent dependencies via undirected graph

**Factor graphs**: General representation for inference algorithms

## Key Takeaways

1. **Probability measures** assign numbers to events, satisfying axioms that ensure consistency.

2. **Conditional probability** updates beliefs based on observed information, fundamental to Bayesian inference.

3. **Bayes' theorem** provides framework for updating prior beliefs with observed data to obtain posterior beliefs.

4. **Random variables** map outcomes to numbers, enabling mathematical analysis of uncertainty.

5. **Probability distributions** (PMF for discrete, PDF for continuous) fully characterize random variables.

6. **Expectation** measures average value, **variance** measures spread around mean.

7. **Common distributions** (Bernoulli, Gaussian, Poisson, etc.) model various real-world phenomena and appear throughout ML.

8. **Independence** simplifies probability calculations and is often assumed in ML models (e.g., i.i.d. data).

9. **Probabilistic modeling** enables uncertainty quantification, which is crucial for reliable ML systems.

10. **Probability theory** provides the foundation for statistical inference, loss function design, and understanding ML algorithm behavior.
