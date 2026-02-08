# Probabilistic Models Naive Bayes

## Table of Contents

1. [Introduction to Probabilistic Models](#introduction-to-probabilistic-models)
2. [Bayes' Theorem](#bayes-theorem)
3. [Naive Bayes Classifier](#naive-bayes-classifier)
4. [Gaussian Naive Bayes](#gaussian-naive-bayes)
5. [Multinomial Naive Bayes](#multinomial-naive-bayes)
6. [Bernoulli Naive Bayes](#bernoulli-naive-bayes)
7. [Conditional Independence Assumption](#conditional-independence-assumption)
8. [Gaussian Discriminant Analysis](#gaussian-discriminant-analysis)
9. [Parameter Estimation](#parameter-estimation)
10. [Key Takeaways](#key-takeaways)

## Introduction to Probabilistic Models

Probabilistic models provide a principled framework for classification by modeling the probability distribution of classes given features.

### What are Probabilistic Models?

Probabilistic classification models:
- Estimate probability distributions $P(y | \mathbf{x})$
- Make predictions using Bayes' decision rule
- Provide uncertainty estimates (probability of each class)
- Can handle missing data naturally

### Advantages

- **Probabilistic Output**: Provides class probabilities, not just predictions
- **Theoretical Foundation**: Based on probability theory and Bayes' theorem
- **Interpretability**: Probabilities are intuitive and interpretable
- **Handles Uncertainty**: Quantifies prediction uncertainty
- **Efficient**: Often fast training and prediction

### Types of Probabilistic Models

- **Generative Models**: Model joint distribution $P(\mathbf{x}, y)$
  - Naive Bayes
  - Gaussian Discriminant Analysis
  - Hidden Markov Models

- **Discriminative Models**: Model conditional distribution $P(y | \mathbf{x})$ directly
  - Logistic Regression
  - Conditional Random Fields

### Generative vs Discriminative

**Generative**: $P(y | \mathbf{x}) = \frac{P(\mathbf{x} | y) P(y)}{P(\mathbf{x})}$
- Model how data is generated
- Can generate new samples
- Often need more data

**Discriminative**: $P(y | \mathbf{x})$ directly
- Focus on classification boundary
- Often better performance with sufficient data
- Cannot generate samples

## Bayes' Theorem

Bayes' theorem provides the foundation for probabilistic classification.

### Statement

For events $A$ and $B$:

$$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$

### Application to Classification

For classification with features $\mathbf{x}$ and class $y$:

$$P(y | \mathbf{x}) = \frac{P(\mathbf{x} | y) P(y)}{P(\mathbf{x})}$$

**Components**:
- **Posterior**: $P(y | \mathbf{x})$ - probability of class given features
- **Likelihood**: $P(\mathbf{x} | y)$ - probability of features given class
- **Prior**: $P(y)$ - prior probability of class
- **Evidence**: $P(\mathbf{x})$ - probability of features (normalizing constant)

### Maximum A Posteriori (MAP) Decision

Choose class with highest posterior probability:

$$\hat{y} = \arg\max_{y} P(y | \mathbf{x}) = \arg\max_{y} P(\mathbf{x} | y) P(y)$$

Since $P(\mathbf{x})$ is the same for all classes, we can ignore it.

### Maximum Likelihood (ML) Decision

If priors are equal ($P(y)$ uniform), MAP reduces to ML:

$$\hat{y} = \arg\max_{y} P(\mathbf{x} | y)$$

### Optimal Decision Rule

Under 0-1 loss, the Bayes optimal classifier minimizes expected risk:

$$\hat{y} = \arg\max_{y} P(y | \mathbf{x})$$

This is the optimal classifier if we know the true distributions.

## Naive Bayes Classifier

Naive Bayes makes the "naive" assumption that features are conditionally independent given the class.

### Conditional Independence Assumption

$$P(\mathbf{x} | y) = \prod_{j=1}^d P(x_j | y)$$

This assumes features are independent given the class label.

### Naive Bayes Formula

Using Bayes' theorem and conditional independence:

$$P(y | \mathbf{x}) = \frac{P(y) \prod_{j=1}^d P(x_j | y)}{P(\mathbf{x})}$$

Since $P(\mathbf{x})$ is constant for all classes:

$$P(y | \mathbf{x}) \propto P(y) \prod_{j=1}^d P(x_j | y)$$

### Prediction Rule

$$\hat{y} = \arg\max_{y} P(y) \prod_{j=1}^d P(x_j | y)$$

### Log-Space Computation

To avoid numerical underflow, work in log space:

$$\log P(y | \mathbf{x}) \propto \log P(y) + \sum_{j=1}^d \log P(x_j | y)$$

$$\hat{y} = \arg\max_{y} \left[\log P(y) + \sum_{j=1}^d \log P(x_j | y)\right]$$

### Why "Naive"?

The independence assumption is rarely true in practice, but Naive Bayes often works well because:
- We only need the correct ordering of probabilities, not exact values
- Dependencies may cancel out
- Works well for high-dimensional data
- Robust to violations of independence

## Gaussian Naive Bayes

Gaussian Naive Bayes assumes continuous features follow Gaussian distributions.

### Model Assumptions

For each class $y$ and feature $j$:

$$P(x_j | y) = \mathcal{N}(x_j; \mu_{yj}, \sigma_{yj}^2) = \frac{1}{\sqrt{2\pi\sigma_{yj}^2}} \exp\left(-\frac{(x_j - \mu_{yj})^2}{2\sigma_{yj}^2}\right)$$

### Parameters

For each class $y$ and feature $j$:
- **Mean**: $\mu_{yj} = \mathbb{E}[x_j | y]$
- **Variance**: $\sigma_{yj}^2 = \text{Var}(x_j | y)$

### Parameter Estimation

**Maximum Likelihood Estimates**:

$$\hat{\mu}_{yj} = \frac{1}{n_y} \sum_{i: y_i = y} x_{ij}$$

$$\hat{\sigma}_{yj}^2 = \frac{1}{n_y} \sum_{i: y_i = y} (x_{ij} - \hat{\mu}_{yj})^2$$

where $n_y$ is the number of samples with class $y$.

**Prior**:
$$\hat{P}(y) = \frac{n_y}{n}$$

### Prediction

For a new example $\mathbf{x}$:

$$\hat{y} = \arg\max_{y} \left[\log \hat{P}(y) + \sum_{j=1}^d \log \mathcal{N}(x_j; \hat{\mu}_{yj}, \hat{\sigma}_{yj}^2)\right]$$

### Decision Boundary

The decision boundary between classes $y_1$ and $y_2$ is quadratic (due to different variances) or linear (if variances are equal).

### Advantages

- Works well with continuous features
- Fast training and prediction
- Handles missing features naturally (marginalize them out)

### Limitations

- Assumes Gaussian distribution (may not hold)
- Assumes feature independence (rarely true)
- Can be sensitive to outliers

## Multinomial Naive Bayes

Multinomial Naive Bayes is designed for discrete count data, commonly used for text classification.

### Model Assumptions

Features are counts (e.g., word counts in documents). For feature $j$ in class $y$:

$$P(x_j | y) = \theta_{yj}$$

where $\theta_{yj}$ is the probability of feature $j$ occurring in class $y$.

### Constraints

For each class $y$:
$$\sum_{j=1}^d \theta_{yj} = 1$$

### Likelihood

For a document with feature vector $\mathbf{x} = [x_1, x_2, \ldots, x_d]$ (counts):

$$P(\mathbf{x} | y) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_{j=1}^d \theta_{yj}^{x_j}$$

The multinomial coefficient is constant, so:

$$P(\mathbf{x} | y) \propto \prod_{j=1}^d \theta_{yj}^{x_j}$$

### Parameter Estimation

**Maximum Likelihood with Laplace Smoothing**:

$$\hat{\theta}_{yj} = \frac{N_{yj} + \alpha}{N_y + \alpha d}$$

where:
- $N_{yj} = \sum_{i: y_i = y} x_{ij}$: total count of feature $j$ in class $y$
- $N_y = \sum_j N_{yj}$: total count of all features in class $y$
- $\alpha$: smoothing parameter (typically $\alpha = 1$ for Laplace smoothing)

**Without Smoothing** ($\alpha = 0$):
$$\hat{\theta}_{yj} = \frac{N_{yj}}{N_y}$$

### Text Classification Example

For document classification:
- Features: word counts or TF-IDF values
- $x_j$: count of word $j$ in document
- $\theta_{yj}$: probability of word $j$ in class $y$

### Advantages

- Natural for count data (text, bags of words)
- Handles high-dimensional sparse data well
- Fast and scalable

### Limitations

- Ignores word order (bag of words assumption)
- Requires smoothing to handle unseen words
- May not work well with very long documents

## Bernoulli Naive Bayes

Bernoulli Naive Bayes models binary features (presence/absence), suitable for binary feature vectors.

### Model Assumptions

Each feature $x_j \in \{0, 1\}$ follows a Bernoulli distribution:

$$P(x_j | y) = \theta_{yj}^{x_j} (1 - \theta_{yj})^{1-x_j}$$

where $\theta_{yj} = P(x_j = 1 | y)$.

### Likelihood

$$P(\mathbf{x} | y) = \prod_{j=1}^d \theta_{yj}^{x_j} (1 - \theta_{yj})^{1-x_j}$$

### Parameter Estimation

$$\hat{\theta}_{yj} = \frac{\sum_{i: y_i = y} \mathbb{1}(x_{ij} = 1) + \alpha}{n_y + 2\alpha}$$

where $\alpha$ is smoothing parameter.

### Text Classification

For binary text features (word present/absent):
- $x_j = 1$ if word $j$ appears in document
- $x_j = 0$ otherwise
- Different from Multinomial: doesn't consider word counts, only presence

### Comparison with Multinomial

**Multinomial**: Models word counts, better for longer documents
**Bernoulli**: Models word presence/absence, better for short documents or when word frequency is less important

## Conditional Independence Assumption

The naive assumption of conditional independence is central to Naive Bayes.

### What It Means

$$P(\mathbf{x} | y) = \prod_{j=1}^d P(x_j | y)$$

Features are independent given the class label.

### Why It Often Works

Despite being unrealistic, Naive Bayes performs well because:

1. **Ranking Matters**: We only need correct class ordering, not exact probabilities
2. **Dependencies Cancel**: Some dependencies may cancel out in the ratio
3. **High Dimensions**: In high dimensions, independence assumption becomes less problematic
4. **Robustness**: Model is robust to violations of independence

### When It Fails

Naive Bayes may perform poorly when:
- Features are highly correlated
- Feature interactions are important
- Dependencies are strong and systematic

### Relaxing the Assumption

**Tree-Augmented Naive Bayes (TAN)**:
- Allows a tree structure of dependencies
- More complex but still tractable

**Bayesian Networks**:
- General graphical models
- Can model arbitrary dependencies
- More complex inference

## Gaussian Discriminant Analysis

Gaussian Discriminant Analysis (GDA) models classes using multivariate Gaussian distributions without the naive independence assumption.

### Model

For each class $y$:

$$P(\mathbf{x} | y) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_y, \Sigma_y)$$

where:
- $\boldsymbol{\mu}_y$: mean vector for class $y$
- $\Sigma_y$: covariance matrix for class $y$

### Quadratic Discriminant Analysis (QDA)

Each class has its own covariance matrix $\Sigma_y$.

**Decision Boundary**: Quadratic (due to different covariances)

**Parameters**: $K$ means + $K$ covariance matrices = $Kd + Kd(d+1)/2$ parameters

### Linear Discriminant Analysis (LDA)

All classes share the same covariance matrix $\Sigma_y = \Sigma$.

**Decision Boundary**: Linear (due to shared covariance)

**Parameters**: $K$ means + 1 covariance matrix = $Kd + d(d+1)/2$ parameters

### Parameter Estimation

**QDA**:
$$\hat{\boldsymbol{\mu}}_y = \frac{1}{n_y} \sum_{i: y_i = y} \mathbf{x}_i$$

$$\hat{\Sigma}_y = \frac{1}{n_y} \sum_{i: y_i = y} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_y)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_y)^T$$

**LDA**:
$$\hat{\boldsymbol{\mu}}_y = \frac{1}{n_y} \sum_{i: y_i = y} \mathbf{x}_i$$

$$\hat{\Sigma} = \frac{1}{n} \sum_{y} \sum_{i: y_i = y} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_y)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_y)^T$$

### Prediction

$$\hat{y} = \arg\max_{y} \left[\log P(y) - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_y)^T \Sigma_y^{-1}(\mathbf{x} - \boldsymbol{\mu}_y) - \frac{1}{2}\log|\Sigma_y|\right]$$

### Comparison with Naive Bayes

**Gaussian Naive Bayes**: Assumes $\Sigma_y$ is diagonal (independent features)
**GDA**: Allows full covariance matrices (correlated features)

**When to Use**:
- **Naive Bayes**: When features are approximately independent or for high-dimensional data
- **LDA**: When features are correlated but classes have similar covariance
- **QDA**: When classes have different covariance structures

## Parameter Estimation

Parameter estimation is crucial for probabilistic models.

### Maximum Likelihood Estimation (MLE)

Find parameters that maximize likelihood of observed data:

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n P(\mathbf{x}_i, y_i | \boldsymbol{\theta})$$

### Maximum A Posteriori (MAP)

Incorporate prior beliefs:

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} P(\boldsymbol{\theta} | \mathcal{D}) = \arg\max_{\boldsymbol{\theta}} P(\mathcal{D} | \boldsymbol{\theta}) P(\boldsymbol{\theta})$$

### Smoothing

**Laplace Smoothing (Additive Smoothing)**:
Add small constant $\alpha$ to counts:

$$\hat{\theta} = \frac{\text{count} + \alpha}{\text{total} + \alpha \cdot \text{number of values}}$$

**Why Smoothing**:
- Prevents zero probabilities (unseen events)
- Incorporates prior belief of uniform distribution
- Improves generalization

### Bayesian Estimation

Full Bayesian approach:
- Specify prior $P(\boldsymbol{\theta})$
- Compute posterior $P(\boldsymbol{\theta} | \mathcal{D})$
- Predict using posterior predictive distribution

More principled but computationally intensive.

## Key Takeaways

1. **Probabilistic Models** provide class probabilities using Bayes' theorem: $P(y | \mathbf{x}) = \frac{P(\mathbf{x} | y) P(y)}{P(\mathbf{x})}$, enabling uncertainty quantification.

2. **Naive Bayes** assumes conditional independence: $P(\mathbf{x} | y) = \prod_j P(x_j | y)$, making it efficient despite unrealistic assumption.

3. **Gaussian Naive Bayes** models continuous features as $P(x_j | y) = \mathcal{N}(\mu_{yj}, \sigma_{yj}^2)$, estimating means and variances per class and feature.

4. **Multinomial Naive Bayes** models count data with $P(\mathbf{x} | y) \propto \prod_j \theta_{yj}^{x_j}$, using Laplace smoothing $\hat{\theta}_{yj} = \frac{N_{yj} + \alpha}{N_y + \alpha d}$.

5. **Bernoulli Naive Bayes** models binary features with $P(x_j | y) = \theta_{yj}^{x_j}(1-\theta_{yj})^{1-x_j}$, suitable for presence/absence features.

6. **Conditional Independence** assumption often works well despite being unrealistic because ranking matters more than exact probabilities, and dependencies may cancel.

7. **Gaussian Discriminant Analysis** models $P(\mathbf{x} | y) = \mathcal{N}(\boldsymbol{\mu}_y, \Sigma_y)$ without independence, with LDA sharing covariance and QDA using class-specific covariances.

8. **Parameter Estimation** uses MLE with smoothing (Laplace) to prevent zero probabilities: $\hat{\theta} = \frac{\text{count} + \alpha}{\text{total} + \alpha \cdot \text{values}}$.

9. **Advantages** include probabilistic output, fast training/prediction, natural handling of missing data, and good performance on high-dimensional data.

10. **Applications** span text classification (Multinomial/Bernoulli), continuous data (Gaussian), and scenarios requiring probability estimates and interpretability.
