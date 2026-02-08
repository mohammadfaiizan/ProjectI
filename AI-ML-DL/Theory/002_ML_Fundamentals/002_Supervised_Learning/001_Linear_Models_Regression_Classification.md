# Linear Models Regression Classification

## Table of Contents

1. [Introduction to Linear Models](#introduction-to-linear-models)
2. [Linear Regression](#linear-regression)
3. [Least Squares Estimation](#least-squares-estimation)
4. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
5. [Logistic Regression](#logistic-regression)
6. [Regularized Linear Models](#regularized-linear-models)
7. [Generalized Linear Models](#generalized-linear-models)
8. [Model Assumptions and Diagnostics](#model-assumptions-and-diagnostics)
9. [Multiclass Classification](#multiclass-classification)
10. [Key Takeaways](#key-takeaways)

## Introduction to Linear Models

Linear models form the foundation of many machine learning algorithms, providing interpretable, efficient, and often effective solutions to regression and classification problems.

### What are Linear Models?

Linear models make predictions using a linear combination of input features:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_d x_d = w_0 + \sum_{i=1}^d w_i x_i$$

In vector notation:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + w_0$$

where $\mathbf{w} = [w_1, w_2, \ldots, w_d]^T$ are weights, $\mathbf{x} = [x_1, x_2, \ldots, x_d]^T$ are features, and $w_0$ is the bias term (intercept).

### Advantages of Linear Models

- **Interpretability**: Coefficients directly indicate feature importance
- **Efficiency**: Fast training and prediction
- **Stability**: Less prone to overfitting with regularization
- **Theoretical Foundation**: Well-understood statistical properties
- **Baseline Performance**: Often serve as strong baselines

### Limitations

- **Linearity Assumption**: Assumes linear relationships
- **Feature Interactions**: Cannot capture interactions without explicit features
- **Non-linear Boundaries**: Limited for complex decision boundaries

### Extending Linearity

Linear models can handle non-linear relationships through:
- **Polynomial Features**: $x^2, x^3, x_1 x_2$
- **Basis Functions**: $\phi(x)$ transforms inputs to feature space
- **Kernel Methods**: Implicit feature transformations

## Linear Regression

Linear regression models the relationship between a continuous target variable and input features using a linear function.

### Problem Formulation

Given training data $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$, where $y_i \in \mathbb{R}$, find weights $\mathbf{w}$ that minimize prediction error.

### Model Definition

The linear regression model is:

$$y = \mathbf{w}^T \mathbf{x} + w_0 + \epsilon$$

where $\epsilon$ is noise (typically assumed Gaussian: $\epsilon \sim \mathcal{N}(0, \sigma^2)$).

Including bias in the weight vector by adding a constant feature $x_0 = 1$:

$$y = \mathbf{w}^T \mathbf{x} + \epsilon$$

where $\mathbf{x} = [1, x_1, x_2, \ldots, x_d]^T$ and $\mathbf{w} = [w_0, w_1, w_2, \ldots, w_d]^T$.

### Matrix Notation

For $n$ samples, the model becomes:

$$\mathbf{y} = X\mathbf{w} + \boldsymbol{\epsilon}$$

where:
- $\mathbf{y} \in \mathbb{R}^n$: target vector
- $X \in \mathbb{R}^{n \times (d+1)}$: design matrix (each row is a sample)
- $\mathbf{w} \in \mathbb{R}^{d+1}$: weight vector
- $\boldsymbol{\epsilon} \in \mathbb{R}^n$: error vector

### Objective Function

The goal is to minimize the sum of squared errors (SSE):

$$\text{SSE} = \sum_{i=1}^n (y_i - \mathbf{w}^T \mathbf{x}_i)^2 = \|\mathbf{y} - X\mathbf{w}\|^2$$

This is equivalent to minimizing mean squared error (MSE):

$$\text{MSE} = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2$$

## Least Squares Estimation

The least squares method finds weights that minimize the sum of squared residuals.

### Normal Equations

Taking the gradient of SSE with respect to $\mathbf{w}$ and setting to zero:

$$\nabla_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|^2 = -2X^T(\mathbf{y} - X\mathbf{w}) = 0$$

Solving gives the normal equations:

$$X^T X \mathbf{w} = X^T \mathbf{y}$$

### Closed-Form Solution

If $X^T X$ is invertible, the solution is:

$$\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$$

This is the ordinary least squares (OLS) estimator.

### Geometric Interpretation

The least squares solution projects $\mathbf{y}$ onto the column space of $X$:

$$\hat{\mathbf{y}} = X\hat{\mathbf{w}} = X(X^T X)^{-1} X^T \mathbf{y} = P_X \mathbf{y}$$

where $P_X = X(X^T X)^{-1} X^T$ is the projection matrix.

### Rank Deficiency

If $X^T X$ is singular (not invertible):
- **Multicollinearity**: Features are linearly dependent
- **More features than samples**: $d+1 > n$
- **Solution**: Use pseudoinverse or regularization

### Computational Considerations

**Time Complexity**: $O(nd^2 + d^3)$ for matrix multiplication and inversion
- For large $d$, use iterative methods (gradient descent)
- For large $n$, use stochastic gradient descent

**Numerical Stability**: 
- Use QR decomposition: $X = QR$, then solve $R\mathbf{w} = Q^T\mathbf{y}$
- Or SVD: $X = U\Sigma V^T$, then $\hat{\mathbf{w}} = V\Sigma^{-1}U^T\mathbf{y}$

## Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) provides a probabilistic framework for linear regression.

### Probabilistic Model

Assuming Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$:

$$y | \mathbf{x}, \mathbf{w} \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}, \sigma^2)$$

The probability density function is:

$$p(y | \mathbf{x}, \mathbf{w}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \mathbf{w}^T \mathbf{x})^2}{2\sigma^2}\right)$$

### Likelihood Function

For independent samples, the likelihood is:

$$L(\mathbf{w}, \sigma^2) = \prod_{i=1}^n p(y_i | \mathbf{x}_i, \mathbf{w}, \sigma^2)$$

Taking the log:

$$\ell(\mathbf{w}, \sigma^2) = \sum_{i=1}^n \log p(y_i | \mathbf{x}_i, \mathbf{w}, \sigma^2)$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

### MLE Solution

Maximizing log-likelihood with respect to $\mathbf{w}$:

$$\frac{\partial \ell}{\partial \mathbf{w}} = \frac{1}{\sigma^2}X^T(\mathbf{y} - X\mathbf{w}) = 0$$

This gives the same solution as least squares:

$$\hat{\mathbf{w}}_{\text{MLE}} = (X^T X)^{-1} X^T \mathbf{y}$$

For variance:

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{\mathbf{w}}^T \mathbf{x}_i)^2$$

### Properties of MLE

- **Consistency**: $\hat{\mathbf{w}} \to \mathbf{w}^*$ as $n \to \infty$
- **Asymptotic Normality**: $\hat{\mathbf{w}} \sim \mathcal{N}(\mathbf{w}^*, (X^T X)^{-1}\sigma^2)$
- **Efficiency**: Achieves Cramér-Rao lower bound
- **Invariance**: MLE of $g(\mathbf{w})$ is $g(\hat{\mathbf{w}})$

## Logistic Regression

Logistic regression extends linear models to binary classification by modeling class probabilities.

### Problem Formulation

For binary classification with $y \in \{0, 1\}$, we model the probability:

$$P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid (logistic) function.

### Sigmoid Function Properties

The sigmoid function maps real numbers to $(0, 1)$:

- $\sigma(z) \to 0$ as $z \to -\infty$
- $\sigma(z) \to 1$ as $z \to \infty$
- $\sigma(0) = 0.5$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### Decision Boundary

The decision boundary occurs where $P(y = 1 | \mathbf{x}) = 0.5$, i.e., $\mathbf{w}^T \mathbf{x} = 0$.

This is a linear decision boundary (hyperplane) in feature space.

### Likelihood Function

For binary classification, the likelihood is:

$$L(\mathbf{w}) = \prod_{i=1}^n P(y_i | \mathbf{x}_i, \mathbf{w}) = \prod_{i=1}^n \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{1-y_i}$$

### Log-Likelihood

$$\ell(\mathbf{w}) = \sum_{i=1}^n [y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i) + (1-y_i) \log(1 - \sigma(\mathbf{w}^T \mathbf{x}_i))]$$

$$= \sum_{i=1}^n [y_i \mathbf{w}^T \mathbf{x}_i - \log(1 + e^{\mathbf{w}^T \mathbf{x}_i})]$$

### Gradient and Hessian

The gradient is:

$$\nabla_{\mathbf{w}} \ell(\mathbf{w}) = \sum_{i=1}^n \mathbf{x}_i (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i)) = X^T(\mathbf{y} - \boldsymbol{\sigma})$$

where $\boldsymbol{\sigma} = [\sigma(\mathbf{w}^T \mathbf{x}_1), \ldots, \sigma(\mathbf{w}^T \mathbf{x}_n)]^T$.

The Hessian is:

$$H(\mathbf{w}) = -\sum_{i=1}^n \sigma(\mathbf{w}^T \mathbf{x}_i)(1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \mathbf{x}_i \mathbf{x}_i^T = -X^T \text{diag}(\boldsymbol{\sigma} \odot (1 - \boldsymbol{\sigma})) X$$

The Hessian is negative definite, ensuring the log-likelihood is concave and has a unique maximum.

### Optimization

Since there's no closed-form solution, use iterative methods:

**Newton's Method**:
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - H^{-1}(\mathbf{w}^{(t)}) \nabla_{\mathbf{w}} \ell(\mathbf{w}^{(t)})$$

**Gradient Ascent**:
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \alpha \nabla_{\mathbf{w}} \ell(\mathbf{w}^{(t)})$$

**IRLS (Iteratively Reweighted Least Squares)**: Reformulates as weighted least squares at each iteration.

## Regularized Linear Models

Regularization prevents overfitting by penalizing large weights.

### Ridge Regression (L2 Regularization)

Adds penalty proportional to squared weights:

$$\min_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_2^2$$

where $\lambda > 0$ is the regularization parameter.

**Solution**:
$$\hat{\mathbf{w}}_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$$

**Properties**:
- Always invertible (adds $\lambda$ to diagonal)
- Shrinks weights toward zero
- Doesn't perform feature selection
- Biased but lower variance than OLS

### Lasso Regression (L1 Regularization)

Adds penalty proportional to absolute weights:

$$\min_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_1$$

where $\|\mathbf{w}\|_1 = \sum_{i=1}^d |w_i|$.

**Properties**:
- Performs feature selection (sets some weights to exactly zero)
- Creates sparse solutions
- Useful for high-dimensional data
- No closed-form solution (use coordinate descent, LARS)

### Elastic Net

Combines L1 and L2 regularization:

$$\min_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

**Advantages**:
- Handles correlated features better than Lasso
- More stable than Lasso
- Can select groups of correlated features

### Regularized Logistic Regression

Apply regularization to logistic regression:

**Ridge Logistic Regression**:
$$\min_{\mathbf{w}} -\ell(\mathbf{w}) + \lambda \|\mathbf{w}\|_2^2$$

**Lasso Logistic Regression**:
$$\min_{\mathbf{w}} -\ell(\mathbf{w}) + \lambda \|\mathbf{w}\|_1$$

### Choosing Regularization Parameter

**Cross-Validation**: Select $\lambda$ that minimizes validation error.

**Information Criteria**: AIC, BIC balance fit and complexity.

**Path Algorithms**: Compute solutions for all $\lambda$ values efficiently (e.g., LARS for Lasso).

## Generalized Linear Models

Generalized Linear Models (GLMs) extend linear models to handle various response distributions.

### GLM Components

A GLM consists of:

1. **Random Component**: Response distribution from exponential family
   $$p(y | \theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)$$

2. **Systematic Component**: Linear predictor $\eta = \mathbf{w}^T \mathbf{x}$

3. **Link Function**: Connects mean to linear predictor $g(\mu) = \eta$

### Common GLMs

**Linear Regression**: 
- Distribution: Gaussian
- Link: Identity $g(\mu) = \mu$

**Logistic Regression**:
- Distribution: Bernoulli
- Link: Logit $g(\mu) = \log(\mu/(1-\mu))$

**Poisson Regression**:
- Distribution: Poisson
- Link: Log $g(\mu) = \log(\mu)$

**Exponential Regression**:
- Distribution: Exponential
- Link: Log $g(\mu) = \log(\mu)$

### Maximum Likelihood for GLMs

The log-likelihood is:

$$\ell(\mathbf{w}) = \sum_{i=1}^n \frac{y_i \theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi)$$

where $\theta_i$ relates to $\mu_i$ through the canonical link.

### Iteratively Reweighted Least Squares

GLMs are fit using IRLS, which iteratively solves weighted least squares problems.

## Model Assumptions and Diagnostics

Linear models make several assumptions that should be validated.

### Assumptions

1. **Linearity**: $E[y | \mathbf{x}] = \mathbf{w}^T \mathbf{x}$
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance $\text{Var}(\epsilon) = \sigma^2$
4. **Normality**: Errors are normally distributed (for inference)
5. **No Multicollinearity**: Features are not perfectly correlated

### Diagnostic Methods

**Residual Plots**: Plot residuals vs. predicted values
- Should show random scatter
- Patterns indicate violations

**Q-Q Plots**: Check normality of residuals
- Points should lie on diagonal line

**Cook's Distance**: Identify influential observations
$$D_i = \frac{(y_i - \hat{y}_i)^2}{p \cdot \text{MSE}} \cdot \frac{h_i}{(1-h_i)^2}$$
where $h_i$ is leverage.

**Variance Inflation Factor (VIF)**: Detect multicollinearity
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is $R^2$ from regressing $x_j$ on other features.

**Durbin-Watson Test**: Check for autocorrelation in residuals.

### Remedies for Violations

- **Non-linearity**: Add polynomial features, use transformations
- **Heteroscedasticity**: Weighted least squares, transformations
- **Non-normality**: Transformations, robust methods
- **Multicollinearity**: Remove features, use regularization, PCA

## Multiclass Classification

Extend binary classification to multiple classes.

### One-vs-Rest (OvR)

Train $K$ binary classifiers, one per class:
- Classifier $k$: class $k$ vs. all others
- Prediction: class with highest score

**Advantages**: Simple, works with any binary classifier

**Disadvantages**: Imbalanced training sets, no probability calibration

### One-vs-One (OvO)

Train $\binom{K}{2}$ binary classifiers for all pairs:
- Classifier $(i,j)$: class $i$ vs. class $j$
- Prediction: majority vote

**Advantages**: Balanced training sets

**Disadvantages**: More classifiers needed, slower

### Multinomial Logistic Regression

Direct extension to $K$ classes using softmax:

$$P(y = k | \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

This is the softmax function, generalizing sigmoid to multiple classes.

**Log-Likelihood**:
$$\ell(\mathbf{W}) = \sum_{i=1}^n \sum_{k=1}^K \mathbb{1}(y_i = k) \log P(y_i = k | \mathbf{x}_i)$$

where $\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_K]$.

**Note**: One weight vector is redundant (set $\mathbf{w}_K = \mathbf{0}$ for identifiability).

## Key Takeaways

1. **Linear Models** make predictions using linear combinations of features: $\hat{y} = \mathbf{w}^T \mathbf{x} + w_0$, providing interpretable and efficient solutions.

2. **Linear Regression** minimizes sum of squared errors, with closed-form solution $\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$ via normal equations.

3. **Maximum Likelihood Estimation** provides probabilistic framework, showing OLS is MLE under Gaussian noise assumptions.

4. **Logistic Regression** extends linear models to binary classification using sigmoid function: $P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$, optimized via gradient ascent or Newton's method.

5. **Ridge Regression** (L2) shrinks weights toward zero: $\min \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_2^2$, always invertible and reducing variance.

6. **Lasso Regression** (L1) performs feature selection: $\min \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_1$, creating sparse solutions with some weights exactly zero.

7. **Elastic Net** combines L1 and L2 regularization, handling correlated features better than Lasso alone.

8. **Generalized Linear Models** extend to various distributions (Gaussian, Bernoulli, Poisson) through exponential family and link functions.

9. **Model Diagnostics** validate assumptions (linearity, independence, homoscedasticity, normality) through residual plots, Q-Q plots, VIF, and other tests.

10. **Multiclass Classification** uses one-vs-rest, one-vs-one, or multinomial logistic regression with softmax function for $K$ classes.
