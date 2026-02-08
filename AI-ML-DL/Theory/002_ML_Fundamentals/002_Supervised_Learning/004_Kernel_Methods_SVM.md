# Kernel Methods SVM

## Table of Contents

1. [Introduction to Support Vector Machines](#introduction-to-support-vector-machines)
2. [Hard Margin SVM](#hard-margin-svm)
3. [Soft Margin SVM](#soft-margin-svm)
4. [Dual Formulation and Kernel Trick](#dual-formulation-and-kernel-trick)
5. [Kernel Functions](#kernel-functions)
6. [Multi-Class SVM](#multi-class-svm)
7. [Support Vector Regression](#support-vector-regression)
8. [Theoretical Foundations](#theoretical-foundations)
9. [Practical Considerations](#practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Support Vector Machines

Support Vector Machines (SVM) are powerful supervised learning algorithms that find optimal separating hyperplanes by maximizing the margin between classes.

### What are SVMs?

SVMs are maximum margin classifiers that:
- Find the hyperplane that maximizes the distance to the nearest data points
- Use only support vectors (points closest to decision boundary) for prediction
- Can handle non-linear boundaries through kernel functions
- Provide strong theoretical guarantees

### Key Concepts

**Separating Hyperplane**: For binary classification, a hyperplane that separates classes:
$$\mathbf{w}^T \mathbf{x} + b = 0$$

**Margin**: The distance between the hyperplane and the nearest data points. Larger margin implies better generalization.

**Support Vectors**: Training examples closest to the decision boundary that determine the hyperplane.

**Kernel Trick**: Implicit mapping to high-dimensional feature space without explicitly computing transformed features.

### Advantages

- Effective in high-dimensional spaces
- Memory efficient (uses only support vectors)
- Versatile (different kernel functions for different data types)
- Strong theoretical foundation (maximizes margin, minimizes VC dimension)

### Limitations

- Doesn't perform well on large datasets (slow training)
- Sensitive to feature scaling
- Doesn't provide probability estimates directly
- Requires careful kernel and parameter selection

## Hard Margin SVM

Hard margin SVM assumes the data is linearly separable and finds the maximum margin hyperplane.

### Problem Formulation

Given training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where $y_i \in \{-1, +1\}$, find hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ that separates classes with maximum margin.

### Geometric Intuition

The margin is the distance between the hyperplane and the nearest points. We want to maximize this distance:

$$\text{margin} = \frac{2}{\|\mathbf{w}\|}$$

Maximizing margin is equivalent to minimizing $\|\mathbf{w}\|^2$.

### Optimization Problem

**Primal Problem**:
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$

subject to constraints:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i = 1, \ldots, n$$

These constraints ensure all points are at least distance $1/\|\mathbf{w}\|$ from the hyperplane.

### Lagrangian Formulation

Introduce Lagrange multipliers $\alpha_i \geq 0$:

$$L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1]$$

### Karush-Kuhn-Tucker (KKT) Conditions

At optimum:
1. **Stationarity**: $\nabla_{\mathbf{w}} L = 0$, $\frac{\partial L}{\partial b} = 0$
2. **Primal Feasibility**: $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$
3. **Dual Feasibility**: $\alpha_i \geq 0$
4. **Complementary Slackness**: $\alpha_i[y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1] = 0$

From stationarity:
$$\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$
$$\sum_{i=1}^n \alpha_i y_i = 0$$

### Support Vectors

From complementary slackness:
- If $\alpha_i > 0$, then $y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1$ (point is on margin)
- If $\alpha_i = 0$, point is not a support vector

Only support vectors contribute to the weight vector $\mathbf{w}$.

### Decision Function

After solving, predictions are made using:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b\right)$$

where $SV$ is the set of support vector indices.

## Soft Margin SVM

Soft margin SVM handles non-separable data by allowing some misclassifications.

### Slack Variables

Introduce slack variables $\xi_i \geq 0$ to allow violations:

$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$$

- $\xi_i = 0$: Point is correctly classified with margin
- $0 < \xi_i < 1$: Point is inside margin but correctly classified
- $\xi_i \geq 1$: Point is misclassified

### Optimization Problem

**Primal Problem**:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$

subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i$$

where $C > 0$ is the regularization parameter controlling the tradeoff between margin size and classification errors.

### Interpretation of C

- **Large $C$**: Penalize errors heavily → narrower margin, fewer misclassifications (may overfit)
- **Small $C$**: Allow more errors → wider margin, more misclassifications (may underfit)
- **$C \to \infty$**: Approaches hard margin SVM

### Hinge Loss

The soft margin objective can be written using hinge loss:

$$\min_{\mathbf{w}, b} \sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)) + \lambda \|\mathbf{w}\|^2$$

where $\lambda = 1/(2C)$.

Hinge loss: $L(y, f(\mathbf{x})) = \max(0, 1 - y f(\mathbf{x}))$

## Dual Formulation and Kernel Trick

The dual formulation reveals the kernel trick, enabling non-linear SVMs.

### Dual Problem

Substituting stationarity conditions into Lagrangian gives dual problem:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

subject to:
$$0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

### Key Observation

The dual problem and decision function depend only on dot products $\mathbf{x}_i^T \mathbf{x}_j$, not on individual vectors.

### Kernel Trick

Replace dot products with kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

where $\phi$ maps to a (possibly infinite-dimensional) feature space.

**Dual Problem with Kernel**:
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

**Decision Function**:
$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

### Benefits

- **Non-linear Boundaries**: Linear in feature space, non-linear in input space
- **Computational Efficiency**: No need to compute $\phi(\mathbf{x})$ explicitly
- **High-Dimensional Spaces**: Can work in infinite-dimensional spaces

### Mercer's Theorem

A function $K$ is a valid kernel if and only if:
1. $K$ is symmetric: $K(\mathbf{x}_i, \mathbf{x}_j) = K(\mathbf{x}_j, \mathbf{x}_i)$
2. $K$ is positive semi-definite: For any $\mathbf{x}_1, \ldots, \mathbf{x}_n$, the matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite

## Kernel Functions

Different kernels capture different types of non-linear relationships.

### Linear Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$

- Equivalent to no kernel transformation
- For linearly separable data
- Fastest to compute

### Polynomial Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$

where $d$ is the degree, $\gamma$ is a scaling parameter, and $r$ is a coefficient.

**Properties**:
- $d=1$: Linear kernel
- Higher $d$: More complex boundaries
- Captures feature interactions

### Radial Basis Function (RBF) Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

where $\gamma > 0$ is a parameter (often written as $\gamma = 1/(2\sigma^2)$).

**Properties**:
- Infinite-dimensional feature space
- Local influence (decays with distance)
- Very flexible, can fit complex boundaries
- Most commonly used kernel

**Parameter $\gamma$**:
- Large $\gamma$: Narrow influence → complex boundaries, risk of overfitting
- Small $\gamma$: Wide influence → smoother boundaries, risk of underfitting

### Sigmoid Kernel

$$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$$

- Similar to neural network activation
- Not always positive semi-definite (use with caution)
- Less commonly used

### Custom Kernels

Domain-specific kernels can be designed:
- **String Kernels**: For text/sequence data
- **Graph Kernels**: For graph-structured data
- **Tree Kernels**: For parse trees
- **Fisher Kernels**: For probabilistic models

### Kernel Selection

**Guidelines**:
- **Linear**: When data is linearly separable or nearly so
- **Polynomial**: When features have meaningful interactions
- **RBF**: Default choice for non-linear problems
- **Domain-specific**: Use domain knowledge

**Parameter Tuning**: Use cross-validation to select kernel parameters ($C$, $\gamma$, $d$, etc.)

## Multi-Class SVM

SVMs are inherently binary classifiers. Several strategies extend them to multiple classes.

### One-vs-Rest (OvR)

Train $K$ binary SVMs:
- SVM$_k$: Class $k$ vs. all others
- Prediction: Class with highest decision function value

**Advantages**: Simple, $K$ classifiers needed

**Disadvantages**: Imbalanced training sets, no probability calibration

### One-vs-One (OvO)

Train $\binom{K}{2}$ binary SVMs for all pairs:
- SVM$_{ij}$: Class $i$ vs. class $j$
- Prediction: Majority vote

**Advantages**: Balanced training sets

**Disadvantages**: $O(K^2)$ classifiers, slower prediction

### Crammer-Singer Formulation

Single optimization problem for all classes:

$$\min_{\mathbf{w}_1, \ldots, \mathbf{w}_K, \boldsymbol{\xi}} \frac{1}{2} \sum_{k=1}^K \|\mathbf{w}_k\|^2 + C \sum_{i=1}^n \xi_i$$

subject to:
$$\mathbf{w}_{y_i}^T \mathbf{x}_i - \mathbf{w}_k^T \mathbf{x}_i \geq 1 - \xi_i, \quad \forall k \neq y_i$$

**Decision**: $\hat{y} = \arg\max_k \mathbf{w}_k^T \mathbf{x}$

**Advantages**: Single optimization, theoretically sound

**Disadvantages**: More complex, slower training

## Support Vector Regression

SVM can be extended to regression by using an $\epsilon$-insensitive loss.

### $\epsilon$-Insensitive Loss

$$L_\epsilon(y, f(\mathbf{x})) = \begin{cases}
0 & \text{if } |y - f(\mathbf{x})| \leq \epsilon \\
|y - f(\mathbf{x})| - \epsilon & \text{otherwise}
\end{cases}$$

Errors within $\epsilon$ are not penalized.

### Optimization Problem

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$

subject to:
$$y_i - (\mathbf{w}^T \mathbf{x}_i + b) \leq \epsilon + \xi_i$$
$$(\mathbf{w}^T \mathbf{x}_i + b) - y_i \leq \epsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

### Support Vectors

Support vectors are points:
- Outside the $\epsilon$-tube ($\xi_i > 0$ or $\xi_i^* > 0$)
- On the boundary of the $\epsilon$-tube

### Parameters

- **$C$**: Controls tradeoff between flatness and tolerance to errors
- **$\epsilon$**: Width of insensitive tube
  - Large $\epsilon$: Fewer support vectors, smoother function
  - Small $\epsilon$: More support vectors, fits data more closely

## Theoretical Foundations

SVMs have strong theoretical guarantees based on statistical learning theory.

### Margin and Generalization

Larger margin implies better generalization. The margin bound states:

With probability at least $1-\delta$:

$$R(h) \leq R_{\text{emp}}(h) + \tilde{O}\left(\sqrt{\frac{d_{\text{VC}}}{n}}\right)$$

where $d_{\text{VC}}$ is related to margin size.

### VC Dimension of SVMs

For SVMs with margin $\rho$:

$$d_{\text{VC}} \leq \min\left(\frac{R^2}{\rho^2}, d\right) + 1$$

where $R$ is the radius of the smallest sphere containing the data.

Maximizing margin minimizes VC dimension, improving generalization.

### Representer Theorem

The optimal solution of regularized risk minimization in RKHS (Reproducing Kernel Hilbert Space) has the form:

$$f^*(\mathbf{x}) = \sum_{i=1}^n \alpha_i K(\mathbf{x}_i, \mathbf{x})$$

This justifies the kernel trick theoretically.

## Practical Considerations

### Feature Scaling

SVMs are sensitive to feature scales:
- Features with larger scales dominate
- Always standardize or normalize features
- Use zero mean, unit variance scaling

### Computational Complexity

**Training**: $O(n^2)$ to $O(n^3)$ depending on implementation
- Efficient for moderate datasets ($n < 10^4$)
- For large datasets, consider:
  - Linear SVM (faster)
  - Approximate methods
  - Stochastic gradient descent variants

**Prediction**: $O(|SV| \cdot d)$ where $|SV|$ is number of support vectors
- Fast prediction (only support vectors needed)

### Parameter Selection

**Grid Search**: Search over $(C, \gamma)$ grid using cross-validation

**Common Ranges**:
- $C \in \{10^{-3}, 10^{-2}, \ldots, 10^3\}$
- $\gamma \in \{10^{-3}, 10^{-2}, \ldots, 10^3\}$ (for RBF)

**Guidelines**:
- Start with default RBF kernel
- Use cross-validation for parameter selection
- Consider linear kernel for large datasets

### Handling Imbalanced Data

**Strateges**:
- Use class weights: $C_i = C \cdot w_i$ where $w_i$ is weight for class $i$
- Adjust decision threshold
- Use appropriate evaluation metrics (precision, recall, F1)

### Probability Estimates

SVMs don't provide probabilities directly. Use:
- **Platt Scaling**: Fit sigmoid to decision function values
- **Cross-Validation**: Estimate probabilities from CV predictions

## Key Takeaways

1. **Support Vector Machines** find maximum margin hyperplanes, using only support vectors for prediction and providing strong generalization guarantees.

2. **Hard Margin SVM** assumes linear separability and maximizes margin by minimizing $\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$.

3. **Soft Margin SVM** handles non-separable data using slack variables, balancing margin size and classification errors via parameter $C$.

4. **Dual Formulation** reveals dependence only on dot products, enabling the kernel trick to work in high-dimensional feature spaces without explicit transformation.

5. **Kernel Functions** (linear, polynomial, RBF) enable non-linear boundaries, with RBF kernel $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ being the most commonly used.

6. **Multi-Class SVM** extends binary classification via one-vs-rest, one-vs-one, or Crammer-Singer formulation for $K$ classes.

7. **Support Vector Regression** uses $\epsilon$-insensitive loss, creating a tube where errors within $\epsilon$ are not penalized, controlled by parameters $C$ and $\epsilon$.

8. **Theoretical Foundations** show that maximizing margin minimizes VC dimension, with representer theorem justifying the kernel trick in RKHS.

9. **Feature Scaling** is critical as SVMs are sensitive to feature scales, requiring standardization before training.

10. **Practical Considerations** include computational complexity ($O(n^2)$-$O(n^3)$ training), parameter selection via grid search with cross-validation, and handling imbalanced data through class weights.
