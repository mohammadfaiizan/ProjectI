# Multivariable Calculus and Optimization

## Partial Derivatives

### Definition

For function $f: \mathbb{R}^n \to \mathbb{R}$, partial derivative with respect to $x_i$:

$$\frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(\mathbf{x})}{h}$$

**Notation**: $f_{x_i}$, $\partial_i f$, $D_i f$.

### Higher-Order Partial Derivatives

**Second-order**: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_j}\right)$

**Schwarz's theorem**: If mixed partials are continuous, then:

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

(order of differentiation doesn't matter).

## Gradient

### Definition

Gradient of $f: \mathbb{R}^n \to \mathbb{R}$:

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$$

**Properties**:
- Points in direction of steepest ascent
- Perpendicular to level sets: $\nabla f(\mathbf{x}) \perp \{\mathbf{y} : f(\mathbf{y}) = f(\mathbf{x})\}$
- Directional derivative: $D_{\mathbf{v}}f(\mathbf{x}) = \nabla f(\mathbf{x})^T \mathbf{v}$

### Chain Rule

For $f(\mathbf{g}(t))$ where $\mathbf{g}: \mathbb{R} \to \mathbb{R}^n$:

$$\frac{d}{dt}f(\mathbf{g}(t)) = \nabla f(\mathbf{g}(t))^T \mathbf{g}'(t) = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \frac{dg_i}{dt}$$

For $f(\mathbf{g}(\mathbf{x}))$ where $\mathbf{g}: \mathbb{R}^m \to \mathbb{R}^n$:

$$\frac{\partial f}{\partial x_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial g_i} \frac{\partial g_i}{\partial x_j}$$

## Hessian

### Definition

Hessian matrix of $f: \mathbb{R}^n \to \mathbb{R}$:

$$\mathbf{H}_f(\mathbf{x}) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

**Properties**:
- Symmetric: $\mathbf{H}_f = \mathbf{H}_f^T$ (if second derivatives continuous)
- Second-order Taylor expansion:

$$f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T\mathbf{h} + \frac{1}{2}\mathbf{h}^T\mathbf{H}_f(\mathbf{x})\mathbf{h}$$

## Jacobian

### Definition

For vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$\mathbf{J}_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

**Properties**:
- Chain rule: $\mathbf{J}_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x}) = \mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \mathbf{J}_{\mathbf{g}}(\mathbf{x})$
- Inverse function theorem: If $\mathbf{J}_{\mathbf{f}}(\mathbf{a})$ invertible, then $\mathbf{f}$ locally invertible near $\mathbf{a}$

### Determinant (Jacobian Determinant)

For $m = n$, Jacobian determinant:

$$J(\mathbf{x}) = \det(\mathbf{J}_{\mathbf{f}}(\mathbf{x}))$$

**Change of variables**: For transformation $\mathbf{y} = \mathbf{f}(\mathbf{x})$:

$$\int_{\mathbf{f}(D)} g(\mathbf{y}) d\mathbf{y} = \int_D g(\mathbf{f}(\mathbf{x})) |J(\mathbf{x})| d\mathbf{x}$$

## Taylor Expansion (Multivariate)

### First-Order (Linear Approximation)

$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^T\mathbf{h} + o(\|\mathbf{h}\|)$$

### Second-Order (Quadratic Approximation)

$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^T\mathbf{h} + \frac{1}{2}\mathbf{h}^T\mathbf{H}_f(\mathbf{x})\mathbf{h} + o(\|\mathbf{h}\|^2)$$

### Multivariate Taylor Series

$$f(\mathbf{x} + \mathbf{h}) = \sum_{k=0}^{\infty} \frac{1}{k!} \sum_{|\boldsymbol{\alpha}|=k} \frac{k!}{\alpha_1!\cdots\alpha_n!} \frac{\partial^k f}{\partial x_1^{\alpha_1}\cdots\partial x_n^{\alpha_n}}(\mathbf{x}) h_1^{\alpha_1}\cdots h_n^{\alpha_n}$$

where $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_n)$ is multi-index and $|\boldsymbol{\alpha}| = \sum \alpha_i$.

## Unconstrained Optimization

### First-Order Conditions

For local extremum of $f: \mathbb{R}^n \to \mathbb{R}$:

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

**Critical points**: Points where gradient vanishes.

### Second-Order Conditions

**Local minimum**: If $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $\mathbf{H}_f(\mathbf{x}^*)$ positive definite, then $\mathbf{x}^*$ is local minimum.

**Local maximum**: If $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $\mathbf{H}_f(\mathbf{x}^*)$ negative definite, then $\mathbf{x}^*$ is local maximum.

**Saddle point**: If $\mathbf{H}_f(\mathbf{x}^*)$ has both positive and negative eigenvalues.

**Definiteness check**:
- Positive definite: All eigenvalues $> 0$ (or all principal minors $> 0$)
- Negative definite: All eigenvalues $< 0$ (or principal minors alternate sign)
- Indefinite: Eigenvalues have mixed signs

## Constrained Optimization

### Lagrange Multipliers

For problem:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g(\mathbf{x}) = 0$$

**Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$

**First-order conditions**:

$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f(\mathbf{x}) - \lambda \nabla g(\mathbf{x}) = \mathbf{0}$$
$$\frac{\partial \mathcal{L}}{\partial \lambda} = -g(\mathbf{x}) = 0$$

**Geometric interpretation**: At optimum, $\nabla f$ and $\nabla g$ are parallel (normal to constraint surface).

### Multiple Constraints

For $m$ equality constraints $g_i(\mathbf{x}) = 0$:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) - \sum_{i=1}^{m} \lambda_i g_i(\mathbf{x})$$

**KKT conditions**:

$$\nabla f(\mathbf{x}) - \sum_{i=1}^{m} \lambda_i \nabla g_i(\mathbf{x}) = \mathbf{0}$$
$$g_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$

### Inequality Constraints (KKT Conditions)

For problem:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m$$

**Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) - \sum_{i=1}^{m} \lambda_i g_i(\mathbf{x})$$

**KKT conditions**:

1. **Stationarity**: $\nabla f(\mathbf{x}) - \sum_{i=1}^{m} \lambda_i \nabla g_i(\mathbf{x}) = \mathbf{0}$
2. **Primal feasibility**: $g_i(\mathbf{x}) \leq 0$ for all $i$
3. **Dual feasibility**: $\lambda_i \geq 0$ for all $i$
4. **Complementary slackness**: $\lambda_i g_i(\mathbf{x}) = 0$ for all $i$

**Interpretation**: 
- If constraint inactive ($g_i(\mathbf{x}) < 0$), then $\lambda_i = 0$
- If constraint active ($g_i(\mathbf{x}) = 0$), then $\lambda_i \geq 0$ (shadow price)

### Second-Order Conditions

**Bordered Hessian**: For constrained optimization, check definiteness of Hessian restricted to tangent space of constraints.

**Sufficient condition**: If $\mathbf{H}_f(\mathbf{x}^*) - \sum \lambda_i \mathbf{H}_{g_i}(\mathbf{x}^*)$ positive definite on $\{\mathbf{v} : \nabla g_i(\mathbf{x}^*)^T\mathbf{v} = 0\}$, then local minimum.

## Convexity

### Convex Sets

Set $C \subseteq \mathbb{R}^n$ is convex if:

$$\lambda\mathbf{x} + (1-\lambda)\mathbf{y} \in C$$

for all $\mathbf{x}, \mathbf{y} \in C$ and $\lambda \in [0,1]$.

### Convex Functions

Function $f: C \to \mathbb{R}$ is convex if:

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

for all $\mathbf{x}, \mathbf{y} \in C$ and $\lambda \in [0,1]$.

**Strictly convex**: Inequality is strict for $\mathbf{x} \neq \mathbf{y}$ and $\lambda \in (0,1)$.

### Characterizations

**First-order**: $f$ convex if and only if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$$

for all $\mathbf{x}, \mathbf{y} \in C$ (supporting hyperplane).

**Second-order**: If $f$ twice differentiable, $f$ convex if and only if $\mathbf{H}_f(\mathbf{x})$ positive semidefinite for all $\mathbf{x} \in C$.

### Properties

1. **Global minimum**: Local minimum of convex function is global minimum
2. **Uniqueness**: Strictly convex function has at most one minimizer
3. **Sum**: Sum of convex functions is convex
4. **Composition**: If $g$ convex and $h$ convex non-decreasing, then $h \circ g$ convex

## Convex Optimization

### Convex Program

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m$$

where $f$ and $g_i$ are convex.

**Properties**:
- Local minimum is global minimum
- KKT conditions are necessary and sufficient (under Slater's condition)
- Efficient algorithms available (interior point methods, gradient descent)

### Slater's Condition

There exists $\mathbf{x}$ such that $g_i(\mathbf{x}) < 0$ for all $i$ (strict feasibility).

**Strong duality**: Under Slater's condition, optimal values of primal and dual coincide.

## Duality

### Lagrangian Dual

For primal problem:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g_i(\mathbf{x}) \leq 0$$

**Dual function**:

$$g(\boldsymbol{\lambda}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = \inf_{\mathbf{x}} \left[f(\mathbf{x}) - \sum_{i=1}^{m} \lambda_i g_i(\mathbf{x})\right]$$

**Dual problem**:

$$\max_{\boldsymbol{\lambda} \geq \mathbf{0}} g(\boldsymbol{\lambda})$$

### Weak Duality

For any feasible $\mathbf{x}$ and $\boldsymbol{\lambda} \geq \mathbf{0}$:

$$g(\boldsymbol{\lambda}) \leq f(\mathbf{x})$$

**Duality gap**: $f(\mathbf{x}^*) - g(\boldsymbol{\lambda}^*) \geq 0$

### Strong Duality

Under Slater's condition (for convex problems):

$$f(\mathbf{x}^*) = g(\boldsymbol{\lambda}^*)$$

**KKT conditions**: Necessary and sufficient for optimality under strong duality.

## Applications in Quantitative Finance

### Portfolio Optimization

**Mean-variance optimization**:

$$\min_{\mathbf{w}} \mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w} \quad \text{subject to } \mathbf{w}^T\mathbf{1} = 1, \quad \mathbf{w}^T\boldsymbol{\mu} = \mu_p$$

**Lagrangian**:

$$\mathcal{L} = \mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w} - \lambda_1(\mathbf{w}^T\mathbf{1} - 1) - \lambda_2(\mathbf{w}^T\boldsymbol{\mu} - \mu_p)$$

**First-order conditions**:

$$2\boldsymbol{\Sigma}\mathbf{w} - \lambda_1\mathbf{1} - \lambda_2\boldsymbol{\mu} = \mathbf{0}$$

**Solution**:

$$\mathbf{w}^* = \boldsymbol{\Sigma}^{-1}\left(\frac{\lambda_1}{2}\mathbf{1} + \frac{\lambda_2}{2}\boldsymbol{\mu}\right)$$

**Efficient frontier**: Parametric curve of optimal portfolios for different $\mu_p$.

### Calibration Problems

**Parameter estimation**: Minimize distance between model and market prices:

$$\min_{\boldsymbol{\theta}} \sum_{i=1}^{n} (P_i^{\text{model}}(\boldsymbol{\theta}) - P_i^{\text{market}})^2$$

where $\boldsymbol{\theta}$ are model parameters.

**Gradient**: 

$$\nabla_{\boldsymbol{\theta}} = 2\sum_{i=1}^{n} (P_i^{\text{model}}(\boldsymbol{\theta}) - P_i^{\text{market}}) \frac{\partial P_i^{\text{model}}}{\partial \boldsymbol{\theta}}$$

**Hessian**: For Newton's method:

$$\mathbf{H} = 2\sum_{i=1}^{n} \left[\frac{\partial P_i^{\text{model}}}{\partial \boldsymbol{\theta}}\left(\frac{\partial P_i^{\text{model}}}{\partial \boldsymbol{\theta}}\right)^T + (P_i^{\text{model}} - P_i^{\text{market}})\frac{\partial^2 P_i^{\text{model}}}{\partial \boldsymbol{\theta}^2}\right]$$

### Risk Budgeting

**Risk parity**: Equal risk contribution from each asset:

$$\min_{\mathbf{w}} \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \Sigma_{ij} \quad \text{subject to } \sum_{i=1}^{n} w_i = 1, \quad \text{RC}_i = \text{RC}_j \text{ for all } i,j$$

where risk contribution:

$$\text{RC}_i = w_i \frac{\partial \sigma_p}{\partial w_i} = \frac{w_i(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_p}$$

### Maximum Likelihood Estimation

**Log-likelihood**: $\ell(\boldsymbol{\theta}) = \sum_{i=1}^{n} \ln f(X_i; \boldsymbol{\theta})$

**MLE**: $\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta})$

**Score function**: $\mathbf{s}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta})$

**Information matrix**: $\mathbf{I}(\boldsymbol{\theta}) = -E[\mathbf{H}_\ell(\boldsymbol{\theta})]$

**Asymptotic distribution**: $\hat{\boldsymbol{\theta}} \sim \mathcal{N}(\boldsymbol{\theta}_0, \mathbf{I}(\boldsymbol{\theta}_0)^{-1}/n)$

### Utility Maximization

**Expected utility**: $E[U(W_T)]$ where $W_T$ is terminal wealth.

**Optimization**:

$$\max_{\mathbf{w}} E[U(\mathbf{w}^T\mathbf{R})] \quad \text{subject to } \mathbf{w}^T\mathbf{1} = 1$$

**First-order condition**: $E[U'(W_T)\mathbf{R}] = \lambda\mathbf{1}$

**Risk aversion**: $U''(W) < 0$ (concave utility).

**Certainty equivalent**: $CE$ such that $U(CE) = E[U(W_T)]$.

### Option Greeks

**Delta**: $\Delta = \frac{\partial C}{\partial S}$ (sensitivity to underlying)

**Gamma**: $\Gamma = \frac{\partial^2 C}{\partial S^2}$ (convexity)

**Theta**: $\Theta = \frac{\partial C}{\partial t}$ (time decay)

**Vega**: $\mathcal{V} = \frac{\partial C}{\partial \sigma}$ (volatility sensitivity)

**Rho**: $\rho = \frac{\partial C}{\partial r}$ (interest rate sensitivity)

**Hedging**: Use Greeks to construct delta-neutral, gamma-neutral portfolios.

### Stochastic Control

**Hamilton-Jacobi-Bellman equation**: For optimal control problem:

$$\max_{\mathbf{u}} E\left[\int_0^T f(t, \mathbf{X}_t, \mathbf{u}_t) dt + g(\mathbf{X}_T)\right]$$

subject to $d\mathbf{X}_t = \boldsymbol{\mu}(t, \mathbf{X}_t, \mathbf{u}_t) dt + \boldsymbol{\sigma}(t, \mathbf{X}_t, \mathbf{u}_t) d\mathbf{W}_t$.

**HJB equation**:

$$0 = \frac{\partial V}{\partial t} + \max_{\mathbf{u}} \left[f + \boldsymbol{\mu}^T \nabla V + \frac{1}{2}\text{tr}(\boldsymbol{\sigma}\boldsymbol{\sigma}^T \mathbf{H}_V)\right]$$

where $V(t, \mathbf{x})$ is value function.
