# Constrained Optimization: Lagrange Multipliers

## Table of Contents

1. [Introduction](#introduction)
2. [Constrained Optimization Problems](#constrained-optimization-problems)
3. [Lagrange Multipliers](#lagrange-multipliers)
4. [KKT Conditions](#kkt-conditions)
5. [Duality Theory](#duality-theory)
6. [Convex Optimization](#convex-optimization)
7. [Applications in Machine Learning](#applications-in-machine-learning)
8. [Support Vector Machines](#support-vector-machines)
9. [Regularization as Constraint](#regularization-as-constraint)
10. [Key Takeaways](#key-takeaways)

## Introduction

Constrained optimization extends unconstrained optimization to problems where variables must satisfy certain constraints (equalities or inequalities). In machine learning, constraints appear naturally: budget constraints in resource allocation, fairness constraints, and regularization can be viewed as constraints. Lagrange multipliers provide a powerful framework for solving constrained problems, and the Karush-Kuhn-Tucker (KKT) conditions generalize to inequality constraints.

This document covers Lagrange multipliers, KKT conditions, duality theory, and their applications in ML, particularly in support vector machines and regularization.

## Constrained Optimization Problems

### Equality Constraints

**Problem form**:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad h_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$

where $f: \mathbb{R}^n \to \mathbb{R}$ is the objective and $h_i: \mathbb{R}^n \to \mathbb{R}$ are constraint functions.

**Example**: Minimize distance to origin subject to linear constraint:
$$\min_{x,y} x^2 + y^2 \quad \text{s.t.} \quad ax + by = c$$

### Inequality Constraints

**Problem form**:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, p$$

**Example**: Minimize function subject to box constraints:
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{x} \geq \mathbf{0}$$

### Mixed Constraints

**General form**:

$$\min_{\mathbf{x}} f(\mathbf{x})$$
$$\text{subject to} \quad h_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$
$$\quad \quad \quad \quad g_j(\mathbf{x}) \leq 0, \quad j = 1, \ldots, p$$

## Lagrange Multipliers

### Method of Lagrange Multipliers

For equality-constrained problem:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad h(\mathbf{x}) = 0$$

Form the **Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda h(\mathbf{x})$$

where $\lambda$ is the **Lagrange multiplier**.

### Necessary Condition

**Theorem**: If $\mathbf{x}^*$ is a local minimum and $\nabla h(\mathbf{x}^*) \neq \mathbf{0}$ (constraint qualification), then there exists $\lambda^*$ such that:

$$\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \lambda^*) = \nabla f(\mathbf{x}^*) + \lambda^*\nabla h(\mathbf{x}^*) = \mathbf{0}$$
$$\frac{\partial \mathcal{L}}{\partial \lambda}(\mathbf{x}^*, \lambda^*) = h(\mathbf{x}^*) = 0$$

**Geometric interpretation**: At optimum, gradient of objective is parallel to gradient of constraint: $\nabla f(\mathbf{x}^*) = -\lambda^*\nabla h(\mathbf{x}^*)$.

### Multiple Equality Constraints

For constraints $h_i(\mathbf{x}) = 0$, $i = 1, \ldots, m$, Lagrangian:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i h_i(\mathbf{x})$$

**KKT conditions** (for equality constraints):
$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla h_i(\mathbf{x}) = \mathbf{0}$$
$$h_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$

### Example: Least Squares with Constraint

Minimize $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$ subject to $\mathbf{c}^T\mathbf{x} = d$.

Lagrangian: $\mathcal{L} = \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 + \lambda(\mathbf{c}^T\mathbf{x} - d)$

Optimality:
$$2\mathbf{A}^T(\mathbf{A}\mathbf{x} - \mathbf{b}) + \lambda\mathbf{c} = \mathbf{0}$$
$$\mathbf{c}^T\mathbf{x} = d$$

Solving gives constrained solution.

## KKT Conditions

### Karush-Kuhn-Tucker Conditions

For problem with inequality constraints:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, p$$

Form Lagrangian:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_{i=1}^p \mu_i g_i(\mathbf{x})$$

**KKT conditions** (necessary for optimality under constraint qualification):

1. **Stationarity**: $\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*) = \mathbf{0}$
   $$\nabla f(\mathbf{x}^*) + \sum_{i=1}^p \mu_i^* \nabla g_i(\mathbf{x}^*) = \mathbf{0}$$

2. **Primal feasibility**: $g_i(\mathbf{x}^*) \leq 0$ for all $i$

3. **Dual feasibility**: $\mu_i^* \geq 0$ for all $i$

4. **Complementary slackness**: $\mu_i^* g_i(\mathbf{x}^*) = 0$ for all $i$

### Interpretation

- **Stationarity**: Gradient of objective is a nonnegative combination of constraint gradients
- **Complementary slackness**: Either constraint is active ($g_i(\mathbf{x}^*) = 0$) or multiplier is zero ($\mu_i^* = 0$)
- **Dual feasibility**: Multipliers are nonnegative (constraints can only "push" in one direction)

### Constraint Qualification

KKT conditions are necessary under **constraint qualification** (e.g., Linear Independence Constraint Qualification, LICQ): gradients of active constraints are linearly independent.

### Sufficient Conditions

For **convex** problems (convex $f$ and $g_i$), KKT conditions are also **sufficient** for global optimality.

## Duality Theory

### Primal Problem

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, p$$

### Lagrangian Dual Function

**Dual function**:

$$g(\boldsymbol{\mu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\mu}) = \inf_{\mathbf{x}} \left[ f(\mathbf{x}) + \sum_{i=1}^p \mu_i g_i(\mathbf{x}) \right]$$

**Properties**:
- $g(\boldsymbol{\mu})$ is concave (pointwise infimum of affine functions)
- $g(\boldsymbol{\mu}) \leq f(\mathbf{x}^*)$ for any feasible $\mathbf{x}^*$ (weak duality)

### Dual Problem

**Dual problem**:

$$\max_{\boldsymbol{\mu} \geq \mathbf{0}} g(\boldsymbol{\mu})$$

This maximizes the best lower bound on primal objective.

### Weak and Strong Duality

**Weak duality**: $d^* \leq p^*$ where:
- $p^*$: optimal value of primal
- $d^*$: optimal value of dual

**Strong duality**: $d^* = p^*$

**Slater's condition**: For convex problems, if there exists strictly feasible point ($g_i(\mathbf{x}) < 0$ for all $i$), then strong duality holds.

### Saddle Point Interpretation

Optimal $(\mathbf{x}^*, \boldsymbol{\mu}^*)$ is a **saddle point**:

$$\mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}) \leq \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*) \leq \mathcal{L}(\mathbf{x}, \boldsymbol{\mu}^*)$$

Minimizes over $\mathbf{x}$, maximizes over $\boldsymbol{\mu}$.

## Convex Optimization

### Convex Constrained Problem

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, p$$
$$\quad \quad \quad \quad h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, m$$

where $f$ and $g_i$ are convex, $h_j$ are affine.

**Properties**:
- KKT conditions are necessary and sufficient
- Strong duality holds under Slater's condition
- Local minimum is global minimum

### Quadratic Programming

**QP problem**:

$$\min_{\mathbf{x}} \frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x} \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} \leq \mathbf{b}, \quad \mathbf{C}\mathbf{x} = \mathbf{d}$$

where $\mathbf{Q} \succeq 0$ (convex QP).

**KKT system** (for equality constraints):
$$\begin{pmatrix} \mathbf{Q} & \mathbf{C}^T \\ \mathbf{C} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{x} \\ \boldsymbol{\lambda} \end{pmatrix} = \begin{pmatrix} -\mathbf{c} \\ \mathbf{d} \end{pmatrix}$$

### Semidefinite Programming

**SDP problem**:

$$\min_{\mathbf{X}} \text{tr}(\mathbf{C}\mathbf{X}) \quad \text{s.t.} \quad \text{tr}(\mathbf{A}_i\mathbf{X}) = b_i, \quad \mathbf{X} \succeq 0$$

where optimization is over positive semidefinite matrices.

## Applications in Machine Learning

### Regularization as Constraint

**L2 regularization** can be viewed as constraint:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) + \lambda\|\boldsymbol{\theta}\|^2$$

Equivalent to:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) \quad \text{s.t.} \quad \|\boldsymbol{\theta}\|^2 \leq t$$

where $t$ depends on $\lambda$. Lagrange multiplier $\lambda$ trades off loss vs. regularization.

**L1 regularization**:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) + \lambda\|\boldsymbol{\theta}\|_1$$

Equivalent to:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) \quad \text{s.t.} \quad \|\boldsymbol{\theta}\|_1 \leq t$$

### Fairness Constraints

**Equalized odds**:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) \quad \text{s.t.} \quad P(\hat{Y} = 1 | Y = y, A = a) = P(\hat{Y} = 1 | Y = y, A = a')$$

for all $y, a, a'$, where $A$ is protected attribute.

### Budget Constraints

**Feature selection** with budget:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) \quad \text{s.t.} \quad \|\boldsymbol{\theta}\|_0 \leq k$$

(Number of nonzero parameters $\leq k$). Often relaxed to L1 constraint.

## Support Vector Machines

### Hard Margin SVM

**Primal**:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

**Lagrangian**:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]$$

**KKT conditions**:
- $\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$
- $\sum_{i=1}^n \alpha_i y_i = 0$
- $\alpha_i \geq 0$
- $\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$

**Support vectors**: Points with $\alpha_i > 0$ (on margin).

### Soft Margin SVM

**Primal**:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i$$
$$\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Dual**:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$
$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

**Kernel trick**: Replace $\mathbf{x}_i^T\mathbf{x}_j$ with $k(\mathbf{x}_i, \mathbf{x}_j)$.

### SVM Derivation via KKT

From KKT stationarity:
$$\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

Substitute into Lagrangian to get dual problem. Complementary slackness shows support vectors are points on or inside margin.

## Regularization as Constraint

### Equivalent Formulations

**Tikhonov regularization**:

$$\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|^2$$

Equivalent to:

$$\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2 \quad \text{s.t.} \quad \|\mathbf{x}\|^2 \leq t$$

**Lagrange multiplier interpretation**: $\lambda$ is the "price" of violating the constraint. Larger $\lambda$ means tighter constraint.

### Elastic Net

Combines L1 and L2:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) + \lambda_1\|\boldsymbol{\theta}\|_1 + \lambda_2\|\boldsymbol{\theta}\|^2$$

Can be viewed as:

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) \quad \text{s.t.} \quad \|\boldsymbol{\theta}\|_1 \leq t_1, \quad \|\boldsymbol{\theta}\|^2 \leq t_2$$

### Group Lasso

$$\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) + \lambda \sum_{g=1}^G \|\boldsymbol{\theta}_g\|_2$$

Promotes group sparsity (entire groups set to zero).

## Key Takeaways

1. **Lagrange multipliers** convert constrained problems to unconstrained optimization of Lagrangian function.

2. **KKT conditions** are necessary (and sufficient for convex problems) optimality conditions for constrained optimization.

3. **Complementary slackness** means inactive constraints have zero multipliers, active constraints can have positive multipliers.

4. **Duality theory** provides lower bounds on optimal value and alternative solution methods via dual problem.

5. **Strong duality** holds for convex problems under Slater's condition, enabling dual-based algorithms.

6. **Support Vector Machines** are naturally formulated as constrained optimization, with dual formulation enabling kernel methods.

7. **Regularization** can be viewed as constraint, with regularization parameter as Lagrange multiplier.

8. **Fairness constraints** in ML can be formulated as equality or inequality constraints on model behavior.

9. **Convex constrained optimization** is tractable: KKT conditions are sufficient, local optima are global.

10. **Constraint qualification** ensures KKT conditions are necessary; LICQ is common sufficient condition.
