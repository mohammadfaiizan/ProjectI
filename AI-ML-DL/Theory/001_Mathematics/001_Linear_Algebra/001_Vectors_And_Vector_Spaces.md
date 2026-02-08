# Vectors and Vector Spaces

## Table of Contents

1. [Introduction](#introduction)
2. [Vector Fundamentals](#vector-fundamentals)
3. [Vector Operations](#vector-operations)
4. [Linear Combinations and Span](#linear-combinations-and-span)
5. [Linear Independence](#linear-independence)
6. [Basis and Dimension](#basis-and-dimension)
7. [Subspaces](#subspaces)
8. [Inner Products and Norms](#inner-products-and-norms)
9. [Orthogonality](#orthogonality)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Vector spaces form the foundational mathematical structure underlying virtually all machine learning and deep learning algorithms. From representing data points as feature vectors to understanding the geometry of neural network transformations, vector space theory provides the language and tools necessary for rigorous analysis of ML systems.

This document covers the fundamental concepts of vectors and vector spaces, including operations, linear combinations, basis, dimension, and subspaces. We develop these concepts with formal mathematical rigor while maintaining focus on their applications in machine learning contexts.

## Vector Fundamentals

### Definition of a Vector

A **vector** is an ordered collection of numbers, typically written as a column:

$$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} \in \mathbb{R}^n$$

where $\mathbb{R}^n$ denotes the $n$-dimensional real vector space. Each component $v_i$ is a real number.

### Vector Space Axioms

A **vector space** $V$ over a field $\mathbb{F}$ (typically $\mathbb{R}$ or $\mathbb{C}$) is a set equipped with two operations:

1. **Vector Addition**: For $\mathbf{u}, \mathbf{v} \in V$, $\mathbf{u} + \mathbf{v} \in V$
2. **Scalar Multiplication**: For $\alpha \in \mathbb{F}$ and $\mathbf{v} \in V$, $\alpha\mathbf{v} \in V$

These operations must satisfy the following axioms:

- **Commutativity**: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- **Associativity**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- **Additive Identity**: There exists $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$
- **Additive Inverse**: For each $\mathbf{v} \in V$, there exists $-\mathbf{v}$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$
- **Multiplicative Identity**: $1 \cdot \mathbf{v} = \mathbf{v}$
- **Distributivity**: $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$ and $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$

### Standard Vector Spaces

The most common vector space in ML is $\mathbb{R}^n$, the space of $n$-tuples of real numbers. Other important examples include:

- **Function Spaces**: The set of all functions $f: \mathbb{R} \to \mathbb{R}$ forms a vector space
- **Polynomial Spaces**: Polynomials of degree at most $n$ form a vector space
- **Matrix Spaces**: $m \times n$ matrices form a vector space $\mathbb{R}^{m \times n}$

## Vector Operations

### Vector Addition

For vectors $\mathbf{u} = (u_1, u_2, \ldots, u_n)^T$ and $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

**Properties**:
- Commutative: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- Associative: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- Zero vector: $\mathbf{v} + \mathbf{0} = \mathbf{v}$ where $\mathbf{0} = (0, 0, \ldots, 0)^T$

### Scalar Multiplication

For scalar $\alpha \in \mathbb{R}$ and vector $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$:

$$\alpha\mathbf{v} = \begin{pmatrix} \alpha v_1 \\ \alpha v_2 \\ \vdots \\ \alpha v_n \end{pmatrix}$$

**Geometric Interpretation**: Scalar multiplication scales the vector by factor $\alpha$. If $\alpha < 0$, the direction is reversed.

### Dot Product (Inner Product)

The **dot product** (or **inner product**) of two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ is:

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T\mathbf{v} = \sum_{i=1}^n u_i v_i$$

**Properties**:
- Commutative: $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
- Distributive: $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$
- Scalar multiplication: $(\alpha\mathbf{u}) \cdot \mathbf{v} = \alpha(\mathbf{u} \cdot \mathbf{v})$
- Positive definite: $\mathbf{v} \cdot \mathbf{v} \geq 0$ with equality iff $\mathbf{v} = \mathbf{0}$

**Geometric Interpretation**: $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ where $\theta$ is the angle between vectors.

### Vector Norm

The **Euclidean norm** (or $L_2$ norm) of a vector $\mathbf{v} \in \mathbb{R}^n$ is:

$$\|\mathbf{v}\|_2 = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{\sum_{i=1}^n v_i^2}$$

Other important norms:
- **$L_1$ norm**: $\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$
- **$L_\infty$ norm**: $\|\mathbf{v}\|_\infty = \max_i |v_i|$
- **$L_p$ norm**: $\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p}$

## Linear Combinations and Span

### Linear Combination

A **linear combination** of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in V$ is an expression of the form:

$$\alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \cdots + \alpha_k\mathbf{v}_k$$

where $\alpha_1, \alpha_2, \ldots, \alpha_k$ are scalars.

### Span

The **span** of a set of vectors $S = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is the set of all possible linear combinations:

$$\text{span}(S) = \left\{ \sum_{i=1}^k \alpha_i\mathbf{v}_i : \alpha_i \in \mathbb{R} \right\}$$

**Geometric Interpretation**: The span is the smallest subspace containing all vectors in $S$. In $\mathbb{R}^2$, the span of a single nonzero vector is a line through the origin. The span of two linearly independent vectors is the entire plane.

### Theorem: Span is a Subspace

**Theorem**: For any set $S \subseteq V$, $\text{span}(S)$ is a subspace of $V$.

**Proof**: 
1. **Closure under addition**: If $\mathbf{u}, \mathbf{v} \in \text{span}(S)$, then $\mathbf{u} = \sum \alpha_i\mathbf{v}_i$ and $\mathbf{v} = \sum \beta_i\mathbf{v}_i$ for some scalars. Then $\mathbf{u} + \mathbf{v} = \sum (\alpha_i + \beta_i)\mathbf{v}_i \in \text{span}(S)$.

2. **Closure under scalar multiplication**: If $\mathbf{u} \in \text{span}(S)$ and $\alpha$ is a scalar, then $\alpha\mathbf{u} = \sum (\alpha\alpha_i)\mathbf{v}_i \in \text{span}(S)$.

3. **Contains zero vector**: $0 \cdot \mathbf{v}_1 + \cdots + 0 \cdot \mathbf{v}_k = \mathbf{0} \in \text{span}(S)$.

## Linear Independence

### Definition

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is **linearly independent** if the only solution to:

$$\alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \cdots + \alpha_k\mathbf{v}_k = \mathbf{0}$$

is $\alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$.

If there exist nonzero scalars such that the above equation holds, the vectors are **linearly dependent**.

### Testing Linear Independence

For vectors in $\mathbb{R}^n$, form the matrix $\mathbf{A} = [\mathbf{v}_1 | \mathbf{v}_2 | \cdots | \mathbf{v}_k]$ with columns as the vectors. The vectors are linearly independent if and only if the only solution to $\mathbf{A}\boldsymbol{\alpha} = \mathbf{0}$ is $\boldsymbol{\alpha} = \mathbf{0}$, which occurs when $\text{rank}(\mathbf{A}) = k$.

### Properties

- A set containing the zero vector is always linearly dependent
- A set with more vectors than dimensions ($k > n$ in $\mathbb{R}^n$) is always linearly dependent
- If a set is linearly independent, any subset is also linearly independent
- If a set is linearly dependent, any superset is also linearly dependent

## Basis and Dimension

### Basis

A **basis** for a vector space $V$ is a linearly independent set $B = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ such that $\text{span}(B) = V$.

**Standard Basis**: For $\mathbb{R}^n$, the **standard basis** is:

$$\mathbf{e}_1 = \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad \mathbf{e}_2 = \begin{pmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{pmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{pmatrix}$$

### Dimension

The **dimension** of a vector space $V$, denoted $\dim(V)$, is the number of vectors in any basis for $V$.

**Theorem**: All bases for a finite-dimensional vector space have the same number of elements.

**Proof Sketch**: Suppose $B_1$ and $B_2$ are two bases with $|B_1| = k$ and $|B_2| = m$. Since $B_1$ spans $V$ and $B_2$ is linearly independent, we have $m \leq k$. Similarly, $k \leq m$. Therefore $k = m$.

### Coordinate Representation

Given a basis $B = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$ for $V$, every vector $\mathbf{u} \in V$ can be uniquely written as:

$$\mathbf{u} = \alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \cdots + \alpha_n\mathbf{v}_n$$

The scalars $(\alpha_1, \alpha_2, \ldots, \alpha_n)$ are called the **coordinates** of $\mathbf{u}$ with respect to basis $B$.

## Subspaces

### Definition

A **subspace** $W$ of a vector space $V$ is a subset $W \subseteq V$ that is itself a vector space under the same operations.

**Subspace Test**: A subset $W \subseteq V$ is a subspace if and only if:
1. $\mathbf{0} \in W$
2. For all $\mathbf{u}, \mathbf{v} \in W$, $\mathbf{u} + \mathbf{v} \in W$
3. For all $\mathbf{u} \in W$ and $\alpha \in \mathbb{R}$, $\alpha\mathbf{u} \in W$

### Examples of Subspaces

1. **Trivial subspaces**: $\{\mathbf{0}\}$ and $V$ itself
2. **Span of vectors**: $\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k)$ for any vectors
3. **Solution space**: The set of solutions to $\mathbf{A}\mathbf{x} = \mathbf{0}$ (null space)
4. **Column space**: The span of columns of a matrix
5. **Row space**: The span of rows of a matrix

### Dimension of Subspaces

**Theorem**: If $W$ is a subspace of finite-dimensional $V$, then:
- $\dim(W) \leq \dim(V)$
- $\dim(W) = \dim(V)$ if and only if $W = V$

## Inner Products and Norms

### Inner Product Spaces

An **inner product** on a vector space $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:

1. **Symmetry**: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. **Linearity**: $\langle \alpha\mathbf{u} + \beta\mathbf{v}, \mathbf{w} \rangle = \alpha\langle \mathbf{u}, \mathbf{w} \rangle + \beta\langle \mathbf{v}, \mathbf{w} \rangle$
3. **Positive definiteness**: $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ with equality iff $\mathbf{v} = \mathbf{0}$

A vector space equipped with an inner product is called an **inner product space**.

### Induced Norm

Every inner product induces a norm:

$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

This norm satisfies the **Cauchy-Schwarz inequality**:

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\|\|\mathbf{v}\|$$

### Weighted Inner Products

For a positive definite matrix $\mathbf{A}$, we can define a weighted inner product:

$$\langle \mathbf{u}, \mathbf{v} \rangle_{\mathbf{A}} = \mathbf{u}^T\mathbf{A}\mathbf{v}$$

This is useful in optimization and kernel methods.

## Orthogonality

### Orthogonal Vectors

Two vectors $\mathbf{u}, \mathbf{v}$ are **orthogonal** if $\mathbf{u} \cdot \mathbf{v} = 0$, denoted $\mathbf{u} \perp \mathbf{v}$.

### Orthogonal Set

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is **orthogonal** if $\mathbf{v}_i \cdot \mathbf{v}_j = 0$ for all $i \neq j$.

### Orthonormal Set

An orthogonal set where each vector has unit norm ($\|\mathbf{v}_i\| = 1$) is called **orthonormal**.

**Properties**:
- Orthogonal sets are automatically linearly independent
- Orthonormal bases simplify coordinate computations
- For orthonormal basis $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$, coordinates are: $\alpha_i = \mathbf{v} \cdot \mathbf{e}_i$

### Orthogonal Complement

For a subspace $W \subseteq V$, the **orthogonal complement** is:

$$W^\perp = \{\mathbf{v} \in V : \mathbf{v} \cdot \mathbf{w} = 0 \text{ for all } \mathbf{w} \in W\}$$

**Theorem**: For any subspace $W$ of finite-dimensional $V$:
- $W^\perp$ is a subspace
- $V = W \oplus W^\perp$ (direct sum)
- $\dim(W) + \dim(W^\perp) = \dim(V)$

## Machine Learning Applications

### Feature Vectors

In supervised learning, each data point is represented as a **feature vector** $\mathbf{x} \in \mathbb{R}^d$ where $d$ is the number of features. The entire dataset forms a collection of vectors in $\mathbb{R}^d$.

**Example**: In image classification, a $28 \times 28$ grayscale image becomes a vector in $\mathbb{R}^{784}$ by flattening the pixel values.

### Word Embeddings

Word embeddings map words to dense vectors in $\mathbb{R}^d$ (typically $d = 100-300$). Words with similar meanings have vectors that are close in the embedding space.

**Properties**:
- Semantic relationships: $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$
- Distance in embedding space reflects semantic similarity
- Learned embeddings capture linguistic structure

### Linear Models

Linear regression and classification models use vector operations:

$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$

where $\mathbf{w} \in \mathbb{R}^d$ is the weight vector and $b$ is the bias. The prediction is computed via dot product between weight vector and feature vector.

### Support Vector Machines

SVMs find the optimal separating hyperplane by maximizing the margin, which involves computing distances between vectors and hyperplanes using inner products and norms.

### Neural Network Layers

A fully connected layer in a neural network computes:

$$\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

where $\mathbf{W}$ is a weight matrix, $\mathbf{x}$ is the input vector, and $\sigma$ is an activation function. This is fundamentally a linear transformation followed by a nonlinearity.

### Principal Component Analysis

PCA finds orthogonal directions of maximum variance in data. The principal components form an orthonormal basis for a lower-dimensional subspace that preserves most information.

### Cosine Similarity

For comparing vectors (e.g., document similarity, recommendation systems):

$$\text{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \cos\theta$$

This measures the angle between vectors, independent of their magnitudes.

## Key Takeaways

1. **Vector spaces** provide the mathematical foundation for representing data and transformations in ML systems.

2. **Linear combinations and span** allow us to understand which vectors can be represented using a given set of basis vectors.

3. **Linear independence** determines the minimal set of vectors needed to represent a space, directly related to the concept of basis.

4. **Basis and dimension** characterize the structure of vector spaces. The dimension equals the number of features or the intrinsic dimensionality of data.

5. **Subspaces** appear naturally in ML as solution spaces, column spaces of data matrices, and lower-dimensional representations.

6. **Inner products and norms** enable measurement of similarity, distance, and angles between vectors, crucial for clustering, nearest neighbor methods, and optimization.

7. **Orthogonality** simplifies computations and appears in PCA, SVD, and many optimization algorithms.

8. **Feature vectors** are the primary data representation in ML, where each component corresponds to a feature or measurement.

9. **Geometric intuition** from vector spaces helps understand high-dimensional ML problems, even when visualization is impossible.

10. **Dimensionality** plays a crucial role: high-dimensional spaces suffer from the curse of dimensionality, while low-dimensional representations enable efficient learning and generalization.
