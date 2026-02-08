# Linear Transformations and Mappings

## Table of Contents

1. [Introduction](#introduction)
2. [Definition and Properties](#definition-and-properties)
3. [Matrix Representation](#matrix-representation)
4. [Kernel and Image](#kernel-and-image)
5. [Composition and Inverses](#composition-and-inverses)
6. [Change of Basis](#change-of-basis)
7. [Similarity Transforms](#similarity-transforms)
8. [Isomorphisms](#isomorphisms)
9. [Projections](#projections)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Linear transformations are fundamental mathematical objects that describe how vectors are mapped from one vector space to another while preserving the structure of vector addition and scalar multiplication. In machine learning and deep learning, every layer of a neural network implements a linear transformation (followed by a nonlinearity), making understanding of these mappings essential for analyzing model behavior.

This document covers linear transformations, their matrix representations, kernel and image spaces, change of basis, and similarity transforms. We develop these concepts with formal rigor while emphasizing their role in neural networks and other ML systems.

## Definition and Properties

### Definition of Linear Transformation

A function $T: V \to W$ between vector spaces $V$ and $W$ is a **linear transformation** (or **linear map**) if for all $\mathbf{u}, \mathbf{v} \in V$ and scalars $\alpha, \beta$:

$$T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$$

This property is called **linearity** and combines two requirements:
1. **Additivity**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity**: $T(\alpha\mathbf{u}) = \alpha T(\mathbf{u})$

### Examples of Linear Transformations

**Rotation in $\mathbb{R}^2$**: Rotating vectors by angle $\theta$:
$$T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix}$$

**Scaling**: Multiplying all components by a constant:
$$T(\mathbf{v}) = \alpha\mathbf{v}$$

**Projection**: Projecting onto a subspace

**Differentiation**: On the space of polynomials, $D(p(x)) = p'(x)$

**Integration**: On function spaces, $I(f)(x) = \int_0^x f(t)dt$

### Properties of Linear Transformations

**Theorem**: If $T: V \to W$ is linear, then:
1. $T(\mathbf{0}_V) = \mathbf{0}_W$ (maps zero to zero)
2. $T(-\mathbf{v}) = -T(\mathbf{v})$ (preserves negatives)
3. $T(\alpha_1\mathbf{v}_1 + \cdots + \alpha_k\mathbf{v}_k) = \alpha_1 T(\mathbf{v}_1) + \cdots + \alpha_k T(\mathbf{v}_k)$ (preserves linear combinations)

**Proof**: 
1. $T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}$
2. $T(-\mathbf{v}) = T((-1)\mathbf{v}) = (-1)T(\mathbf{v}) = -T(\mathbf{v})$
3. Follows by induction from linearity

## Matrix Representation

### Standard Matrix Representation

For linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$, there exists a unique matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ such that:

$$T(\mathbf{x}) = \mathbf{A}\mathbf{x}$$

**Construction**: The matrix $\mathbf{A}$ has columns equal to $T(\mathbf{e}_j)$ where $\mathbf{e}_j$ are standard basis vectors:

$$\mathbf{A} = [T(\mathbf{e}_1) | T(\mathbf{e}_2) | \cdots | T(\mathbf{e}_n)]$$

**Theorem**: Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ has a unique matrix representation.

**Proof**: For any $\mathbf{x} = \sum_{j=1}^n x_j\mathbf{e}_j$:
$$T(\mathbf{x}) = T\left(\sum_{j=1}^n x_j\mathbf{e}_j\right) = \sum_{j=1}^n x_j T(\mathbf{e}_j) = \mathbf{A}\mathbf{x}$$

### Matrix Representation with Respect to Bases

For linear transformation $T: V \to W$ with bases $B_V = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ for $V$ and $B_W = \{\mathbf{w}_1, \ldots, \mathbf{w}_m\}$ for $W$, the matrix representation $[T]_{B_W}^{B_V}$ has columns:

$$[T]_{B_W}^{B_V} = [[T(\mathbf{v}_1)]_{B_W} | \cdots | [T(\mathbf{v}_n)]_{B_W}]$$

where $[T(\mathbf{v}_j)]_{B_W}$ is the coordinate vector of $T(\mathbf{v}_j)$ with respect to basis $B_W$.

### Coordinate Transformation

If $[\mathbf{v}]_B$ denotes coordinates of $\mathbf{v}$ in basis $B$, then:

$$[T(\mathbf{v})]_{B_W} = [T]_{B_W}^{B_V}[\mathbf{v}]_{B_V}$$

## Kernel and Image

### Kernel (Null Space)

The **kernel** (or **null space**) of linear transformation $T: V \to W$ is:

$$\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}_W\}$$

**Properties**:
- $\ker(T)$ is a subspace of $V$
- $T$ is injective (one-to-one) if and only if $\ker(T) = \{\mathbf{0}\}$

**Proof**: 
- Contains zero: $T(\mathbf{0}) = \mathbf{0}$, so $\mathbf{0} \in \ker(T)$
- Closed under addition: If $T(\mathbf{u}) = T(\mathbf{v}) = \mathbf{0}$, then $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) = \mathbf{0}$
- Closed under scalar multiplication: If $T(\mathbf{v}) = \mathbf{0}$, then $T(\alpha\mathbf{v}) = \alpha T(\mathbf{v}) = \mathbf{0}$

### Image (Range)

The **image** (or **range**) of $T: V \to W$ is:

$$\text{im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\} = T(V)$$

**Properties**:
- $\text{im}(T)$ is a subspace of $W$
- $T$ is surjective (onto) if and only if $\text{im}(T) = W$

**Proof**: Similar to kernel, using linearity to show closure properties.

### Rank-Nullity Theorem for Linear Transformations

**Theorem**: For linear transformation $T: V \to W$ with $V$ finite-dimensional:

$$\dim(\ker(T)) + \dim(\text{im}(T)) = \dim(V)$$

**Proof Sketch**: 
1. Let $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ be a basis for $\ker(T)$
2. Extend to basis $\{\mathbf{v}_1, \ldots, \mathbf{v}_k, \mathbf{v}_{k+1}, \ldots, \mathbf{v}_n\}$ for $V$
3. Show $\{T(\mathbf{v}_{k+1}), \ldots, T(\mathbf{v}_n)\}$ is a basis for $\text{im}(T)$
4. Therefore $\dim(\text{im}(T)) = n - k = \dim(V) - \dim(\ker(T))$

### Rank and Nullity

- **Rank**: $\text{rank}(T) = \dim(\text{im}(T))$
- **Nullity**: $\text{nullity}(T) = \dim(\ker(T))$

For matrix representation $\mathbf{A}$, $\text{rank}(T) = \text{rank}(\mathbf{A})$ and $\text{nullity}(T) = \text{nullity}(\mathbf{A})$.

## Composition and Inverses

### Composition

For linear transformations $T: U \to V$ and $S: V \to W$, the **composition** $S \circ T: U \to W$ is:

$$(S \circ T)(\mathbf{u}) = S(T(\mathbf{u}))$$

**Properties**:
- Composition is linear
- Matrix representation: $[S \circ T] = [S][T]$ (matrix multiplication)

**Proof**: 
$$(S \circ T)(\alpha\mathbf{u} + \beta\mathbf{v}) = S(T(\alpha\mathbf{u} + \beta\mathbf{v})) = S(\alpha T(\mathbf{u}) + \beta T(\mathbf{v})) = \alpha S(T(\mathbf{u})) + \beta S(T(\mathbf{v}))$$

### Inverse Transformation

Linear transformation $T: V \to W$ is **invertible** if there exists $T^{-1}: W \to V$ such that:

$$T^{-1} \circ T = \text{id}_V \quad \text{and} \quad T \circ T^{-1} = \text{id}_W$$

**Theorem**: $T$ is invertible if and only if:
1. $T$ is bijective (one-to-one and onto)
2. $\ker(T) = \{\mathbf{0}\}$ and $\text{im}(T) = W$
3. $\dim(V) = \dim(W)$ and $\text{rank}(T) = \dim(V)$

**Matrix Representation**: If $T(\mathbf{x}) = \mathbf{A}\mathbf{x}$, then $T^{-1}(\mathbf{y}) = \mathbf{A}^{-1}\mathbf{y}$.

## Change of Basis

### Change of Basis Matrix

Given two bases $B = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ and $B' = \{\mathbf{v}'_1, \ldots, \mathbf{v}'_n\}$ for vector space $V$, the **change of basis matrix** from $B$ to $B'$ is:

$$\mathbf{P}_{B \to B'} = [[\mathbf{v}_1]_{B'} | \cdots | [\mathbf{v}_n]_{B'}]$$

This matrix converts coordinates: $[\mathbf{x}]_{B'} = \mathbf{P}_{B \to B'}[\mathbf{x}]_B$.

**Properties**:
- $\mathbf{P}_{B \to B'}$ is invertible
- $\mathbf{P}_{B' \to B} = \mathbf{P}_{B \to B'}^{-1}$
- $\mathbf{P}_{B \to B''} = \mathbf{P}_{B' \to B''}\mathbf{P}_{B \to B'}$

### Transformation Matrix Under Change of Basis

For linear transformation $T: V \to V$ with matrix $[T]_B$ in basis $B$ and $[T]_{B'}$ in basis $B'$:

$$[T]_{B'} = \mathbf{P}_{B \to B'}[T]_B\mathbf{P}_{B' \to B} = \mathbf{P}_{B \to B'}[T]_B\mathbf{P}_{B \to B'}^{-1}$$

This is a **similarity transformation**.

## Similarity Transforms

### Similar Matrices

Matrices $\mathbf{A}$ and $\mathbf{B}$ are **similar** if there exists invertible $\mathbf{P}$ such that:

$$\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$$

**Interpretation**: Similar matrices represent the same linear transformation in different bases.

### Properties of Similar Matrices

- **Same eigenvalues**: $\det(\mathbf{A} - \lambda\mathbf{I}) = \det(\mathbf{B} - \lambda\mathbf{I})$
- **Same trace**: $\text{tr}(\mathbf{A}) = \text{tr}(\mathbf{B})$
- **Same determinant**: $\det(\mathbf{A}) = \det(\mathbf{B})$
- **Same rank**: $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{B})$

**Proof**: For eigenvalues:
$$\det(\mathbf{B} - \lambda\mathbf{I}) = \det(\mathbf{P}^{-1}\mathbf{A}\mathbf{P} - \lambda\mathbf{I}) = \det(\mathbf{P}^{-1}(\mathbf{A} - \lambda\mathbf{I})\mathbf{P}) = \det(\mathbf{A} - \lambda\mathbf{I})$$

### Diagonalization

A matrix $\mathbf{A}$ is **diagonalizable** if it is similar to a diagonal matrix:

$$\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$$

where $\boldsymbol{\Lambda}$ contains eigenvalues and columns of $\mathbf{P}$ are eigenvectors.

**Condition**: $\mathbf{A}$ is diagonalizable if and only if it has $n$ linearly independent eigenvectors.

## Isomorphisms

### Definition

A linear transformation $T: V \to W$ is an **isomorphism** if it is bijective (one-to-one and onto).

**Properties**:
- $T$ is invertible
- $\dim(V) = \dim(W)$
- $V$ and $W$ are **isomorphic** (essentially the same structure)

### Isomorphism Theorem

**Theorem**: Two finite-dimensional vector spaces are isomorphic if and only if they have the same dimension.

**Proof**: 
- If $\dim(V) = \dim(W) = n$, choose bases and define $T$ mapping basis to basis
- If isomorphic, then bijective, so $\dim(V) = \dim(W)$

**Corollary**: Every $n$-dimensional vector space over $\mathbb{F}$ is isomorphic to $\mathbb{F}^n$.

## Projections

### Orthogonal Projection

For subspace $W \subseteq V$ with orthonormal basis $\{\mathbf{w}_1, \ldots, \mathbf{w}_k\}$, the **orthogonal projection** onto $W$ is:

$$P_W(\mathbf{v}) = \sum_{i=1}^k (\mathbf{v} \cdot \mathbf{w}_i)\mathbf{w}_i$$

**Matrix Form**: If columns of $\mathbf{Q}$ form orthonormal basis for $W$:

$$P_W(\mathbf{v}) = \mathbf{Q}\mathbf{Q}^T\mathbf{v}$$

**Properties**:
- $P_W$ is linear
- $P_W^2 = P_W$ (idempotent)
- $P_W$ is symmetric: $P_W^T = P_W$ (for orthogonal projection)
- $\mathbf{v} - P_W(\mathbf{v}) \in W^\perp$

### Projection onto Column Space

For matrix $\mathbf{A}$ with full column rank, projection onto $\text{col}(\mathbf{A})$:

$$P_{\text{col}(\mathbf{A})}(\mathbf{b}) = \mathbf{A}(\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$$

This appears in least squares problems: $\mathbf{A}\mathbf{x} = \mathbf{b}$ has solution $\mathbf{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$ when $\mathbf{b} \in \text{col}(\mathbf{A})$.

## Machine Learning Applications

### Neural Network Layers as Linear Transformations

Each fully connected layer in a neural network implements:

$$\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

where:
- $\mathbf{W}\mathbf{x}$ is a linear transformation
- $\mathbf{b}$ is a translation (affine, not linear)
- $\sigma$ is a nonlinear activation function

**Analysis**:
- The kernel of $\mathbf{W}$ contains inputs that produce zero output (before activation)
- The image of $\mathbf{W}$ is the subspace of possible pre-activation outputs
- Rank of $\mathbf{W}$ determines the dimensionality of the transformation

### Convolutional Layers

Convolutional layers implement linear transformations with special structure (sparse, weight-shared):

$$\mathbf{y} = \text{conv}(\mathbf{x}, \mathbf{k})$$

This is linear: $\text{conv}(\alpha\mathbf{x}_1 + \beta\mathbf{x}_2, \mathbf{k}) = \alpha\text{conv}(\mathbf{x}_1, \mathbf{k}) + \beta\text{conv}(\mathbf{x}_2, \mathbf{k})$.

### Attention Mechanisms

Self-attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The matrix multiplication $\mathbf{Q}\mathbf{K}^T$ and $\mathbf{A}\mathbf{V}$ are linear transformations, though the softmax introduces nonlinearity.

### Principal Component Analysis

PCA finds linear transformation that projects data onto principal directions:

$$\mathbf{y} = \mathbf{W}^T\mathbf{x}$$

where columns of $\mathbf{W}$ are principal components. This is a linear dimensionality reduction.

### Linear Regression

Linear regression learns transformation:

$$\hat{y} = \boldsymbol{\theta}^T\mathbf{x}$$

This is a linear map from feature space to prediction space.

### Feature Transformations

Many feature engineering techniques are linear transformations:
- **Polynomial features**: $\phi(\mathbf{x}) = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]^T$ (nonlinear in input, but linear in parameters)
- **Basis expansion**: Representing functions as linear combinations of basis functions

### Change of Basis in Embeddings

Word embeddings can be viewed in different coordinate systems. Changing basis corresponds to:

$$\mathbf{e}' = \mathbf{P}\mathbf{e}$$

where $\mathbf{P}$ is the change of basis matrix. This preserves distances and relationships if $\mathbf{P}$ is orthogonal.

### Kernel Methods

Kernel methods implicitly work in high-dimensional feature spaces via feature map $\phi: \mathcal{X} \to \mathcal{H}$. The feature map is typically nonlinear, but the learned function is linear in the feature space:

$$f(\mathbf{x}) = \sum_{i=1}^n \alpha_i k(\mathbf{x}_i, \mathbf{x}) = \sum_{i=1}^n \alpha_i \phi(\mathbf{x}_i)^T\phi(\mathbf{x})$$

### Matrix Factorization

Low-rank matrix factorization $\mathbf{A} \approx \mathbf{U}\mathbf{V}^T$ represents data transformation:
- $\mathbf{V}^T$ maps from original feature space to latent space
- $\mathbf{U}$ maps from latent space to output space

### Autoencoders

Autoencoders learn:
- **Encoder**: Linear (or nonlinear) transformation $\mathbf{z} = f(\mathbf{x})$ mapping to latent space
- **Decoder**: Transformation $\hat{\mathbf{x}} = g(\mathbf{z})$ mapping back to input space

Even with nonlinear activations, the weight matrices implement linear transformations that are composed with nonlinearities.

## Key Takeaways

1. **Linear transformations** preserve vector space structure (addition and scalar multiplication), making them fundamental building blocks.

2. **Matrix representation** provides concrete computational framework: every linear transformation between finite-dimensional spaces has a matrix representation.

3. **Kernel and image** characterize the transformation: kernel measures what gets mapped to zero, image measures the range of outputs.

4. **Rank-nullity theorem** relates dimensions of kernel and image, fundamental for understanding dimensionality reduction and information loss.

5. **Composition** of linear transformations corresponds to matrix multiplication, enabling analysis of multi-layer networks.

6. **Change of basis** allows representing the same transformation in different coordinate systems, useful for understanding embeddings and feature spaces.

7. **Similarity transforms** preserve eigenvalues and other spectral properties, important for understanding how transformations behave under coordinate changes.

8. **Neural network layers** implement affine transformations (linear + translation) followed by nonlinearities, with the linear part being a matrix multiplication.

9. **Projections** appear in dimensionality reduction (PCA), least squares, and many optimization algorithms.

10. **Isomorphisms** show that vector spaces of the same dimension are essentially equivalent, enabling representation flexibility in ML systems.
