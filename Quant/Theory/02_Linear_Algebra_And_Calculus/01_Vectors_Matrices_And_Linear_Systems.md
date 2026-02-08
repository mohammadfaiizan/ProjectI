# Vectors, Matrices, and Linear Systems

## Vector Spaces

### Definition

A vector space $V$ over field $\mathbb{F}$ (typically $\mathbb{R}$ or $\mathbb{C}$) is a set with operations addition $+$ and scalar multiplication satisfying:

1. **Closure**: $u + v \in V$ and $\alpha u \in V$ for all $u, v \in V$, $\alpha \in \mathbb{F}$
2. **Commutativity**: $u + v = v + u$
3. **Associativity**: $(u + v) + w = u + (v + w)$ and $(\alpha\beta)u = \alpha(\beta u)$
4. **Identity**: $0 \in V$ such that $u + 0 = u$
5. **Inverse**: For each $u$, exists $-u$ such that $u + (-u) = 0$
6. **Distributivity**: $\alpha(u + v) = \alpha u + \alpha v$ and $(\alpha + \beta)u = \alpha u + \beta u$
7. **Scalar identity**: $1 \cdot u = u$

### Examples

- **$\mathbb{R}^n$**: $n$-tuples of real numbers
- **$\mathbb{C}^n$**: $n$-tuples of complex numbers
- **Function spaces**: $C[a,b]$ (continuous functions), $L^2[a,b]$ (square-integrable)
- **Matrix spaces**: $M_{m \times n}(\mathbb{R})$ ($m \times n$ matrices)

## Subspaces

### Definition

A subset $W \subseteq V$ is a subspace if:
1. $0 \in W$
2. $u + v \in W$ for all $u, v \in W$ (closed under addition)
3. $\alpha u \in W$ for all $\alpha \in \mathbb{F}$, $u \in W$ (closed under scalar multiplication)

### Examples

- **Null space**: $\text{Null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$
- **Column space**: $\text{Col}(A) = \text{span}\{\mathbf{a}_1, \ldots, \mathbf{a}_n\}$ where $\mathbf{a}_i$ are columns
- **Row space**: $\text{Row}(A) = \text{span}\{\mathbf{r}_1, \ldots, \mathbf{r}_m\}$ where $\mathbf{r}_i$ are rows

## Span, Basis, Dimension

### Linear Combinations

A linear combination of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is:

$$\alpha_1\mathbf{v}_1 + \cdots + \alpha_k\mathbf{v}_k$$

### Span

The span of $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{\alpha_1\mathbf{v}_1 + \cdots + \alpha_k\mathbf{v}_k : \alpha_i \in \mathbb{F}\}$$

### Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are linearly independent if:

$$\alpha_1\mathbf{v}_1 + \cdots + \alpha_k\mathbf{v}_k = \mathbf{0} \implies \alpha_1 = \cdots = \alpha_k = 0$$

Otherwise, they are linearly dependent.

### Basis

A basis for $V$ is a linearly independent set $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ such that:

$$V = \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$$

**Properties**:
- Every vector in $V$ has unique representation: $\mathbf{v} = \alpha_1\mathbf{v}_1 + \cdots + \alpha_n\mathbf{v}_n$
- All bases have same cardinality

### Dimension

The dimension of $V$ is the number of vectors in any basis:

$$\dim(V) = |\text{basis}|$$

**Examples**:
- $\dim(\mathbb{R}^n) = n$
- $\dim(\text{Null}(A)) = n - \text{rank}(A)$ (nullity)
- $\dim(\text{Col}(A)) = \text{rank}(A)$

## Matrix Operations

### Basic Operations

For matrices $A, B \in M_{m \times n}(\mathbb{R})$:

- **Addition**: $(A + B)_{ij} = A_{ij} + B_{ij}$
- **Scalar multiplication**: $(\alpha A)_{ij} = \alpha A_{ij}$
- **Matrix multiplication**: For $A \in M_{m \times n}$, $B \in M_{n \times p}$:
  $$(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$$

**Properties**:
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B + C) = AB + AC$
- NOT commutative: $AB \neq BA$ in general

### Transpose

$$(A^T)_{ij} = A_{ji}$$

**Properties**:
- $(A^T)^T = A$
- $(AB)^T = B^T A^T$
- $(A + B)^T = A^T + B^T$

### Trace

For square matrix $A$:

$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii}$$

**Properties**:
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(AB) = \text{tr}(BA)$
- $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$ (sum of eigenvalues)

## Rank

### Definition

The rank of $A$ is:

$$\text{rank}(A) = \dim(\text{Col}(A)) = \dim(\text{Row}(A))$$

**Properties**:
- $\text{rank}(A) \leq \min(m, n)$
- $\text{rank}(A) = \text{rank}(A^T)$
- $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$
- $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$

### Rank-Nullity Theorem

For $A \in M_{m \times n}$:

$$\text{rank}(A) + \text{nullity}(A) = n$$

where $\text{nullity}(A) = \dim(\text{Null}(A))$.

## Null Space and Column Space

### Null Space

$$\text{Null}(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$$

**Properties**:
- Subspace of $\mathbb{R}^n$
- $\dim(\text{Null}(A)) = n - \text{rank}(A)$
- $A\mathbf{x} = \mathbf{b}$ has solution if and only if $\mathbf{b} \in \text{Col}(A)$

### Column Space

$$\text{Col}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\} = \text{span}\{\mathbf{a}_1, \ldots, \mathbf{a}_n\}$$

**Properties**:
- Subspace of $\mathbb{R}^m$
- $\dim(\text{Col}(A)) = \text{rank}(A)$
- $A\mathbf{x} = \mathbf{b}$ is consistent if and only if $\mathbf{b} \in \text{Col}(A)$

## Systems of Linear Equations

### General Form

$$A\mathbf{x} = \mathbf{b}$$

where $A \in M_{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

### Existence of Solutions

**Theorem**: System $A\mathbf{x} = \mathbf{b}$ has:
- **No solution**: if $\mathbf{b} \notin \text{Col}(A)$ (inconsistent)
- **Unique solution**: if $\mathbf{b} \in \text{Col}(A)$ and $\text{Null}(A) = \{\mathbf{0}\}$ (full column rank)
- **Infinitely many solutions**: if $\mathbf{b} \in \text{Col}(A)$ and $\text{Null}(A) \neq \{\mathbf{0}\}$

**Criterion**: System is consistent if and only if $\text{rank}(A) = \text{rank}([A|\mathbf{b}])$.

### Homogeneous Systems

For $A\mathbf{x} = \mathbf{0}$:
- Always has solution $\mathbf{x} = \mathbf{0}$ (trivial solution)
- Has non-trivial solutions if and only if $\text{rank}(A) < n$
- Solution set is $\text{Null}(A)$

## Gaussian Elimination

### Row Operations

1. **Swap**: $R_i \leftrightarrow R_j$
2. **Scale**: $R_i \to \alpha R_i$ ($\alpha \neq 0$)
3. **Replace**: $R_i \to R_i + \alpha R_j$

### Row Echelon Form

Matrix is in row echelon form if:
- All zero rows are at bottom
- Leading entry (pivot) of each row is to right of pivot above
- Pivots are 1

### Reduced Row Echelon Form

Additional condition:
- All entries above pivots are 0

### Algorithm

1. Find leftmost non-zero column (pivot column)
2. Select pivot (non-zero entry)
3. Use row operations to create zeros below pivot
4. Repeat for submatrix below pivot
5. (For RREF) Create zeros above pivots

**Complexity**: $O(n^3)$ for $n \times n$ matrix.

## Determinants

### Definition

For $A \in M_{n \times n}$, determinant is:

$$\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} A_{i,\sigma(i)}$$

where $S_n$ is symmetric group and $\text{sgn}(\sigma)$ is sign of permutation.

### Properties

1. **Multilinearity**: Linear in each row/column
2. **Alternating**: Swapping rows/columns changes sign
3. **Multiplicative**: $\det(AB) = \det(A)\det(B)$
4. **Transpose**: $\det(A^T) = \det(A)$
5. **Singularity**: $\det(A) = 0$ if and only if $A$ is singular (not invertible)

### Computation

**Cofactor expansion**: For $A \in M_{n \times n}$:

$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} A_{ij} \det(M_{ij})$$

where $M_{ij}$ is $(i,j)$-minor (submatrix removing row $i$ and column $j$).

**Triangular matrices**: $\det(A) = \prod_{i=1}^{n} A_{ii}$

### Applications

- **Invertibility**: $A$ invertible if and only if $\det(A) \neq 0$
- **Volume**: $|\det(A)|$ is volume of parallelepiped spanned by columns
- **Change of variables**: Jacobian determinant in integration

## Cramer's Rule

For system $A\mathbf{x} = \mathbf{b}$ with $\det(A) \neq 0$:

$$x_i = \frac{\det(A_i)}{\det(A)}$$

where $A_i$ is $A$ with column $i$ replaced by $\mathbf{b}$.

**Note**: Computationally expensive ($O(n!)$), mainly of theoretical interest.

## Inner Products

### Definition

An inner product on $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{F}$ satisfying:

1. **Positive definiteness**: $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ with equality if and only if $\mathbf{v} = \mathbf{0}$
2. **Conjugate symmetry**: $\langle \mathbf{u}, \mathbf{v} \rangle = \overline{\langle \mathbf{v}, \mathbf{u} \rangle}$
3. **Linearity**: $\langle \alpha\mathbf{u} + \beta\mathbf{v}, \mathbf{w} \rangle = \alpha\langle \mathbf{u}, \mathbf{w} \rangle + \beta\langle \mathbf{v}, \mathbf{w} \rangle$

### Standard Inner Product

For $\mathbb{R}^n$:

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T\mathbf{v} = \sum_{i=1}^{n} u_i v_i$$

For $\mathbb{C}^n$:

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^*\mathbf{v} = \sum_{i=1}^{n} \overline{u_i} v_i$$

where $\mathbf{u}^*$ is conjugate transpose.

## Norms

### Definition

A norm on $V$ is a function $\|\cdot\|: V \to \mathbb{R}$ satisfying:

1. **Positive definiteness**: $\|\mathbf{v}\| \geq 0$ with equality if and only if $\mathbf{v} = \mathbf{0}$
2. **Homogeneity**: $\|\alpha\mathbf{v}\| = |\alpha|\|\mathbf{v}\|$
3. **Triangle inequality**: $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$

### Common Norms

**Euclidean norm** ($L^2$):

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} |v_i|^2} = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

**$L^p$ norm**:

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$$

**$L^\infty$ norm** (maximum):

$$\|\mathbf{v}\|_\infty = \max_{1 \leq i \leq n} |v_i|$$

**Matrix norms**:
- **Frobenius**: $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^TA)}$
- **Operator**: $\|A\| = \max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\|$

## Orthogonality

### Definition

Vectors $\mathbf{u}$ and $\mathbf{v}$ are orthogonal if:

$$\langle \mathbf{u}, \mathbf{v} \rangle = 0$$

A set $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is orthogonal if $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0$ for $i \neq j$.

If additionally $\|\mathbf{v}_i\| = 1$ for all $i$, the set is orthonormal.

### Orthogonal Complement

For subspace $W \subseteq V$:

$$W^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ for all } \mathbf{w} \in W\}$$

**Properties**:
- $W^\perp$ is subspace
- $V = W \oplus W^\perp$ (direct sum)
- $(W^\perp)^\perp = W$

### Orthogonal Projection

Projection of $\mathbf{v}$ onto subspace $W$ spanned by orthonormal basis $\{\mathbf{w}_1, \ldots, \mathbf{w}_k\}$:

$$\text{proj}_W(\mathbf{v}) = \sum_{i=1}^{k} \langle \mathbf{v}, \mathbf{w}_i \rangle \mathbf{w}_i$$

**Properties**:
- $\mathbf{v} - \text{proj}_W(\mathbf{v}) \in W^\perp$
- Minimizes distance: $\|\mathbf{v} - \text{proj}_W(\mathbf{v})\| \leq \|\mathbf{v} - \mathbf{w}\|$ for all $\mathbf{w} \in W$

## Gram-Schmidt Process

### Algorithm

Given linearly independent vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$, construct orthonormal basis $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$:

1. $\mathbf{u}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}$
2. For $i = 2, \ldots, k$:
   $$\mathbf{w}_i = \mathbf{v}_i - \sum_{j=1}^{i-1} \langle \mathbf{v}_i, \mathbf{u}_j \rangle \mathbf{u}_j$$
   $$\mathbf{u}_i = \frac{\mathbf{w}_i}{\|\mathbf{w}_i\|}$$

**Result**: $\text{span}\{\mathbf{u}_1, \ldots, \mathbf{u}_i\} = \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_i\}$ for all $i$.

### QR Decomposition

For matrix $A \in M_{m \times n}$ with linearly independent columns, Gram-Schmidt yields:

$$A = QR$$

where:
- $Q \in M_{m \times n}$ has orthonormal columns
- $R \in M_{n \times n}$ is upper triangular with positive diagonal

## Applications in Quantitative Finance

### Portfolio Theory

**Portfolio return**: $R_p = \mathbf{w}^T\mathbf{R}$ where $\mathbf{w}$ is weight vector and $\mathbf{R}$ is return vector.

**Portfolio variance**: $\text{Var}(R_p) = \mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w}$ where $\boldsymbol{\Sigma}$ is covariance matrix.

**Optimization**: Minimize $\mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w}$ subject to $\mathbf{w}^T\mathbf{1} = 1$ and $\mathbf{w}^T\boldsymbol{\mu} = \mu_p$.

### Factor Models

**Linear factor model**: $\mathbf{R} = \boldsymbol{\alpha} + \boldsymbol{\beta}\mathbf{F} + \boldsymbol{\epsilon}$

where $\mathbf{F}$ are factors, $\boldsymbol{\beta}$ are factor loadings.

**Estimation**: Use least squares:

$$\hat{\boldsymbol{\beta}} = (\mathbf{F}^T\mathbf{F})^{-1}\mathbf{F}^T\mathbf{R}$$

### Principal Component Analysis

**Covariance matrix decomposition**: $\boldsymbol{\Sigma} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^T$

where $\mathbf{P}$ has eigenvectors (principal components) and $\boldsymbol{\Lambda}$ has eigenvalues.

**Dimensionality reduction**: Use top $k$ principal components to approximate covariance structure.

### Risk Decomposition

**Portfolio risk**: Decompose into systematic and idiosyncratic components:

$$\text{Var}(R_p) = \mathbf{w}^T\boldsymbol{\beta}\boldsymbol{\Sigma}_F\boldsymbol{\beta}^T\mathbf{w} + \mathbf{w}^T\boldsymbol{\Sigma}_\epsilon\mathbf{w}$$

where first term is systematic risk and second is idiosyncratic risk.
