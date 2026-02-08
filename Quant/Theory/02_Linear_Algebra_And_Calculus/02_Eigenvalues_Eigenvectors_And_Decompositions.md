# Eigenvalues, Eigenvectors, and Decompositions

## Eigenvalue Problems

### Definition

For matrix $A \in M_{n \times n}$, eigenvalue $\lambda$ and eigenvector $\mathbf{v} \neq \mathbf{0}$ satisfy:

$$A\mathbf{v} = \lambda\mathbf{v}$$

**Eigenspace**: $E_\lambda = \{\mathbf{v} : A\mathbf{v} = \lambda\mathbf{v}\} = \text{Null}(A - \lambda I)$

**Geometric multiplicity**: $\dim(E_\lambda)$

### Characteristic Polynomial

Eigenvalues are roots of characteristic polynomial:

$$p_A(\lambda) = \det(A - \lambda I) = 0$$

**Algebraic multiplicity**: Multiplicity of $\lambda$ as root of $p_A(\lambda)$.

**Cayley-Hamilton theorem**: $p_A(A) = 0$ (matrix satisfies its own characteristic equation).

### Properties

1. **Trace**: $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$
2. **Determinant**: $\det(A) = \prod_{i=1}^{n} \lambda_i$
3. **Powers**: Eigenvalues of $A^k$ are $\lambda_i^k$
4. **Inverse**: If $A$ invertible, eigenvalues of $A^{-1}$ are $1/\lambda_i$
5. **Transpose**: $A$ and $A^T$ have same eigenvalues (but different eigenvectors in general)

### Diagonalization

Matrix $A$ is diagonalizable if there exists invertible $P$ such that:

$$A = P\Lambda P^{-1}$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

**Necessary and sufficient condition**: $A$ has $n$ linearly independent eigenvectors (geometric multiplicity equals algebraic multiplicity for each eigenvalue).

**Powers**: $A^k = P\Lambda^k P^{-1}$

## Spectral Theorem for Symmetric Matrices

### Real Symmetric Matrices

For symmetric $A = A^T \in M_{n \times n}(\mathbb{R})$:

1. All eigenvalues are real
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal
3. $A$ is orthogonally diagonalizable: $A = Q\Lambda Q^T$ where $Q$ is orthogonal ($Q^T Q = I$)

**Proof outline**: 
- Eigenvalues real: $\lambda = \overline{\lambda}$ since $A$ symmetric
- Orthogonality: For $\lambda_1 \neq \lambda_2$, $\mathbf{v}_1^T\mathbf{v}_2 = 0$

### Positive Definite Matrices

Symmetric $A$ is positive definite if:

$$\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq \mathbf{0}$$

**Equivalent conditions**:
- All eigenvalues positive: $\lambda_i > 0$
- All principal minors positive
- Exists Cholesky decomposition: $A = LL^T$ with $L$ lower triangular

**Positive semidefinite**: $\mathbf{x}^T A \mathbf{x} \geq 0$ (eigenvalues $\geq 0$)

### Applications

**Covariance matrices**: Always positive semidefinite:

$$\mathbf{x}^T \boldsymbol{\Sigma} \mathbf{x} = \text{Var}(\mathbf{x}^T\mathbf{R}) \geq 0$$

**Quadratic forms**: Portfolio variance $\mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w}$ is quadratic form in positive semidefinite matrix.

## Singular Value Decomposition

### Definition

For $A \in M_{m \times n}$, SVD is:

$$A = U\Sigma V^T$$

where:
- $U \in M_{m \times m}$: orthogonal (left singular vectors)
- $\Sigma \in M_{m \times n}$: diagonal with singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$ ($r = \text{rank}(A)$)
- $V \in M_{n \times n}$: orthogonal (right singular vectors)

**Singular values**: $\sigma_i = \sqrt{\lambda_i(A^TA)} = \sqrt{\lambda_i(AA^T)}$

**Relationship to eigenvalues**:
- Non-zero singular values of $A$ are square roots of non-zero eigenvalues of $A^TA$ or $AA^T$
- Columns of $U$ are eigenvectors of $AA^T$
- Columns of $V$ are eigenvectors of $A^TA$

### Properties

1. **Rank**: $\text{rank}(A) = \text{number of non-zero singular values}$
2. **Frobenius norm**: $\|A\|_F^2 = \sum_{i=1}^{r} \sigma_i^2$
3. **Operator norm**: $\|A\| = \sigma_1$ (largest singular value)
4. **Condition number**: $\kappa(A) = \sigma_1/\sigma_r$ (ratio of largest to smallest singular value)

### Computation

**Algorithm**:
1. Compute eigenvalues/eigenvectors of $A^TA$: $A^TA = V\Lambda V^T$
2. Singular values: $\sigma_i = \sqrt{\lambda_i}$
3. Left singular vectors: $\mathbf{u}_i = \frac{1}{\sigma_i}A\mathbf{v}_i$ (for $\sigma_i > 0$)

**Numerical methods**: Use iterative algorithms (e.g., power method, QR algorithm) for large matrices.

### Applications

**PCA**: SVD of centered data matrix:

$$X = U\Sigma V^T$$

- Principal components: columns of $V$
- Principal component scores: $U\Sigma$
- Explained variance: $\sigma_i^2$

**Low-rank approximation**: Best rank-$k$ approximation:

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

minimizes $\|A - A_k\|_F$ over all rank-$k$ matrices (Eckart-Young theorem).

**Pseudoinverse**: Moore-Penrose pseudoinverse:

$$A^+ = V\Sigma^+ U^T$$

where $\Sigma^+$ has $1/\sigma_i$ for non-zero $\sigma_i$, 0 otherwise.

**Least squares**: Solution to $A\mathbf{x} = \mathbf{b}$:

$$\mathbf{x} = A^+\mathbf{b} = V\Sigma^+ U^T\mathbf{b}$$

## QR Decomposition

### Definition

For $A \in M_{m \times n}$ with $m \geq n$ and linearly independent columns:

$$A = QR$$

where:
- $Q \in M_{m \times n}$: columns are orthonormal ($Q^T Q = I$)
- $R \in M_{n \times n}$: upper triangular with positive diagonal

### Computation

**Gram-Schmidt**: Orthonormalize columns of $A$ to get $Q$, then $R = Q^T A$.

**Householder reflections**: More numerically stable:
1. Apply Householder transformations to zero below diagonal
2. $Q$ is product of Householder matrices
3. $R$ is upper triangular result

**Givens rotations**: Alternative method using plane rotations.

### Properties

1. **Uniqueness**: If $A$ has full column rank and $R$ has positive diagonal, decomposition is unique
2. **Least squares**: Solution to $A\mathbf{x} = \mathbf{b}$:
   - $QR\mathbf{x} = \mathbf{b} \implies R\mathbf{x} = Q^T\mathbf{b}$
   - Solve triangular system for $\mathbf{x}$

### Applications

**Least squares regression**: For $X\boldsymbol{\beta} = \mathbf{y}$:

$$X = QR \implies R\boldsymbol{\beta} = Q^T\mathbf{y}$$

**Eigenvalue computation**: QR algorithm iteratively applies QR decomposition to converge to Schur form.

## Cholesky Decomposition

### Definition

For positive definite $A \in M_{n \times n}$, Cholesky decomposition is:

$$A = LL^T$$

where $L$ is lower triangular with positive diagonal.

**Uniqueness**: If $A$ is positive definite, $L$ is unique.

### Computation

**Algorithm**: For $i = 1, \ldots, n$:

$$L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}$$

$$L_{ji} = \frac{A_{ji} - \sum_{k=1}^{i-1} L_{jk}L_{ik}}{L_{ii}} \quad \text{for } j = i+1, \ldots, n$$

**Complexity**: $O(n^3/3)$ operations (half of LU decomposition).

### Properties

1. **Positive definiteness**: $A$ positive definite if and only if Cholesky exists
2. **Stability**: More stable than LU for positive definite matrices
3. **Determinant**: $\det(A) = \prod_{i=1}^{n} L_{ii}^2$

### Applications

**Covariance matrix**: For $\boldsymbol{\Sigma} = LL^T$, generate correlated random variables:

$$\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I) \implies L\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

**Portfolio optimization**: Solve $\boldsymbol{\Sigma}\mathbf{w} = \boldsymbol{\mu}$ efficiently:

$$LL^T\mathbf{w} = \boldsymbol{\mu} \implies L\mathbf{y} = \boldsymbol{\mu}, \quad L^T\mathbf{w} = \mathbf{y}$$

**Kalman filter**: Update covariance matrices using Cholesky for numerical stability.

## Matrix Exponential

### Definition

For $A \in M_{n \times n}$:

$$e^A = \exp(A) = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

**Convergence**: Series converges for all matrices.

### Properties

1. **Eigenvalues**: If $\lambda$ is eigenvalue of $A$, then $e^\lambda$ is eigenvalue of $e^A$
2. **Determinant**: $\det(e^A) = e^{\text{tr}(A)}$
3. **Inverse**: $(e^A)^{-1} = e^{-A}$
4. **Derivative**: $\frac{d}{dt}e^{tA} = Ae^{tA}$
5. **Non-commutativity**: $e^{A+B} = e^A e^B$ if and only if $AB = BA$

### Computation

**Diagonalizable case**: If $A = P\Lambda P^{-1}$:

$$e^A = P e^\Lambda P^{-1} = P \text{diag}(e^{\lambda_1}, \ldots, e^{\lambda_n}) P^{-1}$$

**General case**: Use Padé approximation or scaling and squaring:

$$e^A = (e^{A/2^m})^{2^m}$$

for large $m$, compute $e^{A/2^m}$ using Padé approximation.

### Applications

**Linear ODEs**: Solution to $\frac{d\mathbf{x}}{dt} = A\mathbf{x}$:

$$\mathbf{x}(t) = e^{At}\mathbf{x}(0)$$

**Stochastic processes**: For diffusion $dX_t = AX_t dt + \sigma dW_t$:

$$X_t = e^{At}X_0 + \int_0^t e^{A(t-s)}\sigma dW_s$$

**Interest rate models**: In Vasicek model, transition density involves matrix exponential.

## Cayley-Hamilton Theorem

### Statement

Every square matrix satisfies its own characteristic equation:

$$p_A(A) = 0$$

where $p_A(\lambda) = \det(A - \lambda I)$.

**Implication**: $A^n$ can be expressed as linear combination of $I, A, \ldots, A^{n-1}$.

### Applications

**Matrix powers**: Compute $A^k$ for large $k$ using recurrence relation.

**Inverse**: If $p_A(\lambda) = \lambda^n + c_{n-1}\lambda^{n-1} + \cdots + c_0$, then:

$$A^{-1} = -\frac{1}{c_0}(A^{n-1} + c_{n-1}A^{n-2} + \cdots + c_1 I)$$

(if $c_0 \neq 0$, i.e., $A$ invertible).

## Applications in Quantitative Finance

### PCA for Factor Models

**Covariance decomposition**: $\boldsymbol{\Sigma} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^T$

- Principal components: columns of $\mathbf{P}$ (eigenvectors)
- Factor loadings: $\boldsymbol{\beta} = \mathbf{P}$
- Factor variances: diagonal of $\boldsymbol{\Lambda}$ (eigenvalues)

**Dimensionality reduction**: Use top $k$ principal components:

$$\boldsymbol{\Sigma} \approx \mathbf{P}_k \boldsymbol{\Lambda}_k \mathbf{P}_k^T$$

where $\mathbf{P}_k$ contains top $k$ eigenvectors.

**Explained variance**: Proportion explained by top $k$ components:

$$\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{n} \lambda_i}$$

### Covariance Matrix Decomposition

**Cholesky for simulation**: Generate correlated returns:

$$\mathbf{R} = \boldsymbol{\mu} + L\mathbf{Z}$$

where $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I)$ and $\boldsymbol{\Sigma} = LL^T$.

**Square root**: For risk metrics requiring $\boldsymbol{\Sigma}^{1/2}$:

$$\boldsymbol{\Sigma}^{1/2} = \mathbf{P}\boldsymbol{\Lambda}^{1/2}\mathbf{P}^T$$

where $\boldsymbol{\Lambda}^{1/2} = \text{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n})$.

### Risk Decomposition

**Eigenvalue decomposition**: Decompose portfolio risk:

$$\text{Var}(R_p) = \mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w} = \mathbf{w}^T\mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^T\mathbf{w} = \sum_{i=1}^{n} \lambda_i (\mathbf{p}_i^T\mathbf{w})^2$$

where $\mathbf{p}_i$ are eigenvectors. Each term represents risk contribution from principal component $i$.

### Matrix Factorization Models

**Low-rank approximation**: Approximate covariance matrix:

$$\boldsymbol{\Sigma} \approx \mathbf{F}\mathbf{F}^T + \boldsymbol{\Psi}$$

where $\mathbf{F}$ is $n \times k$ factor loading matrix and $\boldsymbol{\Psi}$ is diagonal idiosyncratic variance.

**Estimation**: Use SVD or maximum likelihood.

### Kalman Filtering

**State-space model**: 

$$\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t$$
$$\mathbf{y}_t = \mathbf{H}\mathbf{x}_t + \mathbf{v}_t$$

**Covariance update**: Uses Cholesky decomposition for numerical stability:

$$\mathbf{P}_{t|t-1} = \mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}^T + \mathbf{Q}$$

where $\mathbf{P}$ is state covariance matrix.

### Option Pricing

**Multivariate Black-Scholes**: For basket option on $n$ assets:

$$C = e^{-rT} E^Q[\max(\mathbf{w}^T\mathbf{S}_T - K, 0)]$$

where $\mathbf{S}_T \sim \mathcal{N}_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ under $Q$.

**Computation**: Use Cholesky to generate correlated asset prices:

$$\mathbf{S}_T = \exp(\boldsymbol{\mu} + L\mathbf{Z})$$

where $L$ is Cholesky factor of $\boldsymbol{\Sigma}$.
