# Numerical Linear Algebra

## LU Factorization

### Definition

For matrix $A \in M_{n \times n}$, LU decomposition is:

$$A = LU$$

where:
- $L$: lower triangular with unit diagonal
- $U$: upper triangular

**Existence**: If all leading principal minors are non-zero, LU decomposition exists (possibly with row permutations).

### Computation

**Gaussian elimination**: Apply row operations to transform $A$ to upper triangular form.

**Algorithm**: For $k = 1, \ldots, n-1$:
1. Find pivot $A_{kk}$ (may require row swap)
2. For $i = k+1, \ldots, n$:
   - $L_{ik} = A_{ik}/A_{kk}$ (multiplier)
   - $A_{ij} \leftarrow A_{ij} - L_{ik}A_{kj}$ for $j = k, \ldots, n$

**Complexity**: $O(n^3/3)$ operations.

### Partial Pivoting

**LU with pivoting**: $PA = LU$ where $P$ is permutation matrix.

**Strategy**: Choose pivot $A_{kk}$ with largest absolute value in column $k$ below diagonal.

**Benefits**: 
- Numerical stability
- Handles singular matrices gracefully

### Applications

**Linear systems**: Solve $A\mathbf{x} = \mathbf{b}$:
1. Factor: $A = LU$
2. Solve: $L\mathbf{y} = \mathbf{b}$ (forward substitution)
3. Solve: $U\mathbf{x} = \mathbf{y}$ (backward substitution)

**Determinant**: $\det(A) = \det(L)\det(U) = \prod_{i=1}^{n} U_{ii}$ (with sign from $P$)

**Inverse**: Solve $AX = I$ column by column.

## QR Factorization Algorithms

### Householder Reflections

**Householder matrix**: 

$$\mathbf{H} = \mathbf{I} - 2\frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}$$

where $\mathbf{v}$ is chosen to zero out entries below diagonal.

**Properties**:
- Orthogonal: $\mathbf{H}^T\mathbf{H} = \mathbf{I}$
- Symmetric: $\mathbf{H} = \mathbf{H}^T$
- Involutory: $\mathbf{H}^2 = \mathbf{I}$

**Algorithm**: For $k = 1, \ldots, n$:
1. Choose $\mathbf{v}_k$ to zero $A_{k+1:n,k}$
2. Apply: $A \leftarrow \mathbf{H}_k A$
3. Accumulate: $Q \leftarrow Q\mathbf{H}_k$

**Complexity**: $O(2mn^2 - 2n^3/3)$ for $m \times n$ matrix.

### Givens Rotations

**Givens matrix**: Rotation in plane $(i,j)$:

$$G(i,j,\theta) = \begin{pmatrix}
1 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \ddots & \vdots & & \vdots & & \vdots \\
0 & \cdots & c & \cdots & s & \cdots & 0 \\
\vdots & & \vdots & \ddots & \vdots & & \vdots \\
0 & \cdots & -s & \cdots & c & \cdots & 0 \\
\vdots & & \vdots & & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & \cdots & 0 & \cdots & 1
\end{pmatrix}$$

where $c = \cos\theta$, $s = \sin\theta$ chosen to zero $A_{ji}$.

**Advantages**: 
- Useful for sparse matrices
- Can zero single element

**Complexity**: More operations than Householder for dense matrices.

### Modified Gram-Schmidt

**Classical Gram-Schmidt**: Numerically unstable due to cancellation.

**Modified version**: 
1. $\mathbf{q}_1 = \mathbf{a}_1 / \|\mathbf{a}_1\|$
2. For $i = 2, \ldots, n$:
   - $\mathbf{v}_i = \mathbf{a}_i$
   - For $j = 1, \ldots, i-1$:
     - $\mathbf{v}_i \leftarrow \mathbf{v}_i - \langle \mathbf{v}_i, \mathbf{q}_j \rangle \mathbf{q}_j$
   - $\mathbf{q}_i = \mathbf{v}_i / \|\mathbf{v}_i\|$

**Stability**: Better than classical, but Householder preferred for dense matrices.

## Cholesky Factorization Algorithms

### Standard Algorithm

For positive definite $A$:

**Algorithm**: For $i = 1, \ldots, n$:

$$L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}$$

$$L_{ji} = \frac{A_{ji} - \sum_{k=1}^{i-1} L_{jk}L_{ik}}{L_{ii}}, \quad j = i+1, \ldots, n$$

**Complexity**: $O(n^3/3)$ operations (half of LU).

**Stability**: More stable than LU for positive definite matrices.

### Block Cholesky

**Partition**: 

$$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$$

**Block factorization**:

$$L = \begin{pmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{pmatrix}$$

where:
- $L_{11}L_{11}^T = A_{11}$ (recursive)
- $L_{21} = A_{21}L_{11}^{-T}$
- $L_{22}L_{22}^T = A_{22} - L_{21}L_{21}^T$ (Schur complement)

**Benefits**: Better cache performance, parallelization.

## Iterative Methods

### Jacobi Method

For system $A\mathbf{x} = \mathbf{b}$, split $A = D + L + U$ where:
- $D$: diagonal
- $L$: strictly lower triangular
- $U$: strictly upper triangular

**Iteration**:

$$\mathbf{x}^{(k+1)} = D^{-1}(\mathbf{b} - (L+U)\mathbf{x}^{(k)})$$

**Component form**:

$$x_i^{(k+1)} = \frac{1}{A_{ii}}\left(b_i - \sum_{j \neq i} A_{ij}x_j^{(k)}\right)$$

**Convergence**: Converges if $\rho(D^{-1}(L+U)) < 1$ (spectral radius).

### Gauss-Seidel Method

**Iteration**:

$$\mathbf{x}^{(k+1)} = (D+L)^{-1}(\mathbf{b} - U\mathbf{x}^{(k)})$$

**Component form**:

$$x_i^{(k+1)} = \frac{1}{A_{ii}}\left(b_i - \sum_{j=1}^{i-1} A_{ij}x_j^{(k+1)} - \sum_{j=i+1}^{n} A_{ij}x_j^{(k)}\right)$$

**Advantages**: Uses updated values immediately (faster convergence than Jacobi).

**Convergence**: Converges if $A$ is strictly diagonally dominant or positive definite.

### Successive Over-Relaxation (SOR)

**Iteration**:

$$\mathbf{x}^{(k+1)} = (D+\omega L)^{-1}((1-\omega)D\mathbf{x}^{(k)} - \omega U\mathbf{x}^{(k)} + \omega\mathbf{b})$$

where $\omega \in (0,2)$ is relaxation parameter.

**Optimal $\omega$**: For certain matrices, optimal $\omega$ minimizes spectral radius.

### Conjugate Gradient Method

For symmetric positive definite $A$, CG minimizes:

$$f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A\mathbf{x} - \mathbf{b}^T\mathbf{x}$$

**Algorithm**:
1. Initialize: $\mathbf{x}^{(0)} = \mathbf{0}$, $\mathbf{r}^{(0)} = \mathbf{b}$, $\mathbf{p}^{(0)} = \mathbf{r}^{(0)}$
2. For $k = 0, 1, \ldots$:
   - $\alpha_k = \frac{\mathbf{r}^{(k)T}\mathbf{r}^{(k)}}{\mathbf{p}^{(k)T}A\mathbf{p}^{(k)}}$
   - $\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha_k\mathbf{p}^{(k)}$
   - $\mathbf{r}^{(k+1)} = \mathbf{r}^{(k)} - \alpha_k A\mathbf{p}^{(k)}$
   - $\beta_k = \frac{\mathbf{r}^{(k+1)T}\mathbf{r}^{(k+1)}}{\mathbf{r}^{(k)T}\mathbf{r}^{(k)}}$
   - $\mathbf{p}^{(k+1)} = \mathbf{r}^{(k+1)} + \beta_k\mathbf{p}^{(k)}$

**Properties**:
- Converges in at most $n$ iterations (exact arithmetic)
- Optimal: minimizes error in $A$-norm at each step
- Efficient for sparse matrices

**Preconditioning**: Use preconditioner $M \approx A$ to accelerate convergence:

$$M^{-1}A\mathbf{x} = M^{-1}\mathbf{b}$$

## Conditioning and Numerical Stability

### Condition Number

For matrix $A$:

$$\kappa(A) = \|A\|\|A^{-1}\|$$

**For 2-norm**: $\kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$ (ratio of largest to smallest singular value).

**Interpretation**: 
- $\kappa(A) \approx 1$: well-conditioned
- $\kappa(A) \gg 1$: ill-conditioned
- $\kappa(A) = \infty$: singular

**Error bound**: For $A\mathbf{x} = \mathbf{b}$:

$$\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A)\frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$$

### Floating Point Considerations

**IEEE 754**: Standard for floating point arithmetic.

**Machine epsilon**: $\epsilon_{\text{mach}}$ is smallest number such that $1 + \epsilon_{\text{mach}} > 1$.

**Roundoff error**: Operations introduce errors of order $\epsilon_{\text{mach}}$.

**Stability**: Algorithm is stable if errors don't grow faster than condition number.

**Backward stability**: Computed solution is exact solution of nearby problem.

## Sparse Matrices

### Storage Formats

**Compressed Sparse Row (CSR)**:
- `val`: non-zero values
- `col_ind`: column indices
- `row_ptr`: row pointers

**Compressed Sparse Column (CSC)**: Similar, column-oriented.

**Coordinate format**: List of $(i, j, A_{ij})$ for non-zeros.

### Sparse Matrix Operations

**Matrix-vector product**: $O(\text{nnz})$ where $\text{nnz}$ is number of non-zeros.

**Factorization**: Sparse LU/Cholesky preserve sparsity structure (fill-in).

**Reordering**: Minimize fill-in using:
- Minimum degree ordering
- Nested dissection
- Cuthill-McKee algorithm

### Applications

**Finite difference methods**: Discretization of PDEs yields sparse matrices.

**Graph algorithms**: Adjacency matrices are sparse.

**Portfolio optimization**: Covariance matrices often sparse (factor models).

## Randomized Linear Algebra

### Random Projections

**Johnson-Lindenstrauss lemma**: For $n$ points in $\mathbb{R}^d$, can embed into $\mathbb{R}^k$ with $k = O(\epsilon^{-2}\ln n)$ preserving pairwise distances.

**Random projection**: $\mathbf{P} = \frac{1}{\sqrt{k}}\mathbf{R}$ where $\mathbf{R}_{ij} \sim \mathcal{N}(0,1)$.

**Application**: Dimensionality reduction, approximate matrix multiplication.

### Sketching

**Sketch matrix**: $\mathbf{S} \in \mathbb{R}^{m \times n}$ with $m \ll n$ such that $\mathbf{S}A$ preserves properties of $A$.

**Types**:
- Gaussian: $\mathbf{S}_{ij} \sim \mathcal{N}(0, 1/m)$
- Subsampled randomized Hadamard transform
- Count sketch

**Applications**:
- Low-rank approximation
- Least squares: Solve $\min_{\mathbf{x}} \|\mathbf{S}A\mathbf{x} - \mathbf{S}\mathbf{b}\|$
- Matrix multiplication: Approximate $AB$ using sketches

### Randomized SVD

**Algorithm**:
1. Generate random matrix $\mathbf{\Omega} \in \mathbb{R}^{n \times k}$
2. Form $\mathbf{Y} = A\mathbf{\Omega}$
3. Compute QR: $\mathbf{Y} = QR$
4. Form $\mathbf{B} = Q^TA$
5. Compute SVD: $\mathbf{B} = \tilde{U}\Sigma V^T$
6. Set $U = Q\tilde{U}$

**Complexity**: $O(mnk)$ instead of $O(mn^2)$ for full SVD.

**Accuracy**: With high probability, error is small if $k$ sufficiently large.

### Applications

**Principal component analysis**: Approximate top $k$ components efficiently.

**Covariance estimation**: Estimate $\boldsymbol{\Sigma}$ from streaming data.

**Kernel methods**: Approximate kernel matrices for large datasets.

## Applications in Quantitative Finance

### Covariance Matrix Estimation

**Sample covariance**: $\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}(\mathbf{X} - \bar{\mathbf{X}})^T(\mathbf{X} - \bar{\mathbf{X}})$

**Shrinkage estimators**: 

$$\hat{\boldsymbol{\Sigma}}_{\text{shrink}} = \lambda \hat{\boldsymbol{\Sigma}} + (1-\lambda)\mathbf{F}$$

where $\mathbf{F}$ is target (e.g., factor model).

**Regularization**: Add $\epsilon I$ to ensure positive definiteness.

### Factor Model Estimation

**Principal component analysis**: 

$$\boldsymbol{\Sigma} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^T$$

Use top $k$ components for dimensionality reduction.

**Sparse PCA**: Add sparsity constraint to factor loadings.

**Randomized PCA**: Use randomized SVD for large covariance matrices.

### Portfolio Optimization

**Efficient frontier**: Solve quadratic program:

$$\min_{\mathbf{w}} \mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w} \quad \text{subject to } \mathbf{w}^T\mathbf{1} = 1, \quad \mathbf{w}^T\boldsymbol{\mu} = \mu_p$$

**Cholesky method**: Factor $\boldsymbol{\Sigma} = LL^T$, solve triangular systems.

**Conjugate gradient**: For large problems, use CG with preconditioner.

**Sparse methods**: Exploit sparsity in factor model structure.

### Risk Decomposition

**Eigenvalue decomposition**: 

$$\boldsymbol{\Sigma} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^T$$

**Risk contributions**: 

$$\text{RC}_i = w_i \frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_p}$$

**Computation**: Use Cholesky or eigenvalue decomposition.

### Monte Carlo Simulation

**Correlated random variables**: Generate $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I)$, then:

$$\mathbf{X} = \boldsymbol{\mu} + L\mathbf{Z}$$

where $\boldsymbol{\Sigma} = LL^T$ (Cholesky).

**Efficient generation**: Use sparse Cholesky if $\boldsymbol{\Sigma}$ sparse.

### Kalman Filtering

**State covariance update**: 

$$\mathbf{P}_{t|t-1} = \mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}^T + \mathbf{Q}$$

**Cholesky form**: Maintain $\mathbf{P} = LL^T$ for numerical stability.

**Square root filter**: Propagate Cholesky factors instead of covariance matrices.

### High-Frequency Data

**Large covariance matrices**: Use randomized methods for dimensionality reduction.

**Streaming data**: Update covariance estimates incrementally.

**Sparse structure**: Exploit sparsity in correlation matrices (many pairs uncorrelated).
