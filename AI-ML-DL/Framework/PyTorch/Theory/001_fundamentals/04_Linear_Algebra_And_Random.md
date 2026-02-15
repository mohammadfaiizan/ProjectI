# Linear Algebra and Random Generation

## Table of Contents
1. [Matrix Multiplication](#matrix-multiplication)
2. [Vector Operations](#vector-operations)
3. [Matrix Properties](#matrix-properties)
4. [Matrix Norms](#matrix-norms)
5. [Matrix Decompositions](#matrix-decompositions)
6. [Inverse and Pseudoinverse](#inverse-and-pseudoinverse)
7. [Solving Linear Systems](#solving-linear-systems)
8. [Determinant and Rank](#determinant-and-rank)
9. [Batch Linear Algebra](#batch-linear-algebra)
10. [Einstein Summation](#einstein-summation)
11. [Random Number Generation](#random-number-generation)
12. [Probability Distributions](#probability-distributions)
13. [Seed Management and Reproducibility](#seed-management-and-reproducibility)

---

## Matrix Multiplication

### Functions

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `torch.mm(A, B)` | 2D x 2D | 2D | Strict matrix multiply |
| `torch.mv(A, v)` | 2D x 1D | 1D | Matrix-vector multiply |
| `torch.bmm(A, B)` | 3D x 3D | 3D | Batch matrix multiply |
| `torch.matmul(A, B)` | Any x Any | Varies | General-purpose, supports broadcasting |
| `A @ B` | Any x Any | Varies | Operator alias for `matmul` |

### matmul Behavior by Input Dimensions

| A dims | B dims | Behavior |
|--------|--------|----------|
| 1D | 1D | Dot product (scalar) |
| 2D | 1D | Matrix-vector product (1D) |
| 1D | 2D | Vector-matrix product (1D) |
| 2D | 2D | Matrix multiply (2D) |
| ND | ND | Batched matmul on last two dims with broadcasting |

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

torch.mm(A, B)                # (3, 5)
torch.matmul(A, B)            # (3, 5)
A @ B                          # (3, 5)

v = torch.randn(4)
torch.mv(A, v)                # (3,)
torch.matmul(A, v)            # (3,)

# Batch
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
torch.bmm(batch_A, batch_B)   # (10, 3, 5)
```

---

## Vector Operations

### Dot Product

Sum of element-wise products. Only for 1D tensors.

```python
u = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([4.0, 5.0, 6.0])

torch.dot(u, w)                # 32.0  (1*4 + 2*5 + 3*6)
torch.sum(u * w)               # equivalent
```

### Cross Product

Produces a vector perpendicular to both inputs. Only for 3D vectors.

```python
torch.cross(u, w)
```

### Outer Product

```python
torch.outer(u, w)             # shape (3, 3)
# Equivalent to: u.unsqueeze(1) @ w.unsqueeze(0)
```

---

## Matrix Properties

### Transpose

```python
A.t()                          # 2D transpose (shorthand)
A.T                            # same as .t()
A.transpose(0, 1)             # explicit dim swap
```

### Trace

Sum of diagonal elements.

```python
torch.trace(A)                 # sum of A[i, i]
torch.sum(torch.diag(A))      # equivalent
```

### Diagonal

```python
torch.diag(A)                  # extract diagonal from matrix -> 1D
torch.diag(v)                  # create diagonal matrix from vector -> 2D
torch.diag(A, diagonal=1)     # super-diagonal
```

---

## Matrix Norms

### Vector Norms

```python
v = torch.tensor([3.0, 4.0])
torch.norm(v, p=1)            # L1 norm: |3| + |4| = 7
torch.norm(v, p=2)            # L2 norm: sqrt(9 + 16) = 5
torch.norm(v, p=float('inf')) # max norm: max(|3|, |4|) = 4
```

### Matrix Norms

| Norm | Code | Definition |
|------|------|------------|
| Frobenius | `torch.norm(A, p='fro')` | sqrt(sum of squared elements) |
| Nuclear | `torch.norm(A, p='nuc')` | sum of singular values |
| Spectral | `torch.norm(A, p=2)` | largest singular value |

```python
torch.norm(matrix, dim=0)     # column-wise L2 norms
torch.norm(matrix, dim=1)     # row-wise L2 norms
```

---

## Matrix Decompositions

### QR Decomposition

Factorizes A into an orthogonal matrix Q and upper triangular matrix R.

```python
Q, R = torch.linalg.qr(A)
# A = Q @ R
# Q.T @ Q = I
```

### SVD (Singular Value Decomposition)

Factorizes A into U * S * V^T where U and V are orthogonal and S contains singular values.

```python
U, S, Vh = torch.linalg.svd(A)
# A = U @ torch.diag(S) @ Vh
```

### Eigenvalue Decomposition

```python
# For general matrices
eigenvalues, eigenvectors = torch.linalg.eig(A)

# For symmetric/Hermitian matrices (faster, real eigenvalues)
eigenvalues, eigenvectors = torch.linalg.eigh(A_symmetric)
```

### Cholesky Decomposition

For symmetric positive-definite matrices. Returns lower triangular matrix L such that A = L @ L^T.

```python
L = torch.linalg.cholesky(A_spd)
# A_spd = L @ L.T
```

### LU Decomposition

```python
LU, pivots = torch.linalg.lu_factor(A)
```

---

## Inverse and Pseudoinverse

### Matrix Inverse

Exists only for square, non-singular matrices.

```python
A_inv = torch.linalg.inv(A)
# A @ A_inv ~ I
torch.allclose(A @ A_inv, torch.eye(A.size(0)))
```

### Moore-Penrose Pseudoinverse

Works for any matrix (including non-square and singular).

```python
A_pinv = torch.linalg.pinv(A)
# A @ A_pinv @ A ~ A
```

---

## Solving Linear Systems

### Solve Ax = b

```python
A = torch.randn(4, 4) + 2 * torch.eye(4)
b = torch.randn(4)

x = torch.linalg.solve(A, b)
# A @ x ~ b

# Multiple right-hand sides
B = torch.randn(4, 3)
X = torch.linalg.solve(A, B)
```

### Least Squares (Overdetermined Systems)

```python
# Solve min ||Ax - b||^2
result = torch.linalg.lstsq(A, b)
x = result.solution
```

---

## Determinant and Rank

### Determinant

```python
torch.linalg.det(A)
torch.logdet(A)                           # log(det(A)), numerically stable
sign, logabsdet = torch.linalg.slogdet(A) # sign and log|det(A)|
```

### Matrix Rank

```python
torch.linalg.matrix_rank(A)
torch.linalg.matrix_rank(A, tol=1e-6)    # custom tolerance
```

### Condition Number

The ratio of the largest to smallest singular value. Large condition numbers indicate near-singularity.

```python
torch.linalg.cond(A)
```

### Low-Rank Approximation

```python
U, S, Vh = torch.linalg.svd(A)
k = 2
A_approx = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
```

---

## Batch Linear Algebra

Most `torch.linalg` functions support batch dimensions. The last two dimensions are treated as the matrix, and leading dimensions are batched.

```python
batch = torch.randn(10, 4, 4)

torch.linalg.det(batch)           # (10,)
torch.linalg.inv(batch)           # (10, 4, 4)
torch.bmm(batch, batch)           # (10, 4, 4)
```

---

## Einstein Summation

`torch.einsum` expresses tensor operations using Einstein summation notation. It handles matrix multiply, batch operations, traces, and arbitrary contractions in a single interface.

```python
# Matrix multiplication
torch.einsum('ij,jk->ik', A, B)

# Batch matrix multiplication
torch.einsum('bij,bjk->bik', batch_A, batch_B)

# Trace
torch.einsum('ii->', A)

# Tensor contraction
torch.einsum('ijk,kl->ijl', tensor_3d, matrix)

# Diagonal
torch.einsum('ii->i', A)
```

---

## Random Number Generation

### Basic Generators

```python
torch.rand(3, 4)                # uniform [0, 1)
torch.randn(3, 4)               # standard normal N(0, 1)
torch.randint(0, 10, (3, 4))    # integers in [0, 10)
torch.randperm(10)              # random permutation of 0..9
```

### Custom Normal Distribution

```python
torch.normal(mean=5.0, std=2.0, size=(3, 4))

# Per-element parameters
means = torch.tensor([1.0, 2.0, 3.0])
stds = torch.tensor([0.5, 1.0, 1.5])
torch.normal(means, stds)
```

### Like Functions

Create random tensors matching the shape and dtype of an existing tensor.

```python
base = torch.zeros(2, 3, 4)
torch.rand_like(base)
torch.randn_like(base)
torch.randint_like(base, 0, 10)
```

### In-Place Random Filling

```python
tensor = torch.empty(3, 4)
tensor.uniform_(0, 1)          # uniform
tensor.normal_(0, 1)           # normal
tensor.exponential_(1.0)       # exponential
tensor.geometric_(0.5)         # geometric
tensor.cauchy_(0.0, 1.0)       # Cauchy
tensor.log_normal_(0.0, 1.0)   # log-normal
```

---

## Probability Distributions

PyTorch provides a distributions module for sampling and computing log-probabilities.

```python
from torch.distributions import Normal, Uniform, Beta, Gamma, Bernoulli

# Normal distribution
dist = Normal(loc=0.0, scale=1.0)
samples = dist.sample((100,))
log_probs = dist.log_prob(samples)

# Beta distribution
Beta(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0])).sample()

# Gamma distribution
Gamma(concentration=torch.tensor([1.0, 2.0]),
      rate=torch.tensor([1.0, 1.0])).sample()

# Multinomial sampling
weights = torch.tensor([0.1, 0.3, 0.4, 0.2])
torch.multinomial(weights, num_samples=10, replacement=True)

# Poisson
torch.poisson(torch.tensor([1.0, 2.0, 3.0]))
```

---

## Seed Management and Reproducibility

### Setting Seeds

```python
torch.manual_seed(42)                    # CPU random seed
torch.cuda.manual_seed(42)              # current GPU seed
torch.cuda.manual_seed_all(42)          # all GPUs

# Full reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Saving and Restoring State

```python
state = torch.get_rng_state()
# ... generate random tensors ...
torch.set_rng_state(state)               # restore to saved state
```

### Independent Generators

```python
gen = torch.Generator()
gen.manual_seed(999)

tensor_a = torch.randn(2, 2, generator=gen)

gen.manual_seed(999)
tensor_b = torch.randn(2, 2, generator=gen)

torch.equal(tensor_a, tensor_b)           # True
```

### Weight Initialization Patterns

```python
# Xavier/Glorot uniform
def xavier_uniform(tensor):
    fan_in, fan_out = tensor.size(-1), tensor.size(0)
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    tensor.uniform_(-bound, bound)

# He/Kaiming normal
def he_normal(tensor):
    fan_in = tensor.size(-1)
    std = (2.0 / fan_in) ** 0.5
    tensor.normal_(0, std)
```

### Shuffling

```python
data = torch.arange(100)
shuffled = data[torch.randperm(len(data))]
```
