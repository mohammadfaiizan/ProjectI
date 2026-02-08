# Linear Algebra, FFT, Random, and Polynomials in NumPy

## Table of Contents
1. [Linear Algebra (np.linalg)](#linear-algebra-nplinalg)
2. [Fast Fourier Transform (np.fft)](#fast-fourier-transform-npfft)
3. [Random Number Generation (np.random)](#random-number-generation-nprandom)
4. [Polynomials (np.polynomial)](#polynomials-nppolynomial)

---

## Linear Algebra (np.linalg)

NumPy's linear algebra module (`np.linalg`) provides comprehensive tools for matrix operations, decompositions, and solving linear systems.

### Matrix Multiplication

NumPy offers multiple ways to perform matrix multiplication, each with specific use cases:

#### Dot Product and Matrix Multiplication

```python
import numpy as np

# Standard dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)  # Standard matrix multiplication
D = np.matmul(A, B)  # Explicit matrix multiplication
E = A @ B  # Python 3.5+ infix operator (recommended)

# All three produce the same result for 2D arrays
```

**Key Differences:**
- `np.dot`: General dot product, handles 1D vectors and 2D matrices
- `np.matmul`: Explicitly for matrix multiplication, better broadcasting behavior
- `@` operator: Syntactic sugar for `np.matmul`, preferred in modern code

#### Inner and Outer Products

```python
# Inner product (same as dot for 1D vectors)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
inner = np.inner(a, b)  # Same as np.dot(a, b) for 1D

# Outer product
outer = np.outer(a, b)
# Result: [[4, 5, 6],
#          [8, 10, 12],
#          [12, 15, 18]]

# For matrices, inner computes sum of element-wise products
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
inner_matrix = np.inner(A, B)  # Computes inner product along last axis
```

#### Tensor Dot Product

```python
# Tensor contraction along specified axes
a = np.random.rand(3, 4, 5)
b = np.random.rand(4, 3, 2)
result = np.tensordot(a, b, axes=([1, 0], [0, 1]))  # Contract axes 1,0 of a with 0,1 of b
```

#### Kronecker Product

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
K = np.kron(A, B)
# Result: [[5, 6, 10, 12],
#          [7, 8, 14, 16],
#          [15, 18, 20, 24],
#          [21, 24, 28, 32]]
```

### Matrix Properties

#### Determinant

```python
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)  # -2.0

# For singular matrices, determinant is near zero
B = np.array([[1, 2], [2, 4]])
det_singular = np.linalg.det(B)  # Approximately 0.0
```

#### Matrix Rank

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank = np.linalg.matrix_rank(A)  # 2 (linearly dependent rows)

# With tolerance
rank_tol = np.linalg.matrix_rank(A, tol=1e-5)
```

#### Trace

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
trace = np.trace(A)  # 15 (sum of diagonal elements)

# Trace of matrix product: trace(AB) = trace(BA)
```

#### Matrix Norms

```python
A = np.array([[1, 2], [3, 4]])

# Vector norms (when A is treated as vector)
norm_frobenius = np.linalg.norm(A, 'fro')  # Frobenius norm
norm_2 = np.linalg.norm(A, 2)  # Largest singular value
norm_1 = np.linalg.norm(A, 1)  # Max column sum
norm_inf = np.linalg.norm(A, np.inf)  # Max row sum

# For vectors
v = np.array([3, 4])
norm_euclidean = np.linalg.norm(v)  # Default: L2 norm = 5.0
norm_l1 = np.linalg.norm(v, 1)  # L1 norm = 7
```

#### Condition Number

```python
A = np.array([[1, 2], [3, 4]])
cond = np.linalg.cond(A)  # Condition number (ratio of largest to smallest singular value)
# Large condition number indicates near-singularity
```

### Matrix Operations

#### Matrix Inverse

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

# Verify: A @ A_inv should be identity matrix
identity_check = A @ A_inv  # Approximately [[1, 0], [0, 1]]

# For singular matrices, use pseudo-inverse
B = np.array([[1, 2], [2, 4]])  # Singular
B_pinv = np.linalg.pinv(B)  # Moore-Penrose pseudo-inverse
```

#### Matrix Power

```python
A = np.array([[1, 2], [3, 4]])
A_squared = np.linalg.matrix_power(A, 2)  # A @ A
A_cubed = np.linalg.matrix_power(A, 3)  # A @ A @ A
A_inverse = np.linalg.matrix_power(A, -1)  # Same as np.linalg.inv(A)
```

### Matrix Decompositions

#### Eigenvalue Decomposition

```python
A = np.array([[1, 2], [2, 1]])

# Standard eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues: array of eigenvalues
# eigenvectors: columns are eigenvectors

# For symmetric/Hermitian matrices (more efficient)
A_symmetric = np.array([[1, 2], [2, 1]])
eigenvalues_sym, eigenvectors_sym = np.linalg.eigh(A_symmetric)

# Eigenvalues only (faster if eigenvectors not needed)
eigenvals_only = np.linalg.eigvals(A)
```

**Properties:**
- For matrix A, if λ is eigenvalue and v is eigenvector: Av = λv
- Eigenvalues may be complex even for real matrices
- `eigh` is faster and more stable for symmetric matrices

#### Singular Value Decomposition (SVD)

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# Full SVD: A = U @ S @ V^T
U, s, Vt = np.linalg.svd(A, full_matrices=True)
# U: left singular vectors (m x m)
# s: singular values (1D array)
# Vt: right singular vectors transposed (n x n)

# Reduced SVD (more common): A = U @ S @ V^T
U_red, s_red, Vt_red = np.linalg.svd(A, full_matrices=False)
# U_red: (m x min(m,n))
# Vt_red: (min(m,n) x n)

# Reconstruct matrix
S = np.diag(s_red)
A_reconstructed = U_red @ S @ Vt_red
```

**Applications:**
- Principal Component Analysis (PCA)
- Low-rank approximations
- Solving least squares problems
- Image compression

#### QR Decomposition

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# QR decomposition: A = Q @ R
Q, R = np.linalg.qr(A)
# Q: orthogonal matrix (Q^T @ Q = I)
# R: upper triangular matrix

# Verify: Q @ R should equal A
A_reconstructed = Q @ R
```

**Applications:**
- Solving least squares problems
- Finding orthogonal bases
- Gram-Schmidt orthogonalization

#### Cholesky Decomposition

```python
# For positive definite matrices: A = L @ L^T
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
L = np.linalg.cholesky(A)

# Verify: L @ L.T should equal A
A_reconstructed = L @ L.T
```

**Requirements:**
- Matrix must be positive definite
- Faster than general LU decomposition
- Used in solving systems with positive definite matrices

#### LU Decomposition

```python
# Note: NumPy doesn't have direct LU decomposition
# Use scipy.linalg.lu instead:
from scipy.linalg import lu
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
P, L, U = lu(A)
# P: permutation matrix
# L: lower triangular matrix
# U: upper triangular matrix
# A = P @ L @ U
```

### Solving Linear Systems

#### Direct Solution

```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)  # [2., 3.]

# Verify: A @ x should equal b
verification = A @ x  # Approximately [9, 8]
```

#### Least Squares Solution

```python
# For overdetermined or underdetermined systems: minimize ||Ax - b||^2
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([2, 3, 4])
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# x: least squares solution
# residuals: sum of squared residuals
# rank: rank of matrix A
# s: singular values of A
```

#### Tensor Solve

```python
# Solve tensor equation: A @ x = B
A = np.random.rand(3, 3, 3)
B = np.random.rand(3, 3)
x = np.linalg.tensorsolve(A, B)
```

### Special Matrix Operations

#### Optimized Chain Multiplication

```python
# Multi-dot: optimized order for chain multiplication
A = np.random.rand(10, 20)
B = np.random.rand(20, 30)
C = np.random.rand(30, 5)
D = np.random.rand(5, 10)

# Instead of: A @ B @ C @ D (may not be optimal order)
result = np.linalg.multi_dot([A, B, C, D])
# Automatically determines optimal multiplication order
```

### Einstein Summation (np.einsum)

Einstein summation convention provides a powerful way to express tensor operations:

```python
# Trace: sum of diagonal elements
A = np.array([[1, 2], [3, 4]])
trace = np.einsum('ii', A)  # Same as np.trace(A)

# Transpose
transpose = np.einsum('ij->ji', A)

# Matrix multiplication: C_ij = A_ik * B_kj
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
C = np.einsum('ik,kj->ij', A, B)  # Same as A @ B

# Batch matrix multiplication
A = np.random.rand(10, 3, 4)
B = np.random.rand(10, 4, 5)
C = np.einsum('bij,bjk->bik', A, B)  # Batch dimension preserved

# Outer product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
outer = np.einsum('i,j->ij', a, b)

# Element-wise multiplication and sum
A = np.random.rand(3, 4)
B = np.random.rand(3, 4)
result = np.einsum('ij,ij->', A, B)  # Sum of element-wise products

# Diagonal extraction
diag = np.einsum('ii->i', A)

# Sum along specific axes
row_sum = np.einsum('ij->i', A)  # Sum along columns
col_sum = np.einsum('ij->j', A)  # Sum along rows
```

**Advantages:**
- Expressive and readable for complex operations
- Can be more efficient than multiple operations
- Supports broadcasting automatically

---

## Fast Fourier Transform (np.fft)

The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform (DFT), essential for signal processing, image analysis, and frequency domain operations.

### One-Dimensional FFT

#### Forward and Inverse FFT

```python
import numpy as np

# Generate a signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# Forward FFT: time domain -> frequency domain
fft_result = np.fft.fft(signal)

# Inverse FFT: frequency domain -> time domain
reconstructed = np.fft.ifft(fft_result)

# Verify: should recover original signal (within numerical precision)
np.allclose(signal, reconstructed)  # True
```

**Key Points:**
- `fft` returns complex array of same length as input
- `ifft` is the inverse operation
- FFT assumes periodic signal (wraps around)

### Two-Dimensional FFT

```python
# 2D FFT (useful for images)
image = np.random.rand(100, 100)
fft_2d = np.fft.fft2(image)
reconstructed_2d = np.fft.ifft2(fft_2d)
```

### N-Dimensional FFT

```python
# N-D FFT
data = np.random.rand(10, 20, 30)
fft_nd = np.fft.fftn(data)
reconstructed_nd = np.fft.ifftn(data)
```

### Real-Valued Input Optimization

For real-valued inputs, use `rfft` which is faster and uses half the memory:

```python
# Real FFT (input must be real-valued)
real_signal = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
fft_real = np.fft.rfft(real_signal)  # Only positive frequencies
reconstructed_real = np.fft.irfft(fft_real, n=1000)  # Specify original length
```

**Differences:**
- `rfft`: Returns only positive frequencies (half the size)
- `irfft`: Reconstructs real signal from positive frequencies only
- More efficient for real-valued data

### Frequency Bins and Shifting

#### Frequency Bins

```python
# Get frequency bins for FFT output
N = 1000  # Number of samples
sampling_rate = 1000  # Hz
frequencies = np.fft.fftfreq(N, 1/sampling_rate)
# Returns frequencies: [-500, -499, ..., -1, 0, 1, ..., 499]

# For real FFT
frequencies_real = np.fft.rfftfreq(N, 1/sampling_rate)
# Returns only positive frequencies: [0, 1, 2, ..., 500]
```

#### FFT Shift

```python
# Shift zero frequency to center
fft_result = np.fft.fft(signal)
fft_shifted = np.fft.fftshift(fft_result)  # DC component at center

# Inverse shift
fft_unshifted = np.fft.ifftshift(fft_shifted)

# For 2D
fft_2d_shifted = np.fft.fftshift(fft_2d)
```

**Use Case:** Visualizing frequency domain with zero frequency at center

### Practical Example: Frequency Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate signal with multiple frequencies
sampling_rate = 1000
duration = 1.0
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Signal: 50 Hz + 120 Hz components
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# Add noise
noise = 0.1 * np.random.randn(len(signal))
signal_noisy = signal + noise

# Compute FFT
fft_values = np.fft.fft(signal_noisy)
frequencies = np.fft.fftfreq(len(signal_noisy), 1/sampling_rate)

# Get magnitude spectrum
magnitude = np.abs(fft_values)
power_spectrum = magnitude ** 2

# Find dominant frequencies
positive_freq_idx = frequencies > 0
dominant_freqs = frequencies[positive_freq_idx][np.argsort(magnitude[positive_freq_idx])[-2:]]

print(f"Dominant frequencies: {dominant_freqs} Hz")  # Should be ~50 and ~120 Hz
```

---

## Random Number Generation (np.random)

NumPy provides comprehensive random number generation capabilities for simulations, sampling, and statistical analysis.

### Legacy vs New Generator API

#### Legacy API (Deprecated)

```python
# Old style (still works but deprecated)
np.random.seed(42)
values = np.random.rand(10)  # Uniform [0, 1)
normal = np.random.randn(10)  # Standard normal
```

#### New Generator API (Recommended)

```python
# New Generator API (recommended)
rng = np.random.default_rng(42)  # Create generator with seed
values = rng.random(10)  # Uniform [0, 1)
normal = rng.standard_normal(10)  # Standard normal
```

**Advantages of New API:**
- Better statistical properties
- More consistent interface
- Better performance
- More reproducible

### Seeds and Reproducibility

```python
# Basic seeding
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)
# Both generators produce same sequence

# SeedSequence for advanced control
from numpy.random import SeedSequence
seed_seq = SeedSequence(42)
rng = np.random.default_rng(seed_seq)

# Spawn for parallel reproducibility
seed_seq = SeedSequence(42)
children = seed_seq.spawn(4)  # Create 4 independent generators
rngs = [np.random.default_rng(child) for child in children]
```

### Probability Distributions

#### Uniform Distribution

```python
rng = np.random.default_rng(42)

# Uniform [0, 1)
uniform = rng.random(10)

# Uniform [low, high)
uniform_range = rng.uniform(low=0, high=10, size=10)
```

#### Normal/Gaussian Distribution

```python
# Standard normal (mean=0, std=1)
normal = rng.standard_normal(10)

# Normal with specified mean and std
normal_custom = rng.normal(loc=5, scale=2, size=10)
```

#### Binomial Distribution

```python
# Binomial: n trials, p probability
# Returns number of successes
successes = rng.binomial(n=10, p=0.5, size=100)
# Simulates 100 experiments of 10 coin flips
```

#### Poisson Distribution

```python
# Poisson: lambda (rate parameter)
events = rng.poisson(lam=5, size=100)
# Simulates 100 time periods with average 5 events
```

#### Exponential Distribution

```python
# Exponential: scale parameter (1/lambda)
wait_times = rng.exponential(scale=2.0, size=100)
```

#### Beta Distribution

```python
# Beta: shape parameters alpha and beta
beta_samples = rng.beta(a=2, b=5, size=100)
```

#### Gamma Distribution

```python
# Gamma: shape and scale parameters
gamma_samples = rng.gamma(shape=2, scale=2, size=100)
```

#### Chi-Square Distribution

```python
# Chi-square: degrees of freedom
chi2_samples = rng.chisquare(df=5, size=100)
```

#### Multinomial Distribution

```python
# Multinomial: n trials, p probabilities
# Returns counts for each category
counts = rng.multinomial(n=20, pvals=[0.3, 0.5, 0.2], size=10)
# 10 experiments, 20 trials each, 3 categories
```

#### Multivariate Normal

```python
# Multivariate normal: mean vector and covariance matrix
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])
mv_samples = rng.multivariate_normal(mean=mean, cov=cov, size=100)
```

### Random Sampling

#### Choice and Shuffle

```python
rng = np.random.default_rng(42)

# Random choice from array
choices = rng.choice(['a', 'b', 'c'], size=10, p=[0.5, 0.3, 0.2])

# Random integers
integers = rng.integers(low=0, high=10, size=10)

# Shuffle array in-place
arr = np.array([1, 2, 3, 4, 5])
rng.shuffle(arr)  # arr is modified

# Permutation (returns new array)
arr = np.array([1, 2, 3, 4, 5])
permuted = rng.permutation(arr)  # arr unchanged
```

### Practical Examples

#### Monte Carlo Simulation

```python
# Estimate π using Monte Carlo
rng = np.random.default_rng(42)
n_samples = 1_000_000

# Generate random points in unit square
x = rng.random(n_samples)
y = rng.random(n_samples)

# Count points inside unit circle
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * np.sum(inside) / n_samples
print(f"π estimate: {pi_estimate}")  # Should be close to 3.14159...
```

#### Bootstrap Sampling

```python
# Bootstrap resampling for confidence intervals
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
rng = np.random.default_rng(42)

n_bootstrap = 1000
bootstrap_means = []

for _ in range(n_bootstrap):
    # Resample with replacement
    sample = rng.choice(data, size=len(data), replace=True)
    bootstrap_means.append(np.mean(sample))

# Compute confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

---

## Polynomials (np.polynomial)

NumPy provides tools for working with polynomials, including both legacy and modern APIs.

### Legacy Polynomial API

#### Creating Polynomials

```python
# Legacy API: np.poly1d
# Coefficients in descending order: a*x^2 + b*x + c
coeffs = [1, 2, 3]  # Represents x^2 + 2x + 3
p = np.poly1d(coeffs)

# Evaluate polynomial
result = p(5)  # 5^2 + 2*5 + 3 = 38

# Polynomial operations
p1 = np.poly1d([1, 2])
p2 = np.poly1d([3, 4])
p_sum = p1 + p2
p_product = p1 * p2
```

#### Polynomial Fitting

```python
# Fit polynomial to data
x = np.linspace(0, 10, 100)
y = x**2 + 2*x + 3 + np.random.randn(100) * 0.1

# Fit degree-2 polynomial
coeffs = np.polyfit(x, y, deg=2)
p_fitted = np.poly1d(coeffs)

# Evaluate fitted polynomial
y_pred = np.polyval(coeffs, x)
```

#### Polynomial Derivatives and Integrals

```python
p = np.poly1d([1, 2, 3])  # x^2 + 2x + 3

# Derivative
p_deriv = np.polyder(p)  # 2x + 2

# Integral (with constant term)
p_int = np.polyint(p)  # x^3/3 + x^2 + 3x + C
p_int_c = np.polyint(p, k=5)  # With constant = 5
```

#### Finding Roots

```python
# Find roots of polynomial
p = np.poly1d([1, -5, 6])  # x^2 - 5x + 6
roots = np.roots(p)  # [3., 2.] (roots of x^2 - 5x + 6 = 0)
```

### Modern Polynomial API

The modern API provides more consistent interfaces and supports different polynomial bases:

#### Standard Polynomial

```python
from numpy.polynomial import polynomial as P

# Coefficients in ascending order: c[0] + c[1]*x + c[2]*x^2
coeffs = [3, 2, 1]  # Represents 3 + 2x + x^2

# Evaluate
result = P.polyval(x, coeffs)

# Fit polynomial
x = np.linspace(0, 10, 100)
y = 3 + 2*x + x**2 + np.random.randn(100) * 0.1
coeffs_fit = P.polyfit(x, y, deg=2)

# Polynomial arithmetic
p1_coeffs = [1, 2]
p2_coeffs = [3, 4]
p_sum_coeffs = P.polyadd(p1_coeffs, p2_coeffs)
p_prod_coeffs = P.polymul(p1_coeffs, p2_coeffs)

# Derivative and integral
p_coeffs = [3, 2, 1]
p_deriv_coeffs = P.polyder(p_coeffs)
p_int_coeffs = P.polyint(p_coeffs)
```

#### Chebyshev Polynomials

```python
from numpy.polynomial import chebyshev as C

# Chebyshev basis: better numerical properties
coeffs = [1, 2, 3]
result = C.chebval(x, coeffs)

# Convert from standard to Chebyshev basis
std_coeffs = [1, 2, 3]
cheb_coeffs = C.poly2cheb(std_coeffs)
```

#### Legendre Polynomials

```python
from numpy.polynomial import legendre as L

# Legendre basis
coeffs = [1, 2, 3]
result = L.legval(x, coeffs)
```

### Polynomial Fitting Example

```python
import numpy as np
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as plt

# Generate noisy data
x = np.linspace(0, 10, 50)
y_true = 2 + 3*x - 0.5*x**2 + 0.1*x**3
y_noisy = y_true + np.random.randn(50) * 2

# Fit polynomials of different degrees
degrees = [1, 2, 3, 5, 10]
plt.figure(figsize=(12, 8))
plt.scatter(x, y_noisy, alpha=0.5, label='Data')

x_fine = np.linspace(0, 10, 200)
for deg in degrees:
    coeffs = P.polyfit(x, y_noisy, deg=deg)
    y_fit = P.polyval(x_fine, coeffs)
    plt.plot(x_fine, y_fit, label=f'Degree {deg}')

plt.plot(x_fine, y_true, 'k--', label='True', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fitting')
plt.show()
```

**Key Considerations:**
- Higher degree polynomials can overfit
- Chebyshev polynomials have better numerical stability
- Modern API uses ascending order coefficients (opposite of legacy)

---

## Summary

This module covered:

1. **Linear Algebra**: Matrix operations, decompositions (eigenvalue, SVD, QR, Cholesky), solving systems, and Einstein summation
2. **FFT**: One-dimensional, multi-dimensional, real-valued optimizations, and frequency analysis
3. **Random Generation**: Modern Generator API, various distributions, and practical applications
4. **Polynomials**: Legacy and modern APIs, fitting, evaluation, and different polynomial bases

These tools form the foundation for numerical computing, signal processing, statistical analysis, and machine learning applications.
