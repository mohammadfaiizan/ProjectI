#!/usr/bin/env python3
"""PyTorch Linear Algebra Operations - Matrix operations, decompositions, solving"""

import torch

print("=== Basic Matrix Operations ===")

# Create sample matrices
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
B = torch.tensor([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 4.0]])
v = torch.tensor([1.0, 2.0, 3.0])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"Vector v: {v}")

# Matrix multiplication
matmul_AB = torch.matmul(A, B)
mm_AB = torch.mm(A, B)
bmm_result = torch.bmm(A.unsqueeze(0), B.unsqueeze(0))

print(f"A @ B (matmul):\n{matmul_AB}")
print(f"A @ B (mm):\n{mm_AB}")
print(f"Results equal: {torch.equal(matmul_AB, mm_AB)}")
print(f"Batch matmul shape: {bmm_result.shape}")

# Matrix-vector multiplication
mv_result = torch.mv(A, v)
matmul_mv = torch.matmul(A, v)

print(f"A @ v (mv): {mv_result}")
print(f"A @ v (matmul): {matmul_mv}")
print(f"Results equal: {torch.equal(mv_result, matmul_mv)}")

print("\n=== Dot and Cross Products ===")

# Vector operations
u = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([4.0, 5.0, 6.0])

# Dot product
dot_product = torch.dot(u, w)
manual_dot = torch.sum(u * w)

print(f"Vector u: {u}")
print(f"Vector w: {w}")
print(f"Dot product: {dot_product}")
print(f"Manual dot: {manual_dot}")

# Cross product (3D vectors only)
cross_product = torch.cross(u, w)
print(f"Cross product: {cross_product}")

# Outer product
outer_product = torch.outer(u, w)
print(f"Outer product shape: {outer_product.shape}")
print(f"Outer product:\n{outer_product}")

print("\n=== Matrix Properties ===")

# Matrix transpose
A_transpose = A.t()
A_transpose_T = A.T

print(f"A transpose:\n{A_transpose}")
print(f"A.T equal A.t(): {torch.equal(A_transpose, A_transpose_T)}")

# Matrix trace
trace_A = torch.trace(A)
manual_trace = torch.sum(torch.diag(A))

print(f"Trace of A: {trace_A}")
print(f"Manual trace: {manual_trace}")

# Matrix diagonal
diag_A = torch.diag(A)
print(f"Diagonal of A: {diag_A}")

# Create diagonal matrix
diag_matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
print(f"Diagonal matrix:\n{diag_matrix}")

print("\n=== Matrix Norms ===")

# Different matrix norms
frobenius_norm = torch.norm(A, p='fro')
nuclear_norm = torch.norm(A, p='nuc')
spectral_norm = torch.norm(A, p=2)

print(f"Frobenius norm: {frobenius_norm}")
print(f"Nuclear norm: {nuclear_norm}")
print(f"Spectral norm: {spectral_norm}")

# Vector norms
l1_norm = torch.norm(v, p=1)
l2_norm = torch.norm(v, p=2)
inf_norm = torch.norm(v, p=float('inf'))

print(f"Vector L1 norm: {l1_norm}")
print(f"Vector L2 norm: {l2_norm}")
print(f"Vector infinity norm: {inf_norm}")

print("\n=== Matrix Decompositions ===")

# Create a proper matrix for decompositions
M = torch.randn(4, 4)
M_symmetric = M + M.t()  # Make symmetric
M_spd = M_symmetric + 4 * torch.eye(4)  # Make positive definite

print(f"Matrix for decomposition shape: {M.shape}")

# QR decomposition
Q, R = torch.qr(M)
print(f"QR decomposition - Q shape: {Q.shape}, R shape: {R.shape}")
print(f"QR reconstruction error: {torch.norm(M - Q @ R):.6f}")

# SVD decomposition
U, S, Vh = torch.svd(M)
print(f"SVD - U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
print(f"SVD reconstruction error: {torch.norm(M - U @ torch.diag(S) @ Vh.t()):.6f}")

# Eigenvalue decomposition (for symmetric matrices)
eigenvals, eigenvecs = torch.symeig(M_symmetric, eigenvectors=True)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors shape: {eigenvecs.shape}")

# Cholesky decomposition (for positive definite matrices)
try:
    L = torch.cholesky(M_spd)
    print(f"Cholesky factor shape: {L.shape}")
    print(f"Cholesky reconstruction error: {torch.norm(M_spd - L @ L.t()):.6f}")
except RuntimeError as e:
    print(f"Cholesky failed: {str(e)[:50]}...")

print("\n=== Matrix Inverse and Pseudoinverse ===")

# Matrix inverse
invertible_matrix = torch.randn(3, 3)
# Make sure it's invertible by adding to diagonal
invertible_matrix += 2 * torch.eye(3)

try:
    matrix_inv = torch.inverse(invertible_matrix)
    print(f"Matrix inverse shape: {matrix_inv.shape}")
    
    # Verify inverse
    identity_check = invertible_matrix @ matrix_inv
    print(f"A @ A^(-1) close to identity: {torch.allclose(identity_check, torch.eye(3))}")
    
except RuntimeError as e:
    print(f"Inverse failed: {e}")

# Pseudoinverse for non-square matrices
non_square = torch.randn(5, 3)
pinv = torch.pinverse(non_square)
print(f"Pseudoinverse shape: {pinv.shape}")

# Moore-Penrose conditions
print(f"Pseudoinverse condition check: {torch.allclose(non_square @ pinv @ non_square, non_square)}")

print("\n=== Solving Linear Systems ===")

# Solve Ax = b
A_square = torch.randn(4, 4) + 2 * torch.eye(4)  # Well-conditioned
b_vec = torch.randn(4)

# Solve linear system
try:
    x_solution = torch.solve(b_vec, A_square)[0]
    print(f"Solution x shape: {x_solution.shape}")
    
    # Verify solution
    residual = torch.norm(A_square @ x_solution - b_vec)
    print(f"Residual |Ax - b|: {residual:.6f}")
    
except RuntimeError as e:
    print(f"Solve failed: {e}")

# Solve multiple right-hand sides
B_matrix = torch.randn(4, 3)
X_solutions = torch.solve(B_matrix, A_square)[0]
print(f"Multiple solutions shape: {X_solutions.shape}")

print("\n=== Matrix Determinant ===")

# Determinant calculation
det_A = torch.det(A_square)
log_det = torch.logdet(A_square)
sign, log_abs_det = torch.slogdet(A_square)

print(f"Determinant: {det_A}")
print(f"Log determinant: {log_det}")
print(f"Sign and log absolute determinant: {sign}, {log_abs_det}")

# Determinant properties
det_transpose = torch.det(A_square.t())
print(f"det(A) = det(A^T): {torch.allclose(det_A, det_transpose)}")

print("\n=== Matrix Powers ===")

# Matrix power
symmetric_matrix = torch.randn(3, 3)
symmetric_matrix = symmetric_matrix + symmetric_matrix.t()

matrix_squared = torch.matrix_power(symmetric_matrix, 2)
manual_squared = symmetric_matrix @ symmetric_matrix

print(f"Matrix power shape: {matrix_squared.shape}")
print(f"Matrix power equals manual: {torch.allclose(matrix_squared, manual_squared)}")

# Fractional powers using eigendecomposition
eigenvals, eigenvecs = torch.symeig(symmetric_matrix, eigenvectors=True)
# Make positive definite for fractional powers
eigenvals = torch.abs(eigenvals) + 0.1
sqrt_eigenvals = torch.sqrt(eigenvals)
matrix_sqrt = eigenvecs @ torch.diag(sqrt_eigenvals) @ eigenvecs.t()

print(f"Matrix square root shape: {matrix_sqrt.shape}")

print("\n=== Batch Linear Algebra ===")

# Batch operations
batch_matrices = torch.randn(10, 4, 4)
batch_vectors = torch.randn(10, 4)

# Batch matrix multiplication
batch_A = torch.randn(10, 4, 3)
batch_B = torch.randn(10, 3, 5)
batch_result = torch.bmm(batch_A, batch_B)
print(f"Batch matmul shape: {batch_result.shape}")

# Batch determinant
batch_dets = torch.det(batch_matrices)
print(f"Batch determinants shape: {batch_dets.shape}")

# Batch inverse
try:
    # Make well-conditioned
    batch_matrices_stable = batch_matrices + 2 * torch.eye(4).unsqueeze(0)
    batch_inverses = torch.inverse(batch_matrices_stable)
    print(f"Batch inverses shape: {batch_inverses.shape}")
except RuntimeError as e:
    print(f"Batch inverse failed: {e}")

print("\n=== Advanced Linear Algebra ===")

# Matrix exponential approximation
def matrix_exp_approximation(A, n_terms=10):
    """Approximate matrix exponential using Taylor series"""
    result = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
    term = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
    
    for i in range(1, n_terms):
        term = term @ A / i
        result += term
    
    return result

small_matrix = torch.randn(3, 3) * 0.1  # Small values for convergence
approx_exp = matrix_exp_approximation(small_matrix)
print(f"Matrix exponential approximation shape: {approx_exp.shape}")

# Matrix logarithm using eigendecomposition
def matrix_log_symmetric(A):
    """Matrix logarithm for symmetric positive definite matrix"""
    eigenvals, eigenvecs = torch.symeig(A, eigenvectors=True)
    # Ensure positive eigenvalues
    eigenvals = torch.abs(eigenvals) + 1e-8
    log_eigenvals = torch.log(eigenvals)
    return eigenvecs @ torch.diag(log_eigenvals) @ eigenvecs.t()

spd_matrix = symmetric_matrix + 3 * torch.eye(3)
matrix_log = matrix_log_symmetric(spd_matrix)
print(f"Matrix logarithm shape: {matrix_log.shape}")

print("\n=== Condition Numbers and Stability ===")

# Condition number
def condition_number(A):
    """Compute condition number using SVD"""
    _, S, _ = torch.svd(A)
    return S[0] / S[-1]

cond_num = condition_number(A_square)
print(f"Condition number: {cond_num}")

# Well-conditioned vs ill-conditioned
well_conditioned = torch.eye(4) + 0.1 * torch.randn(4, 4)
ill_conditioned = torch.randn(4, 4)
ill_conditioned[0] = ill_conditioned[1] * 1e-10  # Make nearly singular

cond_well = condition_number(well_conditioned)
cond_ill = condition_number(ill_conditioned)

print(f"Well-conditioned matrix condition: {cond_well:.2f}")
print(f"Ill-conditioned matrix condition: {cond_ill:.2e}")

print("\n=== Rank and Nullspace ===")

# Matrix rank using SVD
def matrix_rank(A, tol=1e-8):
    """Compute matrix rank"""
    _, S, _ = torch.svd(A)
    return torch.sum(S > tol).item()

rank_A = matrix_rank(A)
rank_identity = matrix_rank(torch.eye(4))

print(f"Rank of A: {rank_A}")
print(f"Rank of identity: {rank_identity}")

# Low-rank approximation
U, S, Vh = torch.svd(A)
rank_k = 2
A_approx = U[:, :rank_k] @ torch.diag(S[:rank_k]) @ Vh[:rank_k, :]
print(f"Low-rank approximation error: {torch.norm(A - A_approx):.6f}")

print("\n=== Tensor Contractions ===")

# Einstein summation for tensor contractions
tensor_3d = torch.randn(3, 4, 5)
tensor_2d = torch.randn(5, 6)

# Matrix multiplication using einsum
einsum_result = torch.einsum('ij,jk->ik', A_square, B_matrix)
matmul_result = A_square @ B_matrix
print(f"Einsum equals matmul: {torch.allclose(einsum_result, matmul_result)}")

# Tensor contraction
contraction = torch.einsum('ijk,kl->ijl', tensor_3d, tensor_2d)
print(f"Tensor contraction shape: {contraction.shape}")

# Batch matrix multiplication with einsum
batch_einsum = torch.einsum('bij,bjk->bik', batch_A, batch_B)
print(f"Batch einsum shape: {batch_einsum.shape}")

print("\n=== Performance Considerations ===")

import time

# Performance comparison
large_A = torch.randn(1000, 1000)
large_B = torch.randn(1000, 1000)

# Matrix multiplication performance
start_time = time.time()
result_mm = torch.mm(large_A, large_B)
mm_time = time.time() - start_time

start_time = time.time()
result_matmul = torch.matmul(large_A, large_B)
matmul_time = time.time() - start_time

print(f"mm time: {mm_time:.6f} seconds")
print(f"matmul time: {matmul_time:.6f} seconds")
print(f"Results equal: {torch.equal(result_mm, result_matmul)}")

# In-place operations for memory efficiency
large_C = torch.randn(1000, 1000)
original_ptr = large_C.data_ptr()

# In-place matrix addition
large_C.add_(large_A)
print(f"In-place operation same memory: {large_C.data_ptr() == original_ptr}")

print("\n=== Linear Algebra Complete ===") 