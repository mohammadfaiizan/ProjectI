"""
NumPy Linear Algebra, FFT, Random and Polynomials
Comprehensive examples of matrix operations, decompositions, solving systems,
einsum, FFT operations, random number generation, and polynomial operations.
"""

import numpy as np

print("=" * 80)
print("FILE 4: LINEAR ALGEBRA, FFT, RANDOM AND POLYNOMIALS")
print("=" * 80)

# ============================================================================
# Matrix Multiplication
# ============================================================================

print("\n--- Matrix Multiplication ---\n")

MatrixA = np.array([[1, 2], [3, 4]])
MatrixB = np.array([[5, 6], [7, 8]])

# dot product
DotResult = np.dot(MatrixA, MatrixB)
print(f"Matrix A:\n{MatrixA}")
print(f"\nMatrix B:\n{MatrixB}")
print(f"\nnp.dot(A, B):\n{DotResult}")

# matmul
MatMulResult = np.matmul(MatrixA, MatrixB)
print(f"\nnp.matmul(A, B):\n{MatMulResult}")

# @ operator (Python 3.5+)
AtOperatorResult = MatrixA @ MatrixB
print(f"\nA @ B:\n{AtOperatorResult}")

# inner product
VectorA = np.array([1, 2, 3])
VectorB = np.array([4, 5, 6])
InnerResult = np.inner(VectorA, VectorB)
print(f"\nVector A: {VectorA}")
print(f"Vector B: {VectorB}")
print(f"inner product: {InnerResult}")

# outer product
OuterResult = np.outer(VectorA, VectorB)
print(f"\nouter product:\n{OuterResult}")

# tensordot
TensorA = np.array([[1, 2], [3, 4]])
TensorB = np.array([[5, 6], [7, 8]])
TensorDotResult = np.tensordot(TensorA, TensorB, axes=1)
print(f"\ntensordot(axes=1):\n{TensorDotResult}")

# kron - Kronecker product
KronResult = np.kron(MatrixA, MatrixB)
print(f"\nkron product shape: {KronResult.shape}")

# ============================================================================
# Matrix Properties
# ============================================================================

print("\n\n--- Matrix Properties ---\n")

SquareMatrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Determinant
DetResult = np.linalg.det(SquareMatrix)
print(f"Matrix:\n{SquareMatrix}")
print(f"det: {DetResult:.4f}")

# Matrix rank
RankResult = np.linalg.matrix_rank(SquareMatrix)
print(f"matrix_rank: {RankResult}")

# Trace
TraceResult = np.trace(SquareMatrix)
TraceManual = np.sum(np.diag(SquareMatrix))
print(f"trace: {TraceResult} (manual: {TraceManual})")

# Norm
NormFrobenius = np.linalg.norm(SquareMatrix, ord='fro')
Norm2 = np.linalg.norm(SquareMatrix, ord=2)
NormInf = np.linalg.norm(SquareMatrix, ord=np.inf)
print(f"\nFrobenius norm: {NormFrobenius:.4f}")
print(f"2-norm: {Norm2:.4f}")
print(f"inf-norm: {NormInf:.4f}")

# Condition number
CondResult = np.linalg.cond(SquareMatrix)
print(f"condition number: {CondResult:.4f}")

# ============================================================================
# Matrix Operations
# ============================================================================

print("\n\n--- Matrix Operations ---\n")

InvertibleMatrix = np.array([[1, 2], [3, 4]])

# Inverse
InvResult = np.linalg.inv(InvertibleMatrix)
InvCheck = InvertibleMatrix @ InvResult
print(f"Matrix:\n{InvertibleMatrix}")
print(f"\ninverse:\n{InvResult}")
print(f"\nMatrix @ inverse (should be identity):\n{InvCheck}")

# Pseudo-inverse
PseudoInvResult = np.linalg.pinv(SquareMatrix)
print(f"\nPseudo-inverse shape: {PseudoInvResult.shape}")

# Matrix power
MatrixPower2 = np.linalg.matrix_power(InvertibleMatrix, 2)
MatrixPower3 = np.linalg.matrix_power(InvertibleMatrix, 3)
print(f"\nmatrix_power(2):\n{MatrixPower2}")
print(f"\nmatrix_power(3):\n{MatrixPower3}")

# ============================================================================
# Eigenvalue Decomposition
# ============================================================================

print("\n\n--- Eigenvalue Decomposition ---\n")

EigenMatrix = np.array([[4, 1], [2, 3]])

# eig - eigenvalues and eigenvectors
Eigenvalues, Eigenvectors = np.linalg.eig(EigenMatrix)
print(f"Matrix:\n{EigenMatrix}")
print(f"\nEigenvalues: {Eigenvalues}")
print(f"Eigenvectors:\n{Eigenvectors}")

# eigh - for Hermitian/symmetric matrices
SymmetricMatrix = np.array([[1, 2], [2, 1]])
EigenvaluesH, EigenvectorsH = np.linalg.eigh(SymmetricMatrix)
print(f"\nSymmetric matrix:\n{SymmetricMatrix}")
print(f"eigh eigenvalues: {EigenvaluesH}")
print(f"eigh eigenvectors:\n{EigenvectorsH}")

# eigvals - eigenvalues only
EigenvaluesOnly = np.linalg.eigvals(EigenMatrix)
print(f"\neigvals: {EigenvaluesOnly}")

# ============================================================================
# SVD - Singular Value Decomposition
# ============================================================================

print("\n\n--- SVD - Singular Value Decomposition ---\n")

SVDMatrix = np.array([[1, 2], [3, 4], [5, 6]])

# Full SVD
U, S, Vt = np.linalg.svd(SVDMatrix, full_matrices=True)
print(f"Matrix:\n{SVDMatrix}")
print(f"\nU shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vt shape: {Vt.shape}")

# Reduced SVD
U_red, S_red, Vt_red = np.linalg.svd(SVDMatrix, full_matrices=False)
print(f"\nReduced SVD:")
print(f"U_red shape: {U_red.shape}")
print(f"S_red: {S_red}")
print(f"Vt_red shape: {Vt_red.shape}")

# Reconstruct
Sigma = np.zeros((U_red.shape[0], Vt_red.shape[0]))
Sigma[:len(S_red), :len(S_red)] = np.diag(S_red)
Reconstructed = U_red @ Sigma @ Vt_red
print(f"\nReconstructed matrix:\n{Reconstructed}")

# ============================================================================
# QR Decomposition
# ============================================================================

print("\n\n--- QR Decomposition ---\n")

QRMatrix = np.array([[1, 2], [3, 4], [5, 6]])

Q, R = np.linalg.qr(QRMatrix)
print(f"Matrix:\n{QRMatrix}")
print(f"\nQ:\n{Q}")
print(f"\nR:\n{R}")
print(f"\nQ @ R (reconstruction):\n{Q @ R}")

# ============================================================================
# Cholesky Decomposition
# ============================================================================

print("\n\n--- Cholesky Decomposition ---\n")

# Positive definite matrix
PosDefMatrix = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

L = np.linalg.cholesky(PosDefMatrix)
Lt = L.T
ReconstructedChol = L @ Lt

print(f"Positive definite matrix:\n{PosDefMatrix}")
print(f"\nCholesky L:\n{L}")
print(f"\nL @ L.T (reconstruction):\n{ReconstructedChol}")

# ============================================================================
# Solving Systems
# ============================================================================

print("\n\n--- Solving Systems ---\n")

# solve - Ax = b
CoefficientMatrix = np.array([[3, 1], [1, 2]])
RightHandSide = np.array([9, 8])

Solution = np.linalg.solve(CoefficientMatrix, RightHandSide)
Verification = CoefficientMatrix @ Solution

print(f"A:\n{CoefficientMatrix}")
print(f"b: {RightHandSide}")
print(f"\nSolution x: {Solution}")
print(f"Verification A @ x: {Verification}")

# lstsq - least squares
OverdeterminedA = np.array([[1, 1], [1, 2], [1, 3]])
OverdeterminedB = np.array([2, 3, 4])

LeastSquaresResult = np.linalg.lstsq(OverdeterminedA, OverdeterminedB, rcond=None)
LSQSolution = LeastSquaresResult[0]
LSQResiduals = LeastSquaresResult[1]

print(f"\nOverdetermined system:")
print(f"A:\n{OverdeterminedA}")
print(f"b: {OverdeterminedB}")
print(f"Least squares solution: {LSQSolution}")
print(f"Residuals: {LSQResiduals}")

# ============================================================================
# np.einsum
# ============================================================================

print("\n\n--- np.einsum ---\n")

EinsumA = np.array([[1, 2], [3, 4]])
EinsumB = np.array([[5, 6], [7, 8]])

# Matrix multiplication
EinsumMatMul = np.einsum('ij,jk->ik', EinsumA, EinsumB)
print(f"A:\n{EinsumA}")
print(f"B:\n{EinsumB}")
print(f"\neinsum('ij,jk->ik', A, B):\n{EinsumMatMul}")

# Trace
EinsumTrace = np.einsum('ii->', EinsumA)
print(f"\neinsum('ii->', A) (trace): {EinsumTrace}")

# Diagonal
EinsumDiag = np.einsum('ii->i', EinsumA)
print(f"einsum('ii->i', A) (diagonal): {EinsumDiag}")

# Outer product
EinsumOuter = np.einsum('i,j->ij', VectorA, VectorB)
print(f"\neinsum('i,j->ij', a, b) (outer):\n{EinsumOuter}")

# Sum over axis
EinsumSum = np.einsum('ij->i', EinsumA)
print(f"\neinsum('ij->i', A) (sum over columns): {EinsumSum}")

# ============================================================================
# np.linalg.multi_dot
# ============================================================================

print("\n\n--- np.linalg.multi_dot ---\n")

MultiDotA = np.array([[1, 2], [3, 4]])
MultiDotB = np.array([[5, 6], [7, 8]])
MultiDotC = np.array([[9, 10], [11, 12]])

MultiDotResult = np.linalg.multi_dot([MultiDotA, MultiDotB, MultiDotC])
ManualMultiDot = MultiDotA @ MultiDotB @ MultiDotC

print(f"multi_dot([A, B, C]):\n{MultiDotResult}")
print(f"\nManual A @ B @ C:\n{ManualMultiDot}")

# ============================================================================
# FFT - Fast Fourier Transform
# ============================================================================

print("\n\n--- FFT - Fast Fourier Transform ---\n")

# Simple signal
Signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# fft - forward FFT
FFTResult = np.fft.fft(Signal)
print(f"Signal: {Signal}")
print(f"\nfft result (first few): {FFTResult[:4]}")

# ifft - inverse FFT
IFFTResult = np.fft.ifft(FFTResult)
print(f"\nifft result: {IFFTResult.real}")

# fft2 - 2D FFT
Signal2D = np.array([[1, 2], [3, 4]])
FFT2Result = np.fft.fft2(Signal2D)
print(f"\n2D signal:\n{Signal2D}")
print(f"\nfft2 result:\n{FFT2Result}")

# rfft - real FFT (more efficient for real inputs)
RealSignal = np.array([1.0, 2.0, 3.0, 4.0])
RFFTResult = np.fft.rfft(RealSignal)
print(f"\nReal signal: {RealSignal}")
print(f"rfft result: {RFFTResult}")

# fftfreq - frequency bins
FFTFreq = np.fft.fftfreq(8)
print(f"\nfftfreq(8): {FFTFreq}")

# fftshift - shift zero frequency to center
FFTShifted = np.fft.fftshift(FFTResult)
print(f"\nfftshift result (first few): {FFTShifted[:4]}")

# Signal example: sine wave
SampleRate = 1000
Duration = 1.0
Frequency = 50
Time = np.linspace(0, Duration, int(SampleRate * Duration))
SineWave = np.sin(2 * np.pi * Frequency * Time)
SineFFT = np.fft.fft(SineWave)
SineFreq = np.fft.fftfreq(len(SineWave), 1/SampleRate)
SineMagnitude = np.abs(SineFFT)

PeakIndex = np.argmax(SineMagnitude[:len(SineMagnitude)//2])
PeakFreq = abs(SineFreq[PeakIndex])
print(f"\nSine wave example:")
print(f"Frequency: {Frequency} Hz")
print(f"Detected peak frequency: {PeakFreq:.2f} Hz")

# ============================================================================
# Random Number Generation
# ============================================================================

print("\n\n--- Random Number Generation ---\n")

# Modern approach: Generator
Rng = np.random.default_rng(seed=42)

# Integers
RandomInts = Rng.integers(0, 10, size=5)
print(f"integers(0, 10, size=5): {RandomInts}")

# Random floats [0, 1)
RandomFloats = Rng.random(size=5)
print(f"\nrandom(size=5): {RandomFloats}")

# Normal distribution
NormalSamples = Rng.normal(loc=0, scale=1, size=5)
print(f"\nnormal(loc=0, scale=1, size=5): {NormalSamples}")

# Uniform distribution
UniformSamples = Rng.uniform(low=0, high=10, size=5)
print(f"\nuniform(low=0, high=10, size=5): {UniformSamples}")

# Choice - random selection
ChoiceArray = np.array([10, 20, 30, 40, 50])
Chosen = Rng.choice(ChoiceArray, size=3, replace=False)
print(f"\nchoice from {ChoiceArray}: {Chosen}")

# Shuffle
ShuffleArray = np.array([1, 2, 3, 4, 5])
Rng.shuffle(ShuffleArray)
print(f"\nAfter shuffle: {ShuffleArray}")

# Permutation
Permuted = Rng.permutation([1, 2, 3, 4, 5])
print(f"\npermutation: {Permuted}")

# Various distributions
ExponentialSamples = Rng.exponential(scale=2.0, size=5)
PoissonSamples = Rng.poisson(lam=5, size=5)
BinomialSamples = Rng.binomial(n=10, p=0.5, size=5)

print(f"\nexponential(scale=2.0): {ExponentialSamples}")
print(f"poisson(lam=5): {PoissonSamples}")
print(f"binomial(n=10, p=0.5): {BinomialSamples}")

# Seeds
RngSeeded1 = np.random.default_rng(seed=42)
RngSeeded2 = np.random.default_rng(seed=42)
SameSeed1 = RngSeeded1.integers(0, 10, size=3)
SameSeed2 = RngSeeded2.integers(0, 10, size=3)
print(f"\nSame seed produces same sequence:")
print(f"Seed 42, first 3: {SameSeed1}")
print(f"Seed 42, first 3: {SameSeed2}")

# ============================================================================
# Polynomials
# ============================================================================

print("\n\n--- Polynomials ---\n")

# poly1d - polynomial from coefficients
PolyCoeffs = np.array([1, 2, 3])  # 1*x^2 + 2*x + 3
Poly1D = np.poly1d(PolyCoeffs)
print(f"Coefficients: {PolyCoeffs}")
print(f"Polynomial: {Poly1D}")
print(f"Evaluate at x=2: {Poly1D(2)}")

# polyfit - fit polynomial to data
XFit = np.array([0, 1, 2, 3, 4])
YFit = np.array([1, 3, 5, 7, 9])  # y = 2x + 1
FitCoeffs = np.polyfit(XFit, YFit, deg=1)
FitPoly = np.poly1d(FitCoeffs)
print(f"\nData points: x={XFit}, y={YFit}")
print(f"Fitted coefficients: {FitCoeffs}")
print(f"Fitted polynomial: {FitPoly}")

# polyval - evaluate polynomial
EvalX = np.array([0, 1, 2])
PolyValResult = np.polyval(PolyCoeffs, EvalX)
print(f"\npolyval({PolyCoeffs}, {EvalX}): {PolyValResult}")

# roots - find roots
RootsPoly = np.array([1, -5, 6])  # x^2 - 5x + 6 = (x-2)(x-3)
Roots = np.roots(RootsPoly)
print(f"\nPolynomial coefficients: {RootsPoly}")
print(f"Roots: {Roots}")

# polynomial.polynomial module
from numpy.polynomial import Polynomial
PolyObj = Polynomial([3, 2, 1])  # 3 + 2x + x^2
PolyEval = PolyObj(2)
PolyDeriv = PolyObj.deriv()
PolyInteg = PolyObj.integ()

print(f"\nPolynomial object: {PolyObj}")
print(f"Evaluate at 2: {PolyEval}")
print(f"Derivative: {PolyDeriv}")
print(f"Integral: {PolyInteg}")

print("\n" + "=" * 80)
print("END OF FILE 4")
print("=" * 80)
