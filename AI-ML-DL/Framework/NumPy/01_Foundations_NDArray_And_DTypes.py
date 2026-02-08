"""
NumPy Foundations: NDArray Creation and DTypes
Comprehensive examples of array creation, dtype system, type casting, views vs copies,
array attributes, and memory layout.
"""

import numpy as np

print("=" * 80)
print("FILE 1: FOUNDATIONS - NDARRAY AND DTYPES")
print("=" * 80)

# ============================================================================
# Array Creation
# ============================================================================

print("\n--- Array Creation ---\n")

# np.array - basic array creation
ArrayFromList = np.array([1, 2, 3, 4, 5])
print(f"np.array([1,2,3,4,5]): {ArrayFromList}")
print(f"Shape: {ArrayFromList.shape}, Dtype: {ArrayFromList.dtype}")

Array2D = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D array:\n{Array2D}")
print(f"Shape: {Array2D.shape}, NDim: {Array2D.ndim}")

# np.zeros - array filled with zeros
ZerosArray = np.zeros((3, 4))
print(f"\nnp.zeros((3, 4)):\n{ZerosArray}")

ZerosInt = np.zeros((2, 3), dtype=np.int32)
print(f"\nnp.zeros((2, 3), dtype=int32):\n{ZerosInt}")

# np.ones - array filled with ones
OnesArray = np.ones((2, 3))
print(f"\nnp.ones((2, 3)):\n{OnesArray}")

OnesFloat = np.ones((3, 2), dtype=np.float64)
print(f"\nnp.ones((3, 2), dtype=float64):\n{OnesFloat}")

# np.full - array filled with a specific value
FullArray = np.full((2, 3), 7)
print(f"\nnp.full((2, 3), 7):\n{FullArray}")

FullFloat = np.full((3, 2), 3.14)
print(f"\nnp.full((3, 2), 3.14):\n{FullFloat}")

# np.empty - uninitialized array (faster, contains garbage)
EmptyArray = np.empty((2, 3))
print(f"\nnp.empty((2, 3)) (uninitialized):\n{EmptyArray}")

# np.arange - similar to Python range
ArangeArray = np.arange(0, 10, 2)
print(f"\nnp.arange(0, 10, 2): {ArangeArray}")

ArangeFloat = np.arange(0.0, 1.0, 0.2)
print(f"np.arange(0.0, 1.0, 0.2): {ArangeFloat}")

# np.linspace - evenly spaced numbers over interval
LinspaceArray = np.linspace(0, 10, 5)
print(f"\nnp.linspace(0, 10, 5): {LinspaceArray}")

LinspaceEndpoint = np.linspace(0, 1, 5, endpoint=False)
print(f"np.linspace(0, 1, 5, endpoint=False): {LinspaceEndpoint}")

# np.logspace - logarithmically spaced numbers
LogspaceArray = np.logspace(0, 2, 5)
print(f"\nnp.logspace(0, 2, 5): {LogspaceArray}")

# np.eye - identity matrix
EyeMatrix = np.eye(3)
print(f"\nnp.eye(3):\n{EyeMatrix}")

EyeK = np.eye(4, k=1)
print(f"\nnp.eye(4, k=1) (diagonal offset):\n{EyeK}")

# np.identity - square identity matrix
IdentityMatrix = np.identity(3)
print(f"\nnp.identity(3):\n{IdentityMatrix}")

# np.diag - create diagonal array or extract diagonal
DiagArray = np.diag([1, 2, 3, 4])
print(f"\nnp.diag([1, 2, 3, 4]):\n{DiagArray}")

ExtractDiag = np.diag(DiagArray)
print(f"np.diag(matrix) extracts diagonal: {ExtractDiag}")

# np.tri - lower triangular matrix
TriMatrix = np.tri(3, 3)
print(f"\nnp.tri(3, 3):\n{TriMatrix}")

# np.zeros_like - zeros array with same shape and dtype
OriginalArray = np.array([[1, 2], [3, 4]], dtype=np.float32)
ZerosLike = np.zeros_like(OriginalArray)
print(f"\nOriginal: {OriginalArray}")
print(f"np.zeros_like: {ZerosLike}")
print(f"Same dtype: {ZerosLike.dtype == OriginalArray.dtype}")

# np.ones_like - ones array with same shape and dtype
OnesLike = np.ones_like(OriginalArray)
print(f"\nnp.ones_like: {OnesLike}")

# np.fromfunction - create array from function
FromFunction = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
print(f"\nnp.fromfunction(lambda i, j: i + j, (3, 3)):\n{FromFunction}")

# np.meshgrid - coordinate matrices
X, Y = np.meshgrid(np.arange(0, 3), np.arange(0, 3))
print(f"\nnp.meshgrid example:")
print(f"X:\n{X}")
print(f"Y:\n{Y}")

# np.mgrid - meshgrid as array
MgridResult = np.mgrid[0:3, 0:3]
print(f"\nnp.mgrid[0:3, 0:3] shape: {MgridResult.shape}")

# np.ogrid - open meshgrid (saves memory)
OgridResult = np.ogrid[0:3, 0:3]
print(f"np.ogrid[0:3, 0:3] - list of arrays: {[arr.shape for arr in OgridResult]}")

# ============================================================================
# Dtype System
# ============================================================================

print("\n\n--- Dtype System ---\n")

# Integer types
Int8Array = np.array([1, 2, 3], dtype=np.int8)
Int16Array = np.array([1, 2, 3], dtype=np.int16)
Int32Array = np.array([1, 2, 3], dtype=np.int32)
Int64Array = np.array([1, 2, 3], dtype=np.int64)
Uint8Array = np.array([1, 2, 3], dtype=np.uint8)
Uint32Array = np.array([1, 2, 3], dtype=np.uint32)

print(f"int8: {Int8Array.dtype}, itemsize: {Int8Array.itemsize} bytes")
print(f"int16: {Int16Array.dtype}, itemsize: {Int16Array.itemsize} bytes")
print(f"int32: {Int32Array.dtype}, itemsize: {Int32Array.itemsize} bytes")
print(f"int64: {Int64Array.dtype}, itemsize: {Int64Array.itemsize} bytes")
print(f"uint8: {Uint8Array.dtype}, itemsize: {Uint8Array.itemsize} bytes")
print(f"uint32: {Uint32Array.dtype}, itemsize: {Uint32Array.itemsize} bytes")

# Float types
Float16Array = np.array([1.5, 2.5], dtype=np.float16)
Float32Array = np.array([1.5, 2.5], dtype=np.float32)
Float64Array = np.array([1.5, 2.5], dtype=np.float64)

print(f"\nfloat16: {Float16Array.dtype}, itemsize: {Float16Array.itemsize} bytes")
print(f"float32: {Float32Array.dtype}, itemsize: {Float32Array.itemsize} bytes")
print(f"float64: {Float64Array.dtype}, itemsize: {Float64Array.itemsize} bytes")

# Complex types
Complex64Array = np.array([1+2j, 3+4j], dtype=np.complex64)
Complex128Array = np.array([1+2j, 3+4j], dtype=np.complex128)

print(f"\ncomplex64: {Complex64Array.dtype}, itemsize: {Complex64Array.itemsize} bytes")
print(f"complex128: {Complex128Array.dtype}, itemsize: {Complex128Array.itemsize} bytes")

# Boolean type
BoolArray = np.array([True, False, True], dtype=bool)
print(f"\nbool: {BoolArray.dtype}, itemsize: {BoolArray.itemsize} bytes")

# String types
StringArray = np.array(['hello', 'world'], dtype='U10')
BytesArray = np.array([b'hello', b'world'], dtype='S10')

print(f"\nUnicode string (U10): {StringArray.dtype}")
print(f"Bytes string (S10): {BytesArray.dtype}")

# Structured dtypes
StructuredDtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
StructuredArray = np.array([('Alice', 25, 55.5), ('Bob', 30, 70.2)], dtype=StructuredDtype)
print(f"\nStructured array:\n{StructuredArray}")
print(f"Access field 'name': {StructuredArray['name']}")
print(f"Access field 'age': {StructuredArray['age']}")

# Custom dtype with endianness
BigEndian = np.dtype('>i4')  # big-endian int32
LittleEndian = np.dtype('<i4')  # little-endian int32
print(f"\nBig-endian int32: {BigEndian}")
print(f"Little-endian int32: {LittleEndian}")

# ============================================================================
# Type Casting
# ============================================================================

print("\n\n--- Type Casting ---\n")

OriginalInt = np.array([1, 2, 3, 4], dtype=np.int32)
CastToFloat = OriginalInt.astype(np.float64)
print(f"Original (int32): {OriginalInt}, dtype: {OriginalInt.dtype}")
print(f"After astype(float64): {CastToFloat}, dtype: {CastToFloat.dtype}")

# Copy vs no-copy
ViewSameDtype = OriginalInt.astype(OriginalInt.dtype, copy=False)
CopySameDtype = OriginalInt.astype(OriginalInt.dtype, copy=True)
print(f"\nastype(same_dtype, copy=False) shares memory: {np.shares_memory(OriginalInt, ViewSameDtype)}")
print(f"astype(same_dtype, copy=True) shares memory: {np.shares_memory(OriginalInt, CopySameDtype)}")

# Casting rules
IntArray = np.array([1, 2, 3], dtype=np.int32)
FloatArray = np.array([1.5, 2.5, 3.5], dtype=np.float64)

# Safe casting (default)
SafeCast = IntArray.astype(np.float64, casting='safe')
print(f"\nSafe casting int32 -> float64: {SafeCast}")

# Same-kind casting
SameKindCast = FloatArray.astype(np.float32, casting='same_kind')
print(f"Same-kind casting float64 -> float32: {SameKindCast}")

# Unsafe casting (may lose precision)
UnsafeCast = FloatArray.astype(np.int32, casting='unsafe')
print(f"Unsafe casting float64 -> int32: {UnsafeCast}")

# ============================================================================
# Views vs Copies
# ============================================================================

print("\n\n--- Views vs Copies ---\n")

OriginalArray = np.array([1, 2, 3, 4, 5])

# Slicing creates a view
SliceView = OriginalArray[1:4]
print(f"Original: {OriginalArray}")
print(f"Slice [1:4]: {SliceView}")
print(f"Slice is view: {np.shares_memory(OriginalArray, SliceView)}")
print(f"Slice.base is Original: {SliceView.base is OriginalArray}")

# Modifying view affects original
SliceView[0] = 99
print(f"\nAfter modifying slice view: Original = {OriginalArray}")

# Reset for next example
OriginalArray = np.array([1, 2, 3, 4, 5])

# Fancy indexing creates a copy
FancyIndex = OriginalArray[[0, 2, 4]]
print(f"\nFancy indexing [0, 2, 4]: {FancyIndex}")
print(f"Fancy index is copy: {not np.shares_memory(OriginalArray, FancyIndex)}")

# Boolean indexing creates a copy
BoolMask = OriginalArray > 2
BoolIndex = OriginalArray[BoolMask]
print(f"\nBoolean indexing (values > 2): {BoolIndex}")
print(f"Boolean index is copy: {not np.shares_memory(OriginalArray, BoolIndex)}")

# Reshape creates a view (if possible)
ReshapeView = OriginalArray.reshape(5, 1)
print(f"\nReshape to (5, 1):\n{ReshapeView}")
print(f"Reshape is view: {np.shares_memory(OriginalArray, ReshapeView)}")

# Transpose creates a view
Matrix2D = np.array([[1, 2, 3], [4, 5, 6]])
TransposeView = Matrix2D.T
print(f"\nOriginal matrix:\n{Matrix2D}")
print(f"Transpose:\n{TransposeView}")
print(f"Transpose is view: {np.shares_memory(Matrix2D, TransposeView)}")

# ============================================================================
# Array Attributes
# ============================================================================

print("\n\n--- Array Attributes ---\n")

ExampleArray = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"Array:\n{ExampleArray}")
print(f"shape: {ExampleArray.shape}")
print(f"ndim (number of dimensions): {ExampleArray.ndim}")
print(f"size (total elements): {ExampleArray.size}")
print(f"dtype: {ExampleArray.dtype}")
print(f"itemsize (bytes per element): {ExampleArray.itemsize}")
print(f"nbytes (total bytes): {ExampleArray.nbytes}")
print(f"strides: {ExampleArray.strides}")
print(f"flags:\n{ExampleArray.flags}")

# Flat iterator
FlatIterator = ExampleArray.flat
print(f"\nFlat iterator (first 5 elements): {[next(FlatIterator) for _ in range(5)]}")

# Transpose attribute
TransposeAttr = ExampleArray.T
print(f"\nTranspose (.T):\n{TransposeAttr}")

# ============================================================================
# Memory Layout
# ============================================================================

print("\n\n--- Memory Layout ---\n")

# C order (row-major, default)
CArray = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(f"C-order array:\n{CArray}")
print(f"C-order contiguous: {CArray.flags.c_contiguous}")
print(f"F-order contiguous: {CArray.flags.f_contiguous}")

# Fortran order (column-major)
FArray = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f"\nF-order array:\n{FArray}")
print(f"C-order contiguous: {FArray.flags.c_contiguous}")
print(f"F-order contiguous: {FArray.flags.f_contiguous}")

# Convert to C-contiguous
NonContiguous = FArray.T  # Transpose may not be contiguous
ContiguousArray = np.ascontiguousarray(NonContiguous)
print(f"\nNon-contiguous array C-contiguous: {NonContiguous.flags.c_contiguous}")
print(f"After ascontiguousarray: {ContiguousArray.flags.c_contiguous}")

# Memory layout affects performance
LargeArrayC = np.zeros((1000, 1000), order='C')
LargeArrayF = np.zeros((1000, 1000), order='F')

print(f"\nLarge C-order array contiguous: {LargeArrayC.flags.c_contiguous}")
print(f"Large F-order array contiguous: {LargeArrayF.flags.f_contiguous}")

print("\n" + "=" * 80)
print("END OF FILE 1")
print("=" * 80)
