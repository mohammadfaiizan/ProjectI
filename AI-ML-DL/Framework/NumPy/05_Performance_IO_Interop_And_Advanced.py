"""
NumPy Performance, I/O, Interop and Advanced Topics
Comprehensive examples of memory layout, strides, vectorization, file I/O,
structured arrays, masked arrays, datetime arrays, string operations, interop,
common pitfalls, memory-mapped files, and performance comparisons.
"""

import numpy as np
import time
import os

print("=" * 80)
print("FILE 5: PERFORMANCE, I/O, INTEROP AND ADVANCED")
print("=" * 80)

# ============================================================================
# Memory Layout
# ============================================================================

print("\n--- Memory Layout ---\n")

# C order (row-major, default)
CArray = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(f"C-order array:\n{CArray}")
print(f"C-contiguous: {CArray.flags.c_contiguous}")
print(f"F-contiguous: {CArray.flags.f_contiguous}")
print(f"Strides: {CArray.strides}")

# F order (column-major)
FArray = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f"\nF-order array:\n{FArray}")
print(f"C-contiguous: {FArray.flags.c_contiguous}")
print(f"F-contiguous: {FArray.flags.f_contiguous}")
print(f"Strides: {FArray.strides}")

# ascontiguousarray
NonContig = FArray.T
ContigArray = np.ascontiguousarray(NonContig)
print(f"\nNon-contiguous C-contiguous: {NonContig.flags.c_contiguous}")
print(f"After ascontiguousarray: {ContigArray.flags.c_contiguous}")

# Checking contiguity
IsCContig = CArray.flags.c_contiguous
IsFContig = FArray.flags.f_contiguous
print(f"\nC-array is C-contiguous: {IsCContig}")
print(f"F-array is F-contiguous: {IsFContig}")

# ============================================================================
# Strides and Stride Tricks
# ============================================================================

print("\n\n--- Strides and Stride Tricks ---\n")

StrideArray = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Original array:\n{StrideArray}")
print(f"Shape: {StrideArray.shape}")
print(f"Strides: {StrideArray.strides}")
print(f"Itemsize: {StrideArray.itemsize}")

# as_strided - create view with custom strides
StrideView = np.lib.stride_tricks.as_strided(
    StrideArray,
    shape=(2, 2),
    strides=(StrideArray.strides[0] * 2, StrideArray.strides[1] * 2)
)
print(f"\nas_strided view (2x2, stride 2):\n{StrideView}")

# sliding_window_view - sliding window
WindowArray = np.array([1, 2, 3, 4, 5, 6])
SlidingWindow = np.lib.stride_tricks.sliding_window_view(WindowArray, window_shape=3)
print(f"\nArray: {WindowArray}")
print(f"sliding_window_view(window_shape=3):\n{SlidingWindow}")

# ============================================================================
# Vectorization Patterns
# ============================================================================

print("\n\n--- Vectorization Patterns ---\n")

# Replacing loops with vectorized operations
LoopArray = np.arange(10)
LoopResult = np.zeros_like(LoopArray)

# Loop version (slow)
for i in range(len(LoopArray)):
    LoopResult[i] = LoopArray[i] ** 2

# Vectorized version (fast)
VectorizedResult = LoopArray ** 2

print(f"Array: {LoopArray}")
print(f"Loop result: {LoopResult}")
print(f"Vectorized result: {VectorizedResult}")
print(f"Results match: {np.array_equal(LoopResult, VectorizedResult)}")

# np.where for conditional operations
ConditionArray = np.array([1, 5, 3, 8, 2, 7])
WhereResult = np.where(ConditionArray > 4, ConditionArray * 2, ConditionArray)
print(f"\nArray: {ConditionArray}")
print(f"np.where(>4, x*2, x): {WhereResult}")

# np.select - multiple conditions
SelectArray = np.array([1, 5, 3, 8, 2, 7])
Conditions = [SelectArray < 3, SelectArray < 6, SelectArray >= 6]
Choices = [SelectArray * 10, SelectArray * 2, SelectArray]
SelectResult = np.select(Conditions, Choices)
print(f"\nArray: {SelectArray}")
print(f"np.select result: {SelectResult}")

# np.piecewise - piecewise function
PiecewiseArray = np.array([-2, -1, 0, 1, 2])
PiecewiseResult = np.piecewise(
    PiecewiseArray,
    [PiecewiseArray < 0, PiecewiseArray >= 0],
    [lambda x: x**2, lambda x: x*2]
)
print(f"\nArray: {PiecewiseArray}")
print(f"np.piecewise result: {PiecewiseResult}")

# ============================================================================
# np.vectorize Demonstration
# ============================================================================

print("\n\n--- np.vectorize Demonstration ---\n")

def CustomFunction(x):
    return x**2 + 2*x + 1

VectorizedFunc = np.vectorize(CustomFunction)
VectorizedInput = np.array([1, 2, 3, 4, 5])
VectorizedOutput = VectorizedFunc(VectorizedInput)

ManualOutput = VectorizedInput**2 + 2*VectorizedInput + 1

print(f"Input: {VectorizedInput}")
print(f"vectorize result: {VectorizedOutput}")
print(f"Manual calculation: {ManualOutput}")
print(f"Results match: {np.array_equal(VectorizedOutput, ManualOutput)}")

# ============================================================================
# Pre-allocation vs Append
# ============================================================================

print("\n\n--- Pre-allocation vs Append ---\n")

Size = 10000

# Append approach (slow)
AppendList = []
StartTime = time.time()
for i in range(Size):
    AppendList.append(i**2)
AppendTime = time.time() - StartTime
AppendArray = np.array(AppendList)

# Pre-allocation approach (fast)
PreAllocArray = np.zeros(Size)
StartTime = time.time()
for i in range(Size):
    PreAllocArray[i] = i**2
PreAllocTime = time.time() - StartTime

# Vectorized approach (fastest)
StartTime = time.time()
VectorizedArray = np.arange(Size)**2
VectorizedTime = time.time() - StartTime

print(f"Size: {Size}")
print(f"Append time: {AppendTime:.6f} seconds")
print(f"Pre-allocation time: {PreAllocTime:.6f} seconds")
print(f"Vectorized time: {VectorizedTime:.6f} seconds")
print(f"Results match: {np.array_equal(AppendArray, PreAllocArray)}")

# ============================================================================
# In-place Operations
# ============================================================================

print("\n\n--- In-place Operations ---\n")

InPlaceArray = np.array([1, 2, 3, 4, 5])
print(f"Original: {InPlaceArray}")

# In-place addition
InPlaceArray += 10
print(f"After += 10: {InPlaceArray}")

# In-place multiplication
InPlaceArray *= 2
print(f"After *= 2: {InPlaceArray}")

# In-place operations save memory
LargeArray = np.ones(1000000)
IdBefore = id(LargeArray)
LargeArray += 1
IdAfter = id(LargeArray)
print(f"\nLarge array ID before: {IdBefore}")
print(f"Large array ID after += 1: {IdAfter}")
print(f"Same object (in-place): {IdBefore == IdAfter}")

# ============================================================================
# File I/O
# ============================================================================

print("\n\n--- File I/O ---\n")

# Create temporary directory for examples
TempDir = "temp_numpy_io"
if not os.path.exists(TempDir):
    os.makedirs(TempDir)

# save/load - binary format (.npy)
SaveArray = np.array([[1, 2, 3], [4, 5, 6]])
SavePath = os.path.join(TempDir, "array.npy")
np.save(SavePath, SaveArray)
LoadedArray = np.load(SavePath)
print(f"Original:\n{SaveArray}")
print(f"\nLoaded:\n{LoadedArray}")
print(f"Arrays equal: {np.array_equal(SaveArray, LoadedArray)}")

# savez - multiple arrays (.npz)
Array1 = np.array([1, 2, 3])
Array2 = np.array([4, 5, 6])
SavezPath = os.path.join(TempDir, "arrays.npz")
np.savez(SavezPath, arr1=Array1, arr2=Array2)
LoadedNPZ = np.load(SavezPath)
print(f"\nsavez loaded keys: {list(LoadedNPZ.keys())}")
print(f"arr1: {LoadedNPZ['arr1']}")
print(f"arr2: {LoadedNPZ['arr2']}")

# savez_compressed
SavezCompPath = os.path.join(TempDir, "arrays_compressed.npz")
np.savez_compressed(SavezCompPath, arr1=Array1, arr2=Array2)
CompSize = os.path.getsize(SavezCompPath)
UncompSize = os.path.getsize(SavezPath)
print(f"\nCompressed size: {CompSize} bytes")
print(f"Uncompressed size: {UncompSize} bytes")

# savetxt/loadtxt - text format
TxtArray = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
TxtPath = os.path.join(TempDir, "array.txt")
np.savetxt(TxtPath, TxtArray, fmt='%.2f')
LoadedTxt = np.loadtxt(TxtPath)
print(f"\nSaved to text:\n{TxtArray}")
print(f"Loaded from text:\n{LoadedTxt}")

# genfromtxt - more flexible text loading
GenTxtPath = os.path.join(TempDir, "genfromtxt.txt")
with open(GenTxtPath, 'w') as f:
    f.write("1.0, 2.0, 3.0\n")
    f.write("4.0, 5.0, 6.0\n")
GenLoaded = np.genfromtxt(GenTxtPath, delimiter=',')
print(f"\ngenfromtxt result:\n{GenLoaded}")

# Cleanup
import shutil
if os.path.exists(TempDir):
    shutil.rmtree(TempDir)

# ============================================================================
# Structured Arrays
# ============================================================================

print("\n\n--- Structured Arrays ---\n")

# Creation
StructuredDtype = np.dtype([
    ('name', 'U10'),
    ('age', 'i4'),
    ('weight', 'f4'),
    ('height', 'f4')
])

StructuredData = np.array([
    ('Alice', 25, 55.5, 165.0),
    ('Bob', 30, 70.2, 180.0),
    ('Charlie', 35, 65.8, 175.0)
], dtype=StructuredDtype)

print(f"Structured array:\n{StructuredData}")
print(f"\nField 'name': {StructuredData['name']}")
print(f"Field 'age': {StructuredData['age']}")
print(f"Field 'weight': {StructuredData['weight']}")

# Field access
NameField = StructuredData['name']
AgeField = StructuredData['age']
print(f"\nNames: {NameField}")
print(f"Ages: {AgeField}")

# Nested structured arrays
NestedDtype = np.dtype([
    ('person', [('name', 'U10'), ('age', 'i4')]),
    ('location', [('city', 'U10'), ('country', 'U10')])
])

NestedData = np.array([
    (('Alice', 25), ('NYC', 'USA')),
    (('Bob', 30), ('London', 'UK'))
], dtype=NestedDtype)

print(f"\nNested structured array:")
print(f"person.name: {NestedData['person']['name']}")
print(f"location.city: {NestedData['location']['city']}")

# ============================================================================
# Masked Arrays
# ============================================================================

print("\n\n--- Masked Arrays ---\n")

# Create masked array
MaskedData = np.array([1, 2, 3, 4, 5, 6])
Mask = [False, False, True, False, True, False]
MaskedArray = np.ma.masked_array(MaskedData, mask=Mask)
print(f"Data: {MaskedData}")
print(f"Mask: {Mask}")
print(f"Masked array: {MaskedArray}")
print(f"Masked values: {MaskedArray.compressed()}")

# Operations with masked arrays
MaskedArray2 = np.ma.masked_array([10, 20, 30, 40, 50, 60], mask=Mask)
MaskedSum = MaskedArray + MaskedArray2
MaskedMean = np.ma.mean(MaskedArray)
print(f"\nMasked array 2: {MaskedArray2}")
print(f"Sum: {MaskedSum}")
print(f"Mean (ignoring masked): {MaskedMean}")

# Mask where condition
ConditionMasked = np.ma.masked_where(MaskedData < 3, MaskedData)
print(f"\nmasked_where(< 3): {ConditionMasked}")

# ============================================================================
# Datetime Arrays
# ============================================================================

print("\n\n--- Datetime Arrays ---\n")

# datetime64 creation
Dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64')
print(f"Dates: {Dates}")
print(f"Dtype: {Dates.dtype}")

# Different units
DatesDays = np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[D]')
DatesHours = np.array(['2024-01-01T00', '2024-01-01T12'], dtype='datetime64[h]')
DatesSeconds = np.array(['2024-01-01T00:00:00'], dtype='datetime64[s]')

print(f"\nDays: {DatesDays}")
print(f"Hours: {DatesHours}")
print(f"Seconds: {DatesSeconds}")

# timedelta64
TimeDeltas = np.array([1, 2, 3], dtype='timedelta64[D]')
DatePlusDelta = DatesDays[0] + TimeDeltas
print(f"\nDates: {DatesDays}")
print(f"Time deltas: {TimeDeltas}")
print(f"Dates + deltas: {DatePlusDelta}")

# Arithmetic
DateDiff = DatesDays[1] - DatesDays[0]
DateRange = np.arange('2024-01-01', '2024-01-10', dtype='datetime64[D]')
print(f"\nDate difference: {DateDiff}")
print(f"Date range:\n{DateRange}")

# ============================================================================
# String Operations
# ============================================================================

print("\n\n--- String Operations ---\n")

StringArray = np.array(['hello', 'world', 'numpy', 'python'])

# np.char module
CharUpper = np.char.upper(StringArray)
CharLower = np.char.lower(['HELLO', 'WORLD'])
CharCapitalize = np.char.capitalize(StringArray)
CharTitle = np.char.title(StringArray)

print(f"Original: {StringArray}")
print(f"upper: {CharUpper}")
print(f"lower: {CharLower}")
print(f"capitalize: {CharCapitalize}")
print(f"title: {CharTitle}")

# String manipulation
CharAdd = np.char.add(['hello'], [' world'])
CharMultiply = np.char.multiply(['ha'], 3)
CharStrip = np.char.strip(['  hello  ', '  world  '])
CharSplit = np.char.split(['hello world', 'numpy python'])

print(f"\nadd: {CharAdd}")
print(f"multiply('ha', 3): {CharMultiply}")
print(f"strip: {CharStrip}")
print(f"split: {CharSplit}")

# String comparison
CharEqual = np.char.equal(['hello', 'world'], ['hello', 'numpy'])
CharFind = np.char.find(['hello world'], 'world')

print(f"\nequal: {CharEqual}")
print(f"find: {CharFind}")

# ============================================================================
# Interop
# ============================================================================

print("\n\n--- Interop ---\n")

# List conversion
NumpyArray = np.array([1, 2, 3, 4, 5])
PythonList = NumpyArray.tolist()
ListToArray = np.array(PythonList)

print(f"NumPy array: {NumpyArray}")
print(f"To Python list: {PythonList}")
print(f"Back to array: {ListToArray}")

# Pandas compatibility pattern
PandasCompatible = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
PandasMask = np.isnan(PandasCompatible)
PandasClean = PandasCompatible[~PandasMask]
print(f"\nPandas-compatible array: {PandasCompatible}")
print(f"NaN mask: {PandasMask}")
print(f"Cleaned: {PandasClean}")

# PIL/Pillow compatibility pattern
PILCompatible = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
PILShape = PILCompatible.shape
PILDtype = PILCompatible.dtype
print(f"\nPIL-compatible array shape: {PILShape}, dtype: {PILDtype}")

# PyTorch compatibility pattern
PyTorchCompatible = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
PyTorchShape = PyTorchCompatible.shape
PyTorchStrides = PyTorchCompatible.strides
PyTorchContiguous = PyTorchCompatible.flags.c_contiguous
print(f"\nPyTorch-compatible array:")
print(f"Shape: {PyTorchShape}, Strides: {PyTorchStrides}")
print(f"C-contiguous: {PyTorchContiguous}")

# ============================================================================
# Common Pitfalls
# ============================================================================

print("\n\n--- Common Pitfalls ---\n")

# Integer overflow
Int8Array = np.array([100, 200], dtype=np.int8)
Int8Overflow = Int8Array + 100
print(f"int8 array: {Int8Array}")
print(f"After +100 (overflow): {Int8Overflow}")
print(f"Warning: Integer overflow occurred!")

# Float comparison
FloatA = 0.1 + 0.2
FloatB = 0.3
DirectCompare = FloatA == FloatB
IsCloseResult = np.isclose(FloatA, FloatB)
AllCloseResult = np.allclose([FloatA], [FloatB])

print(f"\nFloat comparison:")
print(f"0.1 + 0.2 = {FloatA}")
print(f"0.3 = {FloatB}")
print(f"Direct == : {DirectCompare}")
print(f"isclose: {IsCloseResult}")
print(f"allclose: {AllCloseResult}")

# View vs copy confusion
ViewCopyArray = np.array([1, 2, 3, 4, 5])
ViewSlice = ViewCopyArray[1:4]
ViewSlice[0] = 99
print(f"\nView vs copy:")
print(f"Original after modifying slice view: {ViewCopyArray}")

FancyCopy = ViewCopyArray[[0, 2, 4]]
FancyCopy[0] = 88
print(f"Original after modifying fancy index copy: {ViewCopyArray}")

# ============================================================================
# Memory-mapped Files
# ============================================================================

print("\n\n--- Memory-mapped Files ---\n")

# Create temporary file for memmap
MemmapPath = "temp_memmap.dat"
if os.path.exists(MemmapPath):
    os.remove(MemmapPath)

# Create memmap
MemmapArray = np.memmap(MemmapPath, dtype='float32', mode='w+', shape=(3, 4))
MemmapArray[:] = np.arange(12).reshape(3, 4).astype('float32')
print(f"Created memmap array:\n{MemmapArray}")

# Read memmap
ReadMemmap = np.memmap(MemmapPath, dtype='float32', mode='r', shape=(3, 4))
print(f"\nRead memmap:\n{ReadMemmap}")

# Cleanup
if os.path.exists(MemmapPath):
    os.remove(MemmapPath)

# ============================================================================
# Performance Timing Comparison
# ============================================================================

print("\n\n--- Performance Timing Comparison ---\n")

Size = 1000000
TestArray = np.random.random(Size)

# Loop version
StartTime = time.time()
LoopResult = np.zeros(Size)
for i in range(Size):
    LoopResult[i] = TestArray[i] ** 2
LoopTime = time.time() - StartTime

# Vectorized version
StartTime = time.time()
VectorizedResult = TestArray ** 2
VectorizedTime = time.time() - StartTime

Speedup = LoopTime / VectorizedTime

print(f"Array size: {Size:,}")
print(f"Loop time: {LoopTime:.6f} seconds")
print(f"Vectorized time: {VectorizedTime:.6f} seconds")
print(f"Speedup: {Speedup:.2f}x")
print(f"Results match: {np.allclose(LoopResult, VectorizedResult)}")

# Complex operation comparison
ComplexArray = np.random.random((1000, 1000))

# Loop version
StartTime = time.time()
LoopComplex = np.zeros_like(ComplexArray)
for i in range(ComplexArray.shape[0]):
    for j in range(ComplexArray.shape[1]):
        LoopComplex[i, j] = ComplexArray[i, j] * 2 + 1
LoopComplexTime = time.time() - StartTime

# Vectorized version
StartTime = time.time()
VectorizedComplex = ComplexArray * 2 + 1
VectorizedComplexTime = time.time() - StartTime

ComplexSpeedup = LoopComplexTime / VectorizedComplexTime

print(f"\nComplex operation (1000x1000):")
print(f"Loop time: {LoopComplexTime:.6f} seconds")
print(f"Vectorized time: {VectorizedComplexTime:.6f} seconds")
print(f"Speedup: {ComplexSpeedup:.2f}x")
print(f"Results match: {np.allclose(LoopComplex, VectorizedComplex)}")

print("\n" + "=" * 80)
print("END OF FILE 5")
print("=" * 80)
