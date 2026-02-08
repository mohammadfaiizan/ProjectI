"""
NumPy Mathematics, Statistics and Broadcasting
Comprehensive examples of arithmetic operations, ufuncs, trigonometric functions,
exponential/logarithmic functions, rounding, comparisons, aggregations, broadcasting,
and statistical operations.
"""

import numpy as np

print("=" * 80)
print("FILE 3: MATHEMATICS, STATISTICS AND BROADCASTING")
print("=" * 80)

# ============================================================================
# Arithmetic Operations
# ============================================================================

print("\n--- Arithmetic Operations ---\n")

ArrayA = np.array([1, 2, 3, 4])
ArrayB = np.array([5, 6, 7, 8])

# Addition
AddResult = np.add(ArrayA, ArrayB)
AddOperator = ArrayA + ArrayB
print(f"ArrayA: {ArrayA}")
print(f"ArrayB: {ArrayB}")
print(f"np.add: {AddResult}")
print(f"Operator +: {AddOperator}")

# Subtraction
SubResult = np.subtract(ArrayB, ArrayA)
SubOperator = ArrayB - ArrayA
print(f"\nnp.subtract: {SubResult}")
print(f"Operator -: {SubOperator}")

# Multiplication
MulResult = np.multiply(ArrayA, ArrayB)
MulOperator = ArrayA * ArrayB
print(f"\nnp.multiply: {MulResult}")
print(f"Operator *: {MulOperator}")

# Division
DivResult = np.divide(ArrayB, ArrayA)
DivOperator = ArrayB / ArrayA
print(f"\nnp.divide: {DivResult}")
print(f"Operator /: {DivOperator}")

# Floor division
FloorDivResult = np.floor_divide(ArrayB, ArrayA)
FloorDivOperator = ArrayB // ArrayA
print(f"\nnp.floor_divide: {FloorDivResult}")
print(f"Operator //: {FloorDivOperator}")

# Modulo
ModResult = np.mod(ArrayB, ArrayA)
ModOperator = ArrayB % ArrayA
print(f"\nnp.mod: {ModResult}")
print(f"Operator %: {ModOperator}")

# Power
PowerResult = np.power(ArrayA, 2)
PowerOperator = ArrayA ** 2
print(f"\nnp.power(ArrayA, 2): {PowerResult}")
print(f"Operator **: {PowerOperator}")

# ============================================================================
# Ufunc Methods
# ============================================================================

print("\n\n--- Ufunc Methods ---\n")

UfuncArray = np.array([1, 2, 3, 4, 5])

# reduce - reduce along axis
ReduceSum = np.add.reduce(UfuncArray)
ReduceProd = np.multiply.reduce(UfuncArray)
print(f"Array: {UfuncArray}")
print(f"add.reduce (sum): {ReduceSum}")
print(f"multiply.reduce (product): {ReduceProd}")

Reduce2D = np.array([[1, 2, 3], [4, 5, 6]])
ReduceAxis0 = np.add.reduce(Reduce2D, axis=0)
ReduceAxis1 = np.add.reduce(Reduce2D, axis=1)
print(f"\n2D array:\n{Reduce2D}")
print(f"reduce(axis=0): {ReduceAxis0}")
print(f"reduce(axis=1): {ReduceAxis1}")

# accumulate - cumulative operation
AccumulateSum = np.add.accumulate(UfuncArray)
AccumulateProd = np.multiply.accumulate(UfuncArray)
print(f"\nadd.accumulate: {AccumulateSum}")
print(f"multiply.accumulate: {AccumulateProd}")

# outer - outer product
OuterResult = np.multiply.outer(ArrayA[:3], ArrayB[:3])
print(f"\nmultiply.outer:\n{OuterResult}")

# at - in-place operation at indices
AtArray = np.array([1, 2, 3, 4, 5])
np.add.at(AtArray, [0, 2, 4], 10)
print(f"\nAfter np.add.at([0,2,4], 10): {AtArray}")

# ============================================================================
# Trigonometric Functions
# ============================================================================

print("\n\n--- Trigonometric Functions ---\n")

AngleArray = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

SinResult = np.sin(AngleArray)
CosResult = np.cos(AngleArray)
TanResult = np.tan(AngleArray)

print(f"Angles (radians): {AngleArray}")
print(f"sin: {SinResult}")
print(f"cos: {CosResult}")
print(f"tan: {TanResult}")

# Inverse trigonometric
ArcSinResult = np.arcsin([0, 0.5, 1])
ArcCosResult = np.arccos([1, 0.5, 0])
ArcTanResult = np.arctan([0, 1, np.inf])

print(f"\narcsin([0, 0.5, 1]): {ArcSinResult}")
print(f"arccos([1, 0.5, 0]): {ArcCosResult}")
print(f"arctan([0, 1, inf]): {ArcTanResult}")

# arctan2 - two-argument arctan
Y = np.array([1, 1, -1, -1])
X = np.array([1, -1, -1, 1])
ArcTan2Result = np.arctan2(Y, X)
print(f"\narctan2(Y, X): {ArcTan2Result}")

# Degree/radian conversion
Degrees = np.array([0, 30, 45, 60, 90])
Radians = np.deg2rad(Degrees)
Rad2Deg = np.rad2deg(Radians)
print(f"\nDegrees: {Degrees}")
print(f"deg2rad: {Radians}")
print(f"rad2deg: {Rad2Deg}")

# ============================================================================
# Exponential and Logarithmic Functions
# ============================================================================

print("\n\n--- Exponential and Logarithmic Functions ---\n")

ExpArray = np.array([0, 1, 2, 3])

# exp - e^x
ExpResult = np.exp(ExpArray)
print(f"Array: {ExpArray}")
print(f"exp: {ExpResult}")

# log - natural logarithm
LogArray = np.array([1, np.e, np.e**2, np.e**3])
LogResult = np.log(LogArray)
print(f"\nArray: {LogArray}")
print(f"log (natural): {LogResult}")

# log2, log10
Log2Result = np.log2([1, 2, 4, 8])
Log10Result = np.log10([1, 10, 100, 1000])
print(f"\nlog2([1,2,4,8]): {Log2Result}")
print(f"log10([1,10,100,1000]): {Log10Result}")

# expm1 - exp(x) - 1 (more accurate for small x)
ExpM1Result = np.expm1([0, 0.001, 0.01])
ExpMinus1 = np.exp([0, 0.001, 0.01]) - 1
print(f"\nexpm1([0, 0.001, 0.01]): {ExpM1Result}")
print(f"exp - 1: {ExpMinus1}")

# log1p - log(1 + x) (more accurate for small x)
Log1PResult = np.log1p([0, 0.001, 0.01])
LogPlus1 = np.log([1, 1.001, 1.01])
print(f"\nlog1p([0, 0.001, 0.01]): {Log1PResult}")
print(f"log(1 + x): {LogPlus1}")

# ============================================================================
# Rounding Functions
# ============================================================================

print("\n\n--- Rounding Functions ---\n")

RoundArray = np.array([1.2, 1.5, 1.7, 2.5, -1.2, -1.5])

RoundResult = np.round(RoundArray)
FloorResult = np.floor(RoundArray)
CeilResult = np.ceil(RoundArray)
TruncResult = np.trunc(RoundArray)

print(f"Array: {RoundArray}")
print(f"round: {RoundResult}")
print(f"floor: {FloorResult}")
print(f"ceil: {CeilResult}")
print(f"trunc: {TruncResult}")

# Round to decimals
RoundDecimals = np.round([1.234, 5.678, 9.012], decimals=2)
print(f"\nround([1.234, 5.678, 9.012], decimals=2): {RoundDecimals}")

# ============================================================================
# Comparison and Logical Operations
# ============================================================================

print("\n\n--- Comparison and Logical Operations ---\n")

CompArrayA = np.array([1, 2, 3, 4, 5])
CompArrayB = np.array([5, 4, 3, 2, 1])

# Comparison operators
GreaterResult = np.greater(CompArrayA, CompArrayB)
LessResult = np.less(CompArrayA, CompArrayB)
EqualResult = np.equal(CompArrayA, CompArrayB)

print(f"ArrayA: {CompArrayA}")
print(f"ArrayB: {CompArrayB}")
print(f"greater: {GreaterResult}")
print(f"less: {LessResult}")
print(f"equal: {EqualResult}")

# Logical operations
LogicalAnd = np.logical_and([True, True, False, False], [True, False, True, False])
LogicalOr = np.logical_or([True, True, False, False], [True, False, True, False])
LogicalNot = np.logical_not([True, False])

print(f"\nlogical_and: {LogicalAnd}")
print(f"logical_or: {LogicalOr}")
print(f"logical_not: {LogicalNot}")

# all, any
AllTrue = np.all([True, True, True])
AllFalse = np.all([True, False, True])
AnyTrue = np.any([False, False, True])
AnyFalse = np.any([False, False, False])

print(f"\nall([True, True, True]): {AllTrue}")
print(f"all([True, False, True]): {AllFalse}")
print(f"any([False, False, True]): {AnyTrue}")
print(f"any([False, False, False]): {AnyFalse}")

# ============================================================================
# Aggregations with Axis
# ============================================================================

print("\n\n--- Aggregations with Axis ---\n")

AggArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Array:\n{AggArray}")

SumResult = np.sum(AggArray)
SumAxis0 = np.sum(AggArray, axis=0)
SumAxis1 = np.sum(AggArray, axis=1)

print(f"\nsum(): {SumResult}")
print(f"sum(axis=0): {SumAxis0}")
print(f"sum(axis=1): {SumAxis1}")

ProdResult = np.prod(AggArray)
ProdAxis0 = np.prod(AggArray, axis=0)
print(f"\nprod(): {ProdResult}")
print(f"prod(axis=0): {ProdAxis0}")

MeanResult = np.mean(AggArray)
MeanAxis0 = np.mean(AggArray, axis=0)
print(f"\nmean(): {MeanResult}")
print(f"mean(axis=0): {MeanAxis0}")

StdResult = np.std(AggArray)
StdAxis0 = np.std(AggArray, axis=0)
print(f"\nstd(): {StdResult:.4f}")
print(f"std(axis=0): {StdAxis0}")

VarResult = np.var(AggArray)
VarAxis0 = np.var(AggArray, axis=0)
print(f"\nvar(): {VarResult:.4f}")
print(f"var(axis=0): {VarAxis0}")

MedianResult = np.median(AggArray)
MedianAxis0 = np.median(AggArray, axis=0)
print(f"\nmedian(): {MedianResult}")
print(f"median(axis=0): {MedianAxis0}")

MinResult = np.min(AggArray)
MinAxis0 = np.min(AggArray, axis=0)
MaxResult = np.max(AggArray)
MaxAxis0 = np.max(AggArray, axis=0)
print(f"\nmin(): {MinResult}, min(axis=0): {MinAxis0}")
print(f"max(): {MaxResult}, max(axis=0): {MaxAxis0}")

PtpResult = np.ptp(AggArray)  # peak-to-peak (max - min)
PtpAxis0 = np.ptp(AggArray, axis=0)
print(f"\nptp(): {PtpResult}, ptp(axis=0): {PtpAxis0}")

# ============================================================================
# Cumulative Operations
# ============================================================================

print("\n\n--- Cumulative Operations ---\n")

CumArray = np.array([1, 2, 3, 4, 5])

CumSum = np.cumsum(CumArray)
CumProd = np.cumprod(CumArray)

print(f"Array: {CumArray}")
print(f"cumsum: {CumSum}")
print(f"cumprod: {CumProd}")

# diff - differences between consecutive elements
DiffArray = np.array([1, 4, 6, 7, 12])
DiffResult = np.diff(DiffArray)
DiffN = np.diff(DiffArray, n=2)
print(f"\nArray: {DiffArray}")
print(f"diff: {DiffResult}")
print(f"diff(n=2): {DiffN}")

# gradient - gradient approximation
GradientArray = np.array([1, 2, 4, 7, 11])
GradientResult = np.gradient(GradientArray)
print(f"\nArray: {GradientArray}")
print(f"gradient: {GradientResult}")

# ============================================================================
# argmin, argmax, percentile, quantile
# ============================================================================

print("\n\n--- argmin, argmax, percentile, quantile ---\n")

ArgArray = np.array([3, 1, 4, 1, 5, 9, 2, 6])

ArgMinResult = np.argmin(ArgArray)
ArgMaxResult = np.argmax(ArgArray)
print(f"Array: {ArgArray}")
print(f"argmin: {ArgMinResult} (value: {ArgArray[ArgMinResult]})")
print(f"argmax: {ArgMaxResult} (value: {ArgArray[ArgMaxResult]})")

PercentileResult = np.percentile(ArgArray, [25, 50, 75])
QuantileResult = np.quantile(ArgArray, [0.25, 0.5, 0.75])
print(f"\npercentile([25, 50, 75]): {PercentileResult}")
print(f"quantile([0.25, 0.5, 0.75]): {QuantileResult}")

# ============================================================================
# NaN-Safe Functions
# ============================================================================

print("\n\n--- NaN-Safe Functions ---\n")

NaNArray = np.array([1, 2, np.nan, 4, 5, np.nan])

NaNSum = np.nansum(NaNArray)
NaNMean = np.nanmean(NaNArray)
NaNStd = np.nanstd(NaNArray)
NaNMin = np.nanmin(NaNArray)
NaNMax = np.nanmax(NaNArray)

print(f"Array: {NaNArray}")
print(f"nansum: {NaNSum}")
print(f"nanmean: {NaNMean:.4f}")
print(f"nanstd: {NaNStd:.4f}")
print(f"nanmin: {NaNMin}")
print(f"nanmax: {NaNMax}")

# ============================================================================
# Histogram
# ============================================================================

print("\n\n--- Histogram ---\n")

HistData = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
HistCounts, HistBins = np.histogram(HistData, bins=5)
print(f"Data: {HistData}")
print(f"Histogram counts: {HistCounts}")
print(f"Histogram bins: {HistBins}")

# histogram2d
XData = np.array([1, 2, 2, 3, 3])
YData = np.array([1, 1, 2, 2, 3])
Hist2D, XBins, YBins = np.histogram2d(XData, YData, bins=3)
print(f"\n2D Histogram:\n{Hist2D}")

# ============================================================================
# Broadcasting
# ============================================================================

print("\n\n--- Broadcasting ---\n")

# Broadcasting rule: dimensions are compatible if they are equal or one is 1

# Example 1: Array + scalar
BroadcastArray1 = np.array([[1, 2, 3], [4, 5, 6]])
BroadcastScalar = BroadcastArray1 + 10
print(f"Array:\n{BroadcastArray1}")
print(f"\nArray + 10:\n{BroadcastScalar}")

# Example 2: Array + 1D array
BroadcastArray2 = np.array([[1, 2, 3], [4, 5, 6]])
Broadcast1D = BroadcastArray2 + np.array([10, 20, 30])
print(f"\nArray:\n{BroadcastArray2}")
print(f"1D array: [10, 20, 30]")
print(f"Result:\n{Broadcast1D}")

# Example 3: Column vector + row vector
ColumnVector = np.array([[1], [2], [3]])
RowVector = np.array([10, 20, 30])
Broadcast2D = ColumnVector + RowVector
print(f"\nColumn vector:\n{ColumnVector}")
print(f"Row vector: {RowVector}")
print(f"Result:\n{Broadcast2D}")

# Example 4: Incompatible shapes (will raise error)
try:
    IncompatibleA = np.array([[1, 2], [3, 4]])
    IncompatibleB = np.array([1, 2, 3])
    IncompatibleResult = IncompatibleA + IncompatibleB
except ValueError as e:
    print(f"\nIncompatible shapes error: {e}")

# np.broadcast_to
BroadcastToArray = np.array([1, 2, 3])
BroadcastToResult = np.broadcast_to(BroadcastToArray, (3, 3))
print(f"\nOriginal: {BroadcastToArray}")
print(f"broadcast_to((3, 3)):\n{BroadcastToResult}")

# np.broadcast_shapes
Shape1 = (3, 1, 5)
Shape2 = (1, 4, 5)
BroadcastShape = np.broadcast_shapes(Shape1, Shape2)
print(f"\nbroadcast_shapes({Shape1}, {Shape2}): {BroadcastShape}")

# ============================================================================
# Clipping
# ============================================================================

print("\n\n--- Clipping ---\n")

ClipArray = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

ClipResult = np.clip(ClipArray, 3, 7)
MinimumResult = np.minimum(ClipArray, 6)
MaximumResult = np.maximum(ClipArray, 3)

print(f"Array: {ClipArray}")
print(f"clip(3, 7): {ClipResult}")
print(f"minimum(6): {MinimumResult}")
print(f"maximum(3): {MaximumResult}")

# ============================================================================
# Complex Number Operations
# ============================================================================

print("\n\n--- Complex Number Operations ---\n")

ComplexArray = np.array([1+2j, 3+4j, 5+6j])

RealPart = np.real(ComplexArray)
ImagPart = np.imag(ComplexArray)
ConjResult = np.conj(ComplexArray)
AngleResult = np.angle(ComplexArray)

print(f"Complex array: {ComplexArray}")
print(f"real: {RealPart}")
print(f"imag: {ImagPart}")
print(f"conj: {ConjResult}")
print(f"angle (radians): {AngleResult}")

AngleDegrees = np.angle(ComplexArray, deg=True)
print(f"angle (degrees): {AngleDegrees}")

print("\n" + "=" * 80)
print("END OF FILE 3")
print("=" * 80)
