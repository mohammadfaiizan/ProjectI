"""
NumPy Indexing, Slicing, Reshaping and Manipulation
Comprehensive examples of indexing, slicing, fancy indexing, boolean indexing,
reshaping, stacking, splitting, sorting, searching, set operations, and tiling.
"""

import numpy as np

print("=" * 80)
print("FILE 2: INDEXING, SLICING, RESHAPING AND MANIPULATION")
print("=" * 80)

# ============================================================================
# Basic Indexing
# ============================================================================

print("\n--- Basic Indexing ---\n")

# 1D indexing
Array1D = np.array([10, 20, 30, 40, 50])
print(f"1D array: {Array1D}")
print(f"Index 0: {Array1D[0]}")
print(f"Index -1 (last): {Array1D[-1]}")
print(f"Index 2: {Array1D[2]}")

# 2D indexing
Array2D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D array:\n{Array2D}")
print(f"Array2D[0, 0]: {Array2D[0, 0]}")
print(f"Array2D[1, 2]: {Array2D[1, 2]}")
print(f"Array2D[-1, -1]: {Array2D[-1, -1]}")

# 3D indexing
Array3D = np.arange(24).reshape(2, 3, 4)
print(f"\n3D array shape: {Array3D.shape}")
print(f"Array3D[0, 1, 2]: {Array3D[0, 1, 2]}")
print(f"Array3D[1, 0, :]: {Array3D[1, 0, :]}")

# ============================================================================
# Slicing
# ============================================================================

print("\n\n--- Slicing ---\n")

# Basic slicing: start:stop:step
SlicedArray = Array1D[1:4]
print(f"Array1D[1:4]: {SlicedArray}")

SlicedStep = Array1D[0:5:2]
print(f"Array1D[0:5:2]: {SlicedStep}")

# Multi-dimensional slicing
Sliced2D = Array2D[0:2, 1:3]
print(f"\nArray2D[0:2, 1:3]:\n{Sliced2D}")

SlicedRow = Array2D[1, :]
print(f"Array2D[1, :] (entire row): {SlicedRow}")

SlicedCol = Array2D[:, 1]
print(f"Array2D[:, 1] (entire column): {SlicedCol}")

# Ellipsis (...)
SlicedEllipsis = Array3D[..., 0]
print(f"\nArray3D[..., 0] (first element of last axis):\n{SlicedEllipsis}")

# np.s_ - slice object builder
SliceObj = np.s_[1:4:2]
SlicedWithS = Array1D[SliceObj]
print(f"\nnp.s_[1:4:2]: {SlicedWithS}")

# ============================================================================
# Fancy Indexing
# ============================================================================

print("\n\n--- Fancy Indexing ---\n")

# Integer array indexing (always creates copies)
FancyIndices = Array1D[[0, 2, 4]]
print(f"Array1D[[0, 2, 4]]: {FancyIndices}")
print(f"Is copy: {not np.shares_memory(Array1D, FancyIndices)}")

# Multi-dimensional fancy indexing
Fancy2D = Array2D[[0, 2], [1, 2]]
print(f"\nArray2D[[0, 2], [1, 2]]: {Fancy2D}")

# Using np.ix_ for mesh indexing
MeshIndices = Array2D[np.ix_([0, 2], [0, 2])]
print(f"\nArray2D[np.ix_([0, 2], [0, 2])]:\n{MeshIndices}")

# ============================================================================
# Boolean Indexing
# ============================================================================

print("\n\n--- Boolean Indexing ---\n")

# Boolean mask
Mask = Array1D > 30
BoolIndexed = Array1D[Mask]
print(f"Array: {Array1D}")
print(f"Mask (values > 30): {Mask}")
print(f"Boolean indexed: {BoolIndexed}")

# np.where - conditional selection
WhereResult = np.where(Array1D > 30, Array1D, 0)
print(f"\nnp.where(Array1D > 30, Array1D, 0): {WhereResult}")

IndicesWhere = np.where(Array1D > 30)
print(f"np.where(Array1D > 30) returns indices: {IndicesWhere[0]}")

# np.nonzero - indices of non-zero elements
NonZeroArray = np.array([0, 1, 0, 3, 0, 5])
NonZeroIndices = np.nonzero(NonZeroArray)
print(f"\nArray: {NonZeroArray}")
print(f"np.nonzero indices: {NonZeroIndices[0]}")

# np.argwhere - indices as array
ArgWhereResult = np.argwhere(Array1D > 30)
print(f"\nnp.argwhere(Array1D > 30):\n{ArgWhereResult}")

# ============================================================================
# np.take, np.put, np.choose
# ============================================================================

print("\n\n--- np.take, np.put, np.choose ---\n")

# np.take - take elements by index
TakeResult = np.take(Array1D, [0, 2, 4])
print(f"np.take(Array1D, [0, 2, 4]): {TakeResult}")

TakeAxis = np.take(Array2D, [0, 2], axis=1)
print(f"\nnp.take(Array2D, [0, 2], axis=1):\n{TakeAxis}")

# np.put - put values at indices
PutArray = Array1D.copy()
np.put(PutArray, [1, 3], [99, 88])
print(f"\nAfter np.put([1, 3], [99, 88]): {PutArray}")

# np.choose - choose from arrays by index
ChooseArray = np.array([0, 1, 2, 0, 1])
Choices = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
ChooseResult = np.choose(ChooseArray, Choices)
print(f"\nChoices:\n{Choices}")
print(f"Choose indices: {ChooseArray}")
print(f"np.choose result: {ChooseResult}")

# ============================================================================
# Reshaping
# ============================================================================

print("\n\n--- Reshaping ---\n")

OriginalArray = np.arange(12)
print(f"Original (1D): {OriginalArray}")

# reshape
Reshaped = OriginalArray.reshape(3, 4)
print(f"\nreshape(3, 4):\n{Reshaped}")

ReshapedAuto = OriginalArray.reshape(-1, 4)
print(f"\nreshape(-1, 4) (auto-calculate rows):\n{ReshapedAuto}")

# ravel - flatten (view if possible)
RavelResult = Reshaped.ravel()
print(f"\nravel() (flatten): {RavelResult}")
print(f"Is view: {np.shares_memory(Reshaped, RavelResult)}")

# flatten - always returns copy
FlattenResult = Reshaped.flatten()
print(f"\nflatten() (always copy): {FlattenResult}")
print(f"Is copy: {not np.shares_memory(Reshaped, FlattenResult)}")

# transpose
Transposed = Reshaped.transpose()
print(f"\nTranspose:\n{Transposed}")

# swapaxes
SwapAxesResult = Array3D.swapaxes(0, 2)
print(f"\nswapaxes(0, 2) shape: {SwapAxesResult.shape}")

# moveaxis
MoveAxisResult = np.moveaxis(Array3D, 0, -1)
print(f"moveaxis(0, -1) shape: {MoveAxisResult.shape}")

# expand_dims
Expanded = np.expand_dims(OriginalArray, axis=0)
print(f"\nexpand_dims(axis=0) shape: {Expanded.shape}")

# squeeze - remove dimensions of size 1
Squeezed = np.squeeze(Expanded)
print(f"squeeze() shape: {Squeezed.shape}")

# newaxis
NewAxisResult = OriginalArray[np.newaxis, :]
print(f"\nnewaxis shape: {NewAxisResult.shape}")

# ============================================================================
# Stacking
# ============================================================================

print("\n\n--- Stacking ---\n")

ArrayA = np.array([[1, 2], [3, 4]])
ArrayB = np.array([[5, 6], [7, 8]])

# concatenate
ConcatAxis0 = np.concatenate([ArrayA, ArrayB], axis=0)
print(f"concatenate(axis=0):\n{ConcatAxis0}")

ConcatAxis1 = np.concatenate([ArrayA, ArrayB], axis=1)
print(f"\nconcatenate(axis=1):\n{ConcatAxis1}")

# stack - creates new axis
StackResult = np.stack([ArrayA, ArrayB], axis=0)
print(f"\nstack(axis=0) shape: {StackResult.shape}")

# vstack - vertical stack
VStackResult = np.vstack([ArrayA, ArrayB])
print(f"\nvstack:\n{VStackResult}")

# hstack - horizontal stack
HStackResult = np.hstack([ArrayA, ArrayB])
print(f"\nhstack:\n{HStackResult}")

# dstack - depth stack
DStackResult = np.dstack([ArrayA, ArrayB])
print(f"\ndstack shape: {DStackResult.shape}")

# column_stack
Col1 = np.array([1, 2, 3])
Col2 = np.array([4, 5, 6])
ColStackResult = np.column_stack([Col1, Col2])
print(f"\ncolumn_stack:\n{ColStackResult}")

# block - advanced stacking
BlockResult = np.block([[ArrayA, ArrayB], [ArrayB, ArrayA]])
print(f"\nblock (2x2 grid):\n{BlockResult}")

# ============================================================================
# Splitting
# ============================================================================

print("\n\n--- Splitting ---\n")

SplitArray = np.arange(12).reshape(3, 4)
print(f"Array to split:\n{SplitArray}")

# split
SplitResult = np.split(SplitArray, 3, axis=0)
print(f"\nsplit(3, axis=0) - {len(SplitResult)} arrays:")
for i, arr in enumerate(SplitResult):
    print(f"  Part {i}:\n{arr}")

# hsplit
HSplitResult = np.hsplit(SplitArray, 2)
print(f"\nhsplit(2) - {len(HSplitResult)} arrays:")
for i, arr in enumerate(HSplitResult):
    print(f"  Part {i}:\n{arr}")

# vsplit
VSplitResult = np.vsplit(SplitArray, 3)
print(f"\nvsplit(3) - {len(VSplitResult)} arrays:")
for i, arr in enumerate(VSplitResult):
    print(f"  Part {i}: {arr}")

# array_split - allows uneven splits
ArraySplitResult = np.array_split(np.arange(10), 3)
print(f"\narray_split(arange(10), 3) - uneven split:")
for i, arr in enumerate(ArraySplitResult):
    print(f"  Part {i}: {arr}")

# ============================================================================
# Sorting
# ============================================================================

print("\n\n--- Sorting ---\n")

UnsortedArray = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Unsorted: {UnsortedArray}")

# sort - in-place or return sorted copy
SortedArray = np.sort(UnsortedArray)
print(f"np.sort (returns copy): {SortedArray}")
print(f"Original unchanged: {UnsortedArray}")

UnsortedArray.sort()  # in-place
print(f"After .sort() (in-place): {UnsortedArray}")

# argsort - indices that would sort array
UnsortedForArgsort = np.array([3, 1, 4, 1, 5])
ArgsortIndices = np.argsort(UnsortedForArgsort)
print(f"\nArray: {UnsortedForArgsort}")
print(f"argsort indices: {ArgsortIndices}")
print(f"Sorted using argsort: {UnsortedForArgsort[ArgsortIndices]}")

# lexsort - sort by multiple keys
Keys1 = np.array([2, 2, 1, 1])
Keys2 = np.array([1, 2, 1, 2])
LexsortIndices = np.lexsort((Keys2, Keys1))
print(f"\nKeys1: {Keys1}, Keys2: {Keys2}")
print(f"lexsort indices: {LexsortIndices}")

# partition - partial sort
PartitionArray = np.array([3, 1, 4, 1, 5, 9, 2, 6])
Partitioned = np.partition(PartitionArray, 3)
print(f"\nArray: {PartitionArray}")
print(f"partition(3) - first 3 elements are smallest: {Partitioned}")

# argpartition
ArgPartitionIndices = np.argpartition(PartitionArray, 3)
print(f"argpartition(3) indices: {ArgPartitionIndices}")

# ============================================================================
# Searching
# ============================================================================

print("\n\n--- Searching ---\n")

SearchArray = np.array([1, 3, 5, 7, 9, 11])

# searchsorted - find insertion points
SearchSortedResult = np.searchsorted(SearchArray, [2, 4, 6, 8])
print(f"Sorted array: {SearchArray}")
print(f"searchsorted([2, 4, 6, 8]): {SearchSortedResult}")

# digitize - bin indices
Values = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
Bins = np.array([1, 2, 3, 4])
DigitizeResult = np.digitize(Values, Bins)
print(f"\nValues: {Values}")
print(f"Bins: {Bins}")
print(f"digitize result: {DigitizeResult}")

# bincount - count occurrences
BincountArray = np.array([0, 1, 1, 3, 2, 1, 7])
BincountResult = np.bincount(BincountArray)
print(f"\nArray: {BincountArray}")
print(f"bincount: {BincountResult}")

# ============================================================================
# Set Operations
# ============================================================================

print("\n\n--- Set Operations ---\n")

SetArray1 = np.array([1, 2, 3, 4, 5])
SetArray2 = np.array([3, 4, 5, 6, 7])

# unique
UniqueArray = np.array([1, 2, 2, 3, 3, 3])
UniqueResult = np.unique(UniqueArray)
print(f"Array: {UniqueArray}")
print(f"unique: {UniqueResult}")

# union1d
UnionResult = np.union1d(SetArray1, SetArray2)
print(f"\nArray1: {SetArray1}")
print(f"Array2: {SetArray2}")
print(f"union1d: {UnionResult}")

# intersect1d
IntersectResult = np.intersect1d(SetArray1, SetArray2)
print(f"intersect1d: {IntersectResult}")

# setdiff1d
SetDiffResult = np.setdiff1d(SetArray1, SetArray2)
print(f"setdiff1d (elements in 1 but not in 2): {SetDiffResult}")

# isin
IsInArray = np.array([1, 2, 3, 10, 11])
IsInResult = np.isin(IsInArray, SetArray1)
print(f"\nArray: {IsInArray}")
print(f"Is in SetArray1: {IsInResult}")

# ============================================================================
# Tiling and Repeating
# ============================================================================

print("\n\n--- Tiling and Repeating ---\n")

TileArray = np.array([[1, 2], [3, 4]])

# tile
TileResult = np.tile(TileArray, (2, 3))
print(f"Original:\n{TileArray}")
print(f"\ntile((2, 3)):\n{TileResult}")

# repeat
RepeatResult = np.repeat(TileArray, 2, axis=0)
print(f"\nrepeat(2, axis=0):\n{RepeatResult}")

RepeatAxis1 = np.repeat(TileArray, 3, axis=1)
print(f"\nrepeat(3, axis=1):\n{RepeatAxis1}")

# flip - reverse array
FlipResult = np.flip(Array1D)
print(f"\nOriginal: {Array1D}")
print(f"flip: {FlipResult}")

# fliplr - flip left-right
FlipLRResult = np.fliplr(Array2D)
print(f"\nOriginal:\n{Array2D}")
print(f"\nfliplr:\n{FlipLRResult}")

# flipud - flip up-down
FlipUDResult = np.flipud(Array2D)
print(f"\nflipud:\n{FlipUDResult}")

# rot90 - rotate 90 degrees
Rot90Result = np.rot90(Array2D, k=1)
print(f"\nrot90(k=1):\n{Rot90Result}")

# roll - circular shift
RollResult = np.roll(Array1D, shift=2)
print(f"\nOriginal: {Array1D}")
print(f"roll(shift=2): {RollResult}")

RollAxis = np.roll(Array2D, shift=1, axis=0)
print(f"\nroll(shift=1, axis=0):\n{RollAxis}")

print("\n" + "=" * 80)
print("END OF FILE 2")
print("=" * 80)
