"""
Pandas Foundations: Series, DataFrame, and Data Types
Comprehensive examples covering core Pandas data structures and type system
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("PANDAS FOUNDATIONS: SERIES, DATAFRAME, AND DATA TYPES")
print("=" * 80)

# ============================================================================
# SERIES CREATION
# ============================================================================
print("\n" + "=" * 80)
print("SERIES CREATION")
print("=" * 80)

# Series from list
SeriesFromList = pd.Series([10, 20, 30, 40, 50])
print("\nSeries from list:")
print(SeriesFromList)

# Series from dict
SeriesFromDict = pd.Series({'A': 100, 'B': 200, 'C': 300, 'D': 400})
print("\nSeries from dict:")
print(SeriesFromDict)

# Series from ndarray
NumpyArray = np.array([1.5, 2.5, 3.5, 4.5])
SeriesFromNdarray = pd.Series(NumpyArray)
print("\nSeries from ndarray:")
print(SeriesFromNdarray)

# Series from scalar
SeriesFromScalar = pd.Series(42, index=['a', 'b', 'c', 'd'])
print("\nSeries from scalar:")
print(SeriesFromScalar)

# Series with custom index
SeriesWithCustomIndex = pd.Series([100, 200, 300], index=['X', 'Y', 'Z'])
print("\nSeries with custom index:")
print(SeriesWithCustomIndex)

# Named Series
NamedSeries = pd.Series([1, 2, 3, 4], name='MySeries')
print("\nNamed Series:")
print(NamedSeries)
print(f"Series name: {NamedSeries.name}")

# ============================================================================
# SERIES ATTRIBUTES
# ============================================================================
print("\n" + "=" * 80)
print("SERIES ATTRIBUTES")
print("=" * 80)

SampleSeries = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])

print("\nSeries values (numpy array):")
print(SampleSeries.values)
print(f"Type: {type(SampleSeries.values)}")

print("\nSeries index:")
print(SampleSeries.index)
print(f"Index type: {type(SampleSeries.index)}")

print(f"\nData type: {SampleSeries.dtype}")
print(f"Series name: {SampleSeries.name}")
print(f"Shape: {SampleSeries.shape}")
print(f"Size: {SampleSeries.size}")
print(f"Memory usage (bytes): {SampleSeries.nbytes}")

UniqueSeries = pd.Series([1, 2, 3, 4, 5])
DuplicateSeries = pd.Series([1, 2, 2, 3, 3])
print(f"\nIs unique (no duplicates): {UniqueSeries.is_unique}")
print(f"Is unique (with duplicates): {DuplicateSeries.is_unique}")

MonotonicIncreasing = pd.Series([1, 2, 3, 4, 5])
MonotonicDecreasing = pd.Series([5, 4, 3, 2, 1])
NonMonotonic = pd.Series([1, 3, 2, 4, 5])
print(f"\nIs monotonic increasing: {MonotonicIncreasing.is_monotonic_increasing}")
print(f"Is monotonic increasing (decreasing): {MonotonicDecreasing.is_monotonic_increasing}")
print(f"Is monotonic increasing (non-monotonic): {NonMonotonic.is_monotonic_increasing}")

# ============================================================================
# SERIES OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SERIES OPERATIONS")
print("=" * 80)

SeriesA = pd.Series([10, 20, 30, 40])
SeriesB = pd.Series([5, 15, 25, 35])

print("\nArithmetic operations:")
print(f"Addition:\n{SeriesA + SeriesB}")
print(f"\nSubtraction:\n{SeriesA - SeriesB}")
print(f"\nMultiplication:\n{SeriesA * SeriesB}")
print(f"\nDivision:\n{SeriesA / SeriesB}")

BooleanSeries = pd.Series([True, False, True, False, True])
print("\nBoolean operations:")
print(f"All True: {BooleanSeries.all()}")
print(f"Any True: {BooleanSeries.any()}")

LongSeries = pd.Series(range(100))
print("\nHead (first 5):")
print(LongSeries.head())
print("\nTail (last 5):")
print(LongSeries.tail())
print("\nSample (random 5):")
print(LongSeries.sample(5))

NumericSeries = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("\nDescribe (statistical summary):")
print(NumericSeries.describe())

# ============================================================================
# DATAFRAME CREATION
# ============================================================================
print("\n" + "=" * 80)
print("DATAFRAME CREATION")
print("=" * 80)

# DataFrame from dict of lists
DataFrameFromDictOfLists = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris']
})
print("\nDataFrame from dict of lists:")
print(DataFrameFromDictOfLists)

# DataFrame from dict of Series
SeriesName = pd.Series(['Alice', 'Bob', 'Charlie'])
SeriesAge = pd.Series([25, 30, 35])
SeriesCity = pd.Series(['NYC', 'London', 'Tokyo'])
DataFrameFromDictOfSeries = pd.DataFrame({
    'Name': SeriesName,
    'Age': SeriesAge,
    'City': SeriesCity
})
print("\nDataFrame from dict of Series:")
print(DataFrameFromDictOfSeries)

# DataFrame from list of dicts
ListOfDicts = [
    {'Name': 'Alice', 'Age': 25, 'Score': 85},
    {'Name': 'Bob', 'Age': 30, 'Score': 92},
    {'Name': 'Charlie', 'Age': 35, 'Score': 78}
]
DataFrameFromListOfDicts = pd.DataFrame(ListOfDicts)
print("\nDataFrame from list of dicts:")
print(DataFrameFromListOfDicts)

# DataFrame from 2D ndarray
TwoDArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
DataFrameFromNdarray = pd.DataFrame(TwoDArray, columns=['A', 'B', 'C'])
print("\nDataFrame from 2D ndarray:")
print(DataFrameFromNdarray)

# DataFrame with custom index and columns
DataFrameCustomIndex = pd.DataFrame(
    {'Value': [100, 200, 300, 400]},
    index=['Q1', 'Q2', 'Q3', 'Q4'],
    columns=['Value']
)
print("\nDataFrame with custom index:")
print(DataFrameCustomIndex)

# ============================================================================
# DATAFRAME ATTRIBUTES
# ============================================================================
print("\n" + "=" * 80)
print("DATAFRAME ATTRIBUTES")
print("=" * 80)

SampleDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
})

print("\nDataFrame values (numpy array):")
print(SampleDataFrame.values)
print(f"Values shape: {SampleDataFrame.values.shape}")

print("\nDataFrame index:")
print(SampleDataFrame.index)
print(f"Index type: {type(SampleDataFrame.index)}")

print("\nDataFrame columns:")
print(SampleDataFrame.columns)
print(f"Columns type: {type(SampleDataFrame.columns)}")

print("\nDataFrame dtypes:")
print(SampleDataFrame.dtypes)

print(f"\nShape: {SampleDataFrame.shape}")
print(f"Size: {SampleDataFrame.size}")
print(f"Number of dimensions: {SampleDataFrame.ndim}")
print(f"Axes: {SampleDataFrame.axes}")

EmptyDataFrame = pd.DataFrame()
print(f"\nIs empty DataFrame: {EmptyDataFrame.empty}")
print(f"Is non-empty DataFrame: {SampleDataFrame.empty}")

# ============================================================================
# INFO() AND DESCRIBE() DEMONSTRATIONS
# ============================================================================
print("\n" + "=" * 80)
print("INFO() AND DESCRIBE() DEMONSTRATIONS")
print("=" * 80)

MixedDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Active': [True, True, False, True]
})

print("\nDataFrame info():")
MixedDataFrame.info()

print("\nDataFrame describe() (numeric columns only):")
print(MixedDataFrame.describe())

print("\nDataFrame describe(include='all'):")
print(MixedDataFrame.describe(include='all'))

# ============================================================================
# MEMORY USAGE
# ============================================================================
print("\n" + "=" * 80)
print("MEMORY USAGE")
print("=" * 80)

LargeDataFrame = pd.DataFrame({
    'IntColumn': range(1000),
    'FloatColumn': np.random.randn(1000),
    'StringColumn': ['Text'] * 1000
})

print("\nMemory usage per column (deep=True):")
print(LargeDataFrame.memory_usage(deep=True))
print(f"\nTotal memory usage: {LargeDataFrame.memory_usage(deep=True).sum()} bytes")

# ============================================================================
# COLUMN ACCESS
# ============================================================================
print("\n" + "=" * 80)
print("COLUMN ACCESS")
print("=" * 80)

AccessDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'London', 'Tokyo']
})

print("\nSingle column access (bracket notation):")
print(AccessDataFrame['Name'])

print("\nSingle column access (dot notation):")
print(AccessDataFrame.Name)

print("\nMultiple column selection:")
print(AccessDataFrame[['Name', 'Age']])

# ============================================================================
# ADDING COLUMNS
# ============================================================================
print("\n" + "=" * 80)
print("ADDING COLUMNS")
print("=" * 80)

AddColumnDataFrame = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Assignment
AddColumnDataFrame['C'] = [7, 8, 9]
print("\nAfter assignment:")
print(AddColumnDataFrame)

# Using assign()
AddColumnDataFrame = AddColumnDataFrame.assign(D=[10, 11, 12])
print("\nAfter assign():")
print(AddColumnDataFrame)

# Using insert()
AddColumnDataFrame.insert(1, 'E', [13, 14, 15])
print("\nAfter insert() at position 1:")
print(AddColumnDataFrame)

# ============================================================================
# INDEX OBJECTS
# ============================================================================
print("\n" + "=" * 80)
print("INDEX OBJECTS")
print("=" * 80)

# RangeIndex
RangeIndexExample = pd.RangeIndex(start=0, stop=10, step=1)
print("\nRangeIndex:")
print(RangeIndexExample)

# DatetimeIndex
DatetimeIndexExample = pd.date_range('2024-01-01', periods=5, freq='D')
print("\nDatetimeIndex:")
print(DatetimeIndexExample)

# MultiIndex from tuples
TuplesForMultiIndex = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
MultiIndexFromTuples = pd.MultiIndex.from_tuples(TuplesForMultiIndex, names=['Letter', 'Number'])
print("\nMultiIndex from tuples:")
print(MultiIndexFromTuples)

# MultiIndex from arrays
ArraysForMultiIndex = [['X', 'X', 'Y', 'Y'], [1, 2, 1, 2]]
MultiIndexFromArrays = pd.MultiIndex.from_arrays(ArraysForMultiIndex, names=['Group', 'ID'])
print("\nMultiIndex from arrays:")
print(MultiIndexFromArrays)

# MultiIndex from product
MultiIndexFromProduct = pd.MultiIndex.from_product([['A', 'B'], [1, 2]], names=['Letter', 'Number'])
print("\nMultiIndex from product:")
print(MultiIndexFromProduct)

# ============================================================================
# INDEX OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("INDEX OPERATIONS")
print("=" * 80)

Index1 = pd.Index([1, 2, 3, 4, 5])
Index2 = pd.Index([3, 4, 5, 6, 7])

print(f"\nIndex 1: {Index1}")
print(f"Index 2: {Index2}")

UnionResult = Index1.union(Index2)
print(f"\nUnion: {UnionResult}")

IntersectionResult = Index1.intersection(Index2)
print(f"Intersection: {IntersectionResult}")

DifferenceResult = Index1.difference(Index2)
print(f"Difference (Index1 - Index2): {DifferenceResult}")

# ============================================================================
# DTYPE SYSTEM
# ============================================================================
print("\n" + "=" * 80)
print("DTYPE SYSTEM")
print("=" * 80)

DtypeDataFrame = pd.DataFrame({
    'Int64': pd.Series([1, 2, 3], dtype='int64'),
    'Float64': pd.Series([1.1, 2.2, 3.3], dtype='float64'),
    'Bool': pd.Series([True, False, True], dtype='bool'),
    'String': pd.Series(['a', 'b', 'c'], dtype='string'),
    'Object': pd.Series(['x', 'y', 'z'], dtype='object'),
    'Category': pd.Series(['Low', 'Medium', 'High'], dtype='category'),
    'Datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
    'NullableInt': pd.Series([1, 2, None], dtype='Int64')
})

print("\nDataFrame with various dtypes:")
print(DtypeDataFrame)
print("\nDtypes:")
print(DtypeDataFrame.dtypes)

# ============================================================================
# TYPE CONVERSION
# ============================================================================
print("\n" + "=" * 80)
print("TYPE CONVERSION")
print("=" * 80)

ConversionSeries = pd.Series(['1', '2', '3', '4', '5'])
print("\nOriginal Series:")
print(ConversionSeries)
print(f"Dtype: {ConversionSeries.dtype}")

# Using astype
ConvertedToInt = ConversionSeries.astype('int64')
print("\nAfter astype('int64'):")
print(ConvertedToInt)
print(f"Dtype: {ConvertedToInt.dtype}")

# Using to_numeric
NumericStringSeries = pd.Series(['1.5', '2.7', '3.9', 'invalid', '5.2'])
NumericConverted = pd.to_numeric(NumericStringSeries, errors='coerce')
print("\nUsing to_numeric with errors='coerce':")
print(NumericConverted)

# Using to_datetime
DateStringSeries = pd.Series(['2024-01-01', '2024-02-15', '2024-03-20'])
DatetimeConverted = pd.to_datetime(DateStringSeries)
print("\nUsing to_datetime:")
print(DatetimeConverted)
print(f"Dtype: {DatetimeConverted.dtype}")

# Using convert_dtypes
MixedTypeDataFrame = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [1.1, 2.2, 3.3],
    'C': ['a', 'b', 'c']
})
ConvertedDataFrame = MixedTypeDataFrame.convert_dtypes()
print("\nAfter convert_dtypes():")
print(ConvertedDataFrame.dtypes)

# ============================================================================
# PD.NA VS NP.NAN DEMONSTRATION
# ============================================================================
print("\n" + "=" * 80)
print("PD.NA VS NP.NAN DEMONSTRATION")
print("=" * 80)

SeriesWithNaN = pd.Series([1, 2, np.nan, 4, 5])
SeriesWithNA = pd.Series([1, 2, pd.NA, 4, 5], dtype='Int64')

print("\nSeries with np.nan:")
print(SeriesWithNaN)
print(f"Dtype: {SeriesWithNaN.dtype}")

print("\nSeries with pd.NA:")
print(SeriesWithNA)
print(f"Dtype: {SeriesWithNA.dtype}")

print("\nChecking for NaN:")
print(f"isna() with np.nan: {SeriesWithNaN.isna()}")
print(f"isna() with pd.NA: {SeriesWithNA.isna()}")

print("\nArithmetic operations:")
print(f"np.nan + 1: {np.nan + 1}")
print(f"pd.NA + 1: {pd.NA + 1}")

print("\n" + "=" * 80)
print("END OF FOUNDATIONS DEMONSTRATION")
print("=" * 80)
