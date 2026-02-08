"""
Pandas Performance, Advanced Features, and Best Practices
Comprehensive examples covering optimization, advanced features, and best practices
"""

import pandas as pd
import numpy as np
import time

print("=" * 80)
print("PANDAS PERFORMANCE, ADVANCED FEATURES, AND BEST PRACTICES")
print("=" * 80)

# ============================================================================
# MEMORY USAGE
# ============================================================================
print("\n" + "=" * 80)
print("MEMORY USAGE")
print("=" * 80)

MemoryDataFrame = pd.DataFrame({
    'IntColumn': range(10000),
    'FloatColumn': np.random.randn(10000),
    'StringColumn': ['Text'] * 10000
})

MemoryUsage = MemoryDataFrame.memory_usage(deep=True)
TotalMemory = MemoryUsage.sum()
print("\nMemory usage per column (deep=True):")
print(MemoryUsage)
print(f"\nTotal memory usage: {TotalMemory} bytes ({TotalMemory / 1024:.2f} KB)")

# Comparing dtypes
Int64Series = pd.Series(range(1000), dtype='int64')
Int32Series = pd.Series(range(1000), dtype='int32')
Int16Series = pd.Series(range(1000), dtype='int16')

print("\nMemory comparison by dtype:")
print(f"int64: {Int64Series.memory_usage(deep=True)} bytes")
print(f"int32: {Int32Series.memory_usage(deep=True)} bytes")
print(f"int16: {Int16Series.memory_usage(deep=True)} bytes")

# ============================================================================
# DOWNCASTING
# ============================================================================
print("\n" + "=" * 80)
print("DOWNCASTING")
print("=" * 80)

DowncastSeries = pd.Series([1, 2, 3, 4, 5], dtype='int64')
print(f"\nOriginal dtype: {DowncastSeries.dtype}")
print(f"Original memory: {DowncastSeries.memory_usage(deep=True)} bytes")

Downcasted = pd.to_numeric(DowncastSeries, downcast='integer')
print(f"\nAfter downcast='integer': {Downcasted.dtype}")
print(f"After downcast memory: {Downcasted.memory_usage(deep=True)} bytes")

FloatDowncastSeries = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float64')
DowncastedFloat = pd.to_numeric(FloatDowncastSeries, downcast='float')
print(f"\nFloat64 -> {DowncastedFloat.dtype}")

# ============================================================================
# CATEGORY DTYPE
# ============================================================================
print("\n" + "=" * 80)
print("CATEGORY DTYPE")
print("=" * 80)

CategoryDataFrame = pd.DataFrame({
    'Status': ['Active', 'Inactive', 'Active', 'Pending', 'Active'] * 2000,
    'Value': range(10000)
})

# Before conversion
BeforeMemory = CategoryDataFrame.memory_usage(deep=True).sum()
print(f"\nMemory before conversion: {BeforeMemory} bytes")

# After conversion
CategoryDataFrame['Status'] = CategoryDataFrame['Status'].astype('category')
AfterMemory = CategoryDataFrame.memory_usage(deep=True).sum()
print(f"Memory after conversion: {AfterMemory} bytes")
print(f"Memory saved: {BeforeMemory - AfterMemory} bytes ({((BeforeMemory - AfterMemory) / BeforeMemory * 100):.1f}%)")

# Category accessor
CategorySeries = pd.Series(['Low', 'Medium', 'High', 'Low', 'Medium'], dtype='category')
print("\nCategory accessor:")
print(f"Categories: {CategorySeries.cat.categories}")
print(f"Codes: {CategorySeries.cat.codes}")

# Rename categories
RenamedCategories = CategorySeries.cat.rename_categories({'Low': 'L', 'Medium': 'M', 'High': 'H'})
print("\nRenamed categories:")
print(RenamedCategories)

# Set categories
SetCategories = CategorySeries.cat.set_categories(['Low', 'Medium', 'High', 'VeryHigh'], ordered=True)
print("\nSet categories (with new category):")
print(SetCategories)

# ============================================================================
# VECTORIZED VS APPLY VS ITERROWS
# ============================================================================
print("\n" + "=" * 80)
print("VECTORIZED VS APPLY VS ITERROWS")
print("=" * 80)

PerformanceDataFrame = pd.DataFrame({
    'A': np.random.randn(10000),
    'B': np.random.randn(10000)
})

# Vectorized operation
StartTime = time.time()
VectorizedResult = PerformanceDataFrame['A'] * 2 + PerformanceDataFrame['B']
VectorizedTime = time.time() - StartTime
print(f"\nVectorized operation time: {VectorizedTime:.6f} seconds")

# Apply operation
StartTime = time.time()
ApplyResult = PerformanceDataFrame.apply(lambda row: row['A'] * 2 + row['B'], axis=1)
ApplyTime = time.time() - StartTime
print(f"Apply operation time: {ApplyTime:.6f} seconds")
print(f"Apply is {ApplyTime / VectorizedTime:.1f}x slower than vectorized")

# Iterrows (commented out for speed, but shown for demonstration)
# StartTime = time.time()
# IterrowsResult = []
# for Index, Row in PerformanceDataFrame.iterrows():
#     IterrowsResult.append(Row['A'] * 2 + Row['B'])
# IterrowsTime = time.time() - StartTime
# print(f"Iterrows operation time: {IterrowsTime:.6f} seconds")
# print(f"Iterrows is {IterrowsTime / VectorizedTime:.1f}x slower than vectorized")

print("\nNote: iterrows() is typically 100-1000x slower than vectorized operations")

# ============================================================================
# EVAL() AND QUERY() PERFORMANCE
# ============================================================================
print("\n" + "=" * 80)
print("EVAL() AND QUERY() PERFORMANCE")
print("=" * 80)

EvalDataFrame = pd.DataFrame({
    'A': np.random.randn(10000),
    'B': np.random.randn(10000),
    'C': np.random.randn(10000)
})

# Standard evaluation
StartTime = time.time()
StandardResult = EvalDataFrame['A'] + EvalDataFrame['B'] * EvalDataFrame['C']
StandardTime = time.time() - StartTime
print(f"\nStandard evaluation time: {StandardTime:.6f} seconds")

# Using eval()
StartTime = time.time()
EvalResult = EvalDataFrame.eval('A + B * C')
EvalTime = time.time() - StartTime
print(f"eval() time: {EvalTime:.6f} seconds")

# Query performance
QueryDataFrame = pd.DataFrame({
    'Value': np.random.randint(0, 100, 10000),
    'Category': np.random.choice(['A', 'B', 'C'], 10000)
})

StartTime = time.time()
QueryResult = QueryDataFrame[QueryDataFrame['Value'] > 50]
QueryTime1 = time.time() - StartTime

StartTime = time.time()
QueryResult2 = QueryDataFrame.query('Value > 50')
QueryTime2 = time.time() - StartTime

print(f"\nBoolean indexing time: {QueryTime1:.6f} seconds")
print(f"query() time: {QueryTime2:.6f} seconds")

# ============================================================================
# METHOD CHAINING
# ============================================================================
print("\n" + "=" * 80)
print("METHOD CHAINING")
print("=" * 80)

ChainDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Salary': [50000, 60000, 70000, 55000, 65000],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'IT']
})

# Method chaining example
ChainedResult = (ChainDataFrame
                 .query('Age > 28')
                 .assign(SalaryIncrease=lambda x: x['Salary'] * 1.1)
                 .pipe(lambda df: df[df['Department'] == 'IT'])
                 .sort_values('Salary', ascending=False)
                 .reset_index(drop=True))

print("\nMethod chaining result:")
print(ChainedResult)

# ============================================================================
# AVOIDING SETTINGWITHCOPYWARNING
# ============================================================================
print("\n" + "=" * 80)
print("AVOIDING SETTINGWITHCOPYWARNING")
print("=" * 80)

OriginalDataFrame = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

# Wrong way (may cause SettingWithCopyWarning)
# FilteredWrong = OriginalDataFrame[OriginalDataFrame['A'] > 2]
# FilteredWrong['C'] = 100  # Warning!

# Correct way 1: Use .copy()
FilteredCorrect1 = OriginalDataFrame[OriginalDataFrame['A'] > 2].copy()
FilteredCorrect1['C'] = 100
print("\nUsing .copy():")
print(FilteredCorrect1)

# Correct way 2: Use .loc
OriginalDataFrame.loc[OriginalDataFrame['A'] > 2, 'D'] = 200
print("\nUsing .loc:")
print(OriginalDataFrame)

# ============================================================================
# WINDOW FUNCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("WINDOW FUNCTIONS")
print("=" * 80)

WindowDataFrame = pd.DataFrame({
    'Value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
})

# Rolling with custom function
RollingCustom = WindowDataFrame['Value'].rolling(window=3).apply(lambda x: x.max() - x.min())
print("\nRolling range (max - min):")
print(RollingCustom)

# Grouped rolling
GroupedRolling = WindowDataFrame.groupby('Group')['Value'].rolling(window=2).mean().reset_index(level=0, drop=True)
print("\nGrouped rolling mean:")
print(GroupedRolling)

# ============================================================================
# PD.OPTION_CONTEXT
# ============================================================================
print("\n" + "=" * 80)
print("PD.OPTION_CONTEXT")
print("=" * 80)

LargeDisplayDataFrame = pd.DataFrame({
    'A': range(100),
    'B': range(100, 200)
})

print("\nDefault display (first 5 rows):")
print(LargeDisplayDataFrame.head())

# Using option_context
with pd.option_context('display.max_rows', 10, 'display.max_columns', 2):
    print("\nWith option_context (max_rows=10):")
    print(LargeDisplayDataFrame)

# ============================================================================
# PD.TESTING
# ============================================================================
print("\n" + "=" * 80)
print("PD.TESTING")
print("=" * 80)

TestDataFrame1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
TestDataFrame2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
TestDataFrame3 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 5, 6]})

# assert_frame_equal
try:
    pd.testing.assert_frame_equal(TestDataFrame1, TestDataFrame2)
    print("\nDataFrames are equal (assert_frame_equal passed)")
except AssertionError as E:
    print(f"\nDataFrames are not equal: {E}")

try:
    pd.testing.assert_frame_equal(TestDataFrame1, TestDataFrame3)
except AssertionError as E:
    print(f"DataFrames are not equal (expected): {E}")

# assert_series_equal
TestSeries1 = pd.Series([1, 2, 3])
TestSeries2 = pd.Series([1, 2, 3])
TestSeries3 = pd.Series([1, 2, 4])

try:
    pd.testing.assert_series_equal(TestSeries1, TestSeries2)
    print("Series are equal (assert_series_equal passed)")
except AssertionError as E:
    print(f"Series are not equal: {E}")

# ============================================================================
# INTEROP
# ============================================================================
print("\n" + "=" * 80)
print("INTEROP")
print("=" * 80)

InteropDataFrame = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# DataFrame to NumPy
NumpyArray = InteropDataFrame.values
print("\nDataFrame to NumPy array:")
print(NumpyArray)
print(f"Type: {type(NumpyArray)}")

# NumPy to DataFrame
NumpyToDataFrame = pd.DataFrame(NumpyArray, columns=['X', 'Y'])
print("\nNumPy array to DataFrame:")
print(NumpyToDataFrame)

# DataFrame to dict
DataFrameToDict = InteropDataFrame.to_dict()
print("\nDataFrame to dict:")
print(DataFrameToDict)

# Dict to DataFrame
DictToDataFrame = pd.DataFrame(DataFrameToDict)
print("\nDict to DataFrame:")
print(DictToDataFrame)

# DataFrame to records
DataFrameToRecords = InteropDataFrame.to_dict('records')
print("\nDataFrame to records:")
print(DataFrameToRecords)

# Records to DataFrame
RecordsToDataFrame = pd.DataFrame(DataFrameToRecords)
print("\nRecords to DataFrame:")
print(RecordsToDataFrame)

# ============================================================================
# COPY-ON-WRITE DEMONSTRATION
# ============================================================================
print("\n" + "=" * 80)
print("COPY-ON-WRITE DEMONSTRATION")
print("=" * 80)

# Enable copy-on-write (if available in pandas version)
try:
    pd.options.mode.copy_on_write = True
    CopyOnWriteDataFrame = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    CopyDataFrame = CopyOnWriteDataFrame.copy()
    CopyDataFrame['C'] = [7, 8, 9]
    print("\nCopy-on-write enabled")
    print("Original DataFrame unchanged:")
    print(CopyOnWriteDataFrame)
    print("\nCopy DataFrame modified:")
    print(CopyDataFrame)
except AttributeError:
    print("\nCopy-on-write not available in this pandas version")

# ============================================================================
# COMMON PITFALLS
# ============================================================================
print("\n" + "=" * 80)
print("COMMON PITFALLS")
print("=" * 80)

# Pitfall 1: Chained indexing
PitfallDataFrame = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

print("\nPitfall 1: Chained indexing")
print("Avoid: df[df['A'] > 2]['B'] = 100")
print("Use: df.loc[df['A'] > 2, 'B'] = 100")

# Pitfall 2: Growing DataFrame with loop
print("\nPitfall 2: Growing DataFrame with loop")
print("Avoid: Appending in a loop")
print("Use: Collect in list, then concat once")

ListOfDataFrames = []
for I in range(5):
    ListOfDataFrames.append(pd.DataFrame({'Value': [I * 10]}))
ConcatenatedDataFrame = pd.concat(ListOfDataFrames, ignore_index=True)
print("\nCorrect approach (concat once):")
print(ConcatenatedDataFrame)

# Pitfall 3: Mixed dtypes
MixedDtypeDataFrame = pd.DataFrame({
    'A': [1, 2, 'three', 4],
    'B': [10, 20, 30, 40]
})
print("\nPitfall 3: Mixed dtypes")
print("Column A has mixed types:")
print(MixedDtypeDataFrame['A'].dtype)
print("This can cause performance issues")

# Pitfall 4: Silent NaN in groupby
GroupbyNanDataFrame = pd.DataFrame({
    'Group': ['A', 'A', 'B', 'B', None],
    'Value': [1, 2, 3, 4, 5]
})
GroupbyNanResult = GroupbyNanDataFrame.groupby('Group')['Value'].sum()
print("\nPitfall 4: NaN in groupby")
print("NaN groups are excluded:")
print(GroupbyNanResult)

# Pitfall 5: Merge duplicates
MergeLeft = pd.DataFrame({'ID': [1, 2, 2], 'Value': [10, 20, 30]})
MergeRight = pd.DataFrame({'ID': [2, 3], 'Value': [40, 50]})
MergedDuplicates = pd.merge(MergeLeft, MergeRight, on='ID', suffixes=('_left', '_right'))
print("\nPitfall 5: Merge with duplicates")
print("Duplicates create Cartesian product:")
print(MergedDuplicates)

# ============================================================================
# BEST PRACTICES SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("BEST PRACTICES SUMMARY")
print("=" * 80)

print("""
1. USE CORRECT DTYPES:
   - Use int32/int16 instead of int64 when possible
   - Use category for low-cardinality strings
   - Use nullable integer types (Int64) when needed

2. AVOID LOOPS:
   - Prefer vectorized operations
   - Use apply() only when necessary
   - Never use iterrows() for large DataFrames

3. PRE-ALLOCATE:
   - Collect results in list, then concat once
   - Avoid appending to DataFrame in loops

4. USE CATEGORICAL FOR LOW-CARDINALITY STRINGS:
   - Significant memory savings
   - Faster operations

5. USE .loc/.iloc FOR ASSIGNMENTS:
   - Avoid chained indexing
   - Use .copy() when needed

6. MONITOR MEMORY:
   - Use memory_usage(deep=True)
   - Downcast when possible

7. USE METHOD CHAINING:
   - More readable code
   - Better performance with pipe()

8. AVOID SETTINGWITHCOPYWARNING:
   - Use .copy() or .loc appropriately
   - Understand when views vs copies are created
""")

print("\n" + "=" * 80)
print("END OF PERFORMANCE AND BEST PRACTICES DEMONSTRATION")
print("=" * 80)
