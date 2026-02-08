# Pandas Foundations: Series, DataFrame, and DTypes

## Table of Contents
1. [What is Pandas](#what-is-pandas)
2. [Series](#series)
3. [DataFrame](#dataframe)
4. [Index Objects](#index-objects)
5. [DType System](#dtype-system)
6. [Memory Layout](#memory-layout)

---

## What is Pandas

Pandas is a powerful open-source data analysis and manipulation library built on top of NumPy. It provides high-level data structures and tools designed to make working with structured data fast, easy, and expressive.

### Built on NumPy

Pandas leverages NumPy's efficient array operations and extends them with labeled data structures. While NumPy provides homogeneous n-dimensional arrays, Pandas introduces labeled axes and heterogeneous data types within a single structure.

```python
import pandas as pd
import numpy as np

# NumPy array: homogeneous, unlabeled
arr = np.array([1, 2, 3, 4, 5])

# Pandas Series: labeled, can be heterogeneous
series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
```

### Labeled Data Structures

The core innovation of Pandas is labeling. Every axis (rows and columns) can have meaningful labels, making data manipulation intuitive and less error-prone than positional indexing.

```python
# Instead of arr[0], you can use df.loc['row_label']
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, 
                   index=['row1', 'row2', 'row3'])
print(df.loc['row1', 'A'])  # 1
```

### Data Alignment

Pandas automatically aligns data based on labels during operations. This means operations between Series or DataFrames align on their index/columns before computation.

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20], index=['a', 'c'])
print(s1 + s2)
# a    11.0
# b     NaN
# c    23.0
```

### Integrated Indexing

Pandas provides multiple indexing paradigms:
- Label-based indexing (`loc`)
- Position-based indexing (`iloc`)
- Fast scalar access (`at`, `iat`)
- Boolean indexing
- Fancy indexing

### Series vs DataFrame vs Panel

**Series**: One-dimensional labeled array. Think of it as a single column of data with an index.

```python
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(type(s))  # <class 'pandas.core.series.Series'>
```

**DataFrame**: Two-dimensional labeled data structure with columns of potentially different types. This is the most commonly used Pandas structure.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(type(df))  # <class 'pandas.core.frame.DataFrame'>
```

**Panel**: Three-dimensional structure (deprecated since Pandas 0.25.0). Use MultiIndex DataFrames or xarray instead.

### When to Use Pandas vs NumPy vs SQL

| Tool | Best For | Characteristics |
|------|----------|----------------|
| **NumPy** | Numerical computations, homogeneous arrays, mathematical operations | Fast, memory-efficient, no labels |
| **Pandas** | Structured data analysis, time series, heterogeneous data, data cleaning | Labeled axes, missing data handling, rich operations |
| **SQL** | Large-scale data storage, relational queries, concurrent access | Persistent storage, ACID properties, query optimization |

**Use Pandas when:**
- Working with tabular data (CSV, Excel, databases)
- Need labeled axes for clarity
- Performing data cleaning and transformation
- Working with time series data
- Data fits in memory

**Use NumPy when:**
- Pure numerical computations
- Need maximum performance for homogeneous arrays
- Working with multi-dimensional arrays (3D+)
- Building machine learning models (often convert back to NumPy)

**Use SQL when:**
- Data is too large for memory
- Need persistent storage
- Multiple users need concurrent access
- Complex relational queries

---

## Series

A Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the index.

### Creation

#### From List

```python
# Simple list creates default integer index
s = pd.Series([1, 2, 3, 4, 5])
print(s)
# 0    1
# 1    2
# 2    3
# 3    4
# 4    5
# dtype: int64

# With custom index
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
```

#### From Dictionary

```python
# Dictionary keys become index
d = {'a': 1, 'b': 2, 'c': 3}
s = pd.Series(d)
print(s)
# a    1
# b    2
# c    3
# dtype: int64

# Can specify index separately (missing keys become NaN)
s = pd.Series(d, index=['a', 'b', 'c', 'd'])
# a    1.0
# b    2.0
# c    3.0
# d    NaN
```

#### From NumPy Array

```python
arr = np.array([1, 2, 3, 4, 5])
s = pd.Series(arr, index=['a', 'b', 'c', 'd', 'e'])
```

#### From Scalar

```python
# Scalar value is broadcast to all index labels
s = pd.Series(5, index=['a', 'b', 'c', 'd'])
print(s)
# a    5
# b    5
# c    5
# d    5
# dtype: int64
```

### Attributes

#### .values

Returns the underlying NumPy array representation.

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s.values)  # [1 2 3 4 5]
print(type(s.values))  # <class 'numpy.ndarray'>
```

#### .index

Returns the index object.

```python
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s.index)  # Index(['a', 'b', 'c'], dtype='object')
print(s.index[0])  # 'a'
```

#### .dtype

Returns the data type of the Series.

```python
s = pd.Series([1, 2, 3])
print(s.dtype)  # int64

s = pd.Series([1.0, 2.0, 3.0])
print(s.dtype)  # float64

s = pd.Series(['a', 'b', 'c'])
print(s.dtype)  # object
```

#### .name

The name of the Series (useful when Series becomes a DataFrame column).

```python
s = pd.Series([1, 2, 3], name='my_series')
print(s.name)  # 'my_series'
```

#### .shape

Returns a tuple representing the dimensionality (always 1D for Series).

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s.shape)  # (5,)
```

#### .size

Returns the number of elements.

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s.size)  # 5
```

#### .nbytes

Returns the number of bytes consumed by the data.

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s.nbytes)  # 40 (5 elements * 8 bytes per int64)
```

#### .is_unique

Returns True if all values in the Series are unique.

```python
s1 = pd.Series([1, 2, 3, 4, 5])
print(s1.is_unique)  # True

s2 = pd.Series([1, 2, 2, 3, 4])
print(s2.is_unique)  # False
```

#### .is_monotonic_increasing

Returns True if values are monotonically increasing.

```python
s1 = pd.Series([1, 2, 3, 4, 5])
print(s1.is_monotonic_increasing)  # True

s2 = pd.Series([1, 3, 2, 4, 5])
print(s2.is_monotonic_increasing)  # False
```

### Basic Operations

#### Arithmetic Operations

Series support element-wise arithmetic operations with automatic alignment.

```python
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# Addition
print(s1 + s2)
# a    11.0
# b    22.0
# c    33.0
# d     NaN

# Multiplication
print(s1 * 2)
# a     2
# b     4
# c     6
# d     8
```

#### Vectorized String Operations

Access string methods via the `.str` accessor.

```python
s = pd.Series(['apple', 'banana', 'cherry'])
print(s.str.upper())
# 0      APPLE
# 1     BANANA
# 2     CHERRY

print(s.str.len())
# 0    5
# 1    6
# 2    6
```

#### Boolean Operations

```python
s = pd.Series([1, 2, 3, 4, 5])
print(s > 3)
# 0    False
# 1    False
# 2    False
# 3     True
# 4     True

# Boolean indexing
print(s[s > 3])
# 3    4
# 4    5
```

### Common Methods

#### .head() and .tail()

Return the first/last n elements (default n=5).

```python
s = pd.Series(range(10))
print(s.head(3))
# 0    0
# 1    1
# 2    2

print(s.tail(3))
# 7    7
# 8    8
# 9    9
```

#### .sample()

Return random sample of elements.

```python
s = pd.Series(range(10))
print(s.sample(3))
# Returns 3 random elements
```

#### .describe()

Generate descriptive statistics.

```python
s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(s.describe())
# count    10.000000
# mean      5.500000
# std       3.027650
# min       1.000000
# 25%       3.250000
# 50%       5.500000
# 75%       7.750000
# max      10.000000
```

### Series as Dict-like and Ndarray-like Behavior

Series can be used like dictionaries and NumPy arrays.

```python
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

# Dict-like access
print(s['a'])  # 1
print('a' in s)  # True
print(s.get('a', 0))  # 1

# Ndarray-like access
print(s[0])  # 1
print(s[[0, 2, 4]])  # Fancy indexing
print(s[0:3])  # Slicing
```

---

## DataFrame

A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. It is similar to a spreadsheet or SQL table.

### Creation

#### From Dictionary of Lists

```python
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
print(df)
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9
```

#### From Dictionary of Series

```python
s1 = pd.Series([1, 2, 3])
s2 = pd.Series([4, 5, 6])
df = pd.DataFrame({'A': s1, 'B': s2})
```

#### From List of Dictionaries

```python
data = [{'A': 1, 'B': 4}, {'A': 2, 'B': 5}, {'A': 3, 'B': 6}]
df = pd.DataFrame(data)
```

#### From 2D NumPy Array

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

#### From Records (List of Tuples)

```python
records = [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
df = pd.DataFrame.from_records(records, columns=['A', 'B', 'C'])
```

#### From CSV/Dict

```python
# From CSV
df = pd.read_csv('file.csv')

# From dictionary (useful for API responses)
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame.from_dict(data)
```

### Attributes

#### .values

Returns the underlying NumPy array (may lose dtype information).

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.values)
# [[1 4]
#  [2 5]
#  [3 6]]
```

#### .index

Returns the row index.

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
print(df.index)  # Index(['a', 'b', 'c'], dtype='object')
```

#### .columns

Returns the column index.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.columns)  # Index(['A', 'B'], dtype='object')
```

#### .dtypes

Returns a Series with data type of each column.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0], 'C': ['a', 'b', 'c']})
print(df.dtypes)
# A      int64
# B    float64
# C     object
```

#### .shape

Returns a tuple (rows, columns).

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.shape)  # (3, 2)
```

#### .size

Returns the total number of elements (rows × columns).

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.size)  # 6
```

#### .ndim

Returns the number of dimensions (always 2 for DataFrame).

```python
df = pd.DataFrame({'A': [1, 2, 3]})
print(df.ndim)  # 2
```

#### .axes

Returns a list of the row and column axes.

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
print(df.axes)
# [Index(['a', 'b', 'c'], dtype='object'), Index(['A'], dtype='object')]
```

#### .empty

Returns True if DataFrame is empty.

```python
df1 = pd.DataFrame()
print(df1.empty)  # True

df2 = pd.DataFrame({'A': [1]})
print(df2.empty)  # False
```

### .info() Method

Provides a concise summary of the DataFrame including dtypes, non-null counts, and memory usage.

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4.0, 5.0, 6.0], 'C': ['a', 'b', 'c']})
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 3 entries, 0 to 2
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   A       2 non-null      float64
#  1   B       3 non-null      float64
#  2   C       3 non-null      object
# dtypes: float64(2), object(1)
# memory usage: 200.0 bytes
```

### .describe() Method

Generates descriptive statistics for numeric columns.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
print(df.describe())
#               A          B
# count  5.000000   5.000000
# mean   3.000000  30.000000
# std    1.581139  15.811388
# min    1.000000  10.000000
# 25%    2.000000  20.000000
# 50%    3.000000  30.000000
# 75%    4.000000  40.000000
# max    5.000000  50.000000

# Include all dtypes
print(df.describe(include='all'))

# Exclude certain dtypes
print(df.describe(exclude=[np.number]))
```

### .memory_usage()

Returns memory usage of each column.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
print(df.memory_usage())
# Index    128
# A         24
# B         24

# Deep memory usage (includes object references)
print(df.memory_usage(deep=True))
```

### Column Access

#### Single Column

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Dictionary-style
print(df['A'])
# 0    1
# 1    2
# 2    3

# Attribute-style (only works if column name is valid Python identifier)
print(df.A)
```

#### Multiple Columns

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# Returns DataFrame
print(df[['A', 'C']])
#    A  C
# 0  1  7
# 1  2  8
# 2  3  9
```

### Adding Columns

#### Direct Assignment

```python
df = pd.DataFrame({'A': [1, 2, 3]})
df['B'] = [4, 5, 6]
df['C'] = df['A'] + df['B']
```

#### .assign()

Returns a new DataFrame with added columns (doesn't modify original).

```python
df = pd.DataFrame({'A': [1, 2, 3]})
df_new = df.assign(B=[4, 5, 6], C=lambda x: x['A'] * 2)
```

#### .insert()

Insert column at specific position.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
df.insert(1, 'B', [4, 5, 6])  # Insert at position 1
```

---

## Index Objects

The Index is an immutable array-like object that provides the axis labels for Series and DataFrames.

### RangeIndex

Default index type, memory-efficient for sequential integer indices.

```python
df = pd.DataFrame({'A': [1, 2, 3]})
print(type(df.index))  # <class 'pandas.core.indexes.range.RangeIndex'>
print(df.index)  # RangeIndex(start=0, stop=3, step=1)
```

### Int64Index and Float64Index

Deprecated in favor of generic `Index` with dtype specification.

```python
# Old way (deprecated)
# idx = pd.Int64Index([1, 2, 3])

# New way
idx = pd.Index([1, 2, 3], dtype='int64')
```

### DatetimeIndex

Specialized index for datetime data, enables time-series operations.

```python
dates = pd.date_range('2023-01-01', periods=5, freq='D')
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=dates)
print(type(df.index))  # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

# Time-series operations
print(df.index.year)  # [2023, 2023, 2023, 2023, 2023]
print(df.index.month)  # [1, 1, 1, 1, 1]
```

### PeriodIndex

Index for periods (year, quarter, month, etc.).

```python
periods = pd.period_range('2023-01', periods=5, freq='M')
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=periods)
print(type(df.index))  # <class 'pandas.core.indexes.periods.PeriodIndex'>
```

### TimedeltaIndex

Index for durations.

```python
deltas = pd.timedelta_range('1 day', periods=5)
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=deltas)
print(type(df.index))  # <class 'pandas.core.indexes.timedeltas.TimedeltaIndex'>
```

### CategoricalIndex

Index with categorical data (memory efficient for repeated values).

```python
idx = pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b'])
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=idx)
print(type(df.index))  # <class 'pandas.core.indexes.category.CategoricalIndex'>
```

### MultiIndex

Hierarchical index allowing multiple index levels.

#### From Tuples

```python
tuples = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
idx = pd.MultiIndex.from_tuples(tuples, names=['level1', 'level2'])
df = pd.DataFrame({'data': [1, 2, 3, 4]}, index=idx)
```

#### From Arrays

```python
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
idx = pd.MultiIndex.from_arrays(arrays, names=['level1', 'level2'])
```

#### From Product

```python
idx = pd.MultiIndex.from_product([['A', 'B'], [1, 2]], names=['level1', 'level2'])
```

#### Levels and Codes

```python
idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)])
print(idx.levels)  # [['A', 'B'], [1, 2]]
print(idx.codes)  # [[0, 0, 1], [0, 1, 0]]
```

### Index Attributes

#### .name and .names

```python
# Single index
idx = pd.Index([1, 2, 3], name='my_index')
print(idx.name)  # 'my_index'

# MultiIndex
idx = pd.MultiIndex.from_tuples([('A', 1), ('B', 2)], names=['level1', 'level2'])
print(idx.names)  # ['level1', 'level2']
```

#### .nlevels

Number of levels (1 for regular Index, >1 for MultiIndex).

```python
idx1 = pd.Index([1, 2, 3])
print(idx1.nlevels)  # 1

idx2 = pd.MultiIndex.from_tuples([('A', 1), ('B', 2)])
print(idx2.nlevels)  # 2
```

#### .is_unique

Returns True if all values are unique.

```python
idx1 = pd.Index([1, 2, 3])
print(idx1.is_unique)  # True

idx2 = pd.Index([1, 2, 2])
print(idx2.is_unique)  # False
```

#### .is_monotonic

Returns True if values are monotonically increasing.

```python
idx1 = pd.Index([1, 2, 3, 4])
print(idx1.is_monotonic_increasing)  # True

idx2 = pd.Index([1, 3, 2, 4])
print(idx2.is_monotonic_increasing)  # False
```

### Index Operations

#### Set Operations

```python
idx1 = pd.Index([1, 2, 3, 4])
idx2 = pd.Index([3, 4, 5, 6])

# Union
print(idx1.union(idx2))  # Int64Index([1, 2, 3, 4, 5, 6])

# Intersection
print(idx1.intersection(idx2))  # Int64Index([3, 4])

# Difference
print(idx1.difference(idx2))  # Int64Index([1, 2])

# Symmetric difference
print(idx1.symmetric_difference(idx2))  # Int64Index([1, 2, 5, 6])
```

---

## DType System

Pandas uses NumPy's dtype system with extensions for missing data and categorical data.

### Numeric Types

#### Integer Types

| Type | Size | Range |
|------|------|-------|
| int8 | 1 byte | -128 to 127 |
| int16 | 2 bytes | -32,768 to 32,767 |
| int32 | 4 bytes | -2,147,483,648 to 2,147,483,647 |
| int64 | 8 bytes | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
| uint8 | 1 byte | 0 to 255 |
| uint16 | 2 bytes | 0 to 65,535 |
| uint32 | 4 bytes | 0 to 4,294,967,295 |
| uint64 | 8 bytes | 0 to 18,446,744,073,709,551,615 |

```python
s = pd.Series([1, 2, 3], dtype='int8')
print(s.dtype)  # int8
```

#### Floating Point Types

| Type | Size | Precision |
|------|------|-----------|
| float16 | 2 bytes | Half precision |
| float32 | 4 bytes | Single precision |
| float64 | 8 bytes | Double precision (default) |

```python
s = pd.Series([1.0, 2.0, 3.0], dtype='float32')
print(s.dtype)  # float32
```

### Boolean Types

#### bool

Standard boolean type (True/False).

```python
s = pd.Series([True, False, True])
print(s.dtype)  # bool
```

#### BooleanDtype (Nullable Boolean)

Extension dtype that supports missing values.

```python
s = pd.Series([True, False, None], dtype='boolean')
print(s.dtype)  # boolean
print(s)
# 0     True
# 1    False
# 2     <NA>
```

### String Types

#### object (Default)

Stores strings as Python objects (flexible but memory-intensive).

```python
s = pd.Series(['a', 'b', 'c'])
print(s.dtype)  # object
```

#### StringDtype (Extension Type)

Dedicated string dtype (Pandas 1.0+).

```python
s = pd.Series(['a', 'b', 'c'], dtype=pd.StringDtype())
print(s.dtype)  # string
```

### Categorical Type

Memory-efficient for data with repeated values.

```python
# Create categorical
s = pd.Series(['a', 'b', 'c', 'a', 'b'], dtype='category')
print(s.dtype)  # category

# Ordered categorical
cat = pd.Categorical(['low', 'medium', 'high'], ordered=True)
s = pd.Series(cat)

# Access categories
print(s.cat.categories)  # Index(['low', 'medium', 'high'])

# Memory savings
s1 = pd.Series(['a', 'b', 'c', 'a', 'b'] * 1000)  # object dtype
s2 = pd.Series(['a', 'b', 'c', 'a', 'b'] * 1000, dtype='category')
print(s1.memory_usage(deep=True) > s2.memory_usage(deep=True))  # True
```

### Datetime Types

#### datetime64[ns]

Default datetime dtype (nanosecond precision).

```python
s = pd.Series(pd.date_range('2023-01-01', periods=3))
print(s.dtype)  # datetime64[ns]
```

#### DatetimeTZDtype

Timezone-aware datetime.

```python
s = pd.Series(pd.date_range('2023-01-01', periods=3, tz='UTC'))
print(s.dtype)  # datetime64[ns, UTC]
```

### Timedelta Type

```python
s = pd.Series(pd.timedelta_range('1 day', periods=3))
print(s.dtype)  # timedelta64[ns]
```

### Nullable Integer Types

Extension dtypes that support missing values (capital I).

```python
s = pd.Series([1, 2, None], dtype='Int64')
print(s.dtype)  # Int64
print(s)
# 0       1
# 1       2
# 2    <NA>
```

### pd.NA vs np.nan

**np.nan**: Float-based missing value marker.

```python
s = pd.Series([1.0, np.nan, 3.0])
print(s.isna())
# 0    False
# 1     True
# 2    False
```

**pd.NA**: New unified missing value marker (Pandas 1.0+).

```python
s = pd.Series([1, pd.NA, 3], dtype='Int64')
print(s)
# 0       1
# 1    <NA>
# 2       3
```

### Type Conversion

#### .astype()

Convert to specified dtype.

```python
s = pd.Series([1, 2, 3])
s_float = s.astype('float64')
s_str = s.astype('str')
```

#### pd.to_numeric()

Convert to numeric, handling errors.

```python
s = pd.Series(['1', '2', '3', 'invalid'])
s_num = pd.to_numeric(s, errors='coerce')  # Invalid becomes NaN
```

#### pd.to_datetime()

Convert to datetime.

```python
s = pd.Series(['2023-01-01', '2023-01-02'])
s_dt = pd.to_datetime(s)
```

#### pd.to_timedelta()

Convert to timedelta.

```python
s = pd.Series(['1 day', '2 days'])
s_td = pd.to_timedelta(s)
```

#### convert_dtypes()

Convert to best possible dtypes (including nullable types).

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': ['a', 'b', 'c']})
df_new = df.convert_dtypes()
print(df_new.dtypes)
# A    Int64
# B    string
```

### Dtype Inference

#### infer_objects()

Attempt to infer better dtypes for object columns.

```python
df = pd.DataFrame({'A': [1, 2, 3]}, dtype='object')
df_new = df.infer_objects()
print(df_new.dtypes['A'])  # int64
```

---

## Memory Layout

### BlockManager Internals

Pandas DataFrames store data in blocks organized by dtype. This allows efficient operations on homogeneous columns.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0], 'C': ['a', 'b', 'c']})

# Access internal blocks (advanced)
print(df._mgr.blocks)
# [FloatBlock: [A], 1 x 3, dtype: int64,
#  FloatBlock: [B], 1 x 3, dtype: float64,
#  ObjectBlock: [C], 1 x 3, dtype: object]
```

### Memory Optimization Overview

**Strategies:**

1. **Use appropriate dtypes**: Use int8 instead of int64 when values fit.
2. **Use categoricals**: For repeated string values.
3. **Use nullable types**: When you need to represent missing data efficiently.
4. **Downcast numeric types**: Use `pd.to_numeric()` with `downcast` parameter.

```python
# Example: Downcast integers
s = pd.Series([1, 2, 3], dtype='int64')
s_downcast = pd.to_numeric(s, downcast='integer')
print(s_downcast.dtype)  # int8 (if values fit)

# Example: Use categorical for repeated strings
s = pd.Series(['a', 'b', 'c', 'a', 'b'] * 1000)
s_cat = s.astype('category')
print(s.memory_usage(deep=True) > s_cat.memory_usage(deep=True))  # True
```

### Memory Usage Analysis

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})

# Per-column memory usage
print(df.memory_usage(deep=True))
# Index    128
# A         24
# B         24

# Total memory
print(df.memory_usage(deep=True).sum())
```

---

## Summary

This foundation covers the core building blocks of Pandas:

- **Series**: One-dimensional labeled arrays
- **DataFrame**: Two-dimensional labeled data structures
- **Index Objects**: Various index types for different use cases
- **DType System**: Understanding and working with data types
- **Memory Layout**: How Pandas stores and optimizes data

Mastering these fundamentals is essential for effective data manipulation and analysis with Pandas.
