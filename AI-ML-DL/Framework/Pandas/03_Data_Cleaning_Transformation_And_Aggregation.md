# Pandas Data Cleaning, Transformation, and Aggregation

## Table of Contents
1. [Missing Data](#missing-data)
2. [String Methods (.str accessor)](#string-methods-str-accessor)
3. [Apply, Map, Transform](#apply-map-transform)
4. [GroupBy (Split-Apply-Combine)](#groupby-split-apply-combine)
5. [Pivot and Reshape](#pivot-and-reshape)
6. [Merge, Join, Concat](#merge-join-concat)
7. [Other Transformations](#other-transformations)

---

## Missing Data

Missing data is a common issue in real-world datasets. Pandas provides comprehensive tools for detecting, handling, and filling missing values.

### Types of Missing Data

#### NaN (Not a Number)

Float-based missing value marker. Used by default for numeric columns.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, 2, np.nan, 4]})
print(df)
#      A
# 0  1.0
# 1  2.0
# 2  NaN
# 3  4.0
```

#### None

Python's None object. Used for object dtype columns.

```python
df = pd.DataFrame({'A': [1, 2, None, 4]})
print(df.dtypes)
# A    object
```

#### pd.NaT (Not a Time)

Missing value marker for datetime data.

```python
df = pd.DataFrame({'A': pd.date_range('2023-01-01', periods=3)})
df.loc[1, 'A'] = pd.NaT
print(df)
#            A
# 0 2023-01-01
# 1        NaT
# 2 2023-01-03
```

#### pd.NA (Unified Missing Value)

New unified missing value marker (Pandas 1.0+). Works with nullable dtypes.

```python
df = pd.DataFrame({'A': [1, 2, pd.NA, 4]}, dtype='Int64')
print(df)
#      A
# 0    1
# 1    2
# 2  <NA>
# 3    4
```

### Detection

#### isna() and isnull()

Both methods are identical - check for missing values.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8]})

# Check for missing values
print(df.isna())
#        A      B
# 0  False  False
# 1  False   True
# 2   True  False
# 3  False  False

print(df.isnull())  # Same as isna()
```

#### notna() and notnull()

Inverse of isna() - check for non-missing values.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, 4]})

print(df.notna())
#       A
# 0  True
# 1  True
# 2 False
# 3  True
```

#### .isna().sum()

Count missing values per column.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, np.nan]})

# Count missing per column
print(df.isna().sum())
# A    1
# B    2

# Percentage missing
print(df.isna().sum() / len(df) * 100)
# A    25.0
# B    50.0
```

#### .isna().any()

Check if any missing values exist.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, 4]})

# Check if any missing
print(df.isna().any().any())  # True

# Per column
print(df.isna().any())
# A     True
```

### Removal

#### dropna()

Remove rows or columns with missing values.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, 4], 
                   'B': [5, np.nan, 7, 8],
                   'C': [9, 10, 11, 12]})

# Drop rows with any missing value (default)
df_dropped = df.dropna()
print(df_dropped)
#      A    B   C
# 0  1.0  5.0   9
# 3  4.0  8.0  12

# Drop rows where all values are missing
df_dropped = df.dropna(how='all')

# Drop rows with missing in specific columns
df_dropped = df.dropna(subset=['A', 'B'])

# Threshold: keep rows with at least N non-null values
df_dropped = df.dropna(thresh=2)  # Keep rows with at least 2 non-null

# Drop columns instead of rows
df_dropped = df.dropna(axis=1)  # Drop columns with any missing

# Inplace modification
df.dropna(inplace=True)
```

### Filling

#### fillna()

Fill missing values with specified value or method.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, 4, np.nan], 
                   'B': [5, np.nan, 7, 8, 9]})

# Fill with constant value
df_filled = df.fillna(0)
print(df_filled)
#      A    B
# 0  1.0  5.0
# 1  2.0  0.0
# 2  0.0  7.0
# 3  4.0  8.0
# 4  0.0  9.0

# Fill with different values per column
df_filled = df.fillna({'A': 0, 'B': 999})

# Forward fill (propagate last valid observation)
df_filled = df.fillna(method='ffill')  # or 'pad'
print(df_filled)
#      A    B
# 0  1.0  5.0
# 1  2.0  5.0  # Forward filled from row 0
# 2  2.0  7.0  # Forward filled from row 1
# 3  4.0  8.0
# 4  4.0  9.0  # Forward filled from row 3

# Backward fill (use next valid observation)
df_filled = df.fillna(method='bfill')  # or 'backfill'

# Limit consecutive fills
df_filled = df.fillna(method='ffill', limit=1)  # Fill max 1 consecutive NaN
```

#### interpolate()

Fill missing values using interpolation methods.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, np.nan, 5]})

# Linear interpolation (default)
df_interp = df.interpolate()
print(df_interp)
#      A
# 0  1.0
# 1  2.0
# 2  3.0  # Interpolated: (2 + 5) / 2 = 3.5, but linear between 2 and 5
# 3  4.0
# 4  5.0

# Time-based interpolation
df_time = pd.DataFrame({'A': [1, 2, np.nan, 5]}, 
                       index=pd.date_range('2023-01-01', periods=4))
df_interp = df_time.interpolate(method='time')

# Index-based interpolation
df_interp = df.interpolate(method='index')

# Polynomial interpolation
df_interp = df.interpolate(method='polynomial', order=2)

# Spline interpolation
df_interp = df.interpolate(method='spline', order=2)

# Limit interpolation
df_interp = df.interpolate(limit=1)  # Interpolate max 1 consecutive NaN
```

### Replace

#### replace()

Replace values (including missing values) with other values.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})

# Replace scalar
df_replaced = df.replace(2, 999)
print(df_replaced)
#      A  B
# 0    1  5
# 1  999  6
# 2    3  7
# 3    4  8

# Replace list of values
df_replaced = df.replace([2, 3], 999)

# Replace with dictionary
df_replaced = df.replace({2: 999, 3: 888})

# Replace per column
df_replaced = df.replace({'A': {2: 999}, 'B': {6: 666}})

# Replace with regex
df = pd.DataFrame({'A': ['apple', 'banana', 'cherry']})
df_replaced = df.replace({'A': r'^a'}, {'A': 'fruit'}, regex=True)

# Replace NaN
df = pd.DataFrame({'A': [1, 2, np.nan, 4]})
df_replaced = df.replace(np.nan, 0)
```

---

## String Methods (.str accessor)

Pandas provides vectorized string operations through the `.str` accessor, making string manipulation efficient and intuitive.

### Case Operations

#### lower, upper, title, capitalize, swapcase

```python
s = pd.Series(['Apple', 'BANANA', 'cherry', 'DATE'])

# Lowercase
print(s.str.lower())
# 0     apple
# 1    banana
# 2    cherry
# 3      date

# Uppercase
print(s.str.upper())
# 0     APPLE
# 1    BANANA
# 2    CHERRY
# 3      DATE

# Title case
print(s.str.title())
# 0     Apple
# 1    Banana
# 2    Cherry
# 3      Date

# Capitalize (first letter uppercase, rest lowercase)
print(s.str.capitalize())
# 0     Apple
# 1    Banana
# 2    Cherry
# 3      Date

# Swap case
print(s.str.swapcase())
# 0     aPPLE
# 1    banana
# 2    CHERRY
# 3      date
```

### Whitespace Operations

#### strip, lstrip, rstrip

```python
s = pd.Series(['  apple  ', '  banana', 'cherry  '])

# Strip whitespace from both sides
print(s.str.strip())
# 0     apple
# 1    banana
# 2    cherry

# Left strip
print(s.str.lstrip())
# 0     apple  
# 1    banana
# 2    cherry  

# Right strip
print(s.str.rstrip())
# 0       apple
# 1      banana
# 2      cherry

# Strip specific characters
s = pd.Series(['...apple...', '...banana'])
print(s.str.strip('.'))
# 0     apple
# 1    banana
```

#### pad, center, zfill

```python
s = pd.Series(['apple', 'banana'])

# Pad to width (default: right pad with space)
print(s.str.pad(width=10))
# 0     apple    
# 1    banana    

# Left pad
print(s.str.pad(width=10, side='left'))

# Center
print(s.str.center(width=10))
# 0    apple    
# 1   banana    

# Zero fill (left pad with zeros)
s = pd.Series(['1', '12', '123'])
print(s.str.zfill(5))
# 0    00001
# 1    00012
# 2    00123
```

### Search Operations

#### contains

Check if string contains pattern.

```python
s = pd.Series(['apple', 'banana', 'cherry', 'date'])

# Contains substring
print(s.str.contains('a'))
# 0     True
# 1     True
# 2    False
# 3     True

# Case insensitive
print(s.str.contains('A', case=False))

# Regex
print(s.str.contains('^a', regex=True))  # Starts with 'a'

# na parameter for handling NaN
s = pd.Series(['apple', 'banana', np.nan])
print(s.str.contains('a', na=False))  # NaN becomes False
```

#### startswith, endswith

```python
s = pd.Series(['apple', 'banana', 'cherry'])

# Starts with
print(s.str.startswith('a'))
# 0     True
# 1    False
# 2    False

# Ends with
print(s.str.endswith('y'))
# 0    False
# 1    False
# 2     True
```

#### find, findall

```python
s = pd.Series(['apple', 'banana', 'cherry'])

# Find position of substring (-1 if not found)
print(s.str.find('a'))
# 0    0
# 1    1
# 2   -1

# Find all matches (returns list)
print(s.str.findall('a'))
# 0        [a]
# 1    [a, a]
# 2         []
```

#### match, extract, extractall

```python
s = pd.Series(['apple123', 'banana456', 'cherry789'])

# Match pattern (anchored at start)
print(s.str.match(r'\w+\d+'))
# 0    True
# 1    True
# 2    True

# Extract first match
print(s.str.extract(r'(\w+)(\d+)'))
#        0    1
# 0  apple  123
# 1  banana  456
# 2  cherry  789

# Extract all matches
print(s.str.extractall(r'(\d)'))
#           0
#   match
# 0 0      1
#   1      2
#   2      3
# 1 0      4
#   1      5
#   2      6
```

### Modify Operations

#### replace

Replace substring or pattern.

```python
s = pd.Series(['apple', 'banana', 'cherry'])

# Replace substring
print(s.str.replace('a', 'X'))
# 0     Xpple
# 1     bXnXnX
# 2    cherry

# Replace with regex
print(s.str.replace(r'^a', 'X', regex=True))

# Replace with limit
print(s.str.replace('a', 'X', n=1))  # Replace only first occurrence
```

#### slice

Extract substring by position.

```python
s = pd.Series(['apple', 'banana', 'cherry'])

# Slice
print(s.str.slice(start=0, stop=3))
# 0    app
# 1    ban
# 2    che

# Using indexing syntax
print(s.str[0:3])  # Same as above
```

#### repeat

Repeat strings.

```python
s = pd.Series(['a', 'ab', 'abc'])

# Repeat
print(s.str.repeat(3))
# 0        aaa
# 1    ababab
# 2  abcabcabc
```

#### cat (concatenation)

Concatenate strings.

```python
s = pd.Series(['a', 'b', 'c'])

# Concatenate with separator
print(s.str.cat(sep='-'))
# 'a-b-c'

# Concatenate with other Series
s2 = pd.Series(['1', '2', '3'])
print(s.str.cat(s2, sep=''))
# 0    a1
# 1    b2
# 2    c3

# Concatenate with list
print(s.str.cat(['1', '2', '3'], sep=''))
```

#### join

Join strings in list.

```python
s = pd.Series([['a', 'b', 'c'], ['d', 'e']])

# Join list elements
print(s.str.join('-'))
# 0    a-b-c
# 1      d-e
```

### Split Operations

#### split, rsplit

```python
s = pd.Series(['a-b-c', 'd-e-f', 'g-h'])

# Split
print(s.str.split('-'))
# 0    [a, b, c]
# 1    [d, e, f]
# 2       [g, h]

# Split with max splits
print(s.str.split('-', n=1))  # Max 1 split
# 0    [a, b-c]
# 1    [d, e-f]
# 2       [g, h]

# Right split
print(s.str.rsplit('-', n=1))
```

#### partition, rpartition

Split into three parts (before, separator, after).

```python
s = pd.Series(['a-b-c', 'd-e'])

# Partition
print(s.str.partition('-'))
#     0  1  2
# 0   a  -  b-c
# 1   d  -  e

# Right partition
print(s.str.rpartition('-'))
```

#### get

Get element from split result.

```python
s = pd.Series(['a-b-c', 'd-e-f'])

# Get first element after split
print(s.str.split('-').str.get(0))
# 0    a
# 1    d

# Using indexing
print(s.str.split('-').str[0])  # Same as above
```

### Information Operations

#### len

Get string length.

```python
s = pd.Series(['apple', 'banana', 'cherry'])

print(s.str.len())
# 0    5
# 1    6
# 2    6
```

#### count

Count occurrences of substring.

```python
s = pd.Series(['apple', 'banana', 'cherry'])

print(s.str.count('a'))
# 0    1
# 1    3
# 2    0
```

#### isdigit, isalpha, isnumeric, isdecimal

```python
s = pd.Series(['123', 'abc', '12a', '12.3'])

# Is digit
print(s.str.isdigit())
# 0     True
# 1    False
# 2    False
# 3    False

# Is alpha
print(s.str.isalpha())
# 0    False
# 1     True
# 2    False
# 3    False

# Is numeric
print(s.str.isnumeric())
# 0     True
# 1    False
# 2    False
# 3    False

# Is decimal
print(s.str.isdecimal())
```

---

## Apply, Map, Transform

These methods allow you to apply functions to data, providing flexibility for custom transformations.

### Series.map

Element-wise mapping with function or dictionary.

```python
s = pd.Series([1, 2, 3, 4])

# Map with function
mapped = s.map(lambda x: x * 2)
print(mapped)
# 0    2
# 1    4
# 2    6
# 3    8

# Map with dictionary
mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
mapped = s.map(mapping)
print(mapped)
# 0      one
# 1      two
# 2    three
# 3     four

# Missing values in mapping become NaN
mapping = {1: 'one', 2: 'two'}
mapped = s.map(mapping)
print(mapped)
# 0      one
# 1      two
# 2     NaN
# 3     NaN
```

### Series.apply

Element-wise application with access to index.

```python
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

# Apply function
result = s.apply(lambda x: x ** 2)
print(result)
# a     1
# b     4
# c     9
# d    16

# Apply with index access
result = s.apply(lambda x, idx: f"{idx}:{x}", args=('prefix',))
# Or use a function that receives the value
def square_with_name(val):
    return val ** 2

result = s.apply(square_with_name)
```

### DataFrame.apply

Apply function along axis (rows or columns).

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# Apply along columns (axis=0, default)
result = df.apply(np.sum)
print(result)
# A     6
# B    15
# C    24

# Apply along rows (axis=1)
result = df.apply(np.sum, axis=1)
print(result)
# 0    12
# 1    15
# 2    18

# Apply with custom function
def range_func(series):
    return series.max() - series.min()

result = df.apply(range_func)
print(result)
# A    2
# B    2
# C    2

# Apply with result_type
def list_func(series):
    return [series.min(), series.max()]

result = df.apply(list_func, result_type='expand')
# Returns DataFrame with expanded results
```

### DataFrame.applymap / DataFrame.map

Element-wise application to DataFrame (renamed in Pandas 2.1+).

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Element-wise (applymap in older versions, map in 2.1+)
result = df.map(lambda x: x * 2)
print(result)
#    A   B
# 0  2   8
# 1  4  10
# 2  6  12

# With na_action
df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, 5, 6]})
result = df.map(lambda x: x * 2, na_action='ignore')  # Ignore NaN
```

### np.vectorize and np.where

NumPy alternatives for element-wise operations.

```python
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Vectorized function
def custom_func(x):
    return x ** 2 if x > 2 else x

vectorized_func = np.vectorize(custom_func)
result = vectorized_func(df['A'])
print(result)
# [1 2 9]

# np.where for conditional
df['C'] = np.where(df['A'] > 2, 'high', 'low')
print(df)
#    A  B     C
# 0  1  4   low
# 1  2  5   low
# 2  3  6  high
```

### .pipe()

Method chaining with custom functions.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Pipe for method chaining
def add_column(df, col_name, values):
    df[col_name] = values
    return df

result = df.pipe(add_column, 'C', [7, 8, 9])
print(result)
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9

# Chain multiple operations
result = (df
          .pipe(add_column, 'C', [7, 8, 9])
          .pipe(lambda x: x[x['A'] > 1]))
```

---

## GroupBy (Split-Apply-Combine)

GroupBy is one of Pandas' most powerful features, implementing the split-apply-combine pattern for data analysis.

### Grouping

#### By Column

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo'],
                   'B': [1, 2, 3, 4, 5],
                   'C': [10, 20, 30, 40, 50]})

# Group by single column
grouped = df.groupby('A')
print(grouped.groups)
# {'bar': [1, 3], 'foo': [0, 2, 4]}

# Group by multiple columns
grouped = df.groupby(['A', 'B'])
```

#### By Function

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# Group by function result
grouped = df.groupby(lambda x: x % 2)  # Group by even/odd index
```

#### By Index Level

```python
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]},
                  index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2), ('y', 1), ('y', 2)]))

# Group by index level
grouped = df.groupby(level=0)
```

### GroupBy Object

#### .groups

Dictionary mapping group names to row indices.

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo'], 'B': [1, 2, 3]})
grouped = df.groupby('A')
print(grouped.groups)
# {'bar': [1], 'foo': [0, 2]}
```

#### .ngroups

Number of groups.

```python
grouped = df.groupby('A')
print(grouped.ngroups)  # 2
```

#### .get_group()

Get a specific group.

```python
grouped = df.groupby('A')
foo_group = grouped.get_group('foo')
print(foo_group)
#      A  B
# 0  foo  1
# 2  foo  3
```

#### .size()

Get size of each group.

```python
grouped = df.groupby('A')
print(grouped.size())
# A
# bar    1
# foo    2
```

#### .describe()

Statistical summary for each group.

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [1, 2, 3, 4],
                   'C': [10, 20, 30, 40]})
grouped = df.groupby('A')
print(grouped.describe())
```

### Aggregation

#### .agg() / .aggregate()

Apply aggregation functions.

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [1, 2, 3, 4],
                   'C': [10, 20, 30, 40]})
grouped = df.groupby('A')

# Single function
result = grouped.agg(np.sum)
print(result)
#        B   C
# A
# bar    6  60
# foo    4  40

# Multiple functions
result = grouped.agg([np.sum, np.mean])
print(result)
#        B        C
#     sum mean  sum mean
# A
# bar   6  3.0   60  30.0
# foo   4  2.0   40  20.0

# Different functions per column
result = grouped.agg({'B': np.sum, 'C': np.mean})
print(result)
#        B     C
# A
# bar    6  30.0
# foo    4  20.0

# Named aggregation (Pandas 0.25+)
result = grouped.agg(
    B_sum=('B', 'sum'),
    C_mean=('C', 'mean')
)
```

### Transformation

#### .transform()

Returns same-shaped result, useful for group-wise normalization.

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [1, 2, 3, 4]})
grouped = df.groupby('A')

# Transform (returns same shape)
result = grouped.transform(lambda x: x - x.mean())
print(result)
#      B
# 0 -1.0
# 1 -1.0
# 2  1.0
# 3  1.0

# Common transformations
result = grouped.transform('sum')  # Group sum
result = grouped.transform('mean')  # Group mean
result = grouped.transform(lambda x: x / x.sum())  # Normalize within group
```

### Filtration

#### .filter()

Keep groups meeting condition.

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo'],
                   'B': [1, 2, 3, 4, 5]})
grouped = df.groupby('A')

# Filter groups
result = grouped.filter(lambda x: len(x) > 1)  # Keep groups with more than 1 row
print(result)
#      A  B
# 0  foo  1
# 2  foo  3
# 4  foo  5
```

### Iteration

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [1, 2, 3, 4]})
grouped = df.groupby('A')

# Iterate over groups
for name, group in grouped:
    print(f"Group: {name}")
    print(group)
    print()
# Group: bar
#      A  B
# 1  bar  2
# 3  bar  4
# 
# Group: foo
#      A  B
# 0  foo  1
# 2  foo  3
```

### Common Patterns

#### Group-wise Normalization

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [1, 2, 3, 4]})
grouped = df.groupby('A')
df['B_normalized'] = grouped['B'].transform(lambda x: (x - x.mean()) / x.std())
```

#### Cumulative Within Groups

```python
df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'],
                   'B': [1, 2, 3, 4]})
grouped = df.groupby('A')
df['B_cumsum'] = grouped['B'].cumsum()
```

#### Rank Within Groups

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [3, 1, 4, 2]})
grouped = df.groupby('A')
df['B_rank'] = grouped['B'].rank()
```

#### First/Last/Nth

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [1, 2, 3, 4]})
grouped = df.groupby('A')

# First
first = grouped.first()

# Last
last = grouped.last()

# Nth
nth = grouped.nth(0)  # First row of each group
```

---

## Pivot and Reshape

Reshaping data is crucial for analysis. Pandas provides multiple methods for pivoting and restructuring data.

### pivot

Reshape data by index/columns (no aggregation, requires unique index-column pairs).

```python
df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'],
                   'B': ['one', 'two', 'one', 'two'],
                   'C': [1, 2, 3, 4]})

# Pivot
pivoted = df.pivot(index='A', columns='B', values='C')
print(pivoted)
# B    one  two
# A
# bar    3    4
# foo    1    2

# Without values (creates MultiIndex columns)
pivoted = df.pivot(index='A', columns='B')
```

### pivot_table

Pivot with aggregation (handles duplicate index-column pairs).

```python
df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar', 'foo'],
                   'B': ['one', 'two', 'one', 'two', 'one'],
                   'C': [1, 2, 3, 4, 5]})

# Pivot table with aggregation
pivoted = df.pivot_table(index='A', columns='B', values='C', aggfunc='mean')
print(pivoted)
# B    one  two
# A
# bar  3.0  4.0
# foo  3.0  2.0  # (1+5)/2 = 3.0

# Multiple aggregation functions
pivoted = df.pivot_table(index='A', columns='B', values='C', 
                         aggfunc=['mean', 'sum'])

# Margins (totals)
pivoted = df.pivot_table(index='A', columns='B', values='C', 
                         aggfunc='mean', margins=True)

# Fill missing values
pivoted = df.pivot_table(index='A', columns='B', values='C', 
                         aggfunc='mean', fill_value=0)

# Drop missing
pivoted = df.pivot_table(index='A', columns='B', values='C', 
                         aggfunc='mean', dropna=False)
```

### crosstab

Frequency table (cross-tabulation).

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': ['one', 'one', 'two', 'two']})

# Crosstab
ct = pd.crosstab(df['A'], df['B'])
print(ct)
# B    one  two
# A
# bar    1    1
# foo    1    1

# Normalize
ct_norm = pd.crosstab(df['A'], df['B'], normalize='index')  # Normalize by row
ct_norm = pd.crosstab(df['A'], df['B'], normalize='columns')  # Normalize by column
ct_norm = pd.crosstab(df['A'], df['B'], normalize='all')  # Normalize by all
```

### melt

Transform from wide to long format.

```python
df = pd.DataFrame({'A': ['foo', 'bar'],
                   'B': [1, 2],
                   'C': [3, 4],
                   'D': [5, 6]})

# Melt
melted = pd.melt(df, id_vars=['A'], value_vars=['B', 'C', 'D'])
print(melted)
#      A variable  value
# 0  foo        B      1
# 1  bar        B      2
# 2  foo        C      3
# 3  bar        C      4
# 4  foo        D      5
# 5  bar        D      6

# With custom names
melted = pd.melt(df, id_vars=['A'], value_vars=['B', 'C', 'D'],
                 var_name='column_name', value_name='column_value')
```

### stack/unstack

Move levels between index and columns.

#### stack

```python
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])

# Stack (columns become index level)
stacked = df.stack()
print(stacked)
# x  A    1
#    B    3
# y  A    2
#    B    4
```

#### unstack

```python
# Unstack (index level becomes columns)
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, 
                  index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)]))

unstacked = df.unstack()
print(unstacked)
#      A     B
#      1  2  1  2
# x    1  2  3  4

# Unstack specific level
unstacked = df.unstack(level=0)
```

### explode

Expand list-like columns into rows.

```python
df = pd.DataFrame({'A': [1, 2], 'B': [[1, 2, 3], [4, 5]]})

# Explode
exploded = df.explode('B')
print(exploded)
#    A  B
# 0  1  1
# 0  1  2
# 0  1  3
# 1  2  4
# 1  2  5
```

### get_dummies

One-hot encoding for categorical variables.

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo']})

# Get dummies
dummies = pd.get_dummies(df['A'])
print(dummies)
#    bar  foo
# 0    0    1
# 1    1    0
# 2    0    1

# With prefix
dummies = pd.get_dummies(df['A'], prefix='category')

# Multiple columns
df = pd.DataFrame({'A': ['foo', 'bar'], 'B': ['x', 'y']})
dummies = pd.get_dummies(df, columns=['A', 'B'])
```

---

## Merge, Join, Concat

Combining data from multiple sources is a common task. Pandas provides several methods for this.

### pd.concat

Concatenate DataFrames along axis.

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Concatenate along rows (axis=0, default)
result = pd.concat([df1, df2])
print(result)
#    A  B
# 0  1  3
# 1  2  4
# 0  5  7
# 1  6  8

# Concatenate along columns (axis=1)
result = pd.concat([df1, df2], axis=1)
print(result)
#    A  B  A  B
# 0  1  3  5  7
# 1  2  4  6  8

# Join types
df3 = pd.DataFrame({'A': [1, 2], 'C': [9, 10]})
result = pd.concat([df1, df3], join='inner')  # Only common columns
result = pd.concat([df1, df3], join='outer')  # All columns (default)

# Keys for hierarchical index
result = pd.concat([df1, df2], keys=['first', 'second'])
print(result)
#            A  B
# first  0   1  3
#        1   2  4
# second 0   5  7
#        1   6  8

# Ignore index
result = pd.concat([df1, df2], ignore_index=True)
print(result)
#    A  B
# 0  1  3
# 1  2  4
# 2  5  7
# 3  6  8

# Verify integrity
result = pd.concat([df1, df2], verify_integrity=True)  # Raises error if duplicates
```

### pd.merge

SQL-style merge/join operations.

```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# Inner join (default)
result = pd.merge(df1, df2, on='key')
print(result)
#   key  value1  value2
# 0   B       2       4
# 1   C       3       5

# Left join
result = pd.merge(df1, df2, on='key', how='left')
print(result)
#   key  value1  value2
# 0   A       1     NaN
# 1   B       2     4.0
# 2   C       3     5.0

# Right join
result = pd.merge(df1, df2, on='key', how='right')

# Outer join
result = pd.merge(df1, df2, on='key', how='outer')
print(result)
#   key  value1  value2
# 0   A     1.0     NaN
# 1   B     2.0     4.0
# 2   C     3.0     5.0
# 3   D     NaN     6.0

# Different column names
df1 = pd.DataFrame({'key1': ['A', 'B'], 'value1': [1, 2]})
df2 = pd.DataFrame({'key2': ['A', 'B'], 'value2': [3, 4]})
result = pd.merge(df1, df2, left_on='key1', right_on='key2')

# Suffixes for overlapping columns
df1 = pd.DataFrame({'key': ['A', 'B'], 'value': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'B'], 'value': [3, 4]})
result = pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))

# Indicator
result = pd.merge(df1, df2, on='key', how='outer', indicator=True)
print(result)
#   key  value_left  value_right      _merge
# 0   A         1.0         3.0        both
# 1   B         2.0         4.0        both
# 2   C         NaN         5.0  right_only
# 3   D         3.0         NaN   left_only

# Validate merge
result = pd.merge(df1, df2, on='key', validate='one_to_one')  # 'one_to_many', 'many_to_one', 'many_to_many'
```

### DataFrame.join

Index-based merge (convenience method).

```python
df1 = pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])
df2 = pd.DataFrame({'B': [3, 4]}, index=['a', 'c'])

# Join on index
result = df1.join(df2)
print(result)
#    A    B
# a  1  3.0
# b  2  NaN

# How parameter
result = df1.join(df2, how='left')   # Default
result = df1.join(df2, how='right')
result = df1.join(df2, how='outer')
result = df1.join(df2, how='inner')

# Suffixes
df3 = pd.DataFrame({'A': [5, 6]}, index=['a', 'b'])
result = df1.join([df2, df3], lsuffix='_left', rsuffix='_right')
```

### merge_ordered

Merge for time-series data with ordered filling.

```python
df1 = pd.DataFrame({'key': pd.date_range('2023-01-01', periods=3),
                    'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': pd.date_range('2023-01-02', periods=3),
                    'value2': [4, 5, 6]})

# Merge ordered
result = pd.merge_ordered(df1, df2, on='key', fill_method='ffill')
print(result)
#         key  value1  value2
# 0 2023-01-01     1.0     NaN
# 1 2023-01-02     2.0     4.0
# 2 2023-01-03     3.0     5.0
# 3 2023-01-04     NaN     6.0
```

### merge_asof

Nearest-key merge (as-of merge).

```python
df1 = pd.DataFrame({'key': [1, 5, 10], 'value1': ['a', 'b', 'c']})
df2 = pd.DataFrame({'key': [2, 6, 11], 'value2': ['x', 'y', 'z']})

# Merge asof (backward by default)
result = pd.merge_asof(df1, df2, on='key')
print(result)
#    key value1 value2
# 0    1      a    NaN
# 1    5      b      x  # Closest backward match
# 2   10      c      y

# Direction
result = pd.merge_asof(df1, df2, on='key', direction='forward')
result = pd.merge_asof(df1, df2, on='key', direction='nearest')

# Tolerance
result = pd.merge_asof(df1, df2, on='key', tolerance=2)
```

### combine_first

Patch missing data from another DataFrame.

```python
df1 = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
df2 = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})

# Combine first (fill missing in df1 with df2)
result = df1.combine_first(df2)
print(result)
#       A     B
# 0   1.0   4.0
# 1  20.0   5.0
# 2   3.0  60.0
```

---

## Other Transformations

### clip

Clip values to upper and lower bounds.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Clip
clipped = df.clip(lower=2, upper=4)
print(clipped)
#    A
# 0  2  # Clipped from 1
# 1  2
# 2  3
# 3  4
# 4  4  # Clipped from 5

# Per column
df = pd.DataFrame({'A': [1, 5], 'B': [10, 20]})
clipped = df.clip(lower={'A': 2, 'B': 15}, upper={'A': 4, 'B': 18})
```

### cut / qcut

Bin continuous data into discrete intervals.

#### cut

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Cut into bins
binned = pd.cut(df['A'], bins=3)
print(binned)
# 0    (0.991, 4.0]
# 1    (0.991, 4.0]
# 2    (0.991, 4.0]
# 3    (0.991, 4.0]
# 4    (4.0, 7.0]
# 5    (4.0, 7.0]
# 6    (4.0, 7.0]
# 7    (7.0, 10.0]
# 8    (7.0, 10.0]
# 9    (7.0, 10.0]

# Custom bins
binned = pd.cut(df['A'], bins=[0, 3, 6, 10], labels=['low', 'medium', 'high'])

# Include right edge
binned = pd.cut(df['A'], bins=3, right=False)  # Left-closed intervals
```

#### qcut

Quantile-based binning (equal-sized bins).

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Quantile cut
binned = pd.qcut(df['A'], q=3)
print(binned)
# 0    (0.999, 4.0]
# 1    (0.999, 4.0]
# 2    (0.999, 4.0]
# 3    (4.0, 7.0]
# 4    (4.0, 7.0]
# 5    (4.0, 7.0]
# 6    (7.0, 10.0]
# 7    (7.0, 10.0]
# 8    (7.0, 10.0]
# 9    (7.0, 10.0]

# Custom quantiles
binned = pd.qcut(df['A'], q=[0, 0.25, 0.5, 0.75, 1.0], labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### where / mask

Conditional replacement.

#### where

Replace values where condition is False.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Where (keep where True, replace where False)
result = df.where(df['A'] > 3, 0)
print(result)
#    A
# 0  0  # Replaced (condition False)
# 1  0  # Replaced (condition False)
# 2  0  # Replaced (condition False)
# 3  4  # Kept (condition True)
# 4  5  # Kept (condition True)

# With other DataFrame
df2 = pd.DataFrame({'A': [10, 20, 30, 40, 50]})
result = df.where(df['A'] > 3, df2)
```

#### mask

Inverse of where (replace where condition is True).

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Mask (replace where True, keep where False)
result = df.mask(df['A'] > 3, 0)
print(result)
#    A
# 0  1  # Kept (condition False)
# 1  2  # Kept (condition False)
# 2  3  # Kept (condition False)
# 3  0  # Replaced (condition True)
# 4  0  # Replaced (condition True)
```

### assign

Functional column creation (returns new DataFrame).

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Assign single column
df_new = df.assign(C=[7, 8, 9])
print(df_new)
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9

# Assign multiple columns
df_new = df.assign(C=lambda x: x['A'] * 2, D=lambda x: x['B'] + x['A'])

# Chain assignments
df_new = (df
          .assign(C=lambda x: x['A'] * 2)
          .assign(D=lambda x: x['C'] + x['B']))
```

---

## Summary

This comprehensive guide covers data cleaning, transformation, and aggregation in Pandas:

- **Missing Data**: Detection, removal, filling, and replacement strategies
- **String Methods**: Comprehensive string manipulation via `.str` accessor
- **Apply, Map, Transform**: Custom function application for flexible transformations
- **GroupBy**: Split-apply-combine pattern for grouped operations and aggregations
- **Pivot and Reshape**: Restructuring data for analysis (pivot, melt, stack/unstack)
- **Merge, Join, Concat**: Combining data from multiple sources
- **Other Transformations**: Clipping, binning, conditional replacement, and more

Mastering these techniques enables efficient data cleaning, transformation, and analysis workflows in Pandas.
