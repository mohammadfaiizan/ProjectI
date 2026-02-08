# Pandas Indexing, Selection, Filtering, and Manipulation

## Table of Contents
1. [Label-Based Indexing (loc)](#label-based-indexing-loc)
2. [Position-Based Indexing (iloc)](#position-based-indexing-iloc)
3. [Scalar Access (at/iat)](#scalar-access-atiat)
4. [Boolean Indexing / Filtering](#boolean-indexing--filtering)
5. [query() Method](#query-method)
6. [MultiIndex Operations](#multiindex-operations)
7. [Column/Row Manipulation](#columnrow-manipulation)
8. [Sorting](#sorting)
9. [Duplicates](#duplicates)
10. [Iteration](#iteration)

---

## Label-Based Indexing (loc)

The `.loc` accessor provides label-based indexing, meaning you access data by the actual index labels rather than positions. This is the primary method for label-based data selection.

### Single Row

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])

# Single row returns Series
row = df.loc['a']
print(row)
# A    1
# B    4
# Name: a, dtype: int64
```

### Multiple Rows

```python
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, index=['a', 'b', 'c', 'd'])

# Multiple rows returns DataFrame
rows = df.loc[['a', 'c', 'd']]
print(rows)
#    A  B
# a  1  5
# c  3  7
# d  4  8
```

### Row and Column Selection

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, 
                   index=['a', 'b', 'c'])

# Single row, single column (returns scalar)
value = df.loc['a', 'A']
print(value)  # 1

# Single row, multiple columns (returns Series)
row_cols = df.loc['a', ['A', 'C']]
print(row_cols)
# A    1
# C    7

# Multiple rows, single column (returns Series)
rows_col = df.loc[['a', 'c'], 'B']
print(rows_col)
# a    4
# c    6

# Multiple rows, multiple columns (returns DataFrame)
subset = df.loc[['a', 'b'], ['A', 'C']]
print(subset)
#    A  C
# a  1  7
# b  2  8
```

### Slicing with loc

Slicing with `.loc` is **inclusive on both ends**, unlike Python's standard slicing.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}, 
                   index=['a', 'b', 'c', 'd', 'e'])

# Inclusive slice
subset = df.loc['a':'c']
print(subset)
#    A  B
# a  1  6
# b  2  7
# c  3  8

# Slice rows and columns
subset = df.loc['a':'c', 'A']
print(subset)
# a    1
# b    2
# c    3
```

### Boolean Indexing with loc

Combine boolean conditions with label-based selection.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}, 
                   index=['a', 'b', 'c', 'd', 'e'])

# Boolean mask
mask = df['A'] > 2
subset = df.loc[mask, 'B']
print(subset)
# c    30
# d    40
# e    50

# Multiple conditions
mask = (df['A'] > 2) & (df['B'] < 50)
subset = df.loc[mask]
print(subset)
#    A   B
# c  3  30
# d  4  40
```

### Setting Values with loc

`.loc` can be used to set values.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])

# Set single value
df.loc['a', 'A'] = 10
print(df.loc['a', 'A'])  # 10

# Set entire row
df.loc['a'] = [100, 200]

# Set entire column for specific rows
df.loc[['a', 'b'], 'A'] = [1000, 2000]

# Set with condition
df.loc[df['A'] > 2, 'B'] = 999
```

### MultiIndex with loc

Working with hierarchical indices.

```python
tuples = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
idx = pd.MultiIndex.from_tuples(tuples, names=['level1', 'level2'])
df = pd.DataFrame({'data': [1, 2, 3, 4]}, index=idx)

# Single level selection
subset = df.loc['A']
print(subset)
#          data
# level2
# 1           1
# 2           2

# Tuple selection
value = df.loc[('A', 1), 'data']
print(value)  # 1

# Slice on first level
subset = df.loc['A':'B']
```

---

## Position-Based Indexing (iloc)

The `.iloc` accessor provides integer-location based indexing, meaning you access data by position (0-based indexing).

### Single Row

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Single row returns Series
row = df.iloc[0]
print(row)
# A    1
# B    4
# Name: 0, dtype: int64
```

### Multiple Rows

```python
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})

# Multiple rows returns DataFrame
rows = df.iloc[[0, 2, 3]]
print(rows)
#    A  B
# 0  1  5
# 2  3  7
# 3  4  8
```

### Slicing with iloc

Slicing with `.iloc` follows Python's standard slicing: **exclusive end**.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})

# Exclusive end (like Python slicing)
subset = df.iloc[0:3]  # Rows 0, 1, 2
print(subset)
#    A  B
# 0  1  6
# 1  2  7
# 2  3  8

# Negative indexing
last_row = df.iloc[-1]
first_two = df.iloc[:2]
```

### Row and Column Selection

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# Single row, single column (returns scalar)
value = df.iloc[0, 1]
print(value)  # 4

# Single row, multiple columns (returns Series)
row_cols = df.iloc[0, [0, 2]]
print(row_cols)
# A    1
# C    7

# Multiple rows, single column (returns Series)
rows_col = df.iloc[[0, 2], 1]
print(rows_col)
# 0    4
# 2    6

# Multiple rows, multiple columns (returns DataFrame)
subset = df.iloc[[0, 1], [0, 2]]
print(subset)
#    A  C
# 0  1  7
# 1  2  8

# Slicing rows and columns
subset = df.iloc[0:2, 0:2]
print(subset)
#    A  B
# 0  1  4
# 1  2  5
```

### Setting Values with iloc

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Set single value
df.iloc[0, 0] = 10

# Set entire row
df.iloc[0] = [100, 200]

# Set entire column for specific rows
df.iloc[[0, 1], 0] = [1000, 2000]

# Set slice
df.iloc[0:2, 0:2] = [[11, 22], [33, 44]]
```

---

## Scalar Access (at/iat)

For accessing and setting single scalar values, `.at` and `.iat` provide faster alternatives to `.loc` and `.iloc`.

### df.at (Label-Based Scalar Access)

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])

# Get value
value = df.at['a', 'A']
print(value)  # 1

# Set value
df.at['a', 'A'] = 100
print(df.at['a', 'A'])  # 100
```

### df.iat (Position-Based Scalar Access)

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Get value
value = df.iat[0, 1]
print(value)  # 4

# Set value
df.iat[0, 1] = 400
print(df.iat[0, 1])  # 400
```

### When to Use at/iat vs loc/iloc

| Method | Use Case | Performance |
|--------|----------|-------------|
| `.at` | Single scalar value by label | Fastest for single values |
| `.iat` | Single scalar value by position | Fastest for single values |
| `.loc` | Single or multiple values by label | Slower but more flexible |
| `.iloc` | Single or multiple values by position | Slower but more flexible |

**Use `.at`/`.iat` when:**
- Accessing or setting a single scalar value
- Performance is critical (loops, large DataFrames)

**Use `.loc`/`.iloc` when:**
- Selecting multiple rows/columns
- Using boolean indexing
- Slicing operations

---

## Boolean Indexing / Filtering

Boolean indexing allows you to filter DataFrames based on conditions, returning only rows that meet the criteria.

### Single Condition

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# Simple condition
filtered = df[df['A'] > 3]
print(filtered)
#    A   B
# 3  4  40
# 4  5  50

# Alternative syntax
filtered = df.loc[df['A'] > 3]
```

### Multiple Conditions

When using multiple conditions, you **must** use parentheses and bitwise operators (`&`, `|`, `~`) instead of Python's `and`, `or`, `not`.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# AND condition (both must be True)
filtered = df[(df['A'] > 2) & (df['B'] < 50)]
print(filtered)
#    A   B
# 2  3  30
# 3  4  40

# OR condition (either can be True)
filtered = df[(df['A'] < 2) | (df['A'] > 4)]
print(filtered)
#    A   B
# 0  1  10
# 4  5  50

# NOT condition
filtered = df[~(df['A'] > 3)]
print(filtered)
#    A   B
# 0  1  10
# 1  2  20
# 2  3  30

# Complex conditions
filtered = df[(df['A'] > 2) & (df['B'] < 50) | (df['A'] == 1)]
```

### .isin() for Membership Testing

Check if values are in a list.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']})

# Check membership
filtered = df[df['A'].isin([2, 4, 6])]
print(filtered)
#    A  B
# 1  2  b
# 3  4  d

# Not in
filtered = df[~df['A'].isin([2, 4])]
```

### .between() for Range Testing

Check if values fall within a range (inclusive by default).

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7]})

# Inclusive range
filtered = df[df['A'].between(2, 5)]
print(filtered)
#    A
# 1  2
# 2  3
# 3  4
# 4  5

# Exclusive end
filtered = df[df['A'].between(2, 5, inclusive='right')]
```

### String Methods for Filtering

Use `.str` accessor for string-based filtering.

```python
df = pd.DataFrame({'A': ['apple', 'banana', 'cherry', 'date']})

# Contains
filtered = df[df['A'].str.contains('a')]
print(filtered)
#        A
# 0  apple
# 1  banana
# 2   date

# Startswith
filtered = df[df['A'].str.startswith('a')]

# Endswith
filtered = df[df['A'].str.endswith('e')]

# Case-insensitive
filtered = df[df['A'].str.contains('A', case=False)]

# Regex
filtered = df[df['A'].str.contains('^a', regex=True)]
```

### np.where with DataFrames

NumPy's `where` function can be used for conditional selection.

```python
import numpy as np

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Conditional values
df['B'] = np.where(df['A'] > 3, 'high', 'low')
print(df)
#    A     B
# 0  1   low
# 1  2   low
# 2  3   low
# 3  4  high
# 4  5  high

# Multiple conditions
df['C'] = np.where(df['A'] > 4, 'very_high',
          np.where(df['A'] > 2, 'medium', 'low'))
```

---

## query() Method

The `.query()` method allows you to filter DataFrames using string expressions, making complex filtering more readable.

### Basic Syntax

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# Simple query
result = df.query('A > 3')
print(result)
#    A   B
# 3  4  40
# 4  5  50

# Multiple conditions
result = df.query('A > 2 and B < 50')
result = df.query('A > 4 or A < 2')
```

### Variable Reference with @

Reference external variables using `@`.

```python
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

threshold = 3
result = df.query('A > @threshold')

values = [2, 4, 6]
result = df.query('A in @values')
```

### Advantages of query()

1. **Readability**: More readable for complex conditions.
2. **Performance**: Can be faster with `numexpr` backend for large DataFrames.
3. **String-based**: Useful for dynamic query construction.

```python
# Compare readability
# Traditional way
result = df[(df['A'] > 2) & (df['B'] < 50) & (df['A'] != df['B'])]

# Query way
result = df.query('A > 2 and B < 50 and A != B')
```

---

## MultiIndex Operations

MultiIndex (hierarchical indexing) allows you to have multiple index levels, enabling sophisticated data organization.

### set_index and reset_index

#### set_index

Convert columns to index levels.

```python
df = pd.DataFrame({'A': ['a', 'a', 'b', 'b'], 
                   'B': [1, 2, 1, 2], 
                   'C': [10, 20, 30, 40]})

# Single level
df_indexed = df.set_index('A')
print(df_indexed)
#    B   C
# A
# a  1  10
# a  2  20
# b  1  30
# b  2  40

# Multiple levels
df_multi = df.set_index(['A', 'B'])
print(df_multi)
#        C
# A B
# a 1   10
#   2   20
# b 1   30
#   2   40

# Append to existing index
df_indexed = df.set_index('A', append=True)

# Drop column (default is True)
df_indexed = df.set_index('A', drop=False)
```

#### reset_index

Convert index levels back to columns.

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)]))

# Reset all levels
df_reset = df.reset_index()
print(df_reset)
#   level_0  level_1  A
# 0       a        1  1
# 1       a        2  2
# 2       b        1  3

# Reset specific level
df_reset = df.reset_index(level=0)

# Reset with name
df_reset = df.reset_index(names=['level1', 'level2'])

# Don't add to columns (drop)
df_reset = df.reset_index(drop=True)
```

### xs (Cross-Section)

Extract cross-section from MultiIndex.

```python
tuples = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
idx = pd.MultiIndex.from_tuples(tuples, names=['level1', 'level2'])
df = pd.DataFrame({'data': [1, 2, 3, 4]}, index=idx)

# Extract by first level
subset = df.xs('A', level='level1')
print(subset)
#          data
# level2
# 1           1
# 2           2

# Extract by second level
subset = df.xs(1, level='level2')

# Extract by multiple levels (using tuple)
subset = df.xs(('A', 1), level=['level1', 'level2'])

# Drop level in result
subset = df.xs('A', level='level1', drop_level=False)
```

### IndexSlice

`pd.IndexSlice` provides a convenient way to slice MultiIndex DataFrames.

```python
tuples = [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
idx = pd.MultiIndex.from_tuples(tuples, names=['level1', 'level2'])
df = pd.DataFrame({'X': [1, 2, 3, 4], 'Y': [5, 6, 7, 8]}, index=idx)

idx_slice = pd.IndexSlice

# Slice first level
subset = df.loc[idx_slice['A':], :]

# Slice both levels
subset = df.loc[idx_slice['A', 1:2], :]

# Complex slicing
subset = df.loc[idx_slice[:, [1, 2]], :]
```

### swaplevel and reorder_levels

#### swaplevel

Swap two index levels.

```python
df = pd.DataFrame({'A': [1, 2]}, 
                  index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2)], 
                                                  names=['L1', 'L2']))

# Swap levels
df_swapped = df.swaplevel('L1', 'L2')
df_swapped = df.swaplevel(0, 1)  # By position
```

#### reorder_levels

Reorder index levels.

```python
df = pd.DataFrame({'A': [1, 2]}, 
                  index=pd.MultiIndex.from_tuples([('a', 1, 'x'), ('b', 2, 'y')], 
                                                  names=['L1', 'L2', 'L3']))

# Reorder
df_reordered = df.reorder_levels(['L3', 'L1', 'L2'])
df_reordered = df.reorder_levels([2, 0, 1])  # By position
```

### sort_index

Sort by index values.

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['c', 'a', 'b'])

# Sort index
df_sorted = df.sort_index()
print(df_sorted)
#    A
# a  2
# b  3
# c  1

# MultiIndex sorting
df = pd.DataFrame({'A': [1, 2, 3, 4]}, 
                  index=pd.MultiIndex.from_tuples([('b', 2), ('a', 1), ('b', 1), ('a', 2)]))

# Sort by level
df_sorted = df.sort_index(level=0)
df_sorted = df.sort_index(level='level1')

# Sort remaining levels
df_sorted = df.sort_index(level=0, sort_remaining=True)
```

---

## Column/Row Manipulation

### Adding Columns

#### Direct Assignment

```python
df = pd.DataFrame({'A': [1, 2, 3]})

# Single column
df['B'] = [4, 5, 6]

# Multiple columns
df[['C', 'D']] = [[7, 8, 9], [10, 11, 12]]

# From calculation
df['E'] = df['A'] + df['B']
```

#### assign()

Returns new DataFrame without modifying original.

```python
df = pd.DataFrame({'A': [1, 2, 3]})

# Single column
df_new = df.assign(B=[4, 5, 6])

# Multiple columns
df_new = df.assign(B=[4, 5, 6], C=lambda x: x['A'] * 2)

# Chain assignments
df_new = df.assign(B=[4, 5, 6]).assign(C=lambda x: x['A'] + x['B'])
```

#### insert()

Insert column at specific position.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})

# Insert at position 1
df.insert(1, 'B', [4, 5, 6])
print(df)
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9
```

### Dropping

#### drop() for Columns

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# Drop single column (returns new DataFrame)
df_dropped = df.drop(columns='B')

# Drop multiple columns
df_dropped = df.drop(columns=['B', 'C'])

# Inplace modification
df.drop(columns='B', inplace=True)
```

#### drop() for Rows

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])

# Drop by label
df_dropped = df.drop(index='a')

# Drop multiple rows
df_dropped = df.drop(index=['a', 'c'])

# Drop by position (use iloc for this)
df_dropped = df.drop(df.index[0])
```

#### pop()

Remove column and return it as Series.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Pop column
col_b = df.pop('B')
print(col_b)
# 0    4
# 1    5
# 2    6

# df now only has 'A'
print(df)
#    A
# 0  1
# 1  2
# 2  3
```

### Renaming

#### rename() for Columns

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Rename single column
df_renamed = df.rename(columns={'A': 'Alpha'})

# Rename multiple columns
df_renamed = df.rename(columns={'A': 'Alpha', 'B': 'Beta'})

# Using function
df_renamed = df.rename(columns=str.lower)

# Inplace
df.rename(columns={'A': 'Alpha'}, inplace=True)
```

#### rename() for Index

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])

# Rename index
df_renamed = df.rename(index={'a': 'alpha'})

# Using function
df_renamed = df.rename(index=str.upper)
```

#### set_axis()

Set axis labels.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Set column names
df_new = df.set_axis(['Alpha', 'Beta'], axis=1)

# Set index
df_new = df.set_axis(['x', 'y', 'z'], axis=0)
```

### Reindexing

#### reindex()

Conform DataFrame to new index.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])

# Reindex with new labels
df_new = df.reindex(['a', 'b', 'c', 'd'])
print(df_new)
#      A    B
# a  1.0  4.0
# b  2.0  5.0
# c  3.0  6.0
# d  NaN  NaN

# Fill missing values
df_new = df.reindex(['a', 'b', 'c', 'd'], fill_value=0)

# Reindex columns
df_new = df.reindex(columns=['B', 'A', 'C'])
```

#### reindex_like()

Reindex to match another DataFrame's index/columns.

```python
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'B': [10, 20]}, index=['a', 'd'])

# Reindex df1 to match df2's index
df1_new = df1.reindex_like(df2)
```

---

## Sorting

### sort_values

Sort by column values.

```python
df = pd.DataFrame({'A': [3, 1, 2], 'B': [6, 4, 5], 'C': [9, 7, 8]})

# Sort by single column
df_sorted = df.sort_values('A')
print(df_sorted)
#    A  B  C
# 1  1  4  7
# 2  2  5  8
# 0  3  6  9

# Sort by multiple columns
df_sorted = df.sort_values(['A', 'B'])

# Ascending order
df_sorted = df.sort_values('A', ascending=True)  # Default
df_sorted = df.sort_values('A', ascending=False)  # Descending

# Different order per column
df_sorted = df.sort_values(['A', 'B'], ascending=[True, False])

# Handle NaN position
df_sorted = df.sort_values('A', na_position='first')  # or 'last'

# Custom key function
df_sorted = df.sort_values('A', key=lambda x: x % 2)  # Sort by even/odd

# Sort algorithm
df_sorted = df.sort_values('A', kind='quicksort')  # 'mergesort', 'heapsort', 'stable'
```

### sort_index

Sort by index.

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['c', 'a', 'b'])

# Sort index
df_sorted = df.sort_index()
print(df_sorted)
#    A
# a  2
# b  3
# c  1

# MultiIndex sorting
df = pd.DataFrame({'A': [1, 2, 3, 4]}, 
                  index=pd.MultiIndex.from_tuples([('b', 2), ('a', 1), ('b', 1), ('a', 2)]))

# Sort by level
df_sorted = df.sort_index(level=0)
df_sorted = df.sort_index(level='level1')

# Sort remaining levels
df_sorted = df.sort_index(level=0, sort_remaining=True)
```

### nsmallest and nlargest

Efficient partial sorting for top/bottom n values.

```python
df = pd.DataFrame({'A': [5, 2, 8, 1, 9, 3]})

# Top 3 largest
top3 = df.nlargest(3, 'A')
print(top3)
#    A
# 4  9
# 2  8
# 0  5

# Bottom 3 smallest
bottom3 = df.nsmallest(3, 'A')
print(bottom3)
#    A
# 3  1
# 1  2
# 5  3

# Multiple columns
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
top2 = df.nlargest(2, 'A')
```

### rank

Assign ranks to values.

```python
df = pd.DataFrame({'A': [3, 1, 2, 2, 4]})

# Default ranking (average method)
df['rank'] = df['A'].rank()
print(df)
#    A  rank
# 0  3   4.0
# 1  1   1.0
# 2  2   2.5
# 3  2   2.5
# 4  4   5.0

# Ranking methods
df['min'] = df['A'].rank(method='min')  # Minimum rank for ties
df['max'] = df['A'].rank(method='max')  # Maximum rank for ties
df['first'] = df['A'].rank(method='first')  # First occurrence gets lower rank
df['dense'] = df['A'].rank(method='dense')  # Dense ranking (no gaps)

# Ascending/descending
df['rank_desc'] = df['A'].rank(ascending=False)

# Handle NaN
df['rank'] = df['A'].rank(na_option='keep')  # 'top', 'bottom'

# Percentage rank
df['pct_rank'] = df['A'].rank(pct=True)
```

---

## Duplicates

### duplicated()

Identify duplicate rows.

```python
df = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3], 'B': [4, 5, 5, 6, 6, 7]})

# Find duplicates
duplicates = df.duplicated()
print(duplicates)
# 0    False
# 1    False
# 2     True
# 3    False
# 4     True
# 5    False

# Check specific columns
duplicates = df.duplicated(subset=['A'])

# Keep parameter
duplicates_first = df.duplicated(keep='first')  # Default: mark all but first as duplicate
duplicates_last = df.duplicated(keep='last')  # Mark all but last as duplicate
duplicates_none = df.duplicated(keep=False)  # Mark all duplicates as True
```

### drop_duplicates()

Remove duplicate rows.

```python
df = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3], 'B': [4, 5, 5, 6, 6, 7]})

# Drop duplicates (keeps first by default)
df_unique = df.drop_duplicates()
print(df_unique)
#    A  B
# 0  1  4
# 1  2  5
# 3  3  6
# 5  3  7

# Keep last
df_unique = df.drop_duplicates(keep='last')

# Keep none (remove all duplicates)
df_unique = df.drop_duplicates(keep=False)

# Subset
df_unique = df.drop_duplicates(subset=['A'])

# Inplace
df.drop_duplicates(inplace=True)
```

### Index.duplicated()

Check for duplicate index values.

```python
df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'a', 'b'])

# Check index duplicates
print(df.index.duplicated())
# [False  True False]

# Drop duplicate index
df_unique = df[~df.index.duplicated()]
```

---

## Iteration

While iteration is generally discouraged in favor of vectorized operations, there are cases where it's necessary or acceptable.

### iterrows()

Iterate over DataFrame rows as (index, Series) pairs.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Iterate rows
for idx, row in df.iterrows():
    print(f"Index: {idx}, A: {row['A']}, B: {row['B']}")

# Warning: iterrows() is slow and returns copies, not views
```

### itertuples()

Iterate over DataFrame rows as namedtuples (faster than iterrows).

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Iterate as namedtuples
for row in df.itertuples():
    print(row.Index, row.A, row.B)

# Without index
for row in df.itertuples(index=False):
    print(row.A, row.B)

# Custom name
for row in df.itertuples(name='Point'):
    print(row)
```

### items()

Iterate over (column name, Series) pairs.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Iterate columns
for col_name, col_data in df.items():
    print(f"Column: {col_name}")
    print(col_data)
```

### Why Vectorized Operations are Preferred

**Performance Comparison:**

```python
import time

df = pd.DataFrame({'A': range(10000), 'B': range(10000, 20000)})

# Iteration (slow)
start = time.time()
result = []
for idx, row in df.iterrows():
    result.append(row['A'] + row['B'])
time_iter = time.time() - start

# Vectorized (fast)
start = time.time()
result_vec = df['A'] + df['B']
time_vec = time.time() - start

print(f"Iteration: {time_iter:.4f}s")
print(f"Vectorized: {time_vec:.4f}s")
# Vectorized is typically 100-1000x faster
```

**When Iteration is Acceptable:**

1. **Complex logic**: When operations can't be easily vectorized.
2. **Small DataFrames**: Performance difference is negligible.
3. **External API calls**: When each row requires separate API call.
4. **Custom transformations**: When logic is too complex for vectorization.

**Best Practices:**

- Always prefer vectorized operations when possible.
- Use `.apply()` for row/column-wise operations before falling back to iteration.
- Use `.itertuples()` instead of `.iterrows()` if iteration is necessary.
- Consider using NumPy vectorization or Cython for performance-critical code.

---

## Summary

This comprehensive guide covers all aspects of indexing, selection, filtering, and manipulation in Pandas:

- **Label-based indexing (loc)**: Access data by labels with inclusive slicing
- **Position-based indexing (iloc)**: Access data by position with exclusive slicing
- **Scalar access (at/iat)**: Fast single-value access
- **Boolean indexing**: Filter data based on conditions
- **query() method**: Readable string-based filtering
- **MultiIndex operations**: Work with hierarchical indices
- **Column/row manipulation**: Add, drop, rename, reindex
- **Sorting**: Sort by values or index with various options
- **Duplicates**: Identify and remove duplicate data
- **Iteration**: When and how to iterate (and why to avoid it)

Mastering these techniques is essential for efficient data manipulation and analysis.
