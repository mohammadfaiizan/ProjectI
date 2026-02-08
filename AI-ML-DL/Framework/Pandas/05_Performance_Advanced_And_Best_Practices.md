# Performance, Advanced Features, and Best Practices in Pandas

## Table of Contents
1. [Memory Optimization](#memory-optimization)
2. [Performance Optimization](#performance-optimization)
3. [Method Chaining](#method-chaining)
4. [Window Functions](#window-functions)
5. [Advanced Features](#advanced-features)
6. [Interoperability](#interoperability)
7. [Styling](#styling)
8. [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)

---

## Memory Optimization

### Understanding Memory Usage

#### memory_usage()

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 1000000),
    'float_col': np.random.randn(1000000),
    'str_col': ['string'] * 1000000,
    'category_col': ['A', 'B', 'C'] * 333334
})

# Basic memory usage
df.memory_usage()                    # Per column
df.memory_usage(deep=True)           # Deep inspection (includes object references)
df.memory_usage(deep=True).sum()     # Total memory

# Index memory
df.memory_usage(index=True)         # Include index memory

# Memory usage info
df.info(memory_usage='deep')        # Detailed memory info
```

#### Actual vs Reported Memory

```python
# Reported memory (may not include object overhead)
reported = df.memory_usage(deep=False).sum()

# Actual memory (includes object overhead)
actual = df.memory_usage(deep=True).sum()

print(f"Reported: {reported / 1024**2:.2f} MB")
print(f"Actual: {actual / 1024**2:.2f} MB")
```

### Downcasting Numeric Types

#### pd.to_numeric with downcast

```python
# Original DataFrame
df = pd.DataFrame({
    'int64_col': np.array([1, 2, 3, 4, 5], dtype='int64'),
    'float64_col': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float64')
})

# Downcast integers
df['int64_col'] = pd.to_numeric(df['int64_col'], downcast='integer')
# Result: int8 (if values fit)

# Downcast floats
df['float64_col'] = pd.to_numeric(df['float64_col'], downcast='float')
# Result: float32 (if precision allows)

# Automatic downcasting function
def downcast_numeric(df):
    for col in df.select_dtypes(include=['int64', 'int32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64', 'float32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df = downcast_numeric(df)
```

#### Memory Comparison Table

| Original Type | Downcast Type | Memory Reduction | Use Case |
|--------------|---------------|------------------|----------|
| int64 | int8 | 87.5% | Values 0-255 |
| int64 | int16 | 75% | Values -32,768 to 32,767 |
| int64 | int32 | 50% | Values -2B to 2B |
| float64 | float32 | 50% | When precision allows |

### Category Dtype

#### When to Use Categories

Categories are ideal for:
- Repeated string values
- Ordinal data (e.g., sizes: S, M, L, XL)
- Limited unique values relative to total rows

```python
# Create data with repeated strings
df = pd.DataFrame({
    'size': ['S', 'M', 'L', 'XL'] * 250000,
    'color': ['red', 'blue', 'green'] * 333334
})

# Memory before
before = df.memory_usage(deep=True).sum()

# Convert to category
df['size'] = df['size'].astype('category')
df['color'] = df['color'].astype('category')

# Memory after
after = df.memory_usage(deep=True).sum()
reduction = (1 - after / before) * 100
print(f"Memory reduction: {reduction:.1f}%")
```

#### Category Accessor (.cat)

```python
# Create categorical Series
s = pd.Series(['A', 'B', 'C', 'A', 'B'], dtype='category')

# Categories
s.cat.categories                    # Index(['A', 'B', 'C'])
s.cat.codes                         # [0, 1, 2, 0, 1]

# Add categories
s = s.cat.add_categories(['D', 'E'])

# Remove categories
s = s.cat.remove_categories(['D'])

# Remove unused categories
s = s.cat.remove_unused_categories()

# Rename categories
s = s.cat.rename_categories({'A': 'Alpha', 'B': 'Beta'})

# Set categories (can include new ones)
s = s.cat.set_categories(['A', 'B', 'C', 'D', 'E'])

# Reorder categories
s = s.cat.reorder_categories(['C', 'B', 'A'], ordered=True)

# Ordered vs unordered
s = s.cat.as_ordered()              # Set as ordered
s = s.cat.as_unordered()            # Set as unordered

# Check if ordered
s.cat.ordered                        # True/False
```

#### Memory Savings Calculation

```python
def calculate_category_savings(series):
    """Calculate memory savings from converting to category"""
    original_memory = series.memory_usage(deep=True)
    
    categorical = series.astype('category')
    category_memory = categorical.memory_usage(deep=True)
    
    savings = (1 - category_memory / original_memory) * 100
    
    print(f"Original: {original_memory} bytes")
    print(f"Category: {category_memory} bytes")
    print(f"Savings: {savings:.1f}%")
    print(f"Unique values: {series.nunique()}")
    print(f"Total values: {len(series)}")
    
    return savings

# Example
s = pd.Series(['A', 'B', 'C'] * 100000)
calculate_category_savings(s)
```

### Object to String Dtype Conversion

Pandas 1.0+ introduced dedicated string dtype:

```python
# Object dtype (Python objects)
df = pd.DataFrame({'text': ['string'] * 1000000})
print(df['text'].dtype)  # object

# Convert to string dtype
df['text'] = df['text'].astype('string')
print(df['text'].dtype)  # string

# Memory comparison
obj_memory = pd.Series(['text'] * 1000000, dtype='object').memory_usage(deep=True)
str_memory = pd.Series(['text'] * 1000000, dtype='string').memory_usage(deep=True)
print(f"Object: {obj_memory / 1024**2:.2f} MB")
print(f"String: {str_memory / 1024**2:.2f} MB")
```

### Sparse Arrays

Sparse arrays store only non-zero/non-null values:

```python
# Create sparse data (mostly zeros)
arr = np.zeros(1000000)
arr[::100] = 1  # Every 100th element is 1

# Dense array
dense = pd.Series(arr)
print(f"Dense memory: {dense.memory_usage(deep=True) / 1024**2:.2f} MB")

# Sparse array
sparse = pd.arrays.SparseArray(arr)
sparse_series = pd.Series(sparse)
print(f"Sparse memory: {sparse_series.memory_usage(deep=True) / 1024**2:.2f} MB")

# Sparse dtype
df = pd.DataFrame({'sparse_col': pd.arrays.SparseArray(arr)})
print(df['sparse_col'].dtype)  # Sparse[float64, 0]

# Fill value
sparse = pd.arrays.SparseArray([0, 0, 1, 0, 2], fill_value=0)
print(sparse.density)  # Proportion of non-fill values
```

---

## Performance Optimization

### Vectorized Operations vs apply vs iterrows

#### Timing Comparison

```python
import time

df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000)
})

# Vectorized operation (FASTEST)
start = time.time()
result1 = df['A'] + df['B']
vectorized_time = time.time() - start

# Apply (SLOW)
start = time.time()
result2 = df.apply(lambda row: row['A'] + row['B'], axis=1)
apply_time = time.time() - start

# Iterrows (VERY SLOW)
start = time.time()
result3 = pd.Series([row['A'] + row['B'] for _, row in df.iterrows()])
iterrows_time = time.time() - start

print(f"Vectorized: {vectorized_time:.4f}s")
print(f"Apply: {apply_time:.4f}s")
print(f"Iterrows: {iterrows_time:.4f}s")
print(f"Apply is {apply_time/vectorized_time:.1f}x slower")
print(f"Iterrows is {iterrows_time/vectorized_time:.1f}x slower")
```

#### When to Use Each

- **Vectorized**: Always prefer for simple operations
- **apply**: Use for complex row-wise operations without vectorized alternative
- **iterrows**: Avoid unless absolutely necessary (use itertuples if needed)

### eval() and query()

These use numexpr backend for faster evaluation:

```python
# eval() for complex expressions
df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000),
    'C': np.random.randn(100000)
})

# Standard evaluation
result1 = df['A'] + df['B'] * df['C']

# eval() (faster for complex expressions)
result2 = df.eval('A + B * C')

# query() for filtering
filtered1 = df[(df['A'] > 0) & (df['B'] < 0)]
filtered2 = df.query('A > 0 and B < 0')

# In-place operations
df.eval('D = A + B', inplace=True)
```

### .values vs .to_numpy()

```python
# .values (returns numpy array, may be a view)
arr1 = df['A'].values

# .to_numpy() (explicit conversion, preferred)
arr2 = df['A'].to_numpy()

# Copy vs view
arr_copy = df['A'].to_numpy(copy=True)
arr_view = df['A'].to_numpy(copy=False)

# For DataFrame
arr_df = df[['A', 'B']].to_numpy()
```

### Avoiding Chained Indexing

#### SettingWithCopyWarning

```python
# PROBLEMATIC: Chained indexing
df[df['A'] > 0]['B'] = 1  # May raise SettingWithCopyWarning

# CORRECT: Use .loc
df.loc[df['A'] > 0, 'B'] = 1

# CORRECT: Explicit copy
df_copy = df[df['A'] > 0].copy()
df_copy['B'] = 1
```

#### Copy-on-Write in Pandas 2.0+

```python
# Pandas 2.0+ introduces Copy-on-Write
# Views are created by default, copies only when modified

# Enable Copy-on-Write mode
pd.set_option('mode.copy_on_write', True)

# This prevents accidental modifications
df_view = df[['A', 'B']]
df_view['C'] = 1  # This creates a copy automatically
```

### Using Appropriate Dtypes from Start

```python
# BAD: Reading with default dtypes
df = pd.read_csv('large_file.csv')  # May use object for everything

# GOOD: Specify dtypes upfront
dtypes = {
    'id': 'int32',
    'category': 'category',
    'value': 'float32',
    'date': 'str'  # Parse dates separately
}
df = pd.read_csv('large_file.csv', dtype=dtypes)
df['date'] = pd.to_datetime(df['date'])
```

### Batch Operations with Chunksize

```python
# Process large file in chunks
chunk_size = 10000
results = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed = chunk.groupby('category').sum()
    results.append(processed)

# Combine results
final_result = pd.concat(results).groupby('category').sum()
```

### Using inplace Parameter

```python
# inplace=True doesn't always save memory
# It modifies in place but may create intermediate copies

# Standard approach (creates new DataFrame)
df = df.dropna()

# In-place (modifies existing DataFrame)
df.dropna(inplace=True)

# Note: In modern Pandas, inplace is being deprecated
# Prefer: df = df.dropna()
```

### Avoiding Growing DataFrames

```python
# BAD: Growing DataFrame with append
df = pd.DataFrame()
for i in range(1000):
    df = df.append({'A': i, 'B': i*2}, ignore_index=True)  # SLOW

# GOOD: Collect in list, then concat
rows = []
for i in range(1000):
    rows.append({'A': i, 'B': i*2})
df = pd.DataFrame(rows)  # FAST

# EVEN BETTER: Pre-allocate if size is known
df = pd.DataFrame(index=range(1000), columns=['A', 'B'])
for i in range(1000):
    df.loc[i] = [i, i*2]
```

---

## Method Chaining

### Basic Chaining Pattern

```python
# Chain multiple operations
result = (df
    .query('A > 0')
    .groupby('category')
    .agg({'B': 'sum', 'C': 'mean'})
    .reset_index()
    .sort_values('B', ascending=False)
)
```

### .pipe() for Custom Functions

```python
# Pass DataFrame through custom function
def add_total_column(df):
    df['total'] = df.sum(axis=1)
    return df

def filter_high_values(df, threshold=100):
    return df[df['total'] > threshold]

result = (df
    .pipe(add_total_column)
    .pipe(filter_high_values, threshold=100)
    .groupby('category')
    .sum()
)
```

### .assign() for New Columns

```python
# Create new columns in chain
result = (df
    .assign(
        total=lambda x: x['A'] + x['B'],
        ratio=lambda x: x['A'] / x['B'],
        category=lambda x: pd.cut(x['total'], bins=3, labels=['Low', 'Med', 'High'])
    )
    .query('ratio > 1')
    .groupby('category')
    .mean()
)
```

### .query() for Filtering

```python
# Filter in chain
result = (df
    .query('A > 0 and B < 100')
    .query('C.isin(["X", "Y", "Z"])')
    .groupby('category')
    .sum()
)
```

### Complete ETL Pipeline Example

```python
# Complete ETL pipeline as single chain
processed_data = (
    pd.read_csv('raw_data.csv')
    .pipe(lambda x: x.rename(columns=str.lower))
    .assign(
        date=lambda x: pd.to_datetime(x['date']),
        year=lambda x: x['date'].dt.year,
        month=lambda x: x['date'].dt.month
    )
    .query('value > 0')
    .assign(
        category=lambda x: pd.cut(
            x['value'],
            bins=[0, 50, 100, np.inf],
            labels=['Low', 'Medium', 'High']
        )
    )
    .groupby(['year', 'month', 'category'])
    .agg({
        'value': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    })
    .reset_index()
    .sort_values(['year', 'month', 'value'], ascending=[True, True, False])
    .pipe(lambda x: x[x['value']['sum'] > 1000])
    .to_csv('processed_data.csv', index=False)
)
```

---

## Window Functions

### Rolling with Custom Functions

```python
# Custom rolling function
def rolling_iqr(x):
    """Calculate interquartile range"""
    return x.quantile(0.75) - x.quantile(0.25)

df['rolling_iqr'] = df['value'].rolling(window=30).apply(rolling_iqr)

# Using raw=True for NumPy speed (passes array instead of Series)
def custom_func(arr):
    return np.max(arr) - np.min(arr)

df['rolling_range'] = df['value'].rolling(window=30).apply(custom_func, raw=True)
```

### Expanding Windows

```python
# Expanding mean
df['expanding_mean'] = df['value'].expanding().mean()

# Expanding with min_periods
df['expanding_mean'] = df['value'].expanding(min_periods=10).mean()

# Expanding quantiles
df['expanding_q25'] = df['value'].expanding().quantile(0.25)
df['expanding_q75'] = df['value'].expanding().quantile(0.75)
```

### EWM (Exponential Weighted Moving)

```python
# EWM mean with different parameters
df['ewm_span7'] = df['value'].ewm(span=7).mean()
df['ewm_alpha03'] = df['value'].ewm(alpha=0.3).mean()
df['ewm_halflife'] = df['value'].ewm(halflife='7 days').mean()

# EWM with adjust parameter
df['ewm_adjusted'] = df['value'].ewm(span=7, adjust=True).mean()
df['ewm_unadjusted'] = df['value'].ewm(span=7, adjust=False).mean()
```

### Custom Window: BaseIndexer

```python
from pandas.api.indexers import BaseIndexer

class VariableWidthWindow(BaseIndexer):
    """Custom window that varies width based on condition"""
    def get_window_bounds(self, num_values, min_periods, center, closed):
        start = np.zeros(num_values, dtype=np.int64)
        end = np.zeros(num_values, dtype=np.int64)
        
        for i in range(num_values):
            # Custom logic: window size based on index
            window_size = min(10, i + 1)
            start[i] = max(0, i - window_size + 1)
            end[i] = i + 1
        
        return start, end

# Use custom window
indexer = VariableWidthWindow()
df['custom_rolling'] = df['value'].rolling(window=indexer).mean()
```

### Grouped Rolling

```python
# Rolling within groups
df['grouped_rolling'] = df.groupby('category')['value'].rolling(window=5).mean().reset_index(0, drop=True)

# Multiple rolling windows per group
result = df.groupby('category').apply(
    lambda x: x['value'].rolling(window=5).agg(['mean', 'std'])
)
```

---

## Advanced Features

### MultiIndex Operations

#### swaplevel

```python
# Create MultiIndex DataFrame
df = pd.DataFrame({
    'value': np.random.randn(12)
}, index=pd.MultiIndex.from_product([
    ['A', 'B', 'C'],
    ['X', 'Y', 'Z', 'W']
], names=['level1', 'level2']))

# Swap levels
df_swapped = df.swaplevel('level1', 'level2')
df_swapped = df.swaplevel(0, 1)  # By position
```

#### droplevel

```python
# Drop level
df_dropped = df.droplevel('level1')
df_dropped = df.droplevel(0)  # By position
df_dropped = df.droplevel([0, 1])  # Multiple levels
```

#### sortlevel

```python
# Sort by level
df_sorted = df.sortlevel('level1')
df_sorted = df.sortlevel(0)  # By position
df_sorted = df.sortlevel(['level1', 'level2'])  # Multiple levels
df_sorted = df.sortlevel(0, ascending=False)  # Descending
```

### pd.option_context and pd.set_option

#### Display Options

```python
# Set display options permanently
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Use context manager for temporary settings
with pd.option_context('display.max_rows', 10, 'display.max_columns', 5):
    print(df)  # Uses temporary settings

# Reset to default
pd.reset_option('display.max_rows')
pd.reset_option('all')  # Reset all options
```

#### Common Options

```python
# Display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_colwidth', 50)

# Mode options
pd.set_option('mode.chained_assignment', 'warn')  # 'raise', 'warn', None
pd.set_option('mode.copy_on_write', True)  # Pandas 2.0+

# IO options
pd.set_option('io.excel.xlsx.writer', 'openpyxl')
pd.set_option('io.hdf.default_format', 'table')
```

### Testing Functions

```python
# Assert DataFrame equality
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

pd.testing.assert_frame_equal(df1, df2)

# With tolerance for floats
df1 = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
df2 = pd.DataFrame({'A': [1.001, 2.002, 3.003]})
pd.testing.assert_frame_equal(df1, df2, atol=0.01)

# Assert Series equality
s1 = pd.Series([1, 2, 3])
s2 = pd.Series([1, 2, 3])
pd.testing.assert_series_equal(s1, s2)

# Check index equality
pd.testing.assert_index_equal(df1.index, df2.index)
```

### Extension Types

Creating custom ExtensionDtype and ExtensionArray:

```python
from pandas.api.extensions import ExtensionDtype, ExtensionArray

class CustomDtype(ExtensionDtype):
    name = "custom"
    
    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a '{cls}' from '{string}'")
    
    @property
    def type(self):
        return object
    
    @property
    def na_value(self):
        return pd.NA

class CustomArray(ExtensionArray):
    def __init__(self, data):
        self._data = np.asarray(data)
    
    @property
    def dtype(self):
        return CustomDtype()
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, item):
        return self._data[item]
    
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars)

# Register extension type
pd.api.extensions.register_extension_dtype(CustomDtype)

# Use custom type
arr = CustomArray([1, 2, 3, 4, 5])
s = pd.Series(arr)
```

### Copy-on-Write (CoW) in Pandas 2.0+

```python
# Enable Copy-on-Write
pd.set_option('mode.copy_on_write', True)

# Views are created by default
df_view = df[['A', 'B']]  # View, not copy

# Modifications create copies automatically
df_view['C'] = 1  # Creates copy, doesn't modify original df

# Explicit copy still works
df_copy = df.copy()
```

### Nullable Dtypes Overview

```python
# Nullable integer
s = pd.Series([1, 2, None, 4], dtype='Int64')  # Note capital I
print(s.dtype)  # Int64

# Nullable string
s = pd.Series(['a', 'b', None, 'd'], dtype='string')
print(s.dtype)  # string

# Nullable boolean
s = pd.Series([True, False, None, True], dtype='boolean')
print(s.dtype)  # boolean

# Explicit nullable dtypes
s_int = pd.Series([1, 2, None], dtype=pd.Int64Dtype())
s_str = pd.Series(['a', 'b', None], dtype=pd.StringDtype())
s_bool = pd.Series([True, False, None], dtype=pd.BooleanDtype())
```

---

## Interoperability

### Pandas <-> NumPy

```python
# DataFrame/Series to NumPy
arr = df.values           # Returns numpy array (may be view)
arr = df.to_numpy()       # Explicit conversion (preferred)
arr = df[['A', 'B']].to_numpy()

# NumPy to DataFrame
arr = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(arr, columns=['A', 'B'])

# Series to NumPy
arr = s.values
arr = s.to_numpy()
```

### Pandas <-> Dict/List

```python
# DataFrame to dict
df.to_dict()                    # {'col1': {0: val1, 1: val2}, ...}
df.to_dict('records')           # [{col1: val1, col2: val2}, ...]
df.to_dict('index')             # {0: {col1: val1, col2: val2}, ...}
df.to_dict('list')              # {'col1': [val1, val2], ...}
df.to_dict('series')            # {'col1': Series(...), ...}
df.to_dict('split')             # {'index': [...], 'columns': [...], 'data': [[...]]}

# Dict to DataFrame
d = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(d)

# Records (list of dicts)
records = [{'A': 1, 'B': 4}, {'A': 2, 'B': 5}]
df = pd.DataFrame.from_records(records)

# DataFrame to records
records = df.to_records(index=False)
```

### Pandas <-> SQL

```python
import sqlalchemy

# Create connection
engine = sqlalchemy.create_engine('sqlite:///database.db')

# Read from SQL
df = pd.read_sql('SELECT * FROM table', engine)
df = pd.read_sql_table('table_name', engine)
df = pd.read_sql_query('SELECT * FROM table WHERE id > 100', engine)

# Write to SQL
df.to_sql('table_name', engine, if_exists='replace', index=False)

# SQLAlchemy integration
from sqlalchemy import create_engine, MetaData, Table
engine = create_engine('postgresql://user:pass@localhost/db')
df.to_sql('table', engine, if_exists='append')
```

### Pandas <-> PyTorch/TensorFlow

```python
# Pandas to PyTorch
import torch

tensor = torch.from_numpy(df[['A', 'B']].to_numpy())
tensor = torch.tensor(df[['A', 'B']].values)

# PyTorch to Pandas
df = pd.DataFrame(tensor.numpy(), columns=['A', 'B'])

# Pandas to TensorFlow
import tensorflow as tf

tensor = tf.constant(df[['A', 'B']].to_numpy())
dataset = tf.data.Dataset.from_tensor_slices(df[['A', 'B']].to_dict('list'))
```

### Pandas <-> Scikit-learn

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Prepare data for sklearn
X = df[['feature1', 'feature2']].to_numpy()
y = df['target'].to_numpy()

# Column transformer with feature names
ct = ColumnTransformer([
    ('scaler', StandardScaler(), ['feature1', 'feature2'])
])
X_transformed = ct.fit_transform(df[['feature1', 'feature2']])

# Get feature names
feature_names = ct.get_feature_names_out()
df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
```

### Pandas <-> JSON/API

```python
# Normalized JSON (nested structures)
json_data = {
    'users': [
        {'name': 'Alice', 'scores': [10, 20, 30]},
        {'name': 'Bob', 'scores': [15, 25, 35]}
    ]
}

# Normalize nested JSON
df = pd.json_normalize(json_data, 'users', ['name'])

# API response
import requests
response = requests.get('https://api.example.com/data')
df = pd.json_normalize(response.json())
```

---

## Styling

### Basic Styling

```python
# Access Styler object
styled = df.style

# Highlight maximum values
df.style.highlight_max()

# Highlight minimum values
df.style.highlight_min()

# Background gradient
df.style.background_gradient(cmap='viridis')

# Highlight null values
df.style.highlight_null(null_color='red')
```

### .bar() for Inline Bar Charts

```python
# Inline bar charts
df.style.bar(subset=['A', 'B'], color='lightblue')

# Custom alignment and color
df.style.bar(
    subset=['value'],
    align='mid',
    color=['#d65f5f', '#5fba7d']  # Negative, positive colors
)
```

### .format() for Number Formatting

```python
# Format numbers
df.style.format({'A': '{:.2f}', 'B': '{:.0%}'})

# Format with precision
df.style.format(precision=2)

# Format specific columns
df.style.format({
    'price': '${:.2f}',
    'quantity': '{:.0f}',
    'percentage': '{:.1%}'
})
```

### Custom CSS with .applymap / .apply

```python
# Apply function to each cell
def highlight_positive(val):
    if val > 0:
        return 'background-color: green'
    return 'background-color: red'

df.style.applymap(highlight_positive)

# Apply function to rows/columns
def highlight_max_row(row):
    max_val = row.max()
    return ['background-color: yellow' if val == max_val else '' for val in row]

df.style.apply(highlight_max_row, axis=1)

# Complex styling
def style_dataframe(df):
    return (df.style
        .format({'value': '{:.2f}'})
        .background_gradient(subset=['value'], cmap='YlOrRd')
        .bar(subset=['quantity'], color='lightblue')
        .highlight_max(subset=['value'], color='yellow')
    )

styled_df = style_dataframe(df)
```

---

## Common Pitfalls and Best Practices

### SettingWithCopyWarning

#### Why It Happens

```python
# Chained indexing creates intermediate view
df[df['A'] > 0]['B'] = 1  # Warning: may not modify original

# Pandas can't guarantee if modification affects original or copy
```

#### How to Fix

```python
# Use .loc for assignment
df.loc[df['A'] > 0, 'B'] = 1  # Correct

# Or explicit copy
df_filtered = df[df['A'] > 0].copy()
df_filtered['B'] = 1
```

### Mixed Dtypes in Columns

```python
# PROBLEM: Object dtype for mixed types
df = pd.DataFrame({'col': [1, 'string', 3.0]})  # All become object
print(df['col'].dtype)  # object

# Performance hit: object dtype is slow
# SOLUTION: Use separate columns or consistent types
df = pd.DataFrame({
    'numeric': [1, None, 3.0],
    'text': [None, 'string', None]
})
```

### Chained Indexing Anti-pattern

```python
# BAD: Chained indexing
value = df[df['A'] > 0]['B'].iloc[0]  # Multiple operations

# GOOD: Single .loc
value = df.loc[df['A'] > 0, 'B'].iloc[0]

# BAD: Chained assignment
df[df['A'] > 0]['B'] = 1

# GOOD: Direct assignment
df.loc[df['A'] > 0, 'B'] = 1
```

### Using apply When Vectorized Alternative Exists

```python
# BAD: apply for simple operation
df['sum'] = df.apply(lambda row: row['A'] + row['B'], axis=1)

# GOOD: Vectorized
df['sum'] = df['A'] + df['B']

# BAD: apply for conditional
df['flag'] = df.apply(lambda row: 1 if row['A'] > 0 else 0, axis=1)

# GOOD: Vectorized
df['flag'] = (df['A'] > 0).astype(int)
df['flag'] = np.where(df['A'] > 0, 1, 0)
```

### Silently Dropped NaN in groupby

```python
# NaN values are excluded from groupby by default
df = pd.DataFrame({
    'category': ['A', 'A', None, 'B', 'B'],
    'value': [1, 2, 3, 4, 5]
})

# NaN category is excluded
result = df.groupby('category').sum()  # Only A and B groups

# To include NaN
result = df.groupby('category', dropna=False).sum()  # Includes NaN group
```

### Merge Gotchas

#### Duplicate Keys

```python
# Duplicate keys can cause row multiplication
df1 = pd.DataFrame({'key': [1, 1, 2], 'A': [1, 2, 3]})
df2 = pd.DataFrame({'key': [1, 1, 2], 'B': [4, 5, 6]})

# Result has more rows than expected
result = pd.merge(df1, df2, on='key')  # 4 rows instead of 3

# Check for duplicates before merge
assert df1['key'].is_unique, "Duplicate keys in df1"
assert df2['key'].is_unique, "Duplicate keys in df2"
```

#### Unexpected Row Multiplication

```python
# Many-to-many merge
df1 = pd.DataFrame({'key': [1, 2], 'A': [1, 2]})
df2 = pd.DataFrame({'key': [1, 1, 2], 'B': [3, 4, 5]})

# Result: 3 rows (1*2 + 1*1 = 3)
result = pd.merge(df1, df2, on='key')
```

### Index Alignment in Operations

```python
# Misaligned indices create NaN
s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
s2 = pd.Series([4, 5, 6], index=[1, 2, 3])

result = s1 + s2  # NaN at indices 0 and 3

# Align indices first
s1_aligned, s2_aligned = s1.align(s2, fill_value=0)
result = s1_aligned + s2_aligned
```

### Datetime Parsing Pitfalls

#### Ambiguous Formats

```python
# Ambiguous date: 01/02/03
# Could be: Jan 2, 2003 or Feb 1, 2003 or Jan 2, 2003

# Specify format explicitly
pd.to_datetime('01/02/03', format='%m/%d/%y')  # Explicit format

# Or use dayfirst parameter
pd.to_datetime('01/02/03', dayfirst=True)
```

#### Timezone Handling

```python
# PROBLEM: Mixing timezone-aware and naive
ts1 = pd.Timestamp('2023-01-01', tz='UTC')
ts2 = pd.Timestamp('2023-01-01')  # Naive

# Operations may fail or produce unexpected results
# SOLUTION: Ensure consistent timezone handling
ts2 = ts2.tz_localize('UTC')
result = ts1 + pd.Timedelta(days=1)
```

### Best Practices Summary

1. **Always use vectorized operations** when possible
2. **Avoid chained indexing** - use .loc instead
3. **Specify dtypes** when reading files
4. **Use categories** for repeated string values
5. **Process large files in chunks**
6. **Collect data in lists** before creating DataFrame
7. **Use method chaining** for readability
8. **Test with assert_frame_equal** for DataFrames
9. **Handle timezones explicitly** in datetime operations
10. **Check for duplicates** before merge operations
11. **Use .to_numpy()** instead of .values for clarity
12. **Enable Copy-on-Write** in Pandas 2.0+ for safety
13. **Use nullable dtypes** (Int64, string, boolean) when appropriate
14. **Profile memory usage** with memory_usage(deep=True)
15. **Use query() and eval()** for complex filtering/expressions

This comprehensive guide covers performance optimization, advanced features, and best practices for working effectively with Pandas.
