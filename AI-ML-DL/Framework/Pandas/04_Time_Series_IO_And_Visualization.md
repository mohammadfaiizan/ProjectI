# Time Series, I/O, and Visualization in Pandas

## Table of Contents
1. [Datetime Fundamentals](#datetime-fundamentals)
2. [Time Series Operations](#time-series-operations)
3. [File I/O](#file-io)
4. [Visualization](#visualization)

---

## Datetime Fundamentals

### pd.Timestamp

`pd.Timestamp` is Pandas' replacement for Python's `datetime.datetime`. It provides nanosecond precision and is timezone-aware.

#### Creation

```python
import pandas as pd
import numpy as np

# From string
ts1 = pd.Timestamp('2023-01-15')
ts2 = pd.Timestamp('2023-01-15 14:30:00')
ts3 = pd.Timestamp('2023-01-15 14:30:00.123456789')

# From datetime
from datetime import datetime
ts4 = pd.Timestamp(datetime(2023, 1, 15, 14, 30))

# From Unix timestamp
ts5 = pd.Timestamp(1673808000, unit='s')  # seconds
ts6 = pd.Timestamp(1673808000000000000, unit='ns')  # nanoseconds

# Current time
ts7 = pd.Timestamp.now()
ts8 = pd.Timestamp.today()
```

#### Attributes

`pd.Timestamp` provides access to all datetime components:

```python
ts = pd.Timestamp('2023-03-15 14:30:45')

# Date components
ts.year        # 2023
ts.month       # 3
ts.day         # 15
ts.dayofweek   # 2 (Monday=0, Wednesday=2)
ts.dayofyear   # 74
ts.quarter     # 1
ts.week        # 11

# Time components
ts.hour        # 14
ts.minute      # 30
ts.second      # 45
ts.microsecond # 0
ts.nanosecond  # 0

# Other attributes
ts.date        # datetime.date(2023, 3, 15)
ts.time        # datetime.time(14, 30, 45)
ts.tz          # None (timezone-naive)
ts.freq        # None
```

#### Methods

```python
ts = pd.Timestamp('2023-03-15 14:30:45')

# Conversion methods
ts.strftime('%Y-%m-%d %H:%M:%S')  # '2023-03-15 14:30:45'
ts.isoformat()                     # '2023-03-15T14:30:45'

# Rounding
ts.floor('D')    # Timestamp('2023-03-15 00:00:00')
ts.ceil('H')     # Timestamp('2023-03-15 15:00:00')
ts.round('30min') # Timestamp('2023-03-15 14:30:00')

# Arithmetic
ts + pd.Timedelta(days=5)  # Timestamp('2023-03-20 14:30:45')
ts - pd.Timedelta(hours=2) # Timestamp('2023-03-15 12:30:45')

# Comparison
ts > pd.Timestamp('2023-01-01')  # True
```

### pd.to_datetime

`pd.to_datetime()` converts various input types to datetime objects.

#### Parsing Strings

```python
# Single string
pd.to_datetime('2023-01-15')
pd.to_datetime('15/01/2023', format='%d/%m/%Y')

# List/array of strings
pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-25'])

# Series
dates = pd.Series(['2023-01-15', '2023-02-20', '2023-03-25'])
pd.to_datetime(dates)
```

#### Format Parameter

```python
# Explicit format specification
pd.to_datetime('15-01-2023', format='%d-%m-%Y')
pd.to_datetime('2023/01/15 14:30', format='%Y/%m/%d %H:%M')

# Common format codes:
# %Y: 4-digit year
# %m: month (01-12)
# %d: day (01-31)
# %H: hour (00-23)
# %M: minute (00-59)
# %S: second (00-59)
```

#### Errors Parameter

```python
# raise (default): raise error on unparseable dates
pd.to_datetime(['2023-01-15', 'invalid', '2023-03-25'], errors='raise')
# Raises: ParserError

# coerce: convert unparseable to NaT
pd.to_datetime(['2023-01-15', 'invalid', '2023-03-25'], errors='coerce')
# Returns: DatetimeIndex(['2023-01-15', NaT, '2023-03-25'])

# ignore: return original input
pd.to_datetime(['2023-01-15', 'invalid', '2023-03-25'], errors='ignore')
# Returns: Index(['2023-01-15', 'invalid', '2023-03-25'], dtype='object')
```

#### infer_datetime_format

```python
# When True, attempts to infer format for faster parsing
pd.to_datetime(['2023-01-15', '2023-02-20'], infer_datetime_format=True)
```

#### Additional Parameters

```python
# Parsing dates from multiple columns
df = pd.DataFrame({
    'year': [2023, 2023, 2023],
    'month': [1, 2, 3],
    'day': [15, 20, 25]
})
pd.to_datetime(df[['year', 'month', 'day']])

# Unit parameter for Unix timestamps
pd.to_datetime([1673808000, 1673894400], unit='s')

# Origin parameter for custom epoch
pd.to_datetime([1, 2, 3], unit='D', origin='2023-01-01')
```

### DatetimeIndex

`DatetimeIndex` is an index optimized for datetime data.

#### Creation

```python
# From list of strings
pd.DatetimeIndex(['2023-01-15', '2023-02-20', '2023-03-25'])

# From date_range
pd.date_range('2023-01-01', '2023-12-31', freq='D')

# From period_range converted
pd.period_range('2023-01', '2023-12', freq='M').to_timestamp()
```

#### freq Parameter

The `freq` parameter specifies the frequency of the datetime index:

```python
# Daily
pd.date_range('2023-01-01', periods=10, freq='D')

# Business days
pd.date_range('2023-01-01', periods=10, freq='B')

# Weekly
pd.date_range('2023-01-01', periods=10, freq='W')

# Monthly
pd.date_range('2023-01-01', periods=12, freq='M')  # Month end
pd.date_range('2023-01-01', periods=12, freq='MS') # Month start

# Quarterly
pd.date_range('2023-01-01', periods=4, freq='Q')    # Quarter end
pd.date_range('2023-01-01', periods=4, freq='QS')   # Quarter start

# Yearly
pd.date_range('2023-01-01', periods=5, freq='Y')     # Year end
pd.date_range('2023-01-01', periods=5, freq='YS')    # Year start

# Hourly
pd.date_range('2023-01-01', periods=24, freq='H')

# Minutes
pd.date_range('2023-01-01', periods=60, freq='T')   # or 'min'

# Seconds
pd.date_range('2023-01-01', periods=60, freq='S')
```

#### date_range

```python
# Basic usage
pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Using periods instead of end
pd.date_range(start='2023-01-01', periods=365, freq='D')

# Custom frequency
pd.date_range(start='2023-01-01', periods=10, freq='2D')  # Every 2 days
pd.date_range(start='2023-01-01', periods=10, freq='3H')  # Every 3 hours

# Timezone-aware
pd.date_range(start='2023-01-01', periods=10, freq='D', tz='UTC')
pd.date_range(start='2023-01-01', periods=10, freq='D', tz='America/New_York')

# Closed parameter
pd.date_range('2023-01-01', '2023-01-05', freq='D', closed='left')   # Excludes end
pd.date_range('2023-01-01', '2023-01-05', freq='D', closed='right')  # Excludes start
pd.date_range('2023-01-01', '2023-01-05', freq='D', closed='both')   # Includes both
pd.date_range('2023-01-01', '2023-01-05', freq='D', closed='neither') # Excludes both
```

#### bdate_range

`bdate_range` creates a range of business days:

```python
# Basic business days
pd.bdate_range('2023-01-01', '2023-01-31')

# Custom business days per week
pd.bdate_range('2023-01-01', '2023-01-31', freq='C', weekmask='Mon Wed Fri')

# Using holidays
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
pd.bdate_range('2023-01-01', '2023-12-31', freq='C', calendar=cal)
```

### pd.Period and PeriodIndex

Periods represent a time span (e.g., a month, a quarter) rather than a specific point in time.

#### Period vs Timestamp

```python
# Timestamp: specific point in time
ts = pd.Timestamp('2023-01-15 14:30:00')

# Period: time span
p = pd.Period('2023-01', freq='M')  # Represents entire January 2023

# Period can be converted to timestamp
p.to_timestamp()           # Start of period: Timestamp('2023-01-01')
p.to_timestamp(how='end')  # End of period: Timestamp('2023-01-31 23:59:59.999999999')
```

#### period_range

```python
# Monthly periods
pd.period_range('2023-01', '2023-12', freq='M')

# Daily periods
pd.period_range('2023-01-01', '2023-01-31', freq='D')

# Quarterly periods
pd.period_range('2023Q1', '2023Q4', freq='Q')

# Yearly periods
pd.period_range('2020', '2023', freq='Y')
```

#### to_period and to_timestamp

```python
# Convert DatetimeIndex to PeriodIndex
dates = pd.date_range('2023-01-01', periods=12, freq='M')
dates.to_period('M')  # PeriodIndex with monthly periods

# Convert PeriodIndex to DatetimeIndex
periods = pd.period_range('2023-01', '2023-12', freq='M')
periods.to_timestamp()           # Start of each period
periods.to_timestamp(how='end')  # End of each period
```

#### Period Operations

```python
p = pd.Period('2023-01', freq='M')

# Arithmetic
p + 1  # Period('2023-02', 'M')
p - 1  # Period('2022-12', 'M')

# Properties
p.year   # 2023
p.month  # 1
p.start_time  # Timestamp('2023-01-01 00:00:00')
p.end_time    # Timestamp('2023-01-31 23:59:59.999999999')
```

### pd.Timedelta and TimedeltaIndex

`Timedelta` represents a duration or difference between two dates/times.

#### Creation

```python
# From string
pd.Timedelta('1 days')
pd.Timedelta('1 days 2 hours 30 minutes')
pd.Timedelta('1 days 02:30:00')

# From components
pd.Timedelta(days=1, hours=2, minutes=30, seconds=15)
pd.Timedelta(weeks=2, days=3)

# From numeric with unit
pd.Timedelta(1, unit='D')   # 1 day
pd.Timedelta(24, unit='H')  # 24 hours
pd.Timedelta(3600, unit='S') # 3600 seconds
```

#### timedelta_range

```python
# Create range of timedeltas
pd.timedelta_range(start='1 day', periods=10, freq='D')
pd.timedelta_range(start='1 hour', periods=24, freq='H')
```

#### Components

```python
td = pd.Timedelta('1 days 2 hours 30 minutes 15 seconds')

td.days              # 1
td.seconds           # 9015 (total seconds in the time component)
td.microseconds      # 0
td.nanoseconds       # 0

# Component access
td.components.days        # 1
td.components.hours       # 2
td.components.minutes     # 30
td.components.seconds     # 15
td.components.milliseconds # 0
td.components.microseconds # 0
td.components.nanoseconds  # 0

# Total time as different units
td.total_seconds()   # 95415.0
td / pd.Timedelta(hours=1)  # 39.8375 hours
```

### Date Offsets

Date offsets allow flexible date arithmetic.

#### DateOffset

```python
# Generic date offset
pd.DateOffset(days=5)
pd.DateOffset(months=2, days=3)

# Usage
ts = pd.Timestamp('2023-01-15')
ts + pd.DateOffset(days=5)  # Timestamp('2023-01-20')
ts + pd.DateOffset(months=2) # Timestamp('2023-03-15')
```

#### BDay (Business Day)

```python
# Business day offset
ts = pd.Timestamp('2023-01-15')  # Sunday
ts + pd.offsets.BDay()            # Next business day: Timestamp('2023-01-16')
ts + pd.offsets.BDay(5)           # 5 business days later
```

#### MonthEnd, MonthBegin

```python
ts = pd.Timestamp('2023-01-15')

# Month end
ts + pd.offsets.MonthEnd()    # Timestamp('2023-01-31')
ts + pd.offsets.MonthEnd(2)   # Timestamp('2023-02-28')

# Month begin
ts + pd.offsets.MonthBegin()  # Timestamp('2023-02-01')
ts + pd.offsets.MonthBegin(-1) # Timestamp('2023-01-01')
```

#### YearEnd, YearBegin

```python
ts = pd.Timestamp('2023-06-15')

# Year end
ts + pd.offsets.YearEnd()     # Timestamp('2023-12-31')
ts + pd.offsets.YearEnd(1)    # Timestamp('2024-12-31')

# Year begin
ts + pd.offsets.YearBegin()   # Timestamp('2024-01-01')
ts + pd.offsets.YearBegin(-1) # Timestamp('2023-01-01')
```

#### CustomBusinessDay

```python
# Custom business days
from pandas.tseries.offsets import CustomBusinessDay

# Monday, Wednesday, Friday only
weekmask = 'Mon Wed Fri'
cbd = CustomBusinessDay(weekmask=weekmask)
ts = pd.Timestamp('2023-01-15')  # Sunday
ts + cbd  # Timestamp('2023-01-16') (Monday)

# With holidays
from pandas.tseries.holiday import USFederalHolidayCalendar
cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
```

#### Week

```python
# Week offset
ts = pd.Timestamp('2023-01-15')
ts + pd.offsets.Week()        # Timestamp('2023-01-22')
ts + pd.offsets.Week(2)       # Timestamp('2023-01-29')

# Week of month
ts + pd.offsets.WeekOfMonth(week=1, weekday=0)  # First Monday of month
```

---

## Time Series Operations

### Indexing and Slicing with Datetime Strings

Pandas allows intuitive datetime-based indexing when the index is a DatetimeIndex.

```python
# Create sample data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
df = pd.DataFrame({'value': np.random.randn(len(dates))}, index=dates)

# Single year
df['2023']

# Specific month
df['2023-01']
df['2023-02']

# Date range
df['2023-01':'2023-06']
df['2023-01-15':'2023-03-20']

# Partial string matching
df.loc['2023-01']  # All January 2023
df.loc['2023-01-15':'2023-01-20']  # Specific date range
```

### Resampling

Resampling converts time series data to a different frequency.

#### Downsampling (Aggregation)

Downsampling reduces frequency by aggregating data:

```python
# Daily to monthly
daily_data = pd.DataFrame({
    'value': np.random.randn(365)
}, index=pd.date_range('2023-01-01', periods=365, freq='D'))

# Aggregate to monthly
monthly = daily_data.resample('M').sum()      # Sum
monthly = daily_data.resample('M').mean()    # Mean
monthly = daily_data.resample('M').first()   # First value
monthly = daily_data.resample('M').last()    # Last value
monthly = daily_data.resample('M').count()   # Count

# OHLC (Open-High-Low-Close) for financial data
ohlc = daily_data.resample('M').ohlc()

# Multiple aggregations
daily_data.resample('M').agg(['sum', 'mean', 'std', 'min', 'max'])

# Custom aggregation
daily_data.resample('M').apply(lambda x: x.max() - x.min())  # Range
```

#### Upsampling (Interpolation)

Upsampling increases frequency, requiring interpolation:

```python
# Monthly to daily
monthly_data = pd.DataFrame({
    'value': np.random.randn(12)
}, index=pd.date_range('2023-01-01', periods=12, freq='M'))

# Forward fill (carry last value forward)
daily_ffill = monthly_data.resample('D').ffill()

# Backward fill (carry next value backward)
daily_bfill = monthly_data.resample('D').bfill()

# Interpolation
daily_interp = monthly_data.resample('D').interpolate(method='linear')
daily_interp = monthly_data.resample('D').interpolate(method='spline', order=2)

# Nearest value
daily_nearest = monthly_data.resample('D').nearest()

# As frequency (repeats value)
daily_asfreq = monthly_data.resample('D').asfreq()
```

#### Common Resampling Rules

| Rule | Description | Example |
|------|-------------|---------|
| `D` | Daily | 'D' |
| `B` | Business day | 'B' |
| `W` | Weekly | 'W' |
| `M` | Month end | 'M' |
| `MS` | Month start | 'MS' |
| `Q` | Quarter end | 'Q' |
| `QS` | Quarter start | 'QS' |
| `Y` | Year end | 'Y' |
| `YS` | Year start | 'YS' |
| `H` | Hourly | 'H' |
| `T` or `min` | Minute | 'T' |
| `S` | Second | 'S' |
| `L` or `ms` | Millisecond | 'L' |
| `U` or `us` | Microsecond | 'U' |
| `N` | Nanosecond | 'N' |

#### Resampling with Custom Functions

```python
# Custom aggregation function
def custom_agg(x):
    return x.quantile(0.75) - x.quantile(0.25)  # IQR

daily_data.resample('M').apply(custom_agg)

# Named aggregation
daily_data.resample('M').agg(
    total=('value', 'sum'),
    average=('value', 'mean'),
    range=('value', lambda x: x.max() - x.min())
)
```

### Rolling Windows

Rolling windows compute statistics over a sliding window.

```python
# Create sample data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
df = pd.DataFrame({'value': np.random.randn(100)}, index=dates)

# Simple rolling window
df['rolling_mean'] = df['value'].rolling(window=7).mean()   # 7-day rolling mean
df['rolling_sum'] = df['value'].rolling(window=7).sum()     # 7-day rolling sum
df['rolling_std'] = df['value'].rolling(window=7).std()     # 7-day rolling std
df['rolling_min'] = df['value'].rolling(window=7).min()     # 7-day rolling min
df['rolling_max'] = df['value'].rolling(window=7).max()     # 7-day rolling max

# Rolling window with offset string
df['rolling_mean'] = df['value'].rolling(window='7D').mean()  # 7-day window
df['rolling_mean'] = df['value'].rolling(window='30D').mean() # 30-day window

# Custom function
df['rolling_custom'] = df['value'].rolling(window=7).apply(lambda x: x.max() - x.min())

# Center parameter (center the window)
df['rolling_mean_centered'] = df['value'].rolling(window=7, center=True).mean()

# min_periods (minimum observations required)
df['rolling_mean'] = df['value'].rolling(window=7, min_periods=1).mean()

# Multiple windows
df['rolling_7d'] = df['value'].rolling(window=7).mean()
df['rolling_30d'] = df['value'].rolling(window=30).mean()
df['rolling_90d'] = df['value'].rolling(window=90).mean()
```

### Expanding Windows

Expanding windows compute cumulative statistics:

```python
df['expanding_mean'] = df['value'].expanding().mean()
df['expanding_sum'] = df['value'].expanding().sum()
df['expanding_std'] = df['value'].expanding().std()
df['expanding_min'] = df['value'].expanding().min()
df['expanding_max'] = df['value'].expanding().max()

# Cumulative functions (equivalent)
df['cumsum'] = df['value'].cumsum()
df['cummean'] = df['value'].expanding().mean()
df['cummax'] = df['value'].cummax()
df['cummin'] = df['value'].cummin()
```

### Exponential Weighted Moving (EWM)

Exponential weighted functions give more weight to recent observations:

```python
# Using span (half-life = span / log(2))
df['ewm_mean'] = df['value'].ewm(span=7).mean()
df['ewm_std'] = df['value'].ewm(span=7).std()
df['ewm_var'] = df['value'].ewm(span=7).var()

# Using alpha (smoothing factor, 0 < alpha <= 1)
df['ewm_mean'] = df['value'].ewm(alpha=0.3).mean()

# Using halflife (time for weight to reduce to half)
df['ewm_mean'] = df['value'].ewm(halflife='7 days').mean()

# Adjust parameter (bias adjustment)
df['ewm_mean'] = df['value'].ewm(span=7, adjust=True).mean()   # Default
df['ewm_mean'] = df['value'].ewm(span=7, adjust=False).mean()  # No bias adjustment
```

### shift, diff, pct_change

#### shift (Lagging)

```python
# Shift forward (lag)
df['lag_1'] = df['value'].shift(1)   # Previous value
df['lag_7'] = df['value'].shift(7)   # 7 periods ago

# Shift backward (lead)
df['lead_1'] = df['value'].shift(-1) # Next value

# Shift with frequency
df['lag_1_month'] = df['value'].shift(1, freq='M')  # Previous month
```

#### diff (Differencing)

```python
# First difference
df['diff_1'] = df['value'].diff(1)   # Current - previous

# Second difference
df['diff_2'] = df['value'].diff(2)   # Current - 2 periods ago

# Periods parameter
df['diff'] = df['value'].diff(periods=1)
```

#### pct_change (Percentage Change)

```python
# Percentage change
df['pct_change'] = df['value'].pct_change()        # Default: 1 period
df['pct_change'] = df['value'].pct_change(periods=7) # 7-period change

# Fill method
df['pct_change'] = df['value'].pct_change(fill_method='ffill')  # Forward fill NaN
df['pct_change'] = df['value'].pct_change(fill_method='bfill')  # Backward fill NaN
```

### Timezone Operations

#### tz_localize

Convert timezone-naive to timezone-aware:

```python
# Localize to UTC
dates = pd.date_range('2023-01-01', periods=10, freq='D')
df = pd.DataFrame({'value': np.random.randn(10)}, index=dates)
df.index = df.index.tz_localize('UTC')

# Localize to specific timezone
df.index = df.index.tz_localize('America/New_York')

# Ambiguous times handling
df.index = df.index.tz_localize('America/New_York', ambiguous='infer')
df.index = df.index.tz_localize('America/New_York', ambiguous='NaT')
df.index = df.index.tz_localize('America/New_York', ambiguous='raise')
```

#### tz_convert

Convert between timezones:

```python
# Convert UTC to Eastern Time
df.index = df.index.tz_convert('America/New_York')

# Convert to different timezone
df.index = df.index.tz_convert('Asia/Tokyo')
```

### Time-of-Day Filtering

#### between_time

Filter data between specific times of day:

```python
# Create hourly data
dates = pd.date_range('2023-01-01', periods=48, freq='H')
df = pd.DataFrame({'value': np.random.randn(48)}, index=dates)

# Filter between 9 AM and 5 PM
df.between_time('09:00', '17:00')

# Inclusive/exclusive
df.between_time('09:00', '17:00', inclusive='both')    # Default
df.between_time('09:00', '17:00', inclusive='left')    # Exclude end
df.between_time('09:00', '17:00', inclusive='right')   # Exclude start
df.between_time('09:00', '17:00', inclusive='neither') # Exclude both
```

#### at_time

Filter data at specific time:

```python
# Get all records at 9 AM
df.at_time('09:00')

# Get all records at midnight
df.at_time('00:00')
```

---

## File I/O

### CSV Files

#### read_csv

```python
# Basic reading
df = pd.read_csv('file.csv')

# File path options
df = pd.read_csv('file.csv')
df = pd.read_csv('/absolute/path/to/file.csv')
df = pd.read_csv('https://example.com/data.csv')

# Separator
df = pd.read_csv('file.csv', sep=',')        # Default
df = pd.read_csv('file.csv', delimiter=';')  # Alternative name
df = pd.read_csv('file.tsv', sep='\t')       # Tab-separated

# Header
df = pd.read_csv('file.csv', header=0)       # First row as header (default)
df = pd.read_csv('file.csv', header=None)    # No header
df = pd.read_csv('file.csv', header=[0, 1])  # MultiIndex header

# Column names
df = pd.read_csv('file.csv', names=['col1', 'col2', 'col3'])
df = pd.read_csv('file.csv', names=['col1', 'col2'], header=0)  # Replace header

# Index column
df = pd.read_csv('file.csv', index_col=0)           # First column as index
df = pd.read_csv('file.csv', index_col='date')      # Named column as index
df = pd.read_csv('file.csv', index_col=[0, 1])     # MultiIndex

# Select columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])
df = pd.read_csv('file.csv', usecols=[0, 2, 4])    # By position
df = pd.read_csv('file.csv', usecols=lambda x: 'date' in x.lower())  # By function

# Data types
df = pd.read_csv('file.csv', dtype={'col1': 'int64', 'col2': 'float64'})
df = pd.read_csv('file.csv', dtype=str)            # All as string

# Parse dates
df = pd.read_csv('file.csv', parse_dates=['date'])
df = pd.read_csv('file.csv', parse_dates=[0])      # First column
df = pd.read_csv('file.csv', parse_dates=[['year', 'month', 'day']])
df = pd.read_csv('file.csv', parse_dates=True, index_col=0)  # Parse index

# Date parser (deprecated, use format in to_datetime)
from dateutil.parser import parse
df = pd.read_csv('file.csv', parse_dates=['date'], date_parser=parse)

# Missing values
df = pd.read_csv('file.csv', na_values=['NA', 'NULL', ''])
df = pd.read_csv('file.csv', na_values={'col1': ['NA'], 'col2': ['NULL']})
df = pd.read_csv('file.csv', keep_default_na=False)  # Don't treat NaN as missing

# Skip rows
df = pd.read_csv('file.csv', skiprows=5)            # Skip first 5 rows
df = pd.read_csv('file.csv', skiprows=[0, 2, 5])    # Skip specific rows
df = pd.read_csv('file.csv', skiprows=lambda x: x % 2 == 0)  # Skip even rows

# Number of rows
df = pd.read_csv('file.csv', nrows=1000)            # Read only first 1000 rows

# Chunking (for large files)
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process(chunk)

# Encoding
df = pd.read_csv('file.csv', encoding='utf-8')      # Default
df = pd.read_csv('file.csv', encoding='latin-1')   # Latin-1
df = pd.read_csv('file.csv', encoding='cp1252')    # Windows-1252

# Compression
df = pd.read_csv('file.csv.gz', compression='gzip')
df = pd.read_csv('file.csv.zip', compression='zip')
df = pd.read_csv('file.csv.bz2', compression='bz2')

# Converters
def convert_to_upper(x):
    return x.upper() if pd.notna(x) else x

df = pd.read_csv('file.csv', converters={'name': convert_to_upper})

# Comments
df = pd.read_csv('file.csv', comment='#')          # Skip lines starting with #

# Thousands separator
df = pd.read_csv('file.csv', thousands=',')        # Remove commas from numbers
```

#### to_csv

```python
# Basic writing
df.to_csv('output.csv')

# Path
df.to_csv('output.csv')
df.to_csv('/absolute/path/output.csv')

# Separator
df.to_csv('output.csv', sep=',')        # Default
df.to_csv('output.csv', sep='\t')       # Tab-separated

# Header
df.to_csv('output.csv', header=True)    # Default
df.to_csv('output.csv', header=False)   # No header
df.to_csv('output.csv', header=['col1', 'col2'])  # Custom header

# Index
df.to_csv('output.csv', index=True)     # Default
df.to_csv('output.csv', index=False)    # Don't write index

# Columns
df.to_csv('output.csv', columns=['col1', 'col2'])

# Missing value representation
df.to_csv('output.csv', na_rep='NA')    # Default: empty string

# Float formatting
df.to_csv('output.csv', float_format='%.2f')

# Compression
df.to_csv('output.csv.gz', compression='gzip')
df.to_csv('output.csv.zip', compression='zip')

# Mode
df.to_csv('output.csv', mode='w')       # Overwrite (default)
df.to_csv('output.csv', mode='a')       # Append
```

### Excel Files

#### read_excel

```python
# Basic reading
df = pd.read_excel('file.xlsx')

# File path
df = pd.read_excel('file.xlsx')
df = pd.read_excel('/absolute/path/file.xlsx')

# Sheet selection
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')     # By name
df = pd.read_excel('file.xlsx', sheet_name=0)            # By index
df = pd.read_excel('file.xlsx', sheet_name=[0, 1])       # Multiple sheets (returns dict)
df = pd.read_excel('file.xlsx', sheet_name=None)         # All sheets (returns dict)

# Header
df = pd.read_excel('file.xlsx', header=0)                # First row (default)
df = pd.read_excel('file.xlsx', header=None)             # No header

# Column names
df = pd.read_excel('file.xlsx', names=['col1', 'col2'])

# Select columns
df = pd.read_excel('file.xlsx', usecols='A:C')           # Column range
df = pd.read_excel('file.xlsx', usecols=[0, 2, 4])       # By position
df = pd.read_excel('file.xlsx', usecols=['col1', 'col2']) # By name

# Data types
df = pd.read_excel('file.xlsx', dtype={'col1': 'int64'})

# Engine
df = pd.read_excel('file.xlsx', engine='openpyxl')       # .xlsx files
df = pd.read_excel('file.xls', engine='xlrd')            # .xls files
```

#### to_excel

```python
# Basic writing
df.to_excel('output.xlsx')

# ExcelWriter for multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# Or without context manager
writer = pd.ExcelWriter('output.xlsx')
df1.to_excel(writer, sheet_name='Sheet1')
df2.to_excel(writer, sheet_name='Sheet2')
writer.save()
writer.close()

# Sheet name
df.to_excel('output.xlsx', sheet_name='Data')

# Index and header
df.to_excel('output.xlsx', index=False, header=False)

# Columns
df.to_excel('output.xlsx', columns=['col1', 'col2'])

# Engine
df.to_excel('output.xlsx', engine='openpyxl')
```

### JSON Files

#### read_json

```python
# Basic reading
df = pd.read_json('file.json')

# Orient parameter
# 'split': {index: [...], columns: [...], data: [[...]]}
df = pd.read_json('file.json', orient='split')

# 'records': [{col1: val1, col2: val2}, ...]
df = pd.read_json('file.json', orient='records')

# 'index': {index: {col: val, ...}, ...}
df = pd.read_json('file.json', orient='index')

# 'columns': {col: {index: val, ...}, ...}
df = pd.read_json('file.json', orient='columns')

# 'values': [[val1, val2, ...], ...]
df = pd.read_json('file.json', orient='values')

# Lines format (JSONL)
df = pd.read_json('file.jsonl', lines=True)

# Convert dates
df = pd.read_json('file.json', convert_dates=['date'])
df = pd.read_json('file.json', convert_dates=True)  # Auto-detect
```

#### to_json

```python
# Basic writing
df.to_json('output.json')

# Orient
df.to_json('output.json', orient='split')
df.to_json('output.json', orient='records')
df.to_json('output.json', orient='index')
df.to_json('output.json', orient='columns')
df.to_json('output.json', orient='values')

# Lines format
df.to_json('output.jsonl', orient='records', lines=True)

# Date format
df.to_json('output.json', date_format='epoch')      # Unix timestamp
df.to_json('output.json', date_format='iso')       # ISO format
```

### SQL Databases

#### read_sql

```python
import sqlite3

# Create connection
conn = sqlite3.connect('database.db')

# Read SQL query
df = pd.read_sql('SELECT * FROM table_name', conn)

# With parameters
df = pd.read_sql('SELECT * FROM table WHERE id = ?', conn, params=[1])

# Index column
df = pd.read_sql('SELECT * FROM table', conn, index_col='id')

# Parse dates
df = pd.read_sql('SELECT * FROM table', conn, parse_dates=['date'])

# Chunking
for chunk in pd.read_sql('SELECT * FROM table', conn, chunksize=1000):
    process(chunk)
```

#### read_sql_table and read_sql_query

```python
# Read entire table
df = pd.read_sql_table('table_name', conn)

# Read query
df = pd.read_sql_query('SELECT * FROM table WHERE condition', conn)
```

#### to_sql

```python
# Basic writing
df.to_sql('table_name', conn, if_exists='fail')     # Default: fail if exists
df.to_sql('table_name', conn, if_exists='replace') # Replace table
df.to_sql('table_name', conn, if_exists='append')  # Append rows

# Index
df.to_sql('table_name', conn, index=False)          # Don't write index

# Data types
df.to_sql('table_name', conn, dtype={'col1': 'INTEGER', 'col2': 'TEXT'})

# Method for batch insert
df.to_sql('table_name', conn, method='multi')      # Faster for large DataFrames
```

### Parquet Files

#### read_parquet

```python
# Basic reading
df = pd.read_parquet('file.parquet')

# Engine
df = pd.read_parquet('file.parquet', engine='pyarrow')      # Default
df = pd.read_parquet('file.parquet', engine='fastparquet')

# Select columns
df = pd.read_parquet('file.parquet', columns=['col1', 'col2'])

# Filters (pyarrow)
df = pd.read_parquet('file.parquet', filters=[('col1', '>', 100)])
```

#### to_parquet

```python
# Basic writing
df.to_parquet('output.parquet')

# Engine
df.to_parquet('output.parquet', engine='pyarrow')

# Compression
df.to_parquet('output.parquet', compression='snappy')  # Default
df.to_parquet('output.parquet', compression='gzip')
df.to_parquet('output.parquet', compression='brotli')

# Partition columns
df.to_parquet('output_dir', partition_cols=['year', 'month'])

# Index
df.to_parquet('output.parquet', index=False)
```

### Other Formats

#### HTML

```python
# Read HTML tables
tables = pd.read_html('https://example.com/page.html')
df = tables[0]  # First table

# Write HTML
df.to_html('output.html')
df.to_html('output.html', index=False, classes='table')
```

#### Clipboard

```python
# Read from clipboard
df = pd.read_clipboard()

# Write to clipboard
df.to_clipboard()
df.to_clipboard(index=False)
```

#### HDF5

```python
# Read HDF5
df = pd.read_hdf('file.h5', key='data')

# Write HDF5
df.to_hdf('output.h5', key='data', mode='w')
df.to_hdf('output.h5', key='data', mode='a', format='table')  # Append
```

#### Feather

```python
# Read Feather
df = pd.read_feather('file.feather')

# Write Feather
df.to_feather('output.feather')
```

#### Pickle

```python
# Read Pickle
df = pd.read_pickle('file.pkl')

# Write Pickle
df.to_pickle('output.pkl')
df.to_pickle('output.pkl', compression='gzip')  # Compressed
```

---

## Visualization

### DataFrame.plot() and Series.plot()

Pandas provides built-in plotting capabilities using matplotlib.

#### Basic Plotting

```python
import matplotlib.pyplot as plt

# Series plot
s = pd.Series([1, 3, 2, 4, 5])
s.plot()

# DataFrame plot
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 1, 3, 5],
    'C': [3, 1, 4, 2, 5]
})
df.plot()
```

### Line Plot

```python
# Default line plot
df.plot()                    # All columns
df.plot(kind='line')         # Explicit

# Single column
df['A'].plot()

# Multiple columns
df[['A', 'B']].plot()

# Secondary y-axis
ax = df['A'].plot()
df['B'].plot(ax=ax, secondary_y=True)

# Or
df.plot(y='A')
df.plot(y='B', secondary_y=True, ax=plt.gca())
```

### Bar Plot

```python
# Vertical bar
df.plot(kind='bar')

# Horizontal bar
df.plot(kind='barh')

# Stacked bars
df.plot(kind='bar', stacked=True)

# Grouped bars (default)
df.plot(kind='bar', stacked=False)
```

### Histogram

```python
# Basic histogram
df['A'].plot(kind='hist')

# With bins
df['A'].plot(kind='hist', bins=20)

# Multiple columns
df[['A', 'B']].plot(kind='hist', alpha=0.7)

# By group
df.plot(kind='hist', by='category_column')
```

### Box Plot

```python
# Basic box plot
df.plot(kind='box')

# By group
df.plot(kind='box', by='category_column')

# Vertical (default)
df.plot(kind='box', vert=True)

# Horizontal
df.plot(kind='box', vert=False)
```

### Scatter Plot

```python
# Scatter plot (requires x and y)
df.plot(kind='scatter', x='A', y='B')

# With color
df.plot(kind='scatter', x='A', y='B', c='C', colormap='viridis')

# With size
df.plot(kind='scatter', x='A', y='B', s=df['C']*10)
```

### Area Plot

```python
# Basic area plot
df.plot(kind='area')

# Stacked area (default)
df.plot(kind='area', stacked=True)

# Unstacked area
df.plot(kind='area', stacked=False)
```

### Pie Chart

```python
# Pie chart (Series only)
s.plot(kind='pie')

# With labels
s.plot(kind='pie', labels=['A', 'B', 'C', 'D', 'E'])

# With autopct
s.plot(kind='pie', autopct='%1.1f%%')
```

### Plot Customization

```python
# Figure size
df.plot(figsize=(10, 6))

# Title
df.plot(title='My Plot')

# Axis labels
df.plot(xlabel='X Axis', ylabel='Y Axis')

# Grid
df.plot(grid=True)

# Legend
df.plot(legend=True)          # Default
df.plot(legend=False)
df.plot(legend='reverse')     # Reverse order

# Font size
df.plot(fontsize=12)

# Rotation
df.plot(rot=45)              # Rotate x-axis labels

# Color
df.plot(color=['red', 'blue', 'green'])
df.plot(color='red')         # Single color for all

# Style
df.plot(style=['-', '--', ':'])  # Line styles
df.plot(style='o-')              # Marker and line

# Axes object
fig, ax = plt.subplots()
df.plot(ax=ax)
ax.set_title('Custom Title')
```

### Subplots

```python
# Subplots (one per column)
df.plot(subplots=True)

# Layout
df.plot(subplots=True, layout=(2, 2))  # 2x2 grid
df.plot(subplots=True, layout=(3, 1))  # 3 rows, 1 column

# Share axes
df.plot(subplots=True, sharex=True)
df.plot(subplots=True, sharey=True)
```

### Advanced Plotting Examples

```python
# Time series plot
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_df = pd.DataFrame({
    'value': np.random.randn(100).cumsum()
}, index=dates)
ts_df.plot(title='Time Series')

# Multiple y-axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
df['A'].plot(ax=ax1, color='blue', label='A')
df['B'].plot(ax=ax2, color='red', label='B')
ax1.set_ylabel('A', color='blue')
ax2.set_ylabel('B', color='red')

# Custom styling
ax = df.plot(kind='bar', figsize=(12, 6))
ax.set_title('Custom Bar Chart', fontsize=16, fontweight='bold')
ax.set_xlabel('X Label', fontsize=12)
ax.set_ylabel('Y Label', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()
```

This comprehensive guide covers all aspects of time series operations, file I/O, and visualization in Pandas, providing detailed code examples and explanations for each feature.
