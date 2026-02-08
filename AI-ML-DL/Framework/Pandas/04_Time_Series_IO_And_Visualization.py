"""
Pandas Time Series, I/O, and Visualization
Comprehensive examples covering time series operations, file I/O, and plotting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("PANDAS TIME SERIES, I/O, AND VISUALIZATION")
print("=" * 80)

# ============================================================================
# TIMESTAMP
# ============================================================================
print("\n" + "=" * 80)
print("TIMESTAMP")
print("=" * 80)

# Timestamp creation
Timestamp1 = pd.Timestamp('2024-01-15')
Timestamp2 = pd.Timestamp('2024-01-15 14:30:00')
Timestamp3 = pd.Timestamp(2024, 1, 15, 14, 30, 0)

print("\nTimestamp examples:")
print(f"Timestamp1: {Timestamp1}")
print(f"Timestamp2: {Timestamp2}")
print(f"Timestamp3: {Timestamp3}")

# Timestamp attributes
print("\nTimestamp attributes:")
print(f"Year: {Timestamp2.year}")
print(f"Month: {Timestamp2.month}")
print(f"Day: {Timestamp2.day}")
print(f"Hour: {Timestamp2.hour}")
print(f"Minute: {Timestamp2.minute}")
print(f"Second: {Timestamp2.second}")

# ============================================================================
# TO_DATETIME
# ============================================================================
print("\n" + "=" * 80)
print("TO_DATETIME")
print("=" * 80)

# Parsing strings
DateStrings = ['2024-01-01', '2024-02-15', '2024-03-20']
ParsedDates = pd.to_datetime(DateStrings)
print("\nParsed dates from strings:")
print(ParsedDates)
print(f"Dtype: {ParsedDates.dtype}")

# With format
FormattedDates = pd.to_datetime(['01-15-2024', '02-20-2024'], format='%m-%d-%Y')
print("\nParsed dates with format:")
print(FormattedDates)

# With errors='coerce'
InvalidDates = ['2024-01-01', 'invalid', '2024-03-20']
CoercedDates = pd.to_datetime(InvalidDates, errors='coerce')
print("\nParsed dates with errors='coerce':")
print(CoercedDates)

# ============================================================================
# DATE_RANGE
# ============================================================================
print("\n" + "=" * 80)
print("DATE_RANGE")
print("=" * 80)

# Daily frequency
DailyRange = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
print("\nDaily frequency:")
print(DailyRange)

# Weekly frequency
WeeklyRange = pd.date_range(start='2024-01-01', periods=5, freq='W')
print("\nWeekly frequency:")
print(WeeklyRange)

# Monthly frequency
MonthlyRange = pd.date_range(start='2024-01-01', periods=6, freq='M')
print("\nMonthly frequency:")
print(MonthlyRange)

# Quarterly frequency
QuarterlyRange = pd.date_range(start='2024-01-01', periods=4, freq='Q')
print("\nQuarterly frequency:")
print(QuarterlyRange)

# Yearly frequency
YearlyRange = pd.date_range(start='2020-01-01', periods=5, freq='Y')
print("\nYearly frequency:")
print(YearlyRange)

# Hourly frequency
HourlyRange = pd.date_range(start='2024-01-01 00:00', periods=5, freq='H')
print("\nHourly frequency:")
print(HourlyRange)

# Business days frequency
BusinessRange = pd.date_range(start='2024-01-01', periods=5, freq='B')
print("\nBusiness days frequency:")
print(BusinessRange)

# ============================================================================
# DATETIMEINDEX
# ============================================================================
print("\n" + "=" * 80)
print("DATETIMEINDEX")
print("=" * 80)

DatetimeIndexDataFrame = pd.DataFrame({
    'Value': [100, 200, 300, 400, 500],
    'Count': [10, 20, 30, 40, 50]
}, index=pd.date_range('2024-01-01', periods=5, freq='D'))

print("\nDataFrame with DatetimeIndex:")
print(DatetimeIndexDataFrame)

# Slicing with strings
StringSlice = DatetimeIndexDataFrame['2024-01-02':'2024-01-04']
print("\nSlicing with strings:")
print(StringSlice)

# Partial string indexing
PartialIndex = DatetimeIndexDataFrame['2024-01']
print("\nPartial string indexing (January 2024):")
print(PartialIndex)

# ============================================================================
# TIMEDELTA
# ============================================================================
print("\n" + "=" * 80)
print("TIMEDELTA")
print("=" * 80)

# Timedelta creation
Timedelta1 = pd.Timedelta(days=5)
Timedelta2 = pd.Timedelta(hours=12)
Timedelta3 = pd.Timedelta('2 days 3 hours 30 minutes')

print("\nTimedelta examples:")
print(f"5 days: {Timedelta1}")
print(f"12 hours: {Timedelta2}")
print(f"2 days 3 hours 30 minutes: {Timedelta3}")

# Timedelta arithmetic
BaseDate = pd.Timestamp('2024-01-01')
FutureDate = BaseDate + Timedelta1
PastDate = BaseDate - Timedelta2

print("\nTimedelta arithmetic:")
print(f"Base date: {BaseDate}")
print(f"Base + 5 days: {FutureDate}")
print(f"Base - 12 hours: {PastDate}")

# ============================================================================
# RESAMPLING
# ============================================================================
print("\n" + "=" * 80)
print("RESAMPLING")
print("=" * 80)

# Create daily data
DailyData = pd.DataFrame({
    'Sales': np.random.randint(100, 1000, 30),
    'Customers': np.random.randint(10, 100, 30)
}, index=pd.date_range('2024-01-01', periods=30, freq='D'))

print("\nDaily data (first 5 rows):")
print(DailyData.head())

# Downsample to monthly (sum)
MonthlySum = DailyData.resample('M').sum()
print("\nDownsampled to monthly (sum):")
print(MonthlySum)

# Downsample to monthly (mean)
MonthlyMean = DailyData.resample('M').mean()
print("\nDownsampled to monthly (mean):")
print(MonthlyMean)

# Upsample to hourly (forward fill)
HourlyData = DailyData.resample('H').ffill()
print("\nUpsampled to hourly (ffill) - first 10 rows:")
print(HourlyData.head(10))

# ============================================================================
# ROLLING
# ============================================================================
print("\n" + "=" * 80)
print("ROLLING")
print("=" * 80)

RollingSeries = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Rolling mean
RollingMean = RollingSeries.rolling(window=3).mean()
print("\nRolling mean (window=3):")
print(RollingMean)

# Rolling sum
RollingSum = RollingSeries.rolling(window=3).sum()
print("\nRolling sum (window=3):")
print(RollingSum)

# Rolling standard deviation
RollingStd = RollingSeries.rolling(window=3).std()
print("\nRolling std (window=3):")
print(RollingStd)

# Rolling with custom function
RollingCustom = RollingSeries.rolling(window=3).apply(lambda x: x.max() - x.min())
print("\nRolling range (max - min):")
print(RollingCustom)

# ============================================================================
# EXPANDING
# ============================================================================
print("\n" + "=" * 80)
print("EXPANDING")
print("=" * 80)

ExpandingSeries = pd.Series([10, 20, 30, 40, 50])

# Expanding mean
ExpandingMean = ExpandingSeries.expanding().mean()
print("\nExpanding mean:")
print(ExpandingMean)

# Expanding sum
ExpandingSum = ExpandingSeries.expanding().sum()
print("\nExpanding sum:")
print(ExpandingSum)

# ============================================================================
# EWM (EXPONENTIALLY WEIGHTED MOVING)
# ============================================================================
print("\n" + "=" * 80)
print("EWM (EXPONENTIALLY WEIGHTED MOVING)")
print("=" * 80)

EwmSeries = pd.Series([10, 20, 30, 40, 50, 60])

# EWM mean
EwmMean = EwmSeries.ewm(span=3).mean()
print("\nEWM mean (span=3):")
print(EwmMean)

# ============================================================================
# SHIFT, DIFF, PCT_CHANGE
# ============================================================================
print("\n" + "=" * 80)
print("SHIFT, DIFF, PCT_CHANGE")
print("=" * 80)

ShiftSeries = pd.Series([100, 110, 120, 130, 140])

# Shift
ShiftedForward = ShiftSeries.shift(1)
ShiftedBackward = ShiftSeries.shift(-1)
print("\nShift:")
print(f"Original: {ShiftSeries.values}")
print(f"Shift forward (1): {ShiftedForward.values}")
print(f"Shift backward (-1): {ShiftedBackward.values}")

# Diff
Differenced = ShiftSeries.diff()
print("\nDiff:")
print(Differenced)

# Percent change
PercentChange = ShiftSeries.pct_change()
print("\nPercent change:")
print(PercentChange)

# ============================================================================
# CSV I/O
# ============================================================================
print("\n" + "=" * 80)
print("CSV I/O")
print("=" * 80)

# Create sample DataFrame for CSV operations
CsvDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
})

# Write to CSV
CsvDataFrame.to_csv('sample_data.csv', index=False)
print("\nDataFrame written to CSV")

# Read CSV with various parameters
ReadCsvBasic = pd.read_csv('sample_data.csv')
print("\nRead CSV (basic):")
print(ReadCsvBasic)

# Read CSV with custom separator (if needed)
# ReadCsvSep = pd.read_csv('sample_data.csv', sep=',')

# Read CSV with header specification
# ReadCsvHeader = pd.read_csv('sample_data.csv', header=0)

# Read CSV with index column
CsvDataFrame.to_csv('sample_data_index.csv', index=True)
ReadCsvIndex = pd.read_csv('sample_data_index.csv', index_col=0)
print("\nRead CSV with index_col:")
print(ReadCsvIndex)

# Read CSV with selected columns
ReadCsvUsecols = pd.read_csv('sample_data.csv', usecols=['Name', 'Age'])
print("\nRead CSV with usecols:")
print(ReadCsvUsecols)

# Read CSV with dtype specification
ReadCsvDtype = pd.read_csv('sample_data.csv', dtype={'Age': 'int64', 'Salary': 'float64'})
print("\nRead CSV with dtype:")
print(ReadCsvDtype.dtypes)

# Read CSV with parse_dates
DateCsvDataFrame = pd.DataFrame({
    'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'Value': [100, 200, 300]
})
DateCsvDataFrame.to_csv('sample_dates.csv', index=False)
ReadCsvDates = pd.read_csv('sample_dates.csv', parse_dates=['Date'])
print("\nRead CSV with parse_dates:")
print(ReadCsvDates)
print(f"Date dtype: {ReadCsvDates['Date'].dtype}")

# Read CSV with na_values
NaCsvDataFrame = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': ['x', 'N/A', 'y', 'z']
})
NaCsvDataFrame.to_csv('sample_na.csv', index=False)
ReadCsvNa = pd.read_csv('sample_na.csv', na_values=['N/A'])
print("\nRead CSV with na_values:")
print(ReadCsvNa)

# Read CSV with nrows
ReadCsvNrows = pd.read_csv('sample_data.csv', nrows=2)
print("\nRead CSV with nrows=2:")
print(ReadCsvNrows)

# Read CSV with skiprows
ReadCsvSkiprows = pd.read_csv('sample_data.csv', skiprows=1)
print("\nRead CSV with skiprows=1:")
print(ReadCsvSkiprows)

# ============================================================================
# EXCEL I/O
# ============================================================================
print("\n" + "=" * 80)
print("EXCEL I/O")
print("=" * 80)

ExcelDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Value': [100, 200, 300]
})

# Write to Excel
ExcelDataFrame.to_excel('sample_data.xlsx', index=False, sheet_name='Sheet1')
print("\nDataFrame written to Excel")

# Read Excel
ReadExcel = pd.read_excel('sample_data.xlsx', sheet_name='Sheet1')
print("\nRead Excel:")
print(ReadExcel)

# Write multiple sheets using ExcelWriter
ExcelDataFrame2 = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Sales': [1000, 2000, 3000]
})

with pd.ExcelWriter('sample_multiple.xlsx', engine='openpyxl') as Writer:
    ExcelDataFrame.to_excel(Writer, sheet_name='Employees', index=False)
    ExcelDataFrame2.to_excel(Writer, sheet_name='Products', index=False)

print("\nMultiple sheets written to Excel")

# Read specific sheet
ReadExcelSheet = pd.read_excel('sample_multiple.xlsx', sheet_name='Products')
print("\nRead specific sheet:")
print(ReadExcelSheet)

# ============================================================================
# JSON I/O
# ============================================================================
print("\n" + "=" * 80)
print("JSON I/O")
print("=" * 80)

JsonDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
})

# Write to JSON
JsonDataFrame.to_json('sample_data.json', orient='records')
print("\nDataFrame written to JSON")

# Read JSON
ReadJson = pd.read_json('sample_data.json', orient='records')
print("\nRead JSON:")
print(ReadJson)

# ============================================================================
# PARQUET I/O
# ============================================================================
print("\n" + "=" * 80)
print("PARQUET I/O")
print("=" * 80)

ParquetDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Value': [100, 200, 300]
})

# Write to Parquet
ParquetDataFrame.to_parquet('sample_data.parquet', compression='snappy')
print("\nDataFrame written to Parquet")

# Read Parquet
ReadParquet = pd.read_parquet('sample_data.parquet')
print("\nRead Parquet:")
print(ReadParquet)

# ============================================================================
# OTHER I/O METHODS
# ============================================================================
print("\n" + "=" * 80)
print("OTHER I/O METHODS")
print("=" * 80)

# to_html
HtmlDataFrame = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
HtmlString = HtmlDataFrame.to_html()
print("\nDataFrame to HTML (first 200 chars):")
print(HtmlString[:200])

# to_pickle / read_pickle
PickleDataFrame = pd.DataFrame({'X': [10, 20], 'Y': [30, 40]})
PickleDataFrame.to_pickle('sample_data.pkl')
ReadPickle = pd.read_pickle('sample_data.pkl')
print("\nRead from pickle:")
print(ReadPickle)

# ============================================================================
# CHUNKED READING
# ============================================================================
print("\n" + "=" * 80)
print("CHUNKED READING")
print("=" * 80)

# Create larger CSV for chunking demonstration
LargeCsvDataFrame = pd.DataFrame({
    'ID': range(100),
    'Value': np.random.randn(100)
})
LargeCsvDataFrame.to_csv('large_sample.csv', index=False)

# Read in chunks
ChunkSize = 20
ChunkCount = 0
for Chunk in pd.read_csv('large_sample.csv', chunksize=ChunkSize):
    ChunkCount += 1
    if ChunkCount == 1:
        print(f"\nFirst chunk (size={ChunkSize}):")
        print(Chunk.head())

print(f"\nTotal chunks processed: {ChunkCount}")

# ============================================================================
# PLOTTING
# ============================================================================
print("\n" + "=" * 80)
print("PLOTTING")
print("=" * 80)

# Create sample data for plotting
PlotDataFrame = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
    'Sales': np.random.randint(100, 1000, 30),
    'Profit': np.random.randint(10, 100, 30),
    'Category': np.random.choice(['A', 'B', 'C'], 30)
})

# Line plot
LinePlot = PlotDataFrame.plot(x='Date', y='Sales', kind='line', figsize=(8, 4), title='Sales Over Time')
plt.close()

# Bar plot
BarData = PlotDataFrame.groupby('Category')['Sales'].sum()
BarPlot = BarData.plot(kind='bar', figsize=(8, 4), title='Sales by Category')
plt.close()

# Histogram
HistPlot = PlotDataFrame['Sales'].plot(kind='hist', bins=10, figsize=(8, 4), title='Sales Distribution')
plt.close()

# Box plot
BoxPlot = PlotDataFrame[['Sales', 'Profit']].plot(kind='box', figsize=(8, 4), title='Sales and Profit Distribution')
plt.close()

# Scatter plot
ScatterPlot = PlotDataFrame.plot(x='Sales', y='Profit', kind='scatter', figsize=(8, 4), title='Sales vs Profit')
plt.close()

# Area plot
AreaData = PlotDataFrame.set_index('Date')[['Sales', 'Profit']]
AreaPlot = AreaData.plot(kind='area', figsize=(8, 4), title='Sales and Profit Over Time')
plt.close()

print("\nPlotting examples completed (plots closed to avoid display issues)")

print("\n" + "=" * 80)
print("END OF TIME SERIES, I/O, AND VISUALIZATION DEMONSTRATION")
print("=" * 80)
