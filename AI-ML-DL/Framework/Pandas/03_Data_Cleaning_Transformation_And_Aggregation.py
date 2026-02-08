"""
Pandas Data Cleaning, Transformation, and Aggregation
Comprehensive examples covering data cleaning, transformations, and aggregations
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("PANDAS DATA CLEANING, TRANSFORMATION, AND AGGREGATION")
print("=" * 80)

# ============================================================================
# MISSING DATA
# ============================================================================
print("\n" + "=" * 80)
print("MISSING DATA")
print("=" * 80)

# Creating data with NaN
MissingDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', None, 'Diana', 'Eve'],
    'Age': [25, None, 35, 28, None],
    'Salary': [50000, 60000, None, 55000, 65000],
    'City': ['NYC', 'London', 'Tokyo', None, 'Berlin']
})

print("\nDataFrame with missing values:")
print(MissingDataFrame)

# isna and notna
IsNaResult = MissingDataFrame.isna()
print("\nisna() result:")
print(IsNaResult)

NotNaResult = MissingDataFrame.notna()
print("\nnotna() result:")
print(NotNaResult)

# Count missing values per column
MissingCount = MissingDataFrame.isna().sum()
print("\nMissing values count per column:")
print(MissingCount)

# dropna - drop rows with any NaN
DroppedAny = MissingDataFrame.dropna()
print("\nAfter dropna() (drop rows with any NaN):")
print(DroppedAny)

# dropna - drop rows where all values are NaN
DroppedAll = MissingDataFrame.dropna(how='all')
print("\nAfter dropna(how='all'):")
print(DroppedAll)

# dropna with thresh
DroppedThresh = MissingDataFrame.dropna(thresh=3)
print("\nAfter dropna(thresh=3) (keep rows with at least 3 non-NaN):")
print(DroppedThresh)

# dropna with subset
DroppedSubset = MissingDataFrame.dropna(subset=['Name', 'Age'])
print("\nAfter dropna(subset=['Name', 'Age']):")
print(DroppedSubset)

# fillna with value
FilledValue = MissingDataFrame.fillna(0)
print("\nAfter fillna(0):")
print(FilledValue)

# fillna with method='ffill'
FilledForward = MissingDataFrame.fillna(method='ffill')
print("\nAfter fillna(method='ffill'):")
print(FilledForward)

# fillna with limit
FilledLimit = MissingDataFrame.fillna(method='ffill', limit=1)
print("\nAfter fillna(method='ffill', limit=1):")
print(FilledLimit)

# interpolate
NumericSeries = pd.Series([1, None, None, None, 5])
Interpolated = NumericSeries.interpolate()
print("\nInterpolation example:")
print(f"Original: {NumericSeries.values}")
print(f"Interpolated: {Interpolated.values}")

# ============================================================================
# STRING METHODS
# ============================================================================
print("\n" + "=" * 80)
print("STRING METHODS")
print("=" * 80)

StringDataFrame = pd.DataFrame({
    'Name': ['  ALICE  ', 'bob', 'CHARLIE', '  diana  '],
    'Email': ['ALICE@EXAMPLE.COM', 'bob@test.com', 'CHARLIE@EXAMPLE.COM', 'diana@test.com']
})

print("\nOriginal DataFrame:")
print(StringDataFrame)

# lower
LowerResult = StringDataFrame['Name'].str.lower()
print("\nLowercase:")
print(LowerResult)

# upper
UpperResult = StringDataFrame['Name'].str.upper()
print("\nUppercase:")
print(UpperResult)

# strip
StripResult = StringDataFrame['Name'].str.strip()
print("\nStripped:")
print(StripResult)

# contains
ContainsResult = StringDataFrame[StringDataFrame['Email'].str.contains('EXAMPLE')]
print("\nContains 'EXAMPLE':")
print(ContainsResult)

# replace
ReplaceResult = StringDataFrame['Email'].str.replace('EXAMPLE', 'example')
print("\nReplaced 'EXAMPLE' with 'example':")
print(ReplaceResult)

# extract
ExtractDataFrame = pd.DataFrame({'Text': ['A1', 'B2', 'C3', 'D4']})
Extracted = ExtractDataFrame['Text'].str.extract(r'([A-Z])(\d)')
print("\nExtracted pattern:")
print(Extracted)

# split
SplitResult = StringDataFrame['Email'].str.split('@', expand=True)
print("\nSplit by '@':")
print(SplitResult)

# len
LengthResult = StringDataFrame['Name'].str.len()
print("\nLength of strings:")
print(LengthResult)

# cat
CatResult = StringDataFrame['Name'].str.cat(StringDataFrame['Email'], sep=' - ')
print("\nConcatenated:")
print(CatResult)

# ============================================================================
# REPLACE
# ============================================================================
print("\n" + "=" * 80)
print("REPLACE")
print("=" * 80)

ReplaceDataFrame = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

print("\nOriginal DataFrame:")
print(ReplaceDataFrame)

# Scalar replace
ScalarReplace = ReplaceDataFrame.replace(2, 99)
print("\nReplace scalar (2 -> 99):")
print(ScalarReplace)

# List replace
ListReplace = ReplaceDataFrame.replace([1, 2, 3], [10, 20, 30])
print("\nReplace list:")
print(ListReplace)

# Dict replace
DictReplace = ReplaceDataFrame.replace({'A': {1: 100, 2: 200}, 'B': {'a': 'alpha'}})
print("\nReplace with dict:")
print(DictReplace)

# Regex replace
RegexReplace = ReplaceDataFrame.replace({'B': r'^[a-c]'}, {'B': 'X'}, regex=True)
print("\nRegex replace:")
print(RegexReplace)

# ============================================================================
# APPLY/MAP
# ============================================================================
print("\n" + "=" * 80)
print("APPLY/MAP")
print("=" * 80)

ApplyDataFrame = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

print("\nOriginal DataFrame:")
print(ApplyDataFrame)

# Series.map with dict
MapDict = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
MappedDict = ApplyDataFrame['A'].map(MapDict)
print("\nSeries.map with dict:")
print(MappedDict)

# Series.map with function
MappedFunction = ApplyDataFrame['A'].map(lambda x: x ** 2)
print("\nSeries.map with function:")
print(MappedFunction)

# Series.apply
AppliedSeries = ApplyDataFrame['A'].apply(lambda x: x * 10)
print("\nSeries.apply:")
print(AppliedSeries)

# DataFrame.apply axis=0 (column-wise)
AppliedColumnWise = ApplyDataFrame.apply(lambda x: x.sum(), axis=0)
print("\nDataFrame.apply axis=0 (column-wise):")
print(AppliedColumnWise)

# DataFrame.apply axis=1 (row-wise)
AppliedRowWise = ApplyDataFrame.apply(lambda x: x['A'] + x['B'], axis=1)
print("\nDataFrame.apply axis=1 (row-wise):")
print(AppliedRowWise)

# DataFrame.map
MapDataFrame = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
MappedDataFrame = MapDataFrame.map(str.upper)
print("\nDataFrame.map:")
print(MappedDataFrame)

# ============================================================================
# PIPE() FOR CHAINING
# ============================================================================
print("\n" + "=" * 80)
print("PIPE() FOR CHAINING")
print("=" * 80)

PipeDataFrame = pd.DataFrame({
    'Value': [1, 2, 3, 4, 5]
})

def AddColumn(DataFrame, ColumnName, Value):
    DataFrame[ColumnName] = Value
    return DataFrame

def MultiplyColumn(DataFrame, ColumnName, Factor):
    DataFrame[ColumnName] = DataFrame[ColumnName] * Factor
    return DataFrame

PipedResult = (PipeDataFrame
               .pipe(AddColumn, 'Double', 2)
               .pipe(MultiplyColumn, 'Value', 10))
print("\nUsing pipe() for chaining:")
print(PipedResult)

# ============================================================================
# GROUPBY
# ============================================================================
print("\n" + "=" * 80)
print("GROUPBY")
print("=" * 80)

GroupByDataFrame = pd.DataFrame({
    'Department': ['IT', 'HR', 'IT', 'HR', 'IT', 'Finance'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Salary': [70000, 50000, 80000, 55000, 75000, 60000],
    'Age': [30, 25, 35, 28, 32, 40]
})

print("\nOriginal DataFrame:")
print(GroupByDataFrame)

# Basic groupby
Grouped = GroupByDataFrame.groupby('Department')
print("\nGrouped by Department:")
for Name, Group in Grouped:
    print(f"\n{Name}:")
    print(Group)

# groups attribute
GroupsDict = Grouped.groups
print("\nGroups dictionary:")
print(GroupsDict)

# ngroups
NumberOfGroups = Grouped.ngroups
print(f"\nNumber of groups: {NumberOfGroups}")

# get_group
ItGroup = Grouped.get_group('IT')
print("\nIT group:")
print(ItGroup)

# Single aggregation
SingleAgg = Grouped['Salary'].agg('mean')
print("\nMean salary by department:")
print(SingleAgg)

# Multiple aggregations
MultipleAgg = Grouped['Salary'].agg(['mean', 'sum', 'count'])
print("\nMultiple aggregations:")
print(MultipleAgg)

# Named aggregations
NamedAgg = Grouped.agg(
    MeanSalary=('Salary', 'mean'),
    TotalSalary=('Salary', 'sum'),
    AvgAge=('Age', 'mean')
)
print("\nNamed aggregations:")
print(NamedAgg)

# transform
Transformed = Grouped['Salary'].transform(lambda x: x - x.mean())
print("\nTransform (salary - group mean):")
print(Transformed)

# filter
FilteredGroups = Grouped.filter(lambda x: x['Salary'].mean() > 60000)
print("\nFiltered groups (mean salary > 60000):")
print(FilteredGroups)

# ============================================================================
# GROUP-WISE OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GROUP-WISE OPERATIONS")
print("=" * 80)

GroupWiseDataFrame = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Value': [10, 20, 30, 40, 50, 60]
})

# Normalization within groups
GroupWiseDataFrame['Normalized'] = GroupWiseDataFrame.groupby('Category')['Value'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
print("\nNormalization within groups:")
print(GroupWiseDataFrame)

# Cumulative within groups
GroupWiseDataFrame['Cumulative'] = GroupWiseDataFrame.groupby('Category')['Value'].cumsum()
print("\nCumulative sum within groups:")
print(GroupWiseDataFrame)

# Rank within groups
GroupWiseDataFrame['Rank'] = GroupWiseDataFrame.groupby('Category')['Value'].rank()
print("\nRank within groups:")
print(GroupWiseDataFrame)

# ============================================================================
# PIVOT_TABLE
# ============================================================================
print("\n" + "=" * 80)
print("PIVOT_TABLE")
print("=" * 80)

PivotDataFrame = pd.DataFrame({
    'Date': ['2024-01', '2024-01', '2024-02', '2024-02', '2024-01', '2024-02'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 200, 150, 250, 120, 220],
    'Region': ['North', 'North', 'South', 'South', 'North', 'South']
})

print("\nOriginal DataFrame:")
print(PivotDataFrame)

# Basic pivot_table
BasicPivot = pd.pivot_table(PivotDataFrame, values='Sales', index='Date', columns='Product', aggfunc='sum')
print("\nBasic pivot_table:")
print(BasicPivot)

# Pivot_table with margins
PivotWithMargins = pd.pivot_table(PivotDataFrame, values='Sales', index='Date', columns='Product', 
                                   aggfunc='sum', margins=True)
print("\nPivot_table with margins:")
print(PivotWithMargins)

# Pivot_table with fill_value
PivotWithFill = pd.pivot_table(PivotDataFrame, values='Sales', index='Date', columns='Product',
                                aggfunc='sum', fill_value=0)
print("\nPivot_table with fill_value:")
print(PivotWithFill)

# ============================================================================
# CROSSTAB
# ============================================================================
print("\n" + "=" * 80)
print("CROSSTAB")
print("=" * 80)

CrosstabDataFrame = pd.DataFrame({
    'Department': ['IT', 'HR', 'IT', 'HR', 'IT'],
    'Status': ['Active', 'Active', 'Inactive', 'Active', 'Active']
})

# Basic crosstab
BasicCrosstab = pd.crosstab(CrosstabDataFrame['Department'], CrosstabDataFrame['Status'])
print("\nBasic crosstab:")
print(BasicCrosstab)

# Crosstab with normalize
NormalizedCrosstab = pd.crosstab(CrosstabDataFrame['Department'], CrosstabDataFrame['Status'], 
                                  normalize='index')
print("\nCrosstab normalized by index:")
print(NormalizedCrosstab)

# ============================================================================
# MELT
# ============================================================================
print("\n" + "=" * 80)
print("MELT")
print("=" * 80)

WideDataFrame = pd.DataFrame({
    'ID': [1, 2, 3],
    'Q1': [100, 200, 300],
    'Q2': [150, 250, 350],
    'Q3': [120, 220, 320],
    'Q4': [180, 280, 380]
})

print("\nWide DataFrame:")
print(WideDataFrame)

# Melt (wide to long)
Melted = WideDataFrame.melt(id_vars='ID', value_vars=['Q1', 'Q2', 'Q3', 'Q4'],
                            var_name='Quarter', value_name='Sales')
print("\nMelted (long format):")
print(Melted)

# ============================================================================
# STACK/UNSTACK
# ============================================================================
print("\n" + "=" * 80)
print("STACK/UNSTACK")
print("=" * 80)

StackDataFrame = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['X', 'Y', 'Z'])

print("\nOriginal DataFrame:")
print(StackDataFrame)

# Stack
Stacked = StackDataFrame.stack()
print("\nStacked:")
print(Stacked)

# Unstack
Unstacked = Stacked.unstack()
print("\nUnstacked:")
print(Unstacked)

# ============================================================================
# EXPLODE
# ============================================================================
print("\n" + "=" * 80)
print("EXPLODE")
print("=" * 80)

ExplodeDataFrame = pd.DataFrame({
    'ID': [1, 2, 3],
    'Values': [[10, 20], [30, 40, 50], [60]]
})

print("\nOriginal DataFrame:")
print(ExplodeDataFrame)

# Explode
Exploded = ExplodeDataFrame.explode('Values')
print("\nExploded:")
print(Exploded)

# ============================================================================
# MERGE
# ============================================================================
print("\n" + "=" * 80)
print("MERGE")
print("=" * 80)

LeftDataFrame = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Department': ['IT', 'HR', 'IT', 'Finance']
})

RightDataFrame = pd.DataFrame({
    'ID': [2, 3, 4, 5],
    'Salary': [60000, 70000, 55000, 80000],
    'Age': [30, 35, 28, 40]
})

print("\nLeft DataFrame:")
print(LeftDataFrame)
print("\nRight DataFrame:")
print(RightDataFrame)

# Inner merge
InnerMerge = pd.merge(LeftDataFrame, RightDataFrame, on='ID', how='inner')
print("\nInner merge:")
print(InnerMerge)

# Left merge
LeftMerge = pd.merge(LeftDataFrame, RightDataFrame, on='ID', how='left')
print("\nLeft merge:")
print(LeftMerge)

# Right merge
RightMerge = pd.merge(LeftDataFrame, RightDataFrame, on='ID', how='right')
print("\nRight merge:")
print(RightMerge)

# Outer merge
OuterMerge = pd.merge(LeftDataFrame, RightDataFrame, on='ID', how='outer')
print("\nOuter merge:")
print(OuterMerge)

# Merge with left_on and right_on
LeftOnRightOn = pd.DataFrame({'EmpID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
RightOnRightOn = pd.DataFrame({'StaffID': [1, 2, 4], 'Salary': [50, 60, 70]})
MergedOnOn = pd.merge(LeftOnRightOn, RightOnRightOn, left_on='EmpID', right_on='StaffID')
print("\nMerge with left_on/right_on:")
print(MergedOnOn)

# Merge with suffixes
LeftSuffix = pd.DataFrame({'ID': [1, 2], 'Value': [10, 20]})
RightSuffix = pd.DataFrame({'ID': [1, 2], 'Value': [30, 40]})
MergedSuffix = pd.merge(LeftSuffix, RightSuffix, on='ID', suffixes=('_left', '_right'))
print("\nMerge with suffixes:")
print(MergedSuffix)

# Merge with indicator
MergedIndicator = pd.merge(LeftDataFrame, RightDataFrame, on='ID', how='outer', indicator=True)
print("\nMerge with indicator:")
print(MergedIndicator)

# ============================================================================
# JOIN
# ============================================================================
print("\n" + "=" * 80)
print("JOIN")
print("=" * 80)

JoinLeft = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]}, index=['x', 'y', 'z'])
JoinRight = pd.DataFrame({'C': [100, 200], 'D': [1000, 2000]}, index=['x', 'y'])

# Index-based join
Joined = JoinLeft.join(JoinRight)
print("\nIndex-based join:")
print(Joined)

# ============================================================================
# CONCAT
# ============================================================================
print("\n" + "=" * 80)
print("CONCAT")
print("=" * 80)

ConcatDataFrame1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
ConcatDataFrame2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Concat axis=0 (rows)
ConcatRows = pd.concat([ConcatDataFrame1, ConcatDataFrame2], axis=0)
print("\nConcat axis=0 (rows):")
print(ConcatRows)

# Concat axis=1 (columns)
ConcatColumns = pd.concat([ConcatDataFrame1, ConcatDataFrame2], axis=1)
print("\nConcat axis=1 (columns):")
print(ConcatColumns)

# Concat with keys
ConcatKeys = pd.concat([ConcatDataFrame1, ConcatDataFrame2], keys=['First', 'Second'])
print("\nConcat with keys:")
print(ConcatKeys)

# Concat with ignore_index
ConcatIgnoreIndex = pd.concat([ConcatDataFrame1, ConcatDataFrame2], ignore_index=True)
print("\nConcat with ignore_index:")
print(ConcatIgnoreIndex)

# ============================================================================
# MERGE_ASOF
# ============================================================================
print("\n" + "=" * 80)
print("MERGE_ASOF")
print("=" * 80)

LeftAsof = pd.DataFrame({
    'Time': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:05', '2024-01-01 10:10']),
    'Value': [100, 200, 300]
})

RightAsof = pd.DataFrame({
    'Time': pd.to_datetime(['2024-01-01 10:02', '2024-01-01 10:07']),
    'Price': [50, 60]
})

MergedAsof = pd.merge_asof(LeftAsof, RightAsof, on='Time', direction='backward')
print("\nmerge_asof (backward):")
print(MergedAsof)

# ============================================================================
# CUT/QCUT
# ============================================================================
print("\n" + "=" * 80)
print("CUT/QCUT")
print("=" * 80)

CutSeries = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# cut (equal-width bins)
CutResult = pd.cut(CutSeries, bins=3, labels=['Low', 'Medium', 'High'])
print("\ncut (equal-width bins):")
print(CutResult)

# qcut (equal-frequency bins)
QcutResult = pd.qcut(CutSeries, q=3, labels=['Low', 'Medium', 'High'])
print("\nqcut (equal-frequency bins):")
print(QcutResult)

# ============================================================================
# GET_DUMMIES
# ============================================================================
print("\n" + "=" * 80)
print("GET_DUMMIES")
print("=" * 80)

DummyDataFrame = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C', 'B']
})

# One-hot encoding
Dummies = pd.get_dummies(DummyDataFrame['Category'], prefix='Category')
print("\nOne-hot encoding:")
print(Dummies)

# ============================================================================
# CLIP
# ============================================================================
print("\n" + "=" * 80)
print("CLIP")
print("=" * 80)

ClipSeries = pd.Series([1, 5, 10, 15, 20, 25, 30])

# Clip values
Clipped = ClipSeries.clip(lower=5, upper=25)
print("\nClipped (between 5 and 25):")
print(Clipped)

print("\n" + "=" * 80)
print("END OF DATA CLEANING AND TRANSFORMATION DEMONSTRATION")
print("=" * 80)
