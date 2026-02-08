"""
Pandas Indexing, Selection, Filtering, and Manipulation
Comprehensive examples covering data access, filtering, and manipulation techniques
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("PANDAS INDEXING, SELECTION, FILTERING, AND MANIPULATION")
print("=" * 80)

# ============================================================================
# LOC: LABEL-BASED INDEXING
# ============================================================================
print("\n" + "=" * 80)
print("LOC: LABEL-BASED INDEXING")
print("=" * 80)

LocDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Salary': [50000, 60000, 70000, 55000, 65000],
    'City': ['NYC', 'London', 'Tokyo', 'Paris', 'Berlin']
}, index=['A', 'B', 'C', 'D', 'E'])

print("\nSample DataFrame:")
print(LocDataFrame)

# Single row
print("\nSingle row (loc['A']):")
print(LocDataFrame.loc['A'])

# Multiple rows
print("\nMultiple rows (loc[['A', 'C', 'E']]):")
print(LocDataFrame.loc[['A', 'C', 'E']])

# Row and column
print("\nRow and column (loc['A', 'Name']):")
print(LocDataFrame.loc['A', 'Name'])

# Multiple rows and columns
print("\nMultiple rows and columns:")
print(LocDataFrame.loc[['A', 'B'], ['Name', 'Age']])

# Slicing
print("\nSlicing rows (loc['A':'C']):")
print(LocDataFrame.loc['A':'C'])

# Boolean mask
BooleanMask = LocDataFrame['Age'] > 30
print("\nBoolean mask (Age > 30):")
print(LocDataFrame.loc[BooleanMask])

# Setting values
LocDataFrame.loc['A', 'Age'] = 26
print("\nAfter setting value (loc['A', 'Age'] = 26):")
print(LocDataFrame.loc['A'])

LocDataFrame.loc[LocDataFrame['Age'] > 30, 'Salary'] = 80000
print("\nAfter setting Salary for Age > 30:")
print(LocDataFrame)

# ============================================================================
# ILOC: INTEGER POSITION-BASED INDEXING
# ============================================================================
print("\n" + "=" * 80)
print("ILOC: INTEGER POSITION-BASED INDEXING")
print("=" * 80)

IlocDataFrame = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

print("\nSample DataFrame:")
print(IlocDataFrame)

# Single row
print("\nSingle row (iloc[0]):")
print(IlocDataFrame.iloc[0])

# Multiple rows
print("\nMultiple rows (iloc[[0, 2, 4]]):")
print(IlocDataFrame.iloc[[0, 2, 4]])

# Slicing
print("\nSlicing rows (iloc[1:4]):")
print(IlocDataFrame.iloc[1:4])

# Row and column
print("\nRow and column (iloc[0, 1]):")
print(IlocDataFrame.iloc[0, 1])

# Multiple rows and columns
print("\nMultiple rows and columns (iloc[0:3, 0:2]):")
print(IlocDataFrame.iloc[0:3, 0:2])

# Setting values
IlocDataFrame.iloc[0, 0] = 999
print("\nAfter setting value (iloc[0, 0] = 999):")
print(IlocDataFrame)

# ============================================================================
# AT/IAT: SCALAR ACCESS
# ============================================================================
print("\n" + "=" * 80)
print("AT/IAT: SCALAR ACCESS")
print("=" * 80)

AtDataFrame = pd.DataFrame({
    'X': [10, 20, 30],
    'Y': [40, 50, 60]
}, index=['a', 'b', 'c'])

print("\nSample DataFrame:")
print(AtDataFrame)

# Using at (label-based)
ValueAt = AtDataFrame.at['a', 'X']
print(f"\nValue at ['a', 'X']: {ValueAt}")

# Using iat (position-based)
ValueIat = AtDataFrame.iat[0, 1]
print(f"Value at [0, 1]: {ValueIat}")

# Setting values
AtDataFrame.at['a', 'X'] = 100
AtDataFrame.iat[1, 1] = 200
print("\nAfter setting values:")
print(AtDataFrame)

print("\nNote: at/iat are faster than loc/iloc for scalar access")

# ============================================================================
# BOOLEAN INDEXING
# ============================================================================
print("\n" + "=" * 80)
print("BOOLEAN INDEXING")
print("=" * 80)

BooleanDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Salary': [50000, 60000, 70000, 55000, 65000],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'IT']
})

print("\nSample DataFrame:")
print(BooleanDataFrame)

# Single condition
AgeFilter = BooleanDataFrame[BooleanDataFrame['Age'] > 30]
print("\nSingle condition (Age > 30):")
print(AgeFilter)

# Multiple conditions with &
MultipleConditionsAnd = BooleanDataFrame[(BooleanDataFrame['Age'] > 28) & (BooleanDataFrame['Salary'] > 55000)]
print("\nMultiple conditions with & (Age > 28 AND Salary > 55000):")
print(MultipleConditionsAnd)

# Multiple conditions with |
MultipleConditionsOr = BooleanDataFrame[(BooleanDataFrame['Age'] < 30) | (BooleanDataFrame['Department'] == 'IT')]
print("\nMultiple conditions with | (Age < 30 OR Department == 'IT'):")
print(MultipleConditionsOr)

# Using ~ for negation
NegationFilter = BooleanDataFrame[~(BooleanDataFrame['Department'] == 'IT')]
print("\nNegation with ~ (Department != 'IT'):")
print(NegationFilter)

# Using isin
IsinFilter = BooleanDataFrame[BooleanDataFrame['Department'].isin(['IT', 'HR'])]
print("\nUsing isin (Department in ['IT', 'HR']):")
print(IsinFilter)

# Using between
BetweenFilter = BooleanDataFrame[BooleanDataFrame['Age'].between(28, 32, inclusive='both')]
print("\nUsing between (Age between 28 and 32):")
print(BetweenFilter)

# ============================================================================
# STR ACCESSOR FILTERING
# ============================================================================
print("\n" + "=" * 80)
print("STR ACCESSOR FILTERING")
print("=" * 80)

StringDataFrame = pd.DataFrame({
    'Name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown', 'Diana Williams'],
    'Email': ['alice@example.com', 'bob@test.com', 'charlie@example.com', 'diana@test.com']
})

print("\nSample DataFrame:")
print(StringDataFrame)

# Contains
ContainsFilter = StringDataFrame[StringDataFrame['Name'].str.contains('Smith')]
print("\nContains 'Smith':")
print(ContainsFilter)

# Startswith
StartsWithFilter = StringDataFrame[StringDataFrame['Name'].str.startswith('B')]
print("\nStarts with 'B':")
print(StartsWithFilter)

# Endswith
EndsWithFilter = StringDataFrame[StringDataFrame['Email'].str.endswith('example.com')]
print("\nEnds with 'example.com':")
print(EndsWithFilter)

# ============================================================================
# QUERY() METHOD
# ============================================================================
print("\n" + "=" * 80)
print("QUERY() METHOD")
print("=" * 80)

QueryDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000]
})

print("\nSample DataFrame:")
print(QueryDataFrame)

# Basic query
BasicQuery = QueryDataFrame.query('Age > 30')
print("\nBasic query (Age > 30):")
print(BasicQuery)

# Query with multiple conditions
MultipleQuery = QueryDataFrame.query('Age > 28 and Salary > 55000')
print("\nMultiple conditions (Age > 28 and Salary > 55000):")
print(MultipleQuery)

# Query with @ variables
ThresholdAge = 30
ThresholdSalary = 60000
VariableQuery = QueryDataFrame.query('Age > @ThresholdAge and Salary > @ThresholdSalary')
print("\nQuery with @ variables:")
print(VariableQuery)

# ============================================================================
# MULTIINDEX
# ============================================================================
print("\n" + "=" * 80)
print("MULTIINDEX")
print("=" * 80)

MultiIndexDataFrame = pd.DataFrame({
    'Value': [100, 200, 300, 400, 500, 600],
    'Count': [10, 20, 30, 40, 50, 60]
}, index=pd.MultiIndex.from_tuples([
    ('GroupA', 'X'), ('GroupA', 'Y'), ('GroupA', 'Z'),
    ('GroupB', 'X'), ('GroupB', 'Y'), ('GroupB', 'Z')
], names=['Group', 'SubGroup']))

print("\nMultiIndex DataFrame:")
print(MultiIndexDataFrame)

# set_index
SimpleDataFrame = pd.DataFrame({
    'Group': ['A', 'A', 'B', 'B'],
    'SubGroup': ['X', 'Y', 'X', 'Y'],
    'Value': [100, 200, 300, 400]
})
MultiIndexFromSet = SimpleDataFrame.set_index(['Group', 'SubGroup'])
print("\nAfter set_index:")
print(MultiIndexFromSet)

# reset_index
ResetIndexResult = MultiIndexFromSet.reset_index()
print("\nAfter reset_index:")
print(ResetIndexResult)

# xs (cross-section)
XsResult = MultiIndexDataFrame.xs('GroupA', level='Group')
print("\nCross-section (xs) for GroupA:")
print(XsResult)

# IndexSlice
IndexSliceResult = MultiIndexDataFrame.loc[pd.IndexSlice['GroupA', 'X':'Y'], :]
print("\nUsing IndexSlice:")
print(IndexSliceResult)

# swaplevel
SwapLevelResult = MultiIndexDataFrame.swaplevel('Group', 'SubGroup')
print("\nAfter swaplevel:")
print(SwapLevelResult)

# ============================================================================
# COLUMN MANIPULATION
# ============================================================================
print("\n" + "=" * 80)
print("COLUMN MANIPULATION")
print("=" * 80)

ColumnDataFrame = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9],
    'D': [10, 11, 12]
})

print("\nOriginal DataFrame:")
print(ColumnDataFrame)

# Adding column
ColumnDataFrame['E'] = [13, 14, 15]
print("\nAfter adding column 'E':")
print(ColumnDataFrame)

# Dropping column
DroppedColumn = ColumnDataFrame.drop('C', axis=1)
print("\nAfter dropping column 'C':")
print(DroppedColumn)

# Popping column
PoppedColumn = ColumnDataFrame.pop('D')
print("\nPopped column 'D':")
print(PoppedColumn)
print("\nDataFrame after pop:")
print(ColumnDataFrame)

# Renaming columns
RenamedDataFrame = ColumnDataFrame.rename(columns={'A': 'Alpha', 'B': 'Beta'})
print("\nAfter renaming columns:")
print(RenamedDataFrame)

# Reindexing columns
ReindexedColumns = ColumnDataFrame.reindex(columns=['B', 'A', 'E', 'C'])
print("\nAfter reindexing columns:")
print(ReindexedColumns)

# ============================================================================
# SORTING
# ============================================================================
print("\n" + "=" * 80)
print("SORTING")
print("=" * 80)

SortDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [35, 25, 30, 28],
    'Salary': [70000, 50000, 60000, 55000]
})

print("\nOriginal DataFrame:")
print(SortDataFrame)

# sort_values by single column
SortedByAge = SortDataFrame.sort_values('Age')
print("\nSorted by Age (ascending):")
print(SortedByAge)

# sort_values with ascending=False
SortedDescending = SortDataFrame.sort_values('Age', ascending=False)
print("\nSorted by Age (descending):")
print(SortedDescending)

# sort_values by multiple columns
SortedMultiple = SortDataFrame.sort_values(['Age', 'Salary'], ascending=[True, False])
print("\nSorted by Age (asc) and Salary (desc):")
print(SortedMultiple)

# sort_values with na_position
SortWithNaN = pd.DataFrame({'A': [1, 2, None, 4, None, 3]})
SortedWithNaN = SortWithNaN.sort_values('A', na_position='last')
print("\nSorted with NaN last:")
print(SortedWithNaN)

# sort_index
SortIndexDataFrame = pd.DataFrame({'Value': [10, 20, 30]}, index=['C', 'A', 'B'])
SortedIndex = SortIndexDataFrame.sort_index()
print("\nSorted by index:")
print(SortedIndex)

# nsmallest and nlargest
NsmallestResult = SortDataFrame.nsmallest(2, 'Age')
print("\n2 smallest by Age:")
print(NsmallestResult)

NlargestResult = SortDataFrame.nlargest(2, 'Salary')
print("\n2 largest by Salary:")
print(NlargestResult)

# ============================================================================
# RANKING
# ============================================================================
print("\n" + "=" * 80)
print("RANKING")
print("=" * 80)

RankDataFrame = pd.DataFrame({
    'Value': [100, 200, 150, 200, 180]
})

print("\nOriginal DataFrame:")
print(RankDataFrame)

# Default ranking (average for ties)
RankDefault = RankDataFrame['Value'].rank()
print("\nDefault ranking:")
print(RankDefault)

# Method='min'
RankMin = RankDataFrame['Value'].rank(method='min')
print("\nRanking with method='min':")
print(RankMin)

# Method='max'
RankMax = RankDataFrame['Value'].rank(method='max')
print("\nRanking with method='max':")
print(RankMax)

# Method='dense'
RankDense = RankDataFrame['Value'].rank(method='dense')
print("\nRanking with method='dense':")
print(RankDense)

# Ascending=False
RankDescending = RankDataFrame['Value'].rank(ascending=False)
print("\nRanking descending:")
print(RankDescending)

# ============================================================================
# DUPLICATES
# ============================================================================
print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)

DuplicateDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Age': [25, 30, 25, 35, 30],
    'City': ['NYC', 'London', 'NYC', 'Tokyo', 'London']
})

print("\nOriginal DataFrame:")
print(DuplicateDataFrame)

# Check for duplicates
IsDuplicate = DuplicateDataFrame.duplicated()
print("\nDuplicated rows:")
print(IsDuplicate)

# Drop duplicates (keep first)
DroppedDuplicatesFirst = DuplicateDataFrame.drop_duplicates(keep='first')
print("\nAfter drop_duplicates(keep='first'):")
print(DroppedDuplicatesFirst)

# Drop duplicates (keep last)
DroppedDuplicatesLast = DuplicateDataFrame.drop_duplicates(keep='last')
print("\nAfter drop_duplicates(keep='last'):")
print(DroppedDuplicatesLast)

# Drop duplicates with subset
DroppedDuplicatesSubset = DuplicateDataFrame.drop_duplicates(subset=['Name', 'Age'], keep='first')
print("\nAfter drop_duplicates(subset=['Name', 'Age']):")
print(DroppedDuplicatesSubset)

# ============================================================================
# ITERATION
# ============================================================================
print("\n" + "=" * 80)
print("ITERATION")
print("=" * 80)

IterateDataFrame = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
})

print("\nSample DataFrame:")
print(IterateDataFrame)

# iterrows (returns index and Series)
print("\nUsing iterrows():")
for Index, Row in IterateDataFrame.iterrows():
    print(f"Index: {Index}, Name: {Row['Name']}, Age: {Row['Age']}")

# itertuples (faster than iterrows)
print("\nUsing itertuples():")
for Row in IterateDataFrame.itertuples():
    print(f"Index: {Row.Index}, Name: {Row.Name}, Age: {Row.Age}")

# items (iterates over columns)
print("\nUsing items() (column iteration):")
for ColumnName, ColumnSeries in IterateDataFrame.items():
    print(f"Column: {ColumnName}")
    print(ColumnSeries)
    print()

print("WARNING: iterrows() and itertuples() are slow for large DataFrames.")
print("Prefer vectorized operations or apply() when possible.")

print("\n" + "=" * 80)
print("END OF INDEXING AND MANIPULATION DEMONSTRATION")
print("=" * 80)
