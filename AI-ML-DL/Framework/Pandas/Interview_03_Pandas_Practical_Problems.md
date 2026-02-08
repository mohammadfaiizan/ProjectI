# Pandas Practical Problems Interview Questions

## Q1: How do you find rows where a column has missing values?

**A1:** Use the isnull() or isna() method on the column and use it for boolean indexing. For example, df[df['column'].isnull()] returns all rows where 'column' has missing values. You can also use df[df['column'].isna()] as isna() and isnull() are aliases. To find rows with missing values in any column, use df[df.isnull().any(axis=1)]. To find rows with missing values in all columns, use df[df.isnull().all(axis=1)]. The isnull() method returns a boolean Series that can be used directly for filtering, making it the standard approach for identifying missing data. This is essential for data cleaning and understanding data quality.

## Q2: How do you fill missing values with group means?

**A2:** Use groupby with transform to calculate group means and fill missing values. For example, df['column'] = df.groupby('group_col')['column'].transform(lambda x: x.fillna(x.mean())) fills missing values in 'column' with the mean of each group defined by 'group_col'. Alternatively, you can use df['column'].fillna(df.groupby('group_col')['column'].transform('mean'), inplace=True). The transform method ensures the result has the same length as the original, broadcasting the group mean to each row. This approach is useful when missing values should be imputed based on group characteristics rather than global statistics, which is common in hierarchical data or when groups have different distributions.

## Q3: How do you compute a rolling 7-day average?

**A3:** Use the rolling() method with a window size of 7 and then apply mean(). For datetime-indexed DataFrames, use df['column'].rolling(window='7D').mean() to get a 7-day rolling average. For integer-based windows, use df['column'].rolling(window=7).mean(). The rolling window calculates the mean of the current row and the previous 6 rows. You can specify min_periods to handle the initial rows where there aren't enough previous values. For example, df['column'].rolling(window=7, min_periods=1).mean() will use available data even if there are fewer than 7 previous values. Rolling averages are essential for smoothing time series data and identifying trends.

## Q4: How do you create a new column based on multiple conditions?

**A4:** Use numpy.select() or nested np.where() statements for multiple conditions. For example, df['new_col'] = np.select([df['A'] > 5, df['B'] < 3], ['high', 'low'], default='medium') assigns values based on conditions. Alternatively, use np.where() nested: df['new_col'] = np.where(df['A'] > 5, 'high', np.where(df['B'] < 3, 'low', 'medium')). For boolean conditions, you can also use df.loc[df['A'] > 5, 'new_col'] = 'high' followed by additional assignments. The select approach is cleaner for multiple conditions as it's more readable and maintainable. This pattern is common in feature engineering where you need to categorize or transform data based on multiple criteria.

## Q5: How do you get the top 3 values per group?

**A5:** Use groupby with nlargest() or apply with head() after sorting. For example, df.groupby('group_col')['value_col'].nlargest(3) returns the top 3 values per group. Alternatively, df.groupby('group_col').apply(lambda x: x.nlargest(3, 'value_col')) gives you the top 3 rows per group based on a column. You can also use df.groupby('group_col')['value_col'].apply(lambda x: x.nlargest(3)) for just the values. The nlargest method is optimized for this use case and is more efficient than sorting the entire DataFrame. For bottom values, use nsmallest() instead. This is useful for finding top performers, outliers, or extreme values within categories.

## Q6: How do you convert a column of comma-separated values into separate rows?

**A6:** Use str.split() combined with explode(). First split the column on commas, then explode the resulting list. For example, df['column'].str.split(',', expand=False).explode() converts comma-separated values into separate rows. If you need to strip whitespace, use df['column'].str.split(',').str.strip().explode(). The explode() method creates a new row for each element in the list, duplicating other column values. This is essential for normalizing data where multiple values are stored in a single cell, such as tags, categories, or multi-select responses. After exploding, you may want to reset_index() if you need a clean sequential index.

## Q7: How do you compute percentage of total per group?

**A7:** Use groupby with transform to get group sums, then divide by the total sum. For example, df['pct'] = df.groupby('group_col')['value_col'].transform('sum') / df['value_col'].sum() * 100 calculates the percentage each group represents of the total. Alternatively, group_totals = df.groupby('group_col')['value_col'].sum() and then df['pct'] = df['group_col'].map(group_totals) / df['value_col'].sum() * 100. The transform method ensures the percentage is broadcast to each row in the group. This is useful for understanding group contributions to totals, market share analysis, or distribution analysis. Multiply by 100 to get percentages rather than proportions.

## Q8: How do you find the most frequent value in each column?

**A8:** Use the mode() method which returns the most frequent value(s) for each column. For example, df.mode().iloc[0] gives you the first mode value for each column (handling cases where there are multiple modes). To get a single value per column, use df.apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else None). Alternatively, df.apply(lambda x: x.value_counts().index[0]) gets the most frequent value, but this fails if there are no values. The mode() method handles edge cases better and returns a DataFrame, making it easier to work with. This is useful for understanding data distributions and finding common values, especially for categorical data or when exploring datasets.

## Q9: How do you merge two DataFrames keeping only new rows?

**A9:** Perform an outer merge with indicator, then filter for rows that only exist in the right DataFrame. For example, merged = df1.merge(df2, how='outer', indicator=True) and then new_rows = merged[merged['_merge'] == 'right_only'].drop('_merge', axis=1). Alternatively, use an anti-join pattern: new_rows = df2[~df2['key'].isin(df1['key'])]. The indicator parameter shows the source of each row, making it easy to identify new records. This is useful for incremental data updates, finding new records in a dataset, or synchronizing data sources. The anti-join approach is more efficient if you only need to check key columns, while the merge approach works when you need to compare all columns.

## Q10: How do you create a date range and count events per month?

**A10:** Create a date range using pd.date_range(), then use groupby with Grouper or resample to count events per month. For example, df['date'] = pd.to_datetime(df['date']) and then monthly_counts = df.groupby(pd.Grouper(key='date', freq='M')).size(). Alternatively, if 'date' is the index, use df.resample('M').size(). The Grouper allows you to specify the frequency ('M' for month-end, 'MS' for month-start) and works with a datetime column. Resample works when the datetime is the index. This pattern is essential for time series analysis, reporting, and understanding temporal patterns in data. The result is a time series with monthly counts that can be easily visualized or analyzed.

## Q11: How do you apply different aggregations to different columns?

**A11:** Pass a dictionary to the agg() method mapping column names to aggregation functions. For example, df.groupby('group_col').agg({'col1': 'sum', 'col2': 'mean', 'col3': ['min', 'max']}) applies sum to col1, mean to col2, and both min and max to col3. You can also use a list of functions for multiple aggregations on the same column. The result is a DataFrame with MultiIndex columns when you specify multiple functions. This is much more flexible than applying the same aggregation to all columns and is essential for creating summary statistics tables. The dictionary approach makes it clear which aggregation applies to which column, improving code readability and maintainability.

## Q12: How do you pivot data and fill NaN with 0?

**A12:** Use pivot_table() with fill_value parameter set to 0. For example, df.pivot_table(index='row_col', columns='col_col', values='value_col', fill_value=0) creates a pivot table and fills missing values with 0. Alternatively, use pivot() followed by fillna(0), but pivot_table is preferred as it handles duplicates and provides fill_value directly. The fill_value parameter replaces NaN values that occur when there are no matching combinations of index and columns. This is common when pivoting sparse data where not all combinations exist. Filling with 0 is appropriate for count data or when absence should be represented as zero rather than missing.

## Q13: How do you detect and remove outliers using IQR?

**A13:** Calculate the Interquartile Range (IQR) and identify values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR. For example, Q1 = df['column'].quantile(0.25), Q3 = df['column'].quantile(0.75), IQR = Q3 - Q1, and then df_clean = df[(df['column'] >= Q1 - 1.5*IQR) & (df['column'] <= Q3 + 1.5*IQR)]. This keeps values within 1.5 IQR of the quartiles. You can adjust the multiplier (1.5 is standard) to be more or less strict. The IQR method is robust to outliers and works well for skewed distributions. After detection, you can remove outliers, cap them, or flag them for further investigation depending on your analysis needs.

## Q14: How do you create a lagged feature column?

**A14:** Use the shift() method to create a column with values from previous rows. For example, df['lag_1'] = df['column'].shift(1) creates a column with values from one row before. For time series with datetime index, df['lag_1'] = df['column'].shift(1, freq='D') shifts by one day. You can specify the number of periods to shift (positive shifts backward, negative shifts forward). Lagged features are essential for time series forecasting and feature engineering, as they capture temporal dependencies. The first row will have NaN since there's no previous value, which you may need to handle depending on your use case. Multiple lags can be created by shifting different amounts.

## Q15: How do you one-hot encode a categorical column?

**A15:** Use get_dummies() to create binary columns for each category. For example, pd.get_dummies(df['category_col'], prefix='category') creates binary columns for each unique value. To add these back to the original DataFrame, use pd.concat([df, pd.get_dummies(df['category_col'], prefix='category')], axis=1). Alternatively, use pd.get_dummies(df, columns=['category_col']) to encode specific columns directly in the DataFrame. The prefix parameter adds a prefix to column names to avoid conflicts. One-hot encoding converts categorical variables into a binary matrix, which is required for many machine learning algorithms that can't handle categorical data directly. This is a fundamental preprocessing step for ML pipelines.

## Q16: How do you read a large CSV file in chunks and process it?

**A16:** Use the chunksize parameter in read_csv() to read the file in chunks, then iterate over chunks. For example, chunk_iter = pd.read_csv('large_file.csv', chunksize=10000) and then for chunk in chunk_iter: process(chunk). Each chunk is a DataFrame with the specified number of rows. You can process each chunk independently, aggregate results, or filter before combining. This approach allows processing files larger than available memory. Common patterns include filtering chunks before concatenation, computing running aggregations, or writing processed chunks to a new file. Remember to process chunks efficiently and avoid operations that require the full dataset unless you aggregate results appropriately.

## Q17: How do you convert wide format to long format?

**A17:** Use melt() to transform columns into rows. For example, df_long = df.melt(id_vars=['id_col'], value_vars=['col1', 'col2', 'col3'], var_name='variable', value_name='value') converts specified columns into rows. The id_vars are columns that stay as columns, value_vars are columns that become rows, var_name is the name for the new column indicating the source column, and value_name is the name for the values. This is the inverse of pivot and is essential for data tidying. Long format is often required for visualization libraries, statistical analysis, or when you need to analyze multiple measurements together. Melt is particularly useful when you have repeated measurements stored as separate columns.

## Q18: How do you compute cumulative sum within groups?

**A18:** Use groupby with cumsum() to calculate cumulative sums per group. For example, df['cumsum'] = df.groupby('group_col')['value_col'].cumsum() computes the cumulative sum of 'value_col' within each group defined by 'group_col'. The cumsum() method works with groupby to reset the cumulative sum at the start of each group. This is useful for tracking running totals, computing cumulative distributions, or creating features for time series analysis within groups. The result maintains the same length as the original DataFrame, with cumulative sums calculated independently for each group. This pattern is common in financial analysis, inventory tracking, and sequential data analysis.

## Q19: How do you extract year, month, and day from a datetime column?

**A19:** Use the dt accessor to extract datetime components. For example, df['year'] = df['date_col'].dt.year, df['month'] = df['date_col'].dt.month, and df['day'] = df['date_col'].dt.day. You can also extract other components like dt.quarter, dt.weekday, dt.dayofweek, dt.hour, etc. The dt accessor provides access to all datetime properties and methods. These extracted components are useful for time-based analysis, grouping by time periods, creating time-based features for machine learning, or filtering data by specific time ranges. Ensure the column is datetime type using pd.to_datetime() first if it's stored as strings or other types.

## Q20: How do you find duplicate rows based on specific columns?

**A20:** Use the duplicated() method with subset parameter to check for duplicates in specific columns. For example, df[df.duplicated(subset=['col1', 'col2'], keep=False)] returns all rows that have duplicates based on col1 and col2, including the first occurrence. The keep parameter controls which duplicates to mark: 'first' marks all except first, 'last' marks all except last, and False marks all duplicates. To remove duplicates, use drop_duplicates(subset=['col1', 'col2']). This is essential for data quality checks, identifying duplicate records, and cleaning datasets. Finding duplicates based on specific columns is more flexible than finding exact duplicate rows, as it allows you to focus on the columns that should uniquely identify records.
