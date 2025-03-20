import pandas as pd
import numpy as np

print("Pandas Intermediate Tutorial")
print("==========================")

# Create sample data
np.random.seed(42)
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), 
                 index=dates, 
                 columns=list('ABCD'))

# 1. Advanced Indexing
print("\n1. Advanced Indexing:")
print("Original DataFrame:")
print(df)

print("\nMulti-index selection:")
print(df.loc['2023-01-02':'2023-01-04', ['A', 'B']])

# 2. Data Transformation
print("\n2. Data Transformation:")
# Apply function to column
print("\nApply custom function:")
df['E'] = df['A'].apply(lambda x: 'High' if x > 0 else 'Low')
print(df)

# Map values
mapping = {'High': 1, 'Low': 0}
df['E_numeric'] = df['E'].map(mapping)
print("\nMapped values:")
print(df)

# 3. Grouping Operations
print("\n3. Grouping Operations:")
# Create sample data
data = {
    'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Value': [10, 20, 30, 40, 50, 60],
    'Region': ['East', 'West', 'East', 'West', 'East', 'West']
}
df2 = pd.DataFrame(data)
print("\nOriginal data:")
print(df2)

# Group by one column
print("\nGroup by Category:")
print(df2.groupby('Category')['Value'].mean())

# Group by multiple columns
print("\nGroup by Category and Region:")
print(df2.groupby(['Category', 'Region'])['Value'].mean())

# 4. Merging and Joining
print("\n4. Merging and Joining:")
# Create two dataframes
df_left = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K3'],
    'A': ['A0', 'A1', 'A2', 'A3'],
    'B': ['B0', 'B1', 'B2', 'B3']
})

df_right = pd.DataFrame({
    'key': ['K0', 'K1', 'K2', 'K4'],
    'C': ['C0', 'C1', 'C2', 'C4'],
    'D': ['D0', 'D1', 'D2', 'D4']
})

print("\nLeft DataFrame:")
print(df_left)
print("\nRight DataFrame:")
print(df_right)

# Different types of joins
print("\nInner join:")
print(pd.merge(df_left, df_right, on='key', how='inner'))

print("\nOuter join:")
print(pd.merge(df_left, df_right, on='key', how='outer'))

# 5. Pivot Tables
print("\n5. Pivot Tables:")
# Create sample data
data = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 200, 120, 210],
    'Region': ['East', 'West', 'East', 'West']
}
df3 = pd.DataFrame(data)
print("\nOriginal data:")
print(df3)

print("\nPivot table:")
pivot = pd.pivot_table(df3, 
                      values='Sales', 
                      index=['Date'], 
                      columns=['Product'],
                      aggfunc=np.sum)
print(pivot)

# 6. Time Series Operations
print("\n6. Time Series Operations:")
# Create time series data
dates = pd.date_range('20230101', periods=6)
ts = pd.Series(np.random.randn(6), index=dates)
print("\nTime series data:")
print(ts)

print("\nResample by month:")
print(ts.resample('M').mean())

print("\nShift data:")
print(ts.shift(2))

# 7. String Operations
print("\n7. String Operations:")
# Create sample string data
s = pd.Series(['A1', 'B2', 'C3', 'D4'])
print("\nOriginal string series:")
print(s)

print("\nExtract numbers:")
print(s.str.extract('(\d+)'))

print("\nConvert to uppercase:")
print(s.str.upper())

# 8. Categorical Data
print("\n8. Categorical Data:")
# Create categorical data
cat_data = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
print("\nCategorical data:")
print(cat_data)

# Convert to categorical
df3['Product'] = df3['Product'].astype('category')
print("\nMemory usage after conversion:")
print(df3.memory_usage())
