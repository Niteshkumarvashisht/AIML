import pandas as pd
import numpy as np

# Pandas Basics - Data Analysis Library
print("Pandas Basics Examples:")

# 1. Creating Series and DataFrames
print("\n1. Creating Series and DataFrames:")
# Series (1D)
series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print("Series:")
print(series)

# DataFrame (2D)
data = {
    'Name': ['John', 'Anna', 'Peter'],
    'Age': [28, 22, 35],
    'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)

# 2. Reading and Writing Data
print("\n2. Reading and Writing Data:")
# Writing DataFrame to CSV
df.to_csv('sample_data.csv', index=False)
print("DataFrame saved to 'sample_data.csv'")

# Reading CSV
df_read = pd.read_csv('sample_data.csv')
print("\nRead from CSV:")
print(df_read)

# 3. Basic Operations
print("\n3. Basic Operations:")
# Adding a new column
df['Country'] = ['USA', 'France', 'UK']
print("\nAdded new column:")
print(df)

# Filtering
print("\nPeople older than 25:")
print(df[df['Age'] > 25])

# 4. Data Analysis
print("\n4. Data Analysis:")
print("\nBasic statistics:")
print(df.describe())

print("\nMean age:", df['Age'].mean())
print("Age count:", df['Age'].count())

# 5. Grouping and Aggregation
print("\n5. Grouping and Aggregation:")
# Create more data
df2 = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'A'],
    'Value': [10, 20, 30, 40, 50]
})
print("\nGrouping data:")
print(df2)
print("\nMean by category:")
print(df2.groupby('Category')['Value'].mean())

# 6. Handling Missing Data
print("\n6. Handling Missing Data:")
df3 = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [np.nan, 5, 6]
})
print("\nDataFrame with missing values:")
print(df3)
print("\nDropping NA values:")
print(df3.dropna())
print("\nFilling NA values with 0:")
print(df3.fillna(0))
