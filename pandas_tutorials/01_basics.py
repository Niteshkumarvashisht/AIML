import pandas as pd
import numpy as np

print("Pandas Basics Tutorial")
print("====================")

# 1. Series Creation
print("\n1. Series Creation:")
# From list
s1 = pd.Series([1, 2, 3, 4, 5])
print("Series from list:")
print(s1)

# From dictionary
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print("\nSeries from dictionary:")
print(s2)

# 2. DataFrame Creation
print("\n2. DataFrame Creation:")
# From dictionary
data = {
    'Name': ['John', 'Anna', 'Peter'],
    'Age': [28, 22, 35],
    'City': ['New York', 'Paris', 'London']
}
df1 = pd.DataFrame(data)
print("DataFrame from dictionary:")
print(df1)

# From list of lists
data = [[1, 'John', 28],
        [2, 'Anna', 22],
        [3, 'Peter', 35]]
df2 = pd.DataFrame(data, columns=['ID', 'Name', 'Age'])
print("\nDataFrame from list of lists:")
print(df2)

# 3. Basic DataFrame Operations
print("\n3. Basic DataFrame Operations:")
print("\nDataFrame Info:")
print(df1.info())

print("\nDataFrame Description:")
print(df1.describe())

print("\nColumn Names:")
print(df1.columns)

print("\nFirst 2 rows:")
print(df1.head(2))

# 4. Basic Data Selection
print("\n4. Basic Data Selection:")
# Select column
print("\nSelect 'Name' column:")
print(df1['Name'])

# Select multiple columns
print("\nSelect 'Name' and 'Age' columns:")
print(df1[['Name', 'Age']])

# Select by index
print("\nSelect first row:")
print(df1.iloc[0])

# 5. Basic Data Manipulation
print("\n5. Basic Data Manipulation:")
# Add new column
df1['Country'] = ['USA', 'France', 'UK']
print("\nAdded new column 'Country':")
print(df1)

# Modify values
df1.loc[0, 'Age'] = 29
print("\nModified Age for first row:")
print(df1)

# 6. Basic Filtering
print("\n6. Basic Filtering:")
# Filter by condition
print("\nPeople older than 25:")
print(df1[df1['Age'] > 25])

# Multiple conditions
print("\nPeople older than 25 from USA:")
print(df1[(df1['Age'] > 25) & (df1['Country'] == 'USA')])

# 7. Basic Sorting
print("\n7. Basic Sorting:")
# Sort by Age
print("\nSort by Age:")
print(df1.sort_values('Age'))

# Sort by multiple columns
print("\nSort by Age and Name:")
print(df1.sort_values(['Age', 'Name']))

# 8. Handling Missing Data
print("\n8. Handling Missing Data:")
# Create DataFrame with missing values
df3 = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [np.nan, 5, 6],
    'C': [7, 8, 9]
})
print("\nDataFrame with missing values:")
print(df3)

print("\nDrop NA values:")
print(df3.dropna())

print("\nFill NA values with 0:")
print(df3.fillna(0))
