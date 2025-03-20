import cudf
import cuml

print("RAPIDS Intermediate Tutorial")
print("=============================")

# 1. cuDF DataFrame Operations
print("\n1. cuDF DataFrame Operations:")
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Performing operations
print("Initial DataFrame:", df)

# Adding a new column
print("\n2. Adding a new column:")
df['c'] = df['a'] + df['b']
print("Updated DataFrame:", df)

# 3. cuML Linear Regression
print("\n3. cuML Linear Regression:")
X = cudf.DataFrame({'x': [1.0, 2.0, 3.0]})
y = cudf.Series([1.0, 2.0, 3.0])
model = cuml.LinearRegression()
model.fit(X, y)
print("Model coefficients:", model.coef_)
