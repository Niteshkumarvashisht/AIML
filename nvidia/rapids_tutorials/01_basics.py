import cudf
import cuml

print("RAPIDS Basics Tutorial")
print("======================")

# 1. cuDF DataFrame
print("\n1. cuDF DataFrame:")
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print("cuDF DataFrame:")
print(df)

# 2. cuML Linear Regression
print("\n2. cuML Linear Regression:")
X = cudf.DataFrame({'x': [1.0, 2.0, 3.0]})
y = cudf.Series([1.0, 2.0, 3.0])
model = cuml.LinearRegression()
model.fit(X, y)

print("Model coefficients:")
print(model.coef_)

print("\nNote: This tutorial demonstrates basic RAPIDS usage. Ensure RAPIDS is installed with CUDA support.")
