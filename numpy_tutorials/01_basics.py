import numpy as np

print("NumPy Basics Tutorial")
print("====================")

# 1. Array Creation
print("\n1. Basic Array Creation:")
arr1 = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"1D array: {arr1}")
print(f"2D array:\n{arr2d}")

# 2. Array from Range
print("\n2. Array from Range:")
arr_range = np.arange(0, 10, 2)  # start, stop, step
arr_linspace = np.linspace(0, 1, 5)  # start, stop, num_points
print(f"Range array: {arr_range}")
print(f"Linspace array: {arr_linspace}")

# 3. Special Arrays
print("\n3. Special Arrays:")
zeros = np.zeros((2, 3))
ones = np.ones((2, 2))
identity = np.eye(3)
print(f"Zeros:\n{zeros}")
print(f"Ones:\n{ones}")
print(f"Identity:\n{identity}")

# 4. Array Properties
print("\n4. Array Properties:")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Shape: {arr.shape}")
print(f"Dimensions: {arr.ndim}")
print(f"Size: {arr.size}")
print(f"Data type: {arr.dtype}")

# 5. Basic Operations
print("\n5. Basic Operations:")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")
print(f"Square root: {np.sqrt(a)}")
print(f"Exponential: {np.exp(a)}")

# 6. Array Indexing
print("\n6. Array Indexing:")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Original array:\n{arr}")
print(f"Element at (1,2): {arr[1,2]}")
print(f"First row: {arr[0]}")
print(f"First column: {arr[:,0]}")

# 7. Array Slicing
print("\n7. Array Slicing:")
print(f"First two rows:\n{arr[0:2]}")
print(f"Last two columns:\n{arr[:,1:3]}")

# 8. Basic Statistics
print("\n8. Basic Statistics:")
data = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Variance: {np.var(data)}")
