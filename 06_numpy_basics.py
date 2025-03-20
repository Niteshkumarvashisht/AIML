import numpy as np

# NumPy Basics - Numerical Computing Library
print("NumPy Basics Examples:")

# 1. Creating Arrays
print("\n1. Creating Arrays:")
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros(5)
arr3 = np.ones((2, 3))
arr4 = np.arange(0, 10, 2)  # Start, stop, step

print(f"Basic array: {arr1}")
print(f"Zeros array: {arr2}")
print(f"Ones array:\n{arr3}")
print(f"Range array: {arr4}")

# 2. Array Operations
print("\n2. Array Operations:")
arr5 = np.array([1, 2, 3])
arr6 = np.array([4, 5, 6])

print(f"Addition: {arr5 + arr6}")
print(f"Multiplication: {arr5 * arr6}")
print(f"Square root: {np.sqrt(arr5)}")

# 3. Array Reshaping
print("\n3. Array Reshaping:")
arr7 = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr7.reshape(2, 3)
print(f"Original: {arr7}")
print(f"Reshaped:\n{reshaped}")

# 4. Statistical Operations
print("\n4. Statistical Operations:")
data = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(data)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Max: {np.max(data)}")
print(f"Min: {np.min(data)}")

# 5. Matrix Operations
print("\n5. Matrix Operations:")
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print(f"Matrix multiplication:\n{np.dot(matrix1, matrix2)}")
print(f"Transpose:\n{matrix1.T}")

# 6. Random Numbers
print("\n6. Random Numbers:")
random_array = np.random.rand(5)  # 5 random numbers between 0 and 1
print(f"Random array: {random_array}")
normal_dist = np.random.normal(0, 1, 5)  # 5 numbers from normal distribution
print(f"Normal distribution: {normal_dist}")
