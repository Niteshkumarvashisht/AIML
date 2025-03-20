import numpy as np

print("NumPy Intermediate Tutorial")
print("=========================")

# 1. Advanced Array Operations
print("\n1. Advanced Array Operations:")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original array:\n{arr}")
print(f"Transpose:\n{arr.T}")
print(f"Flattened: {arr.flatten()}")
print(f"Reshaped: {arr.reshape(3, 2)}")

# 2. Broadcasting
print("\n2. Broadcasting:")
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(f"Array a:\n{a}")
print(f"Array b: {b}")
print(f"a + b (broadcasting):\n{a + b}")

# 3. Advanced Indexing
print("\n3. Advanced Indexing:")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Original array:\n{arr}")
# Boolean indexing
mask = arr > 5
print(f"Elements > 5:\n{arr[mask]}")
# Fancy indexing
indices = np.array([0, 2])
print(f"First and last rows:\n{arr[indices]}")

# 4. Array Stacking
print("\n4. Array Stacking:")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Vertical stack:\n{np.vstack((a, b))}")
print(f"Horizontal stack:\n{np.hstack((a, b))}")

# 5. Array Splitting
print("\n5. Array Splitting:")
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Original array:\n{arr}")
# Split horizontally
print("Horizontal split:")
for piece in np.hsplit(arr, 2):
    print(piece)
# Split vertically
print("Vertical split:")
for piece in np.vsplit(arr, 3):
    print(piece)

# 6. Universal Functions (ufuncs)
print("\n6. Universal Functions:")
arr = np.array([0, np.pi/4, np.pi/2])
print(f"Original array: {arr}")
print(f"Sin: {np.sin(arr)}")
print(f"Cos: {np.cos(arr)}")
print(f"Tan: {np.tan(arr)}")

# 7. Statistical Operations
print("\n7. Statistical Operations:")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Original array:\n{arr}")
print(f"Row means: {np.mean(arr, axis=1)}")
print(f"Column means: {np.mean(arr, axis=0)}")
print(f"Cumulative sum:\n{np.cumsum(arr)}")

# 8. Random Number Generation
print("\n8. Random Number Generation:")
np.random.seed(42)  # For reproducibility
print(f"Random integers: {np.random.randint(1, 10, 5)}")
print(f"Random floats: {np.random.random(5)}")
print(f"Normal distribution: {np.random.normal(0, 1, 5)}")
print(f"Random choice: {np.random.choice(['a', 'b', 'c'], 3)}")
