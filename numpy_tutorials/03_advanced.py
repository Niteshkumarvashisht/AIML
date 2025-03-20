import numpy as np

print("NumPy Advanced Tutorial")
print("=====================")

# 1. Custom Data Types and Structured Arrays
print("\n1. Custom Data Types and Structured Arrays:")
# Create a structured array
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('salary', 'f8')])
employees = np.array([
    ('John', 35, 70000.0),
    ('Lisa', 28, 65000.0),
    ('Bob', 45, 80000.0)
], dtype=dt)
print("Employees:")
print(employees)
print(f"Names: {employees['name']}")
print(f"Average salary: {np.mean(employees['salary'])}")

# 2. Memory Views and Buffer Protocol
print("\n2. Memory Views and Buffer Protocol:")
x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
print(f"Original array:\n{x}")
print(f"Memory size: {x.nbytes} bytes")
print(f"Memory layout: {x.strides}")

# 3. Advanced Indexing and Masking
print("\n3. Advanced Indexing and Masking:")
arr = np.random.rand(5, 5)
print(f"Original array:\n{arr}")
# Complex boolean masking
mask = (arr > 0.3) & (arr < 0.7)
print(f"Masked values:\n{arr[mask]}")

# 4. Custom Universal Functions
print("\n4. Custom Universal Functions:")
def custom_sigmoid(x):
    return 1 / (1 + np.exp(-x))

custom_sigmoid_ufunc = np.frompyfunc(custom_sigmoid, 1, 1)
x = np.array([-2, -1, 0, 1, 2])
print(f"Custom sigmoid function: {custom_sigmoid_ufunc(x)}")

# 5. Advanced Linear Algebra
print("\n5. Advanced Linear Algebra:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
# Determinant
print(f"Determinant of A: {np.linalg.det(A)}")
# Eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(A)
print(f"Eigenvalues of A: {eigenvals}")
print(f"Eigenvectors of A:\n{eigenvecs}")
# Matrix inverse
print(f"Inverse of A:\n{np.linalg.inv(A)}")
# Solve linear equations: Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print(f"Solution to Ax = b: {x}")

# 6. Fast Fourier Transform
print("\n6. Fast Fourier Transform:")
# Generate a signal
t = np.linspace(0, 1, 100)
signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
# Compute FFT
fft = np.fft.fft(signal)
freq = np.fft.fftfreq(len(t), t[1] - t[0])
print(f"FFT frequencies: {freq[:5]}")
print(f"FFT magnitudes: {np.abs(fft)[:5]}")

# 7. Advanced Random Number Generation
print("\n7. Advanced Random Number Generation:")
# Custom probability distribution
custom_dist = np.random.default_rng(42)
data = custom_dist.normal(loc=0, scale=1, size=1000)
print(f"Custom distribution stats:")
print(f"Mean: {np.mean(data):.2f}")
print(f"Std: {np.std(data):.2f}")

# 8. Polynomial Operations
print("\n8. Polynomial Operations:")
# Define polynomial coefficients (x^2 + 2x + 1)
p = np.array([1, 2, 1])
# Find roots
roots = np.roots(p)
print(f"Polynomial roots: {roots}")
# Evaluate polynomial
x = np.array([0, 1, 2])
y = np.polyval(p, x)
print(f"Polynomial values at {x}: {y}")

# 9. Advanced Array Operations
print("\n9. Advanced Array Operations:")
# Outer product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
outer_prod = np.outer(a, b)
print(f"Outer product:\n{outer_prod}")
# Kronecker product
kron_prod = np.kron(a, b)
print(f"Kronecker product:\n{kron_prod}")
