import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

print("PyCUDA Basics Tutorial")
print("=====================")

# 1. Basic Vector Addition
print("\n1. Vector Addition on GPU:")
mod = SourceModule("""
__global__ void add_vectors(float *dest, float *a, float *b)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    dest[i] = a[i] + b[i];
}
""")

def vector_add_example(size=1000000):
    # Prepare host data
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    dest = np.zeros_like(a)

    # Allocate memory on GPU
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    dest_gpu = cuda.mem_alloc(dest.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Get the kernel function
    func = mod.get_function("add_vectors")

    # Execute kernel
    block_size = 256
    grid_size = (size + block_size - 1) // block_size
    func(dest_gpu, a_gpu, b_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy result back to host
    cuda.memcpy_dtoh(dest, dest_gpu)
    
    print("Vector addition completed")
    print("First 5 elements:")
    print("A:", a[:5])
    print("B:", b[:5])
    print("Result:", dest[:5])
    print("Verification:", np.allclose(dest, a + b))

# 2. GPU Info
print("\n2. GPU Information:")
def print_gpu_info():
    print(f"CUDA Device Count: {cuda.Device.count()}")
    
    device = cuda.Device(0)
    attrs = device.get_attributes()
    
    print("\nDevice Properties:")
    print(f"  Name: {device.name()}")
    print(f"  Compute Capability: {device.compute_capability()}")
    print(f"  Total Memory: {device.total_memory() // (1024*1024)} MB")
    print(f"  Max Threads per Block: {attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]}")
    print(f"  Max Block Dimensions: {attrs[cuda.device_attribute.MAX_BLOCK_DIM_X]}, "
          f"{attrs[cuda.device_attribute.MAX_BLOCK_DIM_Y]}, "
          f"{attrs[cuda.device_attribute.MAX_BLOCK_DIM_Z]}")

# 3. Matrix Multiplication
print("\n3. Simple Matrix Multiplication:")
mat_mod = SourceModule("""
__global__ void matrix_mul(float *a, float *b, float *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}
""")

def matrix_multiply_example(N=32):
    # Prepare host data
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    c = np.zeros((N, N), dtype=np.float32)

    # Allocate GPU memory
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Get the kernel function
    func = mat_mod.get_function("matrix_mul")

    # Execute kernel
    block_size = 16
    grid_size = (N + block_size - 1) // block_size
    func(a_gpu, b_gpu, c_gpu, np.int32(N),
         block=(block_size, block_size, 1),
         grid=(grid_size, grid_size))

    # Copy result back
    cuda.memcpy_dtoh(c, c_gpu)
    
    print("Matrix multiplication completed")
    print("Verification:", np.allclose(c, np.dot(a, b), rtol=1e-5))

if __name__ == "__main__":
    print_gpu_info()
    vector_add_example(1000000)
    matrix_multiply_example(32)
    
print("\nNote: This tutorial demonstrates basic PyCUDA operations. Make sure you have CUDA toolkit and compatible GPU installed.")
