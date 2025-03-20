import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

print("PyCUDA Intermediate Tutorial")
print("===========================")

# 1. Advanced Matrix Operations
print("\n1. Advanced Matrix Operations on GPU:")
mod = SourceModule("""
__global__ void matrix_add(float *dest, float *a, float *b, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        dest[i] = a[i] + b[i];
    }
}
""")

def advanced_matrix_add_example(size=1000000):
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    dest = np.zeros_like(a)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    dest_gpu = cuda.mem_alloc(dest.nbytes)

    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    block_size = 256
    grid_size = (size + block_size - 1) // block_size
    func = mod.get_function("matrix_add")
    func(dest_gpu, a_gpu, b_gpu, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    cuda.memcpy_dtoh(dest, dest_gpu)
    print("Advanced matrix addition completed")

# 2. Error Handling
print("\n2. Error Handling in PyCUDA:")
def error_handling_example():
    try:
        # Intentionally cause an error by allocating too much memory
        huge_array = np.zeros((1000000000,), dtype=np.float32)
        cuda.mem_alloc(huge_array.nbytes)
    except cuda.MemoryError:
        print("Caught a memory error!")

# 3. Custom Kernel with Shared Memory
print("\n3. Custom Kernel with Shared Memory:")
mod_shared = SourceModule("""
__global__ void add_with_shared_memory(float *a, float *b, float *c, int N)
{
    extern __shared__ float shared[];
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        shared[threadIdx.x] = a[i] + b[i];
        __syncthreads();
        c[i] = shared[threadIdx.x];
    }
}
""")

def shared_memory_example(size=100):
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(a)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    block_size = 32
    grid_size = (size + block_size - 1) // block_size
    func = mod_shared.get_function("add_with_shared_memory")
    func(a_gpu, b_gpu, c_gpu, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1), shared=block_size * 4)

    cuda.memcpy_dtoh(c, c_gpu)
    print("Shared memory example completed")

if __name__ == "__main__":
    advanced_matrix_add_example(1000000)
    error_handling_example()
    shared_memory_example(100)

print("\nNote: This tutorial demonstrates intermediate PyCUDA operations.")
