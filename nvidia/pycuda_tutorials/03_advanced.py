import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

print("PyCUDA Advanced Tutorial")
print("==========================")

# 1. Custom Kernel Optimization
print("\n1. Custom Kernel Optimization:")
mod_opt = SourceModule("""
__global__ void optimized_add(float *dest, float *a, float *b, int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        dest[idx] = a[idx] + b[idx];
    }
}
""")

def optimized_kernel_example(size=1000000):
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    dest = np.zeros_like(a)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    dest_gpu = cuda.mem_alloc(dest.nbytes)

    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    block_size = 512
    grid_size = (size + block_size - 1) // block_size
    func = mod_opt.get_function("optimized_add")
    func(dest_gpu, a_gpu, b_gpu, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    cuda.memcpy_dtoh(dest, dest_gpu)
    print("Optimized kernel completed")

# 2. Performance Measurement
print("\n2. Performance Measurement:")
def measure_performance():
    import time
    start = time.time()
    optimized_kernel_example(1000000)
    end = time.time()
    print(f"Performance measured: {end - start:.6f} seconds")

# 3. Using Multiple GPUs
print("\n3. Using Multiple GPUs:")
def multi_gpu_example():
    num_gpus = cuda.Device.count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        device = cuda.Device(i)
        print(f"Using GPU: {device.name()}")

if __name__ == "__main__":
    optimized_kernel_example(1000000)
    measure_performance()
    multi_gpu_example()

print("\nNote: This tutorial demonstrates advanced PyCUDA operations.")
