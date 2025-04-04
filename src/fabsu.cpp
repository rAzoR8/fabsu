#include <hip/hip_runtime.h>
#include <iostream>

// Functions marked with __device__ are executed on the device and called from the device only.
__device__ unsigned int get_thread_idx()
{
    // Built-in threadIdx returns the 3D coordinate of the active work item in the block of threads.
    return threadIdx.x;
}

// Functions marked with __host__ are executed on the host and called from the host.
__host__ void print_hello_host()
{
    std::cout << "Hello world from host!" << std::endl;
}

// Functions marked with __device__ and __host__ are compiled both for host and device.
// These functions cannot use coordinate built-ins.
__device__ __host__ void print_hello()
{
    // Only printf is supported for printing from device code.
    printf("Hello world from device or host!\n");
}

// Functions marked with __global__ are executed on the device and called from the host only.
__global__ void helloworld_kernel()
{
    unsigned int thread_idx = get_thread_idx();
    // Built-in blockIdx returns the 3D coorindate of the active work item in the grid of blocks.
    unsigned int block_idx = blockIdx.x;

    print_hello();

    // Only printf is supported for printing from device code.
    printf("Hello world from device kernel block %u thread %u!\n", block_idx, thread_idx);
}

int main(){

    int devices{0};
    hipError_t error = hipGetDeviceCount(&devices);
    assert(error == hipError_t::hipSuccess);

    std::cout << "Found " << devices << " HIP Devices"  << std::endl;

    if (devices < 1)
    {
        return 0;
    }

    hipDeviceProp_t props;

    for(int d = 0; d < devices; ++d){
        hipGetDeviceProperties(&props, d);
        std::cout << props.name << ": " << props.gcnArchName << std::endl;
    }

    print_hello_host();

    print_hello();

    // Launch the kernel.
    helloworld_kernel<<<dim3(2), // 3D grid specifying number of blocks to launch: (2, 1, 1)
                        dim3(2), // 3D grid specifying number of threads to launch: (2, 1, 1)
                        0, // number of bytes of additional shared memory to allocate
                        hipStreamDefault // stream where the kernel should execute: default stream
                        >>>();

    error = hipGetLastError();
    assert(error == hipError_t::hipSuccess);

    // Wait on all active streams on the current device.
    error = hipDeviceSynchronize();
    assert(error == hipError_t::hipSuccess);

    return devices;
}