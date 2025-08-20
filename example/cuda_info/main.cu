#include <stdio.h>

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

int main() {
    CudaDeviceInfo();
    return 0;
}