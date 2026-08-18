// Direct achievable-bandwidth measurement on this L4, independent of llama.cpp.
// Streams a buffer far larger than the 48 MB L2 so the number is DRAM, not cache.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void read_kernel(const float4 * __restrict__ src, float4 * __restrict__ sink, size_t n) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (; i < n; i += stride) {
        float4 v = src[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) sink[0] = acc;   // never taken; defeats dead-code elimination
}

int main() {
    const size_t bytes = 8ull << 30;                 // 8 GiB, ~170x the 48 MB L2
    const size_t n = bytes / sizeof(float4);
    float4 *src = nullptr, *sink = nullptr;
    if (cudaMalloc(&src, bytes) != cudaSuccess) { printf("alloc failed\n"); return 1; }
    cudaMalloc(&sink, sizeof(float4));
    cudaMemset(src, 1, bytes);

    int blocks = 0, threads = 256;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, read_kernel, threads, 0);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    const int grid = blocks * p.multiProcessorCount;

    read_kernel<<<grid, threads>>>(src, sink, n);    // warm up
    cudaDeviceSynchronize();

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int iters = 5;
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) read_kernel<<<grid, threads>>>(src, sink, n);
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);

    const double gbps = (double) bytes * iters / (ms * 1e-3) / 1e9;
    printf("device            : %s, %d SMs\n", p.name, p.multiProcessorCount);
    printf("grid              : %d blocks x %d threads\n", grid, threads);
    printf("buffer            : %.1f GiB (L2 is %.0f MB)\n", bytes / 1073741824.0, p.l2CacheSize / 1048576.0);
    printf("streaming read    : %.1f GB/s over %d iterations\n", gbps, iters);
    printf("datasheet peak    : 300.05 GB/s\n");
    printf("achieved fraction : %.1f%%\n", 100.0 * gbps / 300.05);
    cudaFree(src); cudaFree(sink);
    return 0;
}
