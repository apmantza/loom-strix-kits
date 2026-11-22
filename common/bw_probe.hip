// DRAM bandwidth ceilings for the decode floors (devices/<gfx>.md, lanes.json "hardware"): read-only, write-only and copy
// kernels with 128-bit accesses over a buffer far larger than any cache, grid-stride, timed with HIP events after a warm-up.
// Reports best and median GB/s per kernel. The read kernel is the one that bounds decode (weights and KV are read, not written).
// build: common/bw_probe.sh (hipcc; the machine env may name ROCM_SDK for a toolchain with gfx device libraries)
// usage: bw_probe [GiB=4] [reps=20] [blocks_per_CU=8]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void k_read(const uint4* __restrict__ src, size_t n, uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = src[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;   // data-dependent and practically never true: the loads stay live
}

__global__ void k_write(uint4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = make_uint4((uint32_t)i, 1u, 2u, 3u);
}

__global__ void k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

int main(int argc, char** argv) {
  double gib = argc > 1 ? atof(argv[1]) : 4.0;
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  int per_cu = argc > 3 ? atoi(argv[3]) : 8;
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  size_t bytes = (size_t)(gib * (1ull << 30)) & ~(size_t)15, n = bytes / 16;
  uint4 *a, *b;
  uint32_t* sink;
  CHECK(hipMalloc(&a, bytes));
  CHECK(hipMalloc(&b, bytes));
  CHECK(hipMalloc(&sink, 1 << 20));
  CHECK(hipMemset(a, 0x5a, bytes));
  CHECK(hipMemset(b, 0, bytes));
  int threads = 256, blocks = p.multiProcessorCount * per_cu;
  hipEvent_t t0, t1;
  CHECK(hipEventCreate(&t0));
  CHECK(hipEventCreate(&t1));
  printf("device %s (%s), %d CUs; buffer %.2f GiB; %d blocks x %d threads; %d reps\n", p.name, p.gcnArchName, p.multiProcessorCount,
         bytes / double(1ull << 30), blocks, threads, reps);
  const char* names[3] = {"read", "write", "copy"};
  for (int k = 0; k < 3; ++k) {
    std::vector<double> gbs;
    for (int r = -2; r < reps; ++r) {   // two warm-up runs
      CHECK(hipEventRecord(t0));
      if (k == 0) k_read<<<blocks, threads>>>(a, n, sink);
      if (k == 1) k_write<<<blocks, threads>>>(b, n);
      if (k == 2) k_copy<<<blocks, threads>>>(a, b, n);
      CHECK(hipEventRecord(t1));
      CHECK(hipEventSynchronize(t1));
      float ms;
      CHECK(hipEventElapsedTime(&ms, t0, t1));
      double moved = (k == 2 ? 2.0 : 1.0) * bytes;   // copy counts its read and its write
      if (r >= 0) gbs.push_back(moved / (ms * 1e-3) / 1e9);
    }
    std::sort(gbs.begin(), gbs.end());
    printf("%-5s best %7.1f GB/s  median %7.1f GB/s\n", names[k], gbs.back(), gbs[gbs.size() / 2]);
  }
  return 0;
}
