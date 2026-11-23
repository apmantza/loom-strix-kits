// Read bandwidth of a row-strided weight read against the row pitch (HRX-like E299; strix-halo-engine-lab PR #1010 measured BF16
// rows at a pitch that is a multiple of 2048 B colliding on DRAM channels: 121-193 GB/s vs 219-230 GB/s at pitch + 256 B).
// The access pattern of a decode matvec: each workgroup reads whole rows of ROW bytes (128-bit loads, 4 in flight per lane),
// workgroups take consecutive rows, rows PITCH bytes apart. Prints best and median GB/s of the bytes the rows hold (not the pitch).
// usage: pitch_probe ROW_BYTES PITCH_BYTES [GiB=2] [reps=20] [blocks_per_CU=8]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void k_rows(const uint4* __restrict__ a, size_t rows, unsigned row16, unsigned pitch16, unsigned* sink) {
  uint4 acc = {0, 0, 0, 0};
  for (size_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const uint4* p = a + r * pitch16;
    for (unsigned i = threadIdx.x; i < row16; i += 4 * blockDim.x) {
      uint4 v0 = p[i];
      uint4 v1 = i + blockDim.x < row16 ? p[i + blockDim.x] : uint4{0, 0, 0, 0};
      uint4 v2 = i + 2 * blockDim.x < row16 ? p[i + 2 * blockDim.x] : uint4{0, 0, 0, 0};
      uint4 v3 = i + 3 * blockDim.x < row16 ? p[i + 3 * blockDim.x] : uint4{0, 0, 0, 0};
      acc.x ^= v0.x ^ v1.x ^ v2.x ^ v3.x; acc.y ^= v0.y ^ v1.y ^ v2.y ^ v3.y;
      acc.z ^= v0.z ^ v1.z ^ v2.z ^ v3.z; acc.w ^= v0.w ^ v1.w ^ v2.w ^ v3.w;
    }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) sink[0] = 1;   // keeps the loads; never true for the fill below
}

// 8-byte loads for pitches that are not multiples of 16 (Q6_K rows: 4200 / 14280 B, 8-byte aligned)
__global__ void k_rows8(const uint2* __restrict__ a, size_t rows, unsigned row8, unsigned pitch8, unsigned* sink) {
  uint2 acc = {0, 0};
  for (size_t r = blockIdx.x; r < rows; r += gridDim.x) {
    const uint2* p = a + r * pitch8;
    for (unsigned i = threadIdx.x; i < row8; i += 4 * blockDim.x) {
      uint2 v0 = p[i];
      uint2 v1 = i + blockDim.x < row8 ? p[i + blockDim.x] : uint2{0, 0};
      uint2 v2 = i + 2 * blockDim.x < row8 ? p[i + 2 * blockDim.x] : uint2{0, 0};
      uint2 v3 = i + 3 * blockDim.x < row8 ? p[i + 3 * blockDim.x] : uint2{0, 0};
      acc.x ^= v0.x ^ v1.x ^ v2.x ^ v3.x; acc.y ^= v0.y ^ v1.y ^ v2.y ^ v3.y;
    }
  }
  if ((acc.x ^ acc.y) == 0x12345678u) sink[0] = 1;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: pitch_probe ROW_BYTES PITCH_BYTES [GiB=2] [reps=20] [blocks_per_CU=8]\n"); return 2; }
  size_t row = strtoull(argv[1], 0, 0), pitch = strtoull(argv[2], 0, 0);
  double gib = argc > 3 ? atof(argv[3]) : 2.0;
  int reps = argc > 4 ? atoi(argv[4]) : 20, per_cu = argc > 5 ? atoi(argv[5]) : 8;
  bool w8 = (row % 16) || (pitch % 16);                        // 8-byte loads unless both are 16-byte multiples
  if (row % 8 || pitch % 8 || pitch < row) { fprintf(stderr, "ROW and PITCH must be multiples of 8, PITCH >= ROW\n"); return 2; }
  size_t bytes = (size_t)(gib * (1ull << 30)), rows = bytes / pitch;
  hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
  uint4* a; unsigned* sink;
  CHECK(hipMalloc(&a, rows * pitch)); CHECK(hipMalloc(&sink, 64));
  CHECK(hipMemset(a, 0x5a, rows * pitch));
  int blocks = p.multiProcessorCount * per_cu, threads = 256;
  hipEvent_t t0, t1; CHECK(hipEventCreate(&t0)); CHECK(hipEventCreate(&t1));
  std::vector<double> gbs;
  for (int i = 0; i < reps + 2; i++) {
    CHECK(hipEventRecord(t0));
    if (w8) k_rows8<<<blocks, threads>>>((const uint2*)a, rows, (unsigned)(row / 8), (unsigned)(pitch / 8), sink);
    else k_rows<<<blocks, threads>>>(a, rows, (unsigned)(row / 16), (unsigned)(pitch / 16), sink);
    CHECK(hipEventRecord(t1)); CHECK(hipEventSynchronize(t1));
    float ms; CHECK(hipEventElapsedTime(&ms, t0, t1));
    if (i >= 2) gbs.push_back(rows * row / (ms * 1e6));
  }
  std::sort(gbs.begin(), gbs.end());
  printf("row %zu pitch %zu (pitch %% 2048 = %zu, %d-byte loads): %zu rows, %d blocks; read best %.1f median %.1f GB/s\n", row, pitch, pitch % 2048, w8 ? 8 : 16,
         rows, blocks, gbs.back(), gbs[gbs.size() / 2]);
  return 0;
}
