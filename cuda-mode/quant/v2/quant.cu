// nchw_per_channel_quant.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call) do {                                      \
  cudaError_t err = (call);                                        \
  if (err != cudaSuccess) {                                        \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
    std::exit(1);                                                  \
  }                                                                \
} while(0)

__device__ __forceinline__ uint8_t clamp_u8(int v) {
  v = (v < 0) ? 0 : v;
  v = (v > 255) ? 255 : v;
  return static_cast<uint8_t>(v);
}

// NCHW per-channel quant
// i is flat index in [0, N*C*H*W)
// c = (i / (H*W)) % C
__global__ void quant_u8_per_channel_nchw(
    const float* __restrict__ in,
    uint8_t* __restrict__ out,
    int N, int C, int H, int W,
    const float* __restrict__ inv_scales, // [C] = 1/scale[c]
    const int* __restrict__ zps)          // [C]
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nElem = N * C * H * W;
  if (i >= nElem) return;

  int HW = H * W;
  int c = (i / HW) % C;

  float x = in[i];
  int q = (int)nearbyintf(x * inv_scales[c]) + zps[c];
  out[i] = clamp_u8(q);
}

__global__ void dequant_u8_per_channel_nchw(
    const uint8_t* __restrict__ in,
    float* __restrict__ out,
    int N, int C, int H, int W,
    const float* __restrict__ scales, // [C]
    const int* __restrict__ zps)      // [C]
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nElem = N * C * H * W;
  if (i >= nElem) return;

  int HW = H * W;
  int c = (i / HW) % C;

  int q = (int)in[i];
  out[i] = (float)(q - zps[c]) * scales[c];
}

static float randomFloat(float minv, float maxv) {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return minv + r * (maxv - minv);
}

int main() {
  // Example shape (NCHW)
  const int N = 2;
  const int C = 3;
  const int H = 20;
  const int W = 10;
  const int nElem = N * C * H * W;

  // Host buffers
  float* h_in = new float[nElem];
  uint8_t* h_q = new uint8_t[nElem];
  float* h_deq = new float[nElem];

  // Fill input
  for (int i = 0; i < nElem; i++) h_in[i] = randomFloat(-10.0f, 10.0f);

  // Per-channel params (example)
  float h_scales[C];
  int h_zps[C];

  // Example: 서로 다른 채널별 scale/zp (원하면 원하는 값으로)
  // scale이 작을수록 더 촘촘히 양자화됨
  h_scales[0] = 20.0f / 255.0f;
  h_scales[1] = 10.0f / 255.0f;
  h_scales[2] =  5.0f / 255.0f;

  h_zps[0] = 0;
  h_zps[1] = 0;
  h_zps[2] = 0;

  float h_inv_scales[C];
  for (int c = 0; c < C; c++) h_inv_scales[c] = 1.0f / h_scales[c];

  // Device buffers
  float* d_in = nullptr;
  uint8_t* d_q = nullptr;
  float* d_deq = nullptr;
  float* d_scales = nullptr;
  float* d_inv_scales = nullptr;
  int* d_zps = nullptr;

  CUDA_CHECK(cudaMalloc(&d_in, nElem * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, nElem * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_deq, nElem * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scales, C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_inv_scales, C * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_zps, C * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in, nElem * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scales, h_scales, C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_inv_scales, h_inv_scales, C * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_zps, h_zps, C * sizeof(int), cudaMemcpyHostToDevice));

  // Launch
  int threads = 256;
  int blocks = (nElem + threads - 1) / threads;

  quant_u8_per_channel_nchw<<<blocks, threads>>>(d_in, d_q, N, C, H, W, d_inv_scales, d_zps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  dequant_u8_per_channel_nchw<<<blocks, threads>>>(d_q, d_deq, N, C, H, W, d_scales, d_zps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy back
  CUDA_CHECK(cudaMemcpy(h_q, d_q, nElem * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_deq, d_deq, nElem * sizeof(float), cudaMemcpyDeviceToHost));

  // Print a few samples to verify
  std::cout << "Print first 30 elements (NCHW-flat order):\n";
  for (int i = 0; i < 30 && i < nElem; i++) {
    // channel computed for display
    int HW = H * W;
    int c = (i / HW) % C;
    std::cout << "i=" << i
              << " c=" << c
              << " in=" << h_in[i]
              << " q=" << (int)h_q[i]
              << " deq=" << h_deq[i]
              << "\n";
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_deq));
  CUDA_CHECK(cudaFree(d_scales));
  CUDA_CHECK(cudaFree(d_inv_scales));
  CUDA_CHECK(cudaFree(d_zps));

  delete[] h_in;
  delete[] h_q;
  delete[] h_deq;

  return 0;
}
