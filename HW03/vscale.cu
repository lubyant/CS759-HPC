#include "vscale.cuh"

/**
 * @brief kernel function for calcualte vscale
 *
 * @param a array a
 * @param b array b
 * @param n number of entries in array a,b
 * @return __global__
 */
__global__ void vscale(const float *a, float *b, unsigned int n) {
  typedef unsigned int uint;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    b[idx] = a[idx] * b[idx];
  }
}