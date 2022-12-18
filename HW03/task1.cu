#include <cstdio>
#include <cuda.h>
typedef unsigned int uint;
const uint nThreads = 8;

/**
 * @brief Kernel function to calculate factoroial
 *
 * @return __global__
 */
__global__ void fact() {
  if (threadIdx.x < nThreads) {
    // lambda for factorial
    auto fac = [](uint n) {
      uint ret = 1;
      for (uint i = 1; i <= n; i++)
        ret *= i;
      return ret;
    };
    printf("%d!=%d\n", threadIdx.x + 1, fac(threadIdx.x + 1));
  }
}

int main(int argc, char *argv[]) {
  // one block, 8 threads
  fact<<<1, nThreads>>>();

  // Synchronize for printf
  cudaDeviceSynchronize();
  return 0;
}