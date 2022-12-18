#include "matmul.cuh"
#include <cstdio>

int main(int argc, char *argv[]) {
  typedef unsigned int uint;

  // number of rows/cols
  size_t n = (size_t)atoi(argv[1]);
  // size_t n = pow(2,14);

  // number of threads per block
  uint threads_per_block = (uint)atoi(argv[2]);
  // uint threads_per_block = 1024;

  // host array
  float *hA = new float[n * n];
  float *hB = new float[n * n];
  float *hC = new float[n * n]{0};

  // init array by random
  srand(time(NULL));
  for (size_t i = 0; i < n * n; i++) {
    hA[i] = 2 * (float)rand() / ((float)RAND_MAX) - 1;
    hB[i] = 2 * (float)rand() / ((float)RAND_MAX) - 1;
  }

  // timer
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // cal matmul
  matmul(hA, hB, hC, n, threads_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // print out the results
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n", hC[n * n - 1], ms);

  // deallocate
  delete[] hA;
  delete[] hB;
  delete[] hC;
  hA = nullptr;
  hB = nullptr;
  hC = nullptr;

  return 0;
}