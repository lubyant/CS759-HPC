#include "mmul.h"
#include <cstdio>
cudaEvent_t start, stop;

void testCase(uint N, uint N_TESTS);

int main(int argc, char *argv[]) {
  uint N = (uint)atoi(argv[1]);
  uint N_TESTS = (uint)atoi(argv[2]);
  // uint N = pow(2, 15);
  // uint N_TESTS = 1;

  testCase(N, N_TESTS);

  return 0;
}

/**
 * @brief test case
 * 
 * @param N size of the matrices
 * @param N_TESTS number of tests to run
 */
void testCase(uint N, uint N_TESTS) {

  // generate random matrix
  float *A = new float[N * N];
  float *B = new float[N * N];
  float *C = new float[N * N];

  srand(time(NULL));
  // column wise initialization
  for (uint j = 0; j < N; j++) {
    for (uint i = 0; i < N; i++) {
      A[i + j * N] = 2 * (float)rand() / (float)RAND_MAX - 1;
      B[i + j * N] = 2 * (float)rand() / (float)RAND_MAX - 1;
      C[i + j * N] = 2 * (float)rand() / (float)RAND_MAX - 1;
    }
  }

  float *dA, *dB, *dC;
  cudaMallocManaged(&dA, sizeof(float) * N * N);
  cudaMallocManaged(&dB, sizeof(float) * N * N);
  cudaMallocManaged(&dC, sizeof(float) * N * N);

  cudaMemcpy(dA, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float) * N * N, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float total_ms;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (uint i = 0; i < N_TESTS; i++)
    mmul(handle, dA, dB, dC, N);

  // timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_ms, start, stop);

  printf("%f\n", total_ms / N_TESTS);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  delete[] A;
  delete[] B;
  delete[] C;
}
