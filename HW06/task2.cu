#include "scan.cuh"

/**
 * @brief test case generator
 * 
 */
void testCase(uint, uint);

int main(int argc, char *argv[]) {
  // length of array
  uint N = (uint)atoi(argv[1]);

  // number of threads
  uint threads_per_block = (uint)atoi(argv[2]);

  // uint N = 1024;
  // uint threads_per_block = 1024;

  testCase(N, threads_per_block);
  return 0;
}

void testCase(uint N, uint threads_per_block) {

  // allocate managed memory
  float *input, *output;
  cudaMallocManaged(&input, N * sizeof(float));
  cudaMallocManaged(&output, N * sizeof(float));

  // generate random number
  srand(time(NULL));
  for (uint i = 0; i < N; i++)
    input[i] = (float)rand() / (float)RAND_MAX;

  scan(input, output, N, threads_per_block);
  
  // deallocate
  cudaFree(input);
  cudaFree(output);
  input = nullptr;
  output = nullptr;
}