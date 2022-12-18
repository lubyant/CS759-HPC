#include "reduce.cuh"
#include <cstdio>
__host__ void createArr(float *arr, uint N);
__host__ void arrayReduce(float *arr, uint N, uint threads_per_block);

int main(int argc, char *argv[]) {
  uint N = atoi(argv[1]);
  uint threads_per_block = atoi(argv[2]);

  // pointer to input
  float *arr = new float[N];

  // create a random array
  createArr(arr, N);

  // reduce the array
  arrayReduce(arr, N, threads_per_block);

  // deallocate the memory
  delete[] arr;
  return 0;
}

/**
 * @brief Create a Arr object
 * by random number
 * @param arr pointer to array
 * @param N length of the array
 */
void createArr(float *arr, uint N) {
  srand(time(NULL));
  for (uint i = 0; i < N; i++) {
    // random between (-1,1)
    arr[i] = 2 * (float)rand() / (float)RAND_MAX - 1;
  }
}

/**
 * @brief function to allocate memory and call the reduce
 * function
 *
 * @param arr pointer to input array
 * @param N length of array
 * @param threads_per_block number of threads per block
 * @return __host__
 */
__host__ void arrayReduce(float *arr, uint N, uint threads_per_block) {
  // assign the pointers for call in kernel functions
  float *input, *output;

  // timers
  cudaEvent_t start, stop;

  // configurations
  int numBlocks = (N + threads_per_block - 1) / threads_per_block;
  numBlocks = (numBlocks + 1) / 2;

  // allocate memory
  cudaMalloc((void **)&input, sizeof(float) * N);
  cudaMalloc((void **)&output, sizeof(float) * numBlocks);

  // copy data into device
  cudaMemcpy(input, arr, N * sizeof(float), cudaMemcpyHostToDevice);

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // call the function to calculate the reduction
  reduce(&input, &output, N, threads_per_block);

  // timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy back the results
  cudaMemcpy(arr, input, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost);

  // print out the results
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n", arr[0], ms);

  // deallocate
  cudaFree(input);
  cudaFree(output);
  input = nullptr;
  output = nullptr;
}

/**
 * @brief some test cases
 *
 */
// int main(){
//     int N = pow(2,10);
//     float *arr = new float[N];
//     for(int i=0; i<N; i++){
//         arr[i] = i+1;
//     }
//     arrayReduce(arr, N, 1024);
//     return 0;
// }