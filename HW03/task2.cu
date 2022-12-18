#include <cuda.h>
#include <stdio.h>

/**
 * @brief Kernel function for print
 *
 * @param data array for storing the thread id
 * @param a random int of factor
 * @param a length of array
 * @return __global__
 */
__global__ void simpleKernel(int *data, const int a, const int n) {

  if((blockIdx.x * 8 + threadIdx.x) < n) // protection
    data[blockIdx.x * 8 + threadIdx.x] += blockIdx.x + a * threadIdx.x;
}

int main() {
  const int numElems = 16;
  int *dA, hA[numElems];
  // allocate memory on the device; zero out all entries in this device array
  cudaMalloc((void **)&dA, sizeof(int) * numElems);
  cudaMemset(dA, 0, numElems * sizeof(int));
  srand(time(NULL));
  int a = (int)(rand() % (10));
  // printf("%d\n", a);

  // invoke GPU kernel, with two block that has 16 threads totally
  simpleKernel<<<2, 8>>>(dA, a, numElems);

  // bring the result back from the GPU into the hostArray
  cudaMemcpy(&hA, dA, sizeof(int) * numElems, cudaMemcpyDeviceToHost);
  // print out the result to confirm that things are looking good
  for (int i = 0; i < numElems; i++)
    printf("%d ", hA[i]);
  printf("\n");
  // release the memory allocated on the GPU
  cudaFree(dA);
  return 0;
}