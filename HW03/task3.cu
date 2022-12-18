#include "vscale.cuh"
#include <stdio.h>

int main(int argc, char *argv[]) {
  using namespace std;
  cudaEvent_t start;
  cudaEvent_t stop;
  int n = atoi(argv[1]);
  // int n = 4;

  // allocate the dynamic array a and b
  float *a = new float[n];
  float *b = new float[n];

  // random initialization
  for (int i = 0; i < n; i++) {
    a[i] = 20 * ((float)rand()) / ((float)RAND_MAX) - 10;
    b[i] = ((float)rand()) / ((float)RAND_MAX);
  }

  // allocate array da,db in cuda device and copy a,b into da,db
  float *da;
  cudaMalloc((void **)&da, sizeof(float) * n);
  cudaMemcpy(da, a, sizeof(float) * n, cudaMemcpyHostToDevice);
  float *db;
  cudaMalloc((void **)&db, sizeof(float) * n);
  cudaMemcpy(db, b, sizeof(float) * n, cudaMemcpyHostToDevice);

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // allocate the threads
  int threadsPerBlock = 256; // number of threads each block
  int blocksPerGrid =
      (n + threadsPerBlock - 1) / threadsPerBlock; // number of blocks
  vscale<<<blocksPerGrid, threadsPerBlock>>>(da, db, (unsigned int)n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy from device into host for print
  cudaMemcpy(b, db, n * sizeof(float), cudaMemcpyDeviceToHost);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  // print the anser
  printf("%f\n%f\n%f\n", ms, b[0], b[n - 1]);

  // deallocate
  delete[] a;
  delete[] b;
  cudaFree(da);
  cudaFree(db);

  return 0;
}