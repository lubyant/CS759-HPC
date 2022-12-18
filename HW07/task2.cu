#include "count.cuh"
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
cudaEvent_t start, stop;
int main(int argc, char *argv[]) {
  uint n = (uint)atoi(argv[1]);
  // uint n = pow(2,24);

  // initialize the array
  thrust::host_vector<int> hv(n);
  thrust::device_vector<int> dv(n);
  srand(time(NULL));
  for (uint i = 0; i < n; i++) {
    hv[i] = (500 * (float)rand() / (float)RAND_MAX);
  }

  // copy the data into device
  thrust::copy(hv.begin(), hv.end(), dv.begin());
  thrust::device_vector<int> values(n), counts(n);

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  count(dv, values, counts);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  thrust::host_vector<int> values_h = values;
  thrust::host_vector<int> counts_h = counts;

  // print output
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%d\n%d\n%f\n", values_h[values_h.size() - 1],
         counts_h[counts_h.size() - 1], ms);

  return 0;
}