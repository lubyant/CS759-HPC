#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
cudaEvent_t start, stop;
int main(int argc, char *argv[]) {
  // length of array
  uint n = (uint)atoi(argv[1]);
  // uint n = 10;

  // allocate host vector
  thrust::host_vector<float> hv(n);

  // allocate device vector
  thrust::device_vector<float> dv(n);

  // initialization and copy
  for (uint i = 0; i < n; i++)
    hv[i] = 2 * (float)rand() / (float)RAND_MAX - 1;
  thrust::copy(hv.begin(), hv.end(), dv.begin());

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  float reduce = thrust::reduce(thrust::device, dv.begin(), dv.end());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // print output
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n", reduce, ms);
  return 0;
}