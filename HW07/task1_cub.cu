#define CUB_STDERR
#include "cub/util_debug.cuh"
#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>
#include <stdio.h>
using namespace cub;
CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory
cudaEvent_t start, stop;
int main(int argc, char *argv[]) {
  // length of array
  uint num_items = (uint)atoi(argv[1]);

  // Set up host arrays
  float *h_in = new float[num_items];

  // Initialization
  for (uint i = 0; i < num_items; i++)
    h_in[i] = 2 * (float)rand() / (float)RAND_MAX - 1;

  // Set up device arrays
  float *d_in = NULL;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&d_in, sizeof(float) * num_items));

  // Initialize device input
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items,
                          cudaMemcpyHostToDevice));

  // Setup device output array
  float *d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(float) * 1));

  // Request and allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                 d_sum, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Do the actual reduce operation
  // CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
  // d_sum, num_items));
  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copy result
  float gpu_sum;
  CubDebugExit(
      cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));

  // print output
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n", gpu_sum, ms);

  // Cleanup
  if (d_in)
    CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_sum)
    CubDebugExit(g_allocator.DeviceFree(d_sum));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

  return 0;
}