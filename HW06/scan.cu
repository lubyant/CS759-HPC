#include "scan.cuh"
#include <cstdio>
// timer
cudaEvent_t start, stop;
typedef unsigned int uint;
/**
 * @brief scan kernel function using HellisStelle
 * Algorithm
 *
 * @param g_idata input array
 * @param g_odata output array
 * @param n length of array
 * @param ghost_sum ghost sum from the last portion of
 * array, which is also equal to last element of last array
 * @return __global__
 */
__global__ void kernel_HellisStelle(const float *g_idata, float *g_odata,
                                    uint n, float *scan_sum);

__global__ void kernel_HellisStelle(const float *g_idata, float *g_odata,
                                    uint n);
__global__ void kernel_add(float *g_odata, float *scan_sum, uint n);

__host__ void scan(const float *dInput, float *dOutput, unsigned int n,
                   unsigned int threads_per_block) {
  // number of block
  int numBlock = (n + threads_per_block - 1) / threads_per_block;

  // create an array for holding scan sum of each block
  float *scan_sum_i;
  float *scan_sum_o;
  cudaMallocManaged(&scan_sum_i, sizeof(float) * numBlock);
  cudaMallocManaged(&scan_sum_o, sizeof(float) * numBlock);

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // scan the array first time and store it in scanSum
  kernel_HellisStelle<<<numBlock, threads_per_block,
                        sizeof(float) * 2 * threads_per_block>>>(
      dInput, dOutput, n, scan_sum_i);
  cudaDeviceSynchronize();

  // scan the scan_sum_i and store in scan sum output
  kernel_HellisStelle<<<1, threads_per_block,
                        sizeof(float) * 2 * threads_per_block>>>(
      scan_sum_i, scan_sum_o, numBlock);
  cudaDeviceSynchronize();

  // add the scan_sum_o into the scanned array
  kernel_add<<<numBlock, threads_per_block>>>(dOutput, scan_sum_o, n);

  // timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // print out the results
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n", dOutput[n - 1], ms);

  // deallocate
  cudaFree(scan_sum_i);
  cudaFree(scan_sum_o);
}

// implementation of kernel call
__global__ void kernel_HellisStelle(const float *g_idata, float *g_odata,
                                    uint n, float *scan_sum) {
  // allocate shared memory
  extern volatile __shared__ float shrMem[];

  // thread id
  int tid = threadIdx.x;

  // Index
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  // swift count
  int pout = 0, pin = 1;

  // write data into share memory
  if (id < n) {
    shrMem[tid] =
        g_idata[id]; // *inclusive scan*: copy the data into same location
  }

  __syncthreads();

  // length of array
  uint len = blockDim.x;
  if (n % blockDim.x != 0)
    len = blockIdx.x == (gridDim.x - 1) ? n % blockDim.x : blockDim.x;

  // swap add by iteration
  for (int offset = 1; offset < len; offset <<= 1) {
    if (tid < len) {
      // swap the count
      pout = 1 - pout;
      pin = 1 - pin;

      if (tid >= offset) // adding
        shrMem[pout * len + tid] =
            shrMem[pin * len + tid] + shrMem[pin * len + tid - offset];
      else // not adding
        shrMem[pout * len + tid] = shrMem[pin * len + tid];
    }
    // sync before next iteration
    __syncthreads();
  }

  if (id < n) {
    // write output
    g_odata[id] = shrMem[pout * len + tid];
  }
  __syncthreads();

  // save the last output into the scan_sum
  if (tid == len - 1 && id < n) {
    scan_sum[blockIdx.x] = g_odata[id];
    // printf("%f ", scan_sum[blockIdx.x]);
  }
}

__global__ void kernel_add(float *g_odata, float *scan_sum, uint n) {
  // index
  uint index = threadIdx.x + blockIdx.x * blockDim.x;

  // add the scan_sum into g_odata
  if (blockIdx.x > 0 && index < n) {
    g_odata[index] += scan_sum[blockIdx.x - 1];
  }
}

__global__ void kernel_HellisStelle(const float *g_idata, float *g_odata,
                                    uint n) {
  // allocate shared memory
  extern volatile __shared__ float shrMem[];

  // thread id
  int tid = threadIdx.x;

  // Index
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  // swift count
  int pout = 0, pin = 1;

  // write data into share memory
  if (id < n) {
    shrMem[tid] =
        g_idata[id]; // *inclusive scan*: copy the data into same location
  }

  __syncthreads();

  // length of array
  uint len;
  len = n;

  // swap add by iteration
  for (int offset = 1; offset < len; offset <<= 1) {
    if (tid < len) {
      // swap the count
      pout = 1 - pout;
      pin = 1 - pin;

      if (tid >= offset) // adding
        shrMem[pout * len + tid] =
            shrMem[pin * len + tid] + shrMem[pin * len + tid - offset];
      else // not adding
        shrMem[pout * len + tid] = shrMem[pin * len + tid];
    }
    // sync before next iteration
    __syncthreads();
  }

  if (id < n) {
    // write output
    g_odata[id] = shrMem[pout * n + tid];
  }
  __syncthreads();
}

/**
 * @brief test case
 *
 */
// int main(){
//     uint N = pow(2,20);
//     float *input, *output;
//     cudaMallocManaged(&input, N*sizeof(float));
//     cudaMallocManaged(&output, N*sizeof(float));
//     for(uint i=0; i<N; i++)
//         input[i] = 1;

//     scan(input, output, N, 1024);
//     // for(uint i=0; i<N; i++)
//     //     printf("%f ", output[i]);
//     return 0;
// }