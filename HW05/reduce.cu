#include "reduce.cuh"

typedef unsigned int uint;

/**
 * @brief kernel function to calculate reduce using
 * first add after load
 *
 * @param g_idata input data array
 * @param g_odata output data array
 * @param n length of input array
 * @return __global__
 */
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
  // allocate shared memory
  __shared__ extern float shrMem[];

  // load the data into the shared memory and add first
  uint tid = threadIdx.x;
  uint i = blockIdx.x * (blockDim.x * 2) + tid;
  uint bi = i + blockDim.x;

  // load and add
  if (bi < n) {
    shrMem[tid] = g_idata[i] + g_idata[bi];
  }

  // only load not add (lead to divergence to last arr)
  if (i < n && bi >= n) {
    shrMem[tid] = g_idata[i];
  }

  // not load (buffer, lead to divergence to last arr)
  if (i >= n) {
    shrMem[tid] = 0;
  }
  __syncthreads();

  // binary add
  for (uint s = (blockDim.x + 1) / 2; s > 1; s = (s + 1) / 2) {
    if (tid < s) {
      shrMem[tid] += shrMem[tid + s];
      shrMem[tid + s] = 0;
    }

    __syncthreads();
  }

  // put data into the g_odata
  g_odata[blockIdx.x] = shrMem[0] + shrMem[1];
}

/**
 * @brief host function to call kernel function
 *
 * @param input pointer to input array
 * @param output pointer to output array
 * @param N total length of array to reduce
 * @param threads_per_block number of threads per block
 * @return __host__
 */
__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
  // init two pointers for call kernel function
  float *g_idata = *input, *g_odata = *output;
  while (N > 1) {
    // number of blocks
    int numBlocks = (N + threads_per_block - 1) / threads_per_block;
    numBlocks = (numBlocks + 1) / 2; // half due to first add

    // call kernel functions
    reduce_kernel<<<numBlocks, threads_per_block,
                    threads_per_block * sizeof(float)>>>(g_idata, g_odata, N);

    // save the output
    g_idata = g_odata;

    // decrease the number of blocks for next calculations
    N = numBlocks;
  }

  // swap input and output so that result is save at first element of input
  float *temp;
  temp = *input;
  *input = *output;
  *output = temp;
}
