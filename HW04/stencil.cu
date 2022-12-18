#include "stencil.cuh"
#include <stdio.h>
/**
 * @brief Kernel function for calculate the stencil
 *
 * @param image pointer to 1d image array
 * @param mask pointer 1d mask array
 * @param output pointer to 1d output array
 * @param n length of image array
 * @param R radius of mask
 * @return __global__
 */
__global__ void stencil_kernel(const float *image, const float *mask,
                               float *output, unsigned int n, unsigned int R) {

  // allocate shared memory (two ways)

  // statically
  // __shared__ float mask_shared[2048];
  // __shared__ float arr_shared[2048];

  // dynamically
  extern __shared__ float shrMem[];
  float *mask_shared = shrMem;              // first 2*R+1 is for mask
  float *arr_shared = (shrMem + 2 * R + 2); // another 2*R+threads

  // index conversion
  int gindex = threadIdx.x + blockIdx.x * blockDim.x; // index in image
  int lindex = threadIdx.x + R;                       // index in shared array

  // protection for over buffer
  if (gindex < n){
    // read the mask into shared memory
    mask_shared[threadIdx.x] = mask[threadIdx.x];
    if (gindex == n - 1) {
      for (int i = 0; i < 2 * R + 1; i++) {
        mask_shared[i] = mask[i];
      }
    }

    // read iamge into shared memory and halo on sides
    // /--1,1,1,........,1,1,1--/
    // heads
    if (threadIdx.x == 0) {
      for (int i = 0; i < R; i++) {
        if (gindex - 1 - i < 0) {
          arr_shared[i] = 1;
        } else {
          arr_shared[R - i - 1] = image[gindex - 1 - i];
        }
      }
    }
    // tails
    if (threadIdx.x == blockDim.x - 1) {
      for (int i = 0; i < R; i++) {
        if (gindex + 1 + i > n - 1) {
          arr_shared[i + blockDim.x + R] = 1;
        } else {
          arr_shared[i + blockDim.x + R] = image[gindex + 1 + i];
        }
      }
    }
    if (gindex == n - 1) {
      for (int i = 0; i < R; i++) {
        arr_shared[i + R + gindex % blockDim.x + 1] = 1;
      }
    }

    arr_shared[lindex] = image[gindex];
  }
  // synchronize
  __syncthreads();

  // output result
  float result = 0;
  for (int i = 0; i < 2 * R + 1; i++) {
    result += mask_shared[i] * arr_shared[lindex - R + i];
  }
  output[gindex] = result;
}

/**
 * @brief calculate the stencil
 *
 * @param image pointer to 1d image array
 * @param mask pointer 1d mask array
 * @param output pointer to 1d output array
 * @param n length of image array
 * @param R radius of mask
 * @return __host__
 */
__host__ void stencil(const float *image, const float *mask, float *output,
                      unsigned int n, unsigned int R,
                      unsigned int threads_per_block) {
  // device array
  float *dImage, *dMask, *dOutput;

  // timer
  cudaEvent_t start;
  cudaEvent_t stop;

  // allocate cuda memory
  cudaMalloc((void **)&dImage, sizeof(float) * n);
  cudaMalloc((void **)&dOutput, sizeof(float) * n);
  cudaMalloc((void **)&dMask, sizeof(float) * (2 * R + 1));

  // copy data into device
  cudaMemcpy(dImage, image, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dMask, mask, sizeof(float) * (2 * R + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(dOutput, output, sizeof(float) * n, cudaMemcpyHostToDevice);

  // num of grid = grid/size
  int numOfGrid = (n + threads_per_block - 1) / threads_per_block;

  // size of shared memory: mask arr (2*R+1) + image arr (2*R+threads)
  // = total (4*R+1+threads_per_block)
  int sizeOfShrMem = (4 * R + 1 + threads_per_block) * sizeof(float);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // allocate the blocks and call kernel function
  stencil_kernel<<<numOfGrid, threads_per_block, sizeOfShrMem>>>(dImage, dMask,
                                                                 dOutput, n, R);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  // copy result back into host
  cudaMemcpy(output, dOutput, sizeof(float) * n, cudaMemcpyDeviceToHost);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n", output[n-1], ms);

  // deallocate
  cudaFree(dImage);
  cudaFree(dOutput);
  cudaFree(dMask);
  dImage = nullptr;
  dOutput = nullptr;
  dMask = nullptr;
}