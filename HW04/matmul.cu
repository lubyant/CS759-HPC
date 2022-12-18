#include "matmul.cuh"
#include <cstdio>

/**
 * @brief Kernel function for matmul
 * 
 * @param A n*n matrix
 * @param B n*n matrix
 * @param C n*n matrix
 * @param n side of matrix
 * @return __global__ 
 */
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              size_t n) {

  // index conversion:
  // nx = blockIdx.x, ny = blockIdx.y*num+ threadIdx
  typedef unsigned int uint;
  uint nx = blockIdx.x;
  uint ny = blockIdx.y * blockDim.x + threadIdx.x;

  // protection from over buffer
  if (ny > n - 1)
    return;

  // assign a init value
  float value = 0;

  // inner product
  for (size_t i = 0; i < n; i++) {
    value += A[n * nx + i] * B[ny + i * n];
  }
  C[nx * n + ny] = value;
  // printf("%d %d %d %f\n", blockIdx.x, blockIdx.y, threadIdx.x, value);
}

/**
 * @brief calculate the matmul
 * 
 * @param A n*n matrix
 * @param B n*n matrix
 * @param C n*n matrix
 * @param n side of matrix
 */
void matmul(const float *A, const float *B, float *C, size_t n,
            unsigned int threads_per_block) {
  typedef unsigned int uint;

  // device array
  float *dA, *dB, *dC;

  // allocate device memory
  cudaMalloc((void **)&dA, sizeof(float) * n * n);
  cudaMalloc((void **)&dB, sizeof(float) * n * n);
  cudaMalloc((void **)&dC, sizeof(float) * n * n);

  // copy host to device
  cudaMemcpy(dA, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * n * n, cudaMemcpyHostToDevice);

  // num of blocks in a row
  uint block_per_row = (n + threads_per_block - 1) / threads_per_block;

  // num of rows
  uint num_block_row = (uint)n;

  // block dim
  dim3 DimBlock(threads_per_block);

  // grid dim
  dim3 DimGrid(num_block_row, block_per_row);

  // cal the kernel function
  matmul_kernel<<<DimGrid, DimBlock>>>(dA, dB, dC, n);

  // copy device to host
  cudaMemcpy(C, dC, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

  // deallocate
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);  
  dA = nullptr;
  dB = nullptr;
  dC = nullptr;
}

/**
 * @brief test case
 * 
 */
// int main(){
//     size_t n = 3;
//     // host array
//     float *hA = new float[n*n]{1,2,3,4,5,6,7,8,9};
//     float *hB = new float[n*n]{9,8,7,6,5,4,3,2,1};
//     float *hC = new float[n*n]{0};

//     // device array
//     float *dA, *dB, *dC;

//     // allocate device memory
//     cudaMalloc((void **)&dA, sizeof(float) * n*n);
//     cudaMalloc((void **)&dB, sizeof(float) * n*n);
//     cudaMalloc((void **)&dC, sizeof(float) * n*n);

//     // copy host to device
//     cudaMemcpy(dA, hA, sizeof(float) * n*n, cudaMemcpyHostToDevice);
//     cudaMemcpy(dB, hB, sizeof(float) * n*n, cudaMemcpyHostToDevice);

//     // cal matmul
//     matmul(dA, dB, dC, n, 2);

//     // copy device to host
//     cudaMemcpy(hC, dC, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

//     // print results
//     for(size_t i=0; i<n*n; i++)
//         printf("%f\n", hC[i]);

//     // deallocate
//     cudaFree(dA);
//     cudaFree(dB);
//     cudaFree(dC);
//     delete[] hA;
//     delete[] hB;
//     delete[] hC;

//     return 0;
//  }