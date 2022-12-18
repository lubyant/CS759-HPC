#include "matmul.cuh"
#include <stdio.h>
typedef unsigned int uint;
// timer
cudaEvent_t start;
cudaEvent_t stop;

__global__ void kernel_matmul1(int *A, int *B, int *C, uint n, uint block_dim);
__global__ void kernel_matmul2(float *A, float *B, float *C, uint n,
                              uint block_dim);
__global__ void kernel_matmul3(double *A, double *B, double *C, uint n,
                              uint block_dim);

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim) {
  // allocate memory in GPU
  int *dA, *dB, *dC;
  cudaMalloc((void **)&dA, n * n * sizeof(int));
  cudaMalloc((void **)&dB, n * n * sizeof(int));
  cudaMalloc((void **)&dC, n * n * sizeof(int));

  // copy data into device
  cudaMemcpy(dA, A, n * n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, n * n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, n * n * sizeof(int));
  // kernel configuration
  dim3 DimBlock(block_dim, block_dim);
  dim3 DimGrid((n + block_dim - 1) / block_dim,
               (n + block_dim - 1) / block_dim);

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // call the kernel function
  kernel_matmul1<<<DimGrid, DimBlock>>>(
      dA, dB, dC, n, block_dim);

  // timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy the results
  cudaMemcpy(C, dC, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  // print out results
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%d\n%d\n%f\n", C[0], C[n * n - 1], ms);

  // deallocate
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  dA = nullptr;
  dB = nullptr;
  dC = nullptr;
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim) {
  // allocate memory in GPU
  float *dA, *dB, *dC;
  cudaMalloc((void **)&dA, n * n * sizeof(float));
  cudaMalloc((void **)&dB, n * n * sizeof(float));
  cudaMalloc((void **)&dC, n * n * sizeof(float));

  // copy data into device
  cudaMemcpy(dA, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, n * n * sizeof(float));
  // kernel configuration
  dim3 DimBlock(block_dim, block_dim);
  dim3 DimGrid((n + block_dim - 1) / block_dim,
               (n + block_dim - 1) / block_dim);

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // call the kernel function
  kernel_matmul2<<<DimGrid, DimBlock>>>(
      dA, dB, dC, n, block_dim);

  // timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy the results
  cudaMemcpy(C, dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  // print out results
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n%f\n", C[0], C[n * n - 1], ms);

  // deallocate
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  dA = nullptr;
  dB = nullptr;
  dC = nullptr;
}
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim) {
  // allocate memory in GPU
  double *dA, *dB, *dC;
  cudaMalloc((void **)&dA, n * n * sizeof(double));
  cudaMalloc((void **)&dB, n * n * sizeof(double));
  cudaMalloc((void **)&dC, n * n * sizeof(double));

  // copy data into device
  cudaMemcpy(dA, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, n * n * sizeof(double));
  // kernel configuration
  dim3 DimBlock(block_dim, block_dim);
  dim3 DimGrid((n + block_dim - 1) / block_dim,
               (n + block_dim - 1) / block_dim);

  // timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // call the kernel function
  kernel_matmul3<<<DimGrid, DimBlock>>>(
      dA, dB, dC, n, block_dim);

  // timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // copy the results
  cudaMemcpy(C, dC, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  // print out results
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("%f\n%f\n%f\n", C[0], C[n * n - 1], ms);

  // deallocate
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  dA = nullptr;
  dB = nullptr;
  dC = nullptr;
}

__global__ void kernel_matmul1(int *A, int *B, int *C, uint n, uint block_dim) {
  // allocate share memory
  // __shared__ extern int shrMemInt[];
  __shared__ int shrMemInt[2048];
  int *shrMemA = shrMemInt;                             // shrMem for matrix A
  int *shrMemB = shrMemInt + block_dim * block_dim + 1; // shrMem for matrix B
  // memset(shrMem, 0, 2*block_dim*block_dim*sizeof(T));
  /**
   * index conversion
   * grid (blockIndx.x,blockIndex.y)
   * block (threadIdx.x, threadIdx.y)
   * nx = blockIdx.x*block_dim + threadIdx.x
   * ny = blockIdx.y*block_dim + threadIdx.y
   */
  uint nx = blockIdx.x * block_dim + threadIdx.x;
  uint ny = blockIdx.y * block_dim + threadIdx.y;

  for (uint k = 0; k < gridDim.x; k++) {
    // protection
    if (nx < n) {
      // put data into shrMem
      shrMemA[threadIdx.x * block_dim + threadIdx.y] =
          A[k * block_dim + threadIdx.y +
            (blockIdx.x * block_dim + threadIdx.x) * n];
      // shrMemB[threadIdx.x*block_dim+threadIdx.y] =
      // B[(k*block_dim+threadIdx.x)*n+threadIdx.y+blockIdx.y*block_dim];
    }
    if (ny < n) {
      // put data into shrMem
      // shrMemA[threadIdx.x*block_dim+threadIdx.y] =
      // A[k*block_dim+threadIdx.y+(blockIdx.x*block_dim+threadIdx.x)*n];
      shrMemB[threadIdx.x * block_dim + threadIdx.y] =
          B[(k * block_dim + threadIdx.x) * n + threadIdx.y +
            blockIdx.y * block_dim];
    }
    __syncthreads();
    // protection
    if (nx < n && ny < n) {
      // calculate inner product
      for (uint i = 0; i < block_dim; i++) {
        C[nx * n + ny] += shrMemA[threadIdx.x * block_dim + i] *
                          shrMemB[block_dim * i + threadIdx.y];
        // if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 &&
        // blockIdx.y == 2) printf("%d*%d, %d \n",
        // shrMemA[threadIdx.x*block_dim+i],shrMemB[threadIdx.x*block_dim+threadIdx.y+i*block_dim],
        // C[nx*n+ny]);
      }
    }
    __syncthreads();
  }
}

__global__ void kernel_matmul2(float *A, float *B, float *C, uint n,
                              uint block_dim) {
  // allocate share memory
  // __shared__ extern float shrMemFlt[];
  __shared__ float shrMemFlt[2048];
  float *shrMemA = shrMemFlt;                             // shrMem for matrix A
  float *shrMemB = shrMemFlt + block_dim * block_dim + 1; // shrMem for matrix B
  // memset(shrMem, 0, 2*block_dim*block_dim*sizeof(T));
  /**
   * index conversion
   * grid (blockIndx.x,blockIndex.y)
   * block (threadIdx.x, threadIdx.y)
   * nx = blockIdx.x*block_dim + threadIdx.x
   * ny = blockIdx.y*block_dim + threadIdx.y
   */
  uint nx = blockIdx.x * block_dim + threadIdx.x;
  uint ny = blockIdx.y * block_dim + threadIdx.y;

  for (uint k = 0; k < gridDim.x; k++) {
    // protection
    if (nx < n) {
      // put data into shrMem
      shrMemA[threadIdx.x * block_dim + threadIdx.y] =
          A[k * block_dim + threadIdx.y +
            (blockIdx.x * block_dim + threadIdx.x) * n];
      // shrMemB[threadIdx.x*block_dim+threadIdx.y] =
      // B[(k*block_dim+threadIdx.x)*n+threadIdx.y+blockIdx.y*block_dim];
    }
    if (ny < n) {
      // put data into shrMem
      // shrMemA[threadIdx.x*block_dim+threadIdx.y] =
      // A[k*block_dim+threadIdx.y+(blockIdx.x*block_dim+threadIdx.x)*n];
      shrMemB[threadIdx.x * block_dim + threadIdx.y] =
          B[(k * block_dim + threadIdx.x) * n + threadIdx.y +
            blockIdx.y * block_dim];
    }
    __syncthreads();
    // protection
    if (nx < n && ny < n) {
      // calculate inner product
      for (uint i = 0; i < block_dim; i++) {
        C[nx * n + ny] += shrMemA[threadIdx.x * block_dim + i] *
                          shrMemB[block_dim * i + threadIdx.y];
        // if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 &&
        // blockIdx.y == 2) printf("%d*%d, %d \n",
        // shrMemA[threadIdx.x*block_dim+i],shrMemB[threadIdx.x*block_dim+threadIdx.y+i*block_dim],
        // C[nx*n+ny]);
      }
    }
    __syncthreads();
  }
}

__global__ void kernel_matmul3(double *A, double *B, double *C, uint n,
                              uint block_dim) {
  // allocate share memory
  // __shared__ extern double shrMemDouble[];
  __shared__ double shrMemDouble[2048];
  double *shrMemA = shrMemDouble; // shrMem for matrix A
  double *shrMemB =
      shrMemDouble + block_dim * block_dim + 1; // shrMem for matrix B
  // memset(shrMem, 0, 2*block_dim*block_dim*sizeof(T));
  /**
   * index conversion
   * grid (blockIndx.x,blockIndex.y)
   * block (threadIdx.x, threadIdx.y)
   * nx = blockIdx.x*block_dim + threadIdx.x
   * ny = blockIdx.y*block_dim + threadIdx.y
   */
  uint nx = blockIdx.x * block_dim + threadIdx.x;
  uint ny = blockIdx.y * block_dim + threadIdx.y;

  for (uint k = 0; k < gridDim.x; k++) {
    // protection
    if (nx < n) {
      // put data into shrMem
      shrMemA[threadIdx.x * block_dim + threadIdx.y] =
          A[k * block_dim + threadIdx.y +
            (blockIdx.x * block_dim + threadIdx.x) * n];
      // shrMemB[threadIdx.x*block_dim+threadIdx.y] =
      // B[(k*block_dim+threadIdx.x)*n+threadIdx.y+blockIdx.y*block_dim];
    }
    if (ny < n) {
      // put data into shrMem
      // shrMemA[threadIdx.x*block_dim+threadIdx.y] =
      // A[k*block_dim+threadIdx.y+(blockIdx.x*block_dim+threadIdx.x)*n];
      shrMemB[threadIdx.x * block_dim + threadIdx.y] =
          B[(k * block_dim + threadIdx.x) * n + threadIdx.y +
            blockIdx.y * block_dim];
    }
    __syncthreads();
    // protection
    if (nx < n && ny < n) {
      // calculate inner product
      for (uint i = 0; i < block_dim; i++) {
        C[nx * n + ny] += shrMemA[threadIdx.x * block_dim + i] *
                          shrMemB[block_dim * i + threadIdx.y];
        // if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 &&
        // blockIdx.y == 2) printf("%d*%d, %d \n",
        // shrMemA[threadIdx.x*block_dim+i],shrMemB[threadIdx.x*block_dim+threadIdx.y+i*block_dim],
        // C[nx*n+ny]);
      }
    }
    __syncthreads();
  }
}
