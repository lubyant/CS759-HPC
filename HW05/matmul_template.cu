#include "matmul.cuh"
#include "cstdio"
#include <typeinfo>

// timer
cudaEvent_t start;
cudaEvent_t stop;

template<typename T>
__global__ void kernel_matmul(T *A, T *B, T *C, uint n, uint block_dim);

template<typename T>
__host__ void matmul(const T *A, const T *B, const T *C, uint n, uint block_dim);

namespace cuda{
  template<typename T>
  struct SharedMemory
  {
    __device__ T* getPointer(){
      return (T*)0;
    }
  };

  template<>
  struct SharedMemory <int>
  {
    __device__ int* getPointer(){extern __shared__ int s_int[]; return s_int;}
  };

  template<>
  struct SharedMemory <float>
  {
    __device__ float* getPointer(){extern __shared__ float s_float[]; return s_float;}
  };

  template<>
  struct SharedMemory <double>
  {
    __device__ double* getPointer(){extern __shared__ double s_double[]; return s_double;}
  };  
}


__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim)
{
    matmul(A, B, C, n, block_dim);
}

template<typename T>
__global__ void kernel_matmul(T *A, T *B, T *C, uint n,
                              uint block_dim) {
  // allocate share memory
  cuda::SharedMemory<T> sharedMemory;
  T* shrMem = sharedMemory.getPointer();
  T *shrMemA = shrMem; // shrMem for matrix A
  T *shrMemB =
      shrMem + block_dim * block_dim + 1; // shrMem for matrix B
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
    }
    if (ny < n) {
      // put data into shrMem
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

template<typename T>
__host__ void matmul(const T *A, const T *B, const T *C, uint n, uint block_dim){
    // allocate memory in GPU
    T *dA, *dB, *dC;
    cudaMalloc((void **)&dA, n * n * sizeof(T));
    cudaMalloc((void **)&dB, n * n * sizeof(T));
    cudaMalloc((void **)&dC, n * n * sizeof(T));

    // copy data into device
    cudaMemcpy(dA, A, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, n * n * sizeof(T));
    // kernel configuration
    dim3 DimBlock(block_dim, block_dim);
    dim3 DimGrid((n + block_dim - 1) / block_dim,
                (n + block_dim - 1) / block_dim);

    // timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // call the kernel function
    kernel_matmul<<<DimGrid, DimBlock, 2 * block_dim * block_dim * sizeof(T)>>>(
        dA, dB, dC, n, block_dim);
    
    // timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    // copy the results
    cudaMemcpy((void *)C, (void *)dC, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    // print out results
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    char printfCmd[20];
    if (typeid(T) == typeid(int)) {
      strcpy(printfCmd, "%d\n%d\n%f\n");
    } else {
      strcpy(printfCmd, "%f\n%f\n%f\n");
    }
    printf(printfCmd, C[0], C[n * n - 1], ms);

    // deallocate
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    dA = nullptr;
    dB = nullptr;
    dC = nullptr;
}

// int main(){
//     uint N = 5;
//     uint block_dim = 2;

//     // allocate
//     int *A = new int[N*N];
//     int *B = new int[N*N];
//     int *C = new int[N*N]{0};
//     srand(time(NULL));
//     for(uint i=0; i<N*N; i++){
//         // A[i] = (double)rand()/((double)RAND_MAX);
//         // B[i] = (double)rand()/((double)RAND_MAX);
//         A[i] = i;
//         B[i] = N*N-1-i;
//     }

//     matmul_1(A,B,C,N,block_dim);
//     for(uint i=0; i<N*N; i++){
//         printf("%d\n", C[i]);
//     }
// }