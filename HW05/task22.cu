#include "matmul.cuh"
#include <stdio.h>

void matmul_int(uint, uint);
void matmul_float(uint, uint);
void matmul_double(uint, uint);

int main(int argc, char *argv[]) {
  uint N = (uint)atoi(argv[1]);
  uint block_dim = (uint)atoi(argv[2]);
  // uint N = pow(2, 14);
  // uint block_dim = 32;

  // init the test case
  int *A1 = new int[N * N];
  int *B1 = new int[N * N];
  int *C1 = new int[N * N]{0};
  float *A2 = new float[N * N];
  float *B2 = new float[N * N];
  float *C2 = new float[N * N]{0};
  double *A3 = new double[N * N];
  double *B3 = new double[N * N];
  double *C3 = new double[N * N]{0};
  srand(time(NULL));
  for (uint i = 0; i < N * N; i++) {
    A1[i] = (int)(4 * (double)rand() / ((double)RAND_MAX) - 2);
    B1[i] = (int)(4 * (double)rand() / ((double)RAND_MAX) - 2);
    A2[i] = (float)A1[i];
    B2[i] = (float)B1[i];
    A3[i] = (double)A1[i];
    B3[i] = (double)B1[i];
  }

  // allocate memory in GPU
  int *dA1=nullptr, *dB1=nullptr, *dC1=nullptr;
  cudaMalloc((void **)&dA1, N * N * sizeof(int));
  cudaMalloc((void **)&dB1, N * N * sizeof(int));
  cudaMalloc((void **)&dC1, N * N * sizeof(int));

  // copy data into device
  cudaMemcpy(dA1, A1, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dB1, B1, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dC1, 0, N * N * sizeof(int));

  // call matmul1
  matmul_1(dA1, dB1, dC1, N, block_dim);

  // copy the results
  cudaMemcpy(C1, dC1, N * N * sizeof(int), cudaMemcpyDeviceToHost);


  // print out results
  printf("%d\n%d\n", C1[0], C1[N * N - 1]);

  // deallocate
  cudaFree(dA1);
  cudaFree(dB1);
  cudaFree(dC1);
  dA1 = nullptr;
  dB1 = nullptr;
  dC1 = nullptr;
  cudaDeviceSynchronize();

  // deallocate
  delete[] A1;
  delete[] B1;
  delete[] C1;
  A1 = nullptr;
  B1 = nullptr;
  C1 = nullptr;

  // allocate memory in GPU
  float *dA2=nullptr, *dB2=nullptr, *dC2=nullptr;
  cudaMalloc((void **)&dA2, N * N * sizeof(float));
  cudaMalloc((void **)&dB2, N * N * sizeof(float));
  cudaMalloc((void **)&dC2, N * N * sizeof(float));

  // copy data into device
  cudaMemcpy(dA2, A2, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB2, B2, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dC2, 0, N * N * sizeof(float));

  // call matmul1
  matmul_2(dA2, dB2, dC2, N, block_dim);

  // copy the results
  cudaMemcpy(C2, dC2, N * N * sizeof(float), cudaMemcpyDeviceToHost);


  // print out results
  printf("%f\n%f\n", C2[0], C2[N * N - 1]);

  // deallocate
  cudaFree(dA2);
  cudaFree(dB2);
  cudaFree(dC2);
  dA2 = nullptr;
  dB2 = nullptr;
  dC2 = nullptr;
    cudaDeviceSynchronize();
  // deallocate
  delete[] A2;
  delete[] B2;
  delete[] C2;
  A2 = nullptr;
  B2 = nullptr;
  C2 = nullptr;

  // allocate memory in GPU
  double *dA3=nullptr, *dB3=nullptr, *dC3=nullptr;
  cudaMalloc((void **)&dA3, N * N * sizeof(double));
  cudaMalloc((void **)&dB3, N * N * sizeof(double));
  cudaMalloc((void **)&dC3, N * N * sizeof(double));

  // copy data into device
  cudaMemcpy(dA3, A3, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB3, B3, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(dC3, 0, N * N * sizeof(double));

  // call matmul1
  matmul_3(dA3, dB3, dC3, N, block_dim);

  // copy the results
  cudaMemcpy(C3, dC3, N * N * sizeof(double), cudaMemcpyDeviceToHost);


  // print out results
  printf("%f\n%f\n", C3[0], C3[N * N - 1]);

  // deallocate
  cudaFree(dA3);
  cudaFree(dB3);
  cudaFree(dC3);
  dA3 = nullptr;
  dB3 = nullptr;
  dC3 = nullptr;
  cudaDeviceSynchronize();

  // deallocate
  delete[] A3;
  delete[] B3;
  delete[] C3;
  A3 = nullptr;
  B3 = nullptr;
  C3 = nullptr;

  return 0;
}


// int main(){
//     uint N = 5;
//     uint block_dim = 2;

//     // allocate
//     float *A = new float[N*N];
//     float *B = new float[N*N];
//     float *C = new float[N*N]{0};
//     srand(time(NULL));
//     for(uint i=0; i<N*N; i++){
//         // A[i] = (double)rand()/((double)RAND_MAX);
//         // B[i] = (double)rand()/((double)RAND_MAX);
//         A[i] = i;
//         B[i] = N*N-1-i;
//     }

//     matmul_2(A,B,C,N,block_dim);
//     for(uint i=0; i<N*N; i++){
//         printf("%f\n", C[i]);
//     }
// }